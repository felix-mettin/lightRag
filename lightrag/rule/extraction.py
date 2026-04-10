"""Knowledge graph extraction and building functions.

This module contains functions for validating LLM extraction results,
building knowledge graph nodes and edges, and upserting them into storage.
"""

from __future__ import annotations
from copy import deepcopy
import json
import re
from typing import Any
from collections import defaultdict
from pathlib import Path

from lightrag.utils import logger, compute_mdhash_id
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage

from .utils import (
    _normalize_text_key,
    _build_override_path_key,
    _coerce_override_path_parts,
    _json_dumps_compact,
    _merge_evidence,
    _merge_node_data_with_human_override,
    _merge_edge_data_with_human_override,
    _stable_clause_id,
    _stable_equipment_id,
    _stable_report_id,
    _stable_test_id,
    _stable_param_id,
    _stable_rule_id,
)


# Cache for tree override rules
_TREE_OVERRIDE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _validate_controlled_payload(data: dict, chunk_text: str, chunk_meta: dict) -> dict:
    """Validate and normalize controlled extraction payload."""
    required_root = {
        "standard",
        "clause",
        "equipment",
        "report_types",
        "equipment_reports",
        "test_items",
        "quality",
    }
    if not isinstance(data, dict):
        raise ValueError("payload must be object")

    if set(data.keys()) != required_root:
        found_keys = set(data.keys())
        extra_keys = found_keys - required_root
        missing_keys = required_root - found_keys
        known_testitem_spill_keys = {
            "test_item",
            "category",
            "aliases",
            "required_reports",
            "parameters",
            "rules",
            "acceptance_criteria",
            "notes",
            "evidence",
            "confidence",
        }
        if not missing_keys and extra_keys and extra_keys.issubset(known_testitem_spill_keys):
            logger.debug(
                "Controlled payload root spill normalized. extras=%s",
                sorted(list(extra_keys)),
            )
        else:
            logger.warning(
                "Controlled payload root mismatch, normalizing keys. missing=%s extras=%s found=%s",
                sorted(list(missing_keys)),
                sorted(list(extra_keys)),
                sorted(list(found_keys)),
            )
        normalized = {
            "standard": data.get("standard", {}),
            "clause": data.get("clause", {}),
            "equipment": data.get("equipment", []),
            "report_types": data.get("report_types", []),
            "equipment_reports": data.get("equipment_reports", []),
            "test_items": data.get("test_items", []),
            "quality": data.get("quality", {}),
        }
        data = normalized

    standard = data["standard"]
    clause = data["clause"]
    if not isinstance(standard, dict) or not isinstance(clause, dict):
        raise ValueError("standard/clause must be object")

    standard.setdefault("std_id", chunk_meta.get("std_id", ""))
    standard.setdefault("std_name", chunk_meta.get("std_name", ""))
    standard.setdefault("source", "")

    clause.setdefault("clause_id", chunk_meta.get("clause_id", ""))
    clause.setdefault("clause_title", chunk_meta.get("clause_title", ""))
    clause.setdefault("chunk_id", chunk_meta.get("chunk_id", ""))
    clause.setdefault("quote", "")
    clause.setdefault("page_hint", "")

    if clause.get("chunk_id") != chunk_meta["chunk_id"]:
        clause["chunk_id"] = chunk_meta["chunk_id"]

    quote = clause.get("quote", "")
    if quote and (len(quote) > 800 or quote not in chunk_text):
        clause["quote"] = ""

    if not isinstance(data["equipment"], list):
        raise ValueError("equipment must be list")
    if not isinstance(data["report_types"], list):
        raise ValueError("report_types must be list")
    if not isinstance(data["equipment_reports"], list):
        raise ValueError("equipment_reports must be list")
    if not isinstance(data["test_items"], list):
        raise ValueError("test_items must be list")
    if not isinstance(data["quality"], dict):
        raise ValueError("quality must be object")

    return data


def _build_controlled_nodes_edges(
    payload: dict,
    chunk_meta: dict,
    file_path: str,
    schema_cfg: dict | None = None,
    chunk_text: str = "",
) -> tuple[list[tuple[str, dict]], list[tuple[str, str, dict]]]:
    """Build knowledge graph nodes and edges from validated payload."""
    from .tree_override import _load_tree_override_rules
    from .utils import (
        _resolve_override_param_name,
        _resolve_override_value_text,
        _infer_value_source,
        _note_is_remove,
    )
    
    schema_cfg = schema_cfg or {}
    tree_override_rules = _load_tree_override_rules(schema_cfg)
    tree_tests_by_path = tree_override_rules.get("tests_by_path", {}) or {}
    tree_tests_by_name = tree_override_rules.get("tests_by_name", {}) or {}
    tree_add_test_items = tree_override_rules.get("add_test_items", []) or []
    report_aliases = schema_cfg.get("report_aliases", {})
    test_aliases = schema_cfg.get("test_aliases", {})
    configured_test_items = schema_cfg.get("test_items", [])
    configured_report_keys = {
        _normalize_text_key(str(name))
        for name in schema_cfg.get("report_types", [])
        if _normalize_text_key(str(name))
    }
    
    strict_override_match_raw = schema_cfg.get("strict_tree_override_match", True)
    strict_override_match = (
        str(strict_override_match_raw).strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(strict_override_match_raw, str)
        else bool(strict_override_match_raw)
    )
    enforce_param_whitelist_raw = schema_cfg.get("enforce_param_whitelist", True)
    enforce_param_whitelist = (
        str(enforce_param_whitelist_raw).strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(enforce_param_whitelist_raw, str)
        else bool(enforce_param_whitelist_raw)
    )
    override_param_filter_to_template_raw = schema_cfg.get(
        "override_param_filter_to_template", True
    )
    override_param_filter_to_template = (
        str(override_param_filter_to_template_raw).strip().lower()
        in {"1", "true", "yes", "on"}
        if isinstance(override_param_filter_to_template_raw, str)
        else bool(override_param_filter_to_template_raw)
    )
    annotation_guardrail_mode_raw = schema_cfg.get("annotation_guardrail_mode", True)
    annotation_guardrail_mode = (
        str(annotation_guardrail_mode_raw).strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(annotation_guardrail_mode_raw, str)
        else bool(annotation_guardrail_mode_raw)
    )
    param_map = schema_cfg.get("param_map", {})
    required_param_map = schema_cfg.get("test_item_param_requirements", {})

    def _normalize_test_item_lookup_key(value: str) -> str:
        text = _normalize_text_key(str(value))
        return text.replace("（", "(").replace("）", ")")

    required_param_map_normalized = {
        _normalize_test_item_lookup_key(str(k)): v
        for k, v in required_param_map.items()
        if _normalize_test_item_lookup_key(str(k))
    }
    allowed_test_item_keys = {
        _normalize_test_item_lookup_key(str(name))
        for name in configured_test_items
        if _normalize_test_item_lookup_key(str(name))
    }

    std_id = payload["standard"].get("std_id", "")
    std_name = payload["standard"].get("std_name", "")
    clause_id = payload["clause"].get("clause_id", "")
    clause_title = payload["clause"].get("clause_title", "")
    chunk_id = payload["clause"].get("chunk_id", "") or chunk_meta["chunk_id"]

    evidence = [{"std_id": std_id, "clause_id": clause_id, "chunk_id": chunk_id}]

    nodes: list[tuple[str, dict]] = []
    edges: list[tuple[str, str, dict]] = []

    clause_node_id = _stable_clause_id(std_id, clause_id, chunk_id)
    clause_node = {
        "entity_id": clause_node_id,
        "entity_type": "StandardClause",
        "name": clause_title or clause_id or chunk_id,
        "std_id": std_id,
        "std_name": std_name,
        "clause_id": clause_id,
        "clause_title": clause_title,
        "chunk_id": chunk_id,
        "quote": payload["clause"].get("quote", ""),
        "page_hint": payload["clause"].get("page_hint", ""),
        "source_id": chunk_id,
        "file_path": file_path,
        "evidence": _json_dumps_compact(evidence),
        "human_override": False,
    }
    nodes.append((clause_node_id, clause_node))

    # Build equipment nodes
    equipment_nodes: dict[str, str] = {}
    for equip_name in payload.get("equipment", []):
        if not equip_name:
            continue
        equip_id = _stable_equipment_id(str(equip_name))
        equipment_nodes[str(equip_name)] = equip_id
        nodes.append((
            equip_id,
            {
                "entity_id": equip_id,
                "entity_type": "EquipmentType",
                "name": str(equip_name),
                "source_id": chunk_id,
                "file_path": file_path,
                "evidence": _json_dumps_compact(evidence),
                "human_override": False,
            },
        ))
        edges.append((
            clause_node_id,
            equip_id,
            {
                "src_id": clause_node_id,
                "tgt_id": equip_id,
                "rel_type": "DEFINES_EQUIPMENT",
                "source_id": chunk_id,
                "file_path": file_path,
                "evidence": _json_dumps_compact(evidence),
                "human_override": False,
            },
        ))

    # Build report type nodes
    report_nodes: dict[str, str] = {}
    for report_name in payload.get("report_types", []):
        if not report_name:
            continue
        report_id = _stable_report_id(str(report_name))
        report_nodes[str(report_name)] = report_id
        nodes.append((
            report_id,
            {
                "entity_id": report_id,
                "entity_type": "ReportType",
                "name": str(report_name),
                "source_id": chunk_id,
                "file_path": file_path,
                "evidence": _json_dumps_compact(evidence),
                "human_override": False,
            },
        ))
        edges.append((
            clause_node_id,
            report_id,
            {
                "src_id": clause_node_id,
                "tgt_id": report_id,
                "rel_type": "DEFINES_REPORT",
                "source_id": chunk_id,
                "file_path": file_path,
                "evidence": _json_dumps_compact(evidence),
                "human_override": False,
            },
        ))

    # Build test item nodes and connect to reports
    test_nodes: dict[str, str] = {}
    for test_item in payload.get("test_items", []):
        if not isinstance(test_item, dict):
            continue
        test_name = str(test_item.get("test_item", "") or "").strip()
        if not test_name:
            continue
        
        # Skip test items not in configured whitelist if enforcement enabled
        if enforce_param_whitelist:
            lookup_key = _normalize_test_item_lookup_key(test_name)
            if lookup_key not in allowed_test_item_keys:
                logger.debug("Skipping test item not in whitelist: %s", test_name)
                continue

        category = str(test_item.get("category", "") or "").strip()
        report_type = str(test_item.get("report_type", "") or "").strip()
        aliases = test_item.get("aliases", []) or []
        acceptance_criteria = test_item.get("acceptance_criteria", "")
        note = test_item.get("note", "")
        confidence = test_item.get("confidence", 1.0)
        required_reports = test_item.get("required_reports", []) or []

        # Determine ID scope key based on report type
        id_scope_key = ""
        if report_type:
            for cfg_report in schema_cfg.get("report_types", []):
                if str(cfg_report).strip() == report_type:
                    id_scope_key = _normalize_text_key(report_type)
                    break

        test_id = _stable_test_id(test_name, id_scope_key)
        test_nodes[test_name] = test_id

        test_evidence = list(evidence)
        if test_item.get("evidence"):
            test_evidence.extend(test_item["evidence"])

        nodes.append((
            test_id,
            {
                "entity_id": test_id,
                "entity_type": "TestItem",
                "name": test_name,
                "category": category,
                "report_type": report_type,
                "aliases": aliases,
                "acceptance_criteria": str(acceptance_criteria) if acceptance_criteria else "",
                "note": str(note) if note else "",
                "confidence": float(confidence) if isinstance(confidence, (int, float)) else 1.0,
                "required_reports": required_reports,
                "source_id": chunk_id,
                "file_path": file_path,
                "evidence": _json_dumps_compact(test_evidence),
                "human_override": False,
            },
        ))

        # Connect to report type
        target_report = report_type
        if not target_report and report_aliases:
            for alias, canonical in report_aliases.items():
                if alias and canonical and alias in test_name:
                    target_report = canonical
                    break
        if target_report:
            report_id = report_nodes.get(target_report, _stable_report_id(target_report))
            edges.append((
                report_id,
                test_id,
                {
                    "src_id": report_id,
                    "tgt_id": test_id,
                    "rel_type": "INCLUDES_TEST",
                    "source_id": chunk_id,
                    "file_path": file_path,
                    "evidence": _json_dumps_compact(test_evidence),
                    "human_override": False,
                },
            ))

        # Build parameter nodes
        for param in test_item.get("parameters", []):
            if not isinstance(param, dict):
                continue
            param_name = str(param.get("param_name", "") or "").strip()
            if not param_name:
                continue

            param_key = _normalize_text_key(param_name)
            param_id = _stable_param_id(test_name, param_key, id_scope_key)

            value_text = str(param.get("value_text", "") or "").strip()
            value_expr = str(param.get("value_expr", "") or "").strip()
            value_type = str(param.get("value_type", "") or "").strip()
            value_source = str(param.get("value_source", "") or "").strip()
            unit = str(param.get("unit", "") or "").strip()
            constraints = str(param.get("constraints", "") or "").strip()
            calc_rule = str(param.get("calc_rule", "") or "").strip()
            derive_from_rated = str(param.get("derive_from_rated", "") or "").strip()
            table_ref = str(param.get("table_ref", "") or "").strip()

            nodes.append((
                param_id,
                {
                    "entity_id": param_id,
                    "entity_type": "TestParameter",
                    "param_name": param_name,
                    "param_key": param_key,
                    "value_text": value_text,
                    "value_expr": value_expr,
                    "value_type": value_type,
                    "value_source": value_source,
                    "unit": unit,
                    "constraints": constraints,
                    "calc_rule": calc_rule,
                    "derive_from_rated": derive_from_rated,
                    "table_ref": table_ref,
                    "source_id": chunk_id,
                    "file_path": file_path,
                    "evidence": _json_dumps_compact(test_evidence),
                    "human_override": False,
                },
            ))

            edges.append((
                test_id,
                param_id,
                {
                    "src_id": test_id,
                    "tgt_id": param_id,
                    "rel_type": "HAS_PARAMETER",
                    "source_id": chunk_id,
                    "file_path": file_path,
                    "evidence": _json_dumps_compact(test_evidence),
                    "human_override": False,
                },
            ))

    return nodes, edges


async def _upsert_controlled_node(
    node_id: str,
    node_data: dict,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    entity_chunks_storage: BaseKVStorage | None,
) -> None:
    """Upsert a controlled node into the knowledge graph."""
    def _compose_node_description(data: dict) -> str:
        entity_type = str(data.get("entity_type", "") or "")
        name = str(data.get("name", "") or "")
        if entity_type == "TestParameter":
            value_text = str(data.get("value_text", "") or "")
            unit = str(data.get("unit", "") or "")
            source = str(data.get("value_source", "") or "")
            parts = [name]
            if value_text:
                parts.append(value_text)
            if unit:
                parts.append(unit)
            if source:
                parts.append(f"source={source}")
            return " ".join(parts)
        return name

    existing = await knowledge_graph_inst.get_node(node_id)
    merged_data = _merge_node_data_with_human_override(existing, node_data)

    description = _compose_node_description(merged_data)
    content = f"{node_id}\n{description}"

    await knowledge_graph_inst.upsert_node(node_id, merged_data)

    if entity_vdb is not None:
        vdb_id = compute_mdhash_id(node_id, prefix="ent-")
        vdb_data = {
            vdb_id: {
                "content": content,
                "entity_name": node_id,
                "entity_type": merged_data.get("entity_type", "UNKNOWN"),
                "source_id": merged_data.get("source_id", ""),
                "file_path": merged_data.get("file_path", "unknown_source"),
            }
        }
        await entity_vdb.upsert(vdb_data)

    if entity_chunks_storage is not None:
        chunk_ids = [merged_data.get("source_id", "")]
        if chunk_ids[0]:
            await entity_chunks_storage.upsert(
                {node_id: {"chunk_ids": chunk_ids, "count": len(chunk_ids)}}
            )


async def _upsert_controlled_edge(
    src_id: str,
    tgt_id: str,
    edge_data: dict,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage | None,
    relation_chunks_storage: BaseKVStorage | None,
) -> None:
    """Upsert a controlled edge into the knowledge graph."""
    existing = await knowledge_graph_inst.get_edge(src_id, tgt_id)
    merged_data = _merge_edge_data_with_human_override(existing, edge_data)

    await knowledge_graph_inst.upsert_edge(src_id, tgt_id, merged_data)

    if relationships_vdb is not None:
        # Sort for consistent ID generation
        if src_id > tgt_id:
            src_id, tgt_id = tgt_id, src_id
        vdb_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")
        rel_type = merged_data.get("rel_type", "RELATED_TO")
        content = f"{rel_type}\t{src_id}\n{tgt_id}"
        vdb_data = {
            vdb_id: {
                "content": content,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "source_id": merged_data.get("source_id", ""),
                "file_path": merged_data.get("file_path", "unknown_source"),
            }
        }
        await relationships_vdb.upsert(vdb_data)

    if relation_chunks_storage is not None:
        chunk_ids = [merged_data.get("source_id", "")]
        if chunk_ids[0]:
            storage_key = f"{src_id}~{tgt_id}"
            await relation_chunks_storage.upsert(
                {storage_key: {"chunk_ids": chunk_ids, "count": len(chunk_ids)}}
            )
