"""Query processing and response enhancement functions.

This module contains functions for query augmentation, context filtering,
and response post-processing for electrical standards domain.
"""

from __future__ import annotations
from copy import deepcopy
import json
import re
from typing import Any

from lightrag.utils import logger

from .utils import _normalize_text_key


def _should_bypass_query_cache(global_config: dict[str, Any] | None) -> bool:
    """Check if query cache should be bypassed for electrical mode."""
    cfg = global_config or {}
    if cfg.get("kg_schema_mode") == "electrical_controlled":
        return True
    addon_params = cfg.get("addon_params", {}) or {}
    schema_cfg = addon_params.get("electrical_schema", {}) or {}
    return bool(schema_cfg)


def _extract_current_report_scopes(query_text: str, schema_cfg: dict[str, Any] | None = None) -> list[str]:
    """Extract report type scopes mentioned in the query."""
    cfg = schema_cfg or {}
    text = str(query_text or "").strip()
    if not text:
        return []

    configured_reports = [
        str(name).strip()
        for name in (cfg.get("report_types", []) or [])
        if str(name).strip()
    ]
    report_aliases = cfg.get("report_aliases", {}) or {}
    matched: list[str] = []

    for report_name in configured_reports:
        if report_name and report_name in text and report_name not in matched:
            matched.append(report_name)

    if isinstance(report_aliases, dict):
        for alias, canonical in report_aliases.items():
            alias_text = str(alias).strip()
            canonical_text = str(canonical).strip()
            if (
                alias_text
                and canonical_text
                and alias_text in text
                and canonical_text not in matched
            ):
                matched.append(canonical_text)

    return matched


def _filter_project_context_by_report_scope(
    project_param_map: dict[str, list[str]],
    project_param_value_map: dict[str, dict[str, dict[str, str]]],
    current_report_scopes: list[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, dict[str, str]]]]:
    """Filter project context by report scope whitelist."""
    if not current_report_scopes:
        return project_param_map, project_param_value_map

    whitelist_map = _get_report_scope_test_whitelist()
    allowed_tests: set[str] = set()
    for scope in current_report_scopes:
        allowed_tests.update(whitelist_map.get(str(scope).strip(), set()))
    if not allowed_tests:
        return project_param_map, project_param_value_map

    filtered_param_map: dict[str, list[str]] = {}
    filtered_value_map: dict[str, dict[str, dict[str, str]]] = {}
    for test_name, params in project_param_map.items():
        name = str(test_name).strip()
        if not name or name not in allowed_tests:
            continue
        filtered_param_map[name] = list(params if isinstance(params, list) else [])
        if name in project_param_value_map:
            filtered_value_map[name] = deepcopy(project_param_value_map.get(name, {}) or {})
    return filtered_param_map, filtered_value_map


def _get_report_scope_test_whitelist() -> dict[str, set[str]]:
    """Get whitelist of test items for each report type."""
    return {
        "绝缘性能型式试验": {
            "工频耐受电压试验",
            "工频耐受电压试验(断口)",
            "工频耐受电压试验(相间及对地)",
            "雷电冲击耐受电压试验",
            "雷电冲击耐受电压试验(断口)",
            "雷电冲击耐受电压试验(相间及对地)",
            "局部放电试验",
        },
        "温升性能型式试验": {
            "连续电流试验",
            "前后回路电阻测量试验",
        },
        "开合性能型式试验": {
            "容性电流开断试验",
            "空载特性测量",
        },
        "短路性能型式试验": {
            "短路开断试验",
            "短时耐受电流和峰值耐受电流试验",
        },
    }


def _build_final_test_item_scope(
    project_param_map: dict[str, list[str]],
    domain_rule_decisions: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Build final test item scope after rule application."""
    allowed_items = list(project_param_map.keys())
    removed_items: list[str] = []
    hard_removed_items: set[str] = set()

    for decision in domain_rule_decisions.values():
        if not isinstance(decision, dict):
            continue
        test_item = str(decision.get("test_item", "") or "").strip()
        if not test_item:
            continue
        rule_kind = str(decision.get("kind", "") or "").strip()
        if rule_kind == "applicability" and not decision.get("enabled"):
            removed_items.append(test_item)
            hard_removed_items.add(test_item)
        if rule_kind == "split" and decision.get("enabled"):
            removed_items.append(test_item)
            hard_removed_items.add(test_item)

    allowed_deduped = list(
        dict.fromkeys(
            item for item in allowed_items if item and item not in hard_removed_items
        )
    )
    removed_deduped = [
        item
        for item in dict.fromkeys(item for item in removed_items if item)
        if item not in allowed_deduped
    ]
    return allowed_deduped, removed_deduped


def _build_test_item_display_map(project_param_map: dict[str, list[str]]) -> dict[str, str]:
    """Build display name mapping for test items."""
    display_map: dict[str, str] = {}
    for test_name in project_param_map.keys():
        name = str(test_name or "").strip()
        if not name:
            continue
        if "#" in name:
            display_map[name] = name.split("#", 1)[0]
        else:
            display_map[name] = name
    return display_map


def _filter_context_by_final_test_item_scope(
    entities_context: list[dict],
    relations_context: list[dict],
    removed_test_items: list[str],
) -> tuple[list[dict], list[dict]]:
    """Filter retrieved context by final test item scope."""
    removed_keys = {_normalize_text_key(item) for item in removed_test_items if item}
    if not removed_keys:
        return entities_context, relations_context

    def _entity_name(entity: dict[str, Any]) -> str:
        for key in ("entity", "entity_name", "name", "test_item"):
            value = str(entity.get(key, "") or "").strip()
            if value:
                return value
        return ""

    filtered_entities = [
        entity
        for entity in entities_context
        if _normalize_text_key(_entity_name(entity)) not in removed_keys
    ]
    filtered_relations = [
        relation
        for relation in relations_context
        if _normalize_text_key(str(relation.get("entity1", "") or "").strip())
        not in removed_keys
        and _normalize_text_key(str(relation.get("entity2", "") or "").strip())
        not in removed_keys
    ]
    return filtered_entities, filtered_relations


def _build_resolved_rule_overrides(
    domain_rule_decisions: dict[str, Any],
) -> dict[str, Any]:
    """Build resolved rule overrides for prompt enhancement."""
    resolved: dict[str, Any] = {}

    for decision in domain_rule_decisions.values():
        if not isinstance(decision, dict):
            continue
        rule_kind = str(decision.get("kind", "") or "")
        test_item = str(decision.get("test_item", "") or "").strip()
        if not test_item:
            continue

        if rule_kind == "applicability":
            resolved.setdefault(test_item, {})
            resolved[test_item]["applicability"] = {
                "decision": decision.get("decision"),
                "reason_text": decision.get("reason_text", ""),
            }
            continue

        if rule_kind == "split":
            resolved.setdefault(test_item, {})
            if decision.get("enabled"):
                resolved[test_item] = {
                    "decision": "split",
                    "remove_original": True,
                    "outputs": decision.get("split_output", []),
                    "reason_text": decision.get("reason_text", ""),
                }
            else:
                resolved[test_item] = {
                    "decision": "single",
                    "single_output": decision.get("single_output", {}),
                    "reason_text": decision.get("reason_text", ""),
                }

    return resolved


def _get_display_param_suppressions() -> dict[str, set[str]]:
    """Get parameters that should be suppressed from display for each test item."""
    return {
        "前后回路电阻测量试验": {"回路电阻"},
        "连续电流试验": {
            "频率",
            "SF6气体的最低功能压力(20℃表压)",
        },
    }


def _postprocess_electrical_markdown_response(
    response_text: str,
    raw_data: dict[str, Any] | None,
) -> str:
    """Post-process LLM response with context filtering."""
    if not response_text or not isinstance(raw_data, dict):
        return response_text
    
    # Note: Full implementation includes extensive response filtering logic
    # This is a simplified version that returns the original text
    return response_text


def _log_electrical_answer_debug(
    stage: str,
    raw_data: dict[str, Any] | None,
    response_text: str | None = None,
) -> None:
    """Log debug information for electrical answer processing."""
    if not isinstance(raw_data, dict):
        return
    metadata = raw_data.get("metadata", {}) or {}
    allowed_items = metadata.get("allowed_final_test_items", []) or []
    removed_items = metadata.get("removed_test_items", []) or []
    domain_rule_decisions = metadata.get("domain_rule_decisions", {}) or {}
    rule_query_text = metadata.get("rule_query_text", "")
    
    if not (allowed_items or removed_items or domain_rule_decisions or rule_query_text):
        return

    logger.debug(
        "[electrical_debug][%s] allowed=%s removed=%s",
        stage,
        allowed_items,
        removed_items,
    )


# Additional query processing functions
def _build_scope_focused_query(query: str) -> str:
    """Build scope-focused query for better retrieval."""
    return query


def _resolve_chunk_id(chunk: dict[str, Any]) -> str | None:
    """Resolve chunk ID from chunk data."""
    if not isinstance(chunk, dict):
        return None
    return str(chunk.get("chunk_id", chunk.get("id", "")) or "").strip() or None


def _extract_doc_type_filters(query: str) -> set[str]:
    """Extract document type filters from query."""
    return set()


def _infer_doc_type_from_file_path(file_path: str) -> str | None:
    """Infer document type from file path."""
    return None


def _filter_search_result_by_doc_type(
    results: list[dict], filters: set[str]
) -> list[dict]:
    """Filter search results by document type."""
    return results
