"""Tree override and annotation memory functions.

This module handles loading and merging of human annotation rules
from annotation memory files (memory.json).
"""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import re
from typing import Any
from collections import defaultdict

from lightrag.utils import logger

from .utils import (
    _normalize_text_key,
    _build_override_path_key,
    _coerce_override_path_parts,
    _split_name_and_value,
    _infer_value_source,
    _note_is_remove,
    _resolve_override_param_name,
    _resolve_override_value_text,
    _extract_missing_feature_params,
    _extract_missing_test_items,
    _extract_test_item_detail_params,
)


# Cache for tree override rules
_TREE_OVERRIDE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _load_tree_override_rules_single(
    schema_cfg: dict | None = None, override_path: Path | None = None
) -> dict[str, Any]:
    """Load one annotation source (tree JSON or normalized rules JSON)."""
    if override_path is None:
        return {}
    
    configured_report_keys = {
        _normalize_text_key(str(name))
        for name in (schema_cfg or {}).get("report_types", [])
        if _normalize_text_key(str(name))
    }

    def _resolve_legacy_report_type(
        path_parts: list[str], report_type: str, category: str
    ) -> str:
        report_text = str(report_type or "").strip()
        category_text = str(category or "").strip()
        if configured_report_keys:
            report_key = _normalize_text_key(report_text)
            category_key = _normalize_text_key(category_text)
            if report_key in configured_report_keys:
                return report_text
            if category_key in configured_report_keys:
                return category_text
            if len(path_parts) >= 2:
                first = str(path_parts[0]).strip()
                second = str(path_parts[1]).strip()
                if _normalize_text_key(first) in configured_report_keys:
                    return first
                if _normalize_text_key(second) in configured_report_keys:
                    return second
        if report_text:
            return report_text
        if path_parts:
            return str(path_parts[0]).strip()
        return ""

    cache_key = str(override_path)
    try:
        mtime = override_path.stat().st_mtime
    except OSError:
        return {}
        
    cached = _TREE_OVERRIDE_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        payload = json.loads(override_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read tree override JSON %s: %s", override_path, exc)
        return {}

    if isinstance(payload, dict):
        tests_by_path: dict[str, dict[str, Any]] = {}
        tests_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        add_test_items: list[dict[str, Any]] = []

        if isinstance(payload.get("tests_by_path"), dict):
            for path_key, raw_rule in payload.get("tests_by_path", {}).items():
                if not isinstance(raw_rule, dict):
                    continue
                rule = dict(raw_rule)
                path_parts = _coerce_override_path_parts(rule.get("path_parts"))
                if not path_parts:
                    path_parts = _coerce_override_path_parts(path_key)
                if not path_parts:
                    path_parts = _coerce_override_path_parts(rule.get("path"))
                test_name = str(
                    rule.get("test_name", "") or rule.get("test_item", "") or ""
                ).strip()
                if not test_name and path_parts:
                    test_name = str(path_parts[-1]).strip()
                category = str(rule.get("category", "") or "").strip()
                if not category and len(path_parts) >= 2:
                    category = str(path_parts[-2]).strip()
                report_type = _resolve_legacy_report_type(
                    path_parts, str(rule.get("report_type", "") or ""), category
                )
                rule["test_name"] = test_name
                rule["category"] = category
                rule["report_type"] = report_type
                normalized_path = _build_override_path_key([report_type, category, test_name])
                legacy_path = _build_override_path_key(path_parts) or _normalize_text_key(
                    str(path_key)
                )
                if not normalized_path:
                    normalized_path = legacy_path
                if not normalized_path:
                    continue
                rule["path_key"] = normalized_path
                if not isinstance(rule.get("remove_parameters"), list):
                    rule["remove_parameters"] = []
                if not isinstance(rule.get("remove_rules"), list):
                    rule["remove_rules"] = []
                tests_by_path[normalized_path] = rule
                if legacy_path and legacy_path != normalized_path:
                    tests_by_path[legacy_path] = rule

        if isinstance(payload.get("tests_by_name"), dict):
            for name_key, raw_rules in payload.get("tests_by_name", {}).items():
                if isinstance(raw_rules, dict):
                    raw_rules = [raw_rules]
                if not isinstance(raw_rules, list):
                    continue
                normalized_name = _normalize_text_key(str(name_key))
                for raw_rule in raw_rules:
                    if not isinstance(raw_rule, dict):
                        continue
                    rule = dict(raw_rule)
                    test_name = str(rule.get("test_name", "") or "").strip()
                    if test_name:
                        normalized_name = _normalize_text_key(test_name)
                    if not normalized_name:
                        continue
                    if not isinstance(rule.get("remove_parameters"), list):
                        rule["remove_parameters"] = []
                    if not isinstance(rule.get("remove_rules"), list):
                        rule["remove_rules"] = []
                    tests_by_name[normalized_name].append(rule)

        if isinstance(payload.get("tests"), dict):
            for _, raw_rule in payload.get("tests", {}).items():
                if not isinstance(raw_rule, dict):
                    continue
                rule = dict(raw_rule)
                test_name = str(rule.get("test_name", "") or "").strip()
                if not test_name:
                    continue
                if not isinstance(rule.get("remove_parameters"), list):
                    rule["remove_parameters"] = []
                if not isinstance(rule.get("remove_rules"), list):
                    rule["remove_rules"] = []
                name_key = _normalize_text_key(test_name)
                if name_key:
                    tests_by_name[name_key].append(rule)
                path_value = rule.get("path_parts") or rule.get("path") or rule.get("test_path")
                path_key = _build_override_path_key(_coerce_override_path_parts(path_value))
                category = str(rule.get("category", "") or "").strip()
                report_type = _resolve_legacy_report_type(
                    _coerce_override_path_parts(path_value),
                    str(rule.get("report_type", "") or ""),
                    category,
                )
                rule["report_type"] = report_type
                normalized_path = _build_override_path_key([report_type, category, test_name])
                if normalized_path:
                    path_key = normalized_path
                if path_key:
                    rule["path_key"] = path_key
                    tests_by_path[path_key] = rule
                if bool(rule.get("force_add")):
                    add_test_items.append(rule)

        if isinstance(payload.get("add_test_items"), list):
            add_test_items.extend(
                [item for item in payload.get("add_test_items", []) if isinstance(item, dict)]
            )

        if tests_by_path or tests_by_name or add_test_items:
            rules = {
                "tests_by_path": tests_by_path,
                "tests_by_name": dict(tests_by_name),
                "add_test_items": add_test_items,
            }
            _TREE_OVERRIDE_CACHE[cache_key] = (mtime, rules)
            logger.info(
                "Loaded tree override rules: path=%d, name=%d, additions=%d from %s",
                len(tests_by_path),
                len(tests_by_name),
                len(add_test_items),
                override_path,
            )
            return rules

    root = payload.get("tree", payload)
    if not isinstance(root, dict):
        return {}

    test_rules_by_path: dict[str, dict[str, Any]] = {}
    test_rules_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    generated_add_test_items: list[dict[str, Any]] = []
    generated_add_test_item_keys: set[str] = set()

    def _walk(node: dict[str, Any], ancestors: list[str]) -> None:
        if not isinstance(node, dict):
            return
        node_name = str(node.get("name", "") or "").strip()
        node_note = str(node.get("note", "") or "").strip()
        children = node.get("children", [])
        if not isinstance(children, list):
            children = []
        current_path = [*ancestors, node_name] if node_name else list(ancestors)

        if node_name and len(current_path) == 2 and node_note:
            report_name = str(current_path[0]).strip()
            category_name = str(current_path[-1]).strip()
            for missing_test_name, detail in _extract_missing_test_items(node_note):
                add_key = _build_override_path_key([report_name, category_name, missing_test_name])
                if not add_key or add_key in generated_add_test_item_keys:
                    continue
                generated_add_test_item_keys.add(add_key)
                params_payload = []
                for param_name, value_text in _extract_test_item_detail_params(detail):
                    value_source = _infer_value_source(value_text, value_text)
                    resolved_value_text = _resolve_override_value_text(
                        value_text, value_text, value_source
                    )
                    params_payload.append(
                        {
                            "param_name": param_name,
                            "value_text": resolved_value_text,
                            "value_expr": value_text
                            if value_source in {"user_input", "formula", "default"}
                            else "",
                            "value_source": value_source,
                            "value_type": value_source,
                            "constraints": value_text,
                            "calc_rule": value_text if value_source == "formula" else "",
                            "derive_from_rated": value_text
                            if value_source == "user_input"
                            else "",
                        }
                    )
                generated_add_test_items.append(
                    {
                        "test_item": missing_test_name,
                        "category": category_name,
                        "report_type": report_name,
                        "aliases": [],
                        "acceptance_criteria": "",
                        "note": node_note,
                        "confidence": 1.0,
                        "required_reports": [
                            {
                                "report_type": report_name,
                                "is_required": True,
                                "condition": "",
                            }
                        ],
                        "parameters": params_payload,
                        "rules": [],
                    }
                )

        feature_node = None
        for child in children:
            if isinstance(child, dict) and str(child.get("name", "")).strip() == "特征值":
                feature_node = child
                break

        parent_name = current_path[-2] if len(current_path) >= 2 else ""
        is_test_level_node = len(current_path) >= 3 and parent_name != "特征值"
        if node_name and is_test_level_node and (feature_node is not None or _note_is_remove(node_note)):
            category_name = current_path[-2] if len(current_path) >= 2 else ""
            report_name = _resolve_legacy_report_type(current_path, "", category_name)
            path_key = _build_override_path_key([report_name, category_name, node_name])
            rule = {
                "test_name": node_name,
                "report_type": report_name,
                "category": category_name,
                "path_parts": current_path,
                "path_key": path_key,
                "skip": _note_is_remove(node_note),
                "note": node_note,
                "parameters": [],
                "remove_parameters": [],
                "remove_rules": [],
            }
            if feature_node is not None:
                feature_note = str(feature_node.get("note", "") or "").strip()
                param_nodes = feature_node.get("children", [])
                if isinstance(param_nodes, list):
                    for param_node in param_nodes:
                        if not isinstance(param_node, dict):
                            continue
                        param_raw_name = str(param_node.get("name", "") or "").strip()
                        param_note = str(param_node.get("note", "") or "").strip()
                        if not param_raw_name:
                            continue
                        param_name, value_text = _split_name_and_value(param_raw_name)
                        if _note_is_remove(param_note):
                            if param_name:
                                rule["remove_parameters"].append(param_name)
                            continue
                        resolved_param_name = _resolve_override_param_name(param_name, param_note)
                        value_source = _infer_value_source(value_text, param_note)
                        resolved_value_text = _resolve_override_value_text(
                            value_text, param_note, value_source
                        )
                        value_expr = ""
                        if value_source in {"user_input", "formula", "default"} and param_note:
                            value_expr = param_note
                        rule["parameters"].append(
                            {
                                "param_name": resolved_param_name,
                                "value_text": resolved_value_text,
                                "value_expr": value_expr,
                                "value_source": value_source,
                                "value_type": value_source,
                                "constraints": param_note,
                                "calc_rule": param_note if value_source == "formula" else "",
                                "derive_from_rated": param_note
                                if value_source == "user_input"
                                else "",
                            }
                        )
                existing_param_keys = {
                    _normalize_text_key(str(param.get("param_name", "") or ""))
                    for param in rule["parameters"]
                    if isinstance(param, dict)
                }
                existing_remove_keys = {
                    _normalize_text_key(str(name))
                    for name in rule["remove_parameters"]
                    if _normalize_text_key(str(name))
                }
                for missing_param_name, value_hint in _extract_missing_feature_params(feature_note):
                    normalized_missing = _normalize_text_key(missing_param_name)
                    if not normalized_missing:
                        continue
                    if normalized_missing in existing_remove_keys:
                        continue
                    if normalized_missing in existing_param_keys:
                        continue
                    if _note_is_remove(value_hint):
                        rule["remove_parameters"].append(missing_param_name)
                        existing_remove_keys.add(normalized_missing)
                        continue
                    value_source = _infer_value_source(value_hint, value_hint)
                    resolved_value_text = _resolve_override_value_text(
                        value_hint, value_hint, value_source
                    )
                    rule["parameters"].append(
                        {
                            "param_name": missing_param_name,
                            "value_text": resolved_value_text,
                            "value_expr": value_hint
                            if value_source in {"user_input", "formula", "default"}
                            else "",
                            "value_source": value_source,
                            "value_type": value_source,
                            "constraints": value_hint,
                            "calc_rule": value_hint if value_source == "formula" else "",
                            "derive_from_rated": value_hint
                            if value_source == "user_input"
                            else "",
                        }
                    )
                    existing_param_keys.add(normalized_missing)
            condition_node = None
            for child in children:
                if isinstance(child, dict) and str(child.get("name", "")).strip() == "条件/规则":
                    condition_node = child
                    break
            if condition_node is not None:
                rule_nodes = condition_node.get("children", [])
                if isinstance(rule_nodes, list):
                    for rule_node in rule_nodes:
                        if not isinstance(rule_node, dict):
                            continue
                        raw_rule_name = str(rule_node.get("name", "") or "").strip()
                        rule_note = str(rule_node.get("note", "") or "").strip()
                        if not raw_rule_name or not _note_is_remove(rule_note):
                            continue
                        cleaned_rule_name, cleaned_rule_payload = _split_name_and_value(
                            raw_rule_name
                        )
                        if (
                            _normalize_text_key(cleaned_rule_name) == "规则"
                            and cleaned_rule_payload
                        ):
                            cleaned_rule_name = (
                                cleaned_rule_payload.split("|", 1)[0].strip()
                            )
                        if cleaned_rule_name:
                            rule["remove_rules"].append(cleaned_rule_name)
            if path_key:
                test_rules_by_path[path_key] = rule
            test_rules_by_name[_normalize_text_key(node_name)].append(rule)

        for child in children:
            if isinstance(child, dict):
                _walk(child, current_path)

    _walk(root, [])
    rules = {
        "tests_by_path": test_rules_by_path,
        "tests_by_name": dict(test_rules_by_name),
        "add_test_items": generated_add_test_items,
    }
    _TREE_OVERRIDE_CACHE[cache_key] = (mtime, rules)
    logger.info(
        "Loaded tree override rules: path=%d, name=%d from %s",
        len(test_rules_by_path),
        len(test_rules_by_name),
        override_path,
    )
    return rules


def _find_override_rule_by_test_name(
    tree_tests_by_path: dict[str, Any],
    report_type: str,
    category: str,
    test_name: str,
) -> dict[str, Any] | None:
    """Find override rule by test name."""
    if not isinstance(tree_tests_by_path, dict):
        return None
    path_key = _build_override_path_key([report_type, category, test_name])
    if path_key and isinstance(tree_tests_by_path.get(path_key), dict):
        return tree_tests_by_path.get(path_key)
    normalized_name = _normalize_text_key(test_name)
    for rule in tree_tests_by_path.values():
        if not isinstance(rule, dict):
            continue
        if _normalize_text_key(str(rule.get("report_type", "") or "")) != _normalize_text_key(
            report_type
        ):
            continue
        if _normalize_text_key(str(rule.get("category", "") or "")) != _normalize_text_key(
            category
        ):
            continue
        if _normalize_text_key(str(rule.get("test_name", "") or "")) == normalized_name:
            return rule
    return None


def _merge_annotation_rules(
    base_rules: dict[str, Any], patch_rules: dict[str, Any]
) -> dict[str, Any]:
    """Merge patch rules into base rules."""
    if not base_rules:
        base_rules = {}
    merged = {
        "tests_by_path": dict(base_rules.get("tests_by_path", {}) or {}),
        "tests_by_name": dict(base_rules.get("tests_by_name", {}) or {}),
        "add_test_items": list(base_rules.get("add_test_items", []) or []),
    }
    patch_tests_by_path = patch_rules.get("tests_by_path", {}) or {}
    for path_key, raw_patch_rule in patch_tests_by_path.items():
        if not isinstance(raw_patch_rule, dict):
            continue
        patch_rule = dict(raw_patch_rule)
        existing_rule = merged["tests_by_path"].get(path_key)
        if not isinstance(existing_rule, dict):
            merged["tests_by_path"][path_key] = patch_rule
            continue
        combined_rule = dict(existing_rule)
        for top_key in (
            "test_name",
            "category",
            "report_type",
            "path_parts",
            "path_key",
            "note",
            "aliases",
            "acceptance_criteria",
            "required_reports",
            "parameters_mode",
            "template_only",
            "skip",
        ):
            if top_key in patch_rule and patch_rule.get(top_key) not in (None, ""):
                combined_rule[top_key] = patch_rule.get(top_key)
        remove_param_seen = {
            _normalize_text_key(str(name))
            for name in (existing_rule.get("remove_parameters", []) or [])
            if _normalize_text_key(str(name))
        }
        combined_remove_params = list(existing_rule.get("remove_parameters", []) or [])
        for raw_name in patch_rule.get("remove_parameters", []) or []:
            name_key = _normalize_text_key(str(raw_name))
            if not name_key or name_key in remove_param_seen:
                continue
            remove_param_seen.add(name_key)
            combined_remove_params.append(raw_name)
        combined_rule["remove_parameters"] = combined_remove_params

        remove_rule_seen = {
            _normalize_text_key(str(name))
            for name in (existing_rule.get("remove_rules", []) or [])
            if _normalize_text_key(str(name))
        }
        combined_remove_rules = list(existing_rule.get("remove_rules", []) or [])
        for raw_name in patch_rule.get("remove_rules", []) or []:
            name_key = _normalize_text_key(str(raw_name))
            if not name_key or name_key in remove_rule_seen:
                continue
            remove_rule_seen.add(name_key)
            combined_remove_rules.append(raw_name)
        combined_rule["remove_rules"] = combined_remove_rules

        existing_params = list(existing_rule.get("parameters", []) or [])
        param_index: dict[str, int] = {}
        for idx, raw_param in enumerate(existing_params):
            if not isinstance(raw_param, dict):
                continue
            param_key = _normalize_text_key(
                str(raw_param.get("param_key", "") or raw_param.get("param_name", ""))
            )
            if param_key:
                param_index[param_key] = idx
        for raw_patch_param in patch_rule.get("parameters", []) or []:
            if not isinstance(raw_patch_param, dict):
                continue
            patch_param = dict(raw_patch_param)
            param_key = _normalize_text_key(
                str(patch_param.get("param_key", "") or patch_param.get("param_name", ""))
            )
            if param_key and param_key in param_index:
                existing_idx = param_index[param_key]
                merged_param = dict(existing_params[existing_idx])
                merged_param.update(patch_param)
                existing_params[existing_idx] = merged_param
            else:
                existing_params.append(patch_param)
                if param_key:
                    param_index[param_key] = len(existing_params) - 1
        combined_rule["parameters"] = existing_params
        merged["tests_by_path"][path_key] = combined_rule

    combined_add_items = list(merged.get("add_test_items", []) or [])
    add_index: dict[str, int] = {}
    for idx, add_item in enumerate(combined_add_items):
        if not isinstance(add_item, dict):
            continue
        add_key = _build_override_path_key(
            [
                str(add_item.get("report_type", "") or ""),
                str(add_item.get("category", "") or ""),
                str(add_item.get("test_item", "") or add_item.get("test_name", "") or ""),
            ]
        )
        if add_key:
            add_index[add_key] = idx
    for raw_patch_add in patch_rules.get("add_test_items", []) or []:
        if not isinstance(raw_patch_add, dict):
            continue
        patch_add = dict(raw_patch_add)
        add_key = _build_override_path_key(
            [
                str(patch_add.get("report_type", "") or ""),
                str(patch_add.get("category", "") or ""),
                str(patch_add.get("test_item", "") or patch_add.get("test_name", "") or ""),
            ]
        )
        if add_key and add_key in add_index:
            existing_idx = add_index[add_key]
            merged_add = dict(combined_add_items[existing_idx])
            merged_add.update(patch_add)
            existing_params = list(combined_add_items[existing_idx].get("parameters", []) or [])
            param_index: dict[str, int] = {}
            for param_idx, raw_param in enumerate(existing_params):
                if not isinstance(raw_param, dict):
                    continue
                param_key = _normalize_text_key(
                    str(raw_param.get("param_key", "") or raw_param.get("param_name", ""))
                )
                if param_key:
                    param_index[param_key] = param_idx
            for raw_patch_param in patch_add.get("parameters", []) or []:
                if not isinstance(raw_patch_param, dict):
                    continue
                patch_param = dict(raw_patch_param)
                param_key = _normalize_text_key(
                    str(patch_param.get("param_key", "") or patch_param.get("param_name", ""))
                )
                if param_key and param_key in param_index:
                    existing_params[param_index[param_key]] = patch_param
                else:
                    existing_params.append(patch_param)
                    if param_key:
                        param_index[param_key] = len(existing_params) - 1
            merged_add["parameters"] = existing_params
            combined_add_items[existing_idx] = merged_add
        else:
            combined_add_items.append(patch_add)
            if add_key:
                add_index[add_key] = len(combined_add_items) - 1
    merged["add_test_items"] = combined_add_items

    tests_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in (merged.get("tests_by_path", {}) or {}).values():
        if not isinstance(rule, dict):
            continue
        normalized_name = _normalize_text_key(str(rule.get("test_name", "") or ""))
        if normalized_name:
            tests_by_name[normalized_name].append(rule)
    merged["tests_by_name"] = dict(tests_by_name)
    return merged


def _resolve_annotation_source_paths(schema_cfg: dict | None = None) -> tuple[Path | None, list[Path]]:
    """Resolve annotation memory and source JSON paths."""
    schema_cfg = schema_cfg or {}
    memory_path_text = str(schema_cfg.get("annotation_memory_path", "") or "").strip()
    memory_path = Path(memory_path_text).expanduser() if memory_path_text else None
    if memory_path and not memory_path.is_absolute():
        memory_path = Path.cwd() / memory_path

    source_paths: list[Path] = []
    raw_source_paths = str(schema_cfg.get("annotation_source_json_paths", "") or "").strip()
    if raw_source_paths:
        for raw_path in raw_source_paths.split(","):
            text = raw_path.strip()
            if not text:
                continue
            path = Path(text).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                source_paths.append(path)
            else:
                logger.warning("Annotation source JSON not found: %s", path)

    return memory_path, source_paths


def _load_tree_override_rules(schema_cfg: dict | None = None) -> dict[str, Any]:
    """Load cumulative annotation memory and merge optional incremental sources."""
    schema_cfg = schema_cfg or {}
    memory_path, source_paths = _resolve_annotation_source_paths(schema_cfg)
    merged_rules: dict[str, Any] = {}
    if memory_path and memory_path.exists():
        memory_rules = _load_tree_override_rules_single(schema_cfg, memory_path)
        merged_rules = _merge_annotation_rules(merged_rules, memory_rules)

    for source_path in source_paths:
        if memory_path and source_path == memory_path:
            continue
        source_rules = _load_tree_override_rules_single(schema_cfg, source_path)
        merged_rules = _merge_annotation_rules(merged_rules, source_rules)

    auto_merge_raw = schema_cfg.get("annotation_auto_merge_to_memory", False)
    auto_merge_to_memory = (
        str(auto_merge_raw).strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(auto_merge_raw, str)
        else bool(auto_merge_raw)
    )
    if (
        auto_merge_to_memory
        and memory_path is not None
        and (source_paths or memory_path.exists())
        and merged_rules
    ):
        try:
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            memory_path.write_text(
                json.dumps(merged_rules, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to persist annotation memory %s: %s", memory_path, exc)
    return merged_rules
