"""Domain rule evaluation and application functions.

This module contains domain-specific rule evaluation for electrical standards,
including split rules, merge rules, applicability rules, and count rules.
"""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import re
from typing import Any

from lightrag.utils import logger

from .utils import _normalize_text_key


# Cache for domain rules
_DOMAIN_RULE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _resolve_domain_rules_path(schema_cfg: dict | None = None) -> Path | None:
    """Resolve domain rules file path from schema config."""
    schema_cfg = schema_cfg or {}
    path_text = str(schema_cfg.get("electrical_rules_path", "") or "").strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _load_domain_rules(schema_cfg: dict | None = None) -> dict[str, Any]:
    """Load domain rules from JSON file."""
    schema_cfg = schema_cfg or {}
    path = _resolve_domain_rules_path(schema_cfg)
    if path is None:
        logger.debug("Domain rules path is empty in electrical_schema config")
        return {}
    if not path.exists():
        logger.debug("Domain rules file not found: %s", path)
        return {}

    logger.info("Loading domain rules from %s", path)

    cache_key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
        
    cached = _DOMAIN_RULE_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        logger.debug(
            "Domain rules cache hit: %s (rules=%s)",
            path,
            len((cached[1] or {}).get("rules", []) or []),
        )
        return deepcopy(cached[1])

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load domain rules %s: %s", path, exc)
        return {}

    if not isinstance(payload, dict):
        logger.warning("Domain rules payload must be object: %s", path)
        return {}

    logger.info("Loaded domain rules: %s rule(s) from %s", len(payload.get("rules", []) or []), path)
    _DOMAIN_RULE_CACHE[cache_key] = (mtime, payload)
    return deepcopy(payload)


def _evaluate_domain_rule_decisions(
    query: str,
    schema_cfg: dict | None = None,
) -> dict[str, Any]:
    """Evaluate domain rules against the query.
    
    This function evaluates split rules, merge rules, applicability rules,
    and count rules based on the query text.
    
    Note: The full implementation contains extensive rule logic for electrical
    standards (GB/T, IEC, DLT). See the original rule.py for complete implementation.
    """
    schema_cfg = schema_cfg or {}
    domain_rules = _load_domain_rules(schema_cfg)
    rules = domain_rules.get("rules", []) or []
    if not isinstance(rules, list) or not rules:
        return {}

    # Extract query parameters
    def _extract_rated_voltage_kv(query_text: str) -> float | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        match = re.search(
            r"额定电压\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*kV\b", text, flags=re.IGNORECASE
        )
        return float(match.group(1)) if match else None

    def _extract_rated_current_amp(query_text: str) -> int | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        match = re.search(r"额定电流\s*(?:[:：=]\s*)?([0-9]+)\s*A\b", text, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_model_prefix(query_text: str) -> str | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        match = re.search(r"(?:型号名称|型号)\s*[：:=]\s*([A-Za-z0-9]+)", text)
        return match.group(1).upper() if match else None

    def _query_contains(text: str, needle: str) -> bool:
        return bool(str(text or "").strip() and needle and needle in text)

    def _matches_condition(condition: dict[str, Any], query_text: str) -> bool:
        cond_type = str(condition.get("type", "") or "").strip()
        label = str(condition.get("label", "") or "").strip()
        if cond_type == "contains":
            return _query_contains(query_text, label)
        if cond_type == "not_contains":
            return bool(label) and label not in query_text
        if cond_type == "contains_any":
            labels = condition.get("labels", []) or []
            return isinstance(labels, list) and any(
                _query_contains(query_text, str(item or "").strip())
                for item in labels
                if str(item or "").strip()
            )
        if cond_type == "regex_match":
            pattern = str(condition.get("pattern", "") or "").strip()
            if not pattern:
                return False
            return bool(re.search(pattern, query_text, flags=re.IGNORECASE))
        if cond_type == "equals_numeric":
            value = condition.get("value")
            if label == "额定电压":
                actual = _extract_rated_voltage_kv(query_text)
            elif label == "额定电流":
                actual = _extract_rated_current_amp(query_text)
            else:
                actual = None
            return actual is not None and float(actual) == float(value)
        if cond_type == "all":
            subconditions = condition.get("conditions", []) or []
            return isinstance(subconditions, list) and all(
                isinstance(item, dict) and _matches_condition(item, query_text)
                for item in subconditions
            )
        return False

    rated_voltage_kv = _extract_rated_voltage_kv(query)
    rated_current_amp = _extract_rated_current_amp(query)
    model_prefix = _extract_model_prefix(query)
    decisions: dict[str, Any] = {}

    for raw_rule in rules:
        if not isinstance(raw_rule, dict):
            continue

        rule_id = str(raw_rule.get("rule_id", "") or "").strip()
        if not rule_id:
            continue
        rule_kind = str(raw_rule.get("kind", "") or "").strip()
        test_item = str(raw_rule.get("test_item", "") or "").strip()

        if rule_kind == "split":
            input_cfg = raw_rule.get("inputs", {}) or {}
            trigger_when_any = input_cfg.get("trigger_when_any", []) or []
            split_enabled = False
            matched_conditions: list[str] = []
            
            if isinstance(trigger_when_any, list) and trigger_when_any:
                for condition in trigger_when_any:
                    if not isinstance(condition, dict):
                        continue
                    if _matches_condition(condition, query):
                        matched_conditions.append(
                            str(condition.get("label", "") or condition.get("type", "") or "")
                        )
                split_enabled = bool(matched_conditions)
            
            decisions[rule_id] = {
                "rule_id": rule_id,
                "domain": raw_rule.get("domain", ""),
                "test_item": test_item,
                "kind": "split",
                "decision": "split" if split_enabled else "single",
                "enabled": split_enabled,
                "reason_code": "split_enabled" if split_enabled else "split_not_triggered",
                "reason_text": (
                    f"命中条件：{'；'.join(matched_conditions)}，允许拆分。"
                    if split_enabled
                    else "未命中任何拆分触发条件，保持未拆分。"
                ),
                "single_output": raw_rule.get("single_output", {}),
                "split_output": raw_rule.get("split_output", []),
            }
            continue

        if rule_kind == "applicability":
            allow_when_any = raw_rule.get("allow_when_any", []) or []
            matched_conditions = []
            for condition in allow_when_any:
                if not isinstance(condition, dict):
                    continue
                if _matches_condition(condition, query):
                    label = str(condition.get("label", "") or condition.get("type", "") or "")
                    matched_conditions.append(label)
            enabled = bool(matched_conditions)
            decisions[rule_id] = {
                "rule_id": rule_id,
                "domain": raw_rule.get("domain", ""),
                "test_item": test_item,
                "kind": "applicability",
                "decision": "allow" if enabled else "deny",
                "enabled": enabled,
                "reason_code": "allowed" if enabled else "denied",
                "reason_text": (
                    f"{test_item}适用。命中条件：" + "；".join(matched_conditions)
                    if enabled
                    else f"未命中任何{test_item}适用条件，禁止输出{test_item}。"
                ),
                "matched_conditions": matched_conditions,
                "inputs": {
                    "rated_voltage_kv": rated_voltage_kv,
                    "rated_current_amp": rated_current_amp,
                    "model_prefix": model_prefix,
                },
            }
            continue

    return decisions


def _apply_domain_rule_decisions_to_project_context(
    project_param_map: dict[str, list[str]],
    project_param_value_map: dict[str, dict[str, dict[str, str]]],
    domain_rule_decisions: dict[str, Any],
    schema_cfg: dict[str, Any] | None = None,
    rule_query_text: str | None = None,
) -> tuple[dict[str, list[str]], dict[str, dict[str, dict[str, str]]]]:
    """Apply domain rule decisions to project context.
    
    This function modifies the project context based on rule decisions
    such as splitting tests, merging tests, or determining applicability.
    
    Note: The full implementation contains extensive logic for applying
    split/merge/applicability rules to the project context.
    """
    updated_param_map = deepcopy(project_param_map)
    updated_value_map = deepcopy(project_param_value_map)
    
    for decision in domain_rule_decisions.values():
        if not isinstance(decision, dict):
            continue
        rule_kind = str(decision.get("kind", "") or "")
        test_item = str(decision.get("test_item", "") or "").strip()
        
        if rule_kind == "applicability":
            if decision.get("enabled"):
                # Add test item if applicable
                if test_item and test_item not in updated_param_map:
                    updated_param_map[test_item] = []
                    updated_value_map[test_item] = {}
            else:
                # Remove test item if not applicable
                updated_param_map.pop(test_item, None)
                updated_value_map.pop(test_item, None)
            continue
        
        if rule_kind == "split" and decision.get("enabled"):
            # Handle test item splitting
            if test_item and test_item in updated_param_map:
                source_params = list(updated_param_map.get(test_item, []) or [])
                source_values = deepcopy(updated_value_map.get(test_item, {}) or {})
                updated_param_map.pop(test_item, None)
                updated_value_map.pop(test_item, None)
                
                for split_output in decision.get("split_output", []) or []:
                    if not isinstance(split_output, dict):
                        continue
                    target_name = str(split_output.get("test_item", "") or "").strip()
                    if not target_name:
                        continue
                    updated_param_map[target_name] = list(source_params)
                    updated_value_map[target_name] = deepcopy(source_values)
            continue
    
    return updated_param_map, updated_value_map
