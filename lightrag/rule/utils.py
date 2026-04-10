"""Utility functions for the rule module.

This module contains common helper functions used across the rule module,
including text normalization, ID generation, and data extraction utilities.
"""

from __future__ import annotations
from copy import deepcopy
import json
import re
from typing import Any

from lightrag.utils import logger, compute_mdhash_id, remove_think_tags
from lightrag.constants import GRAPH_FIELD_SEP


def _normalize_text_key(value: str) -> str:
    """
    "绝缘试验"        →   "绝缘试验"
    " 绝缘 试验 "     →   "绝缘试验"
    "绝缘   试验\n"   →   "绝缘试验"
    " 工频 耐受 电压 " →   "工频耐受电压"
    None              →   ""
    用来生成统一的实体 / 节点 KEY！
    """
    return re.sub(r"\s+", "", (value or "").strip())


def _build_override_path_key(parts: list[str]) -> str:
    return " > ".join(_normalize_text_key(part) for part in parts if _normalize_text_key(part))


def _coerce_override_path_parts(path_value: Any) -> list[str]:
    if isinstance(path_value, list):
        return [str(part) for part in path_value if str(part).strip()]
    if isinstance(path_value, str):
        text = path_value.strip()
        if not text:
            return []
        if ">" in text:
            return [part.strip() for part in text.split(">") if part.strip()]
        if "/" in text:
            return [part.strip() for part in text.split("/") if part.strip()]
        return [text]
    return []


def _split_name_and_value(raw_name: str) -> tuple[str, str]:
    text = (raw_name or "").strip()
    for sep in ("：", "﹕", "∶", ":"):
        if sep in text:
            left, right = text.split(sep, 1)
            return left.strip(), right.strip()
    return text, ""


def _infer_value_source(value_text: str, note_text: str) -> str:
    merged = f"{value_text} {note_text}".strip()
    if any(
        token in merged
        for token in ("用户录入", "用户输入", "用户提供", "客户录入", "客户输入", "客户提供")
    ):
        return "user_input"
    if any(token in merged for token in ("计算", "公式", "%", "×", "*")):
        return "formula"
    if "默认" in merged:
        return "default"
    return "standard"


def _note_is_remove(note: str) -> bool:
    text = (note or "").strip()
    return any(
        marker in text
        for marker in (
            "这个不要",
            "这个不需要",
            "不需要了",
            "不需要",
            "这个不要了",
            "不涉及",
            "无关",
            "可删除",
            "删除",
            "错",
        )
    )


def _classify_query_value_resolution_mode(
    value_text: str,
    value_source: str,
    value_expr: str = "",
    constraints: str = "",
    calc_rule: str = "",
    derive_from_rated: str = "",
) -> str:
    """Classify whether a graph parameter value can be used as a final answer directly.

    Returns one of:
      - graph_final: safe to output directly
      - needs_user_input: user-supplied device input must be injected
      - needs_formula: formula/calculation must be resolved
      - needs_condition: branch/lookup/default rule must be resolved
      - missing: no usable graph value
    """
    merged = " ".join(
        part.strip()
        for part in (
            value_text,
            value_expr,
            constraints,
            calc_rule,
            derive_from_rated,
        )
        if str(part or "").strip()
    )
    if not merged:
        return "missing"

    if value_source == "user_input" or any(
        token in merged
        for token in ("用户录入", "用户输入", "用户提供", "客户录入", "客户输入", "客户提供")
    ):
        return "needs_user_input"

    if value_source == "formula" or calc_rule.strip():
        return "needs_formula"

    formula_markers = ("%", "×", "*", "√", "/")
    formula_context_markers = ("额定电压", "额定电流", "Ur", "Isc", "根3", "公式", "计算")
    if any(marker in merged for marker in formula_markers) and any(
        marker in merged for marker in formula_context_markers
    ):
        return "needs_formula"

    conditional_markers = (
        "按表",
        "查表",
        "根据",
        "依据",
        "由用户",
        "取决于",
        "与",
        "一致",
        "见表",
        "见图",
        "以上",
        "以下",
        "~",
        "范围",
        "时",
        "若",
        "如果",
        "则",
        "或",
        "和/或",
        "影响",
        "有关",
        "分箱",
        "共箱",
        "单相试验",
        "三相试验",
        "可单相或三相",
        "默认为",
    )
    conditional_patterns = (
        r"S\d+\s*或\s*S\d+",
        r"额定\s*\d+\s*Hz.*做\s*\d+\s*Hz",
        r"\d+(?:\.\d+)?kV.*单相.*三相",
        r"(?:用户|客户).*(?:决定|选择|提供)",
    )
    if derive_from_rated.strip() or any(
        marker in merged for marker in conditional_markers
    ) or any(re.search(pattern, merged) for pattern in conditional_patterns):
        return "needs_condition"

    return "graph_final"


def _extract_user_input_text(note_text: str) -> str:
    text = (note_text or "").strip()
    if not text:
        return ""
    idx = -1
    for token in ("用户录入", "用户输入", "用户提供", "客户录入", "客户输入", "客户提供"):
        token_idx = text.find(token)
        if token_idx >= 0 and (idx < 0 or token_idx < idx):
            idx = token_idx
    if idx < 0:
        return ""
    user_text = text[idx:]
    user_text = re.sub(r"^[：:，,。.、\-\s]+", "", user_text)
    return user_text.strip()


def _extract_corrected_value_text(note_text: str) -> str:
    text = (note_text or "").strip()
    if not text:
        return ""
    markers = ("应当为", "应改为", "应改成", "应该为", "应为", "改为", "改成", "修改为")
    for marker in markers:
        idx = text.find(marker)
        if idx < 0:
            continue
        corrected = text[idx + len(marker) :].strip()
        corrected = re.sub(r"^[：:，,。.、\-\s]+", "", corrected)
        if corrected:
            return corrected
    return ""


def _extract_corrected_param_name(note_text: str) -> str:
    text = (note_text or "").strip()
    if not text:
        return ""
    patterns = (
        r"(?:特征值名称错误|参数名称错误|名称错误)[，,:： ]*(?:应当为|应为|改为|修改为)([^，,。；;]+)",
        r"(?:应当为|应为|改为|修改为)([^，,。；;]+)",
    )
    for pattern in patterns:
        matched = re.search(pattern, text)
        if not matched:
            continue
        name = (matched.group(1) or "").strip()
        name = re.sub(r"^[：:，,。.、\-\s]+", "", name)
        if not name:
            continue
        if any(token in name for token in ("用户录入", "优选值", "额定")) and "试验" not in name:
            continue
        if len(name) > 24:
            continue
        return name
    return ""


def _extract_missing_feature_params(note_text: str) -> list[tuple[str, str]]:
    text = (note_text or "").strip()
    if not text:
        return []
    items: list[tuple[str, str]] = []
    known_prefixes = (
        "首开极系数kpp",
        "SF6气体的最低功能压力(20℃表压)",
        "SF6气体的额定压力(20℃表压)",
        "额定短路关合电流",
        "额定短路开断电流",
        "最短分闸时间",
        "发电机额定容量",
        "线路侧波阻抗",
        "外壳是否带电",
        "断路器等级",
        "操作顺序",
        "试验项数",
        "试验相数",
        "试验次数",
        "试验电流kA",
        "试验电流A",
        "试验电流",
        "试验电压",
        "额定频率",
        "结构特征",
        "直流分量",
        "故障类型",
        "不均匀系数",
        "时间常数",
        "关合电流",
        "介质性质",
        "电压极性",
    )
    segments = [seg.strip() for seg in re.split(r"[；;\n]+", text) if seg.strip()]
    for segment in segments:
        if "缺特征值" in segment or "缺少特征值" in segment:
            matched = re.search(r"缺(?:少)?特征值[：:]\s*(.+)", segment)
            payload = matched.group(1).strip() if matched else ""
            if not payload:
                continue
            payload = re.sub(r"^[:：，,\s]+", "", payload).strip()
            if not payload:
                continue
            param_name = ""
            value_hint = ""
            for sep in ("，", ",", ":", "："):
                if sep in payload:
                    left, right = payload.split(sep, 1)
                    left = left.strip()
                    right = right.strip()
                    if left:
                        param_name = left
                        value_hint = right
                        break
            if not param_name:
                param_name = payload.strip()
                value_hint = ""
            if param_name:
                items.append((param_name, value_hint))
            continue

        short_missing = re.match(r"^缺(?:少)?([^：:，,]+)[：:，,]\s*(.+)$", segment)
        if not short_missing:
            short_missing = re.match(r"^缺(?:少)?([^为是]+?)(?:为|是)\s*(.+)$", segment)
        if short_missing:
            param_name = (short_missing.group(1) or "").strip()
            value_hint = (short_missing.group(2) or "").strip()
            if not param_name:
                continue
            if "特征值" in param_name:
                continue
            if "试验" in param_name and "次数" not in param_name and "相数" not in param_name:
                if not any(token in param_name for token in ("电压", "电流", "频率", "状态", "部位", "介质", "极性", "时间", "类别", "顺序", "项数")):
                    continue
            items.append((param_name, value_hint))
            continue

        matched_prefix = None
        payload = ""
        stripped_segment = segment
        if stripped_segment.startswith("缺少"):
            stripped_segment = stripped_segment[2:].strip()
        elif stripped_segment.startswith("缺"):
            stripped_segment = stripped_segment[1:].strip()
        for prefix in known_prefixes:
            if stripped_segment.startswith(prefix):
                matched_prefix = prefix
                payload = stripped_segment[len(prefix) :].strip("：:，, ")
                break
        if matched_prefix:
            items.append((matched_prefix, payload))
            continue

        explicit_assignment = re.match(r"^([^：:，,]+)[：:，,]\s*(.+)$", segment)
        if explicit_assignment:
            param_name = (explicit_assignment.group(1) or "").strip()
            value_hint = (explicit_assignment.group(2) or "").strip()
            if param_name in known_prefixes and value_hint:
                items.append((param_name, value_hint))
    return items


def _extract_missing_test_items(note_text: str) -> list[tuple[str, str]]:
    text = (note_text or "").strip()
    if not text:
        return []
    items: list[tuple[str, str]] = []
    for matched in re.finditer(r"缺(?:少)?([^\s，,；;。]*(?:试验|试验\([^)]+\)))", text):
        test_name = (matched.group(1) or "").strip()
        if not test_name:
            continue
        detail = ""
        tail = text[matched.end() :]
        detail_matched = re.search(
            r"(?:全量)?特征值(?:应该)?为\s*([^)；;。]+(?:\([^)]*\)[^)；;。]*)?)", tail
        )
        if not detail_matched:
            detail_matched = re.search(r"特征值[，,:：]\s*(.+?)\s*(?:；|。|$)", tail)
        if detail_matched:
            detail = detail_matched.group(1).strip()
        items.append((test_name, detail))
    return items


def _extract_test_item_detail_params(detail_text: str) -> list[tuple[str, str]]:
    text = (detail_text or "").strip()
    if not text:
        return []
    text = re.sub(r"^(?:全量)?特征值(?:应该)?为", "", text).strip()
    if not text:
        return []
    if all(token not in text for token in ("（", "）", "(", ")", "，", ",", ":")):
        return [(seg.strip(), "") for seg in re.split(r"[、；;\n]+", text) if seg.strip()]
    if all(token not in text for token in ("（", "）", "(", ")", ":")) and "。" in text:
        list_like = [seg.strip() for seg in re.split(r"[、，,；;\n]+", text) if seg.strip()]
        if len(list_like) >= 2:
            return [(seg, "") for seg in list_like]
    params: list[tuple[str, str]] = []
    for segment in re.split(r"[、；;\n]+", text):
        seg = segment.strip().strip("，,。")
        if not seg:
            continue
        param_name = seg
        value_text = ""
        if "（" in seg and "）" in seg:
            left, right = seg.split("（", 1)
            param_name = left.strip()
            value_text = right.rsplit("）", 1)[0].strip()
        elif "(" in seg and ")" in seg:
            left, right = seg.split("(", 1)
            param_name = left.strip()
            value_text = right.rsplit(")", 1)[0].strip()
        elif "，" in seg:
            left, right = seg.split("，", 1)
            param_name = left.strip()
            value_text = right.strip()
        elif "," in seg:
            left, right = seg.split(",", 1)
            param_name = left.strip()
            value_text = right.strip()
        if "（" in param_name and "）" in param_name:
            param_name = param_name.split("（", 1)[0].strip()
        if "(" in param_name and ")" in param_name:
            param_name = param_name.split("(", 1)[0].strip()
        if param_name:
            params.append((param_name, value_text))
    return params


def _sanitize_value_text(value_text: str) -> str:
    text = (value_text or "").strip()
    if not text:
        return ""
    text = re.sub(
        r"(?:我们一般用|一般我们用|我们通常用|通常我们用|我们用|一般用)\s*",
        "",
        text,
    ).strip()

    def _is_symbolic_placeholder(seg: str) -> bool:
        s = (seg or "").strip()
        if not s:
            return False
        if re.fullmatch(r"[a-z_][a-z0-9_]*", s):
            return True
        if re.fullmatch(r"(?:i|u|t|k|r)_[a-z0-9_]+", s):
            return True
        return False

    def _normalize_symbolic_value(seg: str) -> str:
        s = (seg or "").strip()
        mapping = {
            "rated_operation_sequence": "额定操作顺序（用户录入）",
            "rated_voltage": "额定电压（用户录入）",
            "i_sc": "额定短路开断电流（用户录入）",
            "ur kv": "额定电压（用户录入）",
            "u_r kv": "额定电压（用户录入）",
            "u_r": "额定电压（用户录入）",
            "ur": "额定电压（用户录入）",
            "rated_short_circuit_current": "额定短路开断电流（用户录入）",
            "rated_short_circuit_current ka": "额定短路开断电流（用户录入）",
            "rated_short_circuit_breaking_current": "额定短路开断电流（用户录入）",
            "rated_short_circuit_breaking_current ka": "额定短路开断电流（用户录入）",
        }
        return mapping.get(s.lower(), s)

    def _contains_chinese(seg: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", seg or ""))

    def _is_english_formula_like(seg: str) -> bool:
        s = (seg or "").strip()
        if not s:
            return False
        if _contains_chinese(s):
            return False
        has_alpha = bool(re.search(r"[A-Za-z]", s))
        has_formula_symbol = bool(re.search(r"[<>=*/^_]", s))
        return has_alpha and has_formula_symbol

    def _segment_quality_score(seg: str) -> tuple[int, int, int, int]:
        s = (seg or "").strip()
        if not s:
            return (-100, 0, 0, 0)
        score = 0
        if any(token in s for token in ("用户录入", "默认为", "默认", "优选值")):
            score += 40
        if _contains_chinese(s):
            score += 12
        if any(token in s for token in ("C1", "C2", "三相", "单相", "对地", "断口", "主回路", "正极性", "负极性")):
            score += 8
        if any(token in s for token in ("应当为", "应为", "改为", "修改为")):
            score += 30
        if "表" in s and any(token in s for token in ("按", "查", "见", "数值")):
            score += 6
        if _is_english_formula_like(s):
            score -= 16
        if re.fullmatch(r"(?:u|ur|u_r|rated_voltage|i_sc|rated_short_circuit_current)(?:\s*(?:kv|ka|v|a))?", s, flags=re.IGNORECASE):
            score -= 24
        if len(s) > 120:
            score -= 8
        return (score, -len(s), int(_contains_chinese(s)), int(bool(re.search(r"\d", s))))

    segments = [seg.strip() for seg in re.split(r"\s*/\s*", text) if seg.strip()]
    cleaned_segments: list[str] = []
    corrected_segments: list[str] = []
    error_markers = (
        "特征值不对",
        "特征值提取错误",
        "参数提取错误",
        "参数错误",
        "名称错误",
        "提取错误",
    )
    for seg in segments:
        cleaned = seg
        marker_hit = False
        for marker in error_markers:
            if marker in cleaned:
                marker_hit = True
                cleaned = cleaned.split(marker, 1)[1].strip()
        cleaned = re.sub(r"^[：:，,。.、\-\s]+", "", cleaned)
        if cleaned:
            cleaned_segments.append(cleaned)
            if marker_hit:
                corrected_segments.append(cleaned)

    if corrected_segments:
        segments = corrected_segments
    elif cleaned_segments:
        segments = cleaned_segments

    preferred_segments = [
        seg for seg in segments if any(token in seg for token in ("默认为", "默认", "用户录入"))
    ]
    if preferred_segments:
        text = preferred_segments[0]
        segments = [text]

    table_segments = [
        seg
        for seg in segments
        if "表" in seg and any(token in seg for token in ("提取", "查表", "见表", "按表"))
    ]
    if table_segments:
        text = table_segments[0]
    elif segments:
        best_segment = max(segments, key=_segment_quality_score)
        if _is_symbolic_placeholder(best_segment) and len(segments) >= 2:
            text = segments[0]
        else:
            text = best_segment

    text = re.sub(r"\s+", " ", text).strip()
    text = _normalize_symbolic_value(text)
    text = re.sub(r"\s*[,，]\s*(?:rated_|ur\b|u_r\b|i_sc\b).*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"%\s*%", "%", text)
    text = re.sub(
        r"(?i)\b(kv|hz|min|ms|ka|a|v|mpa|pa|s|%)\s+\1\b",
        r"\1",
        text,
    )
    return text


def _resolve_override_param_name(param_name: str, note_text: str) -> str:
    corrected_name = _extract_corrected_param_name(note_text)
    if corrected_name:
        return corrected_name
    return (param_name or "").strip()


def _resolve_override_value_text(value_text: str, note_text: str, value_source: str) -> str:
    note_text = (note_text or "").strip()
    if note_text:
        matched_general = re.search(
            r"(?:一般(?:我们)?用|通常用|常用)\s*([^，,。；;\n]+)", note_text
        )
        if matched_general and matched_general.group(1).strip():
            return matched_general.group(1).strip()
        matched_uniform = re.search(r"均(?:为|默认)\s*([^，,。；;\n]+)", note_text)
        if matched_uniform and matched_uniform.group(1).strip():
            return matched_uniform.group(1).strip()
        if "唯一" in note_text and "单相" in note_text:
            return "单相"
        if note_text.startswith("同") and len(note_text) <= 32:
            return note_text

    if value_source == "user_input":
        if any(
            token in note_text
            for token in ("三相", "单相", "合成试验", "试验方式", "共箱", "分箱", "其余")
        ):
            return note_text
        user_text = _extract_user_input_text(note_text)
        if user_text:
            return user_text
    corrected_text = _extract_corrected_value_text(note_text)
    if corrected_text:
        return corrected_text
    if (
        note_text
        and len(note_text) <= 32
        and "\n" not in note_text
        and not note_text.startswith("缺")
        and not _note_is_remove(note_text)
    ):
        return note_text
    if note_text:
        if any(
            token in note_text
            for token in (
                "用户录入",
                "客户录入",
                "默认",
                "根据用户",
                "应当",
                "应为",
                "改为",
                "改成",
                "C1",
                "C2",
                "sqrt",
                "根3",
                "%",
            )
        ):
            return note_text
    return (value_text or "").strip()


# ID Generation Functions

def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _truncate_entity_identifier(identifier: str, limit: int, chunk_key: str, identifier_role: str) -> str:
    """Truncate entity identifiers that exceed the configured length limit."""
    if len(identifier) <= limit:
        return identifier

    display_value = identifier[:limit]
    preview = identifier[:20]
    logger.warning(
        "%s: %s len %d > %d chars (Name: '%s...')",
        chunk_key,
        identifier_role,
        len(identifier),
        limit,
        preview,
    )
    return display_value


def _stable_equipment_id(name: str) -> str:
    return f"equip:{name}"


def _stable_report_id(name: str) -> str:
    return f"report:{name}"


def _stable_test_id(name: str, scope_key: str = "") -> str:
    if scope_key:
        return f"test:{scope_key}:{name}"
    return f"test:{name}"


def _stable_param_id(test_item: str, param_key: str, scope_key: str = "") -> str:
    if scope_key:
        return f"param:{scope_key}:{test_item}:{param_key}"
    return f"param:{test_item}:{param_key}"


def _stable_rule_id(test_item: str, rule_key: str, scope_key: str = "") -> str:
    return compute_mdhash_id(f"{scope_key}:{test_item}:{rule_key}", prefix="rule-")


def _stable_clause_id(std_id: str, clause_id: str, chunk_id: str) -> str:
    suffix = clause_id if clause_id else chunk_id
    return f"clause:{std_id}:{suffix}"


# Merge and Evidence Functions

def _merge_evidence(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for item in existing + incoming:
        key = (item.get("std_id"), item.get("clause_id"), item.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _merge_graph_field(existing_value: str | None, new_values: list[str]) -> str:
    existing_list = [v for v in (existing_value or "").split(GRAPH_FIELD_SEP) if v]
    for value in new_values:
        if value and value not in existing_list:
            existing_list.append(value)
    return GRAPH_FIELD_SEP.join(existing_list)


def _value_source_rank(value_source: str) -> int:
    ranking = {
        "user_input": 4,
        "formula": 3,
        "standard": 2,
        "default": 1,
    }
    return ranking.get(str(value_source or "").strip().lower(), 0)


def _is_reference_only_value_text(text: str) -> bool:
    merged = str(text or "").strip().lower()
    if not merged:
        return False
    ref_markers = ("规定", "依据", "按照", "见")
    has_ref_marker = any(marker in merged for marker in ref_markers)
    has_standard_ref = bool(
        re.search(r"(gb/?t|iec|iectr|t\d{4,}|第\d+(?:\.\d+)*[章节条款])", merged)
    )
    if not (has_ref_marker and has_standard_ref):
        return False
    has_numeric_value = bool(
        re.search(r"\d+(?:\.\d+)?\s*(kv|ka|hz|ms|min|%|a|v|s|次|相|m)\b", merged)
    )
    has_actionable_text = any(
        token in merged
        for token in (
            "用户录入",
            "默认",
            "三相",
            "单相",
            "相间",
            "对地",
            "断口",
            "主回路",
            "正极性",
            "负极性",
            "c1",
            "c2",
            "o-",
            "co",
        )
    )
    return not has_numeric_value and not has_actionable_text


def _parameter_quality_score(node_data: dict) -> tuple[int, int, int, int, int]:
    value_text = str(node_data.get("value_text", "") or "").strip()
    value_expr = str(node_data.get("value_expr", "") or "").strip()
    unit = str(node_data.get("unit", "") or "").strip()
    source_rank = _value_source_rank(str(node_data.get("value_source", "") or ""))
    merged_text = f"{value_text} {value_expr}".strip()
    has_measurable_detail = int(
        bool(re.search(r"\d", merged_text))
        or any(token in merged_text for token in ("%", "×", "*", "√", "/", "kV", "kA", "Hz", "ms", "min"))
    )
    reference_penalty = -1 if _is_reference_only_value_text(merged_text) else 0
    specificity = min(len(merged_text), 80) if has_measurable_detail else 0
    has_unit = int(bool(unit))
    return (source_rank, has_measurable_detail, reference_penalty, has_unit, specificity)


def _prefer_incoming_parameter(existing: dict, incoming: dict) -> bool:
    existing_score = _parameter_quality_score(existing)
    incoming_score = _parameter_quality_score(incoming)
    if incoming_score > existing_score:
        return True
    if incoming_score < existing_score:
        return False

    existing_value = str(existing.get("value_text", "") or "").strip()
    incoming_value = str(incoming.get("value_text", "") or "").strip()
    if not existing_value and incoming_value:
        return True
    if existing_value and not incoming_value:
        return False
    return False


def _merge_node_data_with_human_override(existing: dict | None, incoming: dict) -> dict:
    if not existing:
        return incoming
    merged = dict(existing)
    human_override = bool(existing.get("human_override"))

    merged["source_id"] = _merge_graph_field(
        existing.get("source_id"), [incoming.get("source_id", "")]
    )
    merged["file_path"] = _merge_graph_field(
        existing.get("file_path"), [incoming.get("file_path", "")]
    )
    merged_evidence = _merge_evidence(
        json.loads(existing.get("evidence", "[]") or "[]"),
        json.loads(incoming.get("evidence", "[]") or "[]"),
    )
    merged["evidence"] = _json_dumps_compact(merged_evidence)

    if human_override:
        for key, value in incoming.items():
            if key in {"source_id", "file_path", "evidence", "last_updated_at", "version"}:
                continue
            if not merged.get(key) and value not in (None, "", []):
                merged[key] = value
        return merged

    if (
        str(existing.get("entity_type", "") or "") == "TestParameter"
        and str(incoming.get("entity_type", "") or "") == "TestParameter"
    ):
        guarded_fields = {
            "param_name",
            "param_key",
            "value_text",
            "value_expr",
            "unit",
            "value_type",
            "value_source",
            "constraints",
            "calc_rule",
            "table_ref",
            "derive_from_rated",
        }
        prefer_incoming = _prefer_incoming_parameter(existing, incoming)
        for key, value in incoming.items():
            if key in {"source_id", "file_path", "evidence"}:
                continue
            if key in guarded_fields and not prefer_incoming and merged.get(key) not in (None, "", []):
                continue
            merged[key] = value
        return merged

    for key, value in incoming.items():
        if key in {"source_id", "file_path", "evidence"}:
            continue
        merged[key] = value
    return merged


def _merge_edge_data_with_human_override(existing: dict | None, incoming: dict) -> dict:
    if not existing:
        return incoming
    merged = dict(existing)
    human_override = bool(existing.get("human_override"))

    merged["source_id"] = _merge_graph_field(
        existing.get("source_id"), [incoming.get("source_id", "")]
    )
    merged["file_path"] = _merge_graph_field(
        existing.get("file_path"), [incoming.get("file_path", "")]
    )
    merged_evidence = _merge_evidence(
        json.loads(existing.get("evidence", "[]") or "[]"),
        json.loads(incoming.get("evidence", "[]") or "[]"),
    )
    merged["evidence"] = _json_dumps_compact(merged_evidence)

    if human_override:
        for key, value in incoming.items():
            if key in {"source_id", "file_path", "evidence", "last_updated_at", "version"}:
                continue
            if not merged.get(key) and value not in (None, "", []):
                merged[key] = value
        return merged

    for key, value in incoming.items():
        if key in {"source_id", "file_path", "evidence"}:
            continue
        merged[key] = value
    return merged


def _parse_controlled_json_response(raw_response: str) -> dict:
    """Parse controlled extraction JSON with tolerant cleanup/repair.
    
    Args:
        raw_response: Raw LLM response string
        
    Returns:
        Parsed dict from JSON response
    """
    import json_repair
    
    cleaned = remove_think_tags(raw_response or "").strip()
    
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*$", "", cleaned)
        cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    candidate = cleaned[start : end + 1] if start != -1 and end > start else cleaned

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = json_repair.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    preview = candidate[:180].replace("\n", " ")
    raise ValueError(f"Invalid controlled JSON response: {preview}")
