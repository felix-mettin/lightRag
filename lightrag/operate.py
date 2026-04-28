from __future__ import annotations
from copy import deepcopy
from contextvars import ContextVar
from functools import partial
from pathlib import Path
import os
import math
import time

import asyncio
import html
import json
import json_repair
import re
import uuid
from typing import Any, AsyncIterator, overload, Literal
from collections import Counter, defaultdict
from dotenv import load_dotenv

from lightrag.exceptions import (
    PipelineCancelledException,
    ChunkTokenLimitExceededError,
)
from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
    QueryResult,
    QueryContextResult
)
from lightrag.prompt import PROMPTS
from lightrag.utils import (
    CacheData,
    logger,
    get_file_only_logger,
    compute_mdhash_id,
    compute_args_hash,
    Tokenizer,
    is_float_regex,
    apply_source_ids_limit,
    convert_to_user_format,
    create_prefixed_exception,
    fix_tuple_delimiter_corruption,
    generate_reference_list_from_chunks,
    handle_cache,
    make_relation_chunk_key,
    merge_source_ids,
    pack_user_ass_to_openai_messages,
    pick_by_vector_similarity,
    pick_by_weighted_polling,
    process_chunks_unified,
    remove_think_tags,
    safe_vdb_operation_with_exception,
    save_to_cache,
    sanitize_and_normalize_extracted_text,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    update_chunk_cache_list,
    use_llm_func_with_cache,
)
from lightrag.constants import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_RELATED_CHUNK_NUMBER,
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
    GRAPH_FIELD_SEP,
)
from lightrag.standards import normalize_standard_type
from lightrag.kg.shared_storage import get_storage_keyed_lock

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

_TREE_OVERRIDE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_DOMAIN_RULE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_electrical_debug_logger = get_file_only_logger(
    "lightrag.electrical_debug",
    "lightrag_electrical_debug.log",
    env_var="LIGHTRAG_ELECTRICAL_DEBUG_LOG",
)
_CURRENT_ELECTRICAL_TRACE: ContextVar[dict[str, Any] | None] = ContextVar(
    "lightrag_electrical_trace",
    default=None,
)
_TRACE_STAGE_DESCRIPTIONS = {
    "trace_start": "一次查询调试日志开始",
    "kg_query_request": "查询入口和基础参数",
    "kg_query_keywords": "关键词抽取结果",
    "raw_search_result": "向量检索后的原始实体关系结果",
    "token_truncation": "实体关系 token 截断结果",
    "build_context_inputs": "上下文构建入口参数",
    "project_param_candidates": "试验项候选识别",
    "project_param_graph_lookup": "图中试验项参数和特征值提取",
    "post_rule_application": "规则覆盖后的项目参数结果",
    "report_scope_filter": "报告范围过滤后的项目参数结果",
    "final_test_item_scope": "最终试验项保留和移除结果",
    "pre_model_input": "首轮送模前完整输入",
    "pre_model_input_second_retrieval": "二轮检索后送模前完整输入",
    "naive_pre_model_input": "naive 模式送模前完整输入",
    "pre_llm": "送模前结构化原始数据快照",
    "naive_pre_llm": "naive 模式送模前结构化原始数据快照",
}
_TRACE_LIST_PREVIEW_LIMIT = 8
_TRACE_DICT_PREVIEW_LIMIT = 8
_TRACE_TEXT_PREVIEW_LIMIT = 1200
_ELECTRICAL_DEBUG_VERBOSE = os.getenv(
    "LIGHTRAG_ELECTRICAL_DEBUG_VERBOSE", "0"
).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_operate_standard_type(stand_type: str | None) -> str:
    normalized = normalize_standard_type(stand_type)
    if normalized:
        return normalized

    raw_value = str(stand_type or "").strip().upper()
    return raw_value or "others"


def _normalize_text_key(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").strip())


def _is_power_frequency_test_name(test_name: str | None) -> bool:
    name = str(test_name or "").strip()
    return name.startswith("工频耐受电压试验") or name == "作为状态检查的工频耐受电压试验"


def _uses_normal_count_label(test_name: str | None) -> bool:
    name = str(test_name or "").strip()
    return (
        _is_power_frequency_test_name(name)
        or name.startswith("雷电冲击耐受电压试验")
        or name.startswith("操作冲击耐受电压试验")
        or name == "作为状态检查的雷电冲击耐受电压试验"
    )


def _normalize_count_param_names(
    test_name: str | None,
    param_names: list[str] | None,
) -> list[str]:
    preferred_count_name = "正常次数" if _uses_normal_count_label(test_name) else "试验次数"
    fallback_count_name = "试验次数" if preferred_count_name == "正常次数" else "正常次数"
    normalized_params: list[str] = []
    count_inserted = False

    for raw_param_name in param_names or []:
        param_name = str(raw_param_name or "").strip()
        if not param_name:
            continue
        if param_name in {"试验次数", "正常次数"}:
            if count_inserted:
                continue
            normalized_params.append(preferred_count_name)
            count_inserted = True
            continue
        normalized_params.append(param_name)

    if not count_inserted and fallback_count_name in normalized_params:
        normalized_params.append(preferred_count_name)

    return _dedupe_preserve_order(normalized_params)


def _dedupe_preserve_order(values: list[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_value in values or []:
        value = str(raw_value or "").strip()
        if not value:
            continue
        normalized = _normalize_text_key(value).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result


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


def _split_name_and_value(raw_name: str) -> tuple[str, str]:
    text = (raw_name or "").strip()
    for sep in ("：", "﹕", "∶", ":"):
        if sep in text:
            left, right = text.split(sep, 1)
            return left.strip(), right.strip()
    return text, ""


def _infer_value_source(value_text: str, note_text: str) -> str:
    merged = f"{value_text} {note_text}".strip()
    # 先检查是否是公式（包含计算符号）
    if any(token in merged for token in ("计算", "公式", "%", "×", "*", "/", "÷", "+", "-", "=")):
        return "formula"
    # 再检查是否需要用户输入
    if any(
            token in merged
            for token in ("用户录入", "用户输入", "用户提供", "客户录入", "客户输入", "客户提供")
    ):
        return "user_input"
    if "默认" in merged:
        return "default"
    return "standard"


def _is_concrete_final_value_text(value_text: str) -> bool:
    text = _sanitize_value_text(value_text)
    if not text:
        return False
    unresolved_markers = (
        "用户录入",
        "用户输入",
        "用户提供",
        "客户录入",
        "客户输入",
        "客户提供",
        "默认为",
        "默认",
        "优选值",
        "取决于",
        "根据",
        "按表",
        "查表",
        "见表",
        "见图",
        "若",
        "如果",
        "则",
        "或",
        "和/或",
        "范围",
        "以上",
        "以下",
        "无法确定",
        "--",
    )
    if any(marker in text for marker in unresolved_markers):
        return False
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ ]*", text):
        return False
    return True


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

    if value_source in {"user_input", "default", "standard", "rule"} and _is_concrete_final_value_text(
        value_text
    ):
        return "graph_final"

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
        corrected = text[idx + len(marker):].strip()
        corrected = re.sub(r"^[：:，,。.、\-\s]+", "", corrected)
        if corrected:
            return corrected
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

        # Support concise reviewer notes like:
        # "缺试验次数：24次" / "缺试验相数：..." / "缺试验电流A: ..."
        short_missing = re.match(r"^缺(?:少)?([^：:，,]+)[：:，,]\s*(.+)$", segment)
        if not short_missing:
            short_missing = re.match(r"^缺(?:少)?([^为是]+?)(?:为|是)\s*(.+)$", segment)
        if short_missing:
            param_name = (short_missing.group(1) or "").strip()
            value_hint = (short_missing.group(2) or "").strip()
            if not param_name:
                continue
            # Category-level "缺XX试验" belongs to add_test_items, not feature params.
            if "特征值" in param_name:
                continue
            if "试验" in param_name and "次数" not in param_name and "相数" not in param_name:
                if not any(token in param_name for token in
                           ("电压", "电流", "频率", "状态", "部位", "介质", "极性", "时间", "类别", "顺序", "项数")):
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
                payload = stripped_segment[len(prefix):].strip("：:，, ")
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
        tail = text[matched.end():]
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
    # Support "参数A、参数B、参数C" list-style declarations with no explicit values.
    if all(token not in text for token in ("（", "）", "(", ")", "，", ",", ":")):
        return [(seg.strip(), "") for seg in re.split(r"[、；;\n]+", text) if seg.strip()]
    if all(token not in text for token in ("（", "）", "(", ")", ":")) and "、" in text:
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


def _resolve_override_value_text(value_text: str, note_text: str, value_source: str) -> str:
    note_text = (note_text or "").strip()
    if note_text:
        # Common reviewer phrasing: "一般我们用1min".
        matched_general = re.search(
            r"(?:一般(?:我们)?用|通常用|常用)\s*([^，,。；;\n]+)", note_text
        )
        if matched_general and matched_general.group(1).strip():
            return matched_general.group(1).strip()
        # "均为..." / "均默认..." is a direct replacement hint.
        matched_uniform = re.search(r"均(?:为|默认)\s*([^，,。；;\n]+)", note_text)
        if matched_uniform and matched_uniform.group(1).strip():
            return matched_uniform.group(1).strip()
        # "唯一" usually means a fixed single option in this domain.
        if "唯一" in note_text and "单相" in note_text:
            return "单相"
        # Keep explicit cross-reference notes instead of stale extracted value.
        if note_text.startswith("同") and len(note_text) <= 32:
            return note_text

    if value_source == "user_input":
        # Keep full scenario-driven notes for complex branching rules such as
        # T100s/T100a short-circuit tests where the note contains both the
        # user-input variable and the surrounding applicability logic.
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
    # Reviewer often places the final corrected value directly in the note,
    # e.g. "O", "CO", "2 kV". Treat short standalone notes as authoritative.
    if (
            note_text
            and len(note_text) <= 32
            and "\n" not in note_text
            and not note_text.startswith("缺")
            and not _note_is_remove(note_text)
    ):
        return note_text
    if note_text:
        # In reviewer-driven annotation mode, note text often is the corrected final value.
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


def _resolve_override_param_name(param_name: str, note_text: str) -> str:
    corrected_name = _extract_corrected_param_name(note_text)
    if corrected_name:
        return corrected_name
    return (param_name or "").strip()


def _sanitize_value_text(value_text: str) -> str:
    text = (value_text or "").strip()
    if not text:
        return ""
    # Remove colloquial reviewer phrases from value candidates.
    text = re.sub(
        r"(?:我们一般用|一般我们用|我们通常用|通常我们用|我们用|一般用)\s*",
        "",
        text,
    ).strip()

    def _is_symbolic_placeholder(seg: str) -> bool:
        s = (seg or "").strip()
        if not s:
            return False
        # Unresolved variable-like placeholders.
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
        if re.fullmatch(r"(?:u|ur|u_r|rated_voltage|i_sc|rated_short_circuit_current)(?:\s*(?:kv|ka|v|a))?", s,
                        flags=re.IGNORECASE):
            score -= 24
        if len(s) > 120:
            score -= 8
        # Prefer concise and actionable fragments in ties.
        return (score, -len(s), int(_contains_chinese(s)), int(bool(re.search(r"\d", s))))

    # segments = [seg.strip() for seg in re.split(r"\s*/\s*", text) if seg.strip()]
    segments = [text.strip()]
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
        cleaned = re.sub(r"^[：:。,.、\-\s]+", "", cleaned)
        if cleaned:
            cleaned_segments.append(cleaned)
            if marker_hit:
                corrected_segments.append(cleaned)
    if cleaned_segments:
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
        # Pick highest-quality segment to avoid keeping bilingual translation noise.
        best_segment = max(segments, key=_segment_quality_score)
        if _is_symbolic_placeholder(best_segment) and len(segments) >= 2:
            text = segments[0]
        else:
            text = best_segment

    text = re.sub(r"\s+", " ", text).strip()
    text = _normalize_symbolic_value(text)
    # Trim mixed-language tail like: "... , rated_voltage = ...".
    text = re.sub(r"\s*[,，]\s*(?:rated_|ur\b|u_r\b|i_sc\b).*$", "", text, flags=re.IGNORECASE)
    # Normalize repeated unit symbols.
    text = re.sub(r"%\s*%", "%", text)
    text = re.sub(
        r"(?i)\b(kv|hz|min|ms|ka|a|v|mpa|pa|s|%)\s+\1\b",
        r"\1",
        text,
    )
    return text


def _find_override_rule_by_test_name(
        tree_tests_by_path: dict[str, Any],
        report_type: str,
        category: str,
        test_name: str,
) -> dict[str, Any] | None:
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


def _expand_shorthand_param_value(
        value_text: str,
        param_name: str,
        override_rule: dict[str, Any],
        tree_tests_by_path: dict[str, Any],
) -> str:
    text = (value_text or "").strip()
    if not text:
        return text
    if "同" not in text:
        return text

    report_type = str(override_rule.get("report_type", "") or "").strip()
    category = str(override_rule.get("category", "") or "").strip()
    target_rule: dict[str, Any] | None = None

    if "同工频" in text:
        target_rule = _find_override_rule_by_test_name(
            tree_tests_by_path, report_type, category, "工频耐受电压试验"
        )
    elif any(token in text for token in ("同雷电", "同雷冲")):
        target_rule = _find_override_rule_by_test_name(
            tree_tests_by_path, report_type, category, "雷电冲击耐受电压试验"
        )
    elif "同操作冲击" in text:
        target_rule = _find_override_rule_by_test_name(
            tree_tests_by_path, report_type, category, "操作冲击耐受电压试验"
        )
    else:
        # e.g. 同CC2 / 同T10
        matched = re.search(r"同([A-Za-z0-9()]+)", text)
        if matched:
            token = matched.group(1).upper()
            fallback_candidates: list[dict[str, Any]] = []
            for rule in tree_tests_by_path.values():
                if not isinstance(rule, dict):
                    continue
                candidate = str(rule.get("test_name", "") or "")
                if token and token in candidate.upper():
                    fallback_candidates.append(rule)
            if fallback_candidates:
                for rule in fallback_candidates:
                    if _normalize_text_key(str(rule.get("report_type", "") or "")) != _normalize_text_key(
                            report_type
                    ):
                        continue
                    if _normalize_text_key(str(rule.get("category", "") or "")) != _normalize_text_key(
                            category
                    ):
                        continue
                    target_rule = rule
                    break
                if target_rule is None:
                    target_rule = fallback_candidates[0]

    if not target_rule:
        return text

    target_params = target_rule.get("parameters", [])
    if not isinstance(target_params, list):
        return text
    normalized_param = _normalize_text_key(param_name)
    for target_param in target_params:
        if not isinstance(target_param, dict):
            continue
        target_param_name = str(target_param.get("param_name", "") or "").strip()
        if _normalize_text_key(target_param_name) != normalized_param:
            continue
        resolved = str(target_param.get("value_text", "") or "").strip()
        if resolved:
            return resolved
    return text


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
    mtime = override_path.stat().st_mtime
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


def _merge_annotation_rules(
        base_rules: dict[str, Any], patch_rules: dict[str, Any]
) -> dict[str, Any]:
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


def _resolve_domain_rules_path(schema_cfg: dict | None = None) -> Path | None:
    schema_cfg = schema_cfg or {}
    path_text = str(schema_cfg.get("electrical_rules_path", "") or "").strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _load_domain_rules(schema_cfg: dict | None = None) -> dict[str, Any]:
    schema_cfg = schema_cfg or {}
    path = _resolve_domain_rules_path(schema_cfg)
    if path is None:
        logger.warning("Domain rules path is empty in electrical_schema config")
        return {}
    if not path.exists():
        logger.warning("Domain rules file not found: %s", path)
        return {}

    logger.info("Loading domain rules from %s", path)

    cache_key = str(path.resolve())
    mtime = path.stat().st_mtime
    cached = _DOMAIN_RULE_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        logger.info(
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


def _evaluate_domain_rule_decisions(
    query: str,
    schema_cfg: dict | None = None,
    stand_type: str | None = None
) -> dict[str, Any]:
    schema_cfg = schema_cfg or {}
    stand_type = _normalize_operate_standard_type(stand_type)
    domain_rules = _load_domain_rules(schema_cfg)
    rules = domain_rules.get("rules", []) or []
    if not isinstance(rules, list) or not rules:
        return {}

    def _build_optional_unit_pattern(unit: str) -> str:
        normalized_unit = str(unit or "").strip().lower()
        if normalized_unit == "kv":
            return r"(?:\s*(?:kV|KV|kv))?"
        if normalized_unit == "ka":
            return r"(?:\s*(?:kA|KA|ka))?"
        if normalized_unit == "a":
            return r"(?:\s*A)?"
        if normalized_unit == "hz":
            return r"(?:\s*(?:Hz|HZ|hz))?"
        if normalized_unit in {"p.u.", "pu", "p.u", "p u"}:
            return r"(?:\s*(?:p\.?\s*u\.?))?"
        return r"(?:\s*(?:kV|KV|kv|kA|KA|ka|A|Hz|HZ|hz|p\.?\s*u\.?))?"

    def _extract_named_numeric(
        query_text: str,
        labels: list[str],
        unit: str = "",
    ) -> float | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        normalized = text.replace("（", "(").replace("）", ")")
        unit_pattern = _build_optional_unit_pattern(unit)
        for label in labels:
            if not str(label or "").strip():
                continue
            pattern = (
                rf"{re.escape(str(label).strip())}\s*(?:[:：=]\s*)?"
                rf"([0-9]+(?:\.[0-9]+)?(?:\s*\+\s*[0-9]+(?:\.[0-9]+)?)*){unit_pattern}"
            )
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                parts = re.findall(r"[0-9]+(?:\.[0-9]+)?", match.group(1))
                if parts:
                    return sum(float(part) for part in parts)
        return None

    def _extract_named_voltage_kv(query_text: str, labels: list[str]) -> float | None:
        return _extract_named_numeric(query_text, labels, unit="kV")

    def _extract_named_current_ka(query_text: str, labels: list[str]) -> float | None:
        return _extract_named_numeric(query_text, labels, unit="kA")

    def _extract_named_current_a(query_text: str, labels: list[str]) -> float | None:
        return _extract_named_numeric(query_text, labels, unit="A")

    def _preferred_capacitive_current_a(kind: str, rated_voltage: float | None) -> float | None:
        if rated_voltage is None:
            return None
        i1_table = {
            3.6: 10.0,
            7.2: 10.0,
            12.0: 10.0,
            24.0: 10.0,
            40.5: 10.0,
            72.5: 10.0,
            126.0: 31.5,
            252.0: 125.0,
            363.0: 315.0,
            550.0: 500.0,
            800.0: 900.0,
            1100.0: 1200.0,
        }
        ic_table = {
            3.6: 10.0,
            7.2: 10.0,
            12.0: 25.0,
            24.0: 31.5,
            40.5: 50.0,
            72.5: 125.0,
            126.0: 140.0,
            252.0: 250.0,
            363.0: 355.0,
            550.0: 500.0,
        }
        table = i1_table if kind == "I1" else ic_table if kind == "Ic" else {}
        for key, value in table.items():
            if abs(float(rated_voltage) - key) < 1e-6:
                return value
        return None

    def _format_current_a(value: float) -> str:
        if float(value).is_integer():
            return f"{int(value)}A"
        return f"{value:.2f}".rstrip("0").rstrip(".") + "A"

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

    def _extract_condition_numeric(query_text: str, label: str, unit: str) -> float | None:
        if not label:
            return None
        actual = _extract_named_numeric(query_text, [label], unit=unit)
        if actual is not None:
            return actual
        alias_labels: list[str] = []
        if label.endswith("kpp"):
            alias_labels.append(label.removesuffix("kpp").strip())
        elif label == "首开极系数":
            alias_labels.append("首开极系数kpp")
        if alias_labels:
            return _extract_named_numeric(query_text, alias_labels, unit=unit)
        return None

    def _extract_condition_numeric_from_candidates(
        query_text: str, labels: list[str], unit: str
    ) -> float | None:
        for candidate in labels:
            normalized_candidate = str(candidate or "").strip()
            if not normalized_candidate:
                continue
            actual = _extract_condition_numeric(query_text, normalized_candidate, unit)
            if actual is not None:
                return actual
        return None

    def _matches_condition(condition: dict[str, Any], query_text: str) -> bool:
        cond_type = str(condition.get("type", "") or "").strip()
        label = str(condition.get("label", "") or "").strip()
        configured_reports = {
            str(item).strip()
            for item in (schema_cfg.get("report_types", []) or [])
            if str(item).strip()
        }
        report_alias_map = schema_cfg.get("report_aliases", {}) or {}
        query_report_scopes = set(_extract_current_report_scopes(query_text, schema_cfg))

        def _is_report_scope_label(text: str) -> bool:
            normalized_text = str(text or "").strip()
            if not normalized_text:
                return False
            if normalized_text in configured_reports:
                return True
            if isinstance(report_alias_map, dict):
                canonical_text = str(report_alias_map.get(normalized_text, "") or "").strip()
                return canonical_text in configured_reports if canonical_text else False
            return False

        def _matches_text_label(text: str) -> bool:
            normalized_text = str(text or "").strip()
            if not normalized_text:
                return False
            if _is_report_scope_label(normalized_text):
                if normalized_text in query_report_scopes:
                    return True
                canonical_text = str(report_alias_map.get(normalized_text, "") or "").strip()
                return canonical_text in query_report_scopes if canonical_text else False
            return _query_contains(query_text, normalized_text)

        if cond_type == "contains":
            return _matches_text_label(label)
        if cond_type == "not_contains":
            return bool(label) and label not in query_text
        if cond_type == "contains_any":
            labels = condition.get("labels", []) or []
            return isinstance(labels, list) and any(
                _matches_text_label(str(item or "").strip())
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
            actual = _extract_condition_numeric(
                query_text,
                label,
                str(condition.get("unit", "") or "").strip(),
            )
            return actual is not None and float(actual) == float(value)
        if cond_type == "not_equals_numeric":
            value = condition.get("value")
            actual = _extract_condition_numeric(
                query_text,
                label,
                str(condition.get("unit", "") or "").strip(),
            )
            return actual is not None and float(actual) != float(value)
        if cond_type == "greater_numeric":
            value = condition.get("value")
            actual = _extract_condition_numeric(
                query_text,
                label,
                str(condition.get("unit", "") or "").strip(),
            )
            return actual is not None and float(actual) > float(value)
        if cond_type == "less_or_equal_numeric":
            value = condition.get("value")
            actual = _extract_condition_numeric(
                query_text,
                label,
                str(condition.get("unit", "") or "").strip(),
            )
            return actual is not None and float(actual) <= float(value)
        if cond_type == "greater_or_equal_numeric":
            value = condition.get("value")
            actual = _extract_condition_numeric(
                query_text,
                label,
                str(condition.get("unit", "") or "").strip(),
            )
            return actual is not None and float(actual) >= float(value)
        if cond_type == "less_numeric":
            value = condition.get("value")
            actual = _extract_condition_numeric(
                query_text,
                label,
                str(condition.get("unit", "") or "").strip(),
            )
            return actual is not None and float(actual) < float(value)
        if cond_type == "ratio_greater_or_equal":
            numerator_labels = [
                str(item or "").strip()
                for item in (condition.get("numerator_labels", []) or [])
                if str(item or "").strip()
            ]
            denominator_labels = [
                str(item or "").strip()
                for item in (condition.get("denominator_labels", []) or [])
                if str(item or "").strip()
            ]
            numerator_label = str(condition.get("numerator_label", "") or "").strip()
            denominator_label = str(condition.get("denominator_label", "") or "").strip()
            value = condition.get("value")
            unit = str(condition.get("unit", "") or "").strip()
            if numerator_label:
                numerator_labels.append(numerator_label)
            if denominator_label:
                denominator_labels.append(denominator_label)
            if not numerator_labels or not denominator_labels or value is None:
                return False
            num_val = _extract_condition_numeric_from_candidates(
                query_text, numerator_labels, unit
            )
            den_val = _extract_condition_numeric_from_candidates(
                query_text, denominator_labels, unit
            )
            if num_val is None or den_val is None or float(den_val) == 0:
                return False
            return float(num_val) / float(den_val) >= float(value)
        if cond_type == "ratio_less":
            numerator_labels = [
                str(item or "").strip()
                for item in (condition.get("numerator_labels", []) or [])
                if str(item or "").strip()
            ]
            denominator_labels = [
                str(item or "").strip()
                for item in (condition.get("denominator_labels", []) or [])
                if str(item or "").strip()
            ]
            numerator_label = str(condition.get("numerator_label", "") or "").strip()
            denominator_label = str(condition.get("denominator_label", "") or "").strip()
            value = condition.get("value")
            unit = str(condition.get("unit", "") or "").strip()
            if numerator_label:
                numerator_labels.append(numerator_label)
            if denominator_label:
                denominator_labels.append(denominator_label)
            if not numerator_labels or not denominator_labels or value is None:
                return False
            num_val = _extract_condition_numeric_from_candidates(
                query_text, numerator_labels, unit
            )
            den_val = _extract_condition_numeric_from_candidates(
                query_text, denominator_labels, unit
            )
            if num_val is None or den_val is None or float(den_val) == 0:
                return False
            return float(num_val) / float(den_val) < float(value)
        if cond_type == "regex_extract_not_equals":
            pattern = str(condition.get("pattern", "") or "").strip()
            disallowed = str(condition.get("value", "") or "").strip().upper()
            if not pattern:
                return False
            match = re.search(pattern, query_text)
            return bool(match and match.group(1).strip().upper() != disallowed)
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
    capacitive_grade_match = re.search(
        r"(?:容性电流开合时重击穿等级|开合容性电流能力的级别|重击穿等级|能力级别|试验级别)\s*(?:[:：=]|为|是)?\s*(C[12])",
        query,
        flags=re.IGNORECASE,
    )
    capacitive_grade = (
        capacitive_grade_match.group(1).upper() if capacitive_grade_match else None
    )
    explicit_solid_sealed_pole = _query_contains(query, "元件中含固封极柱")
    pf_base_for_split = _extract_named_voltage_kv(
        query,
        ["额定短时工频耐受电压", "额定工频耐受电压"],
    )
    pf_fracture_for_split = _extract_named_voltage_kv(
        query,
        ["额定短时工频耐受电压(断口)", "额定工频耐受电压(断口)"],
    )
    pf_fracture_split_active = bool(
        pf_base_for_split is not None
        and pf_fracture_for_split is not None
        and pf_fracture_for_split > pf_base_for_split
    )
    current_report_scopes = set(_extract_current_report_scopes(query, schema_cfg))

    def _rule_matches_current_scope(raw_rule: dict[str, Any]) -> bool:
        if not current_report_scopes:
            return True
        rule_domain = str(raw_rule.get("domain", "") or "").strip()
        if not rule_domain:
            return True
        domain_scope_map = {
            "insulation": "绝缘性能型式试验",
            "temperature_rise": "温升性能型式试验",
            "switching": "开合性能型式试验",
            "short_circuit": "短路性能型式试验",
        }
        expected_scope = domain_scope_map.get(rule_domain)
        if not expected_scope:
            return True
        return expected_scope in current_report_scopes

    decisions: dict[str, Any] = {}

    for raw_rule in rules:
        if not isinstance(raw_rule, dict):
            continue
        # If the rule explicitly targets a standard (gb/iec/dlt/etc.), skip it
        # when the current evaluation stand_type is provided and does not match.
        raw_rule_standard = str(raw_rule.get("standard", "") or "").strip()
        if raw_rule_standard:
            normalized_raw_standard = _normalize_operate_standard_type(raw_rule_standard)
            if normalized_raw_standard and normalized_raw_standard != stand_type:
                continue

        if not _rule_matches_current_scope(raw_rule):
            continue

        rule_id = str(raw_rule.get("rule_id", "") or "").strip()
        if not rule_id:
            continue
        rule_kind = str(raw_rule.get("kind", "") or "").strip()

        if rule_kind == "split":
            input_cfg = raw_rule.get("inputs", {}) or {}
            trigger_when_any = input_cfg.get("trigger_when_any", []) or []
            require_when_all = input_cfg.get("require_when_all", []) or []
            base_labels = input_cfg.get("base_voltage_labels", []) or []
            fracture_labels = input_cfg.get("fracture_voltage_labels", []) or []
            split_enabled = False
            base_kv = None
            fracture_kv = None
            fracture_provided = False
            remove_original = bool(raw_rule.get("remove_original", True))

            single_output = raw_rule.get("single_output", {}) or {}
            split_outputs = raw_rule.get("split_output", []) or []

            test_item = str(raw_rule.get("test_item", "") or "")
            if isinstance(trigger_when_any, list) and trigger_when_any:
                matched_conditions: list[str] = []
                for condition in trigger_when_any:
                    if not isinstance(condition, dict):
                        continue
                    if _matches_condition(condition, query):
                        matched_conditions.append(
                            str(condition.get("label", "") or condition.get("type", "") or "")
                        )
                split_enabled = bool(matched_conditions)
                reason_code = "split_enabled" if split_enabled else "split_not_triggered"
                reason_text = (
                    f"命中条件：{'；'.join(matched_conditions)}，允许拆分。"
                    if split_enabled
                    else "未命中任何拆分触发条件，保持未拆分。"
                )
                if split_enabled and isinstance(require_when_all, list) and require_when_all:
                    unmet_conditions = [
                        condition
                        for condition in require_when_all
                        if not isinstance(condition, dict) or not _matches_condition(condition, query)
                    ]
                    if unmet_conditions:
                        split_enabled = False
                        reason_code = "split_constraints_not_met"
                        reason_text = "已命中拆分触发词，但未满足全部附加约束条件，保持未拆分。"
                if rule_id in {"switching.gb.cc2_final_split", "switching.gb.lc2_final_split"}:
                    if capacitive_grade == "C1":
                        split_enabled = False
                        reason_code = "capacitive_grade_c1_single"
                        reason_text = "开合容性电流能力的级别为C1，CC2/LC2 不拆分，按单项 CO、24次执行。"
                        single_output = {
                            "test_item": test_item,
                            "parameter_overrides": {
                                "操作顺序": "CO",
                                "试验次数": "24次",
                            },
                        }
                    else:
                        reason_text = (
                            f"{reason_text} 开合容性电流能力的级别"
                            f"{'未提供，默认按C2处理' if capacitive_grade is None else '为C2'}，保留拆分。"
                        )
                normalized_stand_type = _normalize_operate_standard_type(stand_type)
                expected_rule_id = rule_id
                if normalized_stand_type == "DLT":
                    expected_rule_id = "insulation.gb.power_frequency_outdoor_state_split"
                elif normalized_stand_type == "IEC":
                    expected_rule_id = "insulation.gb.power_frequency_outdoor_state_split"
                else:
                    expected_rule_id = "insulation.gb.power_frequency_outdoor_state_split"
                if (
                    rule_id == expected_rule_id
                    and split_enabled
                    and pf_fracture_split_active
                    and rated_voltage_kv is not None
                    and rated_voltage_kv <= 40.5
                ):
                    split_enabled = False
                    reason_code = "delegated_to_power_frequency_split"
                    reason_text = "户外命名拆分被工频断口拆分接管，最终由工频断口拆分直接产出三条工频项目。"
                if rule_id == expected_rule_id and split_enabled:
                    if rated_voltage_kv is not None and rated_voltage_kv > 252:
                        split_outputs = [
                            {
                                "test_item": "工频耐受电压试验(相间及对地)",
                                "inherits_from": "工频耐受电压试验",
                                "parameter_overrides": {
                                    "试验部位": "相间及对地",
                                    "试验状态": "干",
                                    "正常次数": "9次",
                                },
                            }
                        ]
                        reason_text = "额定电压高于252kV时，高压户外工频不再设置湿态原项，仅保留相间及对地干态项，联合电压由高压联合电压规则承载。"
                    elif rated_voltage_kv is not None and rated_voltage_kv > 40.5:
                        split_outputs = [
                            {
                                "test_item": "工频耐受电压试验(相间及对地)",
                                "inherits_from": "工频耐受电压试验",
                                "parameter_overrides": {
                                    "试验部位": "相间及对地",
                                    "试验状态": "干",
                                    "正常次数": "9次",
                                },
                            },
                            {
                                "test_item": "工频耐受电压试验",
                                "inherits_from": "工频耐受电压试验",
                                "parameter_overrides": {
                                    "试验部位": "相间及对地",
                                    "试验状态": "湿",
                                    "正常次数": "9次",
                                },
                            },
                        ]
                        reason_text = "高压户外工频试验将相间及对地拆分为干态显式项和湿态原项，联合电压由高压联合电压规则承载。"
                joint_voltage_rule_id = rule_id
                if normalized_stand_type == "DLT":
                    joint_voltage_rule_id = "insulation.dlt.power_frequency_joint_voltage_split"
                elif normalized_stand_type == "IEC":
                    joint_voltage_rule_id = "insulation.gb.power_frequency_joint_voltage_split"
                else:
                    joint_voltage_rule_id = "insulation.gb.power_frequency_joint_voltage_split"
                if (
                    rule_id == joint_voltage_rule_id
                    and split_enabled
                    and rated_voltage_kv is not None
                    and rated_voltage_kv > 252
                    and ("户外产品" in query or "户外" in query)
                ):
                    split_outputs = [
                        {
                            "test_item": "工频耐受电压试验(联合电压)",
                            "inherits_from": "工频耐受电压试验",
                            "parameter_overrides": {
                                "试验部位": "断口",
                                "试验状态": "干"
                            },
                            "additional_params": ["交流电压(辅)"],
                        }
                    ]
                    reason_text = "额定电压高于252kV且命中户外条件时，高压工频联合电压仅保留干态联合电压项，不再复用原项。"
                switching_joint_voltage_rule_id = rule_id
                if normalized_stand_type == "DLT":
                    switching_joint_voltage_rule_id = "insulation.dlt.switching_impulse_joint_voltage_split"
                elif normalized_stand_type == "IEC":
                    switching_joint_voltage_rule_id = "insulation.iec.switching_impulse_joint_voltage_split"
                else:
                    switching_joint_voltage_rule_id = "insulation.gb.switching_impulse_joint_voltage_split"
                if (
                    rule_id == switching_joint_voltage_rule_id
                    and split_enabled
                    and rated_voltage_kv is not None
                    and rated_voltage_kv > 252
                    and ("户外产品" in query or "户外" in query)
                ):
                    remove_original = True
                    split_outputs = [
                        {
                            "test_item": "操作冲击耐受电压试验(干)",
                            "inherits_from": "操作冲击耐受电压试验",
                            "parameter_overrides": {
                                "试验部位": "相间及对地",
                                "试验状态": "干"
                            },
                        },
                        {
                            "test_item": "操作冲击耐受电压试验(湿)",
                            "inherits_from": "操作冲击耐受电压试验",
                            "parameter_overrides": {
                                "试验部位": "相间及对地",
                                "试验状态": "湿"
                            },
                        },
                        {
                            "test_item": "操作冲击耐受电压试验(联合电压)",
                            "inherits_from": "操作冲击耐受电压试验",
                            "parameter_overrides": {
                                "试验部位": "断口",
                                "试验状态": "干"
                            },
                        }
                    ]
                    reason_text = "额定电压高于252kV且命中户外条件时，操作冲击耐受电压试验拆分为相间及对地(干)、相间及对地(湿)和联合电压(干)，并移除原项。"
                switching_wet_rule_id = rule_id
                if normalized_stand_type == "DLT":
                    switching_wet_rule_id = "insulation.gb.switching_impulse_wet_split"
                elif normalized_stand_type == "IEC":
                    switching_wet_rule_id = "insulation.iec.switching_impulse_wet_split"
                else:
                    switching_wet_rule_id = "insulation.gb.switching_impulse_wet_split"
                if (
                    rule_id == switching_wet_rule_id
                    and split_enabled
                    and rated_voltage_kv is not None
                    and rated_voltage_kv > 252
                    and ("户外产品" in query or "户外" in query)
                ):
                    split_enabled = False
                    reason_code = "delegated_to_switching_joint_voltage_split"
                    reason_text = "户外高压操作冲击的干/湿/联合电压拆分由联合电压规则统一承载，湿态补充分裂规则不再重复生效。"
            else:
                if not isinstance(base_labels, list) or not isinstance(fracture_labels, list):
                    continue

                base_kv = _extract_named_voltage_kv(query, [str(v) for v in base_labels])
                fracture_kv = _extract_named_voltage_kv(
                    query, [str(v) for v in fracture_labels]
                )
                fracture_provided = fracture_kv is not None
                split_enabled = bool(
                    base_kv is not None and fracture_kv is not None and fracture_kv > base_kv
                )

                domain_label = "雷电" if "雷电" in test_item else "工频"
                reason_code = "split_enabled"
                reason_text = f"断口{domain_label}值严格大于本体值，允许拆分。"
                if not fracture_provided:
                    reason_code = "fracture_not_provided"
                    reason_text = f"未提供断口{domain_label}参数，必须保持未拆分。"
                elif base_kv is None:
                    reason_code = "base_not_provided"
                    reason_text = f"未提供本体{domain_label}参数，不能触发拆分。"
                elif fracture_kv <= base_kv:
                    if rated_voltage_kv == 40.5:
                        reason_code = "ur_40_5_prefer_single"
                        reason_text = (
                            f"Ur=40.5kV 且断口{domain_label}值未严格大于本体值，必须保持未拆分。"
                        )
                    else:
                        reason_code = "fracture_not_greater"
                        reason_text = f"断口{domain_label}值小于或等于本体值，必须保持未拆分。"
                if split_enabled and isinstance(require_when_all, list) and require_when_all:
                    unmet_conditions = [
                        condition
                        for condition in require_when_all
                        if not isinstance(condition, dict) or not _matches_condition(condition, query)
                    ]
                    if unmet_conditions:
                        split_enabled = False
                        reason_code = "split_constraints_not_met"
                        reason_text = "已满足断口值拆分条件，但未满足全部附加约束条件，保持未拆分。"
                normalized_stand_type = _normalize_operate_standard_type(stand_type)
                expected_rule_id = rule_id
                if normalized_stand_type == "DLT":
                    expected_rule_id = "insulation.gb.power_frequency_split"
                elif normalized_stand_type == "IEC":
                    expected_rule_id = "insulation.gb.power_frequency_split"
                else:
                    expected_rule_id = "insulation.gb.power_frequency_split"
                if (
                    rule_id == expected_rule_id
                    and split_enabled
                    and ("户外产品" in query or "户外" in query)
                ):
                    if rated_voltage_kv is not None and rated_voltage_kv > 252:
                        split_outputs = [
                            {
                                "test_item": "工频耐受电压试验(相间及对地)",
                                "inherits_from": "工频耐受电压试验",
                                "parameter_overrides": {
                                    "试验部位": "相间及对地",
                                    "试验状态": "干",
                                    "正常次数": "9次",
                                },
                            }
                        ]
                        reason_text = "断口工频值严格大于本体值且命中户外条件，但额定电压高于252kV时不再设置湿态原项，断口侧由联合电压规则承载，仅保留相间及对地干态项。"
                    elif rated_voltage_kv is not None and rated_voltage_kv > 40.5:
                        split_outputs = [
                            {
                                "test_item": "工频耐受电压试验(相间及对地)",
                                "inherits_from": "工频耐受电压试验",
                                "parameter_overrides": {
                                    "试验部位": "相间及对地",
                                    "试验状态": "干",
                                    "正常次数": "9次",
                                },
                            },
                            {
                                "test_item": "工频耐受电压试验",
                                "inherits_from": "工频耐受电压试验",
                                "parameter_overrides": {
                                    "试验部位": "相间及对地",
                                    "试验状态": "湿",
                                    "正常次数": "9次",
                                },
                            },
                        ]
                        reason_text = "断口工频值严格大于本体值且命中户外条件，在40.5kV以上且不高于252kV时，工频拆分为相间及对地(干)和原项承载的相间及对地(湿)，断口侧由联合电压规则承载。"
                    else:
                        split_outputs = [
                            {
                                "test_item": "工频耐受电压试验(相间及对地)",
                                "inherits_from": "工频耐受电压试验",
                                "parameter_overrides": {
                                    "试验部位": "相间及对地",
                                    "试验状态": "干",
                                    "正常次数": "9次",
                                },
                            },
                            {
                                "test_item": "工频耐受电压试验",
                                "inherits_from": "工频耐受电压试验",
                                "parameter_overrides": {
                                    "试验部位": "相间及对地",
                                    "试验状态": "湿",
                                    "正常次数": "9次",
                                },
                            },
                            {
                                "test_item": "工频耐受电压试验(断口)",
                                "inherits_from": "工频耐受电压试验",
                                "parameter_overrides": {
                                    "试验部位": "开关断口",
                                    "试验状态": "干",
                                    "正常次数": "6次",
                                },
                            },
                        ]
                        reason_text = "断口工频值严格大于本体值且命中户外条件，工频耐受电压试验拆分为相间及对地(干)、原项承载的相间及对地(湿)、断口(干)。"

            decisions[rule_id] = {
                "rule_id": rule_id,
                "domain": raw_rule.get("domain", ""),
                "test_item": raw_rule.get("test_item", ""),
                "kind": "split",
                "inputs": {
                    "rated_voltage_kv": rated_voltage_kv,
                    "base_voltage_kv": base_kv,
                    "fracture_voltage_kv": fracture_kv,
                    "fracture_voltage_provided": fracture_provided,
                },
                "decision": "split" if split_enabled else "single",
                "enabled": split_enabled,
                "remove_original": remove_original,
                "reason_code": reason_code,
                "reason_text": reason_text,
                "single_output": single_output,
                "split_output": split_outputs,
            }
            continue

        if rule_kind == "applicability":
            allow_when_any = raw_rule.get("allow_when_any", []) or []
            deny_when_any = raw_rule.get("deny_when_any", []) or []
            matched_conditions: list[str] = []
            matched_reason_texts: list[str] = []
            matched_deny_conditions: list[str] = []
            matched_deny_reason_texts: list[str] = []
            test_item = str(raw_rule.get("test_item", "") or "").strip()
            for condition in deny_when_any:
                if not isinstance(condition, dict):
                    continue
                if _matches_condition(condition, query):
                    label = str(condition.get("label", "") or condition.get("type", "") or "")
                    if condition.get("type") == "all":
                        label = "组合禁止条件命中"
                        matched_deny_reason_texts.append("命中组合禁止条件")
                    elif condition.get("type") == "contains":
                        matched_deny_reason_texts.append(str(condition.get("label", "") or ""))
                    elif condition.get("type") == "regex_match":
                        matched_deny_reason_texts.append(str(condition.get("pattern", "") or "regex_match"))
                    elif condition.get("type") == "contains_any":
                        matched_deny_reason_texts.append(
                            f"含有{'|'.join(str(l) for l in (condition.get('labels', []) or []))}"
                        )
                    else:
                        matched_deny_reason_texts.append(label)
                    matched_deny_conditions.append(label)
            for condition in allow_when_any:
                if not isinstance(condition, dict):
                    continue
                if _matches_condition(condition, query):
                    label = str(condition.get("label", "") or condition.get("type", "") or "")
                    if condition.get("type") == "all":
                        label = "组合条件命中"
                        matched_reason_texts.append("型号前缀不是VF1且额定电流精确等于4000A")
                    elif condition.get("type") == "contains":
                        matched_reason_texts.append(str(condition.get("label", "") or ""))
                    elif condition.get("type") == "equals_numeric":
                        matched_reason_texts.append(
                            f"{condition.get('label', '')}={condition.get('value', '')}{condition.get('unit', '')}"
                        )
                    elif condition.get("type") == "not_equals_numeric":
                        matched_reason_texts.append(
                            f"{condition.get('label', '')}!={condition.get('value', '')}{condition.get('unit', '')}"
                        )
                    elif condition.get("type") == "greater_numeric":
                        matched_reason_texts.append(
                            f"{condition.get('label', '')}>{condition.get('value', '')}{condition.get('unit', '')}"
                        )
                    elif condition.get("type") == "less_or_equal_numeric":
                        matched_reason_texts.append(
                            f"{condition.get('label', '')}<={condition.get('value', '')}{condition.get('unit', '')}"
                        )
                    elif condition.get("type") == "greater_or_equal_numeric":
                        matched_reason_texts.append(
                            f"{condition.get('label', '')}>={condition.get('value', '')}{condition.get('unit', '')}"
                        )
                    elif condition.get("type") == "less_numeric":
                        matched_reason_texts.append(
                            f"{condition.get('label', '')}<{condition.get('value', '')}{condition.get('unit', '')}"
                        )
                    elif condition.get("type") in ("ratio_greater_or_equal", "ratio_less"):
                        numerator_desc = str(condition.get("numerator_label", "") or "").strip()
                        denominator_desc = str(condition.get("denominator_label", "") or "").strip()
                        if not numerator_desc:
                            numerator_desc = "|".join(
                                str(item or "").strip()
                                for item in (condition.get("numerator_labels", []) or [])
                                if str(item or "").strip()
                            )
                        if not denominator_desc:
                            denominator_desc = "|".join(
                                str(item or "").strip()
                                for item in (condition.get("denominator_labels", []) or [])
                                if str(item or "").strip()
                            )
                        matched_reason_texts.append(
                            f"{numerator_desc}/{denominator_desc}"
                            f"{'>='+str(condition.get('value','')) if condition.get('type')=='ratio_greater_or_equal' else '<'+str(condition.get('value',''))}"
                        )
                    elif condition.get("type") == "contains_any":
                        matched_reason_texts.append(
                            f"含有{'|'.join(str(l) for l in (condition.get('labels', []) or []))}"
                        )
                    matched_conditions.append(label)
            deny_hit = bool(matched_deny_conditions)
            enabled = bool(matched_conditions) and not deny_hit
            reason_code = "allowed" if enabled else "denied"
            if enabled:
                reason_text = (
                    f"{test_item}适用。命中条件："
                    + "；".join(matched_reason_texts or matched_conditions)
                    + (
                        "。该结论只决定是否输出局部放电试验，不决定试验次数。"
                        if test_item == "局部放电试验"
                        else "。"
                    )
                )
            elif deny_hit:
                reason_text = (
                    f"命中{test_item}禁止条件："
                    + "；".join(matched_deny_reason_texts or matched_deny_conditions)
                    + f"，禁止输出{test_item}。"
                )
            else:
                reason_text = f"未命中任何{test_item}适用条件，禁止输出{test_item}。"
            decisions[rule_id] = {
                "rule_id": rule_id,
                "domain": raw_rule.get("domain", ""),
                "test_item": test_item,
                "kind": "applicability",
                "decision": "allow" if enabled else "deny",
                "enabled": enabled,
                "reason_code": reason_code,
                "reason_text": reason_text,
                "matched_conditions": matched_conditions,
                "matched_deny_conditions": matched_deny_conditions,
                "inputs": {
                    "rated_voltage_kv": rated_voltage_kv,
                    "rated_current_amp": rated_current_amp,
                    "model_prefix": model_prefix,
                    "explicit_solid_sealed_pole": explicit_solid_sealed_pole,
                },
            }
            continue

        if rule_kind == "pair_merge":
            input_cfg = raw_rule.get("inputs", {}) or {}
            base_current_labels = [
                str(item).strip()
                for item in (input_cfg.get("base_current_labels", []) or [])
                if str(item).strip()
            ]
            peak_current_labels = [
                str(item).strip()
                for item in (input_cfg.get("peak_current_labels", []) or [])
                if str(item).strip()
            ]
            threshold = input_cfg.get("ratio_threshold", 2.6)
            base_current_ka = _extract_named_current_ka(query, base_current_labels)
            peak_current_ka = _extract_named_current_ka(query, peak_current_labels)
            merge_enabled = bool(
                base_current_ka is not None
                and peak_current_ka is not None
                and float(base_current_ka) != 0
                and float(peak_current_ka) / float(base_current_ka) < float(threshold)
            )
            if merge_enabled:
                reason_code = "merged"
                reason_text = (
                    f"峰值耐受电流/短时耐受电流={peak_current_ka}/{base_current_ka}"
                    f"<{threshold}，合并输出短时耐受电流和峰值耐受电流试验。"
                )
            elif base_current_ka is None or peak_current_ka is None:
                reason_code = "missing_inputs"
                reason_text = "缺少短时耐受电流或峰值耐受电流，保持分开输出。"
            else:
                reason_code = "separate"
                reason_text = (
                    f"峰值耐受电流/短时耐受电流={peak_current_ka}/{base_current_ka}"
                    f">={threshold}，保持分开输出。"
                )
            decisions[rule_id] = {
                "rule_id": rule_id,
                "domain": raw_rule.get("domain", ""),
                "test_item": str(raw_rule.get("test_item", "") or "").strip(),
                "secondary_test_item": str(raw_rule.get("secondary_test_item", "") or "").strip(),
                "kind": "pair_merge",
                "decision": "merge" if merge_enabled else "separate",
                "enabled": merge_enabled,
                "reason_code": reason_code,
                "reason_text": reason_text,
                "inputs": {
                    "base_current_ka": base_current_ka,
                    "peak_current_ka": peak_current_ka,
                    "ratio_threshold": threshold,
                },
                "merged_output": raw_rule.get("merged_output", {}) or {},
            }
            continue

        if rule_kind == "count":
            input_cfg = raw_rule.get("inputs", {}) or {}
            pf_base_labels = input_cfg.get("power_frequency_base_labels", []) or []
            pf_fracture_labels = input_cfg.get("power_frequency_fracture_labels", []) or []
            li_base_labels = input_cfg.get("lightning_base_labels", []) or []
            li_fracture_labels = input_cfg.get("lightning_fracture_labels", []) or []
            pf_base = _extract_named_voltage_kv(query, [str(v) for v in pf_base_labels])
            pf_fracture = _extract_named_voltage_kv(
                query, [str(v) for v in pf_fracture_labels]
            )
            li_base = _extract_named_voltage_kv(query, [str(v) for v in li_base_labels])
            li_fracture = _extract_named_voltage_kv(
                query, [str(v) for v in li_fracture_labels]
            )
            pf_strictly_greater = bool(
                pf_base is not None and pf_fracture is not None and pf_fracture > pf_base
            )
            li_strictly_greater = bool(
                li_base is not None and li_fracture is not None and li_fracture > li_base
            )
            count_cfg = raw_rule.get("decision", {}) or {}
            if rated_voltage_kv == 40.5 and not pf_strictly_greater and not li_strictly_greater:
                count_value = "3次"
                reason_code = "ur_40_5_priority"
                reason_text = "Ur=40.5kV 且断口值未严格大于本体值，局放次数固定为3次。"
            elif pf_strictly_greater or li_strictly_greater:
                count_value = str(
                    count_cfg.get("count_when_fracture_strictly_greater", "9次")
                )
                reason_code = "fracture_strictly_greater"
                reason_text = "断口工频或雷电值严格大于本体值，局放次数按9次。"
            else:
                count_value = str(count_cfg.get("count_otherwise", "3次"))
                reason_code = "default_single_count"
                reason_text = "未命中断口严格大于条件，局放次数按3次。"
            decisions[rule_id] = {
                "rule_id": rule_id,
                "domain": raw_rule.get("domain", ""),
                "test_item": raw_rule.get("test_item", ""),
                "kind": "count",
                "decision": count_value,
                "enabled": True,
                "reason_code": reason_code,
                "reason_text": reason_text,
                "inputs": {
                    "rated_voltage_kv": rated_voltage_kv,
                    "power_frequency_base_kv": pf_base,
                    "power_frequency_fracture_kv": pf_fracture,
                    "lightning_base_kv": li_base,
                    "lightning_fracture_kv": li_fracture,
                    "power_frequency_fracture_strictly_greater": pf_strictly_greater,
                    "lightning_fracture_strictly_greater": li_strictly_greater,
                },
            }

    return decisions


def _apply_domain_rule_decisions_to_project_context(
    project_param_map: dict[str, list[str]],
    project_param_value_map: dict[str, dict[str, dict[str, str]]],
    domain_rule_decisions: dict[str, Any],
    schema_cfg: dict[str, Any] | None = None,
    rule_query_text: str | None = None,
    stand_type: str | None = None
) -> tuple[dict[str, list[str]], dict[str, dict[str, dict[str, str]]]]:
    updated_param_map = deepcopy(project_param_map)
    updated_value_map = deepcopy(project_param_value_map)
    schema_cfg = schema_cfg or {}
    configured_requirements = schema_cfg.get("test_item_param_requirements", {}) or {}
    # Normalized standard type for conditional logic (GB vs IEC vs DLT etc.)
    normalized_stand_type = _normalize_operate_standard_type(stand_type)

    def _normalize_test_item_lookup_key_local(value: str) -> str:
        text = _normalize_text_key(str(value))
        return text.replace("（", "(").replace("）", ")").lower()

    configured_requirements_normalized = {
        _normalize_test_item_lookup_key_local(str(test_name)): list(raw_params)
        for test_name, raw_params in configured_requirements.items()
        if _normalize_test_item_lookup_key_local(str(test_name))
        and isinstance(raw_params, list)
    }

    configured_requirement_names_by_normalized = {
        _normalize_test_item_lookup_key_local(str(test_name)): str(test_name)
        for test_name, raw_params in configured_requirements.items()
        if _normalize_test_item_lookup_key_local(str(test_name))
        and isinstance(raw_params, list)
    }

    def _resolve_existing_test_name(test_name: str) -> str:
        normalized_test_name = _normalize_test_item_lookup_key_local(test_name)
        if not normalized_test_name:
            return str(test_name)
        for existing_name in updated_param_map.keys():
            if _normalize_test_item_lookup_key_local(str(existing_name)) == normalized_test_name:
                return str(existing_name)
        for existing_name in updated_value_map.keys():
            if _normalize_test_item_lookup_key_local(str(existing_name)) == normalized_test_name:
                return str(existing_name)
        return str(test_name)

    def _resolve_config_test_name(test_name: str) -> str:
        if isinstance(configured_requirements.get(test_name), list):
            return str(test_name)
        return configured_requirement_names_by_normalized.get(
            _normalize_test_item_lookup_key_local(test_name),
            str(test_name),
        )

    def _ensure_test(test_name: str) -> None:
        resolved_test_name = _resolve_existing_test_name(test_name)
        updated_param_map.setdefault(resolved_test_name, [])
        updated_value_map.setdefault(resolved_test_name, {})

    def _ensure_param(test_name: str, param_name: str) -> None:
        _ensure_test(test_name)
        resolved_test_name = _resolve_existing_test_name(test_name)
        if param_name not in updated_param_map[resolved_test_name]:
            updated_param_map[resolved_test_name].append(param_name)
        updated_value_map[resolved_test_name].setdefault(param_name, {})

    def _ensure_test_from_config(test_name: str) -> None:
        config_test_name = _resolve_config_test_name(test_name)
        resolved_test_name = _resolve_existing_test_name(config_test_name)
        if resolved_test_name in updated_param_map:
            return
        raw_params = configured_requirements.get(config_test_name, [])
        params = [
            str(param).strip()
            for param in (raw_params if isinstance(raw_params, list) else [])
            if str(param).strip()
        ]
        updated_param_map[resolved_test_name] = params
        updated_value_map.setdefault(resolved_test_name, {})

    def _get_config_required_params(test_name: str) -> list[str]:
        config_test_name = _resolve_config_test_name(test_name)
        raw_params = configured_requirements.get(config_test_name, [])
        return [
            str(param).strip()
            for param in (raw_params if isinstance(raw_params, list) else [])
            if str(param).strip()
        ]

    def _set_param(
        test_name: str,
        param_name: str,
        value_text: str,
        *,
        value_source: str = "rule",
        value_type: str = "text",
        constraints: str | None = None,
        calc_rule: str = "",
        derive_from_rated: str = "",
        resolution_mode: str = "graph_final",
    ) -> None:
        _ensure_param(test_name, param_name)
        # Normalize special count syntax like "2(-1)*次" -> "2次"
        norm_value_text = str(value_text or "")
        calc_rule_text = str(calc_rule or "").strip()
        if str(param_name) in {"试验次数", "正常次数"}:
            m = re.match(r"^\s*([0-9]+)\s*\(\s*\-1\s*\)\*\s*(次)?\s*$", norm_value_text)
            if m:
                original = norm_value_text
                num = m.group(1)
                norm_value_text = f"{num}次"
                note = f"原始值为{original}，表示在满足特定条件时可减一次。"
                calc_rule_text = f"{calc_rule_text} {note}".strip() if calc_rule_text else note

        updated_value_map[test_name][param_name].update(
            {
                "value_text": norm_value_text,
                "value_source": value_source,
                "value_expr": norm_value_text if value_source == "user_input" else "",
                "unit": "",
                "constraints": constraints if constraints is not None else norm_value_text,
                "calc_rule": calc_rule_text,
                "derive_from_rated": derive_from_rated,
                "resolution_mode": resolution_mode,
            }
        )

    def _format_voltage_value(value: float) -> str:
        if float(value).is_integer():
            return f"{int(value)} kV"
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{text} kV"

    def _format_current_a_local(value: float) -> str:
        if float(value).is_integer():
            return f"{int(value)} A"
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{text} A"

    def _format_current_a_local_2(value: float) -> str:
        result = value * 1.1
        if float(result).is_integer():
            return f"{int(result)} A"
        text = f"{result:.2f}".rstrip("0").rstrip(".")
        return f"{text} A"

    def _parse_current_amp_value(value_text: str) -> float | None:
        text = str(value_text or "").strip()
        if not text:
            return None
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*A\b", text, flags=re.IGNORECASE)
        return float(match.group(1)) if match else None

    def _extract_named_current_ka_local(query_text: str, labels: list[str]) -> float | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        normalized = text.replace("（", "(").replace("）", ")")
        for label in labels:
            pattern = rf"{re.escape(label)}\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:kA)?"
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_named_voltage_kv_local(query_text: str, labels: list[str]) -> float | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        normalized = text.replace("（", "(").replace("）", ")")
        for label in labels:
            pattern = rf"{re.escape(label)}\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?(?:\s*\+\s*[0-9]+(?:\.[0-9]+)?)*)\s*(?:kV)?"
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                parts = re.findall(r"[0-9]+(?:\.[0-9]+)?", match.group(1))
                if parts:
                    return sum(float(part) for part in parts)
        return None

    def _extract_named_joint_voltage_parts_local(
        query_text: str, labels: list[str]
    ) -> tuple[float, float | None] | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        normalized = text.replace("（", "(").replace("）", ")")
        for label in labels:
            pattern = (
                rf"{re.escape(label)}\s*(?:[:：=]\s*)?"
                rf"([0-9\.\+\(\)\s]+?)\s*(?:kV\b|$)"
            )
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            value_text = re.sub(r"\s+", "", str(match.group(1) or ""))
            if not value_text:
                continue

            pair_match = re.fullmatch(
                r"([0-9]+(?:\.[0-9]+)?)\(\+([0-9]+(?:\.[0-9]+)?)\)",
                value_text,
            )
            if not pair_match:
                pair_match = re.fullmatch(
                    r"\(([0-9]+(?:\.[0-9]+)?)\+([0-9]+(?:\.[0-9]+)?)\)",
                    value_text,
                )
            if not pair_match:
                pair_match = re.fullmatch(
                    r"([0-9]+(?:\.[0-9]+)?)\+([0-9]+(?:\.[0-9]+)?)",
                    value_text,
                )

            if pair_match:
                primary_value = float(pair_match.group(1))
                auxiliary_value = float(pair_match.group(2))
                return primary_value, auxiliary_value

            single_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)", value_text)
            if not single_match:
                continue
            primary_value = float(single_match.group(1))
            auxiliary_value = None
            return primary_value, auxiliary_value
        return None

    def _extract_named_scalar_local(query_text: str, labels: list[str]) -> float | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        normalized = text.replace("（", "(").replace("）", ")")
        for label in labels:
            pattern = rf"{re.escape(label)}\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)"
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _detect_capacitive_test_phase_local(
        query_text: str,
        rated_voltage: float | None,
    ) -> tuple[str, str]:
        text = str(query_text or "").strip()
        normalized = text.replace("（", "(").replace("）", ")")
        explicit_match = re.search(
            r"(?:试验相数|试验方式)\s*(?:[:：=]\s*)?(单相|三相)",
            normalized,
            flags=re.IGNORECASE,
        )
        if explicit_match:
            phase = explicit_match.group(1)
            return phase, f"用户已明确给出试验相数为{phase}，直接采用。"

        if re.search(r"三相\s*断路器|三相\s*试验", normalized):
            return "三相", "问题中明确提到三相断路器/三相试验，按三相计算。"
        if re.search(r"单相\s*断路器|单相\s*试验", normalized):
            return "单相", "问题中明确提到单相断路器/单相试验，按单相计算。"

        single_count = len(re.findall(r"单相", normalized))
        three_count = len(re.findall(r"三相", normalized))
        if single_count > 0 and three_count == 0:
            return "单相", "问题中只出现单相信息，按单相计算。"
        if three_count > 0 and single_count == 0:
            return "三相", "问题中只出现三相信息，按三相计算。"

        if rated_voltage is not None and rated_voltage <= 40.5:
            return "三相", f"未明确给出试验相数，额定电压 {rated_voltage} kV 不高于40.5 kV，默认按三相计算。"
        if rated_voltage is not None:
            return "单相", f"未明确给出试验相数，额定电压 {rated_voltage} kV 高于40.5 kV，默认按单相计算。"
        return "单相", "未明确给出试验相数且无法识别额定电压，默认按单相计算。"

    def _resolve_capacitive_nonuniform_coefficient_local(
        query_text: str,
    ) -> tuple[float, str, str, bool]:
        explicit_value = _extract_named_scalar_local(query_text, ["不均匀系数"])
        if explicit_value is not None:
            return (
                explicit_value,
                str(explicit_value).rstrip("0").rstrip("."),
                "用户已明确提供不均匀系数，直接采用。",
                True,
            )

        text = str(query_text or "").strip()
        normalized = text.replace("（", "(").replace("）", ")")
        break_count_match = re.search(
            r"断口数量\s*(?:[:：=]\s*)?([0-9]+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if break_count_match:
            break_count = int(break_count_match.group(1))
            if break_count > 1:
                return 1.05, "1.05", f"问题中给出断口数量为 {break_count}，按双断口/多断口默认不均匀系数 1.05 计算。", False
            return 1.0, "1", f"问题中给出断口数量为 {break_count}，按单断口默认不均匀系数 1 计算。", False

        if any(token in normalized for token in ("双断口", "双端口", "多断口", "多端口")):
            return 1.05, "1.05", "问题中命中双断口/多断口描述，默认不均匀系数取 1.05。", False
        if any(token in normalized for token in ("单断口", "单端口")):
            return 1.0, "1", "问题中命中单断口/单端口描述，默认不均匀系数取 1。", False
        return 1.0, "1", "未明确给出不均匀系数且无法识别断口数量，默认按单断口取 1。", False

    def _resolve_break_count_local(query_text: str) -> tuple[int, str]:
        text = str(query_text or "").strip()
        normalized = text.replace("（", "(").replace("）", ")")
        break_count_match = re.search(
            r"断口数量\s*(?:[:：=]\s*)?([0-9]+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if break_count_match:
            break_count = max(int(break_count_match.group(1)), 1)
            return break_count, f"问题中给出断口数量为 {break_count}，按该断口数量计算。"
        if any(token in normalized for token in ("双断口", "双端口")):
            return 2, "问题中命中双断口/双端口描述，断口数量按 2 计算。"
        if any(token in normalized for token in ("多断口", "多端口")):
            return 2, "问题中命中多断口/多端口描述但未明确数量，断口数量默认按 2 计算。"
        if any(token in normalized for token in ("单断口", "单端口")):
            return 1, "问题中命中单断口/单端口描述，断口数量按 1 计算。"
        return 1, "未明确给出断口数量时，默认按单断口即 1 个断口计算。"

    def _resolve_capacitive_test_voltage_local(
        query_text: str,
        rated_voltage: float | None,
    ) -> tuple[str | None, str, str, str]:
        phase_value, phase_rule = _detect_capacitive_test_phase_local(
            query_text, rated_voltage
        )
        nonuniform_value, nonuniform_text, nonuniform_rule, _ = (
            _resolve_capacitive_nonuniform_coefficient_local(query_text)
        )
        break_count, break_count_rule = _resolve_break_count_local(query_text)
        kc_value = _extract_named_scalar_local(
            query_text,
            ["容性电压系数kc", "容性电压系数", "kc"],
        )
        if rated_voltage is None:
            return None, phase_value, nonuniform_text, "未识别到额定电压，无法计算容性电流开断试验试验电压。"
        if phase_value == "三相":
            return (
                _format_voltage_value(rated_voltage),
                phase_value,
                nonuniform_text,
                f"{phase_rule} 三相容性电流开断试验试验电压直接取额定电压 {_format_voltage_value(rated_voltage)}。",
            )

        kc_value = 1.2 if kc_value is None else kc_value
        kc_rule = (
            f"用户已明确给出容性电压系数 kc={str(kc_value).rstrip('0').rstrip('.')}，按该值计算。"
            if _extract_named_scalar_local(query_text, ["容性电压系数kc", "容性电压系数", "kc"]) is not None
            else "未明确给出容性电压系数 kc，默认按 1.2 计算。"
        )
        computed_voltage = (
            ((kc_value * rated_voltage) / math.sqrt(3.0) / break_count)
            * nonuniform_value
        )
        return (
            _format_voltage_value(computed_voltage),
            phase_value,
            nonuniform_text,
            f"{phase_rule} {nonuniform_rule} {break_count_rule} {kc_rule} 单相试验电压按 kc × 额定电压 / √3 / 断口数量 × 不均匀系数 计算，其中 kc={str(kc_value).rstrip('0').rstrip('.')}，即 {str(kc_value).rstrip('0').rstrip('.')} × {rated_voltage} / √3 / {break_count} × {nonuniform_text} = {_format_voltage_value(computed_voltage)}。",
        )

    def _preferred_capacitive_current_a_local(
        kind: str, rated_voltage: float | None
    ) -> float | None:
        if rated_voltage is None:
            return None
        i1_table = {
            3.6: 10.0,
            7.2: 10.0,
            12.0: 10.0,
            24.0: 10.0,
            40.5: 10.0,
            72.5: 10.0,
            126.0: 31.5,
            252.0: 125.0,
            363.0: 315.0,
            550.0: 500.0,
            800.0: 900.0,
            1100.0: 1200.0,
        }
        ic_table = {
            3.6: 10.0,
            7.2: 10.0,
            12.0: 25.0,
            24.0: 31.5,
            40.5: 50.0,
            72.5: 125.0,
            126.0: 140.0,
            252.0: 250.0,
            363.0: 355.0,
            550.0: 500.0,
        }
        table = i1_table if kind == "I1" else ic_table if kind == "Ic" else {}
        for key, value in table.items():
            if abs(float(rated_voltage) - key) < 1e-6:
                return value
        return None

    def _resolve_param_current_a(
        test_name: str,
        param_name: str,
        fallback_kind: str,
    ) -> float | None:
        param_payload = updated_value_map.get(test_name, {}).get(param_name, {}) or {}
        if isinstance(param_payload, dict):
            for field_name in ("value_text", "value_expr", "constraints", "calc_rule"):
                value = _parse_current_amp_value(str(param_payload.get(field_name, "") or ""))
                if value is not None:
                    return value
        return _preferred_capacitive_current_a_local(fallback_kind, rated_voltage_kv)

    query_text = str(rule_query_text or "").strip()
    rated_voltage_match = re.search(
        r"额定电压\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*kV\b", query_text, flags=re.IGNORECASE
    )
    rated_voltage_kv = float(rated_voltage_match.group(1)) if rated_voltage_match else None
    rated_current_match = re.search(
        r"额定电流\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*A\b",
        query_text,
        flags=re.IGNORECASE,
    )
    rated_current_a = float(rated_current_match.group(1)) if rated_current_match else None
    rated_frequency_match = re.search(
        r"额定频率\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*Hz\b",
        query_text,
        flags=re.IGNORECASE,
    )
    rated_frequency_hz = (
        float(rated_frequency_match.group(1)) if rated_frequency_match else 50.0
    )
    normalized_query_text = query_text.replace("（", "(").replace("）", ")")
    rated_closing_ka = _extract_named_current_ka_local(
        query_text,
        ["额定短路关合电流", "短路关合电流", "关合电流"],
    )
    explicit_first_pole_kpp = _extract_named_scalar_local(
        query_text,
        ["首开极系数kpp", "首开极系数"],
    )
    first_pole_kpp_from_user = explicit_first_pole_kpp is not None
    if explicit_first_pole_kpp is not None:
        first_pole_kpp = explicit_first_pole_kpp
        first_pole_kpp_rule = "用户已明确提供首开极系数 kpp，问答阶段直接采用该值。"
    elif rated_voltage_kv is not None and rated_voltage_kv <= 40.5:
        first_pole_kpp = 1.5
        first_pole_kpp_rule = f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV，首开极系数 kpp 默认取 1.5。"
    elif rated_voltage_kv is not None and rated_voltage_kv >= 72.5:
        first_pole_kpp = 1.3
        first_pole_kpp_rule = f"额定电压 {rated_voltage_kv} kV 不低于 72.5 kV，首开极系数 kpp 默认取 1.3。"
    else:
        first_pole_kpp = None
        first_pole_kpp_rule = "未明确提供首开极系数 kpp，且当前额定电压区间无默认值。"
    is_low_voltage = rated_voltage_kv is not None and rated_voltage_kv <= 40.5
    is_breaker_class_not_applicable = rated_voltage_kv is not None and rated_voltage_kv > 72.5
    pf_withstand_kv = _extract_named_voltage_kv_local(
        query_text,
        ["额定短时工频耐受电压", "额定工频耐受电压"],
    )
    pf_fracture_withstand_kv = _extract_named_voltage_kv_local(
        query_text,
        ["额定短时工频耐受电压(断口)", "额定工频耐受电压(断口)"],
    )
    li_withstand_kv = _extract_named_voltage_kv_local(
        query_text,
        ["额定雷电冲击耐受电压"],
    )
    li_fracture_withstand_kv = _extract_named_voltage_kv_local(
        query_text,
        ["额定雷电冲击耐受电压(断口)"],
    )
    pf_joint_voltage_parts = _extract_named_joint_voltage_parts_local(
        query_text,
        ["额定短时工频耐受电压(断口)", "额定工频耐受电压(断口)"],
    )
    li_joint_voltage_parts = _extract_named_joint_voltage_parts_local(
        query_text,
        ["额定雷电冲击耐受电压(断口)"],
    )
    si_withstand_kv = _extract_named_voltage_kv_local(
        query_text,
        ["额定操作冲击耐受电压"],
    )
    si_joint_voltage_parts = _extract_named_joint_voltage_parts_local(
        query_text,
        ["额定操作冲击耐受电压(断口)"],
    )
    bc_current_a = _extract_named_scalar_local(
        query_text,
        [
            "额定电容器组电流",
            "电容器组开断电流",
            "电容器组开合电流",
            "电容器电流",
            "额定背对背电容器组开断电流",
            "背对背电容器组开断电流",
            "额定单个电容器组开断电流",
            "单个电容器组开断电流",
        ],
    )
    bc_test_category = ""
    if bc_current_a is not None:
        if any(token in query_text for token in ("背对背", "Ibb")):
            bc_test_category = "背对背电容器组"
        elif any(token in query_text for token in ("单个电容器组", "单个电容器")):
            bc_test_category = "单个电容器组"
        else:
            bc_test_category = "电容器组"
    cc_current_a = _extract_named_scalar_local(
        query_text,
        [
            "额定电缆充电电流",
            "额定电缆充电开断电流",
            "额定电缆充电开合电流",
            "电缆充电开断电流",
            "电缆充电开合电流",
            "电缆充电电流",
        ],
    )
    lc_current_a = _extract_named_scalar_local(
        query_text,
        [
            "额定线路充电电流",
            "额定线路充电开断电流",
            "额定线路充电开合电流",
            "线路充电开断电流",
            "线路充电开合电流",
            "线路充电电流",
        ],
    )
    c_grade = None
    c_grade_match = re.search(
        r"(?:容性电流开合时重击穿等级|开合容性电流能力的级别|重击穿等级|能力级别|试验级别)\s*(?:[:：=]|为|是)?\s*(C[12])",
        query_text,
        flags=re.IGNORECASE,
    )
    if c_grade_match:
        c_grade = c_grade_match.group(1).upper()
    capacitive_grade_value = c_grade or "C2"
    capacitive_grade_rule = (
        "用户已提供开合容性电流能力级别信息，直接采用。"
        if c_grade
        else "未提供开合容性电流能力级别时默认按C2处理。"
    )
    capacitive_trial_count_text = "48次" if capacitive_grade_value == "C2" else "24次"
    capacitive_trial_count_rule = (
        f"开合容性电流能力级别判定为{capacitive_grade_value}，容性电流开断试验次数取{capacitive_trial_count_text}。"
    )
    capacitive_split_trial_count_text = "24次" if capacitive_grade_value == "C2" else "12次"
    capacitive_split_trial_count_rule = (
        f"开合容性电流能力级别判定为{capacitive_grade_value}，拆分后的容性电流开断试验次数取{capacitive_split_trial_count_text}。"
    )

    def _set_capacitive_current_local(
        test_name: str,
        current_a: float,
        ratio: float,
        source_label: str,
    ) -> None:
        current_text = _format_current_a_local(current_a * ratio)
        ratio_text = str(int(ratio * 100)).rstrip("0").rstrip(".")
        calc_rule = (
            f"{test_name}试验电流取{source_label}的{ratio_text}%，即 {str(current_a).rstrip('0').rstrip('.')} A × {ratio_text}% = {current_text}。"
        )
        for param_name in ("试验电流A", "试验电流"):
            _set_if_present(
                test_name,
                param_name,
                current_text,
                calc_rule=calc_rule,
            )

    capacitive_voltage_targets = (
        "容性电流开断试验(LC1)",
        "容性电流开断试验(LC1)#1",
        "容性电流开断试验(LC1)#2",
        "容性电流开断试验(LC2)",
        "容性电流开断试验(LC2)#1",
        "容性电流开断试验(LC2)#2",
        "容性电流开断试验(CC1)",
        "容性电流开断试验(CC1)#1",
        "容性电流开断试验(CC1)#2",
        "容性电流开断试验(CC2)",
        "容性电流开断试验(CC2)#1",
        "容性电流开断试验(CC2)#2",
        "容性电流开断试验(BC1)",
        "容性电流开断试验(BC2)",
        "BC1(60Hz)",
        "BC2(60Hz)",
        "CC1(60Hz)",
        "CC2(60Hz)",
        "LC1(60Hz)",
        "LC2(60Hz)",
    )
    is_three_phase_default = rated_voltage_kv is not None and rated_voltage_kv <= 40.5

    def _resolve_partial_discharge_parameters_local() -> dict[str, tuple[str, str]]:
        resolved: dict[str, tuple[str, str]] = {}
        if rated_voltage_kv is None:
            return resolved

        if rated_voltage_kv <= 40.5:
            pre_voltage = round(rated_voltage_kv * 1.3, 2)
            ac_voltage = round(rated_voltage_kv * 1.1, 2)
            resolved["局部放电值"] = (
                "≤10 pC",
                f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV，局部放电值限值取 ≤10 pC。",
            )
            resolved["预加电压"] = (
                _format_voltage_value(pre_voltage),
                f"低压局部放电试验预加电压按 1.3 × 额定电压 计算，即 1.3 × {rated_voltage_kv} kV = {_format_voltage_value(pre_voltage)}。",
            )
            resolved["交流电压"] = (
                _format_voltage_value(ac_voltage),
                f"低压局部放电试验交流电压按 1.1 × 额定电压 计算，即 1.1 × {rated_voltage_kv} kV = {_format_voltage_value(ac_voltage)}。",
            )
            return resolved

        resolved["局部放电值"] = (
            "≤5 pC",
            f"额定电压 {rated_voltage_kv} kV 高于 40.5 kV，局部放电值限值取 ≤5 pC。",
        )
        if pf_withstand_kv is not None:
            resolved["预加电压"] = (
                _format_voltage_value(pf_withstand_kv),
                f"高压局部放电试验预加电压取用户提供的额定短时工频耐受电压 {_format_voltage_value(pf_withstand_kv)}。",
            )
        if first_pole_kpp is not None:
            if abs(first_pole_kpp - 1.5) < 1e-6:
                ac_voltage = round(rated_voltage_kv * 1.2, 2)
                resolved["交流电压"] = (
                    _format_voltage_value(ac_voltage),
                    f"高压局部放电试验在首开极系数 kpp={str(first_pole_kpp).rstrip('0').rstrip('.')} 时，交流电压按 1.2 × 额定电压 计算，即 1.2 × {rated_voltage_kv} kV = {_format_voltage_value(ac_voltage)}。",
                )
            elif abs(first_pole_kpp - 1.3) < 1e-6:
                ac_voltage = round((rated_voltage_kv * 1.2) / math.sqrt(3.0), 2)
                resolved["交流电压"] = (
                    _format_voltage_value(ac_voltage),
                    f"高压局部放电试验在首开极系数 kpp={str(first_pole_kpp).rstrip('0').rstrip('.')} 时，交流电压按 1.2 × 额定电压 / √3 计算，即 1.2 × {rated_voltage_kv} / √3 = {_format_voltage_value(ac_voltage)}。",
                )
        return resolved

    def _resolve_state_t10_parameters_local() -> dict[str, tuple[str, str]]:
        resolved: dict[str, tuple[str, str]] = {}
        normalized = query_text.replace("（", "(").replace("）", ")")

        has_multiple_units = any(
            token in normalized
            for token in (
                "断路器串联关合和开断单元数量为多个",
                "断路器串联关合和开断单元数量多个",
                "串联关合和开断单元数量为多个",
                "串联关合和开断单元数量多个",
                "多个关合和开断单元",
                "多个开断单元",
                "多单元",
            )
        ) or (
            "关合和开断单元" in normalized
            and any(token in normalized for token in ("多个", "多组", "多单元"))
        )
        has_asymmetric_path = any(
            token in normalized
            for token in (
                "电流路径不对称",
                "路径不对称",
                "电流通路不对称",
                "电流回路不对称",
                "回路不对称",
            )
        )
        if has_multiple_units and has_asymmetric_path:
            resolved["试验次数"] = (
                "30次",
                "问题中明确提到断路器串联关合和开断单元数量为多个且电流路径不对称，状态检查试验(T10)试验次数取30次。",
            )
        else:
            resolved["试验次数"] = (
                "20次",
                "未识别到“多个串联关合和开断单元且电流路径不对称”的条件时，状态检查试验(T10)试验次数默认取20次。",
            )

        if rated_voltage_kv is not None and rated_voltage_kv <= 40.5:
            resolved["试验相数"] = (
                "三相",
                f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV，状态检查试验(T10)默认按三相。",
            )
        elif rated_voltage_kv is not None and rated_voltage_kv >= 72.5:
            resolved["试验相数"] = (
                "单相",
                f"额定电压 {rated_voltage_kv} kV 不低于 72.5 kV，状态检查试验(T10)默认按单相。",
            )
        elif rated_voltage_kv is not None:
            resolved["试验相数"] = (
                "单相",
                f"额定电压 {rated_voltage_kv} kV 未落在 40.5 kV 及以下档，状态检查试验(T10)默认按单相。",
            )

        if rated_voltage_kv is not None:
            if 72.5 <= rated_voltage_kv <= 252.0 and li_withstand_kv is not None:
                test_voltage = round(li_withstand_kv * 0.6, 3)
                resolved["试验电压"] = (
                    _format_voltage_value(test_voltage),
                    f"72.5 kV ≤ 额定电压 ≤ 252 kV 时，状态检查试验(T10)试验电压取额定雷电冲击耐受电压的60%，即 {li_withstand_kv} kV × 60% = {_format_voltage_value(test_voltage)}。",
                )
            elif abs(rated_voltage_kv - 363.0) < 1e-6 and si_withstand_kv is not None:
                test_voltage = round(si_withstand_kv * 0.8, 3)
                resolved["试验电压"] = (
                    _format_voltage_value(test_voltage),
                    f"额定电压为 363 kV 时，状态检查试验(T10)试验电压取额定操作冲击耐受电压的80%，即 {si_withstand_kv} kV × 80% = {_format_voltage_value(test_voltage)}。",
                )
            elif 550.0 <= rated_voltage_kv <= 1100.0 and si_withstand_kv is not None:
                test_voltage = round(si_withstand_kv * 0.9, 3)
                resolved["试验电压"] = (
                    _format_voltage_value(test_voltage),
                    f"550 kV ≤ 额定电压 ≤ 1100 kV 时，状态检查试验(T10)试验电压取额定操作冲击耐受电压的90%，即 {si_withstand_kv} kV × 90% = {_format_voltage_value(test_voltage)}。",
                )
            else:
                test_voltage = round(rated_voltage_kv * 0.5, 3)
                resolved["试验电压"] = (
                    _format_voltage_value(test_voltage),
                    f"当前额定电压 {rated_voltage_kv} kV 未命中状态检查试验(T10)的特定分段规则，回退按 50% × 额定电压 计算，即 {_format_voltage_value(test_voltage)}。",
                )

        if short_break_ka is not None:
            test_current = round(short_break_ka * 0.1, 3)
            current_text = f"{str(test_current).rstrip('0').rstrip('.')} kA"
            resolved["试验电流"] = (
                current_text,
                f"状态检查试验(T10)试验电流取额定短路开断电流的10%，即 {short_break_ka} kA × 10% = {current_text}。",
            )

        return resolved

    def _resolve_as_state_t10_parameters_local() -> dict[str, tuple[str, str]]:
        resolved: dict[str, tuple[str, str]] = {}
        if rated_voltage_kv is not None:
            if rated_voltage_kv <= 40.5:
                resolved["试验相数"] = (
                    "三相",
                    f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV，作为状态检查的T10试验按三相。",
                )
                test_voltage = round(rated_voltage_kv / 2.0, 3)
                resolved["试验电压"] = (
                    _format_voltage_value(test_voltage),
                    f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV 时，作为状态检查的T10试验电压取额定电压/2，即 {rated_voltage_kv} / 2 = {_format_voltage_value(test_voltage)}。",
                )
            elif rated_voltage_kv >= 72.5:
                resolved["试验相数"] = (
                    "单相",
                    f"额定电压 {rated_voltage_kv} kV 不低于 72.5 kV，作为状态检查的T10试验按单相。",
                )
                test_voltage = round(rated_voltage_kv / math.sqrt(3.0) / 2.0, 3)
                resolved["试验电压"] = (
                    _format_voltage_value(test_voltage),
                    f"额定电压 {rated_voltage_kv} kV 不低于 72.5 kV 时，作为状态检查的T10试验电压取额定电压/√3/2，即 {rated_voltage_kv} / √3 / 2 = {_format_voltage_value(test_voltage)}。",
                )
            else:
                resolved["试验相数"] = (
                    "单相",
                    f"额定电压 {rated_voltage_kv} kV 未落在 40.5 kV 及以下档，作为状态检查的T10试验默认按单相。",
                )
                test_voltage = round(rated_voltage_kv / 2.0, 3)
                resolved["试验电压"] = (
                    _format_voltage_value(test_voltage),
                    f"额定电压 {rated_voltage_kv} kV 未命中特定分段规则时，作为状态检查的T10试验电压取额定电压/2，即 {rated_voltage_kv} / 2 = {_format_voltage_value(test_voltage)}。",
                )

        if short_break_ka is not None:
            test_current = round(short_break_ka * 0.1, 3)
            current_text = f"{str(test_current).rstrip('0').rstrip('.')} kA"
            resolved["试验电流kA"] = (
                current_text,
                f"作为状态检查的T10试验电流取额定短路开断电流的10%，即 {short_break_ka} kA × 10% = {current_text}。",
            )

        return resolved

    def _resolve_op2_parameters_local() -> dict[str, tuple[str, str]]:
        resolved: dict[str, tuple[str, str]] = {}
        if rated_voltage_kv is None or capacitive_nonuniform_value is None:
            return resolved

        nonuniform_text = str(capacitive_nonuniform_value).rstrip("0").rstrip(".")

        if rated_voltage_kv <= 40.5:
            resolved["试验相数"] = (
                "三相",
                f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV，失步关合和开断试验(OP2)按三相。",
            )
            test_voltage = round(
                ((rated_voltage_kv * 2.5) / math.sqrt(3.0))
                * float(capacitive_nonuniform_value),
                3,
            )
            resolved["试验电压"] = (
                _format_voltage_value(test_voltage),
                f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV 时，失步关合和开断试验(OP2)试验电压按 2.5 × 额定电压 / √3 × 不均匀系数 计算，即 2.5 × {rated_voltage_kv} / √3 × {nonuniform_text} = {_format_voltage_value(test_voltage)}。",
            )
            return resolved

        if rated_voltage_kv >= 72.5:
            resolved["试验相数"] = (
                "单相",
                f"额定电压 {rated_voltage_kv} kV 不低于 72.5 kV，失步关合和开断试验(OP2)按单相。",
            )
            if first_pole_kpp is None:
                return resolved
            kpp_text = str(first_pole_kpp).rstrip("0").rstrip(".")
            if abs(first_pole_kpp - 1.3) < 1e-6:
                numerator = 2.0
            elif abs(first_pole_kpp - 1.5) < 1e-6:
                numerator = 2.5
            else:
                return resolved
            numerator_text = str(numerator).rstrip("0").rstrip(".")
            test_voltage = round(
                ((rated_voltage_kv * numerator) / math.sqrt(3.0))
                * float(capacitive_nonuniform_value),
                3,
            )
            resolved["试验电压"] = (
                _format_voltage_value(test_voltage),
                f"额定电压 {rated_voltage_kv} kV 不低于 72.5 kV 且首开极系数 kpp={kpp_text} 时，失步关合和开断试验(OP2)试验电压按 {numerator_text} × 额定电压 / √3 × 不均匀系数 计算，即 {numerator_text} × {rated_voltage_kv} / √3 × {nonuniform_text} = {_format_voltage_value(test_voltage)}。",
            )

        return resolved

    def _resolve_op1_parameters_local() -> dict[str, tuple[str, str]]:
        resolved: dict[str, tuple[str, str]] = {}
        if rated_voltage_kv is None:
            return resolved

        test_voltage = round((rated_voltage_kv * 2.5) / math.sqrt(3.0), 3)
        resolved["试验电压"] = (
            _format_voltage_value(test_voltage),
            f"失步关合和开断试验(OP1)试验电压按 额定电压 × 2.5 / √3 计算，即 {rated_voltage_kv} × 2.5 / √3 = {_format_voltage_value(test_voltage)}。",
        )
        return resolved

    def _resolve_t100s_ab_parameters_local() -> dict[str, dict[str, tuple[str, str]]]:
        resolved: dict[str, dict[str, tuple[str, str]]] = {}
        if rated_voltage_kv is None or capacitive_nonuniform_value is None:
            return resolved

        break_count, break_count_rule = _resolve_break_count_local(query_text)
        nonuniform_text = str(capacitive_nonuniform_value).rstrip("0").rstrip(".")

        t100s_a_voltage = round(
            ((rated_voltage_kv / math.sqrt(3.0)) / break_count)
            * float(capacitive_nonuniform_value),
            3,
        )
        t100s_a_value = (
            _format_voltage_value(t100s_a_voltage),
            f"T100s(a)试验电压按 额定电压 / √3 / 断口数量 × 不均匀系数 计算；{break_count_rule} 即 {rated_voltage_kv} / √3 / {break_count} × {nonuniform_text} = {_format_voltage_value(t100s_a_voltage)}。",
        )
        resolved["T100s(a)"] = {"试验电压": t100s_a_value}
        resolved["T100s(a)(60Hz)"] = {"试验电压": t100s_a_value}

        if first_pole_kpp is not None:
            kpp_text = str(first_pole_kpp).rstrip("0").rstrip(".")
            t100s_b_voltage = round(
                (((rated_voltage_kv / math.sqrt(3.0)) / break_count)
                * float(capacitive_nonuniform_value))
                * float(first_pole_kpp),
                3,
            )
            t100s_b_value = (
                _format_voltage_value(t100s_b_voltage),
                f"T100s(b)试验电压按 额定电压 / √3 / 断口数量 × 不均匀系数 × 首开极系数 计算；{break_count_rule} 即 {rated_voltage_kv} / √3 / {break_count} × {nonuniform_text} × {kpp_text} = {_format_voltage_value(t100s_b_voltage)}。",
            )
            resolved["T100s(b)"] = {"试验电压": t100s_b_value}
            resolved["T100s(b)(60Hz)"] = {"试验电压": t100s_b_value}

        return resolved

    def _resolve_single_phase_ground_fault_parameters_local() -> dict[str, tuple[str, str]]:
        resolved: dict[str, tuple[str, str]] = {}
        if rated_voltage_kv is None or capacitive_nonuniform_value is None:
            return resolved

        break_count, break_count_rule = _resolve_break_count_local(query_text)
        nonuniform_text = str(capacitive_nonuniform_value).rstrip("0").rstrip(".")
        test_voltage = round(
            ((rated_voltage_kv / math.sqrt(3.0)) / break_count)
            * float(capacitive_nonuniform_value),
            3,
        )
        resolved["试验电压"] = (
            _format_voltage_value(test_voltage),
            f"单相接地故障试验试验电压按 额定电压 / √3 / 断口数量 × 不均匀系数 计算；{break_count_rule} 即 {rated_voltage_kv} / √3 / {break_count} × {nonuniform_text} = {_format_voltage_value(test_voltage)}。",
        )
        return resolved

    def _resolve_op2_making_parameters_local() -> dict[str, tuple[str, str]]:
        resolved: dict[str, tuple[str, str]] = {}
        if rated_voltage_kv is None or capacitive_nonuniform_value is None:
            return resolved

        break_count, break_count_rule = _resolve_break_count_local(query_text)
        nonuniform_text = str(capacitive_nonuniform_value).rstrip("0").rstrip(".")
        test_voltage = round(
            (((rated_voltage_kv * 2.0) / break_count) / math.sqrt(3.0))
            * float(capacitive_nonuniform_value),
            3,
        )
        resolved["试验电压"] = (
            _format_voltage_value(test_voltage),
            f"OP2关合试验电压按 2 × 额定电压 / 断口数量 / √3 × 不均匀系数 计算；{break_count_rule} 即 2 × {rated_voltage_kv} / {break_count} / √3 × {nonuniform_text} = {_format_voltage_value(test_voltage)}。",
        )
        return resolved

    def _resolve_60hz_short_circuit_parameters_local() -> dict[str, dict[str, tuple[str, str]]]:
        resolved: dict[str, dict[str, tuple[str, str]]] = {}
        normalized = query_text.replace("（", "(").replace("）", ")")
        operation_sequence_match = re.search(
            r"(?:额定操作顺序|操作顺序)\s*(?:[:：=]|为|是)?\s*([A-Za-z0-9\.\-\s]+(?:CO|O))",
            normalized,
            flags=re.IGNORECASE,
        )
        operation_sequence_text = (
            re.sub(r"\s+", "", operation_sequence_match.group(1))
            if operation_sequence_match
            else "O-0.3s-CO-180s-CO"
        )
        operation_sequence_rule = (
            "用户已提供额定操作顺序，问答阶段直接采用该值。"
            if operation_sequence_match
            else "未提供额定操作顺序时，默认按 O-0.3s-CO-180s-CO 执行。"
        )
        break_count, break_count_rule = _resolve_break_count_local(query_text)
        min_opening_time_ms = _extract_named_scalar_local(query_text, ["最短分闸时间"])
        min_opening_time_ms = 20.0 if min_opening_time_ms is None else min_opening_time_ms
        min_opening_time_text = f"{str(min_opening_time_ms).rstrip('0').rstrip('.')} ms"
        min_opening_time_rule = (
            "用户已明确提供最短分闸时间，问答阶段直接采用该值。"
            if _extract_named_scalar_local(query_text, ["最短分闸时间"]) is not None
            else "未提供最短分闸时间时，默认按 20 ms 计算。"
        )
        time_constant_ms = _extract_named_scalar_local(query_text, ["时间常数"])
        time_constant_ms = 45.0 if time_constant_ms is None else time_constant_ms
        time_constant_text = f"{str(time_constant_ms).rstrip('0').rstrip('.')} ms"
        time_constant_rule = (
            "用户已明确提供时间常数，问答阶段直接采用该值。"
            if _extract_named_scalar_local(query_text, ["时间常数"]) is not None
            else "未提供时间常数时，默认按 45 ms 计算。"
        )

        for test_name in (
            "L75(60Hz)",
            "L90(60Hz)",
            "T100S(60Hz)",
            "近区故障试验(L75)",
            "近区故障试验(L90)",
            "短路开断试验(T100S)",
        ):
            resolved[test_name] = {
                "操作顺序": (operation_sequence_text, operation_sequence_rule),
            }
        for test_name in (
            "T100A(60Hz)",
            "短路开断试验(T100A)",
        ):
            resolved[test_name] = {
                "操作顺序": ("O", "T100A 试验操作顺序为O，无需 CO 循环。"),
            }

        if rated_voltage_kv is not None:
            l_voltage = round(rated_voltage_kv / math.sqrt(3.0), 3)
            l_voltage_text = _format_voltage_value(l_voltage)
            l_voltage_rule = (
                f"L75/L90 试验电压按额定电压 / √3 计算，即 {rated_voltage_kv} / √3 = {l_voltage_text}。"
            )
            for test_name in ("L75(60Hz)", "L90(60Hz)", "近区故障试验(L75)", "近区故障试验(L90)"):
                resolved[test_name]["试验电压"] = (l_voltage_text, l_voltage_rule)

            if rated_voltage_kv <= 40.5:
                rated_voltage_text_local = _format_voltage_value(rated_voltage_kv)
                t100a_voltage_rule = (
                    f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV 且按三相时，T100A 试验电压取额定电压 {rated_voltage_text_local}。"
                )
                t100a_phase_rule = (
                    f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV，T100A 默认按三相。"
                )
                for test_name in ("T100A(60Hz)", "短路开断试验(T100A)"):
                    resolved[test_name]["试验电压"] = (
                        rated_voltage_text_local,
                        t100a_voltage_rule,
                    )
                    resolved[test_name]["试验相数"] = (
                        "三相",
                        t100a_phase_rule,
                    )

                t100s_voltage_rule = (
                    f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV 且按三相时，T100S 试验电压取额定电压 {rated_voltage_text_local}。"
                )
                t100s_phase_rule = (
                    f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV，T100S 默认按三相。"
                )
                for test_name in ("T100S(60Hz)", "短路开断试验(T100S)"):
                    resolved[test_name]["试验电压"] = (
                        rated_voltage_text_local,
                        t100s_voltage_rule,
                    )
                    resolved[test_name]["试验相数"] = (
                        "三相",
                        t100s_phase_rule,
                    )
            elif rated_voltage_kv >= 72.5 and first_pole_kpp is not None and capacitive_nonuniform_value is not None:
                kpp_text = str(first_pole_kpp).rstrip("0").rstrip(".")
                nonuniform_text = str(capacitive_nonuniform_value).rstrip("0").rstrip(".")
                test_voltage = round(
                    (((rated_voltage_kv / math.sqrt(3.0)) / break_count)
                    * float(capacitive_nonuniform_value))
                    * float(first_pole_kpp),
                    3,
                )
                test_voltage_text = _format_voltage_value(test_voltage)
                for test_name in ("T100A(60Hz)", "短路开断试验(T100A)"):
                    resolved[test_name]["试验电压"] = (
                        test_voltage_text,
                        f"额定电压 {rated_voltage_kv} kV 不低于 72.5 kV 时，{test_name}试验电压按 额定电压 / √3 / 断口数量 × 不均匀系数 × 首开极系数 计算；{break_count_rule} 即 {rated_voltage_kv} / √3 / {break_count} × {nonuniform_text} × {kpp_text} = {test_voltage_text}。",
                    )
                    resolved[test_name]["试验相数"] = (
                        "单相",
                        f"额定电压 {rated_voltage_kv} kV 不低于 72.5 kV，{test_name}默认按单相。",
                    )

        if short_break_ka is not None:
            t100a_current_text = f"{str(short_break_ka).rstrip('0').rstrip('.')} kA"
            t100a_current_rule = (
                f"T100A 试验电流取客户输入的额定短路开断电流 {t100a_current_text}。"
            )
            for test_name in ("T100A(60Hz)", "短路开断试验(T100A)"):
                resolved[test_name]["试验电流"] = (
                    t100a_current_text,
                    t100a_current_rule,
                )

            l90_current = round(short_break_ka * 0.9, 3)
            l75_current = round(short_break_ka * 0.75, 3)
            l90_current_text = f"{str(l90_current).rstrip('0').rstrip('.')} kA"
            l75_current_text = f"{str(l75_current).rstrip('0').rstrip('.')} kA"
            l90_current_rule = (
                f"L90 试验电流取额定短路开断电流的90%，即 {short_break_ka} kA × 90% = {l90_current_text}。"
            )
            l75_current_rule = (
                f"L75 试验电流取额定短路开断电流的75%，即 {short_break_ka} kA × 75% = {l75_current_text}。"
            )
            for test_name in ("L90(60Hz)", "近区故障试验(L90)"):
                resolved[test_name]["试验电流"] = (
                    l90_current_text,
                    l90_current_rule,
                )
            for test_name in ("L75(60Hz)", "近区故障试验(L75)"):
                resolved[test_name]["试验电流"] = (
                    l75_current_text,
                    l75_current_rule,
                )

        for test_name, effective_frequency_hz in (
            ("T100A(60Hz)", 60.0),
            ("短路开断试验(T100A)", rated_frequency_hz),
        ):
            frequency_text = f"{str(effective_frequency_hz).rstrip('0').rstrip('.')} Hz"
            resolved[test_name]["最短分闸时间"] = (
                min_opening_time_text,
                min_opening_time_rule,
            )
            resolved[test_name]["时间常数"] = (
                time_constant_text,
                time_constant_rule,
            )
            resolved[test_name]["额定频率"] = (
                frequency_text,
                (
                    "T100A(60Hz)按 60 Hz 固定频率计算。"
                    if test_name == "T100A(60Hz)"
                    else "用户已明确提供额定频率时直接采用；未提供时沿用默认 50 Hz。"
                ),
            )
            frequency_offset_ms = 8.3 if abs(float(effective_frequency_hz) - 60.0) < 1e-6 else 10.0
            dc_component = math.exp(
                -((float(min_opening_time_ms) + frequency_offset_ms) / float(time_constant_ms))
            )
            dc_component_text = f"{dc_component:.4f}".rstrip("0").rstrip(".")
            resolved[test_name]["直流分量(试验)"] = (
                dc_component_text,
                f"{test_name}直流分量(试验)按 e^-[(最短分闸时间+a)/时间常数] 计算；最短分闸时间取 {min_opening_time_text}，时间常数取 {time_constant_text}，额定频率 {frequency_text} 对应 a={str(frequency_offset_ms).rstrip('0').rstrip('.')} ms，因此 e^(-(({str(min_opening_time_ms).rstrip('0').rstrip('.')}+{str(frequency_offset_ms).rstrip('0').rstrip('.')})/{str(time_constant_ms).rstrip('0').rstrip('.')})) = {dc_component_text}。",
            )

        return resolved

    def _resolve_t10_t30_t60_parameters_local() -> dict[str, dict[str, tuple[str, str]]]:
        resolved: dict[str, dict[str, tuple[str, str]]] = {}
        normalized = query_text.replace("（", "(").replace("）", ")")
        operation_sequence_match = re.search(
            r"(?:额定操作顺序|操作顺序)\s*(?:[:：=]|为|是)?\s*([A-Za-z0-9\.\-\s]+(?:CO|O))",
            normalized,
            flags=re.IGNORECASE,
        )
        operation_sequence_text = (
            re.sub(r"\s+", "", operation_sequence_match.group(1))
            if operation_sequence_match
            else "O-0.3s-CO-180s-CO"
        )
        operation_sequence_rule = (
            "用户已提供额定操作顺序，问答阶段直接采用该值。"
            if operation_sequence_match
            else "未提供额定操作顺序时，默认按 O-0.3s-CO-180s-CO 执行。"
        )
        break_count, break_count_rule = _resolve_break_count_local(query_text)

        test_ratio_map = {
            "短路开断试验(T10)": 0.1,
            "T10(60Hz)": 0.1,
            "短路开断试验(T30)": 0.3,
            "短路开断试验(T60)": 0.6,
            "T60(60Hz)": 0.6,
        }
        for test_name in test_ratio_map:
            resolved[test_name] = {
                "操作顺序": (operation_sequence_text, operation_sequence_rule),
            }

        if rated_voltage_kv is not None:
            if rated_voltage_kv <= 40.5:
                rated_voltage_text_local = _format_voltage_value(rated_voltage_kv)
                for test_name in test_ratio_map:
                    resolved[test_name]["试验电压"] = (
                        rated_voltage_text_local,
                        f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV 且按三相时，{test_name}试验电压取额定电压 {rated_voltage_text_local}。",
                    )
                    resolved[test_name]["试验相数"] = (
                        "三相",
                        f"额定电压 {rated_voltage_kv} kV 不高于 40.5 kV，{test_name}默认按三相。",
                    )
            elif rated_voltage_kv >= 72.5 and first_pole_kpp is not None and capacitive_nonuniform_value is not None:
                kpp_text = str(first_pole_kpp).rstrip("0").rstrip(".")
                nonuniform_text = str(capacitive_nonuniform_value).rstrip("0").rstrip(".")
                test_voltage = round(
                    ((rated_voltage_kv / math.sqrt(3.0)) * float(first_pole_kpp) / break_count)
                    * float(capacitive_nonuniform_value),
                    3,
                )
                test_voltage_text = _format_voltage_value(test_voltage)
                for test_name in test_ratio_map:
                    resolved[test_name]["试验电压"] = (
                        test_voltage_text,
                        f"额定电压 {rated_voltage_kv} kV 不低于 72.5 kV 时，{test_name}试验电压按 额定电压 / √3 × 首开极系数 / 断口数量 × 不均匀系数 计算；{break_count_rule} 即 {rated_voltage_kv} / √3 × {kpp_text} / {break_count} × {nonuniform_text} = {test_voltage_text}。",
                    )
                    resolved[test_name]["试验相数"] = (
                        "单相",
                        f"额定电压 {rated_voltage_kv} kV 不低于 72.5 kV，{test_name}默认按单相。",
                    )

        if short_break_ka is not None:
            for test_name, ratio in test_ratio_map.items():
                current_ka = round(short_break_ka * ratio, 3)
                current_text = f"{str(current_ka).rstrip('0').rstrip('.')} kA"
                resolved[test_name]["试验电流"] = (
                    current_text,
                    f"{test_name}试验电流取额定短路开断电流的{int(ratio * 100)}%，即 {short_break_ka} kA × {int(ratio * 100)}% = {current_text}。",
                )

        return resolved

    dielectric_value = (
        "充气/充油"
        if any(token in query_text for token in ("SF6", "六氟化硫", "充气断路器", "充油断路器"))
        else "正常"
    )
    def _set_if_present(
        test_name: str,
        param_name: str,
        value_text: str,
        *,
        calc_rule: str = "",
    ) -> None:
        if test_name not in updated_param_map:
            return
        if param_name not in updated_param_map.get(test_name, []):
            return
        _set_param(
            test_name,
            param_name,
            value_text,
            value_source="rule",
            value_type="text",
            constraints=value_text,
            calc_rule=calc_rule,
            resolution_mode="graph_final",
        )

    def _remove_param_if_present(test_name: str, param_name: str) -> None:
        if test_name not in updated_param_map:
            return
        params = updated_param_map.get(test_name, []) or []
        if param_name in params:
            updated_param_map[test_name] = [item for item in params if item != param_name]
        test_values = updated_value_map.get(test_name, {}) or {}
        if param_name in test_values:
            test_values.pop(param_name, None)

    partial_discharge_parameters = _resolve_partial_discharge_parameters_local()

    def _set_if_value_missing(
        test_name: str,
        param_name: str,
        value_text: str,
        *,
        value_source: str = "rule",
        calc_rule: str = "",
        derive_from_rated: str = "",
    ) -> None:
        if test_name not in updated_param_map:
            return
        if param_name not in updated_param_map.get(test_name, []):
            return
        current_entry = updated_value_map.get(test_name, {}).get(param_name, {}) or {}
        if isinstance(current_entry, dict) and any(
            str(current_entry.get(field_name, "") or "").strip()
            for field_name in (
                "value_text",
                "value_expr",
                "constraints",
                "calc_rule",
                "derive_from_rated",
            )
        ):
            return
        _set_param(
            test_name,
            param_name,
            value_text,
            value_source=value_source,
            value_type="text",
            constraints=value_text,
            calc_rule=calc_rule,
            derive_from_rated=derive_from_rated,
            resolution_mode="graph_final",
        )

    def _entry_has_any_value(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        return any(
            str(entry.get(field_name, "") or "").strip()
            for field_name in (
                "value_text",
                "value_expr",
                "constraints",
                "calc_rule",
                "derive_from_rated",
            )
        )

    def _set_entry_if_value_missing(
        test_name: str,
        param_name: str,
        entry: dict[str, Any],
    ) -> None:
        if not entry:
            return
        _ensure_param(test_name, param_name)
        current_entry = updated_value_map.get(test_name, {}).get(param_name, {}) or {}
        if _entry_has_any_value(current_entry):
            return
        updated_value_map[test_name][param_name] = deepcopy(entry)

    def _copy_param_entry_if_missing(
        test_name: str,
        param_name: str,
        source_specs: list[tuple[str, list[str]]],
    ) -> bool:
        for source_test_name, source_param_names in source_specs:
            source_values = updated_value_map.get(source_test_name, {}) or {}
            if not isinstance(source_values, dict):
                continue
            for source_param_name in source_param_names:
                source_entry = source_values.get(source_param_name, {}) or {}
                if not _entry_has_any_value(source_entry):
                    continue
                _set_entry_if_value_missing(test_name, param_name, deepcopy(source_entry))
                current_entry = updated_value_map.get(test_name, {}).get(param_name, {}) or {}
                if _entry_has_any_value(current_entry):
                    return True
        return False

    def _set_all_if_value_missing(
        param_name: str,
        value_text: str,
        *,
        value_source: str = "rule",
        calc_rule: str = "",
        derive_from_rated: str = "",
    ) -> None:
        for test_name, params in updated_param_map.items():
            if not isinstance(params, list):
                continue
            if param_name not in params:
                continue
            _set_if_value_missing(
                str(test_name),
                param_name,
                value_text,
                value_source=value_source,
                calc_rule=calc_rule,
                derive_from_rated=derive_from_rated,
            )

    for decision in domain_rule_decisions.values():
        if not isinstance(decision, dict):
            continue
        rule_kind = str(decision.get("kind", "") or "")
        test_item = str(decision.get("test_item", "") or "")
        rule_id = str(decision.get("rule_id", "") or "")

        if rule_kind == "applicability":
            if decision.get("enabled"):
                _ensure_test_from_config(test_item)
            else:
                resolved_test_item = _resolve_existing_test_name(test_item)
                updated_param_map.pop(resolved_test_item, None)
                updated_value_map.pop(resolved_test_item, None)
            continue

        if rule_kind == "pair_merge":
            merged_output = decision.get("merged_output", {}) or {}
            merged_test_name = str(merged_output.get("test_item", "") or "").strip()
            if not decision.get("enabled"):
                if merged_test_name:
                    normalized_merged_name = _normalize_test_item_lookup_key_local(
                        merged_test_name
                    )
                    stale_merged_names: list[str] = []
                    for existing_name in tuple(updated_param_map.keys()):
                        if (
                            _normalize_test_item_lookup_key_local(str(existing_name))
                            == normalized_merged_name
                        ):
                            stale_merged_names.append(str(existing_name))
                    for existing_name in tuple(updated_value_map.keys()):
                        if (
                            _normalize_test_item_lookup_key_local(str(existing_name))
                            == normalized_merged_name
                        ):
                            stale_merged_names.append(str(existing_name))
                    stale_merged_names.extend(
                        [
                            _resolve_existing_test_name(merged_test_name),
                            _resolve_config_test_name(merged_test_name),
                            merged_test_name,
                        ]
                    )
                    for stale_name in dict.fromkeys(
                        name for name in stale_merged_names if str(name).strip()
                    ):
                        updated_param_map.pop(str(stale_name), None)
                        updated_value_map.pop(str(stale_name), None)
                continue
            secondary_test_item = str(decision.get("secondary_test_item", "") or "").strip()
            if not test_item or not secondary_test_item or not merged_test_name:
                continue
            primary_params = list(updated_param_map.get(test_item, []) or [])
            secondary_params = list(updated_param_map.get(secondary_test_item, []) or [])
            primary_values = deepcopy(updated_value_map.get(test_item, {}) or {})
            secondary_values = deepcopy(updated_value_map.get(secondary_test_item, {}) or {})
            if not primary_params and not secondary_params:
                continue
            merged_params = _get_config_required_params(merged_test_name) or list(
                dict.fromkeys(primary_params + secondary_params)
            )
            merged_values = deepcopy(primary_values)
            for param_name, param_value in secondary_values.items():
                merged_values.setdefault(str(param_name), deepcopy(param_value))
            updated_param_map.pop(test_item, None)
            updated_param_map.pop(secondary_test_item, None)
            updated_value_map.pop(test_item, None)
            updated_value_map.pop(secondary_test_item, None)
            updated_param_map[merged_test_name] = merged_params
            updated_value_map[merged_test_name] = merged_values
            for param_name, value_text in (
                merged_output.get("parameter_overrides", {}) or {}
            ).items():
                _set_param(
                    merged_test_name,
                    str(param_name),
                    str(value_text),
                    value_source="rule",
                    value_type="text",
                    constraints=str(value_text),
                    calc_rule=str(decision.get("reason_text", "") or ""),
                    resolution_mode="graph_final",
                )
            continue

        if rule_kind == "count" and test_item == "局部放电试验":
            count_value = str(decision.get("decision", "") or "").strip()
            rule_inputs = decision.get("inputs", {}) if isinstance(decision, dict) else {}
            rated_voltage_kv = (
                float(rule_inputs.get("rated_voltage_kv"))
                if isinstance(rule_inputs, dict)
                and rule_inputs.get("rated_voltage_kv") is not None
                else None
            )
            if rated_voltage_kv is not None:
                pre_time = "30s" if rated_voltage_kv == 40.5 else "60s"
                for param_name in ("局部放电值", "预加电压", "交流电压"):
                    resolved_param = partial_discharge_parameters.get(param_name)
                    if not resolved_param:
                        continue
                    value_text, calc_rule = resolved_param
                    _set_param(
                        "局部放电试验",
                        param_name,
                        value_text,
                        value_source="rule",
                        value_type="text",
                        constraints=value_text,
                        calc_rule=calc_rule,
                        resolution_mode="graph_final",
                    )
                _set_param(
                    "局部放电试验",
                    "预加时间",
                    pre_time,
                    value_source="rule",
                    value_type="text",
                    constraints=pre_time,
                    calc_rule=(
                        "Ur=40.5kV 时预加时间为30s。"
                        if rated_voltage_kv == 40.5
                        else "Ur=12kV 时预加时间为60s。"
                    ),
                    resolution_mode="graph_final",
                )
                _set_param(
                    "局部放电试验",
                    "测量时间(min)",
                    "1min",
                    value_source="rule",
                    value_type="text",
                    constraints="1min",
                    calc_rule="局部放电测量时间固定为1min。",
                    resolution_mode="graph_final",
                )
            if count_value:
                _set_param(
                    "局部放电试验",
                    "试验次数",
                    count_value,
                    value_source="rule",
                    value_type="text",
                    constraints=count_value,
                    calc_rule=str(decision.get("reason_text", "") or count_value),
                    resolution_mode="graph_final",
                )
            continue

        if rule_kind != "split" or not test_item:
            continue

        source_params = list(updated_param_map.get(test_item, []) or [])
        source_values = deepcopy(updated_value_map.get(test_item, {}) or {})
        if not source_params:
            continue

        if decision.get("enabled"):
            remove_original = bool(decision.get("remove_original", True))
            if remove_original:
                updated_param_map.pop(test_item, None)
                updated_value_map.pop(test_item, None)
            for split_output in decision.get("split_output", []) or []:
                if not isinstance(split_output, dict):
                    continue
                target_name = str(split_output.get("test_item", "") or "").strip()
                if not target_name:
                    continue
                target_params = _get_config_required_params(target_name) or list(source_params)
                updated_param_map[target_name] = _normalize_count_param_names(
                    target_name,
                    list(target_params),
                )
                inherited_value_map: dict[str, dict[str, str]] = {}
                for param_name in updated_param_map[target_name]:
                    if param_name in source_values:
                        inherited_value_map[param_name] = deepcopy(source_values[param_name])
                    elif param_name == "正常次数" and "试验次数" in source_values:
                        inherited_value_map[param_name] = deepcopy(source_values["试验次数"])
                    elif param_name == "试验次数" and "正常次数" in source_values:
                        inherited_value_map[param_name] = deepcopy(source_values["正常次数"])
                updated_value_map[target_name] = inherited_value_map
                additional_params = split_output.get("additional_params", []) or []
                if isinstance(additional_params, list):
                    for param_name in additional_params:
                        normalized_param_name = str(param_name or "").strip()
                        if not normalized_param_name:
                            continue
                        if normalized_param_name not in updated_param_map[target_name]:
                            updated_param_map[target_name].append(normalized_param_name)
                        updated_value_map[target_name].setdefault(normalized_param_name, {})
                for param_name, value_text in (
                    split_output.get("parameter_overrides", {}) or {}
                ).items():
                    is_count_override = str(param_name) in {"试验次数", "正常次数"}
                    value_source = "formula" if is_count_override else "rule"
                    calc_rule = (
                        str(decision.get("reason_text", "") or value_text)
                        if is_count_override
                        else ""
                    )
                    _set_param(
                        target_name,
                        str(param_name),
                        str(value_text),
                        value_source="rule",
                        value_type="text",
                        constraints=str(value_text),
                        calc_rule=calc_rule,
                        resolution_mode="graph_final",
                    )
        else:
            single_output = decision.get("single_output", {}) or {}
            for param_name, value_text in (single_output.get("parameter_overrides", {}) or {}).items():
                _set_param(
                    test_item,
                    str(param_name),
                    str(value_text),
                    value_source="rule",
                    value_type="text",
                    resolution_mode="graph_final",
                )

    if pf_withstand_kv is not None:
        pf_value_text = _format_voltage_value(pf_withstand_kv)
        pf_fracture_value_text = (
            _format_voltage_value(pf_fracture_withstand_kv)
            if pf_fracture_withstand_kv is not None
            else pf_value_text
        )
        pf_voltage_targets = {
            "工频耐受电压试验": pf_value_text,
            "工频耐受电压试验#干": pf_value_text,
            "工频耐受电压试验#湿": pf_value_text,
            "工频耐受电压试验(干)": pf_value_text,
            "工频耐受电压试验(湿)": pf_value_text,
            "工频耐受电压试验(断口)": pf_fracture_value_text,
            "工频耐受电压试验(相间及对地)": pf_value_text,
        }
        for test_name, target_value_text in pf_voltage_targets.items():
            if test_name not in updated_param_map:
                continue
            if test_name == "工频耐受电压试验(断口)" and pf_fracture_withstand_kv is not None:
                calc_rule = "用户已明确提供额定短时工频耐受电压(断口)，问答阶段直接采用该值，不再输出“按表选取”的条件文本。"
            else:
                calc_rule = "用户已明确提供额定短时工频耐受电压，问答阶段直接采用该值，不再输出“按表选取”的条件文本。"
            _set_param(
                test_name,
                "交流电压",
                target_value_text,
                value_source="user_input",
                value_type="text",
                constraints=target_value_text,
                calc_rule=calc_rule,
                resolution_mode="graph_final",
            )
        if "作为状态检查的工频耐受电压试验" in updated_param_map:
            state_check_pf_kv = round(pf_withstand_kv * 0.8, 2)
            state_check_pf_text = _format_voltage_value(state_check_pf_kv)
            _set_param(
                "作为状态检查的工频耐受电压试验",
                "交流电压",
                state_check_pf_text,
                value_source="rule",
                value_type="text",
                constraints=state_check_pf_text,
                calc_rule=f"状态检查工频耐受电压取额定短时工频耐受电压的80%，即 0.8 × {pf_withstand_kv} kV = {state_check_pf_text}。",
                resolution_mode="graph_final",
            )

    if li_withstand_kv is not None:
        li_value_text = _format_voltage_value(li_withstand_kv)
        li_fracture_value_text = (
            _format_voltage_value(li_fracture_withstand_kv)
            if li_fracture_withstand_kv is not None
            else li_value_text
        )
        li_voltage_targets = {
            "雷电冲击耐受电压试验": li_value_text,
            "雷电冲击耐受电压试验(断口)": li_fracture_value_text,
            "雷电冲击耐受电压试验(相间及对地)": li_value_text,
        }
        for test_name, target_value_text in li_voltage_targets.items():
            if test_name not in updated_param_map:
                continue
            if test_name == "雷电冲击耐受电压试验(断口)" and li_fracture_withstand_kv is not None:
                calc_rule = "用户已明确提供额定雷电冲击耐受电压(断口)，问答阶段直接采用该值，不再输出“按表选取”的条件文本。"
            else:
                calc_rule = "用户已明确提供额定雷电冲击耐受电压，问答阶段直接采用该值，不再输出“按表选取”的条件文本。"
            _set_param(
                test_name,
                "雷电冲击干耐受电压",
                target_value_text,
                value_source="user_input",
                value_type="text",
                constraints=target_value_text,
                calc_rule=calc_rule,
                resolution_mode="graph_final",
            )

    if pf_joint_voltage_parts is not None and "工频耐受电压试验(联合电压)" in updated_param_map:
        pf_joint_primary_kv, pf_joint_auxiliary_kv = pf_joint_voltage_parts
        pf_joint_primary_text = _format_voltage_value(pf_joint_primary_kv)
        _set_if_present(
            "工频耐受电压试验(联合电压)",
            "交流电压",
            pf_joint_primary_text,
            calc_rule=(
                "工频联合电压试验的交流电压取额定短时工频耐受电压(断口)中的主值，"
                f"即 {pf_joint_primary_text}。"
            ),
        )
        if pf_joint_auxiliary_kv is not None:
            pf_joint_auxiliary_text = _format_voltage_value(pf_joint_auxiliary_kv)
            _set_if_present(
                "工频耐受电压试验(联合电压)",
                "交流电压(辅)",
                pf_joint_auxiliary_text,
                calc_rule=(
                    "工频联合电压试验的交流电压(辅)取额定短时工频耐受电压(断口)中括号内的辅值，"
                    f"即 {pf_joint_auxiliary_text}。"
                ),
            )

    if li_joint_voltage_parts is not None and "雷电冲击耐受电压试验(联合电压)" in updated_param_map:
        li_joint_primary_kv, li_joint_auxiliary_kv = li_joint_voltage_parts
        li_joint_primary_text = _format_voltage_value(li_joint_primary_kv)
        _set_if_present(
            "雷电冲击耐受电压试验(联合电压)",
            "雷电冲击干耐受电压",
            li_joint_primary_text,
            calc_rule=(
                "雷电联合电压试验的雷电冲击干耐受电压取额定雷电冲击耐受电压(断口)中的主值，"
                f"即 {li_joint_primary_text}。"
            ),
        )
        if li_joint_auxiliary_kv is not None:
            li_joint_auxiliary_text = _format_voltage_value(li_joint_auxiliary_kv)
            _set_if_present(
                "雷电冲击耐受电压试验(联合电压)",
                "交流电压",
                li_joint_auxiliary_text,
                calc_rule=(
                    "雷电联合电压试验的交流电压取额定雷电冲击耐受电压(断口)中括号内的辅值，"
                    f"即 {li_joint_auxiliary_text}。"
                ),
            )

    if si_joint_voltage_parts is not None and "操作冲击耐受电压试验(联合电压)" in updated_param_map:
        si_joint_primary_kv, si_joint_auxiliary_kv = si_joint_voltage_parts
        si_joint_primary_text = _format_voltage_value(si_joint_primary_kv)
        _set_if_present(
            "操作冲击耐受电压试验(联合电压)",
            "操作冲击电压",
            si_joint_primary_text,
            calc_rule=(
                "操作联合电压试验的操作冲击电压取额定操作冲击耐受电压(断口)中的主值，"
                f"即 {si_joint_primary_text}。"
            ),
        )
        _set_if_present(
            "操作冲击耐受电压试验(联合电压)",
            "操作冲击耐受电压",
            si_joint_primary_text,
            calc_rule=(
                "操作联合电压试验的操作冲击耐受电压取额定操作冲击耐受电压(断口)中的主值，"
                f"即 {si_joint_primary_text}。"
            ),
        )
        if si_joint_auxiliary_kv is not None:
            si_joint_auxiliary_text = _format_voltage_value(si_joint_auxiliary_kv)
            _ensure_param("操作冲击耐受电压试验(联合电压)", "交流电压")
            _set_if_present(
                "操作冲击耐受电压试验(联合电压)",
                "交流电压",
                si_joint_auxiliary_text,
                calc_rule=(
                    "操作联合电压试验的交流电压取额定操作冲击耐受电压(断口)中括号内的辅值，"
                    f"即 {si_joint_auxiliary_text}。"
                ),
            )

    if bc_current_a is not None:
        bc1_current_text = (
            f"{_format_current_a_local(bc_current_a * 0.1)}~{_format_current_a_local(bc_current_a * 0.4)}"
        )
        bc_overrides = {
            "容性电流开断试验(BC1)": {
                "value_text": bc1_current_text,
                "constraints": bc1_current_text,
                "calc_rule": "用户已明确提供背对背电容器组开断电流(Ibb)，BC1 试验电流按 Ibb 的 10%~40% 取值。",
            },
            "容性电流开断试验(BC2)": {
                "value_text": _format_current_a_local(bc_current_a),
                "constraints": _format_current_a_local(bc_current_a),
                "calc_rule": "用户已明确提供背对背电容器组开断电流(Ibb)，BC2 试验电流直接采用 Ibb。",
            },
        }
        for test_name, override in bc_overrides.items():
            if test_name not in updated_param_map:
                continue
            _set_param(
                test_name,
                "试验电流A",
                str(override["value_text"]),
                value_source="user_input",
                value_type="text",
                constraints=str(override["constraints"]),
                calc_rule=str(override["calc_rule"]),
                resolution_mode="graph_final",
            )
            if rated_voltage_kv is not None:
                _set_param(
                    test_name,
                    "试验电压",
                    _format_voltage_value(rated_voltage_kv),
                    value_source="rule",
                    value_type="text",
                    constraints=_format_voltage_value(rated_voltage_kv),
                    calc_rule=f"40.5kV及以下三相试验时，试验电压取额定电压 {rated_voltage_kv} kV。",
                    resolution_mode="graph_final",
                )
            if bc_test_category:
                _set_param(
                    test_name,
                    "试验类别",
                    bc_test_category,
                    value_source="rule",
                    value_type="text",
                    constraints=bc_test_category,
                    calc_rule=f"根据用户提供的开断电流标签，试验类别确定为{bc_test_category}。",
                    resolution_mode="graph_final",
                )
            _set_param(
                test_name,
                "开合容性电流能力的级别",
                c_grade or "C2",
                value_source="rule" if c_grade else "default",
                value_type="text",
                constraints=c_grade or "C2",
                calc_rule="用户已提供开合容性电流能力级别信息，直接采用。" if c_grade else "未提供开合容性电流能力级别时默认按C2处理。",
                resolution_mode="graph_final",
            )

    for temperature_rise_test_item in ("温升试验", "温升试验(60Hz)"):
        if rated_voltage_kv is not None and temperature_rise_test_item in updated_param_map:
            _set_param(
                temperature_rise_test_item,
                "额定电压",
                _format_voltage_value(rated_voltage_kv),
                value_source="user_input",
                value_type="text",
                constraints=_format_voltage_value(rated_voltage_kv),
                calc_rule="用户已明确提供额定电压，温升试验直接采用该值。",
                resolution_mode="graph_final",
            )
    if rated_current_a is not None and any(
        temperature_rise_test_item in updated_param_map
        for temperature_rise_test_item in ("温升试验", "温升试验(60Hz)")
    ):
        normalized_stand_type = _normalize_operate_standard_type(stand_type)
        if normalized_stand_type == "DLT":
            logger.info(f"温升试验_DLT:{normalized_stand_type}")
            current_text = _format_current_a_local_2(rated_current_a)
            calc_rule = (
                f"DL/T标准要求温升试验电流为额定电流的110%，"
                f"即 {int(rated_current_a)}A × 110% = {current_text}。"
            )
            value_source = "formula"
        elif normalized_stand_type == "IEC":
            current_text = _format_current_a_local(rated_current_a)
            calc_rule = (
                f"IEC标准要求温升试验电流取额定电流，"
                f"即 {int(rated_current_a)}A。"
            )
            value_source = "user_input"
        else:
            current_text = _format_current_a_local(rated_current_a)
            calc_rule = (
                f"温升试验电流取额定电流，"
                f"即 {int(rated_current_a)}A。"
            )
            value_source = "user_input"
        for temperature_rise_test_item in ("温升试验", "温升试验(60Hz)"):
            if temperature_rise_test_item in updated_param_map:
                _set_param(
                    temperature_rise_test_item,
                    "试验电流A",
                    current_text,
                    value_source=value_source,
                    value_type="text",
                    constraints=current_text,
                    calc_rule=calc_rule,
                    resolution_mode="graph_final",
                )
    _set_if_present("温升试验", "试验部位", "主回路", calc_rule="温升试验试验部位固定为主回路。")
    _set_if_present("温升试验", "试验次数", "1次", calc_rule="温升试验按1次执行。")
    _set_if_present("温升试验(60Hz)", "试验部位", "主回路", calc_rule="温升试验(60Hz)试验部位固定为主回路。")
    _set_if_present("温升试验(60Hz)", "试验次数", "1次", calc_rule="温升试验(60Hz)按1次执行。")
    _set_if_present("辅助和控制回路温升试验", "试验次数", "1次", calc_rule="辅助和控制回路温升试验默认按1次执行。")
    _set_if_present("回路电阻测量", "试验次数", "2次", calc_rule="回路电阻测量按2次执行。")

    # Programmatically finalize high-frequency insulation defaults so the model
    # does not need to resolve conditional default prose on its own.
    if is_three_phase_default:
        for test_name in (
            "工频耐受电压试验",
            "工频耐受电压试验#干",
            "工频耐受电压试验#湿",
            "工频耐受电压试验(干)",
            "工频耐受电压试验(湿)",
            "工频耐受电压试验(断口)",
            "工频耐受电压试验(相间及对地)",
            "雷电冲击耐受电压试验",
            "雷电冲击耐受电压试验(联合电压)",
            "雷电冲击耐受电压试验(断口)",
            "雷电冲击耐受电压试验(相间及对地)",
            "局部放电试验",
            "T60(预备试验)",
            "温升试验",
            "温升试验(60Hz)",
            "状态检查试验(T10)",
            "短路开断试验(T10)",
            "短路开断试验(T30)",
            "短路开断试验(T60)",
            "短路开断试验(T100S)",
            "容性电流开断试验(LC1)",
            "容性电流开断试验(LC2)",
            "容性电流开断试验(CC1)",
            "容性电流开断试验(CC2)",
            "容性电流开断试验(BC1)",
            "容性电流开断试验(BC2)",
        ):
            _set_if_present(
                test_name,
                "试验相数",
                "三相",
                calc_rule="40.5kV及以下默认三相。",
            )

    rated_voltage_text = _format_voltage_value(rated_voltage_kv) if rated_voltage_kv is not None else ""

    if rated_voltage_kv is not None:
        _set_all_if_value_missing(
            "额定电压",
            rated_voltage_text,
            value_source="user_input",
            calc_rule="用户已明确提供额定电压，问答阶段直接采用该值。",
        )

    if rated_voltage_kv is not None:
        for test_name in (
            "容性电流开断试验(LC1)",
            "容性电流开断试验(LC2)",
            "容性电流开断试验(CC1)",
            "容性电流开断试验(CC2)",
        ):
            _set_if_value_missing(
                test_name,
                "额定电压",
                rated_voltage_text,
                value_source="user_input",
                calc_rule="用户已明确提供额定电压，问答阶段直接采用该值。",
            )
            _set_if_present(
                test_name,
                "开合容性电流能力的级别",
                capacitive_grade_value,
                calc_rule=capacitive_grade_rule,
            )
            _set_if_value_missing(
                test_name,
                "操作顺序",
                "O",
                value_source="standard",
                calc_rule="容性电流开断试验操作顺序按标准值 O 执行。",
            )
            _set_if_value_missing(
                test_name,
                "试验次数",
                capacitive_trial_count_text,
                value_source="standard",
                calc_rule=capacitive_trial_count_rule,
            )

    capacitive_test_voltage_text = None
    capacitive_phase_value = ""
    capacitive_nonuniform_value = None
    capacitive_nonuniform_text = ""
    capacitive_voltage_rule = ""
    capacitive_nonuniform_rule = ""
    capacitive_nonuniform_from_user = False
    global_test_phase_value, global_test_phase_rule = _detect_capacitive_test_phase_local(
        query_text,
        rated_voltage_kv,
    )
    global_test_phase_from_user = re.search(
        r"(?:试验相数|试验方式)\s*(?:[:：=]\s*)?(单相|三相)",
        normalized_query_text,
        flags=re.IGNORECASE,
    ) is not None
    global_break_count, global_break_count_rule = _resolve_break_count_local(query_text)
    if rated_voltage_kv is not None:
        (
            capacitive_test_voltage_text,
            capacitive_phase_value,
            capacitive_nonuniform_text,
            capacitive_voltage_rule,
        ) = _resolve_capacitive_test_voltage_local(query_text, rated_voltage_kv)
        (
            capacitive_nonuniform_value,
            _,
            capacitive_nonuniform_rule,
            capacitive_nonuniform_from_user,
        ) = _resolve_capacitive_nonuniform_coefficient_local(query_text)

    if capacitive_test_voltage_text:
        for test_name in capacitive_voltage_targets:
            _set_if_present(
                test_name,
                "试验电压",
                capacitive_test_voltage_text,
                calc_rule=capacitive_voltage_rule,
            )
            _set_if_present(
                test_name,
                "试验相数",
                capacitive_phase_value,
                calc_rule="容性电流开断试验的试验相数按问题描述和额定电压默认规则判定。",
            )

    for test_name in (
        "容性电流开断试验(BC1)",
        "BC1(60Hz)",
    ):
        if bc_current_a is not None:
            _set_capacitive_current_local(test_name, bc_current_a, 0.4, "额定电容器组电流")
        if bc_test_category:
            _set_if_present(
                test_name,
                "试验类别",
                bc_test_category,
                calc_rule=f"根据用户提供的电流标签，试验类别确定为{bc_test_category}。",
            )

    for test_name in (
        "容性电流开断试验(BC2)",
        "BC2(60Hz)",
    ):
        if bc_current_a is not None:
            _set_capacitive_current_local(test_name, bc_current_a, 1.0, "额定电容器组电流")
        if bc_test_category:
            _set_if_present(
                test_name,
                "试验类别",
                bc_test_category,
                calc_rule=f"根据用户提供的电流标签，试验类别确定为{bc_test_category}。",
            )

    for test_name in (
        "容性电流开断试验(CC1)",
        "容性电流开断试验(CC1)#1",
        "容性电流开断试验(CC1)#2",
        "CC1(60Hz)",
    ):
        if cc_current_a is not None:
            _set_capacitive_current_local(test_name, cc_current_a, 0.4, "额定电缆充电电流")

    for test_name in (
        "容性电流开断试验(CC2)",
        "容性电流开断试验(CC2)#1",
        "容性电流开断试验(CC2)#2",
        "CC2(60Hz)",
    ):
        if cc_current_a is not None:
            _set_capacitive_current_local(test_name, cc_current_a, 1.0, "额定电缆充电电流")

    for test_name in (
        "容性电流开断试验(LC1)",
        "容性电流开断试验(LC1)#1",
        "容性电流开断试验(LC1)#2",
        "LC1(60Hz)",
    ):
        if lc_current_a is not None:
            _set_capacitive_current_local(test_name, lc_current_a, 0.4, "额定线路充电电流")

    for test_name in (
        "容性电流开断试验(LC2)",
        "容性电流开断试验(LC2)#1",
        "容性电流开断试验(LC2)#2",
        "LC2(60Hz)",
    ):
        if lc_current_a is not None:
            _set_capacitive_current_local(test_name, lc_current_a, 1.0, "额定线路充电电流")

    for test_name in (
        "容性电流开断试验(LC1)",
        "容性电流开断试验(LC1)#1",
        "容性电流开断试验(LC1)#2",
        "容性电流开断试验(LC2)",
        "容性电流开断试验(LC2)#1",
        "容性电流开断试验(LC2)#2",
        "容性电流开断试验(CC1)",
        "容性电流开断试验(CC1)#1",
        "容性电流开断试验(CC1)#2",
        "容性电流开断试验(CC2)",
        "容性电流开断试验(CC2)#1",
        "容性电流开断试验(CC2)#2",
    ):
        _set_if_present(
            test_name,
            "开合容性电流能力的级别",
            capacitive_grade_value,
            calc_rule=capacitive_grade_rule,
        )
        _set_if_present(
            test_name,
            "试验次数",
            capacitive_trial_count_text,
            calc_rule=capacitive_trial_count_rule,
        )
    for test_name in (
        "容性电流开断试验(LC2)#1",
        "容性电流开断试验(LC2)#2",
        "容性电流开断试验(CC2)#1",
        "容性电流开断试验(CC2)#2",
    ):
        _set_if_present(
            test_name,
            "试验次数",
            capacitive_split_trial_count_text,
            calc_rule=capacitive_split_trial_count_rule,
        )
    if rated_voltage_kv is not None and "T60(预备试验)" in updated_param_map:
        t60_voltage_kv = round(rated_voltage_kv / 2, 2)
        _set_param(
            "T60(预备试验)",
            "试验电压",
            _format_voltage_value(t60_voltage_kv),
            value_source="rule",
            value_type="text",
            constraints=_format_voltage_value(t60_voltage_kv),
            calc_rule=f"额定电压 {rated_voltage_kv} kV 的50%为 {_format_voltage_value(t60_voltage_kv)}。",
            resolution_mode="graph_final",
        )
    short_break_ka = _extract_named_current_ka_local(
        query_text, ["额定短路开断电流", "短路开断电流"]
    )
    if short_break_ka is not None and "T60(预备试验)" in updated_param_map:
        t60_current_ka = round(short_break_ka * 0.6, 3)
        _set_param(
            "T60(预备试验)",
            "试验电流kA",
            f"{str(t60_current_ka).rstrip('0').rstrip('.')} kA",
            value_source="rule",
            value_type="text",
            constraints=f"{str(t60_current_ka).rstrip('0').rstrip('.')} kA",
            calc_rule=f"额定短路开断电流 {short_break_ka} kA 的60%为 {str(t60_current_ka).rstrip('0').rstrip('.')} kA。",
            resolution_mode="graph_final",
        )

    short_circuit_outputs = {
        "短路开断试验(T10)": 0.1,
        "短路开断试验(T30)": 0.3,
        "短路开断试验(T60)": 0.6,
        "短路开断试验(T100S)": 1.0,
    }
    for test_name, ratio in short_circuit_outputs.items():
        if rated_voltage_kv is not None:
            _set_if_present(
                test_name,
                "额定电压",
                _format_voltage_value(rated_voltage_kv),
                calc_rule="用户已明确提供额定电压，问答阶段直接采用该值。",
            )
            _set_if_present(
                test_name,
                "试验电压",
                _format_voltage_value(rated_voltage_kv),
                calc_rule=f"40.5kV及以下三相试验时，{test_name}试验电压取额定电压 {rated_voltage_kv} kV。",
            )
        if short_break_ka is not None:
            current_ka = round(short_break_ka * ratio, 3)
            _set_if_present(
                test_name,
                "试验电流kA",
                f"{str(current_ka).rstrip('0').rstrip('.')} kA",
                calc_rule=f"额定短路开断电流 {short_break_ka} kA 的{int(ratio * 100)}%为 {str(current_ka).rstrip('0').rstrip('.')} kA。"
                if ratio < 1
                else f"试验电流取额定短路开断电流 {str(short_break_ka).rstrip('0').rstrip('.')} kA。",
            )
        if rated_closing_ka is not None:
            _set_if_present(
                test_name,
                "关合电流",
                f"{str(rated_closing_ka).rstrip('0').rstrip('.')} kA",
                calc_rule="用户已明确提供额定短路关合电流，问答阶段直接采用该值。",
            )
        _set_if_present(test_name, "试验相数", "三相", calc_rule="40.5kV及以下默认三相。")
        _set_if_present(test_name, "额定频率", f"{str(rated_frequency_hz).rstrip('0').rstrip('.')} Hz", calc_rule="用户已明确提供额定频率，问答阶段直接采用该值。")

    if first_pole_kpp is not None:
        first_pole_kpp_text = str(first_pole_kpp).rstrip("0").rstrip(".")
        for test_name, param_names in tuple(updated_param_map.items()):
            if "首开极系数kpp" not in (param_names or []):
                continue
            if first_pole_kpp_from_user:
                _set_if_present(
                    test_name,
                    "首开极系数kpp",
                    first_pole_kpp_text,
                    calc_rule=first_pole_kpp_rule,
                )
            else:
                _set_if_value_missing(
                    test_name,
                    "首开极系数kpp",
                    first_pole_kpp_text,
                    calc_rule=first_pole_kpp_rule,
                )

    if capacitive_nonuniform_text:
        for test_name, param_names in tuple(updated_param_map.items()):
            if "不均匀系数" not in (param_names or []):
                continue
            if capacitive_nonuniform_from_user:
                _set_if_present(
                    test_name,
                    "不均匀系数",
                    capacitive_nonuniform_text,
                    calc_rule=capacitive_nonuniform_rule,
                )
            else:
                _set_if_value_missing(
                    test_name,
                    "不均匀系数",
                    capacitive_nonuniform_text,
                    calc_rule=capacitive_nonuniform_rule,
                )

    for test_name, param_names in tuple(updated_param_map.items()):
        if "断口数量" not in (param_names or []):
            continue
        _set_if_value_missing(
            test_name,
            "断口数量",
            str(global_break_count),
            calc_rule=global_break_count_rule,
        )

    for test_name, param_names in tuple(updated_param_map.items()):
        if "试验相数" not in (param_names or []):
            continue
        if global_test_phase_from_user:
            _set_if_present(
                test_name,
                "试验相数",
                global_test_phase_value,
                calc_rule=global_test_phase_rule,
            )
        else:
            _set_if_value_missing(
                test_name,
                "试验相数",
                global_test_phase_value,
                calc_rule=global_test_phase_rule,
            )

    for test_name, param_names in tuple(updated_param_map.items()):
        if "断路器等级" not in (param_names or []):
            continue
        if is_breaker_class_not_applicable:
            _set_if_present(
                test_name,
                "断路器等级",
                "--",
                calc_rule="额定电压高于 72.5 kV 时无断路器等级概念，最终输出固定为 --。",
            )
            continue
        if is_low_voltage:
            _set_if_value_missing(
                test_name,
                "断路器等级",
                "S1",
                calc_rule="低压试验未提供断路器等级时默认按 S1 处理。",
            )

    state_t10_parameters = _resolve_state_t10_parameters_local()
    for state_t10_test_name in ("状态检查试验(T10)", "状态检查试验(T10)"):
        if rated_voltage_kv is not None:
            _set_if_present(
                state_t10_test_name,
                "额定电压",
                _format_voltage_value(rated_voltage_kv),
                calc_rule="用户已明确提供额定电压，问答阶段直接采用该值。",
            )
        for param_name in ("试验次数", "试验相数", "试验电压", "试验电流", "试验电流kA"):
            resolved_value = state_t10_parameters.get(
                "试验电流" if param_name == "试验电流kA" else param_name
            )
            if not resolved_value:
                continue
            value_text, calc_rule = resolved_value
            _set_if_present(
                state_t10_test_name,
                param_name,
                value_text,
                calc_rule=calc_rule,
            )

    as_state_t10_parameters = _resolve_as_state_t10_parameters_local()
    for param_name in ("试验相数", "试验电压", "试验电流kA"):
        resolved_value = as_state_t10_parameters.get(param_name)
        if not resolved_value:
            continue
        value_text, calc_rule = resolved_value
        _set_if_present(
            "作为状态检查的T10试验",
            param_name,
            value_text,
            calc_rule=calc_rule,
        )
    if rated_voltage_kv is not None:
        _set_if_present(
            "作为状态检查的T10试验",
            "额定电压",
            _format_voltage_value(rated_voltage_kv),
            calc_rule="用户已明确提供额定电压，作为状态检查的T10试验直接采用该值。",
        )
    _set_if_present(
        "作为状态检查的T10试验",
        "试验次数",
        "1次",
        calc_rule="作为状态检查的T10试验按1次执行。",
    )
    _set_if_present(
        "作为状态检查的T10试验",
        "操作顺序",
        "O",
        calc_rule="作为状态检查的T10试验操作顺序固定为O。",
    )

    op2_parameters = _resolve_op2_parameters_local()
    for param_name in ("试验相数", "试验电压"):
        resolved_value = op2_parameters.get(param_name)
        if not resolved_value:
            continue
        value_text, calc_rule = resolved_value
        _set_if_present(
            "失步关合和开断试验(OP2)",
            param_name,
            value_text,
            calc_rule=calc_rule,
        )

    op1_parameters = _resolve_op1_parameters_local()
    for param_name in ("试验电压",):
        resolved_value = op1_parameters.get(param_name)
        if not resolved_value:
            continue
        value_text, calc_rule = resolved_value
        _set_if_present(
            "失步关合和开断试验(OP1)",
            param_name,
            value_text,
            calc_rule=calc_rule,
        )

    t100s_ab_parameters = _resolve_t100s_ab_parameters_local()
    for test_name, param_values in t100s_ab_parameters.items():
        for param_name, (value_text, calc_rule) in param_values.items():
            _set_if_present(
                test_name,
                param_name,
                value_text,
                calc_rule=calc_rule,
            )

    single_phase_ground_fault_parameters = _resolve_single_phase_ground_fault_parameters_local()
    for param_name, (value_text, calc_rule) in single_phase_ground_fault_parameters.items():
        _set_if_present(
            "单相接地故障试验",
            param_name,
            value_text,
            calc_rule=calc_rule,
        )

    op2_making_parameters = _resolve_op2_making_parameters_local()
    for param_name in ("试验电压",):
        resolved_value = op2_making_parameters.get(param_name)
        if not resolved_value:
            continue
        value_text, calc_rule = resolved_value
        _set_if_present(
            "OP2关合",
            param_name,
            value_text,
            calc_rule=calc_rule,
        )

    short_circuit_60hz_parameters = _resolve_60hz_short_circuit_parameters_local()
    for test_name, param_values in short_circuit_60hz_parameters.items():
        for param_name, (value_text, calc_rule) in param_values.items():
            if param_name == "试验电流":
                for current_param_name in ("试验电流", "试验电流kA"):
                    _set_if_present(
                        test_name,
                        current_param_name,
                        value_text,
                        calc_rule=calc_rule,
                    )
                continue
            _set_if_present(
                test_name,
                param_name,
                value_text,
                calc_rule=calc_rule,
            )

    t10_t30_t60_parameters = _resolve_t10_t30_t60_parameters_local()
    for test_name, param_values in t10_t30_t60_parameters.items():
        for param_name, (value_text, calc_rule) in param_values.items():
            if param_name == "试验电流":
                for current_param_name in ("试验电流", "试验电流kA"):
                    _set_if_present(
                        test_name,
                        current_param_name,
                        value_text,
                        calc_rule=calc_rule,
                    )
                continue
            _set_if_present(
                test_name,
                param_name,
                value_text,
                calc_rule=calc_rule,
            )

    if rated_voltage_kv is not None and normalized_stand_type != "IEC":
        for test_name in ("电寿命(单分)", "电寿命(合分)", "电寿命(循环)"):
            _set_if_present(
                test_name,
                "试验电压",
                _format_voltage_value(rated_voltage_kv),
                calc_rule=f"40.5kV及以下时，{test_name}试验电压取额定电压 {rated_voltage_kv} kV。",
            )
    if short_break_ka is not None and normalized_stand_type != "IEC":
        for test_name in ("电寿命(单分)", "电寿命(合分)", "电寿命(循环)"):
            _set_if_present(
                test_name,
                "试验电流kA",
                f"{str(short_break_ka).rstrip('0').rstrip('.')} kA",
                calc_rule="用户已明确提供额定短路开断电流，问答阶段直接采用该值。",
            )
    if rated_closing_ka is not None and normalized_stand_type != "IEC":
        for test_name in ("电寿命(单分)", "电寿命(合分)", "电寿命(循环)"):
            _set_if_present(
                test_name,
                "关合电流",
                f"{str(rated_closing_ka).rstrip('0').rstrip('.')} kA",
                calc_rule="用户已明确提供额定短路关合电流，问答阶段直接采用该值。",
            )

    for test_name in (
        "工频耐受电压试验",
        "工频耐受电压试验#干",
        "工频耐受电压试验#湿",
        "工频耐受电压试验(干)",
        "工频耐受电压试验(湿)",
        "工频耐受电压试验(断口)",
        "工频耐受电压试验(相间及对地)",
        "作为状态检查的工频耐受电压试验",
        "雷电冲击耐受电压试验",
        "雷电冲击耐受电压试验(联合电压)",
        "雷电冲击耐受电压试验(断口)",
        "雷电冲击耐受电压试验(相间及对地)",
        "局部放电试验",
    ):
        _set_if_present(
            test_name,
            "介质性质",
            dielectric_value,
            calc_rule="根据用户输入中的介质/气体信息判定；未命中充气/充油证据时默认为正常。",
        )

    for test_name in (
        "工频耐受电压试验",
        "工频耐受电压试验(干)",
        "工频耐受电压试验(湿)",
        "工频耐受电压试验(断口)",
        "工频耐受电压试验(相间及对地)",
        "作为状态检查的工频耐受电压试验",
    ):
        _set_if_present(test_name, "试验时间", "1min", calc_rule="工频耐受电压试验时间固定为1min。")

    _set_if_present(
        "作为状态检查的工频耐受电压试验",
        "正常次数",
        "1次",
        calc_rule="作为状态检查的工频耐受电压试验正常次数固定为1次。",
    )

    _set_if_value_missing(
        "工频耐受电压试验",
        "试验状态",
        "干",
        calc_rule="未命中户外干/湿拆分时，工频耐受电压试验默认按干态输出。",
    )
    _set_if_value_missing(
        "工频耐受电压试验(断口)",
        "试验状态",
        "干",
        calc_rule="断口工频耐受电压试验按干态输出。",
    )
    _set_if_value_missing(
        "工频耐受电压试验#干",
        "试验状态",
        "干",
        calc_rule="高压户外工频拆分后，内部干态分支按干态输出。",
    )
    _set_if_value_missing(
        "工频耐受电压试验#湿",
        "试验状态",
        "湿",
        calc_rule="高压户外工频拆分后，内部湿态分支按湿态输出。",
    )
    _set_if_value_missing(
        "工频耐受电压试验(干)",
        "试验状态",
        "干",
        calc_rule="户外状态拆分后，工频耐受电压试验(干)按干态输出。",
    )
    _set_if_value_missing(
        "工频耐受电压试验(湿)",
        "试验状态",
        "湿",
        calc_rule="户外状态拆分后，工频耐受电压试验(湿)按湿态输出。",
    )
    _set_if_value_missing(
        "工频耐受电压试验(相间及对地)",
        "试验状态",
        "干",
        calc_rule="未命中特殊拆分规则时，相间及对地工频耐受电压试验默认按干态输出。",
    )
    _set_if_value_missing(
        "工频耐受电压试验(联合电压)",
        "试验状态",
        "干",
        calc_rule="工频耐受联合电压试验固定按干态输出。",
    )
    _set_if_present(
        "雷电冲击耐受电压试验(联合电压)",
        "试验状态",
        "干",
        calc_rule="雷电冲击联合电压试验固定按干态输出。",
    )
    _set_if_present(
        "操作冲击耐受电压试验(联合电压)",
        "试验状态",
        "干",
        calc_rule="操作冲击联合电压试验固定按干态输出。",
    )

    _set_if_present(
        "工频耐受电压试验(联合电压)",
        "试验部位",
        "断口",
        calc_rule="工频联合电压试验的试验部位固定为断口。",
    )
    _set_if_present(
        "工频耐受电压试验#干",
        "试验部位",
        "相间及对地",
        calc_rule="高压户外工频拆分后，内部干态分支的试验部位固定为相间及对地。",
    )
    _set_if_present(
        "工频耐受电压试验#湿",
        "试验部位",
        "相间及对地",
        calc_rule="高压户外工频拆分后，内部湿态分支的试验部位固定为相间及对地。",
    )
    _set_if_present(
        "雷电冲击耐受电压试验(联合电压)",
        "试验部位",
        "断口",
        calc_rule="雷电联合电压试验的试验部位固定为断口。",
    )
    _set_if_present(
        "操作冲击耐受电压试验(联合电压)",
        "试验部位",
        "断口",
        calc_rule="操作联合电压试验的试验部位固定为断口。",
    )

    def _resolve_insulation_phase_local(
        query_text: str,
        rated_voltage: float | None,
    ) -> tuple[int, str, str]:
        normalized = str(query_text or "").replace("（", "(").replace("）", ")")
        if "三相共箱" in normalized:
            return 3, "三相", "问题中明确包含三相共箱，按三相试验计算。"
        if "三相分箱" in normalized:
            return 1, "单相", "问题中明确包含三相分箱，按单相试验计算。"

        phase_text, phase_rule = _detect_capacitive_test_phase_local(query_text, rated_voltage)
        phase_count = 3 if phase_text == "三相" else 1
        return phase_count, phase_text, phase_rule

    insulation_phase_count, insulation_phase_text, insulation_phase_rule = (
        _resolve_insulation_phase_local(query_text, rated_voltage_kv)
    )

    def _normalize_insulation_test_position_for_phase() -> None:
        if insulation_phase_text != "单相":
            return

        single_phase_ground_tests = (
            "工频耐受电压试验",
            "工频耐受电压试验#干",
            "工频耐受电压试验#湿",
            "工频耐受电压试验(干)",
            "工频耐受电压试验(湿)",
            "工频耐受电压试验(相间及对地)",
            "作为状态检查的工频耐受电压试验",
            "雷电冲击耐受电压试验",
            "雷电冲击耐受电压试验(相间及对地)",
            "操作冲击耐受电压试验",
            "操作冲击耐受电压试验(干)",
            "操作冲击耐受电压试验(湿)",
        )
        calc_rule = "高压绝缘试验在未明确三相时默认按单相处理；非断口试验部位统一按对地输出。"
        for test_name in single_phase_ground_tests:
            if test_name not in updated_param_map:
                continue
            if "试验部位" not in updated_param_map.get(test_name, []):
                continue
            current_value = str(
                updated_value_map.get(test_name, {}).get("试验部位", {}).get("value_text", "")
                or ""
            ).strip()
            normalized_current_value = current_value.replace("开关断口", "断口")
            if normalized_current_value == "断口":
                continue
            _set_param(
                test_name,
                "试验部位",
                "对地",
                value_source="rule",
                value_type="text",
                constraints="对地",
                calc_rule=calc_rule,
                resolution_mode="graph_final",
            )

    def _is_fracture_insulation_test(test_name: str) -> bool:
        return test_name in {
            "工频耐受电压试验(断口)",
            "工频耐受电压试验(联合电压)",
            "雷电冲击耐受电压试验(断口)",
            "雷电冲击耐受电压试验(联合电压)",
            "操作冲击耐受电压试验(联合电压)",
        }

    def _set_insulation_normal_count(
        test_names: tuple[str, ...],
        *,
        impulse_factor: int,
        family_label: str,
    ) -> None:
        for test_name in test_names:
            phase_multiplier = 2 if _is_fracture_insulation_test(test_name) else 3
            normal_count = phase_multiplier * insulation_phase_count * impulse_factor
            if impulse_factor == 1:
                count_rule = (
                    f"{family_label}{test_name}的正常次数按{phase_multiplier}*试验相数计算；"
                    f"当前试验相数为{insulation_phase_text}，因此正常次数为{normal_count}次。"
                )
            else:
                count_rule = (
                    f"{family_label}{test_name}的正常次数按{phase_multiplier}*试验相数*2*15计算；"
                    f"当前试验相数为{insulation_phase_text}，因此正常次数为{normal_count}次。"
                )
            _set_if_present(test_name, "试验相数", insulation_phase_text, calc_rule=insulation_phase_rule)
            _set_if_present(test_name, "正常次数", f"{normal_count}次", calc_rule=count_rule)

    _normalize_insulation_test_position_for_phase()

    _set_insulation_normal_count(
        (
            "工频耐受电压试验",
            "工频耐受电压试验#干",
            "工频耐受电压试验#湿",
            "工频耐受电压试验(干)",
            "工频耐受电压试验(湿)",
            "工频耐受电压试验(断口)",
            "工频耐受电压试验(相间及对地)",
            "工频耐受电压试验(联合电压)",
        ),
        impulse_factor=1,
        family_label="工频耐受电压试验族",
    )
    _set_insulation_normal_count(
        (
            "雷电冲击耐受电压试验",
            "雷电冲击耐受电压试验(断口)",
            "雷电冲击耐受电压试验(相间及对地)",
            "雷电冲击耐受电压试验(联合电压)",
        ),
        impulse_factor=30,
        family_label="雷电冲击耐受电压试验族",
    )
    _set_insulation_normal_count(
        (
            "操作冲击耐受电压试验",
            "操作冲击耐受电压试验(干)",
            "操作冲击耐受电压试验(湿)",
            "操作冲击耐受电压试验(联合电压)",
        ),
        impulse_factor=30,
        family_label="操作冲击耐受电压试验族",
    )

    def _recall_full_insulation_type_test_parameters() -> None:
        test_name = "全套绝缘型式试验"
        if test_name not in updated_param_map and test_name not in updated_value_map:
            return

        _ensure_test_from_config(test_name)
        for required_param_name in _get_config_required_params(test_name):
            _ensure_param(test_name, required_param_name)

        break_count, break_count_rule = _resolve_break_count_local(query_text)
        _set_if_value_missing(
            test_name,
            "断口数量",
            str(break_count),
            value_source="rule",
            calc_rule=break_count_rule,
        )

        inherited_rated_voltage = _copy_param_entry_if_missing(
            test_name,
            "额定电压",
            [
                ("工频耐受电压试验", ["额定电压"]),
                ("雷电冲击耐受电压试验", ["额定电压"]),
                ("操作冲击耐受电压试验", ["额定电压"]),
                ("工频耐受电压试验(联合电压)", ["额定电压"]),
                ("雷电冲击耐受电压试验(联合电压)", ["额定电压"]),
            ],
        )
        if not inherited_rated_voltage:
            if rated_voltage_kv is not None:
                rated_voltage_text = _format_voltage_value(rated_voltage_kv)
                _set_param(
                    test_name,
                    "额定电压",
                    rated_voltage_text,
                    value_source="user_input",
                    value_type="text",
                    constraints=rated_voltage_text,
                    calc_rule="用户已明确提供额定电压，问答阶段直接采用该值。",
                    derive_from_rated=rated_voltage_text,
                    resolution_mode="graph_final",
                )
            else:
                _set_entry_if_value_missing(
                    test_name,
                    "额定电压",
                    {
                        "value_text": "用户录入，根据GBT11022 5.2.2及5.2.3向上取数",
                        "value_source": "user_input",
                        "value_expr": "用户录入，根据GBT11022 5.2.2及5.2.3向上取数",
                        "unit": "kV",
                        "constraints": "用户录入，根据GBT11022 5.2.2及5.2.3向上取数",
                        "calc_rule": "",
                        "derive_from_rated": "用户录入，根据GBT11022 5.2.2及5.2.3向上取数",
                        "resolution_mode": "needs_user_input",
                    },
                )

        _set_if_value_missing(
            test_name,
            "开关相数",
            insulation_phase_text,
            value_source="rule",
            calc_rule=f"全套绝缘型式试验的开关相数沿用绝缘试验相数判定；{insulation_phase_rule}",
        )

        _set_entry_if_value_missing(
            test_name,
            "其他细则（充补套修）",
            {
                "value_text": "默认取值为：否否否; 充按照是否充气/充油判断，补（是否补气）默认否，套（指套管材质，陶瓷or复合）需用户输入，修（是否修正海拔系数）默认否，最后仅输出判断结果，不要有其他任何多余内容，取值案例：'是否瓷否'/'是否复否'",
                "value_source": "standard",
                "value_expr": "",
                "unit": "",
                "constraints": "默认取值为：否否否;充按照是否充气/充油判断，补（是否补气）默认否，套（指套管材质，陶瓷or复合）需用户输入，修（是否修正海拔系数）默认否，最后仅输出判断结果，不要有其他任何多余内容取值案例：'是否瓷否'/'是否复否'",
                "calc_rule": "充按照是否充气/充油判断，补（是否补气）默认否，套（指套管材质，陶瓷or复合）需用户输入，修（是否修正海拔系数）默认否，最后仅输出判断结果，不要有其他任何多余内容，取值案例：'是否瓷否'/'是否复否',当前问题未给出可完全展开的充补套修细则时，回落到图谱默认说明。",
                "derive_from_rated": "",
                "resolution_mode": "needs_condition",
            },
        )

        _set_entry_if_value_missing(
            test_name,
            "试验项目（全雷操工）",
            {
                "value_text": "是否否否",
                "value_source": "standard",
                "value_expr": "",
                "unit": "",
                "constraints": "是否否否",
                "calc_rule": "当前问题未提供更细的全雷操工判据时，回落到图谱默认值。回复案例:'是否否否'",
                "derive_from_rated": "",
                "resolution_mode": "needs_condition",
            },
        )
        _set_entry_if_value_missing(
            test_name,
            "正常次数",
            {
                "value_text": "1次",
                "value_source": "default",
                "value_expr": "1次",
                "unit": "count",
                "constraints": "默认1次",
                "calc_rule": "全套绝缘型式试验当前未命中更具体的次数规则时，按图谱默认试验次数1次回填到正常次数。",
                "derive_from_rated": "",
                "resolution_mode": "graph_final",
            },
        )

        inherited_altitude = _copy_param_entry_if_missing(
            test_name,
            "最大(适用)的海拔",
            [
                ("工频耐受电压试验", ["最大(适用)的海拔", "最大(适用)的海拔"]),
                ("雷电冲击耐受电压试验", ["最大(适用)的海拔", "最大(适用)的海拔"]),
                ("操作冲击耐受电压试验", ["最大(适用)的海拔", "最大(适用)的海拔"]),
                ("工频耐受电压试验(联合电压)", ["最大(适用)的海拔", "最大(适用)的海拔"]),
                ("雷电冲击耐受电压试验(联合电压)", ["最大(适用)的海拔", "最大(适用)的海拔"]),
                ("操作冲击耐受电压试验(联合电压)", ["最大(适用)的海拔", "最大(适用)的海拔"]),
            ],
        )
        if not inherited_altitude:
            _set_entry_if_value_missing(
                test_name,
                "最大(适用)的海拔",
                {
                    "value_text": "用户输入，未输入默认1000m",
                    "value_source": "user_input",
                    "value_expr": "用户输入，未输入默认1000m",
                    "unit": "m",
                    "constraints": "用户输入，未输入默认1000m",
                    "calc_rule": "",
                    "derive_from_rated": "用户输入，未输入默认1000m",
                    "resolution_mode": "needs_user_input",
                },
            )

    _recall_full_insulation_type_test_parameters()

    if "空载特性测量" in updated_param_map and "空载特性测量#1" not in updated_param_map:
        source_params = list(updated_param_map.get("空载特性测量", []) or [])
        source_values = deepcopy(updated_value_map.get("空载特性测量", {}) or {})
        if source_params:
            updated_param_map.pop("空载特性测量", None)
            updated_value_map.pop("空载特性测量", None)
            for instance_name in ("空载特性测量#1", "空载特性测量#2"):
                updated_param_map[instance_name] = list(source_params)
                updated_value_map[instance_name] = deepcopy(source_values)

    # IEC electrical life: fallback split, test current assignment and reduction-note logic
    # Only run IEC fallback when normalized standard is IEC and the query/report scope
    # indicates a short-circuit type test (短路性能型式试验).
    if normalized_stand_type == "IEC":
        iec_el_configs = [
            {
                "base_name": "电寿命试验(100%)",
                "suffixes": [("#O-CO-CO", "O-0.3s-CO-180s-CO", "2次")],
                "short_circuit_checks": ["短路开断试验(T100s)", "短路开断试验(T100S)", "T100S(60Hz)", "T100s(60Hz)"],
                "current_ratio": 1.0,
                "reduction_note": "已做过短路开断试验(T100s)或T100s(60Hz)，可减一次。",
            },
            {
                "base_name": "电寿命试验(60%)",
                "suffixes": [
                    ("#O", "O", "2次"),
                    ("#O-CO-CO", "O-0.3s-CO-180s-CO", "2次"),
                ],
                "short_circuit_checks": ["短路开断试验(T60)", "T60(60Hz)"],
                "current_ratio": 0.6,
                "reduction_note": "已做过短路开断试验(T60)或T60(60Hz)，可减一次。",
            },
            {
                "base_name": "电寿命试验(30%)",
                "suffixes": [
                    ("#O", "O", "84次"),
                    ("#O-CO", "O-0.3s-CO", "14次"),
                    ("#O-CO-CO", "O-0.3s-CO-180s-CO", "6次"),
                ],
                "short_circuit_checks": ["短路开断试验(T30)", "T30(60Hz)"],
                "current_ratio": 0.3,
                "reduction_note": "已做过短路开断试验(T30)或T30(60Hz)，可减一次。",
            },
            {
                "base_name": "电寿命试验(10%)",
                "suffixes": [
                    ("#O", "O", "84次"),
                    ("#O-CO", "O-0.3s-CO", "14次"),
                    ("#O-CO-CO", "O-0.3s-CO-180s-CO", "6次"),
                ],
                "short_circuit_checks": ["短路开断试验(T10)", "T10(60Hz)"],
                "current_ratio": 0.1,
                "reduction_note": "已做过短路开断试验(T10)或T10(60Hz)，可减一次。",
            },
        ]

        # Shared activation gate: Ur <= 40.5 kV and breaker class != E1
        iec_el_enabled = bool(
            rated_voltage_kv is not None
            and rated_voltage_kv <= 40.5
            and not re.search(
                r"断路器等级\s*(?:(?:[:：=]|为|是)\s*)?E1(?:级)?",
                query_text,
                flags=re.IGNORECASE,
            )
        )
        # Require explicit short-circuit report scope or token in query before acting.
        current_report_scopes = set(_extract_current_report_scopes(query_text, schema_cfg))
        short_circuit_scope_present = (
            "短路性能型式试验" in current_report_scopes or "短路性能型式试验" in query_text
        )

        # Fallback split: if percentage-based items exist but were not split by domain rules,
        # split them programmatically when the shared gate is satisfied and the query
        # indicates a short-circuit type test.
        if iec_el_enabled and short_circuit_scope_present:
            for cfg in iec_el_configs:
                base_name = cfg["base_name"]
                if base_name not in updated_param_map:
                    continue
                already_split = any(
                    str(k).startswith(base_name + "#")
                    for k in updated_param_map.keys()
                )
                if already_split:
                    continue
                source_params = list(updated_param_map.get(base_name, []) or [])
                source_values = deepcopy(updated_value_map.get(base_name, {}) or {})
                if not source_params:
                    continue
                updated_param_map.pop(base_name, None)
                updated_value_map.pop(base_name, None)
                for suffix, op_seq, count_text in cfg["suffixes"]:
                    child_name = base_name + suffix
                    updated_param_map[child_name] = list(source_params)
                    updated_value_map[child_name] = deepcopy(source_values)
                    _set_param(
                        child_name,
                        "操作顺序",
                        op_seq,
                        value_source="rule",
                        value_type="text",
                        constraints=op_seq,
                        calc_rule=f"IEC电寿命试验操作顺序为 {op_seq}。",
                        resolution_mode="graph_final",
                    )
                    _set_param(
                        child_name,
                        "试验次数",
                        count_text,
                        value_source="rule",
                        value_type="text",
                        constraints=count_text,
                        calc_rule=f"IEC电寿命试验次数为 {count_text}。",
                        resolution_mode="graph_final",
                    )

        # Apply test currents and reduction notes to all IEC electrical-life children
        # Only apply when short-circuit scope is present (same gating as the split above)
        if short_circuit_scope_present:
            for cfg in iec_el_configs:
                for suffix, op_seq, count_text in cfg["suffixes"]:
                    test_name = cfg["base_name"] + suffix
                    if test_name not in updated_param_map:
                        continue
                    # Set test current based on rated short-circuit breaking current
                    if short_break_ka is not None:
                        current_ka = round(short_break_ka * cfg["current_ratio"], 3)
                        current_text = f"{str(current_ka).rstrip('0').rstrip('.')} kA"
                        _set_param(
                            test_name,
                            "试验电流kA",
                            current_text,
                            value_source="rule",
                            value_type="text",
                            constraints=current_text,
                            calc_rule=f"额定短路开断电流 {short_break_ka} kA 的{int(cfg['current_ratio'] * 100)}%为 {current_text}。",
                            resolution_mode="graph_final",
                        )
                    # Reduction-once detection: check if corresponding short-circuit test exists
                    has_reduction = any(
                        sc_test in updated_param_map or sc_test in query_text
                        for sc_test in cfg["short_circuit_checks"]
                    )
                    if has_reduction:
                        tc_entry = updated_value_map.get(test_name, {}).get("试验次数")
                        if isinstance(tc_entry, dict):
                            existing_calc = str(tc_entry.get("calc_rule", "") or "").strip()
                            note = cfg["reduction_note"]
                            new_calc = f"{existing_calc} {note}" if existing_calc else note
                            tc_entry["calc_rule"] = new_calc

    return updated_param_map, updated_value_map


def _build_resolved_rule_overrides(
    domain_rule_decisions: dict[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}

    def _format_voltage_value(value: float) -> str:
        if float(value).is_integer():
            return f"{int(value)} kV"
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{text} kV"

    def _resolve_partial_discharge_parameters(
        rated_voltage_kv: float | None,
        pf_withstand_kv: float | None,
        first_pole_kpp: float | None,
    ) -> dict[str, str]:
        overrides: dict[str, str] = {}
        if rated_voltage_kv is None:
            return overrides
        if rated_voltage_kv <= 40.5:
            overrides["局部放电值"] = "≤10 pC"
            overrides["预加电压"] = _format_voltage_value(round(rated_voltage_kv * 1.3, 2))
            overrides["交流电压"] = _format_voltage_value(round(rated_voltage_kv * 1.1, 2))
            return overrides

        overrides["局部放电值"] = "≤5 pC"
        if pf_withstand_kv is not None:
            overrides["预加电压"] = _format_voltage_value(pf_withstand_kv)
        if first_pole_kpp is not None:
            if abs(first_pole_kpp - 1.5) < 1e-6:
                overrides["交流电压"] = _format_voltage_value(round(rated_voltage_kv * 1.2, 2))
            elif abs(first_pole_kpp - 1.3) < 1e-6:
                overrides["交流电压"] = _format_voltage_value(round((rated_voltage_kv * 1.2) / math.sqrt(3.0), 2))
        return overrides

    for decision in domain_rule_decisions.values():
        if not isinstance(decision, dict):
            continue
        rule_kind = str(decision.get("kind", "") or "")
        test_item = str(decision.get("test_item", "") or "")
        if not test_item:
            continue

        if rule_kind == "applicability":
            resolved.setdefault(test_item, {})
            resolved[test_item]["applicability"] = {
                "decision": decision.get("decision"),
                "reason_text": decision.get("reason_text", ""),
            }
            continue

        if rule_kind == "count":
            resolved.setdefault(test_item, {})
            resolved[test_item]["parameter_overrides"] = resolved[test_item].get(
                "parameter_overrides", {}
            )
            rule_inputs = decision.get("inputs", {}) if isinstance(decision, dict) else {}
            rated_voltage_kv = (
                float(rule_inputs.get("rated_voltage_kv"))
                if isinstance(rule_inputs, dict)
                and rule_inputs.get("rated_voltage_kv") is not None
                else None
            )
            pf_withstand_kv = (
                float(rule_inputs.get("pf_withstand_kv"))
                if isinstance(rule_inputs, dict)
                and rule_inputs.get("pf_withstand_kv") is not None
                else None
            )
            first_pole_kpp = (
                float(rule_inputs.get("first_pole_kpp"))
                if isinstance(rule_inputs, dict)
                and rule_inputs.get("first_pole_kpp") is not None
                else None
            )
            if test_item == "局部放电试验" and rated_voltage_kv is not None:
                resolved[test_item]["parameter_overrides"].update(
                    _resolve_partial_discharge_parameters(
                        rated_voltage_kv,
                        pf_withstand_kv,
                        first_pole_kpp,
                    )
                )
                resolved[test_item]["parameter_overrides"]["预加时间"] = (
                    "30s" if rated_voltage_kv == 40.5 else "60s"
                )
                resolved[test_item]["parameter_overrides"]["测量时间(min)"] = "1min"
            resolved[test_item]["parameter_overrides"]["试验次数"] = decision.get(
                "decision"
            )
            resolved[test_item]["count_reason"] = decision.get("reason_text", "")
            continue

        if rule_kind == "pair_merge":
            secondary_test_item = str(decision.get("secondary_test_item", "") or "").strip()
            if decision.get("enabled"):
                resolved[test_item] = {
                    "decision": "merge",
                    "secondary_test_item": secondary_test_item,
                    "merged_output": decision.get("merged_output", {}),
                    "reason_text": decision.get("reason_text", ""),
                }
            else:
                resolved[test_item] = {
                    "decision": "separate",
                    "secondary_test_item": secondary_test_item,
                    "reason_text": decision.get("reason_text", ""),
                }
            continue

        if rule_kind == "split":
            if decision.get("enabled"):
                remove_original = bool(decision.get("remove_original", True))
                resolved[test_item] = {
                    "decision": "split",
                    "remove_original": remove_original,
                    "outputs": decision.get("split_output", []),
                    "reason_text": decision.get("reason_text", ""),
                }
            else:
                resolved[test_item] = {
                    "decision": "single",
                    "single_output": decision.get("single_output", {}),
                    "reason_text": decision.get("reason_text", ""),
                }
            continue

    return resolved


def _build_final_test_item_scope(
        project_param_map: dict[str, list[str]],
        domain_rule_decisions: dict[str, Any],
) -> tuple[list[str], list[str]]:
    def _normalize_test_item_scope_key(value: str) -> str:
        text = _normalize_text_key(str(value))
        return text.replace("（", "(").replace("）", ")").lower()

    allowed_item_lookup = {
        _normalize_test_item_scope_key(str(item)): str(item)
        for item in project_param_map.keys()
        if _normalize_test_item_scope_key(str(item))
    }

    def _resolve_scope_item_name(test_name: str) -> str:
        normalized_name = _normalize_test_item_scope_key(test_name)
        if not normalized_name:
            return str(test_name).strip()
        return allowed_item_lookup.get(normalized_name, str(test_name).strip())

    allowed_items = list(project_param_map.keys())
    removed_items: list[str] = []
    hard_removed_items: set[str] = set()

    for decision in domain_rule_decisions.values():
        if not isinstance(decision, dict):
            continue
        test_item = _resolve_scope_item_name(str(decision.get("test_item", "") or "").strip())
        if not test_item:
            continue
        rule_kind = str(decision.get("kind", "") or "").strip()
        if rule_kind == "applicability" and not decision.get("enabled"):
            removed_items.append(test_item)
            hard_removed_items.add(test_item)
        if rule_kind == "pair_merge" and decision.get("enabled"):
            removed_items.append(test_item)
            hard_removed_items.add(test_item)
            secondary_test_item = _resolve_scope_item_name(
                str(decision.get("secondary_test_item", "") or "").strip()
            )
            if secondary_test_item:
                removed_items.append(secondary_test_item)
                hard_removed_items.add(secondary_test_item)
        if rule_kind == "pair_merge" and not decision.get("enabled"):
            merged_output = decision.get("merged_output", {}) or {}
            merged_test_item = _resolve_scope_item_name(
                str(merged_output.get("test_item", "") or "").strip()
            )
            if merged_test_item:
                removed_items.append(merged_test_item)
                hard_removed_items.add(merged_test_item)
        if rule_kind == "split" and decision.get("enabled"):
            split_outputs = decision.get("split_output", []) or []
            original_reused_as_split_output = any(
                isinstance(split_output, dict)
                and _resolve_scope_item_name(str(split_output.get("test_item", "") or "").strip()) == test_item
                for split_output in split_outputs
            )
            if bool(decision.get("remove_original", True)) and not original_reused_as_split_output:
                removed_items.append(test_item)
                hard_removed_items.add(test_item)
    # Guard against stale project_param_map entries leaking into the final whitelist.
    # If runtime rule decisions explicitly removed an item, it must never remain allowed.
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
    if (
        "短路开断试验(T60)" in allowed_deduped
        and "T60(预备试验)" in allowed_deduped
    ):
        allowed_deduped = [item for item in allowed_deduped if item != "T60(预备试验)"]
        if "T60(预备试验)" not in removed_deduped:
            removed_deduped.append("T60(预备试验)")
    return allowed_deduped, removed_deduped


def _build_test_item_display_map(project_param_map: dict[str, list[str]]) -> dict[str, str]:
    display_map: dict[str, str] = {}

    def _format_t100s_split_display_name(name: str) -> str | None:
        match = re.match(r"^(T100s\([ab]\)(?:\(60Hz\))?)#.+$", name)
        if match:
            return match.group(1)
        return None

    for test_name in project_param_map.keys():
        name = str(test_name or "").strip()
        if not name:
            continue
        t100s_display_name = _format_t100s_split_display_name(name)
        if t100s_display_name:
            display_map[name] = t100s_display_name
            continue
        if name.endswith("#干"):
            display_map[name] = f"{name[:-2]}(干)"
        elif name.endswith("#湿"):
            display_map[name] = f"{name[:-2]}(湿)"
        elif "#" in name:
            display_map[name] = name.split("#", 1)[0]
        else:
            display_map[name] = name
    return display_map


def _filter_context_by_final_test_item_scope(
    entities_context: list[dict],
    relations_context: list[dict],
    removed_test_items: list[str],
) -> tuple[list[dict], list[dict]]:
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


def _should_bypass_query_cache(global_config: dict[str, Any] | None) -> bool:
    cfg = global_config or {}
    if cfg.get("kg_schema_mode") == "electrical_controlled":
        return True
    addon_params = cfg.get("addon_params", {}) or {}
    schema_cfg = addon_params.get("electrical_schema", {}) or {}
    return bool(schema_cfg)


def _get_display_param_suppressions() -> dict[str, set[str]]:
    # return {}
    suppressions = {
        "回路电阻测量": {"回路电阻", "辅助和控制设备的电阻","回路电阻(μΩ)"},
        "工频耐受电压试验":{"SF6气体的最低功能压力(20℃表压)","放电次数","最大适用海拔"} ,
        "工频耐受电压试验(干)":{"SF6气体的最低功能压力(20℃表压)","放电次数","最大适用海拔"} ,
        "工频耐受电压试验(湿)":{"SF6气体的最低功能压力(20℃表压)","放电次数","最大适用海拔"} ,
        "工频耐受电压试验#干":{"SF6气体的最低功能压力(20℃表压)","放电次数","最大适用海拔"} ,
        "工频耐受电压试验#湿":{"SF6气体的最低功能压力(20℃表压)","放电次数","最大适用海拔"} ,
        "工频耐受电压试验(断口)":{"SF6气体的最低功能压力(20℃表压)","放电次数","最大适用海拔"} ,
        "工频耐受电压试验(相间及对地)":{"SF6气体的最低功能压力(20℃表压)","放电次数","最大适用海拔"} ,
        "工频耐受电压试验(联合电压)":{"SF6气体的最低功能压力(20℃表压)","放电次数","最大适用海拔"} ,
        "雷电冲击耐受电压试验":{"SF6气体的最低功能压力(20℃表压)","额定直流电压(±)","放电次数","最大适用海拔"},
        "雷电冲击耐受电压试验(断口)":{"SF6气体的最低功能压力(20℃表压)","放电次数","额定直流电压(±)","最大适用海拔"},
        "雷电冲击耐受电压试验(相间及对地)": {"SF6气体的最低功能压力(20℃表压)","放电次数","额定直流电压(±)","最大适用海拔"},
        "雷电冲击耐受电压试验(联合电压)":{"SF6气体的最低功能压力(20℃表压)","放电次数","额定直流电压(±)","最大适用海拔"},
        "操作冲击耐受电压试验(联合电压)":{"试验电压","SF6气体的最低功能压力(20℃表压)","放电次数","额定直流电压(±)","最大适用海拔"},
        "操作冲击耐受电压试验":{"试验电压","SF6气体的最低功能压力(20℃表压)","放电次数","额定直流电压(±)","最大适用海拔"},
        "操作冲击耐受电压试验(干)":{"SF6气体的最低功能压力(20℃表压)","放电次数","额定直流电压(±)","最大适用海拔"},
        "操作冲击耐受电压试验(湿)":{"SF6气体的最低功能压力(20℃表压)","放电次数","额定直流电压(±)","最大适用海拔"},
        "局部放电试验":{"辅助和控制设备的电阻","SF6气体的最低功能压力(20℃表压)","回路电阻(μΩ)","回路电阻"},
        "T60(预备试验)":{"SF6气体的最低功能压力(20℃表压)","外壳是否带电"},
        "作为状态检查的工频耐受电压试验":{"SF6气体的最低功能压力(20℃表压)","放电次数"},
        "短时耐受电流和峰值耐受电流试验":{"回路电阻","回路电阻(μΩ)"},
        "电寿命(合分)":{"SF6气体的最低功能压力(20℃表压)","外壳是否带电","金短时间","故障类型"},
        "电寿命(单分)":{"SF6气体的最低功能压力(20℃表压)","外壳是否带电","金短时间","故障类型"},
        "电寿命(循环)":{"SF6气体的最低功能压力(20℃表压)","外壳是否带电","金短时间","故障类型"},
        "温升试验":{"SF6气体的最低功能压力(20℃表压)","是否所配元件","材料绝热等级"},
        "温升试验(60Hz)":{"SF6气体的最低功能压力(20℃表压)","是否所配元件","材料绝热等级"},
        "近区故障试验(L75)":{"线路侧波阻抗","SF6气体的最低功能压力(20℃表压)","外壳是否带电"},
        "近区故障试验(L90)":{"线路侧波阻抗","SF6气体的最低功能压力(20℃表压)","外壳是否带电"},
        "辅助和控制回路温升试验": {"机构是否带合分闸线圈","辅助设备和控制设备的额定电源电压","辅助设备和控制设备的额定电流","材料绝热等级"},
        "全套绝缘型式试验":{"最大适用海拔","试验项目(全雷操工)"},
        "容性电流开断试验(BC1)": {"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电","关合涌流","关合涌流的频率",},
        "容性电流开断试验(BC2)": {"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电",},
        "容性电流开断试验(LC1)": {"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电",},
        "容性电流开断试验(LC2)": {"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电",},
        "容性电流开断试验(LC2)#1": {"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电",},
        "容性电流开断试验(LC2)#2": {"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电",},
        "容性电流开断试验(CC1)": {"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电",},
        "容性电流开断试验(CC2)": {"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电",},
        "容性电流开断试验(CC2)#1": {"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电",},
        "容性电流开断试验(CC2)#2": {"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电",},
        "T100s(a)":{"金短时间","外壳是否带电","SF6气体的最低功能压力(20℃表压)","额定频率"},
        "T100s(b)":{"金短时间","外壳是否带电","SF6气体的最低功能压力(20℃表压)","额定频率","结构特征"},
        "T100s":{"SF6气体的最低功能压力(20℃表压)"},
        "电寿命试验":{"SF6气体的最低功能压力(20℃表压)","外壳是否带电","金短时间","故障类型","SF6气体的额定压力(20℃表压)"},
        "电寿命(单分)":{"SF6气体的最低功能压力(20℃表压)","外壳是否带电","金短时间","故障类型","SF6气体的额定压力(20℃表压)"},
        "电寿命(合分)":{"SF6气体的最低功能压力(20℃表压)","外壳是否带电","金短时间","故障类型","SF6气体的额定压力(20℃表压)"},
        "电寿命(循环)":{"SF6气体的最低功能压力(20℃表压)","外壳是否带电","金短时间","故障类型","SF6气体的额定压力(20℃表压)"},
        "OP2关合":{"试验项数","SF6气体的最低功能压力(20℃表压)","外壳是否带电","直流分量(试验)","额定频率","发电机额定容量","故障类型"},
        "短时耐受电流试验": {"回路电阻","回路电阻(μΩ)","峰值电流kA"},
        "峰值耐受电流试验": {"回路电阻","回路电阻(μΩ)"},
        "状态检查试验(T10)":{"外壳是否带电"},
        "作为状态检查的雷电冲击耐受电压试验":{"SF6气体的最低功能压力(20℃表压)","放电次数"},
        "L60":{"线路侧波阻抗","外壳是否带电"},
        "T100s(60Hz)":{"SF6气体的最低功能压力(20℃表压)","外壳是否带电","故障类型"},
        "T60(60Hz)":{"SF6气体的最低功能压力(20℃表压)","外壳是否带电"},
        "L90(60Hz)":{"线路侧波阻抗","SF6气体的最低功能压力(20℃表压)","外壳是否带电"},
        "L75(60Hz)":{"线路侧波阻抗","SF6气体的最低功能压力(20℃表压)","外壳是否带电"},
        "L60(60Hz)":{"线路侧波阻抗","SF6气体的最低功能压力(20℃表压)","外壳是否带电"},
        "T100A(60Hz)":{"SF6气体的最低功能压力(20℃表压)","外壳是否带电","故障类型"},
        "CC1(60Hz)":{"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电"},
        "CC2(60Hz)":{"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电"},
        "BC1(60Hz)":{"关合涌流","关合涌流的频率","SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电"},
        "BC2(60Hz)":{"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电"},
        "LC1(60Hz)":{"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电"},
        "LC2(60Hz)":{"SF6气体的最低功能压力(20℃表压)","SF6气体的额定压力(20℃表压)","外壳是否带电"},
        "T100s(a)(60Hz)":{"外壳是否带电","SF6气体的最低功能压力(20℃表压)"},
        "T100s(b)(60Hz)": {"金短时间","外壳是否带电","SF6气体的最低功能压力(20℃表压)"},
        "短路开断试验(T10)": {"外壳是否带电", "失败次数","SF6气体的最低功能压力(20℃表压)"},
        "短路开断试验(T30)": {"外壳是否带电", "失败次数","SF6气体的最低功能压力(20℃表压)","额定频率"},
        "短路开断试验(T60)": {"外壳是否带电", "失败次数","SF6气体的最低功能压力(20℃表压)","额定频率"},
        "短路开断试验(T100S)": {"外壳是否带电","SF6气体的最低功能压力(20℃表压)","故障类型","失败次数"},
        "短路开断试验(T100A)": {"外壳是否带电","SF6气体的最低功能压力(20℃表压)","故障类型","失败次数"},
        "异相接地故障试验": {"SF6气体的最低功能压力(20℃表压)","额定频率"},
        "单相接地故障试验": {"SF6气体的最低功能压力(20℃表压)","额定频率"},
        "近区故障试验(L60)": {"外壳是否带电"},
        "失步关合和开断试验(OP1)": {"外壳是否带电","SF6气体的最低功能压力(20℃表压)","故障类型","发电机额定容量"},
        "失步关合和开断试验(OP2)": {"额定频率","试验项数","外壳是否带电","SF6气体的最低功能压力(20℃表压)","故障类型","直流分量(试验)","发电机额定容量"},
        "控制和辅助回路的绝缘试验": {"额定直流电压(±)"},
    }

    internal_aliases = {
        "T100s(a)#1": "T100s(a)",
        "T100s(a)#2": "T100s(a)",
        "T100s(a)#共箱共机构": "T100s(a)",
        "T100s(b)#共机构": "T100s(b)",
        "T100s(b)#共箱共机构": "T100s(b)",
        "T100s(a)(60Hz)#1": "T100s(a)(60Hz)",
        "T100s(a)(60Hz)#2": "T100s(a)(60Hz)",
        "T100s(a)(60Hz)#共箱共机构": "T100s(a)(60Hz)",
        "T100s(b)(60Hz)#共机构": "T100s(b)(60Hz)",
        "T100s(b)(60Hz)#共箱共机构": "T100s(b)(60Hz)",
        "短路开断试验(T100A)#共箱共机构": "短路开断试验(T100A)",
    }
    for internal_name, display_name in internal_aliases.items():
        hidden_params = suppressions.get(display_name)
        if hidden_params:
            suppressions[internal_name] = set(hidden_params)
    return suppressions


def _get_report_scope_test_whitelist(stand_type: str | None = None) -> dict[str, set[str]]:
    normalized = _normalize_operate_standard_type(stand_type)
    insulation_tests = {
            "工频耐受电压试验",
            "工频耐受电压试验(断口)",
            "工频耐受电压试验(相间及对地)",
            "工频耐受电压试验(干)",
            "工频耐受电压试验(湿)",
            "雷电冲击耐受电压试验",
            "雷电冲击耐受电压试验(断口)",
            "雷电冲击耐受电压试验(相间及对地)",
            "控制和辅助回路的绝缘试验",
            "操作冲击耐受电压试验",
            "局部放电试验",
            "全套绝缘型式试验",
    }
    switching_tests = {
            "容性电流开断试验(LC1)",
            "容性电流开断试验(LC2)",
            "容性电流开断试验(CC1)",
            "容性电流开断试验(CC2)",
            "容性电流开断试验(BC1)",
            "容性电流开断试验(BC2)",
            "T60(预备试验)",
            "状态检查试验(T10)",
            "作为状态检查的工频耐受电压试验",
            "空载特性测量",
            "空载特性测量#1",
            "空载特性测量#2",
            "BC1(60Hz)",
            "BC2(60Hz)",
            "CC1(60Hz)",
            "CC2(60Hz)",
            "LC1(60Hz)",
            "LC2(60Hz)",
            "作为状态检查的T10试验"
        }
    short_tests = {
            "短时耐受电流试验",
            "峰值耐受电流试验",
            "短时耐受电流和峰值耐受电流试验",
            "空载特性测量",
            "空载特性测量#1",
            "空载特性测量#2",
            "短路开断试验(T100S)",
            "短路开断试验(T10)",
            "失步关合和开断试验(OP1)",
            "失步关合和开断试验(OP2)",
            "电寿命试验",
            "电寿命(单分)",
            "电寿命(合分)",
            "电寿命(循环)",
            "作为状态检查的工频耐受电压试验",
            "状态检查试验(T10)",
            "单相接地故障试验",
            "异相接地故障试验",
            "短路开断试验(T30)",
            "短路开断试验(T60)",
            "T60(60Hz)",
            "T100S(60Hz)",
            "短路开断试验(T100A)",
            "T100A(60Hz)",
            "近区故障试验(L90)",
            "近区故障试验(L75)",
            "L75(60Hz)",
            "L90(60Hz)",
            "OP2关合",
            "T100s(a)",
            "T100s(b)",
            "T100s(a)(60Hz)",
            "T100s(b)(60Hz)",
            "T100s(三相共机构的验证试验)",
            "作为状态检查的T10试验"
        }
    # IEC DLT 标准不包含局部放电试验
    # NOTE: set.discard() and set.add() are in-place operations returning None,
    # so do NOT reassign the variable.
    if normalized == "IEC":
        insulation_tests.discard("局部放电试验")
        short_tests.update({
            "电寿命试验(100%)",
            "电寿命试验(100%)#O-CO-CO",
            "电寿命试验(60%)",
            "电寿命试验(60%)#O",
            "电寿命试验(60%)#O-CO-CO",
            "电寿命试验(30%)",
            "电寿命试验(30%)#O",
            "电寿命试验(30%)#O-CO",
            "电寿命试验(30%)#O-CO-CO",
            "电寿命试验(10%)",
            "电寿命试验(10%)#O",
            "电寿命试验(10%)#O-CO",
            "电寿命试验(10%)#O-CO-CO",
        })
    elif normalized == "DLT":
        insulation_tests.discard("局部放电试验")
        short_tests.add("作为状态检查的雷电冲击耐受电压试验")
        switching_tests.add("作为状态检查的雷电冲击耐受电压试验")
    return {
        "绝缘性能型式试验": insulation_tests,
        "温升性能型式试验": {
            "回路电阻测量",
            "辅助和控制回路温升试验",
            "温升试验",
            "温升试验(60Hz)",
        },
        "开合性能型式试验": switching_tests,
        "短路性能型式试验": short_tests,
    }


def _extract_current_report_scopes(query_text: str, schema_cfg: dict[str, Any] | None = None) -> list[str]:
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

    # Handle compound Chinese expressions like "绝缘性能及温升性能型式试验"
    # which means both "绝缘性能型式试验" and "温升性能型式试验".
    # The simple substring match misses "绝缘性能型式试验" because "及温升性能"
    # is inserted between "绝缘性能" and "型式试验".
    unmatched = [r for r in configured_reports if r not in matched]
    if len(configured_reports) >= 2 and unmatched:
        # Find the longest suffix shared by ALL configured report types
        min_rpt_len = min(len(r) for r in configured_reports)
        max_shared_suffix = ""
        for i in range(1, min_rpt_len):
            s = configured_reports[0][-i:]
            if all(r.endswith(s) for r in configured_reports):
                max_shared_suffix = s
            else:
                break

        if len(max_shared_suffix) >= 2:
            # Try suffix lengths from short to long; stop at first that produces matches
            for suffix_len in range(2, len(max_shared_suffix) + 1):
                candidate_suffix = configured_reports[0][-suffix_len:]
                if candidate_suffix not in text:
                    continue

                prefix_map: dict[str, str] = {}
                for rn in configured_reports:
                    p = rn[:-suffix_len]
                    if p:
                        prefix_map[p] = rn

                if len(prefix_map) < 2:
                    continue

                newly_matched: list[str] = []
                for prefix, report_name in prefix_map.items():
                    if report_name in matched:
                        continue
                    other_alts = "|".join(
                        re.escape(p) for p in prefix_map if p != prefix
                    )
                    if not other_alts:
                        continue
                    # Pattern: this prefix, then (connector + another known prefix)*,
                    # then optional connector, then the shared suffix.
                    pat = (
                            re.escape(prefix)
                            + r"(?:[及和、,，\s]+(?:" + other_alts + r"))*"
                            + r"[及和、,，\s]*"
                            + re.escape(candidate_suffix)
                    )
                    if re.search(pat, text):
                        newly_matched.append(report_name)

                if newly_matched:
                    matched.extend(newly_matched)
                    break

    return matched


def _merge_keywords_with_report_scope_fallback(
        query_text: str,
        hl_keywords: list[str] | None,
        ll_keywords: list[str] | None,
        schema_cfg: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    cfg = schema_cfg or {}
    current_report_scopes = _extract_current_report_scopes(query_text, cfg)
    merged_hl = _dedupe_preserve_order(list(hl_keywords or []))
    merged_ll = _dedupe_preserve_order(list(ll_keywords or []))
    if not current_report_scopes:
        return merged_hl, merged_ll

    is_multi_scope = len(current_report_scopes) > 1
    default_hl_map = cfg.get("report_scope_fallback_high_level_keywords", {}) or {}
    default_ll_map = cfg.get("report_scope_fallback_low_level_keywords", {}) or {}
    multi_hl_map = cfg.get("report_scope_multi_fallback_high_level_keywords", {}) or {}
    multi_ll_map = cfg.get("report_scope_multi_fallback_low_level_keywords", {}) or {}

    fallback_hl: list[str] = []
    fallback_ll: list[str] = []
    for scope in current_report_scopes:
        scope_name = str(scope or "").strip()
        if not scope_name:
            continue
        configured_hl = multi_hl_map.get(scope_name) if is_multi_scope else None
        configured_ll = multi_ll_map.get(scope_name) if is_multi_scope else None
        if not configured_hl:
            configured_hl = default_hl_map.get(scope_name) or [scope_name]
        if not configured_ll:
            configured_ll = default_ll_map.get(scope_name) or []
        fallback_hl.extend(
            configured_hl if isinstance(configured_hl, list) else [configured_hl]
        )
        fallback_ll.extend(
            configured_ll if isinstance(configured_ll, list) else [configured_ll]
        )

    return (
        _dedupe_preserve_order(merged_hl + fallback_hl),
        _dedupe_preserve_order(merged_ll + fallback_ll),
    )


def _filter_project_context_by_report_scope(
        project_param_map: dict[str, list[str]],
        project_param_value_map: dict[str, dict[str, dict[str, str]]],
        current_report_scopes: list[str],
        domain_rule_decisions: dict[str, Any] | None = None,
        stand_type: str | None = None,
) -> tuple[dict[str, list[str]], dict[str, dict[str, dict[str, str]]]]:
    if not current_report_scopes:
        return project_param_map, project_param_value_map

    whitelist_map = _get_report_scope_test_whitelist(stand_type)
    allowed_tests: set[str] = set()
    for scope in current_report_scopes:
        allowed_tests.update(whitelist_map.get(str(scope).strip(), set()))
    for decision in (domain_rule_decisions or {}).values():
        if not isinstance(decision, dict):
            continue
        if str(decision.get("kind", "") or "").strip() != "split":
            continue
        if not decision.get("enabled"):
            continue
        for split_output in decision.get("split_output", []) or []:
            if not isinstance(split_output, dict):
                continue
            split_name = str(split_output.get("test_item", "") or "").strip()
            if split_name:
                allowed_tests.add(split_name)
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


def _postprocess_electrical_markdown_response(
        response_text: str,
        raw_data: dict[str, Any] | None,
) -> str:
    if not response_text or not isinstance(raw_data, dict):
        return response_text

    metadata = raw_data.get("metadata", {}) if isinstance(raw_data, dict) else {}
    allowed_items = metadata.get("allowed_final_test_items", []) or []
    removed_items = metadata.get("removed_test_items", []) or []
    allowed_items_raw = metadata.get("allowed_final_test_items_raw", allowed_items) or []
    removed_items_raw = metadata.get("removed_test_items_raw", removed_items) or []
    value_map = metadata.get("project_param_value_map", {}) or {}
    param_map = metadata.get("project_param_map", {}) or {}
    display_map = metadata.get("test_item_display_map", {}) or {}
    rule_query_text = str(metadata.get("rule_query_text", "") or "")
    if not allowed_items and not removed_items:
        return response_text

    normalized_display_map = {
        str(k).strip(): str(v).strip()
        for k, v in display_map.items()
        if str(k).strip() and str(v).strip()
    }
    ordered_allowed_items = [
        str(item).strip() for item in allowed_items_raw if str(item).strip()
    ]
    allowed_set = set(ordered_allowed_items)
    removed_set = set(str(item).strip() for item in removed_items_raw if str(item).strip())
    allowed_display_set = {
        normalized_display_map.get(item, item) for item in allowed_set
    }
    removed_display_set = {
        normalized_display_map.get(item, item) for item in removed_set
    }
    explicit_gas_or_oil = any(
        token in rule_query_text
        for token in ("SF6", "六氟化硫", "充气断路器", "充油断路器")
    )
    suppressed_display_params = _get_display_param_suppressions()

    def _canonical_test_name(test_name: str) -> str:
        name = str(test_name or "").strip()
        if not name:
            return name
        if name in value_map:
            return name
        matched_keys = [
            key for key, display_name in normalized_display_map.items() if display_name == name
        ]
        if len(matched_keys) == 1:
            return matched_keys[0]
        return name

    def _display_test_name(test_name: str) -> str:
        canonical_name = _canonical_test_name(test_name)
        return normalized_display_map.get(canonical_name, canonical_name)

    allowed_display_items = [
        (canonical_name, _display_test_name(canonical_name))
        for canonical_name in ordered_allowed_items
    ]

    def _parse_sections(text: str) -> tuple[list[str], dict[str, list[str]]]:
        lines = text.splitlines()
        order: list[str] = []
        sections: dict[str, list[str]] = {}
        current = "__prefix__"
        sections[current] = []
        order.append(current)
        for line in lines:
            if line.startswith("### "):
                current = line.strip()
                sections.setdefault(current, [])
                order.append(current)
                continue
            sections.setdefault(current, []).append(line)
        return order, sections

    def _format_param_line(test_name: str, param_name: str, entry: dict[str, Any]) -> str:
        entry = dict(entry or {})
        value_text = _sanitize_value_text(str(entry.get("value_text", "") or ""))
        if param_name == "介质性质" and not explicit_gas_or_oil:
            value_text = "正常"
            entry.update(
                {
                    "value_source": str(entry.get("value_source", "") or "").strip()
                                    or "rule",
                    "calc_rule": "未命中SF6、六氟化硫、充气断路器、充油断路器等明确介质证据时，介质性质强制按正常输出。",
                }
            )
        if not value_text:
            return ""
        resolution_mode = str(entry.get("resolution_mode", "") or "").strip()
        if resolution_mode != "graph_final" and not _is_concrete_final_value_text(value_text):
            return ""
        suffix = ""
        source = str(entry.get("value_source", "") or "").strip()
        calc_rule = str(entry.get("calc_rule", "") or "").strip()
        if source:
            suffix += f"；source：{source}"
        if calc_rule:
            suffix += f"；calculation：{calc_rule}"
        return f"- {param_name}：{value_text}{suffix}"

    def _build_missing_c_block(test_name: str) -> list[str]:
        canonical_name = _canonical_test_name(test_name)
        params = param_map.get(canonical_name, []) if isinstance(param_map, dict) else []
        param_values = value_map.get(canonical_name, {}) if isinstance(value_map, dict) else {}
        block_lines = [f"## 试验项目：{_display_test_name(canonical_name)}"]
        for param_name in params:
            if param_name in suppressed_display_params.get(canonical_name, set()):
                continue
            entry = param_values.get(param_name, {}) if isinstance(param_values, dict) else {}
            if not isinstance(entry, dict):
                continue
            line = _format_param_line(canonical_name, param_name, entry)
            if line:
                block_lines.append(line)
        return block_lines if len(block_lines) > 1 else []

    def _trim_trailing_blank_lines(lines: list[str]) -> None:
        while lines and not lines[-1].strip():
            lines.pop()

    def _is_visual_separator_line(line: str) -> bool:
        stripped = line.strip()
        return stripped in {"---", "***", "___"}

    def _resolve_test_count_text(test_name: str) -> str:
        canonical_name = _canonical_test_name(test_name)
        param_values = value_map.get(canonical_name, {}) if isinstance(value_map, dict) else {}
        if not isinstance(param_values, dict):
            return ""
        count_fields = (
            ("正常次数", "试验次数")
            if _uses_normal_count_label(canonical_name)
            else ("试验次数", "正常次数")
        )
        for count_name in count_fields:
            entry = param_values.get(count_name, {})
            if not isinstance(entry, dict):
                continue
            value_text = str(entry.get("value_text", "") or "").strip()
            if value_text:
                return value_text
        return ""

    lc_cc_coverage_required_items = {
        "容性电流开断试验(LC1)",
        "容性电流开断试验(LC2)",
        "容性电流开断试验(CC1)",
        "容性电流开断试验(CC2)",
    }
    op1_optional_required_item = "失步关合和开断试验(OP1)"

    def _extract_note_rated_voltage_kv(query_text: str) -> float | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        match = re.search(
            r"额定电压\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*kV\b",
            text,
            flags=re.IGNORECASE,
        )
        return float(match.group(1)) if match else None

    note_rated_voltage_kv = _extract_note_rated_voltage_kv(rule_query_text)

    def _should_keep_a_section_note(line: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped:
            return True
        if "LC1，LC2可被CC1、CC2覆盖" in stripped or "LC1,LC2可被CC1、CC2覆盖" in stripped:
            return lc_cc_coverage_required_items.issubset(allowed_display_set)
        if "失步关合和开断试验(OP1)试验可免做" in stripped:
            return (
                op1_optional_required_item in allowed_display_set
                and note_rated_voltage_kv is not None
                and note_rated_voltage_kv >= 72.5
            )
        return True

    def _build_required_a_section_notes(existing_lines: list[str]) -> list[str]:
        existing_text = "\n".join(
            str(line or "").strip()
            for line in existing_lines
            if str(line or "").strip()
        )
        required_notes: list[str] = []

        if lc_cc_coverage_required_items.issubset(allowed_display_set) and (
            "LC1，LC2可被CC1、CC2覆盖" not in existing_text
            and "LC1,LC2可被CC1、CC2覆盖" not in existing_text
        ):
            required_notes.append("LC1，LC2可被CC1、CC2覆盖")

        if (
            op1_optional_required_item in allowed_display_set
            and note_rated_voltage_kv is not None
            and note_rated_voltage_kv >= 72.5
            and "失步关合和开断试验(OP1)试验可免做" not in existing_text
        ):
            required_notes.append("失步关合和开断试验(OP1)试验可免做")

        return required_notes

    def _filter_a_section(lines: list[str]) -> list[str]:
        filtered: list[str] = []
        seen_items: set[str] = set()
        for line in lines:
            stripped = line.strip()
            if not stripped or _is_visual_separator_line(line):
                continue
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                canonical_name = _canonical_test_name(item)
                display_name = _display_test_name(item)
                if canonical_name in allowed_set or (
                    display_name in allowed_display_set and display_name not in removed_display_set
                ):
                    filtered.append(f"- {display_name}")
                    seen_items.add(canonical_name or item)
            else:
                if _should_keep_a_section_note(line):
                    filtered.append(line)
        _trim_trailing_blank_lines(filtered)
        for canonical_name, display_name in allowed_display_items:
            if canonical_name not in seen_items:
                filtered.append(f"- {display_name}")
        required_notes = _build_required_a_section_notes(filtered)
        if required_notes:
            if filtered and filtered[-1].strip():
                filtered.append("")
            for index, note in enumerate(required_notes):
                if index < len(required_notes) - 1:
                    filtered.append(f"{note}  ")
                else:
                    filtered.append(note)
        return filtered

    def _extract_test_name(heading_line: str) -> str | None:
        match = re.match(r"^(?:##\s*)?试验项目[:：]\s*(.+?)\s*$", heading_line.strip())
        return match.group(1).strip() if match else None

    def _rewrite_param_line(test_name: str, line: str) -> str:
        match = re.match(r"^(\s*(?:-\s*)?)([^：:]+)([：:])\s*(.+)$", line)
        if not match:
            return line
        prefix, param_name, _colon, remainder = match.groups()
        canonical_name = _canonical_test_name(test_name)
        param_name = param_name.strip()
        if param_name in suppressed_display_params.get(canonical_name, set()):
            return ""
        param_values = value_map.get(canonical_name, {}) if isinstance(value_map, dict) else {}
        entry = param_values.get(param_name, {}) if isinstance(param_values, dict) else {}
        if not isinstance(entry, dict):
            return line
        if "无法确定" in remainder and not str(entry.get("value_text", "") or "").strip():
            return ""
        formatted_line = _format_param_line(canonical_name, param_name, entry)
        if not formatted_line.startswith("- "):
            if _is_concrete_final_value_text(remainder):
                return line
            return line
        return f"{prefix or '- '}{formatted_line[2:]}"

    def _looks_like_param_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped or _extract_test_name(stripped) is not None:
            return False
        return re.match(r"^(?:-\s*)?[^：:]+[：:].+$", stripped) is not None

    def _filter_c_section(lines: list[str]) -> list[str]:
        filtered: list[str] = []
        current_block: list[str] = []
        current_name: str | None = None
        seen_items: set[str] = set()

        def _flush() -> None:
            nonlocal current_block, current_name
            if not current_block:
                return
            canonical_name = _canonical_test_name(current_name or "") if current_name else ""
            display_name = _display_test_name(current_name or "") if current_name else ""
            if canonical_name and (
                    canonical_name in allowed_set
                    or (display_name in allowed_display_set and display_name not in removed_display_set)
            ):
                seen_items.add(canonical_name)
                seen_param_names: set[str] = set()
                for index, block_line in enumerate(current_block):
                    if index == 0 and current_name:
                        filtered.append(f"## 试验项目：{display_name}")
                        continue
                    if _looks_like_param_line(block_line):
                        match = re.match(r"^\s*-\s*([^：:]+)[：:]", block_line)
                        if not match:
                            match = re.match(r"^\s*([^：:]+)[：:]", block_line)
                        if match:
                            seen_param_names.add(match.group(1).strip())
                        rewritten = _rewrite_param_line(canonical_name, block_line)
                        if rewritten:
                            filtered.append(rewritten)
                    else:
                        filtered.append(block_line)
                test_params = param_map.get(canonical_name, []) if isinstance(param_map, dict) else []
                param_values = value_map.get(canonical_name, {}) if isinstance(value_map, dict) else {}
                for param_name in test_params:
                    if param_name in suppressed_display_params.get(canonical_name, set()):
                        continue
                    if param_name in seen_param_names:
                        continue
                    entry = param_values.get(param_name, {}) if isinstance(param_values, dict) else {}
                    if not isinstance(entry, dict):
                        continue
                    line = _format_param_line(canonical_name, param_name, entry)
                    if line:
                        filtered.append(line)
            elif current_name is None:
                filtered.extend(current_block)
            current_block = []
            current_name = None

        for line in lines:
            if _extract_test_name(line) is not None:
                _flush()
                current_name = _extract_test_name(line)
                current_block = [line]
            else:
                current_block.append(line)
        _flush()

        for canonical_name, _display_name in allowed_display_items:
            if canonical_name in seen_items:
                continue
            missing_block = _build_missing_c_block(canonical_name)
            if missing_block:
                _trim_trailing_blank_lines(filtered)
                filtered.extend(missing_block)
        return filtered

    def _filter_d_section(lines: list[str]) -> list[str]:
        filtered: list[str] = []
        total_items: list[tuple[str, str]] = []
        total_line: str | None = None
        seen_items: set[str] = set()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- ") and ("试验次数：" in stripped or "正常次数：" in stripped):
                count_label = "正常次数：" if "正常次数：" in stripped else "试验次数："
                item = stripped[2:].split(count_label, 1)[0].strip()
                canonical_name = _canonical_test_name(item)
                display_name = _display_test_name(item)
                if canonical_name in allowed_set or (
                        display_name in allowed_display_set and display_name not in removed_display_set
                ):
                    count_text = stripped.split(count_label, 1)[1].strip()
                    preferred_count_label = (
                        "正常次数" if _uses_normal_count_label(canonical_name) else "试验次数"
                    )
                    filtered.append(f"- {display_name}{preferred_count_label}：{count_text}")
                    total_items.append((display_name, count_text))
                    seen_items.add(canonical_name or item)
                continue
            if stripped.startswith("- 绝缘性能型式试验总次数："):
                total_line = line
                continue
            filtered.append(line)

        for canonical_name, display_name in allowed_display_items:
            if canonical_name in seen_items:
                continue
            count_text = _resolve_test_count_text(canonical_name)
            if not count_text:
                continue
            _trim_trailing_blank_lines(filtered)
            count_label = "正常次数" if _uses_normal_count_label(canonical_name) else "试验次数"
            filtered.append(f"- {display_name}{count_label}：{count_text}")
            total_items.append((display_name, count_text))

        if total_items:
            total = 0
            for _item, count_text in total_items:
                count_match = re.search(r"([0-9]+)", count_text)
                if not count_match:
                    continue
                count = int(count_match.group(1))
                total += count
            filtered.append(f"- 绝缘性能型式试验总次数：{total}")
        elif total_line:
            filtered.append(total_line)
        return filtered

    def _filter_e_section(lines: list[str]) -> list[str]:
        filtered: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- 缺失项：") and "无" not in stripped:
                continue
            if stripped.startswith("- 建议补检："):
                continue
            if stripped.startswith("- 图谱参数缺失项：") and "无" not in stripped:
                continue
            if stripped.startswith("- 参数覆盖校验："):
                continue
            if stripped.startswith("- 被白名单抑制参数："):
                continue
            if stripped.startswith("- 未能判定的条件值：") and "无" not in stripped:
                continue
            if stripped.startswith("- 未能计算的公式值：") and "无" not in stripped:
                continue
            filtered.append(line)
        return filtered

    order, sections = _parse_sections(response_text)
    rebuilt: list[str] = []
    for key in order:
        if key == "__prefix__":
            rebuilt.extend(sections.get(key, []))
            continue
        if rebuilt and rebuilt[-1].strip():
            rebuilt.append("")
        rebuilt.append(key)
        section_lines = sections.get(key, [])
        if key.startswith("### A."):
            rebuilt.extend(_filter_a_section(section_lines))
        elif key.startswith("### C."):
            rebuilt.extend(_filter_c_section(section_lines))
        elif key.startswith("### D."):
            rebuilt.extend(_filter_d_section(section_lines))
        elif key.startswith("### E."):
            rebuilt.extend(_filter_e_section(section_lines))
        else:
            rebuilt.extend(section_lines)
    return "\n".join(rebuilt)


def _cleanup_model_response_text(
        response_text: str,
        sys_prompt: str,
        query: str,
) -> str:
    cleaned = str(response_text or "")
    if sys_prompt and len(cleaned) > len(sys_prompt):
        cleaned = cleaned.replace(sys_prompt, "")
    return (
        cleaned.replace("user", "")
        .replace("model", "")
        .replace(query, "")
        .replace("<system>", "")
        .replace("</system>", "")
        .strip()
    )


async def _stream_postprocessed_electrical_response(
    response_stream: AsyncIterator[str],
    raw_data: dict[str, Any] | None,
    sys_prompt: str,
    query: str,
    debug_stage_prefix: str,
    chunk_size: int = 1024,
) -> AsyncIterator[str]:
    chunks: list[str] = []
    async for chunk in response_stream:
        if chunk:
            chunks.append(chunk)

    response_text = _cleanup_model_response_text("".join(chunks), sys_prompt, query)
    _log_electrical_answer_debug(
        f"{debug_stage_prefix}_before_postprocess",
        raw_data,
        response_text,
    )
    processed_text = _postprocess_electrical_markdown_response(
        _enforce_formula_consistency(response_text),
        raw_data,
    )
    if not processed_text:
        processed_text = "No relevant context found for the query."
    _log_electrical_answer_debug(
        f"{debug_stage_prefix}_after_postprocess",
        raw_data,
        processed_text,
    )

    for start in range(0, len(processed_text), chunk_size):
        yield processed_text[start : start + chunk_size]


def _has_electrical_postprocess_metadata(raw_data: dict[str, Any] | None) -> bool:
    if not isinstance(raw_data, dict):
        return False
    metadata = raw_data.get("metadata", {}) or {}
    allowed_items = metadata.get("allowed_final_test_items", []) or []
    removed_items = metadata.get("removed_test_items", []) or []
    domain_rule_decisions = metadata.get("domain_rule_decisions", {}) or {}
    rule_query_text = metadata.get("rule_query_text", "")
    return bool(allowed_items or removed_items or domain_rule_decisions or rule_query_text)


def _build_electrical_a_section_note_patch(
    raw_data: dict[str, Any] | None,
    response_text: str,
) -> list[str]:
    if not isinstance(raw_data, dict):
        return []
    metadata = raw_data.get("metadata", {}) or {}
    allowed_items = metadata.get("allowed_final_test_items_raw") or metadata.get(
        "allowed_final_test_items", []
    ) or []
    allowed_set = {str(item).strip() for item in allowed_items if str(item).strip()}
    display_map = metadata.get("test_item_display_map", {}) or {}
    normalized_display_map = {
        str(k).strip(): str(v).strip()
        for k, v in display_map.items()
        if str(k).strip() and str(v).strip()
    }
    allowed_display_set = {
        normalized_display_map.get(item, item) for item in allowed_set
    }
    if not allowed_set:
        return []

    existing_text = str(response_text or "")
    notes: list[str] = []
    lc_cc_coverage_required_items = {
        "容性电流开断试验(LC1)",
        "容性电流开断试验(LC2)",
        "容性电流开断试验(CC1)",
        "容性电流开断试验(CC2)",
    }
    if lc_cc_coverage_required_items.issubset(allowed_display_set) and (
        "LC1，LC2可被CC1、CC2覆盖" not in existing_text
        and "LC1,LC2可被CC1、CC2覆盖" not in existing_text
    ):
        notes.append("LC1，LC2可被CC1、CC2覆盖")

    rule_query_text = str(metadata.get("rule_query_text", "") or "")
    rated_voltage_match = re.search(
        r"额定电压\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*kV\b",
        rule_query_text,
        flags=re.IGNORECASE,
    )
    rated_voltage_kv = float(rated_voltage_match.group(1)) if rated_voltage_match else None
    op1_optional_required_item = "失步关合和开断试验(OP1)"
    if (
        op1_optional_required_item in allowed_display_set
        and rated_voltage_kv is not None
        and rated_voltage_kv >= 72.5
        and "失步关合和开断试验(OP1)试验可免做" not in existing_text
    ):
        notes.append("失步关合和开断试验(OP1)试验可免做")

    return notes


async def _stream_electrical_response_with_final_event(
    response_stream: AsyncIterator[str],
    raw_data: dict[str, Any] | None,
    sys_prompt: str,
    query: str,
    debug_stage_prefix: str,
) -> AsyncIterator[str | dict[str, str]]:
    chunks: list[str] = []
    a_section_notes = _build_electrical_a_section_note_patch(raw_data, "")
    line_buffer = ""
    inside_a_section = False
    notes_emitted = False

    async def _emit_notes_if_needed() -> AsyncIterator[dict[str, str]]:
        nonlocal notes_emitted
        if a_section_notes and not notes_emitted:
            notes_emitted = True
            yield {"event": "a_section_notes", "response": "\n".join(a_section_notes)}

    async def _flush_complete_lines(text: str) -> AsyncIterator[str | dict[str, str]]:
        nonlocal line_buffer, inside_a_section
        line_buffer += text
        while True:
            newline_index = line_buffer.find("\n")
            if newline_index < 0:
                break
            line = line_buffer[: newline_index + 1]
            line_buffer = line_buffer[newline_index + 1 :]
            stripped = line.strip()
            if stripped.startswith("### "):
                is_a_heading = stripped.startswith("### A.")
                if inside_a_section and not is_a_heading:
                    async for note_event in _emit_notes_if_needed():
                        yield note_event
                inside_a_section = is_a_heading
            yield line

    async for chunk in response_stream:
        if chunk:
            chunks.append(chunk)
            async for output in _flush_complete_lines(chunk):
                yield output

    if line_buffer:
        stripped = line_buffer.strip()
        if stripped.startswith("### "):
            is_a_heading = stripped.startswith("### A.")
            if inside_a_section and not is_a_heading:
                async for note_event in _emit_notes_if_needed():
                    yield note_event
            inside_a_section = is_a_heading
        yield line_buffer
        line_buffer = ""

    if inside_a_section:
        async for note_event in _emit_notes_if_needed():
            yield note_event

    response_text = _cleanup_model_response_text("".join(chunks), sys_prompt, query)
    _log_electrical_answer_debug(
        f"{debug_stage_prefix}_before_postprocess",
        raw_data,
        response_text,
    )
    processed_text = _postprocess_electrical_markdown_response(
        _enforce_formula_consistency(response_text),
        raw_data,
    )
    if not processed_text:
        processed_text = "No relevant context found for the query."
    _log_electrical_answer_debug(
        f"{debug_stage_prefix}_after_postprocess",
        raw_data,
        processed_text,
    )
    if not notes_emitted:
        a_section_notes = _build_electrical_a_section_note_patch(raw_data, response_text)
        if a_section_notes:
            yield {"event": "a_section_notes", "response": "\n".join(a_section_notes)}


def _log_electrical_answer_debug(
        stage: str,
        raw_data: dict[str, Any] | None,
        response_text: str | None = None,
) -> None:
    if not isinstance(raw_data, dict):
        return
    metadata = raw_data.get("metadata", {}) or {}
    allowed_items = metadata.get("allowed_final_test_items", []) or []
    removed_items = metadata.get("removed_test_items", []) or []
    domain_rule_decisions = metadata.get("domain_rule_decisions", {}) or {}
    rule_query_text = metadata.get("rule_query_text", "")
    project_param_map = metadata.get("project_param_map", {}) or {}
    project_param_value_map = metadata.get("project_param_value_map", {}) or {}
    if not (allowed_items or removed_items or domain_rule_decisions or rule_query_text):
        return
    _log_electrical_trace(
        stage,
        allowed_final_test_items=allowed_items,
        removed_test_items=removed_items,
        rule_query_text=rule_query_text,
        domain_rule_decisions=domain_rule_decisions,
        project_param_map=project_param_map,
        project_param_value_map=project_param_value_map,
        response=response_text,
    )


def _serialize_debug_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _compact_text_for_log(value: str, limit: int = _TRACE_TEXT_PREVIEW_LIMIT) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[omitted {omitted} chars]..."


def _compact_sequence_for_log(value: Any, limit: int = _TRACE_LIST_PREVIEW_LIMIT) -> Any:
    if not isinstance(value, (list, tuple, set)):
        return value
    seq = list(value)
    if len(seq) <= limit:
        return [_compact_value_for_log(item) for item in seq]
    preview = [_compact_value_for_log(item) for item in seq[:limit]]
    return {
        "count": len(seq),
        "preview": preview,
        "omitted_count": len(seq) - len(preview),
    }


def _compact_mapping_for_log(value: Any, limit: int = _TRACE_DICT_PREVIEW_LIMIT) -> Any:
    if not isinstance(value, dict):
        return value
    items = list(value.items())
    if len(items) <= limit:
        return {str(key): _compact_value_for_log(item) for key, item in items}
    preview_items = items[:limit]
    return {
        "count": len(items),
        "keys_preview": [str(key) for key, _ in preview_items],
        "preview": {str(key): _compact_value_for_log(item) for key, item in preview_items},
        "omitted_key_count": len(items) - len(preview_items),
    }


def _compact_value_for_log(value: Any) -> Any:
    if isinstance(value, str):
        return _compact_text_for_log(value)
    if isinstance(value, dict):
        return _compact_mapping_for_log(value)
    if isinstance(value, (list, tuple, set)):
        return _compact_sequence_for_log(value)
    return value


def _is_insulation_relevant_test_item(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return any(
        token in text
        for token in (
            "工频耐受电压试验",
            "雷电冲击耐受电压试验",
            "操作冲击耐受电压试验",
            "局部放电试验",
            "控制和辅助回路的绝缘试验",
        )
    )


def _extract_insulation_relevant_items_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        if isinstance(value.get("preview"), list):
            filtered_preview = [
                _compact_value_for_log(item)
                for item in value.get("preview", [])
                if _is_insulation_relevant_test_item(item)
            ]
            if filtered_preview:
                summarized_value = {
                    str(key): _compact_value_for_log(item)
                    for key, item in value.items()
                    if key != "preview"
                }
                summarized_value["preview"] = filtered_preview
                return summarized_value
        filtered = {
            str(key): _compact_value_for_log(item)
            for key, item in value.items()
            if _is_insulation_relevant_test_item(key)
        }
        return filtered or None
    if isinstance(value, (list, tuple, set)):
        filtered = [
            _compact_value_for_log(item)
            for item in value
            if _is_insulation_relevant_test_item(item)
        ]
        return filtered or None
    return None


def _should_log_electrical_stage(stage: str) -> bool:
    return True


def _should_keep_full_trace_value(stage: str, key: str) -> bool:
    return stage == "kg_query_keywords" and key in {
        "high_level_keywords",
        "low_level_keywords",
    }


def _build_project_param_coverage_summary(
        project_param_map: dict[str, Any] | None,
        project_param_value_map: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    normalized_project_param_map = project_param_map or {}
    normalized_project_param_value_map = project_param_value_map or {}
    all_test_items = sorted(
        {
            *(str(key) for key in normalized_project_param_map.keys()),
            *(str(key) for key in normalized_project_param_value_map.keys()),
        }
    )
    for test_name in all_test_items:
        expected_params = [
            str(param)
            for param in (normalized_project_param_map.get(test_name, []) or [])
            if str(param).strip()
        ]
        actual_value_map = normalized_project_param_value_map.get(test_name, {}) or {}
        actual_params = [
            str(param)
            for param in actual_value_map.keys()
            if str(param).strip()
        ]
        missing_params = [param for param in expected_params if param not in actual_params]
        extra_params = [param for param in actual_params if param not in expected_params]
        summary[test_name] = {
            "expected_param_count": len(expected_params),
            "value_param_count": len(actual_params),
            "missing_params": missing_params,
            "extra_params": extra_params,
            "expected_params_preview": expected_params[:_TRACE_LIST_PREVIEW_LIMIT],
            "value_params_preview": actual_params[:_TRACE_LIST_PREVIEW_LIMIT],
        }
    return summary


def _summarize_trace_payload(stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    summarized: dict[str, Any] = {}
    project_param_map = payload.get("project_param_map")
    project_param_value_map = payload.get("project_param_value_map")
    if project_param_map is None and "display_project_param_map" in payload:
        project_param_map = payload.get("display_project_param_map")
    if project_param_value_map is None and "display_project_param_value_map" in payload:
        project_param_value_map = payload.get("display_project_param_value_map")

    if project_param_map is not None or project_param_value_map is not None:
        summarized["test_item_param_coverage"] = _build_project_param_coverage_summary(
            project_param_map if isinstance(project_param_map, dict) else {},
            project_param_value_map if isinstance(project_param_value_map, dict) else {},
        )
    if isinstance(project_param_map, dict):
        insulation_project_param_map = _extract_insulation_relevant_items_for_log(
            project_param_map
        )
        if insulation_project_param_map:
            summarized["project_param_map_insulation_items"] = insulation_project_param_map
    if isinstance(project_param_value_map, dict):
        insulation_project_param_value_map = _extract_insulation_relevant_items_for_log(
            project_param_value_map
        )
        if insulation_project_param_value_map:
            summarized["project_param_value_map_insulation_items"] = (
                insulation_project_param_value_map
            )

    for key, value in payload.items():
        if key in {
            "project_param_map",
            "project_param_value_map",
            "display_project_param_map",
            "display_project_param_value_map",
        }:
            continue
        if key == "user_prompt":
            prompt_text = str(value or "")
            summarized["user_prompt_provided"] = bool(prompt_text.strip())
            summarized["user_prompt_length"] = len(prompt_text)
            continue
        if key == "history_messages":
            history_messages = value if isinstance(value, list) else []
            summarized["history_message_count"] = len(history_messages)
            if history_messages:
                summarized["history_messages_preview"] = _compact_sequence_for_log(
                    history_messages, limit=3
                )
            continue
        if key == "response":
            response_text = str(value or "")
            summarized["response_length"] = len(response_text)
            if response_text:
                summarized["response_preview"] = _compact_text_for_log(response_text)
            continue
        if key == "assembled_prompt":
            if _ELECTRICAL_DEBUG_VERBOSE:
                summarized[key] = str(value or "")
            continue
        if _should_keep_full_trace_value(stage, key):
            if isinstance(value, (list, tuple, set)):
                summarized[key] = list(value)
            else:
                summarized[key] = value
            continue
        summarized[key] = _compact_value_for_log(value)
        if key in {
            "allowed_final_test_items",
            "allowed_final_test_items_raw",
            "removed_test_items",
            "removed_test_items_raw",
            "project_param_map_keys",
            "project_param_value_map_keys",
        }:
            insulation_relevant_items = _extract_insulation_relevant_items_for_log(value)
            if insulation_relevant_items:
                summarized[f"{key}_insulation_items"] = insulation_relevant_items

    return summarized


def _start_electrical_trace_session(
        *,
        source: str,
        mode: str,
        query: str,
        retrieval_query: str,
        user_prompt: str,
        response_type: str,
        current_report_scopes: list[str] | None = None,
        scope_focused_query_applied: bool | None = None,
        scope_focused_query_reason: str = "",
) -> str:
    trace_id = f"{source}-{uuid.uuid4().hex[:8]}"
    trace_context = {
        "trace_id": trace_id,
        "source": source,
        "mode": mode,
        "query": query,
        "retrieval_query": retrieval_query,
        "user_prompt": user_prompt,
        "response_type": response_type,
        "current_report_scopes": list(current_report_scopes or []),
        "scope_focused_query_applied": scope_focused_query_applied,
        "scope_focused_query_reason": scope_focused_query_reason,
        "stage_headers_logged": set(),
    }
    _CURRENT_ELECTRICAL_TRACE.set(trace_context)
    _electrical_debug_logger.info(
        "========== ELECTRICAL TRACE START trace=%s ==========",
        trace_id,
    )
    _log_electrical_trace(
        "trace_start",
        source=source,
        mode=mode,
        query=query,
        retrieval_query=retrieval_query,
        user_prompt=user_prompt,
        response_type=response_type,
        current_report_scopes=current_report_scopes or [],
        scope_focused_query_applied=scope_focused_query_applied,
        scope_focused_query_reason=scope_focused_query_reason,
    )
    return trace_id


def _log_electrical_trace(stage: str, **payload: Any) -> None:
    if not payload:
        return
    if not _should_log_electrical_stage(stage):
        return
    trace_context = _CURRENT_ELECTRICAL_TRACE.get() or {}
    trace_id = str(trace_context.get("trace_id", "no-trace"))
    step_desc = _TRACE_STAGE_DESCRIPTIONS.get(stage, stage)
    stage_headers_logged = trace_context.setdefault("stage_headers_logged", set())
    if stage not in stage_headers_logged:
        _electrical_debug_logger.info(
            "========== **[%s] %s** trace=%s ==========",
            stage,
            step_desc,
            trace_id,
        )
        stage_headers_logged.add(stage)
    summarized_payload = _summarize_trace_payload(stage, payload)
    for key, value in summarized_payload.items():
        _electrical_debug_logger.info(
            "[electrical_trace][trace=%s][%s][%s] %s=%s",
            trace_id,
            stage,
            step_desc,
            key,
            _serialize_debug_value(value),
        )


def _extract_calculation_result_value(line: str) -> str | None:
    if "calculation" not in line.lower():
        return None
    normalized = line.replace("：", ":")
    calc_match = re.search(
        r"calculation\s*:\s*(.+)$", normalized, flags=re.IGNORECASE
    )
    if not calc_match:
        return None
    calc_body = calc_match.group(1).strip()
    if "=" not in calc_body:
        return None
    rhs = calc_body.rsplit("=", 1)[-1].strip()
    rhs = re.split(r"[；;（(。,\n]", rhs, maxsplit=1)[0].strip()
    if not rhs:
        return None
    value_match = re.match(
        r"^(?:≤|≥|<|>)?\s*-?\d+(?:\.\d+)?\s*(?:kV|V|mV|A|kA|s|min|ms|pC|%|m)?$",
        rhs,
        flags=re.IGNORECASE,
    )
    return rhs if value_match else None


def _normalize_value_for_compare(value: str) -> str:
    return (
        value.replace(" ", "")
        .replace("：", ":")
        .replace("（", "(")
        .replace("）", ")")
        .strip()
        .lower()
    )


def _enforce_formula_consistency(response_text: str) -> str:
    if not response_text or "calculation" not in response_text.lower():
        return response_text

    corrected_lines: list[str] = []
    corrected_count = 0

    for line in response_text.splitlines():
        if "calculation" not in line.lower():
            corrected_lines.append(line)
            continue

        calc_value = _extract_calculation_result_value(line)
        param_match = re.match(r"^(\s*-\s*[^：:\n]+[：:]\s*)([^；;\n]+)(.*)$", line)
        if not calc_value or not param_match:
            corrected_lines.append(line)
            continue

        prefix, current_value, suffix = param_match.groups()
        if _normalize_value_for_compare(current_value) == _normalize_value_for_compare(
                calc_value
        ):
            corrected_lines.append(line)
            continue

        corrected_lines.append(f"{prefix}{calc_value}{suffix}")
        corrected_count += 1

    if corrected_count:
        logger.info(
            "Corrected %s formula-derived parameter value(s) to match calculation result",
            corrected_count,
        )

    return "\n".join(corrected_lines)


def _log_model_input_trace(
        stage: str,
        *,
        system_prompt: str,
        user_query: str,
        context_data: str,
        history_messages: list[dict[str, str]] | None,
) -> None:
    assembled_prompt = "\n\n".join([system_prompt, "---User Query---", user_query])
    _log_electrical_trace(
        stage,
        prompt_stats={
            "system_prompt_length": len(system_prompt),
            "context_data_length": len(context_data),
            "user_query_length": len(user_query),
        },
        history_messages=history_messages or [],
        assembled_prompt=assembled_prompt,
    )


def _truncate_entity_identifier(
        identifier: str, limit: int, chunk_key: str, identifier_role: str
) -> str:
    """Truncate entity identifiers that exceed the configured length limit."""

    if len(identifier) <= limit:
        return identifier

    display_value = identifier[:limit]
    preview = identifier[:20]  # Show first 20 characters as preview
    logger.warning(
        "%s: %s len %d > %d chars (Name: '%s...')",
        chunk_key,
        identifier_role,
        len(identifier),
        limit,
        preview,
    )
    return display_value


def chunking_by_token_size(
        tokenizer: Tokenizer,
        content: str,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        chunk_overlap_token_size: int = 100,
        chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    def _prepare_chunk_content_for_table_chunking(raw_text: str) -> str:
        text = str(raw_text or "")
        if not any(tag in text.lower() for tag in ("<table", "<tr", "<td", "</tr>", "</td>")):
            return text
        normalized = html.unescape(text)
        normalized = re.sub(r"(?i)</tr>", "</tr>\n", normalized)
        normalized = re.sub(r"(?i)<tr\b", "\n<tr", normalized)
        normalized = re.sub(r"(?i)</td>", " | ", normalized)
        normalized = re.sub(r"(?i)<br\s*/?>", "\n", normalized)
        normalized = re.sub(r"<[^>]+>", " ", normalized)
        normalized = re.sub(r"[ \t]+\|\s*\|+", " | ", normalized)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    content_for_chunking = _prepare_chunk_content_for_table_chunking(content)
    tokens = tokenizer.encode(content_for_chunking)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content_for_chunking.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    logger.warning(
                        "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                        len(_tokens),
                        chunk_token_size,
                    )
                    raise ChunkTokenLimitExceededError(
                        chunk_tokens=len(_tokens),
                        chunk_token_limit=chunk_token_size,
                        chunk_preview=chunk[:120],
                    )
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    for start in range(
                            0, len(_tokens), chunk_token_size - chunk_overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start: start + chunk_token_size]
                        )
                        new_chunks.append(
                            (min(chunk_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
                range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start: start + chunk_token_size])
            results.append(
                {
                    "tokens": min(chunk_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
        description_type: str,
        entity_or_relation_name: str,
        description_list: list[str],
        seperator: str,
        global_config: dict,
        llm_response_cache: BaseKVStorage | None = None,
) -> tuple[str, bool]:
    """Handle entity relation description summary using map-reduce approach.

    This function summarizes a list of descriptions using a map-reduce strategy:
    1. If total tokens < summary_context_size and len(description_list) < force_llm_summary_on_merge, no need to summarize
    2. If total tokens < summary_max_tokens, summarize with LLM directly
    3. Otherwise, split descriptions into chunks that fit within token limits
    4. Summarize each chunk, then recursively process the summaries
    5. Continue until we get a final summary within token limits or num of descriptions is less than force_llm_summary_on_merge

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        description_list: List of description strings to summarize
        global_config: Global configuration containing tokenizer and limits
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Tuple of (final_summarized_description_string, llm_was_used_boolean)
    """
    # Handle empty input
    if not description_list:
        return "", False

    # If only one description, return it directly (no need for LLM call)
    if len(description_list) == 1:
        return description_list[0], False

    # Get configuration
    tokenizer: Tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]
    summary_max_tokens = global_config["summary_max_tokens"]
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    current_list = description_list[:]  # Copy the list to avoid modifying original
    llm_was_used = False  # Track whether LLM was used during the entire process

    # Iterative map-reduce process
    while True:
        # Calculate total tokens in current list
        total_tokens = sum(len(tokenizer.encode(desc)) for desc in current_list)

        # If total length is within limits, perform final summarization
        if total_tokens <= summary_context_size or len(current_list) <= 2:
            if (
                    len(current_list) < force_llm_summary_on_merge
                    and total_tokens < summary_max_tokens
            ):
                # no LLM needed, just join the descriptions
                final_description = seperator.join(current_list)
                return final_description if final_description else "", llm_was_used
            else:
                if total_tokens > summary_context_size and len(current_list) <= 2:
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize descpriton found"
                    )
                # Final summarization of remaining descriptions - LLM will be used
                final_summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    current_list,
                    global_config,
                    llm_response_cache,
                )
                return final_summary, True  # LLM was used for final summarization

        # Need to split into chunks - Map phase
        # Ensure each chunk has minimum 2 descriptions to guarantee progress
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Currently least 3 descriptions in current_list
        for i, desc in enumerate(current_list):
            desc_tokens = len(tokenizer.encode(desc))

            # If adding current description would exceed limit, finalize current chunk
            if current_tokens + desc_tokens > summary_context_size and current_chunk:
                # Ensure we have at least 2 descriptions in the chunk (when possible)
                if len(current_chunk) == 1:
                    # Force add one more description to ensure minimum 2 per chunk
                    current_chunk.append(desc)
                    chunks.append(current_chunk)
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize descpriton found"
                    )
                    current_chunk = []  # next group is empty
                    current_tokens = 0
                else:  # curren_chunk is ready for summary in reduce phase
                    chunks.append(current_chunk)
                    current_chunk = [desc]  # leave it for next group
                    current_tokens = desc_tokens
            else:
                current_chunk.append(desc)
                current_tokens += desc_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"   Summarizing {entity_or_relation_name}: Map {len(current_list)} descriptions into {len(chunks)} groups"
        )

        # Reduce phase: summarize each group from chunks
        new_summaries = []
        for chunk in chunks:
            if len(chunk) == 1:
                # Optimization: single description chunks don't need LLM summarization
                new_summaries.append(chunk[0])
            else:
                # Multiple descriptions need LLM summarization
                summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    chunk,
                    global_config,
                    llm_response_cache,
                )
                new_summaries.append(summary)
                llm_was_used = True  # Mark that LLM was used in reduce phase

        # Update current list with new summaries for next iteration
        current_list = new_summaries


async def _summarize_descriptions(
        description_type: str,
        description_name: str,
        description_list: list[str],
        global_config: dict,
        llm_response_cache: BaseKVStorage | None = None,
) -> str:
    """Helper function to summarize a list of descriptions using LLM.

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        descriptions: List of description strings to summarize
        global_config: Global configuration containing LLM function and settings
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Summarized description string
    """
    use_llm_func: callable = global_config["llm_model_func"]
    # Apply higher priority (8) to entity/relation summary tasks
    use_llm_func = partial(use_llm_func, _priority=8)

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)

    summary_length_recommended = global_config["summary_length_recommended"]

    prompt_template = PROMPTS["summarize_entity_descriptions"]

    # Convert descriptions to JSONL format and apply token-based truncation
    tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]

    # Create list of JSON objects with "Description" field
    json_descriptions = [{"Description": desc} for desc in description_list]

    # Use truncate_list_by_token_size for length truncation
    truncated_json_descriptions = truncate_list_by_token_size(
        json_descriptions,
        key=lambda x: json.dumps(x, ensure_ascii=False),
        max_token_size=summary_context_size,
        tokenizer=tokenizer,
    )

    # Convert to JSONL format (one JSON object per line)
    joined_descriptions = "\n".join(
        json.dumps(desc, ensure_ascii=False) for desc in truncated_json_descriptions
    )

    # Prepare context for the prompt
    context_base = dict(
        description_type=description_type,
        description_name=description_name,
        description_list=joined_descriptions,
        summary_length=summary_length_recommended,
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)

    # Use LLM function with cache (higher priority for summary generation)
    summary, _ = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        cache_type="summary",
    )

    # Check summary token length against embedding limit
    embedding_token_limit = global_config.get("embedding_token_limit")
    if embedding_token_limit is not None and summary:
        tokenizer = global_config["tokenizer"]
        summary_token_count = len(tokenizer.encode(summary))
        threshold = int(embedding_token_limit)

        if summary_token_count > threshold:
            logger.warning(
                f"Summary tokens({summary_token_count}) exceeds embedding_token_limit({embedding_token_limit}) "
                f" for {description_type}: {description_name}"
            )

    return summary


async def _handle_single_entity_extraction(
        record_attributes: list[str],
        chunk_key: str,
        timestamp: int,
        file_path: str = "unknown_source",
):
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        if len(record_attributes) > 1 and "entity" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/4 feilds on ENTITY `{record_attributes[1]}` @ `{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        entity_name = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )

        # Validate entity name after all cleaning steps
        if not entity_name or not entity_name.strip():
            logger.info(
                f"Empty entity name found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        # Process entity type with same cleaning pipeline
        entity_type = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        if not entity_type.strip() or any(
                char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
        ):
            logger.warning(
                f"Entity extraction error: invalid entity type in: {record_attributes}"
            )
            return None

        # Remove spaces and convert to lowercase
        entity_type = entity_type.replace(" ", "").lower()

        # Process entity description with same cleaning pipeline
        entity_description = sanitize_and_normalize_extracted_text(record_attributes[3])

        if not entity_description.strip():
            logger.warning(
                f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
            )
            return None

        return dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.error(
            f"Entity extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Entity extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def _handle_single_relationship_extraction(
        record_attributes: list[str],
        chunk_key: str,
        timestamp: int,
        file_path: str = "unknown_source",
):
    if (
            len(record_attributes) != 5 or "relation" not in record_attributes[0]
    ):  # treat "relationship" and "relation" interchangeable
        if len(record_attributes) > 1 and "relation" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/5 fields on REALTION `{record_attributes[1]}`~`{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        source = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )
        target = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        # Validate entity names after all cleaning steps
        if not source:
            logger.info(
                f"Empty source entity found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        if not target:
            logger.info(
                f"Empty target entity found after sanitization. Original: '{record_attributes[2]}'"
            )
            return None

        if source == target:
            logger.debug(
                f"Relationship source and target are the same in: {record_attributes}"
            )
            return None

        # Process keywords with same cleaning pipeline
        edge_keywords = sanitize_and_normalize_extracted_text(
            record_attributes[3], remove_inner_quotes=True
        )
        edge_keywords = edge_keywords.replace("，", ",")

        # Process relationship description with same cleaning pipeline
        edge_description = sanitize_and_normalize_extracted_text(record_attributes[4])

        edge_source_id = chunk_key
        weight = (
            float(record_attributes[-1].strip('"').strip("'"))
            if is_float_regex(record_attributes[-1].strip('"').strip("'"))
            else 1.0
        )

        return dict(
            src_id=source,
            tgt_id=target,
            weight=weight,
            description=edge_description,
            keywords=edge_keywords,
            source_id=edge_source_id,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.warning(
            f"Relationship extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.warning(
            f"Relationship extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def rebuild_knowledge_from_chunks(
    entities_to_rebuild: dict[str, list[str]],
    relationships_to_rebuild: dict[tuple[str, str], list[str]],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_storage: BaseKVStorage,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
) -> None:
    """Rebuild entity and relationship descriptions from cached extraction results with parallel processing

    This method uses cached LLM extraction results instead of calling LLM again,
    following the same approach as the insert process. Now with parallel processing
    controlled by llm_model_max_async and using get_storage_keyed_lock for data consistency.

    Args:
        entities_to_rebuild: Dict mapping entity_name -> list of remaining chunk_ids
        relationships_to_rebuild: Dict mapping (src, tgt) -> list of remaining chunk_ids
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_storage: Text chunks storage
        llm_response_cache: LLM response cache
        global_config: Global configuration containing llm_model_max_async
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        entity_chunks_storage: KV storage maintaining full chunk IDs per entity
        relation_chunks_storage: KV storage maintaining full chunk IDs per relation
    """
    if not entities_to_rebuild and not relationships_to_rebuild:
        return

    # Get all referenced chunk IDs
    all_referenced_chunk_ids = set()
    for chunk_ids in entities_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)
    for chunk_ids in relationships_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)

    status_message = f"Rebuilding knowledge from {len(all_referenced_chunk_ids)} cached chunk extractions (parallel processing)"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Get cached extraction results for these chunks using storage
    # cached_results： chunk_id -> [list of (extraction_result, create_time) from LLM cache sorted by create_time of the first extraction_result]
    cached_results = await _get_cached_extraction_results(
        llm_response_cache,
        all_referenced_chunk_ids,
        text_chunks_storage=text_chunks_storage,
    )

    if not cached_results:
        status_message = "No cached extraction results found, cannot rebuild"
        logger.warning(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
        return

    # Process cached results to get entities and relationships for each chunk
    chunk_entities = {}  # chunk_id -> {entity_name: [entity_data]}
    chunk_relationships = {}  # chunk_id -> {(src, tgt): [relationship_data]}

    for chunk_id, results in cached_results.items():
        try:
            # Handle multiple extraction results per chunk
            chunk_entities[chunk_id] = defaultdict(list)
            chunk_relationships[chunk_id] = defaultdict(list)

            # process multiple LLM extraction results for a single chunk_id
            for result in results:
                entities, relationships = await _rebuild_from_extraction_result(
                    text_chunks_storage=text_chunks_storage,
                    chunk_id=chunk_id,
                    extraction_result=result[0],
                    timestamp=result[1],
                )

                # Merge entities and relationships from this extraction result
                # Compare description lengths and keep the better version for the same chunk_id
                for entity_name, entity_list in entities.items():
                    if entity_name not in chunk_entities[chunk_id]:
                        # New entity for this chunk_id
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    elif len(chunk_entities[chunk_id][entity_name]) == 0:
                        # Empty list, add the new entities
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_entities[chunk_id][entity_name][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(entity_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new entity that has longer description
                            chunk_entities[chunk_id][entity_name] = list(entity_list)
                        # Otherwise keep existing version

                # Compare description lengths and keep the better version for the same chunk_id
                for rel_key, rel_list in relationships.items():
                    if rel_key not in chunk_relationships[chunk_id]:
                        # New relationship for this chunk_id
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    elif len(chunk_relationships[chunk_id][rel_key]) == 0:
                        # Empty list, add the new relationships
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_relationships[chunk_id][rel_key][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(rel_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new relationship that has longer description
                            chunk_relationships[chunk_id][rel_key] = list(rel_list)
                        # Otherwise keep existing version

        except Exception as e:
            status_message = (
                f"Failed to parse cached extraction result for chunk {chunk_id}: {e}"
            )
            logger.info(status_message)  # Per requirement, change to info
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            continue

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # Counters for tracking progress
    rebuilt_entities_count = 0
    rebuilt_relationships_count = 0
    failed_entities_count = 0
    failed_relationships_count = 0

    async def _locked_rebuild_entity(entity_name, chunk_ids):
        nonlocal rebuilt_entities_count, failed_entities_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    await _rebuild_single_entity(
                        knowledge_graph_inst=knowledge_graph_inst,
                        entities_vdb=entities_vdb,
                        entity_name=entity_name,
                        chunk_ids=chunk_ids,
                        chunk_entities=chunk_entities,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        entity_chunks_storage=entity_chunks_storage,
                    )
                    rebuilt_entities_count += 1
                except Exception as e:
                    failed_entities_count += 1
                    status_message = f"Failed to rebuild `{entity_name}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    async def _locked_rebuild_relationship(src, tgt, chunk_ids):
        nonlocal rebuilt_relationships_count, failed_relationships_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            # Sort src and tgt to ensure order-independent lock key generation
            sorted_key_parts = sorted([src, tgt])
            async with get_storage_keyed_lock(
                    sorted_key_parts,
                    namespace=namespace,
                    enable_logging=False,
            ):
                try:
                    await _rebuild_single_relationship(
                        knowledge_graph_inst=knowledge_graph_inst,
                        relationships_vdb=relationships_vdb,
                        entities_vdb=entities_vdb,
                        src=src,
                        tgt=tgt,
                        chunk_ids=chunk_ids,
                        chunk_relationships=chunk_relationships,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        relation_chunks_storage=relation_chunks_storage,
                        entity_chunks_storage=entity_chunks_storage,
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                    )
                    rebuilt_relationships_count += 1
                except Exception as e:
                    failed_relationships_count += 1
                    status_message = f"Failed to rebuild `{src}`~`{tgt}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    # Create tasks for parallel processing
    tasks = []

    # Add entity rebuilding tasks
    for entity_name, chunk_ids in entities_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_entity(entity_name, chunk_ids))
        tasks.append(task)

    # Add relationship rebuilding tasks
    for (src, tgt), chunk_ids in relationships_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_relationship(src, tgt, chunk_ids))
        tasks.append(task)

    # Log parallel processing start
    status_message = f"Starting parallel rebuild of {len(entities_to_rebuild)} entities and {len(relationships_to_rebuild)} relationships (async: {graph_max_async})"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Execute all tasks in parallel with semaphore control and early failure detection
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None

    for task in done:
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                # Task completed successfully, retrieve result to mark as processed
                task.result()
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Re-raise the first exception to notify the caller
        raise first_exception

    # Final status report
    status_message = f"KG rebuild completed: {rebuilt_entities_count} entities and {rebuilt_relationships_count} relationships rebuilt successfully."
    if failed_entities_count > 0 or failed_relationships_count > 0:
        status_message += f" Failed: {failed_entities_count} entities, {failed_relationships_count} relationships."

    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _get_cached_extraction_results(
        llm_response_cache: BaseKVStorage,
        chunk_ids: set[str],
        text_chunks_storage: BaseKVStorage,
) -> dict[str, list[str]]:
    """Get cached extraction results for specific chunk IDs

    This function retrieves cached LLM extraction results for the given chunk IDs and returns
    them sorted by creation time. The results are sorted at two levels:
    1. Individual extraction results within each chunk are sorted by create_time (earliest first)
    2. Chunks themselves are sorted by the create_time of their earliest extraction result

    Args:
        llm_response_cache: LLM response cache storage
        chunk_ids: Set of chunk IDs to get cached results for
        text_chunks_storage: Text chunks storage for retrieving chunk data and LLM cache references

    Returns:
        Dict mapping chunk_id -> list of extraction_result_text, where:
        - Keys (chunk_ids) are ordered by the create_time of their first extraction result
        - Values (extraction results) are ordered by create_time within each chunk
    """
    cached_results = {}

    # Collect all LLM cache IDs from chunks
    all_cache_ids = set()

    # Read from storage
    chunk_data_list = await text_chunks_storage.get_by_ids(list(chunk_ids))
    for chunk_data in chunk_data_list:
        if chunk_data and isinstance(chunk_data, dict):
            llm_cache_list = chunk_data.get("llm_cache_list", [])
            if llm_cache_list:
                all_cache_ids.update(llm_cache_list)
        else:
            logger.warning(f"Chunk data is invalid or None: {chunk_data}")

    if not all_cache_ids:
        logger.warning(f"No LLM cache IDs found for {len(chunk_ids)} chunk IDs")
        return cached_results

    # Batch get LLM cache entries
    cache_data_list = await llm_response_cache.get_by_ids(list(all_cache_ids))

    # Process cache entries and group by chunk_id
    valid_entries = 0
    for cache_entry in cache_data_list:
        if (
                cache_entry is not None
                and isinstance(cache_entry, dict)
                and cache_entry.get("cache_type") == "extract"
                and cache_entry.get("chunk_id") in chunk_ids
        ):
            chunk_id = cache_entry["chunk_id"]
            extraction_result = cache_entry["return"]
            create_time = cache_entry.get(
                "create_time", 0
            )  # Get creation time, default to 0
            valid_entries += 1

            # Support multiple LLM caches per chunk
            if chunk_id not in cached_results:
                cached_results[chunk_id] = []
            # Store tuple with extraction result and creation time for sorting
            cached_results[chunk_id].append((extraction_result, create_time))

    # Sort extraction results by create_time for each chunk and collect earliest times
    chunk_earliest_times = {}
    for chunk_id in cached_results:
        # Sort by create_time (x[1]), then extract only extraction_result (x[0])
        cached_results[chunk_id].sort(key=lambda x: x[1])
        # Store the earliest create_time for this chunk (first item after sorting)
        chunk_earliest_times[chunk_id] = cached_results[chunk_id][0][1]

    # Sort cached_results by the earliest create_time of each chunk
    sorted_chunk_ids = sorted(
        chunk_earliest_times.keys(), key=lambda chunk_id: chunk_earliest_times[chunk_id]
    )

    # Rebuild cached_results in sorted order
    sorted_cached_results = {}
    for chunk_id in sorted_chunk_ids:
        sorted_cached_results[chunk_id] = cached_results[chunk_id]

    logger.info(
        f"Found {valid_entries} valid cache entries, {len(sorted_cached_results)} chunks with results"
    )
    return sorted_cached_results  # each item: list(extraction_result, create_time)


async def _process_extraction_result(
        result: str,
        chunk_key: str,
        timestamp: int,
        file_path: str = "unknown_source",
        tuple_delimiter: str = "<|#|>",
        completion_delimiter: str = "<|COMPLETE|>",
) -> tuple[dict, dict]:
    """Process a single extraction result (either initial or gleaning)
    Args:
        result (str): The extraction result to process
        chunk_key (str): The chunk key for source tracking
        file_path (str): The file path for citation
        tuple_delimiter (str): Delimiter for tuple fields
        record_delimiter (str): Delimiter for records
        completion_delimiter (str): Delimiter for completion
    Returns:
        tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
    """
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    if completion_delimiter not in result:
        logger.warning(
            f"{chunk_key}: Complete delimiter can not be found in extraction result"
        )

    # Split LLL output result to records by "\n"
    records = split_string_by_multi_markers(
        result,
        ["\n", completion_delimiter, completion_delimiter.lower()],
    )

    # Fix LLM output format error which use tuple_delimiter to seperate record instead of "\n"
    fixed_records = []
    for record in records:
        record = record.strip()
        if record is None:
            continue
        entity_records = split_string_by_multi_markers(
            record, [f"{tuple_delimiter}entity{tuple_delimiter}"]
        )
        for entity_record in entity_records:
            if not entity_record.startswith("entity") and not entity_record.startswith(
                    "relation"
            ):
                entity_record = f"entity<|{entity_record}"
            entity_relation_records = split_string_by_multi_markers(
                # treat "relationship" and "relation" interchangeable
                entity_record,
                [
                    f"{tuple_delimiter}relationship{tuple_delimiter}",
                    f"{tuple_delimiter}relation{tuple_delimiter}",
                ],
            )
            for entity_relation_record in entity_relation_records:
                if not entity_relation_record.startswith(
                        "entity"
                ) and not entity_relation_record.startswith("relation"):
                    entity_relation_record = (
                        f"relation{tuple_delimiter}{entity_relation_record}"
                    )
                fixed_records = fixed_records + [entity_relation_record]

    if len(fixed_records) != len(records):
        logger.warning(
            f"{chunk_key}: LLM output format error; find LLM use {tuple_delimiter} as record seperators instead new-line"
        )

    for record in fixed_records:
        record = record.strip()
        if record is None:
            continue

        # Fix various forms of tuple_delimiter corruption from the LLM output using the dedicated function
        delimiter_core = tuple_delimiter[2:-2]  # Extract "#" from "<|#|>"
        record = fix_tuple_delimiter_corruption(record, delimiter_core, tuple_delimiter)
        if delimiter_core != delimiter_core.lower():
            # change delimiter_core to lower case, and fix again
            delimiter_core = delimiter_core.lower()
            record = fix_tuple_delimiter_corruption(
                record, delimiter_core, tuple_delimiter
            )

        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

        # Try to parse as entity
        entity_data = await _handle_single_entity_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if entity_data is not None:
            truncated_name = _truncate_entity_identifier(
                entity_data["entity_name"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Entity name",
            )
            entity_data["entity_name"] = truncated_name
            maybe_nodes[truncated_name].append(entity_data)
            continue

        # Try to parse as relationship
        relationship_data = await _handle_single_relationship_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if relationship_data is not None:
            truncated_source = _truncate_entity_identifier(
                relationship_data["src_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            truncated_target = _truncate_entity_identifier(
                relationship_data["tgt_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            relationship_data["src_id"] = truncated_source
            relationship_data["tgt_id"] = truncated_target
            maybe_edges[(truncated_source, truncated_target)].append(relationship_data)

    return dict(maybe_nodes), dict(maybe_edges)


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


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


def _is_reference_only_value_text(text: str, stand_type: str | None = None) -> bool:
    merged = str(text or "").strip().lower()
    if not merged:
        return False
    ref_markers = ("规定", "依据", "按照", "见")
    has_ref_marker = any(marker in merged for marker in ref_markers)
    normalized_stand_type = _normalize_operate_standard_type(stand_type)
    if normalized_stand_type == "DLT":
        has_standard_ref = bool(
            re.search(r"(dlt/?t|iec|iectr|t\d{4,}|第\d+(?:\.\d+)*[章节条款])", merged)
        )
    else:
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
            "极间",
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


def _parameter_quality_score(node_data: dict, stand_type: str | None = None) -> tuple[int, int, int, int]:
    value_text = str(node_data.get("value_text", "") or "").strip()
    value_expr = str(node_data.get("value_expr", "") or "").strip()
    unit = str(node_data.get("unit", "") or "").strip()
    source_rank = _value_source_rank(str(node_data.get("value_source", "") or ""))
    merged_text = f"{value_text} {value_expr}".strip()
    has_measurable_detail = int(
        bool(re.search(r"\d", merged_text))
        or any(token in merged_text for token in ("%", "×", "*", "√", "/", "kV", "kA", "Hz", "ms", "min"))
    )
    reference_penalty = -1 if _is_reference_only_value_text(merged_text, stand_type=stand_type) else 0
    specificity = min(len(merged_text), 80) if has_measurable_detail else 0
    has_unit = int(bool(unit))
    return (source_rank, has_measurable_detail, reference_penalty, has_unit, specificity)


def _prefer_incoming_parameter(existing: dict, incoming: dict, stand_type: str | None = None) -> bool:
    existing_score = _parameter_quality_score(existing, stand_type=stand_type)
    incoming_score = _parameter_quality_score(incoming, stand_type=stand_type)
    if incoming_score > existing_score:
        return True
    if incoming_score < existing_score:
        return False

    # Tie-breaker: prefer incoming only if it brings explicit non-empty value fields.
    existing_value = str(existing.get("value_text", "") or "").strip()
    incoming_value = str(incoming.get("value_text", "") or "").strip()
    if not existing_value and incoming_value:
        return True
    if existing_value and not incoming_value:
        return False
    return False


def _merge_node_data_with_human_override(
        existing: dict | None, incoming: dict, stand_type: str | None = None
) -> dict:
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
        prefer_incoming = _prefer_incoming_parameter(existing, incoming, stand_type=stand_type)
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


def _merge_edge_data_with_human_override(
        existing: dict | None, incoming: dict
) -> dict:
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


def _validate_controlled_payload(data: dict, chunk_text: str, chunk_meta: dict) -> dict:
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
        # Common/benign case: model spills test_item-level keys to root while still
        # providing all required root fields. Normalize silently to avoid log spam.
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
        stand_type: str | None = None
) -> tuple[list[tuple[str, dict]], list[tuple[str, str, dict]]]:
    schema_cfg = schema_cfg or {}
    stand_type = _normalize_operate_standard_type(stand_type)
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
    scoped_ids_raw = schema_cfg.get("use_scoped_ids", False)
    use_scoped_ids = (
        str(scoped_ids_raw).strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(scoped_ids_raw, str)
        else bool(scoped_ids_raw)
    )
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
    annotation_guardrail_only_override_raw = schema_cfg.get(
        "annotation_guardrail_only_override", False
    )
    annotation_guardrail_only_override = (
        str(annotation_guardrail_only_override_raw).strip().lower()
        in {"1", "true", "yes", "on"}
        if isinstance(annotation_guardrail_only_override_raw, str)
        else bool(annotation_guardrail_only_override_raw)
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

    def _is_placeholder_param_value(param: dict, stand_type: str | None = None) -> bool:
        value_text = str(param.get("value_text", "") or "").strip().lower()
        value_expr = str(param.get("value_expr", "") or "").strip().lower()
        value_type = str(param.get("value_type", "") or "").strip().lower()

        if value_type == "missing":
            return True

        merged = f"{value_text} {value_expr}".strip()
        if not merged:
            return True

        placeholders = (
            "missing",
            "未明确数值",
            "应符合标准要求",
            "按标准执行",
            "见标准",
        )
        if any(p in merged for p in placeholders):
            return True

        # formula类型参数通常包含计算表达式，不应被过滤
        if value_source == "formula":
            return False

        # Drop reference-only texts (e.g. "按 GB/Txxxx 规定", "T11022-2011规定")
        # when they do not contain executable values/defaults/enums.
        ref_markers = ("规定", "依据", "按照", "见")
        has_ref_marker = any(marker in merged for marker in ref_markers)
        if stand_type == "DLT":
            has_standard_ref = bool(
                re.search(r"(dlt/?t|iec|iectr|t\d{4,}|第\d+(?:\.\d+)*[章节条款])", merged)
            )
        else:
            has_standard_ref = bool(
                re.search(r"(gb/?t|iec|iectr|t\d{4,}|第\d+(?:\.\d+)*[章节条款])", merged)
            )
        if has_ref_marker and has_standard_ref:
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
                    "极间",
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
            if not has_numeric_value and not has_actionable_text:
                return True

        return False

    def _param_name_signature(text: str) -> str:
        normalized = _normalize_text_key(str(text))
        normalized = normalized.replace("（", "(").replace("）", ")")
        normalized = re.sub(r"\((?:ka|kv|a|v|hz|min|ms)\)", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"(?:ka|kv|hz|min|ms|mm|mpa|kpa|pa|a|v)$", "", normalized, flags=re.IGNORECASE)
        return normalized

    def _align_param_name_with_requirements(
            raw_param_name: str, requirement_names: list[str]
    ) -> str:
        param_name = str(raw_param_name or "").strip()
        if not param_name or not requirement_names:
            return param_name
        normalized_name = _normalize_text_key(param_name)
        for requirement_name in requirement_names:
            if normalized_name == _normalize_text_key(str(requirement_name)):
                return str(requirement_name)
        signature = _param_name_signature(param_name)
        if not signature:
            return param_name
        matched = [
            str(requirement_name)
            for requirement_name in requirement_names
            if _param_name_signature(requirement_name) == signature
        ]
        if len(matched) == 1:
            return matched[0]
        return param_name

    def _is_noisy_param_value_text(value_text: str) -> bool:
        text = str(value_text or "").strip()
        if not text:
            return False
        if "\n" in text:
            return True
        # 不再过滤长文本，因为公式类型参数通常包含较长表达式
        # 只过滤明显是条款叙述泄漏的文本
        if any(token in text for token in ("适用于", "规定了", "应装设", "本标准", "本规范")):
            return True
        return False

    std_id = payload["standard"].get("std_id", "")
    std_name = payload["standard"].get("std_name", "")
    clause_id = payload["clause"].get("clause_id", "")
    clause_title = payload["clause"].get("clause_title", "")
    chunk_id = payload["clause"].get("chunk_id", "") or chunk_meta["chunk_id"]

    evidence = [{"std_id": std_id, "clause_id": clause_id, "chunk_id": chunk_id}]

    def _extract_numeric_ids(raw_text: str) -> list[int]:
        text = str(raw_text or "")
        if not text:
            return []
        ids: list[int] = []
        for match in re.finditer(r"(\d+)\s*(?:~|～|-|—|至|到)\s*(\d+)", text):
            start = int(match.group(1))
            end = int(match.group(2))
            if start <= end and end - start <= 50:
                ids.extend(list(range(start, end + 1)))
        # Remove matched ranges to avoid duplicate single-number extraction.
        text = re.sub(r"(\d+)\s*(?:~|～|-|—|至|到)\s*(\d+)", " ", text)
        ids.extend(int(v) for v in re.findall(r"\b\d+\b", text))
        deduped = sorted(set(ids))
        return deduped[:64]

    def _extract_counting_evidence_rules(content: str, test_item: str) -> list[dict[str, Any]]:
        text = str(content or "")
        if not text:
            return []
        lowered = text.lower()
        has_table10 = "表10" in text or "table 10" in lowered or "table10" in lowered
        has_table13 = "表13" in text or "table 13" in lowered or "table13" in lowered

        close_ids: list[int] = []
        open_ids: list[int] = []
        fracture_ids: list[int] = []

        close_match = re.search(r"合闸(?:条件)?(?:编号)?[：: ]*([^\n；;。]+)", text)
        open_match = re.search(r"分闸(?:条件)?(?:编号)?[：: ]*([^\n；;。]+)", text)
        fracture_match = re.search(r"断口(?:条件)?(?:编号)?[：: ]*([^\n；;。]+)", text)
        if close_match:
            close_ids = _extract_numeric_ids(close_match.group(1))
        if open_match:
            open_ids = _extract_numeric_ids(open_match.group(1))
        if fracture_match:
            fracture_ids = _extract_numeric_ids(fracture_match.group(1))

        if not close_ids:
            range_close = re.search(r"(\d+\s*(?:~|～|-|—|至|到)\s*\d+)\s*[^。\n]*合闸", text)
            if range_close:
                close_ids = _extract_numeric_ids(range_close.group(1))
        if not open_ids:
            range_open = re.search(r"(\d+\s*(?:~|～|-|—|至|到)\s*\d+)\s*[^。\n]*分闸", text)
            if range_open:
                open_ids = _extract_numeric_ids(range_open.group(1))

        rules: list[dict[str, Any]] = []
        if has_table10 and (close_ids or open_ids):
            close_desc = ",".join(str(v) for v in close_ids) if close_ids else "未提取"
            open_desc = ",".join(str(v) for v in open_ids) if open_ids else "未提取"
            rules.append(
                {
                    "rule_id": "count-evidence-table10",
                    "rule_name": "表10计数证据",
                    "rule_type": "table",
                    "report_type": "",
                    "condition": f"表10；合闸编号[{close_desc}]；分闸编号[{open_desc}]",
                    "expression": f"close_rows={len(close_ids)}; open_rows={len(open_ids)}",
                    "inputs": [{"name": "表10", "source": "table"}],
                    "outputs": [
                        {
                            "param_key": "test_count",
                            "value_expr": f"close_rows={len(close_ids)}; open_rows={len(open_ids)}",
                            "unit": "次",
                        }
                    ],
                    "target_param_key": "test_count",
                    "confidence": 0.85,
                }
            )
        if has_table13 and fracture_ids:
            fracture_desc = ",".join(str(v) for v in fracture_ids)
            rules.append(
                {
                    "rule_id": "count-evidence-table13",
                    "rule_name": "表13断口计数证据",
                    "rule_type": "table",
                    "report_type": "",
                    "condition": f"表13；断口编号[{fracture_desc}]",
                    "expression": f"fracture_rows={len(fracture_ids)}",
                    "inputs": [{"name": "表13", "source": "table"}],
                    "outputs": [
                        {
                            "param_key": "test_count",
                            "value_expr": f"fracture_rows={len(fracture_ids)}",
                            "unit": "次",
                        }
                    ],
                    "target_param_key": "test_count",
                    "confidence": 0.85,
                }
            )
        # Only attach to insulation withstand projects to avoid polluting unrelated tests.
        if test_item not in {"工频耐受电压试验", "雷电冲击耐受电压试验"}:
            return []
        return rules

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
        "evidence": _json_dumps_compact(evidence),
        "source_id": chunk_id,
        "file_path": file_path,
        "human_override": False,
    }
    nodes.append((clause_node_id, clause_node))

    equipment_nodes = {}
    for item in payload["equipment"]:
        name = item.get("equipment_name", "")
        if not name:
            continue
        node_id = _stable_equipment_id(name)
        equipment_nodes[name] = node_id
        nodes.append(
            (
                node_id,
                {
                    "entity_id": node_id,
                    "entity_type": "Equipment",
                    "name": name,
                    "equipment_name": name,
                    "aliases": _json_dumps_compact(item.get("aliases", [])),
                    "scope_condition": item.get("scope_condition", ""),
                    "evidence": _json_dumps_compact(evidence),
                    "source_id": chunk_id,
                    "file_path": file_path,
                    "human_override": False,
                },
            )
        )

    report_nodes = {}
    for item in payload["report_types"]:
        name = item.get("report_type", "")
        name = report_aliases.get(name, name)
        if not name:
            continue
        node_id = _stable_report_id(name)
        report_nodes[name] = node_id
        nodes.append(
            (
                node_id,
                {
                    "entity_id": node_id,
                    "entity_type": "ReportType",
                    "name": name,
                    "report_type": name,
                    "aliases": _json_dumps_compact(item.get("aliases", [])),
                    "confidence": float(item.get("confidence", 0.0)),
                    "evidence": _json_dumps_compact(evidence),
                    "source_id": chunk_id,
                    "file_path": file_path,
                    "human_override": False,
                },
            )
        )

    for item in payload["equipment_reports"]:
        equip_name = item.get("equipment_name", "")
        report_name = report_aliases.get(item.get("report_type", ""), item.get("report_type", ""))
        if not equip_name or not report_name:
            continue
        equip_id = equipment_nodes.get(equip_name, _stable_equipment_id(equip_name))
        report_id = report_nodes.get(report_name, _stable_report_id(report_name))
        edges.append(
            (
                equip_id,
                report_id,
                {
                    "src_id": equip_id,
                    "tgt_id": report_id,
                    "rel_type": "REQUIRES_REPORT",
                    "condition": item.get("condition", ""),
                    "confidence": float(item.get("confidence", 0.0)),
                    "weight": float(item.get("confidence", 0.0)) or 1.0,
                    "evidence": _json_dumps_compact(evidence),
                    "source_id": chunk_id,
                    "file_path": file_path,
                    "human_override": False,
                },
            )
        )

    payload_report_names = {
        _normalize_text_key(
            report_aliases.get(item.get("report_type", ""), item.get("report_type", ""))
        )
        for item in payload.get("report_types", [])
        if isinstance(item, dict)
    }
    payload_category_names = {
        _normalize_text_key(str(item.get("category", "") or ""))
        for item in payload.get("test_items", [])
        if isinstance(item, dict) and _normalize_text_key(str(item.get("category", "") or ""))
    }

    # 查找 tests_by_name 规则,补全试验的
    def _match_override_rule(item: dict[str, Any]) -> dict[str, Any]:
        raw_test_name = item.get("test_item", "")
        test_name = test_aliases.get(raw_test_name, raw_test_name)
        test_key = _normalize_text_key(test_name)
        if not test_key:
            return {}

        category = str(item.get("category", "") or "").strip()
        category_key = _normalize_text_key(category)

        report_keys: list[str] = []
        required_reports = item.get("required_reports", []) or []
        if isinstance(required_reports, list):
            for req in required_reports:
                if not isinstance(req, dict):
                    continue
                report_name = report_aliases.get(
                    req.get("report_type", ""), req.get("report_type", "")
                )
                report_key = _normalize_text_key(str(report_name or ""))
                if report_key:
                    report_keys.append(report_key)

        candidates = tree_tests_by_name.get(test_key, [])
        if not isinstance(candidates, list) or not candidates:
            return {}
        if len(candidates) == 1:
            return candidates[0]

        if category_key:
            category_matched = [
                cand
                for cand in candidates
                if _normalize_text_key(str(cand.get("category", "") or "")) == category_key
            ]
            if len(category_matched) == 1:
                return category_matched[0]
            if category_matched:
                candidates = category_matched

        if report_keys:
            report_matched = [
                cand
                for cand in candidates
                if _normalize_text_key(str(cand.get("report_type", "") or "")) in report_keys
            ]
            if len(report_matched) == 1:
                return report_matched[0]
            if report_matched:
                candidates = report_matched

        if len(candidates) > 1:
            # Collapse effectively-equivalent candidates produced from multiple path aliases.
            deduped: list[dict[str, Any]] = []
            seen_signatures: set[tuple[str, str, str, tuple[str, ...]]] = set()
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                param_names = tuple(
                    sorted(
                        {
                            _normalize_text_key(str(p.get("param_name", "") or ""))
                            for p in (cand.get("parameters", []) or [])
                            if isinstance(p, dict)
                               and _normalize_text_key(str(p.get("param_name", "") or ""))
                        }
                    )
                )
                signature = (
                    _normalize_text_key(str(cand.get("report_type", "") or "")),
                    _normalize_text_key(str(cand.get("category", "") or "")),
                    _normalize_text_key(str(cand.get("test_name", "") or "")),
                    param_names,
                )
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                deduped.append(cand)
            if deduped:
                candidates = deduped

        if len(candidates) > 1:
            # Prefer richer rule templates when ambiguity remains.
            scored = sorted(
                candidates,
                key=lambda cand: (
                    len(cand.get("parameters", []) or []),
                    len(cand.get("rules", []) or []),
                ),
                reverse=True,
            )
            if (
                    len(scored) >= 2
                    and (
                    len(scored[0].get("parameters", []) or []),
                    len(scored[0].get("rules", []) or []),
            )
                    > (
                    len(scored[1].get("parameters", []) or []),
                    len(scored[1].get("rules", []) or []),
            )
            ):
                return scored[0]

        if strict_override_match and len(candidates) > 1:
            logger.debug(
                "Ambiguous override match in strict mode, drop candidate: %s (count=%d)",
                test_name,
                len(candidates),
            )
            return {}

        logger.debug(
            "Ambiguous test override by name, falling back to first candidate: %s (count=%d)",
            test_name,
            len(candidates),
        )
        return candidates[0]

    effective_test_items = list(payload.get("test_items", []))
    existing_item_keys = set()
    for item in effective_test_items:
        if not isinstance(item, dict):
            continue
        raw_test_name = item.get("test_item", "")
        test_name = test_aliases.get(raw_test_name, raw_test_name)
        category_key = _normalize_text_key(str(item.get("category", "") or ""))
        existing_item_keys.add(f"{category_key}::{_normalize_text_key(test_name)}")

    for add_rule in tree_add_test_items:
        if not isinstance(add_rule, dict):
            continue
        test_name = str(add_rule.get("test_item", "") or add_rule.get("test_name", "") or "").strip()
        if not test_name:
            continue
        category = str(add_rule.get("category", "") or "").strip()
        report_scope = _normalize_text_key(str(add_rule.get("report_type", "") or ""))
        if report_scope and payload_report_names and report_scope not in payload_report_names:
            # Allow report-level additions (e.g. "型式试验") to be injected when
            # the current payload only carries category-level report scopes.
            category_scope = _normalize_text_key(category)
            if (
                    not category_scope
                    or (
                    category_scope not in payload_report_names
                    and category_scope not in payload_category_names
            )
            ):
                continue
        item_key = f"{_normalize_text_key(category)}::{_normalize_text_key(test_name)}"
        if item_key in existing_item_keys:
            continue
        required_reports = add_rule.get("required_reports", [])
        if not isinstance(required_reports, list):
            required_reports = []
        if not required_reports and add_rule.get("report_type"):
            required_reports = [
                {
                    "report_type": add_rule.get("report_type", ""),
                    "is_required": True,
                    "condition": "",
                }
            ]
        effective_test_items.append(
            {
                "test_item": test_name,
                "category": category,
                "__manual_add": True,
                "aliases": add_rule.get("aliases", []),
                "acceptance_criteria": add_rule.get("acceptance_criteria", ""),
                "notes": add_rule.get("note", ""),
                "confidence": float(add_rule.get("confidence", 1.0)),
                "required_reports": required_reports,
                "parameters": add_rule.get("parameters", []),
                "rules": add_rule.get("rules", []),
                "evidence": {},
            }
        )
        existing_item_keys.add(item_key)
        logger.debug("Inject manual test item from override: %s", test_name)

    for item in effective_test_items:
        if not isinstance(item, dict):
            continue
        test_name = item.get("test_item", "")
        test_name = test_aliases.get(test_name, test_name)
        if not test_name:
            continue
        test_name_key = _normalize_test_item_lookup_key(test_name)
        if allowed_test_item_keys and test_name_key not in allowed_test_item_keys:
            logger.debug("Drop out-of-schema test item: %s", test_name)
            continue
        override_rule = _match_override_rule(item)
        if (tree_tests_by_path or tree_tests_by_name) and not override_rule:
            if bool(item.get("__manual_add")):
                override_rule = {}
            elif strict_override_match:
                logger.debug("Drop test item due to missing override match: %s", test_name)
                continue
            else:
                logger.debug(
                    "No override match for test item, keep extracted item as-is: %s", test_name
                )
                override_rule = {}
        item_category_key = _normalize_text_key(str(item.get("category", "") or ""))
        rule_category_key = _normalize_text_key(str(override_rule.get("category", "") or ""))
        if item_category_key and rule_category_key and item_category_key != rule_category_key:
            logger.debug(
                "Category mismatch on matched override, trust override category: %s (item=%s, rule=%s)",
                test_name,
                item.get("category", ""),
                override_rule.get("category", ""),
            )
        if override_rule.get("skip"):
            logger.debug("Skip test item by tree override: %s", test_name)
            continue
        category = override_rule.get("category", item.get("category", ""))
        aliases = override_rule.get("aliases", item.get("aliases", []))
        acceptance_criteria = override_rule.get(
            "acceptance_criteria", item.get("acceptance_criteria", "")
        )
        required_reports = item.get("required_reports", [])
        if isinstance(override_rule.get("required_reports"), list):
            required_reports = override_rule.get("required_reports", [])
        override_report = str(override_rule.get("report_type", "") or "").strip()
        if override_report and configured_report_keys:
            override_report_key = _normalize_text_key(override_report)
            if override_report_key not in configured_report_keys:
                category_report_key = _normalize_text_key(str(category))
                if category_report_key in configured_report_keys:
                    override_report = str(category)
                else:
                    override_report = ""
        if override_report:
            required_reports = [
                {
                    "report_type": override_report,
                    "is_required": True,
                    "condition": "",
                }
            ]
        scope_key = _build_override_path_key([override_report, category, test_name])
        id_scope_key = scope_key if use_scoped_ids else ""
        test_id = _stable_test_id(test_name, id_scope_key)
        test_evidence = item.get("evidence", {}) or {}
        test_ev_list = [
            {
                "std_id": test_evidence.get("std_id", std_id),
                "clause_id": test_evidence.get("clause_id", clause_id),
                "chunk_id": test_evidence.get("chunk_id", chunk_id),
            }
        ]
        nodes.append(
            (
                test_id,
                {
                    "entity_id": test_id,
                    "entity_type": "TestItem",
                    "name": test_name,
                    "test_item": test_name,
                    "category": category,
                    "aliases": _json_dumps_compact(aliases if isinstance(aliases, list) else []),
                    "acceptance_criteria": acceptance_criteria,
                    "notes": override_rule.get("note") or item.get("notes", ""),
                    "confidence": float(item.get("confidence", 0.0)),
                    "evidence": _json_dumps_compact(test_ev_list),
                    "source_id": chunk_id,
                    "file_path": file_path,
                    "human_override": False,
                },
            )
        )

        edges.append(
            (
                test_id,
                clause_node_id,
                {
                    "src_id": test_id,
                    "tgt_id": clause_node_id,
                    "rel_type": "BASED_ON",
                    "confidence": float(item.get("confidence", 0.0)),
                    "evidence": _json_dumps_compact(test_ev_list),
                    "source_id": chunk_id,
                    "file_path": file_path,
                    "human_override": False,
                },
            )
        )

        for req in required_reports if isinstance(required_reports, list) else []:
            if not isinstance(req, dict):
                continue
            report_name = report_aliases.get(req.get("report_type", ""), req.get("report_type", ""))
            if not report_name:
                continue
            report_id = report_nodes.get(report_name, _stable_report_id(report_name))
            edges.append(
                (
                    report_id,
                    test_id,
                    {
                        "src_id": report_id,
                        "tgt_id": test_id,
                        "rel_type": "INCLUDES_TEST",
                        "is_required": bool(req.get("is_required", True)),
                        "condition": req.get("condition", ""),
                        "confidence": float(item.get("confidence", 0.0)),
                        "weight": float(item.get("confidence", 0.0)) or 1.0,
                        "evidence": _json_dumps_compact(test_ev_list),
                        "source_id": chunk_id,
                        "file_path": file_path,
                        "human_override": False,
                    },
                )
            )

        required_param_names = required_param_map.get(test_name, [])
        if not required_param_names:
            required_param_names = required_param_map_normalized.get(
                _normalize_test_item_lookup_key(test_name), []
            )
        required_param_name_keys = {
            _normalize_text_key(str(name))
            for name in required_param_names
            if _normalize_text_key(str(name))
        }  # 下面获得特征映射
        required_param_key_keys = {
            _normalize_text_key(str(param_map.get(str(name), str(name))))
            for name in required_param_names
            if _normalize_text_key(str(param_map.get(str(name), str(name))))
        }
        required_name_by_key: dict[str, str] = {}
        for req_name in required_param_names:
            req_key = _normalize_text_key(str(param_map.get(str(req_name), str(req_name))))
            if req_key and req_key not in required_name_by_key:
                required_name_by_key[req_key] = str(req_name)
        # Force canonical display name for test_count family.
        if "test_count" in required_name_by_key:
            required_name_by_key["test_count"] = (
                "正常次数" if _uses_normal_count_label(test_name) else "试验次数"
            )
        extracted_params = item.get("parameters", []) or []
        if not isinstance(extracted_params, list):
            extracted_params = []
        override_params = override_rule.get("parameters", []) or []
        if not isinstance(override_params, list):
            override_params = []
        remove_params_raw = override_rule.get("remove_parameters", []) or []
        if not isinstance(remove_params_raw, list):
            remove_params_raw = []
        remove_param_keys = set()
        for remove_name in remove_params_raw:
            normalized_remove_name = _normalize_text_key(str(remove_name))
            if normalized_remove_name:
                remove_param_keys.add(normalized_remove_name)
            mapped_remove_key = _normalize_text_key(str(param_map.get(str(remove_name), "")))
            if mapped_remove_key:
                remove_param_keys.add(mapped_remove_key)
        param_mode = str(override_rule.get("parameters_mode", "") or "").strip().lower()
        # In merge mode, override parameters are incremental corrections.
        # Filtering extracted parameters to override template keys would drop
        # many valid schema parameters (e.g. T10/T30/T60 only keeping a subset).
        # Keep template filtering opt-in via rule flag `template_only`.
        template_only_mode = bool(override_rule.get("template_only", False))
        override_param_keys = set()  # override_param_filter_to_template false时，不执行参数覆盖
        if override_param_filter_to_template and template_only_mode and override_params:
            for override_param in override_params:
                if not isinstance(override_param, dict):
                    continue
                override_key = _normalize_text_key(
                    str(override_param.get("param_key", "") or override_param.get("param_name", ""))
                )
                if override_key:
                    override_param_keys.add(override_key)

        def _param_identity_key(param: dict[str, Any]) -> str:
            name_part = str(param.get("param_name", "") or "")
            key_part = str(param.get("param_key", "") or "")
            normalized_key = _normalize_text_key(key_part)
            if normalized_key:
                return normalized_key
            mapped_key = _normalize_text_key(str(param_map.get(name_part, "")))
            if mapped_key:
                return mapped_key
            return _normalize_text_key(name_part)

        # 没有 找到replace 类型的
        if param_mode == "replace":
            param_candidates = [param for param in override_params if isinstance(param, dict)]
        else:
            base_params = []
            for param in extracted_params:
                if not isinstance(param, dict):
                    continue
                identity_key = _param_identity_key(param)
                if identity_key in remove_param_keys:
                    continue
                if override_param_keys and identity_key and identity_key not in override_param_keys:
                    logger.debug(
                        "Drop parameter not in override template set: test=%s param=%s",
                        test_name,
                        str(param.get("param_name", "") or param.get("param_key", "")),
                    )
                    continue
                base_params.append(dict(param))

            index_by_key: dict[str, int] = {}
            for idx, param in enumerate(base_params):
                key = _param_identity_key(param)
                if key:
                    index_by_key[key] = idx
            # 模板值 替换 模型提取值???
            for override_param in override_params:
                if not isinstance(override_param, dict):
                    continue
                key = _param_identity_key(override_param)
                if key and key in remove_param_keys:
                    continue
                if key and key in index_by_key:
                    existing_idx = index_by_key[key]
                    merged_param = dict(base_params[existing_idx])
                    merged_param.update(override_param)
                    base_params[existing_idx] = merged_param
                else:
                    base_params.append(dict(override_param))
                    if key:
                        index_by_key[key] = len(base_params) - 1

            param_candidates = base_params

        override_param_by_key: dict[str, dict[str, Any]] = {}
        for override_param in override_params:
            if not isinstance(override_param, dict):
                continue
            ov_name = str(override_param.get("param_name", "") or "")
            ov_key = str(override_param.get("param_key", "") or param_map.get(ov_name, ""))
            ov_norm_key = _normalize_text_key(ov_key or ov_name)
            if ov_norm_key in remove_param_keys:
                continue
            if ov_norm_key and ov_norm_key not in override_param_by_key:
                override_param_by_key[ov_norm_key] = dict(override_param)
        guardrail_allowed_keys = set(required_param_key_keys)
        if annotation_guardrail_only_override and override_param_by_key:
            guardrail_allowed_keys = set(override_param_by_key.keys())
        elif annotation_guardrail_mode and override_param_by_key:
            guardrail_allowed_keys.update(override_param_by_key.keys())

        seen_param_keys: set[str] = set()
        for param in param_candidates:
            raw_param_name = str(param.get("param_name", "") or "")
            param_note_text = str(param.get("constraints", "") or "") or str(
                param.get("value_expr", "") or ""
            )
            param_name = _resolve_override_param_name(raw_param_name, param_note_text)
            param_name = _align_param_name_with_requirements(param_name, required_param_names)
            param_key = param.get("param_key", "") or param_map.get(param_name, "")
            if not param_key and param_name in required_param_names:
                param_key = param_name
            if not param_key and param_name:
                param_key = _normalize_text_key(param_name)
            if not param_key:
                continue
            normalized_param_key = _normalize_text_key(param_key)
            if (
                    annotation_guardrail_mode
                    and guardrail_allowed_keys
                    and normalized_param_key not in guardrail_allowed_keys
            ):
                logger.debug(
                    "Drop parameter by annotation guardrail: test=%s param_name=%s param_key=%s",
                    test_name,
                    param_name,
                    param_key,
                )
                continue
            if annotation_guardrail_mode:
                guardrail_param = override_param_by_key.get(normalized_param_key)
                if guardrail_param:
                    merged_guardrail_param = dict(param)
                    merged_guardrail_param.update(guardrail_param)
                    param = merged_guardrail_param
                    raw_param_name = str(param.get("param_name", "") or "")
                    param_note_text = str(param.get("constraints", "") or "") or str(
                        param.get("value_expr", "") or ""
                    )
                    param_name = _resolve_override_param_name(raw_param_name, param_note_text)
                    param_name = _align_param_name_with_requirements(
                        param_name, required_param_names
                    )
                    param_key = param.get("param_key", "") or param_map.get(param_name, "")
                    if not param_key and param_name:
                        param_key = _normalize_text_key(param_name)
                    normalized_param_key = _normalize_text_key(param_key)
            if normalized_param_key in required_name_by_key:
                # Canonicalize parameter names to project-defined whitelist labels.
                param_name = required_name_by_key[normalized_param_key]
            if normalized_param_key == "test_count":
                param_name = "正常次数" if _uses_normal_count_label(test_name) else "试验次数"
            if enforce_param_whitelist and not required_param_names:
                logger.debug(
                    "Drop parameter because whitelist is enabled but no requirements found: test=%s param=%s",
                    test_name,
                    param_name or param_key,
                )
                continue
            if enforce_param_whitelist and required_param_names:
                normalized_param_name = _normalize_text_key(param_name)
                normalized_param_key = _normalize_text_key(param_key)
                if (
                        normalized_param_name not in required_param_name_keys
                        and normalized_param_key not in required_param_key_keys
                ):
                    logger.debug(
                        "Drop non-whitelist parameter: test=%s param_name=%s param_key=%s",
                        test_name,
                        param_name,
                        param_key,
                    )
                    continue
            if normalized_param_key in seen_param_keys:
                logger.debug(
                    "Drop duplicate parameter by key: test=%s param_key=%s",
                    test_name,
                    param_key,
                )
                continue
            seen_param_keys.add(normalized_param_key)
            if _is_placeholder_param_value(param, stand_type):  # 缺失则直接跳过
                continue
            value_source = param.get("value_source", "")
            if not value_source:
                value_source = _infer_value_source(
                    str(param.get("value_text", "") or ""),
                    str(param.get("constraints", "") or ""),
                )
            resolved_value_text = _resolve_override_value_text(
                str(param.get("value_text", "") or ""),
                param_note_text,
                value_source,
            )
            resolved_value_text = _expand_shorthand_param_value(
                resolved_value_text,
                param_name or param_key,
                override_rule,
                tree_tests_by_path,
            )
            resolved_value_text = _sanitize_value_text(resolved_value_text)
            if _is_noisy_param_value_text(resolved_value_text):
                logger.debug(
                    "Drop noisy parameter value text: test=%s param=%s value=%s",
                    test_name,
                    param_name or param_key,
                    resolved_value_text[:80],
                )
                continue
            resolved_value_expr = _sanitize_value_text(str(param.get("value_expr", "") or ""))
            param_id = _stable_param_id(test_name, param_key, id_scope_key)
            nodes.append(
                (
                    param_id,
                    {
                        "entity_id": param_id,
                        "entity_type": "TestParameter",
                        "name": param_name or param_key,
                        "param_name": param_name,
                        "param_key": param_key,
                        "value_text": resolved_value_text,
                        "value_expr": resolved_value_expr,
                        "unit": param.get("unit", ""),
                        "value_type": param.get("value_type", ""),
                        "value_source": value_source,
                        "constraints": param.get("constraints", ""),
                        "calc_rule": param.get("calc_rule", ""),
                        "table_ref": param.get("table_ref", ""),
                        "derive_from_rated": param.get("derive_from_rated", ""),
                        "evidence": _json_dumps_compact(test_ev_list),
                        "source_id": chunk_id,
                        "file_path": file_path,
                        "human_override": False,
                    },
                )
            )
            edges.append(
                (
                    test_id,
                    param_id,
                    {
                        "src_id": test_id,
                        "tgt_id": param_id,
                        "rel_type": "HAS_PARAMETER",
                        "confidence": float(item.get("confidence", 0.0)),
                        "weight": float(item.get("confidence", 0.0)) or 1.0,
                        "evidence": _json_dumps_compact(test_ev_list),
                        "source_id": chunk_id,
                        "file_path": file_path,
                        "human_override": False,
                    },
                )
            )
            edges.append(
                (
                    param_id,
                    clause_node_id,
                    {
                        "src_id": param_id,
                        "tgt_id": clause_node_id,
                        "rel_type": "BASED_ON",
                        "confidence": float(item.get("confidence", 0.0)),
                        "weight": float(item.get("confidence", 0.0)) or 1.0,
                        "evidence": _json_dumps_compact(test_ev_list),
                        "source_id": chunk_id,
                        "file_path": file_path,
                        "human_override": False,
                    },
                )
            )

        # Fill missing required parameters from curated override templates.
        if required_name_by_key and override_param_by_key:
            missing_keys = [
                req_key
                for req_key in required_name_by_key.keys()
                if req_key not in seen_param_keys and req_key not in remove_param_keys
            ]
            for missing_key in missing_keys:
                fallback_param = override_param_by_key.get(missing_key)
                if not fallback_param:
                    continue
                fallback_name = required_name_by_key.get(missing_key, str(fallback_param.get("param_name", "")))
                fallback_key = str(
                    fallback_param.get("param_key", "") or param_map.get(fallback_name, "") or missing_key)
                fallback_value_text = _sanitize_value_text(str(fallback_param.get("value_text", "") or ""))
                fallback_value_text = _expand_shorthand_param_value(
                    fallback_value_text,
                    fallback_name or fallback_key,
                    override_rule,
                    tree_tests_by_path,
                )
                fallback_value_text = _sanitize_value_text(fallback_value_text)
                if not fallback_value_text or _is_noisy_param_value_text(fallback_value_text):
                    continue
                fallback_value_source = str(fallback_param.get("value_source", "") or "")
                if not fallback_value_source:
                    fallback_value_source = _infer_value_source(
                        fallback_value_text,
                        str(fallback_param.get("constraints", "") or ""),
                    )
                fallback_value_expr = _sanitize_value_text(str(fallback_param.get("value_expr", "") or ""))
                fallback_param_id = _stable_param_id(test_name, fallback_key, id_scope_key)
                nodes.append(
                    (
                        fallback_param_id,
                        {
                            "entity_id": fallback_param_id,
                            "entity_type": "TestParameter",
                            "name": fallback_name or fallback_key,
                            "param_name": fallback_name,
                            "param_key": fallback_key,
                            "value_text": fallback_value_text,
                            "value_expr": fallback_value_expr,
                            "unit": fallback_param.get("unit", ""),
                            "value_type": fallback_param.get("value_type", ""),
                            "value_source": fallback_value_source,
                            "constraints": fallback_param.get("constraints", ""),
                            "calc_rule": fallback_param.get("calc_rule", ""),
                            "table_ref": fallback_param.get("table_ref", ""),
                            "derive_from_rated": fallback_param.get("derive_from_rated", ""),
                            "evidence": _json_dumps_compact(test_ev_list),
                            "source_id": chunk_id,
                            "file_path": file_path,
                            "human_override": False,
                        },
                    )
                )
                edges.append(
                    (
                        test_id,
                        fallback_param_id,
                        {
                            "src_id": test_id,
                            "tgt_id": fallback_param_id,
                            "rel_type": "HAS_PARAMETER",
                            "confidence": float(item.get("confidence", 0.0)),
                            "weight": float(item.get("confidence", 0.0)) or 1.0,
                            "evidence": _json_dumps_compact(test_ev_list),
                            "source_id": chunk_id,
                            "file_path": file_path,
                            "human_override": False,
                        },
                    )
                )
                edges.append(
                    (
                        fallback_param_id,
                        clause_node_id,
                        {
                            "src_id": fallback_param_id,
                            "tgt_id": clause_node_id,
                            "rel_type": "BASED_ON",
                            "confidence": float(item.get("confidence", 0.0)),
                            "weight": float(item.get("confidence", 0.0)) or 1.0,
                            "evidence": _json_dumps_compact(test_ev_list),
                            "source_id": chunk_id,
                            "file_path": file_path,
                            "human_override": False,
                        },
                    )
                )
                seen_param_keys.add(_normalize_text_key(fallback_key))

        extracted_rules = item.get("rules", [])
        if not isinstance(extracted_rules, list):
            extracted_rules = []
        inferred_rules = _extract_counting_evidence_rules(chunk_text, test_name)
        remove_rules_raw = override_rule.get("remove_rules", []) or []
        if not isinstance(remove_rules_raw, list):
            remove_rules_raw = []
        remove_rule_markers = {
            _normalize_text_key(str(name))
            for name in remove_rules_raw
            if _normalize_text_key(str(name))
        }
        remove_rule_match_markers = {
            re.sub(r"[^\w\u4e00-\u9fff]", "", marker).lower()
            for marker in remove_rule_markers
            if marker
        }

        def _should_remove_rule(rule_obj: dict[str, Any]) -> bool:
            if not remove_rule_markers:
                return False
            merged_text = "|".join(
                [
                    str(rule_obj.get("rule_name", "") or ""),
                    str(rule_obj.get("rule_type", "") or ""),
                    str(rule_obj.get("condition", "") or ""),
                    str(rule_obj.get("expression", "") or ""),
                ]
            )
            norm_text = _normalize_text_key(merged_text)
            norm_text_match = re.sub(r"[^\w\u4e00-\u9fff]", "", norm_text).lower()
            if not norm_text:
                return False
            for marker in remove_rule_markers:
                if marker and (marker in norm_text or norm_text in marker):
                    return True
            for marker in remove_rule_match_markers:
                if marker and norm_text_match and (
                        marker in norm_text_match or norm_text_match in marker
                ):
                    return True
            return False

        merged_rules: list[dict[str, Any]] = []
        seen_rule_keys: set[str] = set()
        for rule in [*extracted_rules, *inferred_rules]:
            if not isinstance(rule, dict):
                continue
            if _should_remove_rule(rule):
                logger.debug(
                    "Drop rule by override remove_rules: test=%s rule=%s",
                    test_name,
                    str(rule.get("rule_name", "") or rule.get("rule_type", "")),
                )
                continue
            rule_marker = str(
                rule.get("rule_id")
                or f"{rule.get('rule_name', '')}|{rule.get('condition', '')}|{rule.get('expression', '')}"
            )
            if rule_marker in seen_rule_keys:
                continue
            seen_rule_keys.add(rule_marker)
            merged_rules.append(rule)

        for rule in merged_rules:
            rule_type = rule.get("rule_type", "")
            condition = rule.get("condition", "")
            expression = rule.get("expression", "")
            rule_key = rule.get("rule_id") or f"{rule_type}|{condition}|{expression}"
            rule_id = _stable_rule_id(test_name, rule_key, id_scope_key)
            nodes.append(
                (
                    rule_id,
                    {
                        "entity_id": rule_id,
                        "entity_type": "TestRule",
                        "name": rule.get("rule_name", "") or rule_type or "rule",
                        "rule_type": rule_type,
                        "condition": condition,
                        "expression": expression,
                        "description": " | ".join(v for v in [condition, expression] if v),
                        "inputs": _json_dumps_compact(rule.get("inputs", [])),
                        "outputs": _json_dumps_compact(rule.get("outputs", [])),
                        "confidence": float(rule.get("confidence", 0.0)),
                        "evidence": _json_dumps_compact(test_ev_list),
                        "source_id": chunk_id,
                        "file_path": file_path,
                        "human_override": False,
                    },
                )
            )
            edges.append(
                (
                    test_id,
                    rule_id,
                    {
                        "src_id": test_id,
                        "tgt_id": rule_id,
                        "rel_type": "HAS_RULE",
                        "confidence": float(rule.get("confidence", 0.0)),
                        "weight": float(rule.get("confidence", 0.0)) or 1.0,
                        "evidence": _json_dumps_compact(test_ev_list),
                        "source_id": chunk_id,
                        "file_path": file_path,
                        "human_override": False,
                    },
                )
            )

            target_report = report_aliases.get(
                rule.get("report_type", ""), rule.get("report_type", "")
            )
            if target_report:
                report_id = report_nodes.get(target_report, _stable_report_id(target_report))
                edges.append(
                    (
                        report_id,
                        rule_id,
                        {
                            "src_id": report_id,
                            "tgt_id": rule_id,
                            "rel_type": "GOVERNS_RULE",
                            "confidence": float(rule.get("confidence", 0.0)),
                            "weight": float(rule.get("confidence", 0.0)) or 1.0,
                            "evidence": _json_dumps_compact(test_ev_list),
                            "source_id": chunk_id,
                            "file_path": file_path,
                            "human_override": False,
                        },
                    )
                )

            target_param_key = rule.get("target_param_key", "")
            if target_param_key:
                target_param_id = _stable_param_id(test_name, target_param_key, id_scope_key)
                edges.append(
                    (
                        rule_id,
                        target_param_id,
                        {
                            "src_id": rule_id,
                            "tgt_id": target_param_id,
                            "rel_type": "TARGETS_PARAMETER",
                            "confidence": float(rule.get("confidence", 0.0)),
                            "weight": float(rule.get("confidence", 0.0)) or 1.0,
                            "evidence": _json_dumps_compact(test_ev_list),
                            "source_id": chunk_id,
                            "file_path": file_path,
                            "human_override": False,
                        },
                    )
                )
    # logger.info(f"nodes:{nodes}\n\n")
    # logger.info(f"edges:{edges}")
    return nodes, edges


async def _upsert_controlled_node(
        node_id: str,
        node_data: dict,
        knowledge_graph_inst: BaseGraphStorage,
        entity_vdb: BaseVectorStorage | None,
        entity_chunks_storage: BaseKVStorage | None,
        stand_type: str | None = None,
) -> None:
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
            return " | ".join(parts)
        if entity_type == "TestRule":
            condition = str(data.get("condition", "") or "")
            expression = str(data.get("expression", "") or "")
            return " | ".join(v for v in [name, condition, expression] if v)
        if entity_type == "TestItem":
            category = str(data.get("category", "") or "")
            notes = str(data.get("notes", "") or "")
            return " | ".join(v for v in [name, category, notes] if v)
        if entity_type == "StandardClause":
            clause_id = str(data.get("clause_id", "") or "")
            quote = str(data.get("quote", "") or "")
            return " | ".join(v for v in [clause_id, name, quote[:240]] if v)
        return name

    existing = await knowledge_graph_inst.get_node(node_id)
    merged = _merge_node_data_with_human_override(existing, node_data, stand_type)
    if not str(merged.get("description", "") or "").strip():
        merged["description"] = _compose_node_description(merged)
    await knowledge_graph_inst.upsert_node(node_id, merged)

    if entity_chunks_storage is not None:
        chunk_ids = [v for v in merged.get("source_id", "").split(GRAPH_FIELD_SEP) if v]
        if chunk_ids:
            await entity_chunks_storage.upsert(
                {node_id: {"chunk_ids": chunk_ids, "count": len(chunk_ids)}}
            )

    if entity_vdb is not None:
        description = (
                str(merged.get("description", "") or "").strip()
                or str(merged.get("name", "") or "")
                or str(merged.get("test_item", "") or "")
        )
        content = f"{node_id}\n{description}"
        entity_vdb_id = compute_mdhash_id(str(node_id), prefix="ent-")
        data_for_vdb = {
            entity_vdb_id: {
                "entity_name": node_id,
                "entity_type": merged.get("entity_type", ""),
                "content": content,
                "source_id": merged.get("source_id", ""),
                "file_path": merged.get("file_path", "unknown_source"),
            }
        }
        await safe_vdb_operation_with_exception(
            operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload),
            operation_name="entity_upsert",
            entity_name=node_id,
            max_retries=3,
            retry_delay=0.1,
        )


async def _upsert_controlled_edge(
        src_id: str,
        tgt_id: str,
        edge_data: dict,
        knowledge_graph_inst: BaseGraphStorage,
        relationships_vdb: BaseVectorStorage | None,
        relation_chunks_storage: BaseKVStorage | None,
) -> None:
    def _compose_edge_description(data: dict) -> str:
        rel_type = str(data.get("rel_type", "") or "")
        condition = str(data.get("condition", "") or "")
        expression = str(data.get("expression", "") or "")
        extras = [rel_type, condition, expression]
        return " | ".join(v for v in extras if v)

    existing = await knowledge_graph_inst.get_edge(src_id, tgt_id)
    merged = _merge_edge_data_with_human_override(existing, edge_data)
    if not str(merged.get("description", "") or "").strip():
        merged["description"] = _compose_edge_description(merged)
    await knowledge_graph_inst.upsert_edge(src_id, tgt_id, merged)

    if relation_chunks_storage is not None:
        storage_key = make_relation_chunk_key(src_id, tgt_id)
        chunk_ids = [v for v in merged.get("source_id", "").split(GRAPH_FIELD_SEP) if v]
        if chunk_ids:
            await relation_chunks_storage.upsert(
                {storage_key: {"chunk_ids": chunk_ids, "count": len(chunk_ids)}}
            )

    if relationships_vdb is not None:
        rel_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")
        keywords = merged.get("rel_type", "")
        description = str(merged.get("description", "") or merged.get("rel_type", ""))
        content = f"{src_id}\t{tgt_id}\n{keywords}\n{description}"
        data_for_vdb = {
            rel_id: {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "keywords": keywords,
                "description": description,
                "content": content,
                "source_id": merged.get("source_id", ""),
                "file_path": merged.get("file_path", "unknown_source"),
            }
        }
        await safe_vdb_operation_with_exception(
            operation=lambda payload=data_for_vdb: relationships_vdb.upsert(payload),
            operation_name="relationship_upsert",
            entity_name=f"{src_id}~{tgt_id}",
            max_retries=3,
            retry_delay=0.1,
        )


async def _rebuild_from_extraction_result(
        text_chunks_storage: BaseKVStorage,
        extraction_result: str,
        chunk_id: str,
        timestamp: int,
) -> tuple[dict, dict]:
    """Parse cached extraction result using the same logic as extract_entities

    Args:
        text_chunks_storage: Text chunks storage to get chunk data
        extraction_result: The cached LLM extraction result
        chunk_id: The chunk ID for source tracking

    Returns:
        Tuple of (entities_dict, relationships_dict)
    """

    # Get chunk data for file_path from storage
    chunk_data = await text_chunks_storage.get_by_id(chunk_id)
    file_path = (
        chunk_data.get("file_path", "unknown_source")
        if chunk_data
        else "unknown_source"
    )

    # Call the shared processing function
    return await _process_extraction_result(
        extraction_result,
        chunk_id,
        timestamp,
        file_path,
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    )


async def _rebuild_single_entity(
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        entity_name: str,
        chunk_ids: list[str],
        chunk_entities: dict,
        llm_response_cache: BaseKVStorage,
        global_config: dict[str, str],
        entity_chunks_storage: BaseKVStorage | None = None,
        pipeline_status: dict | None = None,
        pipeline_status_lock=None,
) -> None:
    """Rebuild a single entity from cached extraction results"""

    # Get current entity data
    current_entity = await knowledge_graph_inst.get_node(entity_name)
    if not current_entity:
        return

    # Helper function to update entity in both graph and vector storage
    async def _update_entity_storage(
            final_description: str,
            entity_type: str,
            file_paths: list[str],
            source_chunk_ids: list[str],
            truncation_info: str = "",
    ):
        try:
            # Update entity in graph storage (critical path)
            updated_entity_data = {
                **current_entity,
                "description": final_description,
                "entity_type": entity_type,
                "source_id": GRAPH_FIELD_SEP.join(source_chunk_ids),
                "file_path": GRAPH_FIELD_SEP.join(file_paths)
                if file_paths
                else current_entity.get("file_path", "unknown_source"),
                "created_at": int(time.time()),
                "truncate": truncation_info,
            }
            await knowledge_graph_inst.upsert_node(entity_name, updated_entity_data)

            # Update entity in vector database (equally critical)
            entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
            entity_content = f"{entity_name}\n{final_description}"

            vdb_data = {
                entity_vdb_id: {
                    "content": entity_content,
                    "entity_name": entity_name,
                    "source_id": updated_entity_data["source_id"],
                    "description": final_description,
                    "entity_type": entity_type,
                    "file_path": updated_entity_data["file_path"],
                }
            }

            # Use safe operation wrapper - VDB failure must throw exception
            await safe_vdb_operation_with_exception(
                operation=lambda: entities_vdb.upsert(vdb_data),
                operation_name="rebuild_entity_upsert",
                entity_name=entity_name,
                max_retries=3,
                retry_delay=0.1,
            )

        except Exception as e:
            error_msg = f"Failed to update entity storage for `{entity_name}`: {e}"
            logger.error(error_msg)
            raise  # Re-raise exception

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    normalized_chunk_ids = chunk_ids

    if entity_chunks_storage is not None and normalized_chunk_ids:
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
            global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_entity"],
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # Collect all entity data from relevant (limited) chunks
    all_entity_data = []
    for chunk_id in limited_chunk_ids:
        if chunk_id in chunk_entities and entity_name in chunk_entities[chunk_id]:
            all_entity_data.extend(chunk_entities[chunk_id][entity_name])

    if not all_entity_data:
        logger.warning(
            f"No entity data found for `{entity_name}`, trying to rebuild from relationships"
        )

        # Get all edges connected to this entity
        edges = await knowledge_graph_inst.get_node_edges(entity_name)
        if not edges:
            logger.warning(f"No relations attached to entity `{entity_name}`")
            return

        # Collect relationship data to extract entity information
        relationship_descriptions = []
        file_paths = set()

        # Get edge data for all connected relationships
        for src_id, tgt_id in edges:
            edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
            if edge_data:
                if edge_data.get("description"):
                    relationship_descriptions.append(edge_data["description"])

                if edge_data.get("file_path"):
                    edge_file_paths = edge_data["file_path"].split(GRAPH_FIELD_SEP)
                    file_paths.update(edge_file_paths)

        # deduplicate descriptions
        description_list = list(dict.fromkeys(relationship_descriptions))

        # Generate final description from relationships or fallback to current
        if description_list:
            final_description, _ = await _handle_entity_relation_summary(
                "Entity",
                entity_name,
                description_list,
                GRAPH_FIELD_SEP,
                global_config,
                llm_response_cache=llm_response_cache,
            )
        else:
            final_description = current_entity.get("description", "")

        entity_type = current_entity.get("entity_type", "UNKNOWN")
        await _update_entity_storage(
            final_description,
            entity_type,
            file_paths,
            limited_chunk_ids,
        )
        return

    # Process cached entity data
    descriptions = []
    entity_types = []
    file_paths_list = []
    seen_paths = set()

    for entity_data in all_entity_data:
        if entity_data.get("description"):
            descriptions.append(entity_data["description"])
        if entity_data.get("entity_type"):
            entity_types.append(entity_data["entity_type"])
        if entity_data.get("file_path"):
            file_path = entity_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply MAX_FILE_PATHS limit
    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{entity_name}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    entity_types = list(dict.fromkeys(entity_types))

    # Get most common entity type
    entity_type = (
        max(set(entity_types), key=entity_types.count)
        if entity_types
        else current_entity.get("entity_type", "UNKNOWN")
    )

    # Generate final description from entities or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Entity",
            entity_name,
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        final_description = current_entity.get("description", "")

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    await _update_entity_storage(
        final_description,
        entity_type,
        file_paths_list,
        limited_chunk_ids,
        truncation_info,
    )

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{entity_name}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    logger.info(status_message)
    # Update pipeline status
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _rebuild_single_relationship(
        knowledge_graph_inst: BaseGraphStorage,
        relationships_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage,
        src: str,
        tgt: str,
        chunk_ids: list[str],
        chunk_relationships: dict,
        llm_response_cache: BaseKVStorage,
        global_config: dict[str, str],
        relation_chunks_storage: BaseKVStorage | None = None,
        entity_chunks_storage: BaseKVStorage | None = None,
        pipeline_status: dict | None = None,
        pipeline_status_lock=None,
) -> None:
    """Rebuild a single relationship from cached extraction results

    Note: This function assumes the caller has already acquired the appropriate
    keyed lock for the relationship pair to ensure thread safety.
    """

    # Get current relationship data
    current_relationship = await knowledge_graph_inst.get_edge(src, tgt)
    if not current_relationship:
        return

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    normalized_chunk_ids = chunk_ids

    if relation_chunks_storage is not None and normalized_chunk_ids:
        storage_key = make_relation_chunk_key(src, tgt)
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
            global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )
    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_relation"],
        limit_method,
        identifier=f"`{src}`~`{tgt}`",
    )

    # Collect all relationship data from relevant chunks
    all_relationship_data = []
    for chunk_id in limited_chunk_ids:
        if chunk_id in chunk_relationships:
            # Check both (src, tgt) and (tgt, src) since relationships can be bidirectional
            for edge_key in [(src, tgt), (tgt, src)]:
                if edge_key in chunk_relationships[chunk_id]:
                    all_relationship_data.extend(
                        chunk_relationships[chunk_id][edge_key]
                    )

    if not all_relationship_data:
        logger.warning(f"No relation data found for `{src}-{tgt}`")
        return

    # Merge descriptions and keywords
    descriptions = []
    keywords = []
    weights = []
    file_paths_list = []
    seen_paths = set()

    for rel_data in all_relationship_data:
        if rel_data.get("description"):
            descriptions.append(rel_data["description"])
        if rel_data.get("keywords"):
            keywords.append(rel_data["keywords"])
        if rel_data.get("weight"):
            weights.append(rel_data["weight"])
        if rel_data.get("file_path"):
            file_path = rel_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply count limit
    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{src}`~`{tgt}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    keywords = list(dict.fromkeys(keywords))

    combined_keywords = (
        ", ".join(set(keywords))
        if keywords
        else current_relationship.get("keywords", "")
    )

    weight = sum(weights) if weights else current_relationship.get("weight", 1.0)

    # Generate final description from relations or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Relation",
            f"{src}-{tgt}",
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        # fallback to keep current(unchanged)
        final_description = current_relationship.get("description", "")

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    # Update relationship in graph storage
    updated_relationship_data = {
        **current_relationship,
        "description": final_description
        if final_description
        else current_relationship.get("description", ""),
        "keywords": combined_keywords,
        "weight": weight,
        "source_id": GRAPH_FIELD_SEP.join(limited_chunk_ids),
        "file_path": GRAPH_FIELD_SEP.join([fp for fp in file_paths_list if fp])
        if file_paths_list
        else current_relationship.get("file_path", "unknown_source"),
        "truncate": truncation_info,
    }

    # Ensure both endpoint nodes exist before writing the edge back
    # (certain storage backends require pre-existing nodes).
    node_description = (
        updated_relationship_data["description"]
        if updated_relationship_data.get("description")
        else current_relationship.get("description", "")
    )
    node_source_id = updated_relationship_data.get("source_id", "")
    node_file_path = updated_relationship_data.get("file_path", "unknown_source")

    for node_id in {src, tgt}:
        if not (await knowledge_graph_inst.has_node(node_id)):
            node_created_at = int(time.time())
            node_data = {
                "entity_id": node_id,
                "source_id": node_source_id,
                "description": node_description,
                "entity_type": "UNKNOWN",
                "file_path": node_file_path,
                "created_at": node_created_at,
                "truncate": "",
            }
            await knowledge_graph_inst.upsert_node(node_id, node_data=node_data)

            # Update entity_chunks_storage for the newly created entity
            if entity_chunks_storage is not None and limited_chunk_ids:
                await entity_chunks_storage.upsert(
                    {
                        node_id: {
                            "chunk_ids": limited_chunk_ids,
                            "count": len(limited_chunk_ids),
                        }
                    }
                )

            # Update entity_vdb for the newly created entity
            if entities_vdb is not None:
                entity_vdb_id = compute_mdhash_id(node_id, prefix="ent-")
                entity_content = f"{node_id}\n{node_description}"
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_name": node_id,
                        "source_id": node_source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": node_file_path,
                    }
                }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entities_vdb.upsert(payload),
                    operation_name="rebuild_added_entity_upsert",
                    entity_name=node_id,
                    max_retries=3,
                    retry_delay=0.1,
                )

    await knowledge_graph_inst.upsert_edge(src, tgt, updated_relationship_data)

    # Update relationship in vector database
    # Sort src and tgt to ensure consistent ordering (smaller string first)
    if src > tgt:
        src, tgt = tgt, src
    try:
        rel_vdb_id = compute_mdhash_id(src + tgt, prefix="rel-")
        rel_vdb_id_reverse = compute_mdhash_id(tgt + src, prefix="rel-")

        # Delete old vector records first (both directions to be safe)
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )

        # Insert new vector record
        rel_content = f"{combined_keywords}\t{src}\n{tgt}\n{final_description}"
        vdb_data = {
            rel_vdb_id: {
                "src_id": src,
                "tgt_id": tgt,
                "source_id": updated_relationship_data["source_id"],
                "content": rel_content,
                "keywords": combined_keywords,
                "description": final_description,
                "weight": weight,
                "file_path": updated_relationship_data["file_path"],
            }
        }

        # Use safe operation wrapper - VDB failure must throw exception
        await safe_vdb_operation_with_exception(
            operation=lambda: relationships_vdb.upsert(vdb_data),
            operation_name="rebuild_relationship_upsert",
            entity_name=f"{src}-{tgt}",
            max_retries=3,
            retry_delay=0.2,
        )

    except Exception as e:
        error_msg = f"Failed to rebuild relationship storage for `{src}-{tgt}`: {e}"
        logger.error(error_msg)
        raise  # Re-raise exception

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{src}`~`{tgt}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    # Add truncation info from apply_source_ids_limit if truncation occurred
    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f" ({limit_method}:{len(limited_chunk_ids)}/{len(normalized_chunk_ids)})"
        )
        status_message += truncation_info

    logger.info(status_message)

    # Update pipeline status
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _merge_nodes_then_upsert(
        entity_name: str,
        nodes_data: list[dict],
        knowledge_graph_inst: BaseGraphStorage,
        entity_vdb: BaseVectorStorage | None,
        global_config: dict,
        pipeline_status: dict = None,
        pipeline_status_lock=None,
        llm_response_cache: BaseKVStorage | None = None,
        entity_chunks_storage: BaseKVStorage | None = None,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    # 1. Get existing node data from knowledge graph
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(already_node["source_id"].split(GRAPH_FIELD_SEP))
        already_file_paths.extend(already_node["file_path"].split(GRAPH_FIELD_SEP))
        already_description.extend(already_node["description"].split(GRAPH_FIELD_SEP))

    new_source_ids = [dp["source_id"] for dp in nodes_data if dp.get("source_id")]

    existing_full_source_ids = []
    if entity_chunks_storage is not None:
        stored_chunks = await entity_chunks_storage.get_by_id(entity_name)
        if stored_chunks and isinstance(stored_chunks, dict):
            existing_full_source_ids = [
                chunk_id for chunk_id in stored_chunks.get("chunk_ids", []) if chunk_id
            ]

    if not existing_full_source_ids:
        existing_full_source_ids = [
            chunk_id for chunk_id in already_source_ids if chunk_id
        ]

    # 2. Merging new source ids with existing ones
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    if entity_chunks_storage is not None and full_source_ids:
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": full_source_ids,
                    "count": len(full_source_ids),
                }
            }
        )

    # 3. Finalize source_id by applying source ids limit
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_entity")
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # 4. Only keep nodes not filter by apply_source_ids_limit if limit_method is KEEP
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        allowed_source_ids = set(source_ids)
        filtered_nodes = []
        for dp in nodes_data:
            source_id = dp.get("source_id")
            # Skip descriptions sourced from chunks dropped by the limitation cap
            if (
                    source_id
                    and source_id not in allowed_source_ids
                    and source_id not in existing_full_source_ids
            ):
                continue
            filtered_nodes.append(dp)
        nodes_data = filtered_nodes
    else:  # In FIFO mode, keep all nodes - truncation happens at source_ids level only
        nodes_data = list(nodes_data)

    # 5. Check if we need to skip summary due to source_ids limit
    if (
            limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
            and len(existing_full_source_ids) >= max_source_limit
            and not nodes_data
    ):
        if already_node:
            logger.info(
                f"Skipped `{entity_name}`: KEEP old chunks {already_source_ids}/{len(full_source_ids)}"
            )
            existing_node_data = dict(already_node)
            return existing_node_data
        else:
            logger.error(f"Internal Error: already_node missing for `{entity_name}`")
            raise ValueError(
                f"Internal Error: already_node missing for `{entity_name}`"
            )

    # 6.1 Finalize source_id
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize entity type by highest count
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # 7. Deduplicate nodes by description, keeping first occurrence in the same document
    unique_nodes = {}
    for dp in nodes_data:
        desc = dp.get("description")
        if not desc:
            continue
        if desc not in unique_nodes:
            unique_nodes[desc] = dp

    # Sort description by timestamp, then by description length when timestamps are the same
    sorted_nodes = sorted(
        unique_nodes.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )
    sorted_descriptions = [dp["description"] for dp in sorted_nodes]

    # Combine already_description with sorted new sorted descriptions
    description_list = already_description + sorted_descriptions
    if not description_list:
        logger.error(f"Entity {entity_name} has no description")
        raise ValueError(f"Entity {entity_name} has no description")

    # Check for cancellation before LLM summary
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during entity summary")

    # 8. Get summary description an LLM usage status
    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        entity_name,
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS
    file_paths_list = []
    seen_paths = set()
    has_placeholder = False  # Indicating file_path has been truncated before

    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )

    # Collect from already_file_paths, excluding placeholder
    for fp in already_file_paths:
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    for dp in nodes_data:
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)

    # Apply count limit
    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )
        # Add + sign to indicate actual file count is higher
        original_count_str = (
            f"{len(file_paths_list)}+" if has_placeholder else str(len(file_paths_list))
        )

        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{entity_name}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)

    # 10.Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)
    if llm_was_used:
        status_message = f"LLMmrg: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"

    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"

    deduplicated_num = already_fragment + len(nodes_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"

    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.debug(status_message)

    # 11. Update both graph and vector db
    node_data = dict(
        entity_id=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        file_path=file_path,
        created_at=int(time.time()),
        truncate=truncation_info,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    if entity_vdb is not None:
        entity_vdb_id = compute_mdhash_id(str(entity_name), prefix="ent-")
        entity_content = f"{entity_name}\n{description}"
        data_for_vdb = {
            entity_vdb_id: {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "content": entity_content,
                "source_id": source_id,
                "file_path": file_path,
            }
        }
        await safe_vdb_operation_with_exception(
            operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload),
            operation_name="entity_upsert",
            entity_name=entity_name,
            max_retries=3,
            retry_delay=0.1,
        )
    return node_data


async def _merge_edges_then_upsert(
        src_id: str,
        tgt_id: str,
        edges_data: list[dict],
        knowledge_graph_inst: BaseGraphStorage,
        relationships_vdb: BaseVectorStorage | None,
        entity_vdb: BaseVectorStorage | None,
        global_config: dict,
        pipeline_status: dict = None,
        pipeline_status_lock=None,
        llm_response_cache: BaseKVStorage | None = None,
        added_entities: list = None,  # New parameter to track entities added during edge processing
        relation_chunks_storage: BaseKVStorage | None = None,
        entity_chunks_storage: BaseKVStorage | None = None,
):
    if src_id == tgt_id:
        return None

    already_edge = None
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    # 1. Get existing edge data from graph storage
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 1.0 if missing
            already_weights.append(already_edge.get("weight", 1.0))

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    already_edge["source_id"].split(GRAPH_FIELD_SEP)
                )

            # Get file_path with empty string default if missing or None
            if already_edge.get("file_path") is not None:
                already_file_paths.extend(
                    already_edge["file_path"].split(GRAPH_FIELD_SEP)
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.extend(
                    already_edge["description"].split(GRAPH_FIELD_SEP)
                )

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )

    new_source_ids = [dp["source_id"] for dp in edges_data if dp.get("source_id")]

    storage_key = make_relation_chunk_key(src_id, tgt_id)
    existing_full_source_ids = []
    if relation_chunks_storage is not None:
        stored_chunks = await relation_chunks_storage.get_by_id(storage_key)
        if stored_chunks and isinstance(stored_chunks, dict):
            existing_full_source_ids = [
                chunk_id for chunk_id in stored_chunks.get("chunk_ids", []) if chunk_id
            ]

    if not existing_full_source_ids:
        existing_full_source_ids = [
            chunk_id for chunk_id in already_source_ids if chunk_id
        ]

    # 2. Merge new source ids with existing ones
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    if relation_chunks_storage is not None and full_source_ids:
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": full_source_ids,
                    "count": len(full_source_ids),
                }
            }
        )

    # 3. Finalize source_id by applying source ids limit
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_relation")
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{src_id}`~`{tgt_id}`",
    )
    limit_method = (
            global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    # 4. Only keep edges with source_id in the final source_ids list if in KEEP mode
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        allowed_source_ids = set(source_ids)
        filtered_edges = []
        for dp in edges_data:
            source_id = dp.get("source_id")
            # Skip relationship fragments sourced from chunks dropped by keep oldest cap
            if (
                    source_id
                    and source_id not in allowed_source_ids
                    and source_id not in existing_full_source_ids
            ):
                continue
            filtered_edges.append(dp)
        edges_data = filtered_edges
    else:  # In FIFO mode, keep all edges - truncation happens at source_ids level only
        edges_data = list(edges_data)

    # 5. Check if we need to skip summary due to source_ids limit
    if (
            limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
            and len(existing_full_source_ids) >= max_source_limit
            and not edges_data
    ):
        if already_edge:
            logger.info(
                f"Skipped `{src_id}`~`{tgt_id}`: KEEP old chunks  {already_source_ids}/{len(full_source_ids)}"
            )
            existing_edge_data = dict(already_edge)
            return existing_edge_data
        else:
            logger.error(
                f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
            )
            raise ValueError(
                f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
            )

    # 6.1 Finalize source_id
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize weight by summing new edges and existing weights
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)

    # 6.2 Finalize keywords by merging existing and new keywords
    all_keywords = set()
    # Process already_keywords (which are comma-separated)
    for keyword_str in already_keywords:
        if keyword_str:  # Skip empty strings
            all_keywords.update(k.strip() for k in keyword_str.split(",") if k.strip())
    # Process new keywords from edges_data
    for edge in edges_data:
        if edge.get("keywords"):
            all_keywords.update(
                k.strip() for k in edge["keywords"].split(",") if k.strip()
            )
    # Join all unique keywords with commas
    keywords = ",".join(sorted(all_keywords))

    # 7. Deduplicate by description, keeping first occurrence in the same document
    unique_edges = {}
    for dp in edges_data:
        description_value = dp.get("description")
        if not description_value:
            continue
        if description_value not in unique_edges:
            unique_edges[description_value] = dp

    # Sort description by timestamp, then by description length (largest to smallest) when timestamps are the same
    sorted_edges = sorted(
        unique_edges.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )
    sorted_descriptions = [dp["description"] for dp in sorted_edges]

    # Combine already_description with sorted new descriptions
    description_list = already_description + sorted_descriptions
    if not description_list:
        logger.error(f"Relation {src_id}~{tgt_id} has no description")
        raise ValueError(f"Relation {src_id}~{tgt_id} has no description")

    # Check for cancellation before LLM summary
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during relation summary"
                )

    # 8. Get summary description an LLM usage status
    description, llm_was_used = await _handle_entity_relation_summary(
        "Relation",
        f"({src_id}, {tgt_id})",
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS limit
    file_paths_list = []
    seen_paths = set()
    has_placeholder = False  # Track if already_file_paths contains placeholder

    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )

    # Collect from already_file_paths, excluding placeholder
    for fp in already_file_paths:
        # Check if this is a placeholder record
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    for dp in edges_data:
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)

    # Apply count limit
    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )

        # Add + sign to indicate actual file count is higher
        original_count_str = (
            f"{len(file_paths_list)}+" if has_placeholder else str(len(file_paths_list))
        )

        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{src_id}`~`{tgt_id}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)

    # 10. Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)
    if llm_was_used:
        status_message = f"LLMmrg: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"

    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"

    deduplicated_num = already_fragment + len(edges_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"

    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.debug(status_message)

    # 11. Update both graph and vector db
    for need_insert_id in [src_id, tgt_id]:
        # Optimization: Use get_node instead of has_node + get_node
        existing_node = await knowledge_graph_inst.get_node(need_insert_id)

        if existing_node is None:
            # Node doesn't exist - create new node
            node_created_at = int(time.time())
            node_data = {
                "entity_id": need_insert_id,
                "source_id": source_id,
                "description": description,
                "entity_type": "UNKNOWN",
                "file_path": file_path,
                "created_at": node_created_at,
                "truncate": "",
            }
            await knowledge_graph_inst.upsert_node(need_insert_id, node_data=node_data)

            # Update entity_chunks_storage for the newly created entity
            if entity_chunks_storage is not None:
                chunk_ids = [chunk_id for chunk_id in full_source_ids if chunk_id]
                if chunk_ids:
                    await entity_chunks_storage.upsert(
                        {
                            need_insert_id: {
                                "chunk_ids": chunk_ids,
                                "count": len(chunk_ids),
                            }
                        }
                    )

            if entity_vdb is not None:
                entity_vdb_id = compute_mdhash_id(need_insert_id, prefix="ent-")
                entity_content = f"{need_insert_id}\n{description}"
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_name": need_insert_id,
                        "source_id": source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": file_path,
                    }
                }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                    operation_name="added_entity_upsert",
                    entity_name=need_insert_id,
                    max_retries=3,
                    retry_delay=0.1,
                )

            # Track entities added during edge processing
            if added_entities is not None:
                entity_data = {
                    "entity_name": need_insert_id,
                    "entity_type": "UNKNOWN",
                    "description": description,
                    "source_id": source_id,
                    "file_path": file_path,
                    "created_at": node_created_at,
                }
                added_entities.append(entity_data)
        else:
            # Node exists - update its source_ids by merging with new source_ids
            updated = False  # Track if any update occurred

            # 1. Get existing full source_ids from entity_chunks_storage
            existing_full_source_ids = []
            if entity_chunks_storage is not None:
                stored_chunks = await entity_chunks_storage.get_by_id(need_insert_id)
                if stored_chunks and isinstance(stored_chunks, dict):
                    existing_full_source_ids = [
                        chunk_id
                        for chunk_id in stored_chunks.get("chunk_ids", [])
                        if chunk_id
                    ]

            # If not in entity_chunks_storage, get from graph database
            if not existing_full_source_ids:
                if existing_node.get("source_id"):
                    existing_full_source_ids = existing_node["source_id"].split(
                        GRAPH_FIELD_SEP
                    )

            # 2. Merge with new source_ids from this relationship
            new_source_ids_from_relation = [
                chunk_id for chunk_id in source_ids if chunk_id
            ]
            merged_full_source_ids = merge_source_ids(
                existing_full_source_ids, new_source_ids_from_relation
            )

            # 3. Save merged full list to entity_chunks_storage (conditional)
            if (
                    entity_chunks_storage is not None
                    and merged_full_source_ids != existing_full_source_ids
            ):
                updated = True
                await entity_chunks_storage.upsert(
                    {
                        need_insert_id: {
                            "chunk_ids": merged_full_source_ids,
                            "count": len(merged_full_source_ids),
                        }
                    }
                )

            # 4. Apply source_ids limit for graph and vector db
            limit_method = global_config.get(
                "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
            )
            max_source_limit = global_config.get("max_source_ids_per_entity")
            limited_source_ids = apply_source_ids_limit(
                merged_full_source_ids,
                max_source_limit,
                limit_method,
                identifier=f"`{need_insert_id}`",
            )

            # 5. Update graph database and vector database with limited source_ids (conditional)
            limited_source_id_str = GRAPH_FIELD_SEP.join(limited_source_ids)

            if limited_source_id_str != existing_node.get("source_id", ""):
                updated = True
                updated_node_data = {
                    **existing_node,
                    "source_id": limited_source_id_str,
                }
                await knowledge_graph_inst.upsert_node(
                    need_insert_id, node_data=updated_node_data
                )

                # Update vector database
                if entity_vdb is not None:
                    entity_vdb_id = compute_mdhash_id(need_insert_id, prefix="ent-")
                    entity_content = (
                        f"{need_insert_id}\n{existing_node.get('description', '')}"
                    )
                    vdb_data = {
                        entity_vdb_id: {
                            "content": entity_content,
                            "entity_name": need_insert_id,
                            "source_id": limited_source_id_str,
                            "entity_type": existing_node.get("entity_type", "UNKNOWN"),
                            "file_path": existing_node.get(
                                "file_path", "unknown_source"
                            ),
                        }
                    }
                    await safe_vdb_operation_with_exception(
                        operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                        operation_name="existing_entity_update",
                        entity_name=need_insert_id,
                        max_retries=3,
                        retry_delay=0.1,
                    )

            # 6. Log once at the end if any update occurred
            if updated:
                status_message = f"Chunks appended from relation: `{need_insert_id}`"
                logger.info(status_message)
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = status_message
                        pipeline_status["history_messages"].append(status_message)

    edge_created_at = int(time.time())
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            file_path=file_path,
            created_at=edge_created_at,
            truncate=truncation_info,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
        created_at=edge_created_at,
        truncate=truncation_info,
        weight=weight,
    )

    # Sort src_id and tgt_id to ensure consistent ordering (smaller string first)
    if src_id > tgt_id:
        src_id, tgt_id = tgt_id, src_id

    if relationships_vdb is not None:
        rel_vdb_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")
        rel_vdb_id_reverse = compute_mdhash_id(tgt_id + src_id, prefix="rel-")
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )
        rel_content = f"{keywords}\t{src_id}\n{tgt_id}\n{description}"
        vdb_data = {
            rel_vdb_id: {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "source_id": source_id,
                "content": rel_content,
                "keywords": keywords,
                "description": description,
                "weight": weight,
                "file_path": file_path,
            }
        }
        await safe_vdb_operation_with_exception(
            operation=lambda payload=vdb_data: relationships_vdb.upsert(payload),
            operation_name="relationship_upsert",
            entity_name=f"{src_id}-{tgt_id}",
            max_retries=3,
            retry_delay=0.2,
        )

    return edge_data


async def merge_nodes_and_edges(
        chunk_results: list,
        knowledge_graph_inst: BaseGraphStorage,
        entity_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        global_config: dict[str, str],
        full_entities_storage: BaseKVStorage = None,
        full_relations_storage: BaseKVStorage = None,
        doc_id: str = None,
        pipeline_status: dict = None,
        pipeline_status_lock=None,
        llm_response_cache: BaseKVStorage | None = None,
        entity_chunks_storage: BaseKVStorage | None = None,
        relation_chunks_storage: BaseKVStorage | None = None,
        current_file_number: int = 0,
        total_files: int = 0,
        file_path: str = "unknown_source",
        stand_type: str | None = None
) -> None:
    """Two-phase merge: process all entities first, then all relationships

    This approach ensures data consistency by:
    1. Phase 1: Process all entities concurrently
    2. Phase 2: Process all relationships concurrently (may add missing entities)
    3. Phase 3: Update full_entities and full_relations storage with final results

    Args:
        chunk_results: List of tuples (maybe_nodes, maybe_edges) containing extracted entities and relationships
        knowledge_graph_inst: Knowledge graph storage
        entity_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration
        full_entities_storage: Storage for document entity lists
        full_relations_storage: Storage for document relation lists
        doc_id: Document ID for storage indexing
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        llm_response_cache: LLM response cache
        entity_chunks_storage: Storage tracking full chunk lists per entity
        relation_chunks_storage: Storage tracking full chunk lists per relation
        current_file_number: Current file number for logging
        total_files: Total files for logging
        file_path: File path for logging
    """

    # Check for cancellation at the start of merge
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during merge phase")

    if global_config.get("kg_schema_mode") == "electrical_controlled":
        stand_type = _normalize_operate_standard_type(
            stand_type
            or global_config.get("addon_params", {}).get("standard_type")
            or global_config.get("workspace")
        )
        log_message = f"Merging stage {current_file_number}/{total_files}: {file_path}"
        logger.info(log_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Aggregate first to avoid repeated upsert/get cycles for duplicated ids across chunks.
        aggregated_nodes: dict[str, dict] = {}
        aggregated_edges: dict[tuple[str, str], dict] = {}
        total_node_events = 0
        total_edge_events = 0

        for chunk_result in chunk_results:
            nodes = chunk_result.get("nodes", [])
            edges = chunk_result.get("edges", [])
            for node_id, node_data in nodes:
                total_node_events += 1
                existing = aggregated_nodes.get(node_id)
                if existing is None:
                    aggregated_nodes[node_id] = dict(node_data)
                else:
                    aggregated_nodes[node_id] = _merge_node_data_with_human_override(
                        existing, node_data, stand_type
                    )
            for src_id, tgt_id, edge_data in edges:
                total_edge_events += 1
                edge_key = (src_id, tgt_id)
                existing = aggregated_edges.get(edge_key)
                if existing is None:
                    aggregated_edges[edge_key] = dict(edge_data)
                else:
                    aggregated_edges[edge_key] = _merge_edge_data_with_human_override(
                        existing, edge_data
                    )

        graph_max_async = max(int(global_config.get("llm_model_max_async", 4)), 1)
        semaphore = asyncio.Semaphore(graph_max_async)

        log_message = (
            "Controlled merge aggregated: "
            f"nodes {total_node_events}->{len(aggregated_nodes)}, "
            f"edges {total_edge_events}->{len(aggregated_edges)}, "
            f"async={graph_max_async}"
        )
        logger.info(log_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        async def _upsert_node_task(node_id: str, node_data: dict) -> None:
            async with semaphore:
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        if pipeline_status.get("cancellation_requested", False):
                            raise PipelineCancelledException(
                                "User cancelled during controlled node merge"
                            )
                await _upsert_controlled_node(
                    node_id,
                    node_data,
                    knowledge_graph_inst,
                    entity_vdb,
                    entity_chunks_storage,
                    stand_type,
                )

        async def _upsert_edge_task(src_id: str, tgt_id: str, edge_data: dict) -> None:
            async with semaphore:
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        if pipeline_status.get("cancellation_requested", False):
                            raise PipelineCancelledException(
                                "User cancelled during controlled edge merge"
                            )
                await _upsert_controlled_edge(
                    src_id,
                    tgt_id,
                    edge_data,
                    knowledge_graph_inst,
                    relationships_vdb,
                    relation_chunks_storage,
                )

        node_tasks = [
            asyncio.create_task(_upsert_node_task(node_id, node_data))
            for node_id, node_data in aggregated_nodes.items()
        ]
        if node_tasks:
            done, pending = await asyncio.wait(
                node_tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            first_exception = None
            for task in done:
                try:
                    task.result()
                except BaseException as exc:
                    if first_exception is None:
                        first_exception = exc
            if pending:
                for task in pending:
                    task.cancel()
                pending_results = await asyncio.gather(*pending, return_exceptions=True)
                if first_exception is None:
                    for result in pending_results:
                        if isinstance(result, BaseException):
                            first_exception = result
                            break
            if first_exception is not None:
                raise first_exception

        edge_tasks = [
            asyncio.create_task(_upsert_edge_task(src_id, tgt_id, edge_data))
            for (src_id, tgt_id), edge_data in aggregated_edges.items()
        ]
        if edge_tasks:
            done, pending = await asyncio.wait(
                edge_tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            first_exception = None
            for task in done:
                try:
                    task.result()
                except BaseException as exc:
                    if first_exception is None:
                        first_exception = exc
            if pending:
                for task in pending:
                    task.cancel()
                pending_results = await asyncio.gather(*pending, return_exceptions=True)
                if first_exception is None:
                    for result in pending_results:
                        if isinstance(result, BaseException):
                            first_exception = result
                            break
            if first_exception is not None:
                raise first_exception

        log_message = (
            f"Completed controlled merge: {len(chunk_results)} chunks, "
            f"{len(aggregated_nodes)} nodes, {len(aggregated_edges)} edges"
        )
        logger.info(log_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)
        return

    # Collect all nodes and edges from all chunks
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    for maybe_nodes, maybe_edges in chunk_results:
        # Collect nodes
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            all_edges[sorted_edge_key].extend(edges)

    total_entities_count = len(all_nodes)
    total_relations_count = len(all_edges)

    log_message = f"Merging stage {current_file_number}/{total_files}: {file_path}"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # ===== Phase 1: Process all entities concurrently =====
    log_message = f"Phase 1: Processing {total_entities_count} entities from {doc_id} (async: {graph_max_async})"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    async def _locked_process_entity_name(entity_name, entities):
        async with semaphore:
            # Check for cancellation before processing entity
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during entity merge"
                        )

            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                    [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    logger.debug(f"Processing entity {entity_name}")
                    entity_data = await _merge_nodes_then_upsert(
                        entity_name,
                        entities,
                        knowledge_graph_inst,
                        entity_vdb,
                        global_config,
                        pipeline_status,
                        pipeline_status_lock,
                        llm_response_cache,
                        entity_chunks_storage,
                    )

                    return entity_data

                except Exception as e:
                    error_msg = f"Error processing entity `{entity_name}`: {e}"
                    logger.error(error_msg)

                    # Try to update pipeline status, but don't let status update failure affect main exception
                    try:
                        if (
                                pipeline_status is not None
                                and pipeline_status_lock is not None
                        ):
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(error_msg)
                    except Exception as status_error:
                        logger.error(
                            f"Failed to update pipeline status: {status_error}"
                        )

                    # Re-raise the original exception with a prefix
                    prefixed_exception = create_prefixed_exception(
                        e, f"`{entity_name}`"
                    )
                    raise prefixed_exception from e

    # Create entity processing tasks
    entity_tasks = []
    for entity_name, entities in all_nodes.items():
        task = asyncio.create_task(_locked_process_entity_name(entity_name, entities))
        entity_tasks.append(task)

    # Execute entity tasks with error handling
    processed_entities = []
    if entity_tasks:
        done, pending = await asyncio.wait(
            entity_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        first_exception = None
        processed_entities = []

        for task in done:
            try:
                result = task.result()
            except BaseException as e:
                if first_exception is None:
                    first_exception = e
            else:
                processed_entities.append(result)

        if pending:
            for task in pending:
                task.cancel()
            pending_results = await asyncio.gather(*pending, return_exceptions=True)
            for result in pending_results:
                if isinstance(result, BaseException):
                    if first_exception is None:
                        first_exception = result
                else:
                    processed_entities.append(result)

        if first_exception is not None:
            raise first_exception

    # ===== Phase 2: Process all relationships concurrently =====
    log_message = f"Phase 2: Processing {total_relations_count} relations from {doc_id} (async: {graph_max_async})"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    async def _locked_process_edges(edge_key, edges):
        async with semaphore:
            # Check for cancellation before processing edges
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during relation merge"
                        )

            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            sorted_edge_key = sorted([edge_key[0], edge_key[1]])

            async with get_storage_keyed_lock(
                    sorted_edge_key,
                    namespace=namespace,
                    enable_logging=False,
            ):
                try:
                    added_entities = []  # Track entities added during edge processing

                    logger.debug(f"Processing relation {sorted_edge_key}")
                    edge_data = await _merge_edges_then_upsert(
                        edge_key[0],
                        edge_key[1],
                        edges,
                        knowledge_graph_inst,
                        relationships_vdb,
                        entity_vdb,
                        global_config,
                        pipeline_status,
                        pipeline_status_lock,
                        llm_response_cache,
                        added_entities,  # Pass list to collect added entities
                        relation_chunks_storage,
                        entity_chunks_storage,  # Add entity_chunks_storage parameter
                    )

                    if edge_data is None:
                        return None, []

                    return edge_data, added_entities

                except Exception as e:
                    error_msg = f"Error processing relation `{sorted_edge_key}`: {e}"
                    logger.error(error_msg)

                    # Try to update pipeline status, but don't let status update failure affect main exception
                    try:
                        if (
                                pipeline_status is not None
                                and pipeline_status_lock is not None
                        ):
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(error_msg)
                    except Exception as status_error:
                        logger.error(
                            f"Failed to update pipeline status: {status_error}"
                        )

                    # Re-raise the original exception with a prefix
                    prefixed_exception = create_prefixed_exception(
                        e, f"{sorted_edge_key}"
                    )
                    raise prefixed_exception from e

    # Create relationship processing tasks
    edge_tasks = []
    for edge_key, edges in all_edges.items():
        task = asyncio.create_task(_locked_process_edges(edge_key, edges))
        edge_tasks.append(task)

    # Execute relationship tasks with error handling
    processed_edges = []
    all_added_entities = []

    if edge_tasks:
        done, pending = await asyncio.wait(
            edge_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        first_exception = None

        for task in done:
            try:
                edge_data, added_entities = task.result()
            except BaseException as e:
                if first_exception is None:
                    first_exception = e
            else:
                if edge_data is not None:
                    processed_edges.append(edge_data)
                all_added_entities.extend(added_entities)

        if pending:
            for task in pending:
                task.cancel()
            pending_results = await asyncio.gather(*pending, return_exceptions=True)
            for result in pending_results:
                if isinstance(result, BaseException):
                    if first_exception is None:
                        first_exception = result
                else:
                    edge_data, added_entities = result
                    if edge_data is not None:
                        processed_edges.append(edge_data)
                    all_added_entities.extend(added_entities)

        if first_exception is not None:
            raise first_exception

    # ===== Phase 3: Update full_entities and full_relations storage =====
    if full_entities_storage and full_relations_storage and doc_id:
        try:
            # Merge all entities: original entities + entities added during edge processing
            final_entity_names = set()

            # Add original processed entities
            for entity_data in processed_entities:
                if entity_data and entity_data.get("entity_name"):
                    final_entity_names.add(entity_data["entity_name"])

            # Add entities that were added during relationship processing
            for added_entity in all_added_entities:
                if added_entity and added_entity.get("entity_name"):
                    final_entity_names.add(added_entity["entity_name"])

            # Collect all relation pairs
            final_relation_pairs = set()
            for edge_data in processed_edges:
                if edge_data:
                    src_id = edge_data.get("src_id")
                    tgt_id = edge_data.get("tgt_id")
                    if src_id and tgt_id:
                        relation_pair = tuple(sorted([src_id, tgt_id]))
                        final_relation_pairs.add(relation_pair)

            log_message = f"Phase 3: Updating final {len(final_entity_names)}({len(processed_entities)}+{len(all_added_entities)}) entities and  {len(final_relation_pairs)} relations from {doc_id}"
            logger.info(log_message)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

            # Update storage
            if final_entity_names:
                await full_entities_storage.upsert(
                    {
                        doc_id: {
                            "entity_names": list(final_entity_names),
                            "count": len(final_entity_names),
                        }
                    }
                )

            if final_relation_pairs:
                await full_relations_storage.upsert(
                    {
                        doc_id: {
                            "relation_pairs": [
                                list(pair) for pair in final_relation_pairs
                            ],
                            "count": len(final_relation_pairs),
                        }
                    }
                )

            logger.debug(
                f"Updated entity-relation index for document {doc_id}: {len(final_entity_names)} entities (original: {len(processed_entities)}, added: {len(all_added_entities)}), {len(final_relation_pairs)} relations"
            )

        except Exception as e:
            logger.error(
                f"Failed to update entity-relation index for document {doc_id}: {e}"
            )
            # Don't raise exception to avoid affecting main flow

    log_message = f"Completed merging: {len(processed_entities)} entities, {len(all_added_entities)} extra entities, {len(processed_edges)} relations"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)


async def extract_entities(
        chunks: dict[str, TextChunkSchema],
        global_config: dict[str, str],
        pipeline_status: dict = None,
        pipeline_status_lock=None,
        llm_response_cache: BaseKVStorage | None = None,
        text_chunks_storage: BaseKVStorage | None = None,
        stand_type: str | None = None
) -> list:
    # Check for cancellation at the start of entity extraction
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during entity extraction"
                )

    use_llm_func: callable = global_config["llm_model_func"]
    kg_schema_mode = global_config.get("kg_schema_mode", "")
    schema_extract_max_retries = global_config.get("schema_extract_max_retries", 2)
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    if kg_schema_mode == "electrical_controlled":
        processed_chunks = 0
        total_chunks = len(ordered_chunks)
        schema_cfg = global_config.get("addon_params", {}).get("electrical_schema", {})
        config_json = _json_dumps_compact(schema_cfg) if schema_cfg else "{}"
        clause_pattern = schema_cfg.get("clause_pattern", r"^(\d+(?:\.\d+)+)\s*(.*)$")
        try:
            clause_regex = re.compile(clause_pattern)
        except re.error:
            clause_regex = re.compile(r"^(\d+(?:\.\d+)+)\s*(.*)$")

        def _is_transient_gateway_error(exc: Exception) -> bool:
            text = str(exc).lower()
            transient_markers = (
                "504",
                "gateway time-out",
                "gateway timeout",
                "internalservererror",
                "timed out",
                "timeout",
            )
            return any(marker in text for marker in transient_markers)

        def _split_content_for_retry(content: str) -> list[str]:
            if not content or len(content) < 1200:
                return [content]

            lines = content.splitlines()
            if len(lines) < 4:
                mid = len(content) // 2
                return [content[:mid], content[mid:]]

            target_parts = 3
            chunk_size = max(1, len(lines) // target_parts)
            parts: list[str] = []
            for i in range(0, len(lines), chunk_size):
                part = "\n".join(lines[i: i + chunk_size]).strip()
                if part:
                    parts.append(part)
            return parts or [content]

        def _parse_controlled_json_response(raw_response: str) -> dict:
            """Parse controlled extraction JSON with tolerant cleanup/repair."""

            cleaned = remove_think_tags(raw_response or "").strip()

            # Strip fenced code blocks if model returns ```json ... ```
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s*```$", "", cleaned)
                cleaned = cleaned.strip()

            # Try extracting the primary JSON object region first.
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            candidate = cleaned[start: end + 1] if start != -1 and end > start else cleaned

            # Fast path: strict JSON
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            # Fallback: tolerant JSON repair parser
            try:
                parsed = json_repair.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            preview = candidate[:180].replace("\n", " ")
            raise ValueError(f"Invalid controlled JSON response: {preview}")

        def _safe_prompt_format(template: str, values: dict[str, Any]) -> str:
            """Format prompt safely even when template contains raw JSON braces."""

            # 1) Protect known placeholders with sentinels.
            output = template
            sentinels: dict[str, str] = {}
            for idx, (key, value) in enumerate(values.items()):
                sentinel = f"__LIGHTRAG_PLACEHOLDER_{idx}__"
                output = output.replace("{" + key + "}", sentinel)
                sentinels[sentinel] = str(value)

            # 2) Escape all remaining braces so JSON schema blocks are treated as literals.
            output = output.replace("{", "{{").replace("}", "}}")

            # 3) Restore sentinels to concrete values.
            for sentinel, value in sentinels.items():
                output = output.replace(sentinel, value)

            return output

        async def _process_single_content_controlled(chunk_key_dp: tuple[str, TextChunkSchema]):
            nonlocal processed_chunks
            chunk_key = chunk_key_dp[0]
            chunk_dp = chunk_key_dp[1]
            content = chunk_dp["content"]
            file_path = chunk_dp.get("file_path", "unknown_source")

            chunk_meta = {
                "std_id": chunk_dp.get("std_id", "")
                          or schema_cfg.get("standard_id", ""),
                "std_name": chunk_dp.get("std_name", "")
                            or schema_cfg.get("standard_name", ""),
                "clause_id": chunk_dp.get("clause_id", ""),
                "clause_title": chunk_dp.get("clause_title", ""),
                "chunk_id": chunk_key,
            }

            if not chunk_meta["clause_id"]:
                first_line = content.strip().splitlines()[0] if content.strip() else ""
                match = clause_regex.match(first_line)
                if match:
                    chunk_meta["clause_id"] = match.group(1)
                    if not chunk_meta["clause_title"]:
                        chunk_meta["clause_title"] = match.group(2).strip()

            cache_keys_collector = []
            system_prompt = _safe_prompt_format(
                PROMPTS["electrical_schema_extraction_system_prompt"],
                {"config_json": config_json},
            )

            async def _extract_with_retries(content_text: str) -> tuple[list, list, str | None]:
                user_prompt = _safe_prompt_format(
                    PROMPTS["electrical_schema_extraction_user_prompt"],
                    {
                        "std_id": chunk_meta["std_id"],
                        "std_name": chunk_meta["std_name"],
                        "clause_id": chunk_meta["clause_id"],
                        "clause_title": chunk_meta["clause_title"],
                        "chunk_id": chunk_meta["chunk_id"],
                        "chunk_text": content_text,
                    },
                )
                last_error = None
                for attempt in range(schema_extract_max_retries + 1):
                    prompt = user_prompt
                    if last_error:
                        prompt = f"{user_prompt}\nPrevious output invalid: {last_error}"
                    try:
                        result, timestamp = await use_llm_func_with_cache(
                            prompt,
                            use_llm_func,
                            system_prompt=system_prompt,
                            llm_response_cache=llm_response_cache,
                            cache_type="extract",
                            chunk_id=chunk_key,
                            cache_keys_collector=cache_keys_collector,
                        )
                    except Exception as exc:
                        last_error = str(exc)
                        logger.warning(
                            "Chunk %s LLM call failed (attempt %d/%d): %s",
                            chunk_key,
                            attempt + 1,
                            schema_extract_max_retries + 1,
                            last_error,
                        )
                        continue

                    try:  # json解析
                        parsed = _parse_controlled_json_response(result or "")
                        validated = _validate_controlled_payload(
                            parsed, content_text, chunk_meta
                        )
                        nodes, edges = _build_controlled_nodes_edges(
                            validated, chunk_meta, file_path, schema_cfg, content_text, stand_type
                        )
                        return nodes, edges, None
                    except Exception as exc:
                        last_error = str(exc)
                        logger.warning(
                            "Chunk %s schema validation failed (attempt %d/%d): %s",
                            chunk_key,
                            attempt + 1,
                            schema_extract_max_retries + 1,
                            last_error,
                        )
                        continue
                return [], [], last_error

            nodes, edges, last_error = await _extract_with_retries(content)
            if not nodes and not edges and last_error and _is_transient_gateway_error(Exception(last_error)):
                fallback_parts = _split_content_for_retry(content)
                if len(fallback_parts) > 1:
                    logger.warning(
                        "Chunk %s failed with transient error, fallback split into %d parts",
                        chunk_key,
                        len(fallback_parts),
                    )
                    merged_nodes: list = []
                    merged_edges: list = []
                    for idx, part in enumerate(fallback_parts, start=1):
                        part_nodes, part_edges, part_err = await _extract_with_retries(part)
                        if part_err:
                            logger.warning(
                                "Chunk %s fallback part %d failed: %s",
                                chunk_key,
                                idx,
                                part_err,
                            )
                        merged_nodes.extend(part_nodes)
                        merged_edges.extend(part_edges)
                    nodes, edges = merged_nodes, merged_edges

            if nodes or edges:
                if cache_keys_collector and text_chunks_storage:
                    await update_chunk_cache_list(
                        chunk_key,
                        text_chunks_storage,
                        cache_keys_collector,
                        "entity_extraction",
                    )
                processed_chunks += 1
                log_message = (
                    f"Chunk {processed_chunks} of {total_chunks} extracted "
                    f"{len(nodes)} Nodes + {len(edges)} Edges {chunk_key}"
                )
                logger.info(log_message)
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)
                return {"nodes": nodes, "edges": edges}

            processed_chunks += 1
            log_message = (
                f"Chunk {processed_chunks} of {total_chunks} extracted 0 Nodes + 0 Edges {chunk_key} (skipped)"
            )
            logger.info(log_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
            return {"nodes": [], "edges": []}

        chunk_max_async = global_config.get("llm_model_max_async", 4)
        semaphore = asyncio.Semaphore(chunk_max_async)

        async def _process_with_semaphore_controlled(chunk):
            async with semaphore:
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        if pipeline_status.get("cancellation_requested", False):
                            raise PipelineCancelledException(
                                "User cancelled during chunk processing"
                            )
                return await _process_single_content_controlled(chunk)

        tasks = [asyncio.create_task(_process_with_semaphore_controlled(c)) for c in ordered_chunks]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        results = []
        for task in done:
            try:
                results.append(task.result())
            except Exception as e:
                logger.error(
                    "Controlled extraction task failed and will be skipped: %s", e
                )
                results.append({"nodes": [], "edges": []})
        return results
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)
    entity_types = global_config["addon_params"].get(
        "entity_types", DEFAULT_ENTITY_TYPES
    )

    examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # Get file path from chunk data or use default
        file_path = chunk_dp.get("file_path", "unknown_source")

        # Create cache keys collector for batch processing
        cache_keys_collector = []

        # Get initial extraction
        # Format system prompt without input_text for each chunk (enables OpenAI prompt caching across chunks)
        entity_extraction_system_prompt = PROMPTS[
            "entity_extraction_system_prompt"
        ].format(**context_base)
        # Format user prompts with input_text for each chunk
        entity_extraction_user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
            **{**context_base, "input_text": content}
        )
        entity_continue_extraction_user_prompt = PROMPTS[
            "entity_continue_extraction_user_prompt"
        ].format(**{**context_base, "input_text": content})

        final_result, timestamp = await use_llm_func_with_cache(
            entity_extraction_user_prompt,
            use_llm_func,
            system_prompt=entity_extraction_system_prompt,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
        )

        history = pack_user_ass_to_openai_messages(
            entity_extraction_user_prompt, final_result
        )

        # Process initial extraction with file path
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result,
            chunk_key,
            timestamp,
            file_path,
            tuple_delimiter=context_base["tuple_delimiter"],
            completion_delimiter=context_base["completion_delimiter"],
        )

        # Process additional gleaning results only 1 time when entity_extract_max_gleaning is greater than zero.
        if entity_extract_max_gleaning > 0:
            glean_result, timestamp = await use_llm_func_with_cache(
                entity_continue_extraction_user_prompt,
                use_llm_func,
                system_prompt=entity_extraction_system_prompt,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
                chunk_id=chunk_key,
                cache_keys_collector=cache_keys_collector,
            )

            # Process gleaning result separately with file path
            glean_nodes, glean_edges = await _process_extraction_result(
                glean_result,
                chunk_key,
                timestamp,
                file_path,
                tuple_delimiter=context_base["tuple_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
            )

            # Merge results - compare description lengths to choose better version
            for entity_name, glean_entities in glean_nodes.items():
                if entity_name in maybe_nodes:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_nodes[entity_name][0].get("description", "") or ""
                    )
                    glean_desc_len = len(glean_entities[0].get("description", "") or "")

                    if glean_desc_len > original_desc_len:
                        maybe_nodes[entity_name] = list(glean_entities)
                    # Otherwise keep original version
                else:
                    # New entity from gleaning stage
                    maybe_nodes[entity_name] = list(glean_entities)

            for edge_key, glean_edges in glean_edges.items():
                if edge_key in maybe_edges:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_edges[edge_key][0].get("description", "") or ""
                    )
                    glean_desc_len = len(glean_edges[0].get("description", "") or "")

                    if glean_desc_len > original_desc_len:
                        maybe_edges[edge_key] = list(glean_edges)
                    # Otherwise keep original version
                else:
                    # New edge from gleaning stage
                    maybe_edges[edge_key] = list(glean_edges)

        # Batch update chunk's llm_cache_list with all collected cache keys
        if cache_keys_collector and text_chunks_storage:
            await update_chunk_cache_list(
                chunk_key,
                text_chunks_storage,
                cache_keys_collector,
                "entity_extraction",
            )

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel {chunk_key}"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    # Get max async tasks limit from global_config
    chunk_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk):
        async with semaphore:
            # Check for cancellation before processing chunk
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during chunk processing"
                        )

            try:
                return await _process_single_content(chunk)
            except Exception as e:
                chunk_id = chunk[0]  # Extract chunk_id from chunk[0]
                prefixed_exception = create_prefixed_exception(e, chunk_id)
                raise prefixed_exception from e

    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # Wait for tasks to complete or for the first exception to occur
    # This allows us to cancel remaining tasks if any task fails
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None
    chunk_results = []

    for task in done:
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                chunk_results.append(task.result())
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Add progress prefix to the exception message
        progress_prefix = f"C[{processed_chunks + 1}/{total_chunks}]"

        # Re-raise the original exception with a prefix
        prefixed_exception = create_prefixed_exception(first_exception, progress_prefix)
        raise prefixed_exception from first_exception

    # If all tasks completed successfully, chunk_results already contains the results
    # Return the chunk_results for later processing in merge_nodes_and_edges
    return chunk_results


async def kg_query(
        query: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
        system_prompt: str | None = None,
        chunks_vdb: BaseVectorStorage = None,
        stand_type: str | None = None
) -> QueryResult | None:
    """
    Execute knowledge graph query and return unified QueryResult object.

    Args:
        query: Query string
        knowledge_graph_inst: Knowledge graph storage instance
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_db: Text chunks storage
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt
        chunks_vdb: Document chunks vector database

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Based on different query_param settings, different fields will be populated:
        - only_need_context=True: content contains context string
        - only_need_prompt=True: content contains complete prompt
        - stream=True: response_iterator contains streaming response, raw_data contains complete data
        - default: content contains LLM response text, raw_data contains complete data

        Returns None when no relevant context could be constructed for the query.
    """
    print("111111111111111111111111111stand_type:", stand_type)
    stand_type = _normalize_operate_standard_type(
        stand_type or global_config.get("addon_params", {}).get("standard_type")
    )
    print("22222222222222222222222stand_type:", stand_type)
    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    schema_cfg = global_config.get("addon_params", {}).get("electrical_schema", {}) or {}
    retrieval_query, scope_query_meta = _prepare_scope_focused_query(query, schema_cfg)
    _start_electrical_trace_session(
        source="kg_query",
        mode=query_param.mode,
        query=query,
        retrieval_query=retrieval_query,
        user_prompt=query_param.user_prompt or "",
        response_type=query_param.response_type or "Multiple Paragraphs",
        current_report_scopes=scope_query_meta.get("current_report_scopes", []),
        scope_focused_query_applied=scope_query_meta.get("scope_focused_query_applied"),
        scope_focused_query_reason=scope_query_meta.get("scope_focused_query_reason", ""),
    )
    _log_electrical_trace(
        "kg_query_request",
        mode=query_param.mode,
        query=query,
        retrieval_query=retrieval_query,
        current_report_scopes=scope_query_meta.get("current_report_scopes", []),
        scope_focused_query_applied=scope_query_meta.get("scope_focused_query_applied"),
        scope_focused_query_reason=scope_query_meta.get("scope_focused_query_reason", ""),
        user_prompt=query_param.user_prompt or "",
        response_type=query_param.response_type or "Multiple Paragraphs",
        top_k=query_param.top_k,
        chunk_top_k=query_param.chunk_top_k,
        max_entity_tokens=query_param.max_entity_tokens,
        max_relation_tokens=query_param.max_relation_tokens,
        max_total_tokens=query_param.max_total_tokens,
        only_need_context=query_param.only_need_context,
        only_need_prompt=query_param.only_need_prompt,
        stream=query_param.stream,
    )

    def _extract_calculation_result_value(line: str) -> str | None:
        if "calculation" not in line.lower():
            return None
        normalized = line.replace("：", ":")
        calc_match = re.search(
            r"calculation\s*:\s*(.+)$", normalized, flags=re.IGNORECASE
        )
        if not calc_match:
            return None
        calc_body = calc_match.group(1).strip()
        if "=" not in calc_body:
            return None
        rhs = calc_body.rsplit("=", 1)[-1].strip()
        rhs = re.split(r"[；;（(。,\n]", rhs, maxsplit=1)[0].strip()
        if not rhs:
            return None
        value_match = re.match(
            r"^(?:≤|≥|<|>)?\s*-?\d+(?:\.\d+)?\s*(?:kV|V|mV|A|kA|s|min|ms|pC|%|m)?$",
            rhs,
            flags=re.IGNORECASE,
        )
        return rhs if value_match else None

    def _normalize_value_for_compare(value: str) -> str:
        return (
            value.replace(" ", "")
            .replace("：", ":")
            .replace("（", "(")
            .replace("）", ")")
            .strip()
            .lower()
        )

    def _enforce_formula_consistency(response_text: str) -> str:
        if not response_text or "calculation" not in response_text.lower():
            return response_text

        corrected_lines: list[str] = []
        corrected_count = 0

        for line in response_text.splitlines():
            if "calculation" not in line.lower():
                corrected_lines.append(line)
                continue

            calc_value = _extract_calculation_result_value(line)
            param_match = re.match(r"^(\s*-\s*[^：:\n]+[：:]\s*)([^；;\n]+)(.*)$", line)
            if not calc_value or not param_match:
                corrected_lines.append(line)
                continue

            prefix, current_value, suffix = param_match.groups()
            if _normalize_value_for_compare(current_value) == _normalize_value_for_compare(
                    calc_value
            ):
                corrected_lines.append(line)
                continue

            corrected_lines.append(f"{prefix}{calc_value}{suffix}")
            corrected_count += 1

        if corrected_count:
            logger.info(
                "Corrected %s formula-derived parameter value(s) to match calculation result",
                corrected_count,
            )

        return "\n".join(corrected_lines)

    hl_keywords, ll_keywords = await get_keywords_from_query(
        retrieval_query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
        logger.warning("low_level_keywords is empty")
    if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
        logger.warning("high_level_keywords is empty")
    if hl_keywords == [] and ll_keywords == []:
        if len(query) < 50:
            logger.warning(f"Forced low_level_keywords to origin query: {query}")
            ll_keywords = [retrieval_query]
        else:
            return QueryResult(content=PROMPTS["fail_response"])

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""
    _log_electrical_trace(
        "kg_query_keywords",
        high_level_keywords=hl_keywords,
        low_level_keywords=ll_keywords,
    )

    def _extract_second_retrieval_hints(response_text: str) -> list[str]:
        if not response_text:
            return []
        hints: list[str] = []
        patterns = [
            r"(?:需要二次检索|需要检索|需检索|缺失条款/表号|缺失条款|缺失表号)[：:]\s*(.+)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, response_text):
                content = match.group(1).strip()
                if not content:
                    continue
                parts = re.split(r"[，,；;、\n]", content)
                for part in parts:
                    part = part.strip()
                    if part:
                        hints.append(part)
        deduped = []
        for item in hints:
            if item not in deduped:
                deduped.append(item)
        return deduped

    # Build query context (unified interface)
    context_result = await _build_query_context(
        retrieval_query,
        query,
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
        stand_type=stand_type
    )
    # logger.info(f"context_result:{context_result}")
    if context_result is None:
        logger.info("[kg_query] No query context could be built; returning no-result.")
        return None

    # Return different content based on query parameters
    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(
            content=context_result.context, raw_data=context_result.raw_data
        )

    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Build system prompt
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        response_type=response_type,
        user_prompt=user_prompt,
        context_data=context_result.context,
    )

    user_query = query
    _log_model_input_trace(
        "pre_model_input",
        system_prompt=sys_prompt,
        user_query=user_query,
        context_data=context_result.context,
        history_messages=query_param.conversation_history,
    )

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=context_result.raw_data)

    # Call LLM
    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(
        f"[kg_query] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )
    _log_electrical_answer_debug("pre_llm", context_result.raw_data)

    # Handle cache
    bypass_query_cache = _should_bypass_query_cache(global_config)
    args_hash = compute_args_hash(
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        hl_keywords_str,
        ll_keywords_str,
        query_param.user_prompt or "",
        query_param.enable_rerank,
    )

    cached_result = None
    if not bypass_query_cache:
        cached_result = await handle_cache(
            hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
        )
    else:
        logger.info("[kg_query] Bypassing query cache for electrical controlled mode")

    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True,
            stream=query_param.stream,
        )

        if (
                not bypass_query_cache
                and hashing_kv
                and hashing_kv.global_config.get("enable_llm_cache")
        ):
            queryparam_dict = {
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "hl_keywords": hl_keywords_str,
                "ll_keywords": ll_keywords_str,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        if not query_param.stream:
            second_hints = _extract_second_retrieval_hints(response)
            if second_hints:
                second_query = f"{query} " + " ".join(second_hints)
                second_retrieval_query, second_scope_query_meta = _prepare_scope_focused_query(
                    second_query,
                    schema_cfg,
                )
                logger.info(
                    "[kg_query] Second retrieval triggered with hints: %s | rewrite_applied=%s | reason=%s",
                    "; ".join(second_hints),
                    second_scope_query_meta.get("scope_focused_query_applied"),
                    second_scope_query_meta.get("scope_focused_query_reason", ""),
                )
                second_context_result = await _build_query_context(
                    second_retrieval_query,
                    query,
                    ll_keywords_str,
                    hl_keywords_str,
                    knowledge_graph_inst,
                    entities_vdb,
                    relationships_vdb,
                    text_chunks_db,
                    query_param,
                    chunks_vdb,
                    stand_type=stand_type
                )
                if second_context_result is not None:
                    sys_prompt_2 = sys_prompt_temp.format(
                        response_type=response_type,
                        user_prompt=user_prompt,
                        context_data=second_context_result.context,
                    )
                    _log_model_input_trace(
                        "pre_model_input_second_retrieval",
                        system_prompt=sys_prompt_2,
                        user_query=user_query,
                        context_data=second_context_result.context,
                        history_messages=query_param.conversation_history,
                    )
                    response_2 = await use_model_func(
                        user_query,
                        system_prompt=sys_prompt_2,
                        history_messages=query_param.conversation_history,
                        enable_cot=True,
                        stream=query_param.stream,
                    )
                    if isinstance(response_2, str):
                        if len(response_2) > len(sys_prompt_2):
                            response_2 = (
                                response_2.replace(sys_prompt_2, "")
                                .replace("user", "")
                                .replace("model", "")
                                .replace(query, "")
                                .replace("<system>", "")
                                .replace("</system>", "")
                                .strip()
                            )
                        _log_electrical_answer_debug(
                            "second_before_postprocess",
                            second_context_result.raw_data,
                            response_2,
                        )
                        response_2 = _postprocess_electrical_markdown_response(
                            _enforce_formula_consistency(response_2),
                            second_context_result.raw_data,
                        )
                        _log_electrical_answer_debug(
                            "second_after_postprocess",
                            second_context_result.raw_data,
                            response_2,
                        )
                        if "metadata" not in second_context_result.raw_data:
                            second_context_result.raw_data["metadata"] = {}
                        second_context_result.raw_data["metadata"][
                            "second_retrieval"
                        ] = {"hints": second_hints}
                        return QueryResult(
                            content=response_2, raw_data=second_context_result.raw_data
                        )
                    else:
                        if "metadata" not in second_context_result.raw_data:
                            second_context_result.raw_data["metadata"] = {}
                        second_context_result.raw_data["metadata"][
                            "second_retrieval"
                        ] = {"hints": second_hints}
                        return QueryResult(
                            response_iterator=_stream_electrical_response_with_final_event(
                                response_2,
                                second_context_result.raw_data,
                                sys_prompt_2,
                                query,
                                "second_stream",
                            ),
                            raw_data=second_context_result.raw_data,
                            is_streaming=True,
                        )
        _log_electrical_answer_debug(
            "before_postprocess",
            context_result.raw_data,
            response,
        )
        response = _postprocess_electrical_markdown_response(
            _enforce_formula_consistency(response),
            context_result.raw_data,
        )
        _log_electrical_answer_debug(
            "after_postprocess",
            context_result.raw_data,
            response,
        )
        return QueryResult(content=response, raw_data=context_result.raw_data)
    else:
        # Streaming response (AsyncIterator)
        _log_electrical_answer_debug("stream_return", context_result.raw_data)
        return QueryResult(
            response_iterator=_stream_electrical_response_with_final_event(
                response,
                context_result.raw_data,
                sys_prompt,
                query,
                "stream",
            ),
            raw_data=context_result.raw_data,
            is_streaming=True,
        )


async def get_keywords_from_query(
        query: str,
        query_param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Retrieves high-level and low-level keywords for RAG operations.

    This function checks if keywords are already provided in query parameters,
    and if not, extracts them from the query text using LLM.

    Args:
        query: The user's query text
        query_param: Query parameters that may contain pre-defined keywords
        global_config: Global configuration dictionary
        hashing_kv: Optional key-value storage for caching results

    Returns:
        A tuple containing (high_level_keywords, low_level_keywords)
    """
    # Check if pre-defined keywords are already provided
    if query_param.hl_keywords or query_param.ll_keywords:
        return query_param.hl_keywords, query_param.ll_keywords

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )
    return hl_keywords, ll_keywords


def _prepare_scope_focused_query(
        query: str,
        schema_cfg: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    current_report_scopes = _extract_current_report_scopes(query, schema_cfg)
    metadata = {
        "current_report_scopes": current_report_scopes,
        "scope_focused_query_applied": False,
        "scope_focused_query_reason": "no_scope_detected",
    }

    if not current_report_scopes:
        return query, metadata

    if len(current_report_scopes) > 1:
        metadata["scope_focused_query_reason"] = "multi_scope_bypass"
        return query, metadata

    if current_report_scopes[0] != "绝缘性能型式试验":
        metadata["scope_focused_query_reason"] = "non_insulation_scope_bypass"
        return query, metadata

    retrieval_query = _build_scope_focused_query(query)
    metadata["scope_focused_query_applied"] = retrieval_query != query
    metadata["scope_focused_query_reason"] = (
        "single_insulation_scope_applied"
        if retrieval_query != query
        else "single_insulation_scope_no_change"
    )
    return retrieval_query, metadata


# 提问 绝缘性能型式试验，获取参数
def _build_scope_focused_query(query: str) -> str:
    """Reduce noisy cross-domain parameters before retrieval for known test scopes."""
    text = str(query or "").strip()
    if not text:
        return text

    if "绝缘性能型式试验" not in text:
        return text

    normalized = " ".join(text.split())
    fragments: list[str] = []

    model_match = re.search(r"型号名称\s*[：:]\s*([^，。；;\n]+)", normalized)
    if model_match:
        fragments.append(f"型号名称：{model_match.group(1).strip()}")

    if "绝缘性能型式试验" in normalized:
        fragments.append("断路器需要进行绝缘性能型式试验")

    insulation_patterns = [
        r"额定电压\s*[0-9]+(?:\.[0-9]+)?\s*kV",
        r"额定电流\s*[0-9]+(?:\.[0-9]+)?\s*A",
        r"额定频率\s*[0-9]+(?:\.[0-9]+)?\s*Hz?",
        r"额定短时工频耐受电压(?:\(断口\))?\s*[0-9]+(?:\.[0-9]+)?\s*k?V?",
        r"额定雷电冲击耐受电压(?:\(断口\))?\s*[0-9]+(?:\.[0-9]+)?\s*k?V?",
        r"最大\(适用\)的海拔\s*[0-9]+(?:\.[0-9]+)?\s*m",
        r"SF6气体的最低功能压力\(20℃表压\)\s*[0-9]+(?:\.[0-9]+)?\s*MPa",
        r"SF6气体的额定压力\(20℃表压\)\s*[0-9]+(?:\.[0-9]+)?\s*MPa",
        r"额定直流电压\s*[0-9]+(?:\.[0-9]+)?\s*k?V?",
        r"元件中含固封极柱",
        r"户内[^，。；;\n]*断路器",
        r"户外[^，。；;\n]*断路器",
        r"固封式",
        r"充气断路器",
        r"充油断路器",
    ]

    for pattern in insulation_patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            value = match.group(0).strip("，。；; ")
            if value and value not in fragments:
                fragments.append(value)

    # Fall back to original query if trimming would over-prune context.
    return "，".join(fragments) if len(fragments) >= 3 else text


async def extract_keywords_only(
        text: str,
        param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Build the examples
    examples = "\n".join(PROMPTS["keywords_extraction_examples"])

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)

    # 2. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(
        param.mode,
        text,
        language,
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        try:
            keywords_data = json_repair.loads(cached_response)
            return keywords_data.get("high_level_keywords", []), keywords_data.get(
                "low_level_keywords", []
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 3. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text,
        examples=examples,
        language=language,
    )

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(kw_prompt))
    logger.debug(
        f"[extract_keywords] Sending to LLM: {len_of_prompts:,} tokens (Prompt: {len_of_prompts})"
    )

    # 4. Call the LLM for keyword extraction
    if param.model_func:
        use_model_func = param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 5. Parse out JSON from the LLM response
    result = remove_think_tags(result)
    try:
        keywords_data = json_repair.loads(result)
        if not keywords_data:
            logger.error("No JSON-like structure found in the LLM respond.")
            return [], []
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"LLM respond: {result}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # Domain-specific keyword augmentation for electrical standards queries.
    if global_config.get("kg_schema_mode") == "electrical_controlled":
        extra_ll_keywords: list[str] = []

        # Clause numbers like 7.2.6.3
        extra_ll_keywords.extend(re.findall(r"\b\d+(?:\.\d+)+\b", text))

        # Table references like 表10 / 表 13
        extra_ll_keywords.extend([f"表{m}" for m in re.findall(r"表\s*(\d+)", text)])

        # Standard identifiers like GB/T 11022-2020, IEC 62271-100
        extra_ll_keywords.extend(
            re.findall(r"\b(?:GB/T|GBT|DL/T|DLT|IEC)\s*\d+(?:[-—]\d+)?\b", text)
        )

        if extra_ll_keywords:
            ll_keywords = list(dict.fromkeys(ll_keywords + extra_ll_keywords))

    # 6. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        if hashing_kv.global_config.get("enable_llm_cache"):
            # Save to cache with query parameters
            queryparam_dict = {
                "mode": param.mode,
                "response_type": param.response_type,
                "top_k": param.top_k,
                "chunk_top_k": param.chunk_top_k,
                "max_entity_tokens": param.max_entity_tokens,
                "max_relation_tokens": param.max_relation_tokens,
                "max_total_tokens": param.max_total_tokens,
                "user_prompt": param.user_prompt or "",
                "enable_rerank": param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=json.dumps(cache_data),
                    prompt=text,
                    mode=param.mode,
                    cache_type="keywords",
                    queryparam=queryparam_dict,
                ),
            )

    return hl_keywords, ll_keywords


async def _get_vector_context(
        query: str,
        chunks_vdb: BaseVectorStorage,
        query_param: QueryParam,
        query_embedding: list[float] = None,
) -> list[dict]:
    """
    Retrieve text chunks from the vector database without reranking or truncation.

    This function performs vector search to find relevant text chunks for a query.
    Reranking and truncation will be handled later in the unified processing.

    Args:
        query: The query string to search for
        chunks_vdb: Vector database containing document chunks
        query_param: Query parameters including chunk_top_k and ids
        query_embedding: Optional pre-computed query embedding to avoid redundant embedding calls

    Returns:
        List of text chunks with metadata
    """
    try:
        # Use chunk_top_k if specified, otherwise fall back to top_k
        search_top_k = query_param.chunk_top_k or query_param.top_k
        cosine_threshold = chunks_vdb.cosine_better_than_threshold

        results = await chunks_vdb.query(
            query, top_k=search_top_k, query_embedding=query_embedding
        )
        if not results:
            logger.info(
                f"Naive query: 0 chunks (chunk_top_k:{search_top_k} cosine:{cosine_threshold})"
            )
            return []

        valid_chunks = []
        for result in results:
            if "content" in result:
                chunk_id = _resolve_chunk_id(result)
                if not chunk_id:
                    logger.warning(
                        "Vector chunk missing identity fields and content hash fallback failed"
                    )
                    continue
                chunk_with_metadata = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                    "source_type": "vector",  # Mark the source type
                    "chunk_id": chunk_id,
                }
                valid_chunks.append(chunk_with_metadata)

        # Domain-aware rerank: prioritize chunks that match clause/table hints.
        if valid_chunks and chunks_vdb.global_config.get("kg_schema_mode") == "electrical_controlled":
            clause_nums = set(re.findall(r"\b\d+(?:\.\d+)+\b", query or ""))
            table_nums = set(re.findall(r"表\\s*(\\d+)", query or ""))
            table_tags = {f"表{n}" for n in table_nums}

            if clause_nums or table_tags:
                def _score(chunk: dict) -> int:
                    content = chunk.get("content") or ""
                    score = 0
                    for num in clause_nums:
                        if num in content:
                            score += 3
                    for tag in table_tags:
                        if tag in content:
                            score += 2
                    return score

                scored = [(i, _score(c), c) for i, c in enumerate(valid_chunks)]
                scored.sort(key=lambda x: (-x[1], x[0]))
                valid_chunks = [c for _, _, c in scored]

        logger.info(
            f"Naive query: {len(valid_chunks)} chunks (chunk_top_k:{search_top_k} cosine:{cosine_threshold})"
        )
        return valid_chunks

    except Exception as e:
        logger.error(f"Error in _get_vector_context: {e}")
        return []


def _resolve_chunk_id(chunk: dict[str, Any]) -> str | None:
    """Resolve stable chunk ID from known fields, with content-hash fallback."""

    chunk_id = (
            chunk.get("chunk_id")
            or chunk.get("id")
            or chunk.get("__id__")
            or chunk.get("source_id")
    )
    if chunk_id:
        return str(chunk_id)

    content = chunk.get("content")
    if not content:
        return None

    file_path = chunk.get("file_path", "")
    created_at = chunk.get("created_at", "")
    return compute_mdhash_id(
        f"{file_path}|{created_at}|{content}",
        prefix="chunk-fallback-",
    )


def _extract_doc_type_filters(query: str) -> set[str]:
    if not query:
        return set()
    text = query.upper()
    filters = set()

    if re.search(r"\bGB\s*[/\-_]?\s*T\b", text) or re.search(r"\bGBT\b", text) or "GB_T" in text:
        filters.add("GB/T")
    if re.search(r"\bDL\s*[/\-_]?\s*T\b", text) or re.search(r"\bDLT\b", text) or "DL_T" in text:
        filters.add("DL/T")
    if re.search(r"\bIEC\b", text):
        filters.add("IEC")
    if re.search(r"\bSTL\b", text):
        filters.add("STL")

    if "国标" in query or "国家标准" in query:
        filters.add("GB/T")
    if "电力行业标准" in query:
        filters.add("DL/T")
    if "国际" in query or"国际标准" in query:
        filters.add("IEC")

    return filters


def _infer_doc_type_from_file_path(file_path: str) -> str | None:
    if not file_path:
        return None
    base_name = re.split(r"[\\/]", str(file_path).strip())[-1]
    if not base_name:
        return None
    text = base_name.upper()

    if re.search(r"\bGB\s*[/\-_]?\s*T\b", text) or re.search(r"\bGBT\b", text) or text.startswith("GB_T"):
        return "GB/T"
    if re.search(r"\bDL\s*[/\-_]?\s*T\b", text) or re.search(r"\bDLT\b", text) or text.startswith("DL_T"):
        return "DL/T"
    if re.search(r"\bIEC\b", text) or text.startswith("IEC"):
        return "IEC"
    if re.search(r"\bSTL\b", text) or text.startswith("STL"):
        return "STL"
    return None


def _filter_search_result_by_doc_type(
        search_result: dict[str, Any], doc_type_filters: set[str]
) -> dict[str, Any]:
    if not doc_type_filters:
        return search_result

    def _is_match(item: dict) -> bool:
        inferred = _infer_doc_type_from_file_path(item.get("file_path", ""))
        return inferred in doc_type_filters if inferred else False

    def _boost(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        matched = []
        unmatched = []
        for item in items:
            (matched if _is_match(item) else unmatched).append(item)
        return matched + unmatched

    filtered_entities = _boost(list(search_result["final_entities"]))
    filtered_relations = _boost(list(search_result["final_relations"]))
    filtered_vector_chunks = _boost(list(search_result["vector_chunks"]))

    filtered_chunk_tracking = dict(search_result.get("chunk_tracking", {}))

    logger.info(
        "Doc type filter %s: boosted %d entities, %d relations, %d vector chunks",
        ",".join(sorted(doc_type_filters)),
        len(filtered_entities),
        len(filtered_relations),
        len(filtered_vector_chunks),
    )

    return {
        **search_result,
        "final_entities": filtered_entities,
        "final_relations": filtered_relations,
        "vector_chunks": filtered_vector_chunks,
        "chunk_tracking": filtered_chunk_tracking,
    }


async def _perform_kg_search(
        query: str,
        ll_keywords: str,
        hl_keywords: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
        chunks_vdb: BaseVectorStorage = None,
) -> dict[str, Any]:
    """
    Pure search logic that retrieves raw entities, relations, and vector chunks.
    No token truncation or formatting - just raw search results.
    """

    # Initialize result containers
    local_entities = []
    local_relations = []
    global_entities = []
    global_relations = []
    vector_chunks = []
    chunk_tracking = {}

    # Handle different query modes

    # Track chunk sources and metadata for final logging
    chunk_tracking = {}  # chunk_id -> {source, frequency, order}

    # Pre-compute query embedding once for all vector operations
    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    query_embedding = None
    if query and (kg_chunk_pick_method == "VECTOR" or chunks_vdb):
        actual_embedding_func = text_chunks_db.embedding_func
        if actual_embedding_func:
            try:
                query_embedding = await actual_embedding_func([query])
                query_embedding = query_embedding[
                    0
                ]  # Extract first embedding from batch result
                logger.debug("Pre-computed query embedding for all vector operations")
            except Exception as e:
                logger.warning(f"Failed to pre-compute query embedding: {e}")
                query_embedding = None

    # Handle local and global modes
    if query_param.mode == "local" and len(ll_keywords) > 0:
        local_entities, local_relations = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
        )

    elif query_param.mode == "global" and len(hl_keywords) > 0:
        global_relations, global_entities = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )

    else:  # hybrid or mix mode
        if len(ll_keywords) > 0:
            local_entities, local_relations = await _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                query_param,
            )
        if len(hl_keywords) > 0:
            global_relations, global_entities = await _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                query_param,
            )

        # Get vector chunks for mix mode
        if query_param.mode == "mix" and chunks_vdb:
            vector_chunks = await _get_vector_context(
                query,
                chunks_vdb,
                query_param,
                query_embedding,
            )
            # Track vector chunks with source metadata
            for i, chunk in enumerate(vector_chunks):
                chunk_id = _resolve_chunk_id(chunk)
                if chunk_id:
                    chunk["chunk_id"] = chunk_id
                    chunk_tracking[chunk_id] = {
                        "source": "C",
                        "frequency": 1,  # Vector chunks always have frequency 1
                        "order": i + 1,  # 1-based order in vector search results
                    }
                else:
                    logger.warning(f"Vector chunk missing chunk_id: {chunk}")

    # Round-robin merge entities
    final_entities = []
    seen_entities = set()
    max_len = max(len(local_entities), len(global_entities))
    for i in range(max_len):
        # First from local
        if i < len(local_entities):
            entity = local_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

        # Then from global
        if i < len(global_entities):
            entity = global_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

    # Round-robin merge relations
    final_relations = []
    seen_relations = set()
    max_len = max(len(local_relations), len(global_relations))
    for i in range(max_len):
        # First from local
        if i < len(local_relations):
            relation = local_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

        # Then from global
        if i < len(global_relations):
            relation = global_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

    logger.info(
        f"Raw search results: {len(final_entities)} entities, {len(final_relations)} relations, {len(vector_chunks)} vector chunks"
    )

    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "vector_chunks": vector_chunks,
        "chunk_tracking": chunk_tracking,
        "query_embedding": query_embedding,
    }


async def _apply_token_truncation(
        search_result: dict[str, Any],
        query_param: QueryParam,
        global_config: dict[str, str],
) -> dict[str, Any]:
    """
    Apply token-based truncation to entities and relations for LLM efficiency.
    """
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.warning("No tokenizer found, skipping truncation")
        return {
            "entities_context": [],
            "relations_context": [],
            "filtered_entities": search_result["final_entities"],
            "filtered_relations": search_result["final_relations"],
            "entity_id_to_original": {},
            "relation_id_to_original": {},
        }

    # Get token limits from query_param with fallbacks
    max_entity_tokens = getattr(
        query_param,
        "max_entity_tokens",
        global_config.get("max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS),
    )
    max_relation_tokens = getattr(
        query_param,
        "max_relation_tokens",
        global_config.get("max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS),
    )

    final_entities = search_result["final_entities"]
    final_relations = search_result["final_relations"]

    # Create mappings from entity/relation identifiers to original data
    entity_id_to_original = {}
    relation_id_to_original = {}

    # Generate entities context for truncation
    entities_context = []
    for i, entity in enumerate(final_entities):
        entity_name = entity["entity_name"]
        created_at = entity.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Store mapping from entity name to original data
        entity_id_to_original[entity_name] = entity

        entities_context.append(
            {
                "entity": entity_name,
                "type": entity.get("entity_type", "UNKNOWN"),
                "description": entity.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": entity.get("file_path", "unknown_source"),
            }
        )

    # Generate relations context for truncation
    relations_context = []
    for i, relation in enumerate(final_relations):
        created_at = relation.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Handle different relation data formats
        if "src_tgt" in relation:
            entity1, entity2 = relation["src_tgt"]
        else:
            entity1, entity2 = relation.get("src_id"), relation.get("tgt_id")

        # Store mapping from relation pair to original data
        relation_key = (entity1, entity2)
        relation_id_to_original[relation_key] = relation

        relations_context.append(
            {
                "entity1": entity1,
                "entity2": entity2,
                "description": relation.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": relation.get("file_path", "unknown_source"),
            }
        )

    logger.debug(
        f"Before truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Apply token-based truncation
    if entities_context:
        # Remove file_path and created_at for token calculation
        entities_context_for_truncation = []
        for entity in entities_context:
            entity_copy = entity.copy()
            entity_copy.pop("file_path", None)
            entity_copy.pop("created_at", None)
            entities_context_for_truncation.append(entity_copy)

        entities_context = truncate_list_by_token_size(
            entities_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_entity_tokens,
            tokenizer=tokenizer,
        )

    if relations_context:
        # Remove file_path and created_at for token calculation
        relations_context_for_truncation = []
        for relation in relations_context:
            relation_copy = relation.copy()
            relation_copy.pop("file_path", None)
            relation_copy.pop("created_at", None)
            relations_context_for_truncation.append(relation_copy)

        relations_context = truncate_list_by_token_size(
            relations_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_relation_tokens,
            tokenizer=tokenizer,
        )

    logger.info(
        f"After truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Create filtered original data based on truncated context
    filtered_entities = []
    filtered_entity_id_to_original = {}
    if entities_context:
        final_entity_names = {e["entity"] for e in entities_context}
        seen_nodes = set()
        for entity in final_entities:
            name = entity.get("entity_name")
            if name in final_entity_names and name not in seen_nodes:
                filtered_entities.append(entity)
                filtered_entity_id_to_original[name] = entity
                seen_nodes.add(name)

    filtered_relations = []
    filtered_relation_id_to_original = {}
    if relations_context:
        final_relation_pairs = {(r["entity1"], r["entity2"]) for r in relations_context}
        seen_edges = set()
        for relation in final_relations:
            src, tgt = relation.get("src_id"), relation.get("tgt_id")
            if src is None or tgt is None:
                src, tgt = relation.get("src_tgt", (None, None))

            pair = (src, tgt)
            if pair in final_relation_pairs and pair not in seen_edges:
                filtered_relations.append(relation)
                filtered_relation_id_to_original[pair] = relation
                seen_edges.add(pair)

    return {
        "entities_context": entities_context,
        "relations_context": relations_context,
        "filtered_entities": filtered_entities,
        "filtered_relations": filtered_relations,
        "entity_id_to_original": filtered_entity_id_to_original,
        "relation_id_to_original": filtered_relation_id_to_original,
    }


async def _merge_all_chunks(
        filtered_entities: list[dict],
        filtered_relations: list[dict],
        vector_chunks: list[dict],
        query: str = "",
        knowledge_graph_inst: BaseGraphStorage = None,
        text_chunks_db: BaseKVStorage = None,
        query_param: QueryParam = None,
        chunks_vdb: BaseVectorStorage = None,
        chunk_tracking: dict = None,
        query_embedding: list[float] = None,
) -> list[dict]:
    """
    Merge chunks from different sources: vector_chunks + entity_chunks + relation_chunks.
    """
    if chunk_tracking is None:
        chunk_tracking = {}

    # Get chunks from entities
    entity_chunks = []
    if filtered_entities and text_chunks_db:
        entity_chunks = await _find_related_text_unit_from_entities(
            filtered_entities,
            query_param,
            text_chunks_db,
            knowledge_graph_inst,
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Get chunks from relations
    relation_chunks = []
    if filtered_relations and text_chunks_db:
        relation_chunks = await _find_related_text_unit_from_relations(
            filtered_relations,
            query_param,
            text_chunks_db,
            entity_chunks,  # For deduplication
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Round-robin merge chunks from different sources with deduplication
    merged_chunks = []
    seen_chunk_ids = set()
    max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks))
    origin_len = len(vector_chunks) + len(entity_chunks) + len(relation_chunks)

    for i in range(max_len):
        # Add from vector chunks first (Naive mode)
        if i < len(vector_chunks):
            chunk = vector_chunks[i]
            chunk_id = _resolve_chunk_id(chunk)
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

        # Add from entity chunks (Local mode)
        if i < len(entity_chunks):
            chunk = entity_chunks[i]
            chunk_id = _resolve_chunk_id(chunk)
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

        # Add from relation chunks (Global mode)
        if i < len(relation_chunks):
            chunk = relation_chunks[i]
            chunk_id = _resolve_chunk_id(chunk)
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

    logger.info(
        f"Round-robin merged chunks: {origin_len} -> {len(merged_chunks)} (deduplicated {origin_len - len(merged_chunks)})"
    )

    return merged_chunks


async def _build_context_str(
        entities_context: list[dict],
        relations_context: list[dict],
        merged_chunks: list[dict],
        query: str,
        rule_query: str | None,
        query_param: QueryParam,
        global_config: dict[str, str],
        chunk_tracking: dict = None,
        entity_id_to_original: dict = None,
        relation_id_to_original: dict = None,
        knowledge_graph_inst: BaseGraphStorage | None = None,
        stand_type: str = None
) -> tuple[str, dict[str, Any]]:
    """
    Build the final LLM context string with token processing.
    This includes dynamic token calculation and final chunk truncation.
    """
    stand_type = _normalize_operate_standard_type(stand_type)
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.error("Missing tokenizer, cannot build LLM context")
        # Return empty raw data structure when no tokenizer
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Missing tokenizer, cannot build LLM context."
        return "", empty_raw_data

    # Get token limits
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Get the system prompt template from PROMPTS or global_config
    sys_prompt_template = global_config.get(
        "system_prompt_template", PROMPTS["rag_response"]
    )

    kg_context_template = PROMPTS["kg_query_context"]
    user_prompt = query_param.user_prompt if query_param.user_prompt else ""
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )
    addon_params = global_config.get("addon_params", {}) or {}
    schema_cfg = addon_params.get("electrical_schema", {}) or {}

    def _extract_named_voltage_kv(query_text: str, labels: list[str]) -> float | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        normalized = text.replace("（", "(").replace("）", ")")
        for label in labels:
            pattern = rf"{re.escape(label)}\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?(?:\s*\+\s*[0-9]+(?:\.[0-9]+)?)*)\s*(?:kV)?"
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                parts = re.findall(r"[0-9]+(?:\.[0-9]+)?", match.group(1))
                if parts:
                    return sum(float(part) for part in parts)
        return None

    def _extract_model_prefix(query_text: str) -> str | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        match = re.search(r"(?:型号名称|型号)\s*[：:=]\s*([A-Za-z0-9]+)", text)
        return match.group(1).upper() if match else None

    def _extract_rated_current_amp(query_text: str) -> int | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        match = re.search(r"额定电流\s*(?:[:：=]\s*)?([0-9]+)\s*A\b", text, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_rated_voltage_kv(query_text: str) -> float | None:
        text = str(query_text or "").strip()
        if not text:
            return None
        match = re.search(
            r"额定电压\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*kV\b", text, flags=re.IGNORECASE
        )
        return float(match.group(1)) if match else None

    def _query_has_explicit_solid_sealed_pole(query_text: str) -> bool:
        text = str(query_text or "").strip()
        if not text:
            return False
        return bool(re.search(r"元件中含固封极柱", text))

    rule_query_text = str(rule_query or query or "").strip()
    model_prefix = _extract_model_prefix(rule_query_text)
    rated_current_amp = _extract_rated_current_amp(rule_query_text)
    rated_voltage_kv = _extract_rated_voltage_kv(rule_query_text)
    explicit_solid_sealed_pole = _query_has_explicit_solid_sealed_pole(rule_query_text)
    domain_rule_decisions = _evaluate_domain_rule_decisions(rule_query_text, schema_cfg, stand_type=stand_type)
    normalized_stand_type = _normalize_operate_standard_type(stand_type)
    if normalized_stand_type == "DLT":
        pf_split_rule = domain_rule_decisions.get("insulation.gb.power_frequency_split", {})
        li_split_rule = domain_rule_decisions.get("insulation.gb.lightning_impulse_split", {})
        pd_app_rule = domain_rule_decisions.get(
            "insulation.gb.partial_discharge_applicability", {}
        )
    elif normalized_stand_type == "IEC":
        pf_split_rule = domain_rule_decisions.get("insulation.gb.power_frequency_split", {}) or domain_rule_decisions.get(
            "insulation.gb.power_frequency_joint_voltage_split", {}
        )
        li_split_rule = domain_rule_decisions.get("insulation.gb.lightning_impulse_split", {}) or domain_rule_decisions.get(
            "insulation.gb.lightning_impulse_joint_voltage_split", {}
        )
        pd_app_rule = None
        # pd_app_rule = domain_rule_decisions.get(
        #     "insulation.gb.partial_discharge_applicability", {}
        # )
    else:
        pf_split_rule = domain_rule_decisions.get("insulation.gb.power_frequency_split", {}) or domain_rule_decisions.get(
            "insulation.gb.power_frequency_joint_voltage_split", {}
        )
        li_split_rule = domain_rule_decisions.get("insulation.gb.lightning_impulse_split", {}) or domain_rule_decisions.get(
            "insulation.gb.lightning_impulse_joint_voltage_split", {}
        )
        pd_app_rule = domain_rule_decisions.get(
            "insulation.gb.partial_discharge_applicability", {}
        )
    pf_split_inputs = pf_split_rule.get("inputs", {}) if isinstance(pf_split_rule, dict) else {}
    li_split_inputs = li_split_rule.get("inputs", {}) if isinstance(li_split_rule, dict) else {}
    pd_app_inputs = pd_app_rule.get("inputs", {}) if isinstance(pd_app_rule, dict) else {}

    fracture_pf_enabled = bool(pf_split_rule.get("enabled")) if isinstance(pf_split_rule, dict) else False
    fracture_li_enabled = bool(li_split_rule.get("enabled")) if isinstance(li_split_rule, dict) else False
    fracture_pf_provided = bool(pf_split_inputs.get("fracture_voltage_provided")) if isinstance(pf_split_inputs,
                                                                                                dict) else False
    fracture_li_provided = bool(li_split_inputs.get("fracture_voltage_provided")) if isinstance(li_split_inputs,
                                                                                                dict) else False
    pd_allowed = bool(pd_app_rule.get("enabled")) if isinstance(pd_app_rule, dict) else False
    pd_allowed_by_voltage = bool(rated_voltage_kv == 40.5)
    pd_allowed_by_model = bool(
        model_prefix and model_prefix != "VF1" and rated_current_amp == 4000
    )
    if isinstance(pd_app_inputs, dict):
        explicit_solid_sealed_pole = bool(
            pd_app_inputs.get("explicit_solid_sealed_pole", explicit_solid_sealed_pole)
        )

    async def _build_project_param_context() -> tuple[
        dict[str, list[str]], dict[str, dict[str, dict[str, str]]]
    ]:
        """Build graph-backed parameter name/value context for current test items."""
        if global_config.get("kg_schema_mode") != "electrical_controlled":
            return {}, {}
        if knowledge_graph_inst is None:
            return {}, {}

        configured_test_items = schema_cfg.get("test_items", []) or []
        configured_param_requirements = (
                schema_cfg.get("test_item_param_requirements", {}) or {}
        )
        if not configured_test_items:
            return {}, {}

        normalized_requirements: dict[str, list[str]] = {}
        if isinstance(configured_param_requirements, dict):
            for raw_test_name, raw_params in configured_param_requirements.items():
                normalized_test_name = _normalize_text_key(str(raw_test_name))
                if not normalized_test_name:
                    continue
                params = [
                    str(p).strip()
                    for p in (raw_params if isinstance(raw_params, list) else [])
                    if str(p).strip()
                ]
                if params:
                    normalized_requirements[normalized_test_name] = params

        def _get_config_required_params(test_name: str) -> list[str]:
            return list(normalized_requirements.get(_normalize_text_key(test_name), []))

        normalized_to_name: dict[str, str] = {}
        for test_name in configured_test_items:
            normalized = _normalize_text_key(str(test_name))
            if normalized and normalized not in normalized_to_name:
                normalized_to_name[normalized] = str(test_name)
        # Identify test items from current retrieved entities first.
        test_candidates: dict[str, str] = {}
        for entity_name, original in (entity_id_to_original or {}).items():
            if not isinstance(original, dict):
                continue
            raw_name = str(
                original.get("name")
                or original.get("test_item")
                or original.get("entity_name")
                or entity_name
            ).strip()
            normalized = _normalize_text_key(raw_name)
            canonical_name = normalized_to_name.get(normalized, "")
            if not canonical_name:
                continue
            test_id = str(original.get("entity_id", "") or "").strip() or _stable_test_id(
                canonical_name
            )
            test_candidates[canonical_name] = test_id

        # Fallback from relation endpoints if entity list has no explicit test item.
        if not test_candidates:
            for relation in relations_context:
                for key in ("entity1", "entity2"):
                    raw_name = str(relation.get(key, "") or "").strip()
                    normalized = _normalize_text_key(raw_name)
                    canonical_name = normalized_to_name.get(normalized, "")
                    if canonical_name and canonical_name not in test_candidates:
                        test_candidates[canonical_name] = _stable_test_id(canonical_name)

        current_report_scopes = _extract_current_report_scopes(rule_query_text, schema_cfg)
        if current_report_scopes:
            whitelist_map = _get_report_scope_test_whitelist(normalized_stand_type)
            scoped_test_items: set[str] = set()
            for scope in current_report_scopes:
                scoped_test_items.update(whitelist_map.get(str(scope).strip(), set()))
            for test_name in configured_test_items:
                canonical_name = str(test_name).strip()
                if canonical_name and canonical_name in scoped_test_items:
                    test_candidates.setdefault(
                        canonical_name, _stable_test_id(canonical_name)
                    )

        if not test_candidates:
            # Retrieval may miss explicit test-item entities. Fall back to config
            # whitelist so downstream QA still sees the full expected parameter set.
            fallback_map: dict[str, list[str]] = {}
            for test_name in configured_test_items:
                params = _get_config_required_params(str(test_name))
                if params:
                    fallback_map[str(test_name)] = params
            return fallback_map, {}

        project_param_map: dict[str, list[str]] = {}
        project_param_value_map: dict[str, dict[str, dict[str, str]]] = {}
        for test_name, test_id in test_candidates.items():
            edges = await knowledge_graph_inst.get_node_edges(test_id)
            if not edges:
                fallback_params = _get_config_required_params(test_name)
                if fallback_params:
                    project_param_map[test_name] = fallback_params
                continue
            param_ids: list[str] = []
            for src, tgt in edges:
                edge = await knowledge_graph_inst.get_edge(src, tgt)
                if (
                        edge
                        and edge.get("rel_type") == "HAS_PARAMETER"
                        and edge.get("src_id") == test_id
                ):
                    param_id = str(edge.get("tgt_id", "") or "").strip()
                    if param_id:
                        param_ids.append(param_id)
            if not param_ids:
                fallback_params = _get_config_required_params(test_name)
                if fallback_params:
                    project_param_map[test_name] = fallback_params
                continue
            ordered_param_ids = list(dict.fromkeys(param_ids))
            param_nodes = await knowledge_graph_inst.get_nodes_batch(ordered_param_ids)
            params: list[str] = []
            seen_param_names: set[str] = set()
            param_value_entries: dict[str, dict[str, str]] = {}
            for param_id in ordered_param_ids:
                node = param_nodes.get(param_id) if isinstance(param_nodes, dict) else None
                if not isinstance(node, dict):
                    continue
                param_name = str(
                    node.get("param_name", "") or node.get("name", "")
                ).strip()
                normalized = _normalize_text_key(param_name)
                if not normalized or normalized in seen_param_names:
                    continue
                seen_param_names.add(normalized)
                params.append(param_name)
                value_text = str(node.get("value_text", "") or "").strip()
                value_source = str(node.get("value_source", "") or "").strip()
                value_expr = str(node.get("value_expr", "") or "").strip()
                unit = str(node.get("unit", "") or "").strip()
                constraints = str(node.get("constraints", "") or "").strip()
                calc_rule = str(node.get("calc_rule", "") or "").strip()
                derive_from_rated = str(node.get("derive_from_rated", "") or "").strip()
                resolution_mode = _classify_query_value_resolution_mode(
                    value_text=value_text,
                    value_source=value_source,
                    value_expr=value_expr,
                    constraints=constraints,
                    calc_rule=calc_rule,
                    derive_from_rated=derive_from_rated,
                )
                if any((value_text, value_source, value_expr, unit, constraints, calc_rule, derive_from_rated)):
                    param_value_entries[param_name] = {
                        "value_text": value_text,
                        "value_source": value_source,
                        "value_expr": value_expr,
                        "unit": unit,
                        "constraints": constraints,
                        "calc_rule": calc_rule,
                        "derive_from_rated": derive_from_rated,
                        "resolution_mode": resolution_mode,
                    }
            if params:
                required_params = _get_config_required_params(test_name)
                project_param_map[test_name] = required_params or params
                if param_value_entries:
                    ordered_value_entries: dict[str, dict[str, str]] = {}
                    for param_name in project_param_map[test_name]:
                        if param_name in param_value_entries:
                            ordered_value_entries[param_name] = param_value_entries[param_name]
                    if ordered_value_entries:
                        project_param_value_map[test_name] = ordered_value_entries
            else:
                fallback_params = _get_config_required_params(test_name)
                if fallback_params:
                    project_param_map[test_name] = fallback_params

        if not project_param_map:
            fallback_map: dict[str, list[str]] = {}
            for test_name in configured_test_items:
                params = _get_config_required_params(str(test_name))
                if params:
                    fallback_map[str(test_name)] = params
            return fallback_map, {}
        return project_param_map, project_param_value_map

    project_param_map, project_param_value_map = await _build_project_param_context()
    project_param_map, project_param_value_map = _apply_domain_rule_decisions_to_project_context(
        project_param_map,
        project_param_value_map,
        domain_rule_decisions,
        schema_cfg,
        rule_query_text,
        stand_type=stand_type
    )
    _log_electrical_trace(
        "post_rule_application",
        project_param_map_keys=list(project_param_map.keys()),
        project_param_value_map_keys=list(project_param_value_map.keys()),
        project_param_map=project_param_map,
        project_param_value_map=project_param_value_map,
    )
    project_param_map_raw = deepcopy(project_param_map)
    project_param_value_map_raw = deepcopy(project_param_value_map)
    current_report_scopes = _extract_current_report_scopes(rule_query_text, schema_cfg)
    scoped_out_test_items: list[str] = []
    if current_report_scopes:
        pre_scope_test_items = {
            str(test_name).strip()
            for test_name in project_param_map.keys()
            if str(test_name).strip()
        }
        project_param_map, project_param_value_map = _filter_project_context_by_report_scope(
            project_param_map,
            project_param_value_map,
            current_report_scopes,
            domain_rule_decisions,
            stand_type=normalized_stand_type,
        )
        post_scope_test_items = {
            str(test_name).strip()
            for test_name in project_param_map.keys()
            if str(test_name).strip()
        }
        scoped_out_test_items = sorted(pre_scope_test_items - post_scope_test_items)
    suppressed_display_params = _get_display_param_suppressions()
    display_project_param_map: dict[str, list[str]] = {}
    display_project_param_value_map: dict[str, dict[str, dict[str, str]]] = {}
    for test_name, params in project_param_map.items():
        suppressed = suppressed_display_params.get(str(test_name), set())
        display_project_param_map[str(test_name)] = [
            str(param)
            for param in (params if isinstance(params, list) else [])
            if str(param) and str(param) not in suppressed
        ]
        raw_values = project_param_value_map.get(str(test_name), {}) or {}
        if isinstance(raw_values, dict):
            filtered_values = {
                str(param_name): deepcopy(param_value)
                for param_name, param_value in raw_values.items()
                if str(param_name) and str(param_name) not in suppressed
            }
            if filtered_values:
                display_project_param_value_map[str(test_name)] = filtered_values
    resolved_rule_overrides = _build_resolved_rule_overrides(domain_rule_decisions)
    allowed_final_test_items, removed_test_items = _build_final_test_item_scope(
        project_param_map,
        domain_rule_decisions,
    )
    if scoped_out_test_items:
        removed_test_items = sorted(
            {str(item).strip() for item in (removed_test_items + scoped_out_test_items) if str(item).strip()}
        )
    test_item_display_map = _build_test_item_display_map(display_project_param_map)
    allowed_final_test_items_display = [
        test_item_display_map.get(str(item).strip(), str(item).strip())
        for item in allowed_final_test_items
        if str(item).strip()
    ]
    removed_test_items_display = [
        test_item_display_map.get(str(item).strip(), str(item).strip())
        for item in removed_test_items
        if str(item).strip()
    ]
    entities_context, relations_context = _filter_context_by_final_test_item_scope(
        entities_context,
        relations_context,
        removed_test_items,
    )
    project_param_map_str = (
        json.dumps(display_project_param_map, ensure_ascii=False, indent=2)
        if display_project_param_map
        else "{}"
    )
    project_param_value_map_str = (
        json.dumps(display_project_param_value_map, ensure_ascii=False, indent=2)
        if display_project_param_value_map
        else "{}"
    )
    domain_rule_decisions_str = (
        json.dumps(domain_rule_decisions, ensure_ascii=False, indent=2)
        if domain_rule_decisions
        else "{}"
    )
    resolved_rule_overrides_str = (
        json.dumps(resolved_rule_overrides, ensure_ascii=False, indent=2)
        if resolved_rule_overrides
        else "{}"
    )
    allowed_final_test_items_str = (
        json.dumps(allowed_final_test_items_display, ensure_ascii=False, indent=2)
        if allowed_final_test_items_display
        else "[]"
    )
    removed_test_items_str = (
        json.dumps(removed_test_items_display, ensure_ascii=False, indent=2)
        if removed_test_items_display
        else "[]"
    )
    test_item_display_map_str = (
        json.dumps(test_item_display_map, ensure_ascii=False, indent=2)
        if test_item_display_map
        else "{}"
    )

    entities_str = "\n".join(
        json.dumps(entity, ensure_ascii=False) for entity in entities_context
    )
    relations_str = "\n".join(
        json.dumps(relation, ensure_ascii=False) for relation in relations_context
    )

    # Calculate preliminary kg context tokens
    pre_kg_context = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        project_param_map_str=project_param_map_str,
        project_param_value_map_str=project_param_value_map_str,
        domain_rule_decisions_str=domain_rule_decisions_str,
        resolved_rule_overrides_str=resolved_rule_overrides_str,
        allowed_final_test_items_str=allowed_final_test_items_str,
        removed_test_items_str=removed_test_items_str,
        text_chunks_str="",
        reference_list_str="",
    )
    kg_context_tokens = len(tokenizer.encode(pre_kg_context))

    # Calculate preliminary system prompt tokens
    pre_sys_prompt = sys_prompt_template.format(
        context_data="",  # Empty for overhead calculation
        response_type=response_type,
        user_prompt=user_prompt,
    )
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))

    # Calculate available tokens for text chunks
    query_tokens = len(tokenizer.encode(query))
    buffer_tokens = 200  # reserved for reference list and safety buffer
    available_chunk_tokens = max_total_tokens - (
            sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
    )

    # Keep a minimum chunk budget in mix mode to avoid "retrieved but empty chunk context".
    if available_chunk_tokens <= 0 and query_param.mode == "mix":
        logger.warning(
            "Chunk token budget exhausted in mix mode (available=%d). Dropping entity/relation context to preserve chunk recall.",
            available_chunk_tokens,
        )
        entities_context = []
        relations_context = []
        entities_str = ""
        relations_str = ""
        pre_kg_context = kg_context_template.format(
            entities_str=entities_str,
            relations_str=relations_str,
            project_param_map_str=project_param_map_str,
            project_param_value_map_str=project_param_value_map_str,
            domain_rule_decisions_str=domain_rule_decisions_str,
            resolved_rule_overrides_str=resolved_rule_overrides_str,
            text_chunks_str="",
            reference_list_str="",
        )
        kg_context_tokens = len(tokenizer.encode(pre_kg_context))
        available_chunk_tokens = max_total_tokens - (
                sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
        )

    if available_chunk_tokens <= 0:
        fallback_chunk_budget = max(200, min(1200, max_total_tokens // 4))
        logger.warning(
            "Chunk token budget is still non-positive (%d), using fallback chunk budget %d",
            available_chunk_tokens,
            fallback_chunk_budget,
        )
        available_chunk_tokens = fallback_chunk_budget

    logger.debug(
        f"Token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, KG: {kg_context_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Apply token truncation to chunks using the dynamic limit
    truncated_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=merged_chunks,
        query_param=query_param,
        global_config=global_config,
        source_type=query_param.mode,
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
    )

    # Generate reference list from truncated chunks using the new common function
    reference_list, truncated_chunks = generate_reference_list_from_chunks(
        truncated_chunks
    )

    # Rebuild chunks_context with truncated chunks
    # The actual tokens may be slightly less than available_chunk_tokens due to deduplication logic
    chunks_context = []
    for i, chunk in enumerate(truncated_chunks):
        chunks_context.append(
            {
                "reference_id": chunk["reference_id"],
                "content": chunk["content"],
            }
        )

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    logger.info(
        f"Final context: {len(entities_context)} entities, {len(relations_context)} relations, {len(chunks_context)} chunks"
    )

    # not necessary to use LLM to generate a response
    if not entities_context and not relations_context and not chunks_context:
        # Return empty raw data structure when no entities/relations
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Query returned empty dataset."
        return "", empty_raw_data

    # output chunks tracking infomations
    # format: <source><frequency>/<order> (e.g., E5/2 R2/1 C1/1)
    if truncated_chunks and chunk_tracking:
        chunk_tracking_log = []
        for chunk in truncated_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id in chunk_tracking:
                tracking_info = chunk_tracking[chunk_id]
                source = tracking_info["source"]
                frequency = tracking_info["frequency"]
                order = tracking_info["order"]
                chunk_tracking_log.append(f"{source}{frequency}/{order}")
            else:
                chunk_tracking_log.append("?0/0")

        if chunk_tracking_log:
            logger.info(f"Final chunks S+F/O: {' '.join(chunk_tracking_log)}")

    result = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        project_param_map_str=project_param_map_str,
        project_param_value_map_str=project_param_value_map_str,
        domain_rule_decisions_str=domain_rule_decisions_str,
        resolved_rule_overrides_str=resolved_rule_overrides_str,
        allowed_final_test_items_str=allowed_final_test_items_str,
        removed_test_items_str=removed_test_items_str,
        test_item_display_map_str=test_item_display_map_str,
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )

    # Always return both context and complete data structure (unified approach)
    logger.debug(
        f"[_build_context_str] Converting to user format: {len(entities_context)} entities, {len(relations_context)} relations, {len(truncated_chunks)} chunks"
    )
    final_data = convert_to_user_format(
        entities_context,
        relations_context,
        truncated_chunks,
        reference_list,
        query_param.mode,
        entity_id_to_original,
        relation_id_to_original,
    )
    if "metadata" not in final_data:
        final_data["metadata"] = {}
    final_data["metadata"]["project_param_map_raw"] = project_param_map_raw
    final_data["metadata"]["project_param_map"] = display_project_param_map
    final_data["metadata"]["project_param_value_map_raw"] = project_param_value_map_raw
    final_data["metadata"]["project_param_value_map"] = display_project_param_value_map
    final_data["metadata"]["current_report_scopes"] = current_report_scopes
    final_data["metadata"]["project_param_value_map_display"] = display_project_param_value_map
    final_data["metadata"]["rule_query_text"] = rule_query_text
    final_data["metadata"]["domain_rule_decisions"] = domain_rule_decisions
    final_data["metadata"]["resolved_rule_overrides"] = resolved_rule_overrides
    final_data["metadata"]["allowed_final_test_items_raw"] = allowed_final_test_items
    final_data["metadata"]["allowed_final_test_items"] = allowed_final_test_items_display
    final_data["metadata"]["removed_test_items_raw"] = removed_test_items
    final_data["metadata"]["removed_test_items"] = removed_test_items_display
    final_data["metadata"]["test_item_display_map"] = test_item_display_map
    final_data["metadata"]["project_split_rules"] = {
        "pf_fracture_enabled": fracture_pf_enabled,
        "li_fracture_enabled": fracture_li_enabled,
        "pf_fracture_provided": fracture_pf_provided,
        "li_fracture_provided": fracture_li_provided,
        "partial_discharge_enabled": pd_allowed,
        "partial_discharge_by_explicit_solid_sealed_pole": explicit_solid_sealed_pole,
        "partial_discharge_by_model_rule": pd_allowed_by_model,
        "partial_discharge_by_voltage_rule": pd_allowed_by_voltage,
    }
    logger.debug(
        f"[_build_context_str] Final data after conversion: {len(final_data.get('entities', []))} entities, {len(final_data.get('relationships', []))} relationships, {len(final_data.get('chunks', []))} chunks"
    )
    return result, final_data


# Now let's update the old _build_query_context to use the new architecture
async def _build_query_context(
        query: str,
        rule_query: str | None,
        ll_keywords: str,
        hl_keywords: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
        chunks_vdb: BaseVectorStorage = None,
        stand_type: str | None = None
) -> QueryContextResult | None:
    """
    Main query context building function using the new 4-stage architecture:
    1. Search -> 2. Truncate -> 3. Merge chunks -> 4. Build LLM context

    Returns unified QueryContextResult containing both context and raw_data.
    """

    if not query:
        logger.warning("Query is empty, skipping context building")
        return None

    # Stage 1: Pure search
    search_result = await _perform_kg_search(
        query,
        ll_keywords,
        hl_keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )

    doc_type_filters = _extract_doc_type_filters(query)
    if doc_type_filters:
        search_result = _filter_search_result_by_doc_type(
            search_result, doc_type_filters
        )

    if not search_result["final_entities"] and not search_result["final_relations"]:
        if query_param.mode != "mix":
            return None
        else:
            if not search_result["vector_chunks"]:
                return None

    # Stage 2: Apply token truncation for LLM efficiency
    truncation_result = await _apply_token_truncation(
        search_result,
        query_param,
        text_chunks_db.global_config,
    )

    # Stage 3: Merge chunks using filtered entities/relations
    merged_chunks = await _merge_all_chunks(
        filtered_entities=truncation_result["filtered_entities"],
        filtered_relations=truncation_result["filtered_relations"],
        vector_chunks=search_result["vector_chunks"],
        query=query,
        knowledge_graph_inst=knowledge_graph_inst,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
        chunks_vdb=chunks_vdb,
        chunk_tracking=search_result["chunk_tracking"],
        query_embedding=search_result["query_embedding"],
    )

    if (
            not merged_chunks
            and not truncation_result["entities_context"]
            and not truncation_result["relations_context"]
    ):
        return None

    # Stage 4: Build final LLM context with dynamic token processing
    # _build_context_str now always returns tuple[str, dict]
    context, raw_data = await _build_context_str(
        entities_context=truncation_result["entities_context"],
        relations_context=truncation_result["relations_context"],
        merged_chunks=merged_chunks,
        query=query,
        rule_query=rule_query,
        query_param=query_param,
        global_config=text_chunks_db.global_config,
        chunk_tracking=search_result["chunk_tracking"],
        entity_id_to_original=truncation_result["entity_id_to_original"],
        relation_id_to_original=truncation_result["relation_id_to_original"],
        knowledge_graph_inst=knowledge_graph_inst,
        stand_type=stand_type,
    )

    # Convert keywords strings to lists and add complete metadata to raw_data
    hl_keywords_list = hl_keywords.split(", ") if hl_keywords else []
    ll_keywords_list = ll_keywords.split(", ") if ll_keywords else []

    # Add complete metadata to raw_data (preserve existing metadata including query_mode)
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}

    # Update keywords while preserving existing metadata
    raw_data["metadata"]["keywords"] = {
        "high_level": hl_keywords_list,
        "low_level": ll_keywords_list,
    }
    raw_data["metadata"]["processing_info"] = {
        "total_entities_found": len(search_result.get("final_entities", [])),
        "total_relations_found": len(search_result.get("final_relations", [])),
        "entities_after_truncation": len(
            truncation_result.get("filtered_entities", [])
        ),
        "relations_after_truncation": len(
            truncation_result.get("filtered_relations", [])
        ),
        "merged_chunks_count": len(merged_chunks),
        "final_chunks_count": len(raw_data.get("data", {}).get("chunks", [])),
    }

    logger.debug(
        f"[_build_query_context] Context length: {len(context) if context else 0}"
    )
    logger.debug(
        f"[_build_query_context] Raw data entities: {len(raw_data.get('data', {}).get('entities', []))}, relationships: {len(raw_data.get('data', {}).get('relationships', []))}, chunks: {len(raw_data.get('data', {}).get('chunks', []))}"
    )

    return QueryContextResult(context=context, raw_data=raw_data)


async def _get_node_data(
        query: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        query_param: QueryParam,
):
    # get similar entities
    logger.info(
        f"Query nodes: {query} (top_k:{query_param.top_k}, cosine:{entities_vdb.cosine_better_than_threshold})"
    )

    results = await entities_vdb.query(query, top_k=query_param.top_k)

    if not len(results):
        return [], []

    # Extract all entity IDs from your results list
    node_ids = [r["entity_name"] for r in results]

    # Call the batch node retrieval and degree functions concurrently.
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(node_ids),
        knowledge_graph_inst.node_degrees_batch(node_ids),
    )

    # Now, if you need the node data and degree in order:
    node_datas = [nodes_dict.get(nid) for nid in node_ids]
    node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {
            **n,
            "entity_name": k["entity_name"],
            "rank": d,
            "created_at": k.get("created_at"),
        }
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    use_relations = await _find_most_related_edges_from_entities(
        node_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Local query: {len(node_datas)} entites, {len(use_relations)} relations"
    )

    # Entities are sorted by cosine similarity
    # Relations are sorted by rank + weight
    return node_datas, use_relations


async def _find_most_related_edges_from_entities(
        node_datas: list[dict],
        query_param: QueryParam,
        knowledge_graph_inst: BaseGraphStorage,
):
    node_names = [dp["entity_name"] for dp in node_datas]
    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)

    all_edges = []
    seen = set()

    for node_name in node_names:
        this_edges = batch_edges_dict.get(node_name, [])
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
    # For edge degrees, use tuples.
    edge_pairs_tuples = list(all_edges)  # all_edges is already a list of tuples

    # Call the batched functions concurrently.
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
    )

    # Reconstruct edge_datas list in the same order as the deduplicated results.
    all_edges_data = []
    for pair in all_edges:
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            combined = {
                "src_tgt": pair,
                "rank": edge_degrees_dict.get(pair, 0),
                **edge_props,
            }
            all_edges_data.append(combined)

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    return all_edges_data


async def _find_related_text_unit_from_entities(
        node_datas: list[dict],
        query_param: QueryParam,
        text_chunks_db: BaseKVStorage,
        knowledge_graph_inst: BaseGraphStorage,
        query: str = None,
        chunks_vdb: BaseVectorStorage = None,
        chunk_tracking: dict = None,
        query_embedding=None,
):
    """
    Find text chunks related to entities using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(node_datas)} entities")

    if not node_datas:
        return []

    # Step 1: Collect all text chunks for each entity
    entities_with_chunks = []
    for entity in node_datas:
        if entity.get("source_id"):
            chunks = split_string_by_multi_markers(
                entity["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                entities_with_chunks.append(
                    {
                        "entity_name": entity["entity_name"],
                        "chunks": chunks,
                        "entity_data": entity,
                    }
                )

    if not entities_with_chunks:
        logger.warning("No entities with text chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned entities)
    chunk_occurrence_count = {}
    for entity_info in entities_with_chunks:
        deduplicated_chunks = []
        for chunk_id in entity_info["chunks"]:
            chunk_occurrence_count[chunk_id] = (
                    chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier entity, so skip it

        # Update entity's chunks to deduplicated chunks
        entity_info["chunks"] = deduplicated_chunks

    # Step 3: Sort chunks for each entity by occurrence count (higher count = higher priority)
    total_entity_chunks = 0
    for entity_info in entities_with_chunks:
        sorted_chunks = sorted(
            entity_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        entity_info["sorted_chunks"] = sorted_chunks
        total_entity_chunks += len(sorted_chunks)

    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    # Step 4: Apply the selected chunk selection algorithm
    # Pick by vector similarity:
    #     The order of text chunks aligns with the naive retrieval's destination.
    #     When reranking is disabled, the text chunks delivered to the LLM tend to favor naive retrieval.
    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(entities_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=entities_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No entity-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Pick by entity and chunk weight:
        #     When reranking is disabled, delivered more solely KG related chunks to the LLM
        selected_chunk_ids = pick_by_weighted_polling(
            entities_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by weighted polling"
        )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "entity"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "E",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final entity-related results
                }

    return result_chunks


async def _get_edge_data(
        keywords,
        knowledge_graph_inst: BaseGraphStorage,
        relationships_vdb: BaseVectorStorage,
        query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords} (top_k:{query_param.top_k}, cosine:{relationships_vdb.cosine_better_than_threshold})"
    )

    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return [], []

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
    edge_data_dict = await knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)

    # Reconstruct edge_datas list in the same order as results.
    edge_datas = []
    for k in results:
        pair = (k["src_id"], k["tgt_id"])
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            # Keep edge data without rank, maintain vector search order
            combined = {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "created_at": k.get("created_at", None),
                **edge_props,
            }
            edge_datas.append(combined)

    # Relations maintain vector search order (sorted by similarity)

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Global query: {len(use_entities)} entites, {len(edge_datas)} relations"
    )

    return edge_datas, use_entities


async def _find_most_related_entities_from_relationships(
        edge_datas: list[dict],
        query_param: QueryParam,
        knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    # Only get nodes data, no need for node degrees
    nodes_dict = await knowledge_graph_inst.get_nodes_batch(entity_names)

    # Rebuild the list in the same order as entity_names
    node_datas = []
    for entity_name in entity_names:
        node = nodes_dict.get(entity_name)
        if node is None:
            logger.warning(f"Node '{entity_name}' not found in batch retrieval.")
            continue
        # Combine the node data with the entity name, no rank needed
        combined = {**node, "entity_name": entity_name}
        node_datas.append(combined)

    return node_datas


async def _find_related_text_unit_from_relations(
        edge_datas: list[dict],
        query_param: QueryParam,
        text_chunks_db: BaseKVStorage,
        entity_chunks: list[dict] = None,
        query: str = None,
        chunks_vdb: BaseVectorStorage = None,
        chunk_tracking: dict = None,
        query_embedding=None,
):
    """
    Find text chunks related to relationships using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(edge_datas)} relations")

    if not edge_datas:
        return []

    # Step 1: Collect all text chunks for each relationship
    relations_with_chunks = []
    for relation in edge_datas:
        if relation.get("source_id"):
            chunks = split_string_by_multi_markers(
                relation["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                # Build relation identifier
                if "src_tgt" in relation:
                    rel_key = tuple(sorted(relation["src_tgt"]))
                else:
                    rel_key = tuple(
                        sorted([relation.get("src_id"), relation.get("tgt_id")])
                    )

                relations_with_chunks.append(
                    {
                        "relation_key": rel_key,
                        "chunks": chunks,
                        "relation_data": relation,
                    }
                )

    if not relations_with_chunks:
        logger.warning("No relation-related chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned relationships)
    # Also remove duplicates with entity_chunks

    # Extract chunk IDs from entity_chunks for deduplication
    entity_chunk_ids = set()
    if entity_chunks:
        for chunk in entity_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                entity_chunk_ids.add(chunk_id)

    chunk_occurrence_count = {}
    # Track unique chunk_ids that have been removed to avoid double counting
    removed_entity_chunk_ids = set()

    for relation_info in relations_with_chunks:
        deduplicated_chunks = []
        for chunk_id in relation_info["chunks"]:
            # Skip chunks that already exist in entity_chunks
            if chunk_id in entity_chunk_ids:
                # Only count each unique chunk_id once
                removed_entity_chunk_ids.add(chunk_id)
                continue

            chunk_occurrence_count[chunk_id] = (
                    chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier relationship, so skip it

        # Update relationship's chunks to deduplicated chunks
        relation_info["chunks"] = deduplicated_chunks

    # Check if any relations still have chunks after deduplication
    relations_with_chunks = [
        relation_info
        for relation_info in relations_with_chunks
        if relation_info["chunks"]
    ]

    if not relations_with_chunks:
        logger.info(
            f"Find no additional relations-related chunks from {len(edge_datas)} relations"
        )
        return []

    # Step 3: Sort chunks for each relationship by occurrence count (higher count = higher priority)
    total_relation_chunks = 0
    for relation_info in relations_with_chunks:
        sorted_chunks = sorted(
            relation_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        relation_info["sorted_chunks"] = sorted_chunks
        total_relation_chunks += len(sorted_chunks)

    logger.info(
        f"Find {total_relation_chunks} additional chunks in {len(relations_with_chunks)} relations (deduplicated {len(removed_entity_chunk_ids)})"
    )

    # Step 4: Apply the selected chunk selection algorithm
    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(relations_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=relations_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No relation-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Apply linear gradient weighted polling algorithm
        selected_chunk_ids = pick_by_weighted_polling(
            relations_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by weighted polling"
        )

    logger.debug(
        f"KG related chunks: {len(entity_chunks)} from entitys, {len(selected_chunk_ids)} from relations"
    )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "relationship"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "R",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final relation-related results
                }

    return result_chunks


@overload
async def naive_query(
        query: str,
        chunks_vdb: BaseVectorStorage,
        query_param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
        system_prompt: str | None = None,
        return_raw_data: Literal[True] = True,
) -> dict[str, Any]: ...


@overload
async def naive_query(
        query: str,
        chunks_vdb: BaseVectorStorage,
        query_param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
        system_prompt: str | None = None,
        return_raw_data: Literal[False] = False,
) -> str | AsyncIterator[str]: ...


async def naive_query(
        query: str,
        chunks_vdb: BaseVectorStorage,
        query_param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
        system_prompt: str | None = None,
) -> QueryResult | None:
    """
    Execute naive query and return unified QueryResult object.

    Args:
        query: Query string
        chunks_vdb: Document chunks vector database
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Returns None when no relevant chunks are retrieved.
    """

    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    tokenizer: Tokenizer = global_config["tokenizer"]
    if not tokenizer:
        logger.error("Tokenizer not found in global configuration.")
        return QueryResult(content=PROMPTS["fail_response"])

    chunks = await _get_vector_context(query, chunks_vdb, query_param, None)

    if chunks is None or len(chunks) == 0:
        logger.info(
            "[naive_query] No relevant document chunks found; returning no-result."
        )
        return None

    # Calculate dynamic token limit for chunks
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Calculate system prompt template tokens (excluding content_data)
    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Use the provided system prompt or default
    sys_prompt_template = (
        system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    )

    # Create a preliminary system prompt with empty content_data to calculate overhead
    pre_sys_prompt = sys_prompt_template.format(
        response_type=response_type,
        user_prompt=user_prompt,
        content_data="",  # Empty for overhead calculation
    )

    # Calculate available tokens for chunks
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))
    query_tokens = len(tokenizer.encode(query))
    buffer_tokens = 200  # reserved for reference list and safety buffer
    available_chunk_tokens = max_total_tokens - (
            sys_prompt_tokens + query_tokens + buffer_tokens
    )

    logger.debug(
        f"Naive query token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Process chunks using unified processing with dynamic token limit
    processed_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=chunks,
        query_param=query_param,
        global_config=global_config,
        source_type="vector",
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
    )

    # Generate reference list from processed chunks using the new common function
    reference_list, processed_chunks_with_ref_ids = generate_reference_list_from_chunks(
        processed_chunks
    )

    logger.info(f"Final context: {len(processed_chunks_with_ref_ids)} chunks")

    # Build raw data structure for naive mode using processed chunks with reference IDs
    raw_data = convert_to_user_format(
        [],  # naive mode has no entities
        [],  # naive mode has no relationships
        processed_chunks_with_ref_ids,
        reference_list,
        "naive",
    )

    # Add complete metadata for naive mode
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}
    raw_data["metadata"]["keywords"] = {
        "high_level": [],  # naive mode has no keyword extraction
        "low_level": [],  # naive mode has no keyword extraction
    }
    raw_data["metadata"]["processing_info"] = {
        "total_chunks_found": len(chunks),
        "final_chunks_count": len(processed_chunks_with_ref_ids),
    }

    # Build chunks_context from processed chunks with reference IDs
    chunks_context = []
    for i, chunk in enumerate(processed_chunks_with_ref_ids):
        chunks_context.append(
            {
                "reference_id": chunk["reference_id"],
                "content": chunk["content"],
            }
        )

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    naive_context_template = PROMPTS["naive_query_context"]
    context_content = naive_context_template.format(
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )

    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(content=context_content, raw_data=raw_data)

    sys_prompt = sys_prompt_template.format(
        response_type=query_param.response_type,
        user_prompt=user_prompt,
        content_data=context_content,
    )

    user_query = query

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=raw_data)

    # Handle cache
    _log_electrical_answer_debug("naive_pre_llm", raw_data)
    bypass_query_cache = _should_bypass_query_cache(global_config)
    args_hash = compute_args_hash(
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        query_param.user_prompt or "",
        query_param.enable_rerank,
    )
    cached_result = None
    if not bypass_query_cache:
        cached_result = await handle_cache(
            hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
        )
    else:
        logger.info("[naive_query] Bypassing query cache for electrical controlled mode")
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True,
            stream=query_param.stream,
        )

        if (
            not bypass_query_cache
            and hashing_kv
            and hashing_kv.global_config.get("enable_llm_cache")
        ):
            queryparam_dict = {
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        if len(response) > len(sys_prompt):
            response = (
                response[len(sys_prompt):]
                .replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        _log_electrical_answer_debug("naive_before_postprocess", raw_data, response)
        response = _postprocess_electrical_markdown_response(
            _enforce_formula_consistency(response),
            raw_data,
        )
        _log_electrical_answer_debug("naive_after_postprocess", raw_data, response)

        return QueryResult(content=response, raw_data=raw_data)
    else:
        # Streaming response (AsyncIterator)
        _log_electrical_answer_debug("naive_stream_return", raw_data)
        return QueryResult(
            response_iterator=_stream_electrical_response_with_final_event(
                response,
                raw_data,
                sys_prompt,
                query,
                "naive_stream",
            ),
            raw_data=raw_data,
            is_streaming=True,
        )