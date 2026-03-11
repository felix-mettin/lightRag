#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip()).lower()


def is_malformed_add_item(item: dict[str, Any]) -> bool:
    report_type = str(item.get("report_type") or "")
    test_item = str(item.get("test_item") or item.get("test_name") or "")
    if report_type != "型式试验":
        return False
    return True


def is_bad_param_name(param_name: str) -> bool:
    text = str(param_name or "").strip()
    if not text:
        return True
    if text.startswith("最大适用海拔:"):
        return True
    return False


def normalize_value_text(value_text: str) -> str:
    text = str(value_text or "").strip()
    replacements = {
        "额定电压（用户录入）": "用户录入额定电压",
        "用户录入（试验电压基础）": "用户录入额定电压",
        "制造厂）": "制造厂给定",
        "75%×额定短路开断电流（用户录入）": "用户录入额定短路开断电流的75%",
        "87%×额定短路开断电流（用户录入）": "用户录入额定短路开断电流的87%",
    }
    return replacements.get(text, text)


def infer_better_value_source(param_name: str, value_text: str, value_source: str) -> str:
    text = normalize_value_text(value_text)
    source = str(value_source or "").strip() or "standard"

    has_user = any(token in text for token in ("用户录入", "用户输入", "用户提供", "客户录入", "客户输入", "客户提供"))
    has_default = "默认" in text or "未输入" in text or "未录入" in text
    has_formula = any(token in text for token in ("公式", "%", "×", "*", "除2", "除根", "乘", "倍"))

    if has_formula:
        return "formula"
    if has_default:
        return "default"
    if has_user:
        return "user_input"

    # Naked literals like "1" are fixed values, not defaults.
    if source == "default" and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:次|相|kV|A|kA|ms|s|min)?", text):
        return "standard"

    return source


def normalize_param(param: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(param)
    value_text = normalize_value_text(str(cleaned.get("value_text") or ""))
    value_source = infer_better_value_source(
        str(cleaned.get("param_name") or cleaned.get("param_key") or ""),
        value_text,
        str(cleaned.get("value_source") or ""),
    )

    cleaned["value_text"] = value_text
    cleaned["value_source"] = value_source
    cleaned["value_type"] = value_source
    cleaned["constraints"] = value_text
    cleaned["calc_rule"] = value_text if value_source == "formula" else ""
    if value_source == "user_input":
        cleaned["value_expr"] = value_text
        cleaned["derive_from_rated"] = value_text
    elif value_source == "default":
        cleaned["value_expr"] = value_text
        cleaned["derive_from_rated"] = ""
    else:
        cleaned["value_expr"] = ""
        cleaned["derive_from_rated"] = ""

    return cleaned


def dedupe_params(params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for param in params:
        if not isinstance(param, dict):
            continue
        if is_bad_param_name(str(param.get("param_name") or param.get("param_key") or "")):
            continue
        param = normalize_param(param)
        key = normalize(str(param.get("param_key") or param.get("param_name") or ""))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        cleaned.append(param)
    return cleaned


def rebuild_tests_by_name(payload: dict[str, Any]) -> None:
    tests_by_name: dict[str, list[dict[str, Any]]] = {}
    for rule in (payload.get("tests_by_path") or {}).values():
        if not isinstance(rule, dict):
            continue
        key = normalize(str(rule.get("test_name") or ""))
        if not key:
            continue
        tests_by_name.setdefault(key, []).append(rule)
    payload["tests_by_name"] = tests_by_name


def clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tests_by_path = payload.get("tests_by_path") or {}
    for rule in tests_by_path.values():
        if not isinstance(rule, dict):
            continue
        rule["parameters"] = dedupe_params(list(rule.get("parameters") or []))

    add_items = []
    seen_add_keys: set[tuple[str, str, str]] = set()
    for item in payload.get("add_test_items") or []:
        if not isinstance(item, dict):
            continue
        if is_malformed_add_item(item):
            continue
        item["parameters"] = dedupe_params(list(item.get("parameters") or []))
        key = (
            str(item.get("report_type") or ""),
            str(item.get("category") or ""),
            str(item.get("test_item") or item.get("test_name") or ""),
        )
        if key in seen_add_keys:
            continue
        seen_add_keys.add(key)
        add_items.append(item)
    payload["add_test_items"] = add_items

    rebuild_tests_by_name(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean malformed historical entries from annotation_memory.json.")
    parser.add_argument("input", help="Input annotation_memory.json path")
    parser.add_argument("--output", help="Output path; defaults to overwrite input")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    cleaned = clean_payload(payload)
    output_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"cleaned: tests_by_path={len(cleaned.get('tests_by_path') or {})}, "
        f"tests_by_name={len(cleaned.get('tests_by_name') or {})}, "
        f"add_test_items={len(cleaned.get('add_test_items') or [])}"
    )


if __name__ == "__main__":
    main()
