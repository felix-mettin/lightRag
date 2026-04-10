from __future__ import annotations

from pathlib import Path

STANDARD_TYPES = ["GB", "DLT", "IEC", "others"]

STANDARD_TYPE_KEYWORDS = {
    "GB": ["国标", "国家标准", "GB/T", "GBT", "GB-T", "GB", "gb/t", "gbt", "gb-t", "gb"],
    "DLT": ["行标", "行业标准", "HB", "hb", "DL/T", "DLT", "DL-T", "DL", "dl/t", "dlt", "dl-t", "dl"],
    "IEC": ["国际标准", "国际标", "IEC", "ICE", "ISO", "IEEE", "GJB", "gjb", "iec", "ice", "iso", "ieee", "国际"],
}

STANDARD_CONFIG_FILES = {
    "GB": "config_gb.ini",
    "DLT": "config_dlt.ini",
    "IEC": "config_ice.ini",
    "others": "config.ini",
}

STANDARD_WORKSPACES = {
    "GB": "GB",
    "DLT": "HB",
    "IEC": "GJB",
    "others": "others",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_standard_type(value: str | None) -> str | None:
    if not value:
        return None

    normalized = value.strip().upper()
    if normalized in {"GB", "DLT", "IEC", "OTHERS"}:
        return "others" if normalized == "OTHERS" else normalized

    alias_map = {
        "HB": "DLT",
        "DL/T": "DLT",
        "DL-T": "DLT",
        "GJB": "IEC",
        "ICE": "IEC",
        "ISO": "IEC",
        "IEEE": "IEC",
    }
    if normalized in alias_map:
        return alias_map[normalized]

    lowered = value.strip().lower()
    for standard_type, keywords in STANDARD_TYPE_KEYWORDS.items():
        if any(keyword.lower() == lowered for keyword in keywords):
            return standard_type

    return None


def detect_standard_types_from_query(query: str, default_standard: str = "GB") -> list[str]:
    if not query:
        return [default_standard]

    query_upper = query.upper()
    matches: list[tuple[int, str]] = []
    for standard_type, keywords in STANDARD_TYPE_KEYWORDS.items():
        positions = [query_upper.find(keyword.upper()) for keyword in keywords]
        positions = [position for position in positions if position >= 0]
        if positions:
            matches.append((min(positions), standard_type))

    if not matches:
        return [default_standard]

    ordered: list[str] = []
    seen: set[str] = set()
    for _, standard_type in sorted(matches, key=lambda item: item[0]):
        if standard_type not in seen:
            seen.add(standard_type)
            ordered.append(standard_type)
    return ordered


def detect_standard_type_from_query(query: str, default_standard: str = "GB") -> str:
    return detect_standard_types_from_query(query, default_standard=default_standard)[0]


def get_standard_config_path(standard_type: str | None) -> str:
    normalized = normalize_standard_type(standard_type) or "others"
    config_name = STANDARD_CONFIG_FILES.get(normalized, STANDARD_CONFIG_FILES["others"])
    return str((_PROJECT_ROOT / config_name).resolve())


def get_standard_workspace(standard_type: str | None) -> str:
    normalized = normalize_standard_type(standard_type) or "others"
    return STANDARD_WORKSPACES.get(normalized, STANDARD_WORKSPACES["others"])