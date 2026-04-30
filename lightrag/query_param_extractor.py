"""
Query parameter extractor for electrical test queries.

Extracts structured parameters (e.g. altitude, voltage, current) from
user query text. Designed for easy extension with additional parameters.

Parameters with dynamic defaults (dependent on other extracted values)
are resolved via _computed_params after the first extraction pass.

Each parameter also generates a human-readable description explaining
how the value was determined (user-provided vs default fallback).
"""

import re
from typing import Any, Callable


class QueryParamExtractor:
    """Extracts structured parameters from electrical test query text.

    Each parameter is defined as a dict with:
      - key: internal parameter name
      - pattern: regex to extract the value from query text
      - default: fallback value when not found in query
      - cast: type conversion function (optional)
      - description: human-readable parameter name (for generated descriptions)

    Parameters whose default depends on other extracted values (e.g.
    首开极系数 depends on 额定电压) are handled by _computed_params.

    Usage:
        extractor = QueryParamExtractor()
        result = extractor.extract("型号名称：LW30B-550&罐式断路器 ...")
        # result == {
        #     "values": {"altitude_m": 1000, "rated_voltage_kv": 550.0, "first_pole_kpp": 1.3},
        #     "descriptions": {
        #         "altitude_m": "当前未检测到用户提供的最大(适用)的海拔，默认按1000m回填。",
        #         "first_pole_kpp": "额定电压 550.0 kV 不低于 72.5 kV，首开极系数 kpp 默认取 1.3。",
        #     },
        # }
        altitude = result["values"]["altitude_m"]
        altitude_desc = result["descriptions"]["altitude_m"]
    """

    def __init__(self, stand_type: str = "") -> None:
        self._stand_type = stand_type
        # ---- simple params (fixed default) ----
        self._param_defs: list[dict[str, Any]] = [
            {
                "key": "altitude_m",
                "pattern": re.compile(
                    r"最大\(适用\)的海拔\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*m"
                ),
                "default": 1000,
                "cast": int,
                "description": "最大(适用)的海拔",
                "default_desc": "当前未检测到用户提供的{description}，默认按{default_display}回填。",
                "user_desc": "用户已明确提供{description}为 {value_display}，直接采用。",
            },
            {
                "key": "rated_voltage_kv",
                "pattern": re.compile(
                    r"额定电压\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*kV\b"
                ),
                "default": None,
                "cast": float,
                "description": "额定电压",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display}，直接采用。",
            },
            {
                "key": "first_pole_kpp",
                "pattern": re.compile(
                    r"首开极系数(?:\s*kpp)?\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)"
                ),
                "default": None,
                "cast": float,
                "description": "首开极系数 kpp",
                # default_desc 不使用，由 _computed_params 中的 _describe_first_pole_kpp 生成
                "user_desc": "用户已明确提供{description}为 {value_display}，直接采用。",
            },
            {
                "key": "rated_current_a",
                "pattern": re.compile(
                    r"额定电流\s*(?:[:：=]\s*)?([0-9]+)\s*A\b"
                ),
                "default": None,
                "cast": int,
                "description": "额定电流",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} A，直接采用。",
            },
            {
                "key": "model_prefix",
                "pattern": re.compile(
                    r"(?:型号名称|型号)\s*[：:=]\s*([A-Za-z0-9]+)"
                ),
                "default": None,
                "cast": str,
                "description": "型号前缀",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display}，直接采用。",
            },
            {
                "key": "rated_frequency_hz",
                "pattern": re.compile(
                    r"额定频率\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*Hz\b"
                ),
                "default": 50.0,
                "cast": float,
                "description": "额定频率",
                "default_desc": "当前未检测到用户提供的{description}，默认按{default_display} Hz 回填。",
                "user_desc": "用户已明确提供{description}为 {value_display} Hz，直接采用。",
            },
            {
                "key": "rated_closing_ka",
                "pattern": re.compile(
                    r"(?:额定短路关合电流|短路关合电流|关合电流)\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:kA)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定短路关合电流",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} kA，直接采用。",
            },
            {
                "key": "short_break_ka",
                "pattern": re.compile(
                    r"(?:额定短路开断电流|短路开断电流)\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:kA)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定短路开断电流",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} kA，直接采用。",
            },
            {
                "key": "rated_short_time_withstand_ka",
                "pattern": re.compile(
                    r"(?:额定短时耐受电流|短时耐受电流)\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:kA)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定短时耐受电流",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} kA，直接采用。",
            },
            {
                "key": "rated_peak_withstand_ka",
                "pattern": re.compile(
                    r"(?:额定峰值耐受电流|峰值耐受电流|额定峰值电流|峰值电流)\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:kA)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定峰值耐受电流",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} kA，直接采用。",
            },
            {
                "key": "rated_short_circuit_duration_s",
                "pattern": re.compile(
                    r"(?:额定短路持续时间|短路持续时间)\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)"
                ),
                "default": None,
                "cast": float,
                "description": "额定短路持续时间",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} s，直接采用。",
            },
            {
                "key": "rated_out_of_step_break_ka",
                "pattern": re.compile(
                    r"(?:额定失步开断电流|失步开断电流)\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:kA)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定失步开断电流",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} kA，直接采用。",
            },
            {
                "key": "pf_withstand_kv",
                "pattern": re.compile(
                    r"(?:额定短时工频耐受电压|额定工频耐受电压)\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?(?:\s*\+\s*[0-9]+(?:\.[0-9]+)?)*)\s*(?:kV)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定短时工频耐受电压",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} kV，直接采用。",
            },
            {
                "key": "pf_fracture_withstand_kv",
                "pattern": re.compile(
                    r"(?:额定短时工频耐受电压\s*[（(]\s*断口\s*[）)]|额定工频耐受电压\s*[（(]\s*断口\s*[）)])\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?(?:\s*\+\s*[0-9]+(?:\.[0-9]+)?)*)\s*(?:kV)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定短时工频耐受电压(断口)",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} kV，直接采用。",
            },
            {
                "key": "li_withstand_kv",
                "pattern": re.compile(
                    r"额定雷电冲击耐受电压\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?(?:\s*\+\s*[0-9]+(?:\.[0-9]+)?)*)\s*(?:kV)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定雷电冲击耐受电压",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} kV，直接采用。",
            },
            {
                "key": "li_fracture_withstand_kv",
                "pattern": re.compile(
                    r"额定雷电冲击耐受电压\s*[（(]\s*断口\s*[）)]\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?(?:\s*\+\s*[0-9]+(?:\.[0-9]+)?)*)\s*(?:kV)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定雷电冲击耐受电压(断口)",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} kV，直接采用。",
            },
            {
                "key": "si_withstand_kv",
                "pattern": re.compile(
                    r"额定操作冲击耐受电压\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?(?:\s*\+\s*[0-9]+(?:\.[0-9]+)?)*)\s*(?:kV)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定操作冲击耐受电压",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} kV，直接采用。",
            },
            {
                "key": "si_fracture_withstand_kv",
                "pattern": re.compile(
                    r"额定操作冲击耐受电压\s*[（(]\s*断口\s*[）)]\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?(?:\s*\+\s*[0-9]+(?:\.[0-9]+)?)*)\s*(?:kV)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定操作冲击耐受电压(断口)",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} kV，直接采用。",
            },
            {
                "key": "bc_current_a",
                "pattern": re.compile(
                    r"(?:额定电容器组电流|电容器组开断电流|电容器组开合电流|电容器电流|额定背对背电容器组开断电流|背对背电容器组开断电流|额定单个电容器组开断电流|单个电容器组开断电流)\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:A)?"
                ),
                "default": None,
                "cast": float,
                "description": "额定电容器组电流",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display} A，直接采用。",
            },
            {
                "key": "pf_joint_voltage_parts",
                "pattern": re.compile(
                    r"(?:额定短时工频耐受电压|额定工频耐受电压)\s*[（(]\s*断口\s*[）)]\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*[（(]\s*\+\s*([0-9]+(?:\.[0-9]+)?)\s*[）)]"
                ),
                "default": None,
                "cast": self._to_voltage_pair,
                "description": "额定短时工频耐受电压(断口)联合电压",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display}，直接采用。",
            },
            {
                "key": "li_joint_voltage_parts",
                "pattern": re.compile(
                    r"额定雷电冲击耐受电压\s*[（(]\s*断口\s*[）)]\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*[（(]\s*\+\s*([0-9]+(?:\.[0-9]+)?)\s*[）)]"
                ),
                "default": None,
                "cast": self._to_voltage_pair,
                "description": "额定雷电冲击耐受电压(断口)联合电压",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display}，直接采用。",
            },
            {
                "key": "si_joint_voltage_parts",
                "pattern": re.compile(
                    r"额定操作冲击耐受电压\s*[（(]\s*断口\s*[）)]\s*(?:[:：=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*[（(]\s*\+\s*([0-9]+(?:\.[0-9]+)?)\s*[）)]"
                ),
                "default": None,
                "cast": self._to_voltage_pair,
                "description": "额定操作冲击耐受电压(断口)联合电压",
                "default_desc": "当前未检测到用户提供的{description}，需后续补充。",
                "user_desc": "用户已明确提供{description}为 {value_display}，直接采用。",
            },
        ]

        # ---- computed params (dynamic default based on other params) ----
        # 每个条目：(computer_func, describe_func)
        self._computed_params: dict[str, tuple[Callable, Callable]] = {
            "first_pole_kpp": (self._default_first_pole_kpp, self._describe_first_pole_kpp),
        }


    @staticmethod
    def _sum_voltage_parts(raw: str) -> str:
        """电压值原样返回，不做求和。"""
        return raw.strip()

    @staticmethod
    def _to_voltage_pair(*groups: str) -> tuple[float, float | None]:
        """将 '(230)(+92)' 格式转为 (230.0, 92.0) 元组。"""
        if len(groups) >= 2 and groups[1]:
            return (float(groups[0]), float(groups[1]))
        if groups:
            return (float(groups[0]), None)
        return (None, None)

    # ============================================================
    # 首开极系数：默认值计算 & 描述生成
    # ============================================================

    @staticmethod
    def _default_first_pole_kpp(params: dict[str, Any]) -> float | None:
        """
        首开极系数默认值规则：
          - 用户已提供 → 用用户值（已在第一遍提取）
          - ≤40.5kV → 1.5
          - ≥72.5kV → 1.3
          - 其他 → None
        """
        if params.get("first_pole_kpp") is not None:
            return params["first_pole_kpp"]
        ur = params.get("rated_voltage_kv")
        if ur is not None and ur <= 40.5:
            return 1.5
        if ur is not None and ur >= 72.5:
            return 1.3
        return None

    @staticmethod
    def _describe_first_pole_kpp(
        params: dict[str, Any], raw_value: Any, used_default: bool
    ) -> str:
        """生成首开极系数的描述文本。"""
        if not used_default:
            return f"用户已明确提供首开极系数 kpp 为 {raw_value}，问答阶段直接采用该值。"
        ur = params.get("rated_voltage_kv")
        if ur is not None and ur <= 40.5:
            return f"额定电压 {ur} kV 不高于 40.5 kV，首开极系数 kpp 默认取 1.5。"
        if ur is not None and ur >= 72.5:
            return f"额定电压 {ur} kV 不低于 72.5 kV，首开极系数 kpp 默认取 1.3。"
        return "未明确提供首开极系数 kpp，且当前额定电压区间无默认值。"

    # ============================================================
    # 主提取方法
    # ============================================================

    def extract(self, query_text: str) -> dict[str, Any]:
        """Extract all defined parameters from query text.

        Two-pass extraction:
          1. First pass: extract simple params via regex.
          2. Second pass: resolve computed params with dynamic defaults.

        Args:
            query_text: Raw user query string.

        Returns:
            dict with two keys:
              - "values": dict mapping parameter keys to extracted values (or defaults)
              - "descriptions": dict mapping parameter keys to human-readable descriptions
        """
        values: dict[str, Any] = {}
        descriptions: dict[str, str] = {}
        text = str(query_text or "").strip()
        # 统一预处理：中文括号 → 英文括号
        text = text.replace("（", "(").replace("）", ")")

        # ---- first pass: simple regex extraction ----
        for param_def in self._param_defs:
            match = param_def["pattern"].search(text)
            if match:
                groups = match.groups()
                raw = groups[0] if groups else ""
                try:
                    # 传所有分组给 cast，支持多分组提取（如联合电压元组）
                    values[param_def["key"]] = param_def["cast"](*groups)
                except (ValueError, TypeError):
                    values[param_def["key"]] = param_def["default"]
                # 用户明确提供 → 用 user_desc 模板
                if values[param_def["key"]] is not None:
                    desc = param_def.get("user_desc", "")
                    if desc:
                        descriptions[param_def["key"]] = desc.format(
                            description=param_def["description"],
                            value_display=raw,
                        )
                    else:
                        descriptions[param_def["key"]] = ""
                else:
                    # 提取到但 cast 失败导致为 None，走 default_desc
                    default_display = self._format_default_display(param_def)
                    desc = param_def.get("default_desc", "")
                    descriptions[param_def["key"]] = desc.format(
                        description=param_def["description"],
                        default_display=default_display,
                    ) if desc else ""
            else:
                values[param_def["key"]] = param_def["default"]
                # 未匹配 → 用 default_desc 模板
                default_display = self._format_default_display(param_def)
                desc = param_def.get("default_desc", "")
                descriptions[param_def["key"]] = desc.format(
                    description=param_def["description"],
                    default_display=default_display,
                ) if desc else ""

        # ---- second pass: resolve computed params ----
        for key, (computer, describer) in self._computed_params.items():
            used_default = key not in values or values[key] is None
            if used_default:
                computed = computer(values)
                if computed is not None:
                    values[key] = computed
            # 生成描述（无论是否使用默认值，都由 describer 统一处理）
            descriptions[key] = describer(values, values.get(key), used_default)

        # 将 stand_type 也加入返回结果
        values["stand_type"] = self._stand_type
        descriptions["stand_type"] = f"标准类型：{self._stand_type}"

        # ---- 中文参数名映射（用于日志输出） ----
        _PARAM_CN = {
            "altitude_m": "最大(适用)的海拔",
            "rated_voltage_kv": "额定电压",
            "rated_current_a": "额定电流",
            "rated_frequency_hz": "额定频率",
            "rated_closing_ka": "额定短路关合电流",
            "short_break_ka": "额定短路开断电流",
            "rated_short_time_withstand_ka": "额定短时耐受电流",
            "rated_peak_withstand_ka": "额定峰值耐受电流",
            "rated_short_circuit_duration_s": "额定短路持续时间",
            "rated_out_of_step_break_ka": "额定失步开断电流",
            "first_pole_kpp": "首开极系数",
            "pf_withstand_kv": "额定短时工频耐受电压",
            "pf_fracture_withstand_kv": "额定短时工频耐受电压(断口)",
            "li_withstand_kv": "额定雷电冲击耐受电压",
            "li_fracture_withstand_kv": "额定雷电冲击耐受电压(断口)",
            "pf_joint_voltage_parts": "工频断口联合电压",
            "li_joint_voltage_parts": "雷电冲击断口联合电压",
            "si_withstand_kv": "额定操作冲击耐受电压",
            "si_fracture_withstand_kv": "额定操作冲击耐受电压(断口)",
            "si_joint_voltage_parts": "操作冲击断口联合电压",
            "bc_current_a": "额定电容器组电流",
            "model_prefix": "型号前缀",
        }
        _log_lines = [
            f"  {_PARAM_CN.get(k, k)} = {v!r}"
            for k, v in sorted(values.items())
            if v is not None and k != "stand_type"
        ]
        if _log_lines:
            import logging
            logging.getLogger(self.__class__.__name__).info(
                "【参数提取】QueryParamExtractor 提取结果:\n%s",
                "\n".join(_log_lines),
            )

        return {"values": values, "descriptions": descriptions}

    @staticmethod
    def _format_default_display(param_def: dict[str, Any]) -> str:
        """将默认值格式化为可读字符串，用于描述模板。"""
        default = param_def.get("default")
        if default is None:
            return "暂无默认值"
        return str(default)

    # ============================================================
    # 动态注册新参数
    # ============================================================

    def add_param(
        self,
        key: str,
        pattern: str | re.Pattern,
        default: Any = None,
        cast: type | None = None,
        description: str = "",
        default_desc: str = "",
        user_desc: str = "",
    ) -> None:
        """Register a new parameter for extraction (for future extension).

        Args:
            key: Internal parameter name (e.g. "altitude_m").
            pattern: Regex pattern with one capture group for the value.
            default: Fallback value when not found.
            cast: Type conversion function (e.g. int, float).
            description: Human-readable parameter name.
            default_desc: Template for default-value description.
                         Available placeholders: {description}, {default_display}.
            user_desc: Template for user-provided description.
                      Available placeholders: {description}, {value_display}.
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self._param_defs.append({
            "key": key,
            "pattern": pattern,
            "default": default,
            "cast": cast or (lambda x: x),
            "description": description,
            "default_desc": default_desc,
            "user_desc": user_desc,
        })

