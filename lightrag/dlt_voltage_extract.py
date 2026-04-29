"""
已添加三个方法：

1. get_insulation_by_voltage — 取小于输入的最大 key

key 从大到小排序，找到第一个 voltage > key 的
42kV → 40.5 的数据（to_ground=95kV）
126kV → 72.5 的数据（to_ground=160kV）

现在使用的
2. get_insulation_ge — 取大于等于输入的最小 key

key 从小到大排序，找到第一个 key >= voltage 的
42kV → 72.5 的数据（to_ground=160kV）
126kV → 126 的数据（to_ground=230kV）
3. get_insulation_mid — 中间值就近

找到相邻两个 key（lower < voltage < upper）
计算 mid = (lower + upper) / 2
voltage > mid 取 upper，否则取 lower
42kV（40.5~72.5, mid=56.5）→ 42 < 56.5 → 取 40.5（to_ground=95kV）
60kV（40.5~72.5, mid=56.5）→ 60 > 56.5 → 取 72.5（to_ground=160kV）
"""
extract_datas = {
  "3.6": {
    "额定工频短时耐受电压 (通用值)": "25/18",
    "额定工频短时耐受电压 (隔离断口)": "27/20",
    "额定雷电冲击耐受电压 (通用值)": "40/20",
    "额定雷电冲击耐受电压 (隔离断口)": "46/23"
  },
  "7.2": {
    "额定工频短时耐受电压 (通用值)": "30/23",
    "额定工频短时耐受电压 (隔离断口)": "34/27",
    "额定雷电冲击耐受电压 (通用值)": "60/40",
    "额定雷电冲击耐受电压 (隔离断口)": "70/46"
  },
  "12": {
    "额定工频短时耐受电压 (通用值)": "42/30",
    "额定工频短时耐受电压 (隔离断口)": "48/36",
    "额定雷电冲击耐受电压 (通用值)": "75/60",
    "额定雷电冲击耐受电压 (隔离断口)": "85/70"
  },
  "24": {
    "额定工频短时耐受电压 (通用值)": "65/50",
    "额定工频短时耐受电压 (隔离断口)": "79/64",
    "额定雷电冲击耐受电压 (通用值)": "125/95",
    "额定雷电冲击耐受电压 (隔离断口)": "145/115"
  },
  "40.5": {
    "额定工频短时耐受电压 (通用值)": "95/80",
    "额定工频短时耐受电压 (隔离断口)": "118/103",
    "额定雷电冲击耐受电压 (通用值)": "185/170",
    "额定雷电冲击耐受电压 (隔离断口)": "215/200"
  },
  "72.5": {
    "额定工频短时耐受电压 (通用值)": "160",
    "额定工频短时耐受电压 (隔离断口)": "200",
    "额定雷电冲击耐受电压 (通用值)": "350",
    "额定雷电冲击耐受电压 (隔离断口)": "410"
  },
  "126": {
    "额定工频短时耐受电压 (通用值)": "230",
    "额定工频短时耐受电压 (隔离断口)": "230 (+70)",
    "额定雷电冲击耐受电压 (通用值)": "550",
    "额定雷电冲击耐受电压 (隔离断口)": "550 (+100)"
  },
  "252": {
    "额定工频短时耐受电压 (通用值)": "460",
    "额定工频短时耐受电压 (隔离断口)": "460 (+145)",
    "额定雷电冲击耐受电压 (通用值)": "1050",
    "额定雷电冲击耐受电压 (隔离断口)": "1050 (+200)"
  },
  "363": {
    "额定短时工频耐受电压 (相对地及相间)": "510",
    "额定短时工频耐受电压 (开关闸口及隔离闸口)": "510 (+210)",
    "额定操作冲击耐受电压 (相对地)": "950",
    "额定操作冲击耐受电压 (相间)": "1425",
    "额定操作冲击耐受电压 (开关闸口及隔离闸口)": "850 (+295)",
    "额定雷电冲击耐受电压 (相对地及相间)": "1175",
    "额定雷电冲击耐受电压 (开关闸口及隔离闸口)": "1175 (+295)"
  },
  "550": {
    "额定短时工频耐受电压 (相对地及相间)": "740",
    "额定短时工频耐受电压 (开关闸口及隔离闸口)": "740 (+315)",
    "额定操作冲击耐受电压 (相对地)": "1300",
    "额定操作冲击耐受电压 (相间)": "1950",
    "额定操作冲击耐受电压 (开关闸口及隔离闸口)": "1175 (+450)",
    "额定雷电冲击耐受电压 (相对地及相间)": "1675",
    "额定雷电冲击耐受电压 (开关闸口及隔离闸口)": "1675 (+450)"
  },
  "800": {
    "额定短时工频耐受电压 (相对地及相间)": "960",
    "额定短时工频耐受电压 (开关闸口及隔离闸口)": "960 (+460)",
    "额定操作冲击耐受电压 (相对地)": "1550",
    "额定操作冲击耐受电压 (相间)": "2480",
    "额定操作冲击耐受电压 (开关闸口及隔离闸口)": "1425 (+650)",
    "额定雷电冲击耐受电压 (相对地及相间)": "2100",
    "额定雷电冲击耐受电压 (开关闸口及隔离闸口)": "2100 (+650)"
  },
  "1100": {
    "额定短时工频耐受电压 (相对地及相间)": "1100",
    "额定短时工频耐受电压 (开关闸口及隔离闸口)": "1100 (+635)",
    "额定操作冲击耐受电压 (相对地)": "1800",
    "额定操作冲击耐受电压 (相间)": "2700",
    "额定操作冲击耐受电压 (开关闸口及隔离闸口)": "1675 (+900)",
    "额定雷电冲击耐受电压 (相对地及相间)": "2400",
    "额定雷电冲击耐受电压 (开关闸口及隔离闸口)": "2400 (+900)"
  }
}

import json
import re
from typing import Any


def _parse_value(value: str) -> dict[str, float]:
    """解析绝缘值字符串为结构化数据。

    - "95/80"       → {"to_ground": 95, "fracture": 80}
    - "230 (+70)"   → {"to_ground": 230, "fracture_aux": 70}
    - "160"         → {"to_ground": 160}
    """
    value = str(value or "").strip()
    if not value:
        return {}

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)$", value)
    if m:
        return {"to_ground": float(m.group(1)), "fracture": float(m.group(2))}

    m = re.match(
        r"^([0-9]+(?:\.[0-9]+)?)\s*\(\s*\+\s*([0-9]+(?:\.[0-9]+)?)\s*\)$", value
    )
    if m:
        return {"to_ground": float(m.group(1)), "fracture_aux": float(m.group(2))}

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)$", value)
    if m:
        return {"to_ground": float(m.group(1))}

    return {"raw": value}


def _resolve_key(target_voltage: float, data: dict[str, Any]) -> str:
    """将浮点数 key 转为字典中实际的字符串 key，处理 '126' vs '126.0' 问题。"""
    target_key = str(target_voltage)
    if target_key not in data:
        alt_key = str(int(target_voltage)) if target_voltage == int(target_voltage) else target_key
        if alt_key in data:
            target_key = alt_key
    return target_key


def _format_result(target_key: str, data: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        str(key): _parse_value(str(val))
        for key, val in data[target_key].items()
    }


def get_insulation_by_voltage(
    voltage: float,
    data: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """取小于输入电压的最大 key 对应的绝缘数据。

    规则：key 从大到小排序，找到第一个输入电压 > key 的，返回该 key 的数据。
    若输入电压小于等于最小 key，返回最小 key 的数据。
    """
    if not data:
        data = extract_datas

    voltages = sorted((float(k) for k in data.keys()), reverse=True)

    target = None
    for v in voltages:
        if voltage > v:
            target = v
            break

    if target is None:
        target = voltages[-1]

    return _format_result(_resolve_key(target, data), data)


def get_insulation_ge(
    voltage: float,
    data: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """取大于输入电压的最小 key 对应的绝缘数据。

    规则：key 从小到大排序，找到第一个 key >= 输入电压的，返回该 key 的数据。
    若输入电压大于最大 key，返回最大 key 的数据。
    """
    if not data:
        data = extract_datas

    voltages = sorted(float(k) for k in data.keys())

    target = None
    for v in voltages:
        if v >= voltage:
            target = v
            break

    if target is None:
        target = voltages[-1]

    return _format_result(_resolve_key(target, data), data)


def _to_test_name(key: str) -> str:
    """将绝缘数据 key 转为试验名称。

    转换规则：
    - 额定工频短时耐受电压 (隔离断口) → 工频耐受电压试验(断口)
    - 额定雷电冲击耐受电压 (隔离断口) → 雷电冲击耐受电压试验(断口)
    - 额定短时工频耐受电压 (开关闸口及隔离闸口) → 工频耐受电压试验(开关闸口及隔离闸口)
    - 额定操作冲击耐受电压 (开关闸口及隔离闸口) → 操作冲击耐受电压试验(开关闸口及隔离闸口)
    - 额定雷电冲击耐受电压 (开关闸口及隔离闸口) → 雷电冲击耐受电压试验(开关闸口及隔离闸口)
    """
    name = key.strip()
    # 去掉括号及括号内内容（如 (隔离断口)、(开关闸口及隔离闸口)）
    name = re.sub(r"\s*\([^)]*\)\s*", "", name)
    return name.strip()


def get_fracture_voltage_ge(
    voltage: float,
    data: dict[str, Any] | None = None,
) -> dict[str, float]:
    """获取大于等于输入电压的等级中，断口相关的对地电压值。

    内部调用 get_insulation_ge，然后过滤出 key 包含"断口"/"闸口"/"隔离"的项，
    返回 {试验名称: to_ground 值} 的字典。

    Args:
        voltage: 输入的额定电压数值
        data: 绝缘数据字典，为 None 时使用默认的 extract_datas

    Returns:
        断口相关试验的 {试验名称: to_ground 值} 字典。
    """
    result = get_insulation_ge(voltage, data)
    fracture_keys = ["断口", "闸口", "隔离"]
    out = {}
    for key, val in result.items():
        if any(fk in key for fk in fracture_keys):
            tg = val.get("to_ground")
            if tg is not None:
                out[_to_test_name(key)] = int(tg)
    return out


def get_insulation_mid(
    voltage: float,
    data: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """取中间值就近的 key 对应的绝缘数据。

    规则：找到输入电压所在的两个相邻 key（lower < voltage < upper），
    计算中间值 mid = (lower + upper) / 2，
    若 voltage > mid 取 upper，否则取 lower。
    若 voltage 小于最小 key，取最小 key；大于最大 key，取最大 key。
    """
    if not data:
        data = extract_datas

    voltages = sorted(float(k) for k in data.keys())

    # 小于等于最小 key
    if voltage <= voltages[0]:
        return _format_result(_resolve_key(voltages[0], data), data)

    # 大于等于最大 key
    if voltage >= voltages[-1]:
        return _format_result(_resolve_key(voltages[-1], data), data)

    # 找到相邻的两个 key
    lower = upper = None
    for i in range(len(voltages) - 1):
        if voltages[i] <= voltage < voltages[i + 1]:
            lower, upper = voltages[i], voltages[i + 1]
            break

    if lower is None or upper is None:
        return {}

    mid = (lower + upper) / 2
    target = upper if voltage > mid else lower
    return _format_result(_resolve_key(target, data), data)


# ===== 示例 =====

if __name__ == "__main__":
    insulation_data = extract_datas

    def _show(v: float, r: dict) -> str:
        first = next(iter(r.values()))
        tg = first.get("to_ground", "?")
        return f"{v}kV -> to_ground={tg}kV"

    # print("=== get_insulation_by_voltage（取小于输入的最大 key）===")
    # for v in [3, 40.5, 42, 126, 1000]:
    #     print(f"  {_show(v, get_insulation_by_voltage(v, insulation_data))}")

    print("\n=== get_insulation_ge（取大于等于输入的最小 key）===")
    for v in [3, 40.5, 42, 126, 1000]:
        print(f"  { get_fracture_voltage_ge(v, insulation_data)}")

    # print("\n=== get_insulation_mid（中间值就近）===")
    # for v in [3, 30, 42, 60, 100, 126, 1000]:
    #     print(f"  {_show(v, get_insulation_mid(v, insulation_data))}")