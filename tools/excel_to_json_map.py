'''
在线治理表 -> annotation_memory.*.json
'''
import pandas as pd
import json
import sys
from collections import defaultdict
import copy


def str_format(value:None):
    return str(value).strip()


def format_dict(data):
    result = []

    for level1, level2_dict in data.items():
        # 一级标题
        result.append(f"- {level1}")

        for level2, desc in level2_dict.items():
            # 如果有描述就加，没有就只写名字
            if desc:
                result.append(f"  - {level2}  {desc}")
            else:
                result.append(f"  - {level2}")

    return "\n".join(result)

def parse_excel_to_json(excel_path, sheet_name=0, required_cols=None, allowed_categories=None):
    df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=str)
    df = df.fillna('')

    for col in required_cols:
        if col not in df.columns:
            df[col] = ''  # 缺失列用空字符串填充

    groups = defaultdict(list)
    for _, row in df.iterrows():
        subcat = str_format(row['型式试验的小类'])
        # 过滤：只有小类在允许列表中才处理
        if subcat not in allowed_categories:
            continue
        test = str_format(row['试验'])
        actual = str_format(row['实际试验名称'])
        proj = str_format(row['项目试验名称'])
        # note = row['note']
        test_name = actual if actual else proj
        key = (test, subcat, test_name)
        groups[key].append(row)

    tests_by_path = {}
    tests_by_name = defaultdict(list)
    add_test_items = []

    for (test, subcat, test_name), rows in groups.items():
        # ===== 合并 note =====
        notes = []
        for row in rows:
            note_val = str_format(row['note'])
            if note_val:
                notes.append(note_val)
        # 去重并合并，用分号分隔
        unique_notes = list(dict.fromkeys(notes))  # 保持顺序去重
        merged_note = "; ".join(unique_notes) if unique_notes else ""

        # 构建 path_parts: 使用统一的第一个部分
        if test == "型式试验":
            path_parts = [test, subcat, test_name]
            path_parts_2 = [subcat, subcat, test_name]
        else:
            path_parts = ["型式试验", subcat, test_name]
            path_parts_2 = [subcat, subcat, test_name]

        path_key = " > ".join(path_parts)
        path_key_2 = " > ".join(path_parts_2)


        # 收集参数、remove_parameters、remove_rules、触发条件、别名等
        parameters = []
        remove_params_set = set()
        remove_rules_set = set()
        trigger_conditions = set()
        aliases_set = set()
        acceptance_criteria = ""
        confidence = 1.0
        has_required_params = False  # 新增：标记是否有需要输出的参数


        for row in rows:
            # 需要输出的参数 - 始终为每个参数生成完整的字段
            need = str(row['是否需要输出在试验参数中']).strip()
            if need in ('是', 'true', '1', 'True'):
                has_required_params = True  # 标记有需要输出的参数
                value_text = row['取值范围']
                value_source = row['来源']
                param = {
                    "param_name": row['特性名称'],
                    "value_text": value_text,
                    "value_expr": value_text if value_source in {"user_input", "formula", "default"} else "",
                    "value_source": value_source,
                    "value_type": value_source,
                    "constraints": value_text,
                    "calc_rule": value_text if value_source == "formula" else "" ,
                    "derive_from_rated": value_text if value_source == "user_input" else ""
                }
                parameters.append(param)

            # remove_parameters / remove_rules
            if row['remove_parameters']:
                for item in str(row['remove_parameters']).split(','):
                    if item.strip():
                        remove_params_set.add(item.strip())
            if row['remove_rules']:
                for item in str(row['remove_rules']).split(','):
                    if item.strip():
                        remove_rules_set.add(item.strip())

            # 别名
            if row['别名']:
                for alias in str(row['别名']).split(','):
                    if alias.strip():
                        aliases_set.add(alias.strip())


            # 置信度（取第一个非空）
            if row['置信度'] and confidence == 1.0:
                try:
                    confidence = float(row['置信度'])
                except:
                    pass

        remove_parameters = list(remove_params_set)
        remove_rules = list(remove_rules_set)
        
        # 新增：如果没有需要输出的参数，跳过该测试项
        if not has_required_params:
            continue


        # ---------- 1. tests_by_path ----------
        test_obj_path = {
            "test_name": test_name,
            "report_type": subcat,
            "category": subcat,
            "path_parts": path_parts,
            "path_key": path_key,
            "skip": False,
            "note": merged_note,
            "parameters": parameters,
            "remove_parameters": remove_parameters,
            "remove_rules": remove_rules
        }
        test_obj_path_2 = {
            "test_name": test_name,
            "report_type": subcat,
            "category": subcat,
            "path_parts": path_parts,
            "path_key": path_key_2,
            "skip": False,
            "note": merged_note,
            "parameters": parameters,
            "remove_parameters": remove_parameters,
            "remove_rules": remove_rules
        }
        tests_by_path[path_key] = test_obj_path
        tests_by_path[path_key_2] = test_obj_path_2

        # ---------- 2. tests_by_name ----------
        test_obj_name = copy.deepcopy(test_obj_path)
        test_obj_name_2 = copy.deepcopy(test_obj_path_2)
        tests_by_name[test_name].append(test_obj_name)
        tests_by_name[test_name].append(test_obj_name_2)

        # ---------- 3. add_test_items ----------
        required_reports = []
        if trigger_conditions:
            for cond in trigger_conditions:
                required_reports.append({
                    "report_type": subcat,
                    "is_required": True,
                    "condition": cond
                })
        else:
            # 如果没有触发条件，添加一个默认的 required_report
            required_reports.append({
                "report_type": subcat,
                "is_required": True,
                "condition": ""
            })

        add_item_obj = {
            "test_item": test_name,
            "category": subcat,
            "report_type": subcat,
            "aliases": list(aliases_set),
            "acceptance_criteria": acceptance_criteria,
            "note": merged_note,
            "confidence": confidence,
            "required_reports": required_reports,
            "parameters": parameters,
            "rules": remove_rules
        }
        add_test_items.append(add_item_obj)

    # ===== 打印每个试验包含的特性名称 =====
    # 对应的提示词
    chu_fa_tiao_jian = {"短路性能型式试验":{},"绝缘性能型式试验":{},"开合性能型式试验":{},"温升性能型式试验":{}}

    test_params = {}
    for (test, subcat, test_name), rows in groups.items():
        # 检查该测试项是否有需要输出的参数
        has_required_params_for_test = False
        param_names_for_test = []
        
        for row in rows:
            need = str(row['是否需要输出在试验参数中']).strip()
            if need in ('是', 'true', '1', 'True'):
                has_required_params_for_test = True
                param_name = row['特性名称']
                if param_name and param_name not in param_names_for_test:
                    param_names_for_test.append(param_name)
                
                chu_fa_tiao_jian[row["型式试验的小类"]][row["实际试验名称"]]=row["触发条件"]
        
        # 只有有需要输出的参数时，才添加到test_params
        if has_required_params_for_test and param_names_for_test:
            test_params[test_name] = param_names_for_test

    test_item_param_requirements = "test_item_param_requirements = "
    test_items = "test_items = "


    for test_name, param_list in test_params.items():
        if param_list:
            param_str = '|'.join(param_list)
            test_item_param_requirements += f"{test_name}:{param_str}; "
            test_items += f"{test_name}, "


    print(test_items)
    print(test_item_param_requirements)

    print("\n")
    print("试验触发条件提示词")
    print("\n")
    print(format_dict(chu_fa_tiao_jian))

    return {
        "tests_by_path": tests_by_path,
        "tests_by_name": tests_by_name,
        "add_test_items": add_test_items
    }


def pare_english(excel_path, sheet_name="参数中英文最终版"):
    df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=str)
    df = df.fillna('')
    param_map = []
    for _, row in df.iterrows():
        cn_str = str_format(row['特性名称'])
        en_str = str_format(row['特性英文名称'])
        param_map.append(f"{cn_str}:{str(en_str)}")
    text = ";".join(param_map)
    print("param_map = {}".format(text))


if __name__ == "__main__":
    # 需要检查 , '型式试验的小类', '实际试验名称', '项目试验名称', 试验 这些列不能为空
    # 定义期望的列名（可根据实际 Excel 调整）
    required_cols = [
        '试验', '型式试验的小类', '实际试验名称', '项目试验名称',
        '特性名称', '是否需要输出在试验参数中', '取值范围', '来源',
        'note', 'remove_rules', 'remove_parameters', '其他备注',
        '标准桂发文件中描述', '触发条件', '别名', '置信度'
    ]
    # , '判定标准'
    # 允许的型式试验小类列表
    allowed_categories = [
        "温升性能型式试验",
        "开合性能型式试验",
        "绝缘性能型式试验",
        "短路性能型式试验"
        # ,"EMC性能型式试验"
    ]
    import time

    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    input_file = "../../副本项目-特征逻辑映射表-完整版_20260324_1508.xlsx"

    """
    读取中英文
    """
    pare_english(excel_path=input_file,sheet_name="参数中英文最终版")

    # # 处理 GBT
    # output_file_gb = f"{time_str}_annotation_memory.gb_gy.json"
    # sheet_name_gb = "GBT"
    # result_gb = parse_excel_to_json(excel_path=input_file, sheet_name=sheet_name_gb, required_cols=required_cols, allowed_categories=allowed_categories)
    #
    # with open(output_file_gb, 'w', encoding='utf-8') as f:
    #     json.dump(result_gb, f, ensure_ascii=False, indent=2)
    #
    # print(f"转换完成，结果已保存至 {output_file_gb}")


    # # 处理 DLT
    # output_file_dlt = f"{time_str}_annotation_memory.dlt_gy.json"
    # sheet_name_dlt = "DLT"
    # result_dlt = parse_excel_to_json(input_file, sheet_name=sheet_name_dlt, required_cols=required_cols, allowed_categories=allowed_categories)
    #
    # with open(output_file_dlt, 'w', encoding='utf-8') as f:
    #     json.dump(result_dlt, f, ensure_ascii=False, indent=2)
    #
    # print(f"转换完成，结果已保存至 {output_file_dlt}")

    # 处理 IEC
    output_file_iec = f"{time_str}_annotation_memory.iec_gy.json"
    sheet_name_iec = "IEC"
    result_iec = parse_excel_to_json(input_file, sheet_name=sheet_name_iec, required_cols=required_cols, allowed_categories=allowed_categories)

    with open(output_file_iec, 'w', encoding='utf-8') as f:
        json.dump(result_iec, f, ensure_ascii=False, indent=2)

    print(f"转换完成，结果已保存至 {output_file_iec}")