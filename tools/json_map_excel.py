import json
import pandas as pd
from typing import List, Dict, Any
import os

import time

time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

class TreeExtractor:
    def __init__(self, json_file_path: str):
        """
        初始化提取器

        Args:
            json_file_path: JSON文件路径
        """
        self.json_file_path = json_file_path
        self.data = None
        self.results = []
        # 需要提取的特性节点名称
        self.extract_nodes = ["特征值"]
        # 需要跳过的节点名称（不提取其子节点）
        self.skip_nodes = ["条件/规则", "原文切块"]

    def load_json(self):
        """加载JSON文件"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 处理可能的嵌套结构
        if 'tree' in self.data:
            self.data = self.data['tree']

        return self.data

    def extract_names(self, node: Dict, level: int = 0, path: List[str] = None):
        """
        递归提取树结构中的所有name值

        Args:
            node: 当前节点
            level: 当前层级
            path: 路径列表（记录所有父节点名称）
        """
        if path is None:
            path = []

        current_name = node.get('name', '')
        children = node.get('children', [])

        # 构建当前路径
        current_path = path + [current_name]

        # 检查当前节点是否是特征值节点
        is_feature_node = current_name in self.extract_nodes

        # 检查当前节点是否需要跳过
        is_skip_node = current_name in self.skip_nodes

        if is_feature_node:
            # 找到特征值节点，提取其所有子节点
            # 此时路径应该是: [型式试验, 小类, 试验名称, 特征值]
            # 我们需要前三个作为前三列
            for child in children:
                child_name = child.get('name', '')
                if child_name:
                    # 提取前三层路径
                    test_type = current_path[0] if len(current_path) > 0 else '型式试验'
                    sub_test = current_path[1] if len(current_path) > 1 else ''
                    actual_test = current_path[2] if len(current_path) > 2 else ''

                    self.results.append({
                        '试验': test_type,
                        '型式试验的小类': sub_test,
                        '实际试验名称': actual_test,
                        '特性名称': child_name
                    })
            return

        elif is_skip_node:
            # 跳过节点，不处理其子节点
            return

        elif children:
            # 有子节点，继续递归
            for child in children:
                self.extract_names(child, level + 1, current_path)
        else:
            # 叶子节点，但不是特征值节点，不处理
            pass

    def extract_all(self):
        """提取所有数据"""
        if self.data:
            self.extract_names(self.data, 0, [])
        return self.results

    def save_to_excel(self, output_file: str = '试验树提取结果.xlsx'):
        """
        保存结果到Excel文件

        Args:
            output_file: 输出Excel文件名
        """
        if not self.results:
            print("没有数据可保存")
            return

        # 创建DataFrame
        df = pd.DataFrame(self.results)

        # 调整列顺序
        columns_order = ['试验', '型式试验的小类', '实际试验名称', '特性名称']
        df = df[columns_order]

        # 按层级排序
        df = df.sort_values(['试验', '型式试验的小类', '实际试验名称', '特性名称'])

        # 保存到Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='试验树结构', index=False)

            # 调整列宽
            worksheet = writer.sheets['试验树结构']
            column_widths = {'A': 20, 'B': 30, 'C': 35, 'D': 50}
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width

        print(f"数据已保存到: {output_file}")
        print(f"共提取 {len(self.results)} 条记录")

        # 显示数据统计
        print(f"\n=== 数据统计 ===")
        print(f"试验类型: {df['试验'].unique().tolist()}")
        print(f"型式试验的小类: {df['型式试验的小类'].unique().tolist()}")
        print(f"实际试验名称: {df['实际试验名称'].unique().tolist()}")

        # 显示前20条预览
        print("\n=== 数据预览（前20条）===")
        for i, row in df.head(20).iterrows():
            print(f"{i + 1}. 小类: {row['型式试验的小类']:20s} | "
                  f"试验: {row['实际试验名称']:25s} | "
                  f"特性: {row['特性名称'][:40]}")

        return df

    def debug_tree_structure(self, node: Dict, level: int = 0, prefix: str = ""):
        """
        调试函数：打印树结构，用于查看层级关系

        Args:
            node: 当前节点
            level: 当前层级
            prefix: 前缀
        """
        current_name = node.get('name', '')
        children = node.get('children', [])

        indent = "  " * level
        print(f"{indent}├─ {current_name} (层级: {level})")

        for child in children:
            self.debug_tree_structure(child, level + 1)


def extract_from_json_file(json_file_path: str, output_excel: str = None):
    """
    从JSON文件提取数据到Excel

    Args:
        json_file_path: JSON文件路径
        output_excel: 输出Excel文件路径（可选）
    """
    # 创建提取器
    extractor = TreeExtractor(json_file_path)

    # 加载JSON
    print(f"加载JSON文件: {json_file_path}")
    data = extractor.load_json()

    # 打印树结构用于调试
    print("\n=== 树结构（用于调试）===")
    extractor.debug_tree_structure(data)
    print("\n" + "=" * 60 + "\n")

    # 提取数据
    print("提取树结构数据（只提取特征值节点）...")
    extractor.extract_all()

    # 设置输出文件名
    if output_excel is None:
        output_excel = os.path.splitext(json_file_path)[0] + '_特征值提取结果.xlsx'

    # 保存到Excel
    extractor.save_to_excel(output_excel)

    return extractor.results


def main():
    """主函数"""
    # JSON文件路径
    json_file = r'./electrical_test_tree_20260409_180242.json'
    out_file= f'electrical_test_tree_{time_str}.xlsx'
    # 提取数据
    results = extract_from_json_file(json_file,out_file)

    print("\n✅ 提取完成！")
    print(f"共提取 {len(results)} 条特征值记录")


if __name__ == '__main__':
    """

    """
    main()