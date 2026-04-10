'''
从图谱接口迭代拉取形成json树
'''
import json
import requests
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import time



class LightRAGTreeBuilder:
    def __init__(self, base_url: str = "http://127.0.0.1:9621", token: str = None):
        """
        初始化LightRAG树构建器

        Args:
            base_url: API基础URL
            token: Bearer Token（可选）
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {}
        if token:
            self.headers['Authorization'] = f'Bearer {token}'

        # 四大试验根节点
        self.roots = [
            "绝缘性能型式试验",
            "温升性能型式试验",
            "开合性能型式试验",
            "短路性能型式试验"
        ]

    def fetch_graph_by_label(self, label: str, max_depth: int = 4, max_nodes: int = 1200) -> Dict:
        """
        通过标签获取图数据

        Args:
            label: 节点标签
            max_depth: 最大深度
            max_nodes: 最大节点数

        Returns:
            图数据字典
        """
        params = {
            'label': label,
            'max_depth': max_depth,
            'max_nodes': max_nodes
        }

        response = requests.get(
            f"{self.base_url}/graphs",
            params=params,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def search_labels(self, query: str, limit: int = 30) -> List[str]:
        """
        搜索标签

        Args:
            query: 搜索关键词
            limit: 返回数量限制

        Returns:
            标签列表
        """
        params = {
            'q': query,
            'limit': limit
        }

        try:
            response = requests.get(
                f"{self.base_url}/graph/label/search",
                params=params,
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []

    def resolve_best_root_label(self, expected_root: str) -> str:
        """
        解析最佳根节点标签

        Args:
            expected_root: 期望的根节点名称

        Returns:
            最佳匹配的标签
        """
        candidates = self.search_labels(expected_root, 30)

        if not candidates:
            return expected_root

        # 精确匹配 report:xxx
        exact_report = next((x for x in candidates if x == f"report:{expected_root}"), None)
        if exact_report:
            return exact_report

        # 精确匹配
        exact = next((x for x in candidates if x == expected_root), None)
        if exact:
            return exact

        # 包含匹配
        contains = next((x for x in candidates if expected_root in x), None)
        if contains:
            return contains

        # 反向包含匹配
        reverse_contains = next((x for x in candidates if x in expected_root), None)
        if reverse_contains:
            return reverse_contains

        return candidates[0]

    def node_props(self, node: Dict) -> Dict:
        """获取节点属性"""
        return node.get('properties', {})

    def node_display_name(self, node: Dict, fallback_id: str = "") -> str:
        """获取节点显示名称"""
        p = self.node_props(node)
        return (p.get('name') or p.get('report_type') or
                p.get('test_item') or p.get('param_name') or
                node.get('id', fallback_id))

    def build_schema_tree_from_graph(self, root_label: str, report_node_id: str, graph: Dict) -> Dict:
        """
        从图数据构建树结构

        Args:
            root_label: 根节点标签
            report_node_id: 报告节点ID
            graph: 图数据

        Returns:
            树结构字典
        """
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])

        # 构建节点映射
        node_by_id = {node['id']: node for node in nodes}

        # 构建出边映射
        outgoing = defaultdict(list)

        for edge in edges:
            ep = edge.get('properties', {})
            rel_type = ep.get('rel_type', edge.get('type', ''))
            src = ep.get('src_id', edge.get('source'))
            tgt = ep.get('tgt_id', edge.get('target'))

            if not src or not tgt or not rel_type:
                continue

            key = f"{src}||{rel_type}"
            outgoing[key].append({
                'targetId': tgt,
                'edgeProps': ep
            })

        def get_targets(src_id: str, rel_type: str) -> List[Dict]:
            """获取指定关系类型的目标节点"""
            return outgoing.get(f"{src_id}||{rel_type}", [])

        # 查找真实的报告节点ID
        real_report_id = report_node_id
        report_node = node_by_id.get(real_report_id)
        rp = self.node_props(report_node) if report_node else {}

        if rp.get('entity_type') != 'ReportType':
            for node in nodes:
                p = self.node_props(node)
                if (p.get('entity_type') == 'ReportType' and
                        (p.get('report_type') == root_label or p.get('name') == root_label)):
                    real_report_id = node['id']
                    break

        # 获取试验项目
        test_links = get_targets(real_report_id, 'INCLUDES_TEST')
        test_items = []

        for link in test_links:
            test_id = link['targetId']
            test_node = node_by_id.get(test_id)
            if not test_node:
                continue

            tp = self.node_props(test_node)
            if tp.get('entity_type') != 'TestItem':
                continue

            # 获取参数（特征值）
            param_links = get_targets(test_id, 'HAS_PARAMETER')
            params = []
            for pl in param_links:
                param_node = node_by_id.get(pl['targetId'])
                if not param_node:
                    continue

                pp = self.node_props(param_node)
                if pp.get('entity_type') != 'TestParameter':
                    continue

                cn_name = pp.get('param_name') or pp.get('name') or pp.get('param_key') or '参数'
                value_core = ' / '.join(filter(None, [pp.get('value_text'), pp.get('value_expr')]))
                value = ' '.join(filter(None, [value_core, pp.get('unit')])).strip()
                if not value:
                    value = '（未提取到具体值）'

                params.append({'name': f"{cn_name}: {value}"})

            # 获取条件/规则
            condition_items = []

            # 边上的条件
            if link.get('edgeProps', {}).get('condition'):
                condition_items.append({
                    'name': f"执行条件: {link['edgeProps']['condition']}"
                })

            # HAS_RULE关系
            rule_links = get_targets(test_id, 'HAS_RULE')
            for rl in rule_links:
                rule_node = node_by_id.get(rl['targetId'])
                if not rule_node:
                    continue

                rp = self.node_props(rule_node)
                if rp.get('entity_type') != 'TestRule':
                    continue

                parts = [rp.get('name'), rp.get('condition'), rp.get('expression')]
                parts = [p for p in parts if p]
                if parts:
                    # 修复：使用字符串拼接而不是嵌套f-string
                    rule_text = ' | '.join(parts)
                    condition_items.append({'name': f"规则: {rule_text}"})

            # 获取标准条款（原文切块）
            clause_links = get_targets(test_id, 'BASED_ON')
            clauses = []
            for cl in clause_links:
                clause_node = node_by_id.get(cl['targetId'])
                if not clause_node:
                    continue

                cp = self.node_props(clause_node)
                if cp.get('entity_type') != 'StandardClause':
                    continue

                quote = cp.get('quote', '').strip()
                if quote:
                    if len(quote) > 160:
                        quote = f"{quote[:160]}..."
                    clauses.append({'name': quote})
                else:
                    clause_info = ' '.join(filter(None, [cp.get('clause_id'), cp.get('clause_title')]))
                    if clause_info:
                        clauses.append({'name': clause_info})

            # 构建试验节点
            test_name = tp.get('test_item') or tp.get('name') or self.node_display_name(test_node, test_id)
            test_children = []

            if params:
                test_children.append({'name': '特征值', 'children': params})
            if condition_items:
                test_children.append({'name': '条件/规则', 'children': condition_items})
            if clauses:
                test_children.append({'name': '原文切块', 'children': clauses})

            test_items.append({
                'name': test_name,
                'children': test_children
            })

        return {
            'name': root_label,
            'children': test_items
        }

    def fetch_graph_recursive(self, root_label: str, max_layers: int = 6) -> Dict:
        """
        递归抓取图数据

        Args:
            root_label: 根节点标签
            max_layers: 最大递归层数

        Returns:
            合并后的图数据
        """
        visited_labels = {root_label}
        frontier = [root_label]
        all_graphs = []

        for layer in range(1, max_layers + 1):
            if not frontier:
                break

            print(f"递归抓取中：{root_label}，第 {layer}/{max_layers} 层（{len(frontier)} 个节点）")

            layer_graphs = []
            for label in frontier:
                try:
                    graph = self.fetch_graph_by_label(label, max_depth=1)
                    layer_graphs.append(graph)
                    all_graphs.append(graph)
                except Exception as e:
                    print(f"  获取 {label} 失败: {e}")

            # 收集下一层的标签
            next_labels = set()
            for graph in layer_graphs:
                for node in graph.get('nodes', []):
                    # 添加节点ID作为标签
                    if node.get('id'):
                        next_labels.add(node['id'])
                    # 添加节点标签
                    labels = node.get('labels', [])
                    for label in labels:
                        if label:
                            next_labels.add(label)

            # 过滤已访问的标签
            frontier = [label for label in next_labels if label not in visited_labels]
            visited_labels.update(frontier)

        # 合并所有图
        return self.merge_graphs(all_graphs)

    def merge_graphs(self, graphs: List[Dict]) -> Dict:
        """
        合并多个图数据

        Args:
            graphs: 图数据列表

        Returns:
            合并后的图数据
        """
        node_map = {}
        edge_map = {}

        for graph in graphs:
            # 合并节点
            for node in graph.get('nodes', []):
                node_map[node['id']] = node

            # 合并边
            for edge in graph.get('edges', []):
                key = f"{edge.get('source')}=>{edge.get('target')}::{edge.get('type', '')}"
                edge_map[key] = edge

        return {
            'nodes': list(node_map.values()),
            'edges': list(edge_map.values())
        }

    def load_and_build_tree(self, recursive: bool = True, max_depth: int = 4, max_layers: int = 6) -> Dict:
        """
        加载数据并构建树

        Args:
            recursive: 是否递归抓取
            max_depth: 非递归模式下的最大深度
            max_layers: 递归模式下的最大层数

        Returns:
            完整的树结构
        """
        trees = []
        summary = []

        for root_label in self.roots:
            print(f"处理: {root_label}")

            # 解析最佳根节点标签
            resolved_root = self.resolve_best_root_label(root_label)
            print(f"  解析后的标签: {resolved_root}")

            # 获取图数据
            if recursive:
                graph = self.fetch_graph_recursive(resolved_root, max_layers)
            else:
                graph = self.fetch_graph_by_label(resolved_root, max_depth)

            # 构建树
            tree = self.build_schema_tree_from_graph(root_label, resolved_root, graph)
            trees.append(tree)

            nodes_count = len(graph.get('nodes', []))
            edges_count = len(graph.get('edges', []))
            summary.append(f"{root_label}:{nodes_count}N/{edges_count}E")
            print(f"  {root_label}: {nodes_count}节点, {edges_count}边")

        # 构建完整树
        full_tree = {
            'name': '型式试验',
            'children': trees,
            'metadata': {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'recursive': recursive,
                'max_depth': max_depth,
                'max_layers': max_layers,
                'summary': ' | '.join(summary)
            }
        }

        return full_tree

    def export_to_json(self, output_file: str = 'electrical_test_tree.json', **kwargs):
        """
        导出树结构为JSON文件

        Args:
            output_file: 输出文件名
            **kwargs: 传递给load_and_build_tree的参数
        """
        print("开始构建试验树...")
        tree = self.load_and_build_tree(**kwargs)

        print(f"导出到文件: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tree, f, ensure_ascii=False, indent=2)

        print(f"导出完成！")
        print(f"总结: {tree['metadata']['summary']}")


def main():
    """主函数"""
    # 配置参数
    config = {
        'base_url': 'http://172.31.22.13:9625/',  # API地址
        'token': None,  # 如果需要认证，填入token
        'recursive': False,  # 改为False避免内存问题
        'max_depth': 2,  # 降低深度
        'max_layers': 3,  # 降低层数
        'output_file': 'electrical_test_tree.json'  # 输出文件
    }

    # 创建构建器实例
    builder = LightRAGTreeBuilder(
        base_url=config['base_url'],
        token=config['token']
    )

    # 导出JSON
    builder.export_to_json(
        output_file=config['output_file'],
        recursive=config['recursive'],
        max_depth=config['max_depth'],
        max_layers=config['max_layers']
    )


if __name__ == '__main__':
    main()