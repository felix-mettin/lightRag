# LightRAG 智能体开发指南

本文件专为 AI 编程智能体设计，提供项目架构、开发流程和最佳实践的完整参考。

## 项目概述

**LightRAG** 是一个简单且快速的检索增强生成（RAG）框架，通过基于图的知识表示来增强信息检索与生成能力。项目由香港大学数据科学实验室（HKUDS）开发维护。

### 核心特性

- **双级检索系统**：结合局部和全局检索方法，支持多种查询模式（local, global, hybrid, naive, mix, bypass）
- **知识图谱支持**：基于图的实体和关系存储，支持多种图数据库后端
- **多模态文档处理**：通过 RAG-Anything 集成支持文本、图像、表格和公式
- **Ollama 兼容接口**：提供 Ollama 风格的 API，便于 AI 聊天机器人集成
- **重排序支持**：集成 Cohere、Jina、阿里云等重排序服务

### 项目元数据

- **包名**: `lightrag-hku`
- **版本**: 1.4.9.12
- **Python 要求**: >= 3.10
- **许可证**: MIT
- **仓库**: https://github.com/HKUDS/LightRAG

---

## 项目架构

### 目录结构

```
lightrag/                      # 核心 Python 包
├── lightrag.py               # 核心编排器，LightRAG 主类
├── base.py                   # 抽象基类和数据模型（QueryParam, TextChunkSchema 等）
├── operate.py                # 核心操作逻辑（实体/关系提取、知识图谱操作）
├── rule/                       # 业务规则模块
│   ├── __init__.py             # Workflow 类导出
│   ├── utils.py                # 通用工具函数
│   ├── tree_override.py        # 注释记忆和树覆盖规则
│   ├── domain_rules.py         # 领域规则评估
│   ├── extraction.py           # 知识图谱提取和构建
│   ├── query_processing.py     # 查询处理和响应增强
│   ├── workflows.py            # Workflow 类
│   └── README.md               # 模块文档
├── prompt.py                 # LLM 提示词模板
├── utils.py                  # 通用工具函数
├── utils_graph.py            # 图相关工具函数
├── rerank.py                 # 重排序实现
├── constants.py              # 全局常量配置
├── types.py                  # 类型定义
├── exceptions.py             # 自定义异常
├── namespace.py              # 命名空间管理
├── api/                      # FastAPI 服务
│   ├── lightrag_server.py    # API 服务主入口
│   ├── config.py             # API 配置管理
│   ├── auth.py               # 认证处理
│   ├── utils_api.py          # API 工具函数
│   ├── gunicorn_config.py    # Gunicorn 配置
│   ├── run_with_gunicorn.py  # Gunicorn 启动器
│   ├── routers/              # API 路由
│   │   ├── document_routes.py   # 文档管理接口
│   │   ├── query_routes.py      # 查询接口
│   │   ├── graph_routes.py      # 图谱操作接口
│   │   └── ollama_api.py        # Ollama 兼容接口
│   ├── static/               # 静态资源
│   └── webui/                # WebUI 构建产物
├── kg/                       # 知识图谱存储适配器
│   ├── neo4j_impl.py         # Neo4j 实现
│   ├── postgres_impl.py      # PostgreSQL + pgvector 实现
│   ├── mongo_impl.py         # MongoDB 实现
│   ├── milvus_impl.py        # Milvus 实现
│   ├── qdrant_impl.py        # Qdrant 实现
│   ├── redis_impl.py         # Redis 实现
│   ├── memgraph_impl.py      # Memgraph 实现
│   ├── networkx_impl.py      # NetworkX 实现
│   ├── faiss_impl.py         # FAISS 实现
│   ├── nano_vector_db_impl.py  # NanoVectorDB 实现
│   ├── json_doc_status_impl.py # JSON 文档状态存储
│   ├── json_kv_impl.py       # JSON KV 存储
│   └── shared_storage.py     # 共享存储管理
├── llm/                      # LLM 提供商绑定
│   ├── openai.py             # OpenAI / Azure OpenAI
│   ├── gemini.py             # Google Gemini
│   ├── anthropic.py          # Anthropic Claude
│   ├── ollama.py             # Ollama
│   ├── bedrock.py            # AWS Bedrock
│   ├── hf.py                 # HuggingFace
│   ├── zhipu.py              # 智谱 AI
│   ├── jina.py               # Jina AI
│   ├── lmdeploy.py           # LMDeploy
│   ├── lollms.py             # LoLLMs
│   ├── nvidia_openai.py      # NVIDIA OpenAI
│   ├── llama_index_impl.py   # LlamaIndex 集成
│   └── binding_options.py    # 绑定配置选项
├── config/                   # 配置文件目录
│   ├── annotation_memory.json         # 注释记忆库
│   ├── profiles/                      # 标准配置文件
│   │   ├── annotation_memory.gb.json  # 国标注释记忆
│   │   ├── annotation_memory.iec.json # IEC 标准注释记忆
│   │   └── annotation_memory.dlt.json # DLT 标准注释记忆
│   └── domain_rules/                  # 领域规则配置
│       └── insulation.gb.json         # 绝缘规则配置（国标）
└── tools/                    # 工具脚本
    ├── download_cache.py           # 缓存下载工具
    ├── clean_llm_query_cache.py    # 清理 LLM 查询缓存
    ├── build_annotation_memory.py  # 构建注释记忆
    └── clean_annotation_memory.py  # 清理注释记忆

lightrag_webui/               # React + TypeScript Web 前端
├── src/
│   ├── components/          # UI 组件
│   ├── features/            # 功能模块
│   ├── hooks/               # React Hooks
│   ├── services/            # API 服务
│   ├── stores/              # Zustand 状态管理
│   ├── locales/             # 国际化资源
│   └── utils/               # 工具函数
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json

tests/                        # 测试套件
├── conftest.py               # Pytest 配置和固件
├── test_*.py                 # 测试文件
└── README_WORKSPACE_ISOLATION_TESTS.md  # 工作空间隔离测试文档

examples/                     # 使用示例
├── lightrag_openai_demo.py
├── lightrag_ollama_demo.py
├── lightrag_gemini_demo.py
├── lightrag_azure_openai_demo.py
├── lightrag_vllm_demo.py
└── unofficial-sample/        # 社区贡献示例

k8s-deploy/                   # Kubernetes 部署配置
├── databases/                # 数据库部署配置
├── lightrag/                 # LightRAG 应用部署
├── install_lightrag.sh       # 安装脚本
└── README.md

tools/                        # 项目工具
├── activate_profile.sh       # 激活配置文件
├── build_annotation_memory.py
├── clean_annotation_memory.py
├── excel_to_json_map.py
├── graph_json.py
├── json_map_excel.py
└── *.json                    # 电气试验树数据文件

data/                         # 数据目录（多标准支持）
├── gbt/                      # 国标数据
├── gbt_gy/                   # 国高压数据
├── iec/                      # IEC 标准数据
├── iec_gy/                   # IEC 高压数据
├── dlt/                      # DLT 标准数据
├── dlt_gy/                  # DLT 高压数据
└── debug/                    # 调试数据
```

---

## 构建与测试命令

### 环境设置

```bash
# 使用 uv（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv sync --extra api
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows
```

### 安装命令

```bash
# 安装核心包
pip install -e .

# 安装 API 支持
pip install -e ".[api]"

# 安装完整离线包（含所有存储后端和 LLM 提供商）
pip install -e ".[offline]"

# 安装开发测试依赖
pip install -e ".[test]"

# 使用 uv 安装
uv sync --extra api
```

### WebUI 构建

```bash
cd lightrag_webui

# 安装依赖（需要 Bun）
bun install --frozen-lockfile

# 开发模式
bun run dev

# 构建（输出到 lightrag/api/webui）
bun run build

# 运行测试
bun test
bun run test:coverage

# 代码检查
bun run lint
```

### 服务启动

```bash
# 标准启动（使用当前目录 .env 配置）
lightrag-server

# 或使用 uvicorn
uvicorn lightrag.api.lightrag_server:app --reload

# 使用 Gunicorn（生产环境）
lightrag-gunicorn

# 指定工作目录启动
cd /path/to/workspace && lightrag-server
```

### 测试命令

```bash
# 运行离线测试（默认，不依赖外部服务）
python -m pytest tests

# 运行集成测试（需要外部数据库/API 服务）
python -m pytest tests --run-integration
# 或设置环境变量
LIGHTRAG_RUN_INTEGRATION=true python -m pytest tests

# 保留测试产物用于检查
python -m pytest tests --keep-artifacts
# 或
LIGHTRAG_KEEP_ARTIFACTS=true python -m pytest tests

# 压力测试模式
python -m pytest tests --stress-test --test-workers 5

# 代码检查
ruff check .
ruff check . --fix
```

### Docker 部署

```bash
# 构建并启动
docker compose up

# 构建特定服务
docker compose up lightrag_gbt

# 使用生产镜像
docker pull ghcr.io/hkuds/lightrag:latest
```

---

## 代码风格指南

### Python 代码规范

- **缩进**: 4 个空格
- **行长度**: 遵循 PEP 8，建议 88-100 字符
- **类型注解**: 为函数参数和返回值添加类型注解
- **文档字符串**: 使用双引号的多行字符串
- **日志**: 使用 `lightrag.utils.logger`，避免直接使用 `print`

```python
from lightrag.utils import logger

logger.info("处理文档: %s", doc_id)
logger.debug("调试信息: %s", details)
```

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块 | 小写 + 下划线 | `lightrag_server.py`, `neo4j_impl.py` |
| 类 | PascalCase | `LightRAG`, `QueryParam`, `OllamaServerInfos` |
| 函数/方法 | 小写 + 下划线 | `async_query`, `extract_entities` |
| 常量 | 大写 + 下划线 | `DEFAULT_TOP_K`, `GRAPH_FIELD_SEP` |
| 私有成员 | 前缀下划线 | `_internal_cache`, `_validate_input` |

### 代码组织原则

1. **单一职责**: 每个模块/类只负责一个明确的职责
2. **抽象基类**: 存储后端和 LLM 绑定均基于抽象基类实现
3. **配置集中**: 常量定义在 `constants.py`，避免硬编码
4. **错误处理**: 使用自定义异常类，提供有意义的错误信息

### 导入规范

```python
# 标准库
import os
import logging
from typing import Optional, Dict, Any

# 第三方库
import numpy as np
from fastapi import FastAPI

# 项目内部
from lightrag.base import QueryParam
from lightrag.utils import logger
from lightrag.constants import DEFAULT_TOP_K
```

---

## 前端开发规范

### 技术栈

- **框架**: React 19
- **语言**: TypeScript
- **构建**: Vite + Bun
- **样式**: Tailwind CSS 4
- **状态管理**: Zustand
- **组件库**: Radix UI
- **图表**: Sigma.js / Graphology

### 代码规范

- **缩进**: 2 个空格
- **组件**: 使用函数组件 + Hooks，PascalCase 命名
- **文件**: 组件文件使用 `.tsx`，工具函数使用 `.ts`
- **导入顺序**: React → 第三方 → 项目内部 → 类型定义

```typescript
import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useGraphStore } from '@/stores/graphStore';
import type { NodeData } from '@/types/graph';
```

---

## 测试策略

### 测试标记

| 标记 | 描述 | 运行条件 |
|------|------|----------|
| `offline` | 离线测试，无外部依赖 | 默认运行 |
| `integration` | 集成测试，需要外部服务 | 需 `--run-integration` |
| `requires_db` | 需要数据库连接 | 需配置环境变量 |
| `requires_api` | 需要 API 服务 | 需配置环境变量 |

### 测试配置

```python
# conftest.py 提供的固件
keep_test_artifacts    # 是否保留测试产物
stress_test_mode       # 压力测试模式
parallel_workers       # 并行工作线程数
run_integration_tests  # 是否运行集成测试
```

### 环境变量配置

```bash
# 测试相关
LIGHTRAG_KEEP_ARTIFACTS=true      # 保留测试产物
LIGHTRAG_STRESS_TEST=true         # 启用压力测试
LIGHTRAG_TEST_WORKERS=5           # 测试工作线程数
LIGHTRAG_RUN_INTEGRATION=true     # 运行集成测试

# 存储后端配置（用于集成测试）
LIGHTRAG_NEO4J_URI=bolt://localhost:7687
LIGHTRAG_NEO4J_USER=neo4j
LIGHTRAG_NEO4J_PASSWORD=password

LIGHTRAG_POSTGRES_HOST=localhost
LIGHTRAG_POSTGRES_PORT=5432
LIGHTRAG_POSTGRES_USER=postgres
LIGHTRAG_POSTGRES_PASSWORD=password
```

---

## 配置系统

### 配置文件优先级

1. 环境变量（最高优先级）
2. `.env` 文件（当前工作目录）
3. `config.ini`（当前工作目录）
4. 默认值（`constants.py`）

### 关键配置文件

#### .env 文件

环境变量配置文件，包含：
- **服务器配置**: `HOST`, `PORT`, `WORKERS`, `TIMEOUT`
- **LLM 配置**: `LLM_BINDING`, `LLM_MODEL`, `LLM_BINDING_HOST`, `LLM_BINDING_API_KEY`
- **嵌入配置**: `EMBEDDING_BINDING`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`
- **存储配置**: `STORAGE_NAMESPACE`, `LIGHTRAG_KV_STORAGE`, `LIGHTRAG_VECTOR_STORAGE`, `LIGHTRAG_GRAPH_STORAGE`
- **认证配置**: `AUTH_ACCOUNTS`, `TOKEN_SECRET`, `LIGHTRAG_API_KEY`
- **查询配置**: `TOP_K`, `MAX_ENTITY_TOKENS`, `MAX_RELATION_TOKENS`, `MAX_TOTAL_TOKENS`

#### config.ini 文件

电气标准领域专用配置：

```ini
[ELECTRICAL_SCHEMA]
standard_id =                  # 标准ID
standard_name =                # 标准名称
report_types =                 # 报告类型列表（逗号分隔）
report_aliases =               # 报告类型别名（key:value; 格式）
test_items =                   # 试验项目列表
test_aliases =                 # 试验项目别名
param_map =                    # 参数映射表
annotation_memory_path =       # 注释记忆文件路径
electrical_rules_path =        # 电气规则文件路径
```

### 存储后端类型

| 存储类型 | KV 存储 | 向量存储 | 图存储 |
|---------|---------|---------|--------|
| JSON | `JsonKVStorage` | `NanoVectorDBStorage` | `NetworkXStorage` |
| PostgreSQL | `PGKVStorage` | `PGVectorStorage` | `PGGraphStorage` |
| MongoDB | `MongoKVStorage` | `MongoVectorStorage` | `MongoGraphStorage` |
| Neo4j | - | - | `Neo4JStorage` |
| Milvus | - | `MilvusVectorStorage` | - |
| Qdrant | - | `QdrantStorage` | - |
| Redis | `RedisKVStorage` | - | - |
| Memgraph | - | - | `MemgraphStorage` |
| FAISS | - | `FaissVectorStorage` | - |

---

## 部署流程

### 本地开发部署

```bash
# 1. 克隆仓库
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG

# 2. 配置环境
cp env.example .env
# 编辑 .env 配置 LLM 和 Embedding

# 3. 构建 WebUI
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..

# 4. 启动服务
lightrag-server
```

### Docker 部署

```bash
# 多实例部署（支持不同标准）
docker compose up -d lightrag_gbt      # 国标
docker compose up -d lightrag_gbt_gy   # 国高压
docker compose up -d lightrag_iec      # IEC
docker compose up -d lightrag_iec_gy   # IEC 高压
docker compose up -d lightrag_dlt      # DLT
docker compose up -d lightrag_dlt_gy   # DLT 高压
```

### Kubernetes 部署

```bash
cd k8s-deploy

# 安装
cd databases && kubectl apply -f .  # 先部署数据库
cd ../lightrag && kubectl apply -f .  # 再部署应用

# 卸载
./uninstall_lightrag.sh
```

---

## 安全注意事项

### 认证与授权

1. **账号配置**: 使用 `AUTH_ACCOUNTS` 环境变量配置基本认证账号
   ```bash
   AUTH_ACCOUNTS='admin:admin123,user1:pass456'
   ```

2. **API 密钥**: 使用 `LIGHTRAG_API_KEY` 配置 API 访问密钥
   ```bash
   LIGHTRAG_API_KEY=your-secure-api-key-here
   ```

3. **JWT 配置**: 配置 Token 密钥和过期时间
   ```bash
   TOKEN_SECRET=Your-Key-For-LightRAG-API-Server
   TOKEN_EXPIRE_HOURS=48
   TOKEN_AUTO_RENEW=true
   TOKEN_RENEW_THRESHOLD=0.5
   ```

### 敏感数据处理

1. **绝不提交**: `.env` 文件、包含真实连接字符串的配置文件
2. **日志脱敏**: 确保日志中不包含 API 密钥、密码等敏感信息
3. **Token 管理**: `lightrag.log*` 文件作为本地产物，分享前需清理敏感信息

### 网络安全

1. **CORS 配置**: 生产环境限制 `CORS_ORIGINS`
   ```bash
   CORS_ORIGINS=https://your-domain.com
   ```

2. **SSL/TLS**: 生产环境启用 HTTPS
   ```bash
   SSL=true
   SSL_CERTFILE=/path/to/cert.pem
   SSL_KEYFILE=/path/to/key.pem
   ```

---

## Electrical Controlled 模式 Workflow 架构

### 概述

`electrical_controlled` 模式用于电气标准领域的知识图谱提取和查询，采用 Workflow 架构将复杂逻辑模块化。

### 核心 Workflow 类

#### 1. SchemaExtractionWorkflow

负责从电气标准文档中提取结构化知识并构建图谱。

```python
from lightrag.rule import SchemaExtractionWorkflow

workflow = SchemaExtractionWorkflow(global_config)
nodes, edges = await workflow.process_extraction(
    raw_response=llm_response,
    chunk_text=chunk_content,
    chunk_id="chunk_001",
    file_path="/docs/standard.pdf"
)
```

**处理流程**:
1. Configuration Loading - 加载 schema 配置和注释记忆
2. Chunk Processing - 解析条款信息
3. Payload Validation - 验证 LLM 提取结果
4. Tree Override Resolution - 应用人工标注规则
5. Node/Edge Building - 构建知识图谱节点和边

#### 2. QueryAugmentationWorkflow

负责查询时的上下文增强和领域规则应用。

```python
from lightrag.rule import QueryAugmentationWorkflow

workflow = QueryAugmentationWorkflow(global_config)
context = await workflow.augment_query(
    query="额定电压12kV的断路器需要做哪些绝缘试验？",
    entities_context=retrieved_entities,
    relations_context=retrieved_relations,
    text_chunks=retrieved_chunks
)

project_param_map = context["project_param_map"]
domain_rule_decisions = context["domain_rule_decisions"]
allowed_test_items = context["allowed_final_test_items"]
```

**处理流程**:
1. Domain Rule Evaluation - 评估拆分/合并/适用性规则
2. Report Scope Extraction - 提取报告类型作用域
3. Project Context Building - 构建项目参数上下文
4. Context Filtering - 应用报告作用域过滤
5. Prompt Enhancement - 生成增强的提示词上下文

### 配置文件

#### annotation_memory.json

人工标注的覆盖规则，用于修正 LLM 提取结果：

```json
{
  "tests_by_path": {
    "绝缘性能型式试验 > 工频耐受电压试验": {
      "test_name": "工频耐受电压试验",
      "parameters": [...],
      "remove_parameters": [...]
    }
  },
  "add_test_items": [...]
}
```

#### domain_rules (insulation.gb.json)

领域特定规则，用于查询时决策：

```json
{
  "rules": [
    {
      "rule_id": "insulation.gb.power_frequency_split",
      "kind": "split",
      "test_item": "工频耐受电压试验",
      "inputs": {...},
      "split_output": [...]
    }
  ]
}
```

### 向后兼容

现有函数（`_validate_controlled_payload`, `_build_controlled_nodes_edges` 等）仍然可用。Workflow 类是增量添加的，不破坏现有代码。

---

## 开发工作流

### 提交规范

- 使用简洁的祈使式提交信息
- 重大变更在 PR 中说明操作影响
- 提交前运行 `ruff check .` 和 `python -m pytest`

### 新增存储后端

1. 在 `lightrag/kg/` 创建新的实现文件
2. 继承相应的抽象基类（`BaseKVStorage`, `BaseVectorStorage`, `BaseGraphStorage`）
3. 实现所有抽象方法
4. 在 `lightrag/kg/__init__.py` 注册新后端
5. 添加对应的测试用例

### 新增 LLM 提供商

1. 在 `lightrag/llm/` 创建新的绑定文件
2. 实现模型调用函数和嵌入函数
3. 在 `lightrag/api/lightrag_server.py` 添加配置支持
4. 更新配置选项类（`lightrag/llm/binding_options.py`）

### 调试技巧

```python
# 启用详细调试日志
VERBOSE=True lightrag-server

# 或在代码中
from lightrag.utils import set_verbose_debug
set_verbose_debug(True)

# 查看日志
 tail -f lightrag.log
```

---

## 常见问题

### 依赖管理

项目使用 `uv` 进行快速可靠的 Python 包管理。`uv.lock` 文件确保依赖版本一致性。

### 离线部署

参考 `docs/OfflineDeployment.md` 获取离线环境部署指南，包括预安装所有依赖和缓存文件的方法。

### 内存优化

对于大规模数据集，配置适当的 `MAX_ENTITY_TOKENS`、`MAX_RELATION_TOKENS` 和 `MAX_TOTAL_TOKENS` 以控制上下文大小。

---

*本文件最后更新: 2026-04-10*
