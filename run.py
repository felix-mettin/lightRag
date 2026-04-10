import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed,openai_complete_if_cache
from lightrag.utils import setup_logger
from lightrag.utils import EmbeddingFunc
from functools import partial
from lightrag.operate import chunking_by_token_size
setup_logger("lightrag", level="INFO")

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "Qwen3-Next-80B-A3B-Instruct",  # 你的本地模型名称
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="your-api-key",  # 如果需要的话
        # base_url="http://10.8.113.78:18088/llm_api_test4/v1",  # 你的本地API地址
        base_url="http://172.17.80.23:18088/llm_api_test4/v1",  # 你的本地API地址
        **kwargs
    )
#

# 方式一：使用 partial 包装你的配置（推荐）
embedding_func = EmbeddingFunc(
    embedding_dim=1024,  # bge-m3 的维度是 1024
    max_token_size=8192,
    model_name="bge-m3",
    func=partial(
        openai_embed.func,  # 使用 .func 获取未包装的原始函数
        model="bge-m3",
        api_key="your-api-key",
        # base_url="http://10.8.113.78:18088/vllm-bge-m3/v1",
        base_url="http://172.17.80.23:18088/vllm-bge-m3/v1",
    ),
)


async def create_knowledge_graph():
    # 1. 初始化LightRAG实例
    WORKING_DIR = "./data/debug/data/rag_storage"
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    rag = LightRAG(working_dir=WORKING_DIR,
                   embedding_func=embedding_func,
                   llm_model_func=llm_model_func,
                   # chunking_func=lambda tokenizer, content, split_by_character, split_by_character_only, chunk_overlap_token_size, chunk_token_size:
                   #                  chunking_by_token_size(
                   #                      tokenizer, content,
                   #                      split_by_character="\n\n",  # 按段落分割
                   #                      split_by_character_only=True,  # 只按段落分割，忽略token大小
                   #                      chunk_overlap_token_size=chunk_overlap_token_size,
                   #                      chunk_token_size=chunk_token_size
                   #                  ),
                   )

    # 2. 初始化存储（必需步骤）
    await rag.initialize_storages()

    # 3. 处理本地文档
    document_files = [
        # "./data/STL Guide to IEC 62271-100 Issue 4.0_20250601 - clean version.md"
        "/home/ict/下载/IEC/IEC 62271-100 2024 A.md"
    ]

    # 读取并插入文档
    for doc_file in document_files:
        if os.path.exists(doc_file):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
                await rag.ainsert(content)
                print(f"已处理文档: {doc_file}")
        else:
            print(f"文档不存在: {doc_file}")

    # 4. 使用mix模式查询知识图谱
    # query_param = QueryParam(mode="global")  # 使用mix模式
    #
    # # 示例查询
    # queries = [
    #     # "这个知识图谱中有哪些主要实体？",
    #     "40.5kV及以下断路器(用于提取参考文献)需要进行绝缘性能型式试验、 温升性能型式试验，"
    #     "现需要根据GB/T与额定电压 24 kV、额定雷电冲击耐受电压 125 kV、额定雷电冲击耐受电压（断口） 145 kV、"
    #     "额定短时工频耐受电压 65 kV、额定短时工频耐受电压（断口） 80 kV、额定电流 16000 A、额定频率 50 Hz、"
    #     "额定短路关合电流 343 kA、额定短路开断电流 125 kA、直流分量 75 %、"
    #     "最短分闸时间 6 ms、额定操作顺序 CO-1800s-CO 、"
    #     "额定短时耐受电流 125 kA、额定峰值耐受电流 343 kA、额定失步开断电流 62.5 kA输出试验逻辑。"
    # ]
    #
    # for query in queries:
    #     result = await rag.aquery(query, param=query_param)
    #     print(f"\n查询: {query}")
    #     print(f"回答: {result}")

        # 5. 清理资源
    await rag.finalize_storages()


if __name__ == "__main__":
    # 设置OpenAI API密钥
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    # 运行知识图谱创建
    asyncio.run(create_knowledge_graph())