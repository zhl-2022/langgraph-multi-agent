# config.py
import os
from typing import Dict, Any

class Config:
    # Milvus配置
    MILVUS_URI = "https://in03-f2c7c2ce6a1bdff.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
    MILVUS_USER = "db_f2c7c2ce6a1bdff"
    MILVUS_PASSWORD = "Ga7(rHn0xu(Uud"
    MILVUS_TOKEN = "85afa5d1098a63edeefad6c703e22e1bf9d8aa6cacdad29adf5fe13e82df4b4d647d646530fe098f79f0dc70da223673855f2b3d"
    
    # Qwen模型路径
    LLM_MODEL_PATH = "/workspace/models/Qwen/Qwen2.5-3B-Instruct-AWQ"
    EMBEDDING_MODEL_PATH = "/workspace/models/Qwen/Qwen3-Embedding-0.6B"
    RERANKER_MODEL_PATH = "/workspace/models/BAAI/bge-reranker-large"
    # RERANKER_MODEL_PATH = None
    
    # RAG配置
    COLLECTION_NAME = "enterprise_knowledge"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # vLLM配置
    MAX_MODEL_LEN = 8192
    GPU_MEMORY_UTILIZATION = 0.7

config = Config()