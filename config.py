# config.py
import os
from typing import Dict, Any

class Config:
    # 从环境变量获取模型路径，如果没有则使用默认值
    MODEL_BASE_PATH = os.getenv('MODEL_BASE_PATH', '/workspace/models')
    
    # Milvus配置
    MILVUS_URI = os.getenv('MILVUS_URI', "https://in03-f2c7c2ce6a1bdff.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn")
    MILVUS_USER = os.getenv('MILVUS_USER', "db_f2c7c2ce6a1bdff")
    MILVUS_PASSWORD = os.getenv('MILVUS_PASSWORD', "Ga7(rHn0xu(Uud")
    MILVUS_TOKEN = os.getenv('MILVUS_TOKEN', "85afa5d1098a63edeefad6c703e22e1bf9d8aa6cacdad29adf5fe13e82df4b4d647d646530fe098f79f0dc70da223673855f2b3d")
    
    # 模型路径 - 使用环境变量或默认路径
    LLM_MODEL_PATH = os.getenv('LLM_MODEL_PATH', f"{MODEL_BASE_PATH}/Qwen/Qwen2.5-3B-Instruct-AWQ")
    EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH', f"{MODEL_BASE_PATH}/Qwen/Qwen3-Embedding-0.6B")
    RERANKER_MODEL_PATH = os.getenv('RERANKER_MODEL_PATH', f"{MODEL_BASE_PATH}/BAAI/bge-reranker-large")
    
    # RAG配置
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', "enterprise_knowledge")
    
    # vLLM配置
    MAX_MODEL_LEN = int(os.getenv('MAX_MODEL_LEN', '8192'))
    GPU_MEMORY_UTILIZATION = float(os.getenv('GPU_MEMORY_UTILIZATION', '0.7'))

config = Config()