# verify_qwen_models.py
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import Config

def verify_qwen_models():
    """验证Qwen模型配置"""
    config = Config()
    
    print("🔍 验证Qwen模型配置")
    print("=" * 50)
    
    try:
        # 验证Embedding模型
        print("📐 验证Embedding模型...")
        embedding_tokenizer = AutoTokenizer.from_pretrained(
            config.EMBEDDING_MODEL_PATH,
            trust_remote_code=True
        )
        embedding_model = AutoModel.from_pretrained(
            config.EMBEDDING_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 测试Embedding
        texts = ["这是一个测试文本"]
        embeddings = embedding_model.embed_documents(texts)
        print(f"✅ Embedding模型: {config.EMBEDDING_MODEL_PATH}")
        print(f"📐 Embedding维度: {embeddings.shape[1]}")
        print(f"📐 样本数量: {embeddings.shape[0]}")
        
        # 验证Reranker模型
        print("\n🔍 验证Reranker模型...")
        reranker_tokenizer = AutoTokenizer.from_pretrained(
            config.RERANKER_MODEL_PATH,
            trust_remote_code=True
        )
        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            config.RERANKER_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✅ Reranker模型: {config.RERANKER_MODEL_PATH}")
        print("🎯 Reranker模型加载成功")
        
        return embeddings.shape[1]
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return None

if __name__ == "__main__":
    verify_qwen_models()