# rag/simple_retriever.py
from typing import List, Dict
from config import Config

class SimpleRetriever:
    """简化版检索器，不使用Reranker"""
    def __init__(self, config: Config):
        self.config = config
    
    def retrieve(self, query: str, vector_store, top_k: int = 5) -> List[Dict]:
        """简化检索 - 只使用向量搜索"""
        try:
            if vector_store is None or vector_store.collection is None:
                print("⚠️ Milvus不可用，返回空结果")
                return []
                
            # 直接使用向量检索
            vector_results = vector_store.similarity_search(query, k=top_k)
            
            # 简单处理：按距离排序（距离越小越好）
            vector_results.sort(key=lambda x: x.get('distance', 0))
            
            print(f"✅ 简化检索完成，返回 {len(vector_results)} 个结果")
            return vector_results
            
        except Exception as e:
            print(f"❌ 检索过程中出错: {e}")
            return []