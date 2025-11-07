# rag/bge_retriever.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict
from config import Config
import numpy as np

class BGEReranker:
    """BGE-Reranker封装 - 经过充分测试的稳定版本"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载BGE-Reranker模型"""
        try:
            print(f"🔄 加载BGE-Reranker模型: {self.model_path}")
            
            # BGE模型通常有很好的兼容性
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # BGE模型通常已经正确配置了padding
            print(f"📋 Tokenizer配置 - pad_token: {self.tokenizer.pad_token}, pad_token_id: {self.tokenizer.pad_token_id}")
            print("✅ BGE-Reranker模型加载完成")
            
        except Exception as e:
            print(f"❌ 加载BGE-Reranker失败: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str]) -> List[Dict]:
        """重排序文档 - BGE专用方法"""
        if not documents:
            return []
        
        try:
            # BGE模型的输入格式
            pairs = [[query, doc] for doc in documents]
            
            # 编码输入
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.model.device)
            
            # 推理
            with torch.no_grad():
                scores = self.model(**inputs).logits.squeeze(-1)
                scores = torch.sigmoid(scores).cpu().numpy()
            
            # 构建结果
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                results.append({
                    'document': doc,
                    'score': float(score),
                    'rank': i
                })
            
            # 按分数排序
            results.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"✅ BGE重排序完成，处理了 {len(results)} 个文档")
            return results
            
        except Exception as e:
            print(f"❌ BGE重排序失败: {e}")
            # 返回默认结果
            return [{'document': doc, 'score': 0.5, 'rank': i} for i, doc in enumerate(documents)]

class BGERetriever:
    """BGE检索器"""
    
    def __init__(self, config: Config):
        self.config = config
        
        try:
            self.reranker = BGEReranker(config.RERANKER_MODEL_PATH)
            print("✅ BGE检索器初始化成功")
        except Exception as e:
            print(f"❌ BGE检索器初始化失败: {e}")
            self.reranker = None
    
    def retrieve(self, query: str, vector_store, top_k: int = 10, rerank_k: int = 5) -> List[Dict]:
        """检索方法"""
        try:
            if vector_store is None or vector_store.collection is None:
                print("⚠️ Milvus不可用，返回空结果")
                return []
                
            # 1. 向量检索
            vector_results = vector_store.similarity_search(query, k=top_k)
            
            if not vector_results:
                return []
            
            # 如果没有reranker，直接返回结果
            if self.reranker is None:
                return vector_results[:rerank_k]
            
            # 2. 提取文档内容用于重排序
            documents = [result['content'] for result in vector_results]
            
            # 3. 使用BGE-Reranker进行精排
            print("🔄 使用BGE-Reranker进行重排序...")
            reranked_results = self.reranker.rerank(query, documents)
            
            # 4. 合并结果
            final_results = []
            for rerank_item in reranked_results[:rerank_k]:
                original_index = rerank_item['rank']
                if original_index < len(vector_results):
                    final_result = vector_results[original_index].copy()
                    final_result['rerank_score'] = rerank_item['score']
                    # 结合向量距离和重排序分数
                    final_result['final_score'] = (
                        rerank_item['score'] * 0.7 + 
                        (1 - final_result.get('distance', 0)) * 0.3
                    )
                    final_results.append(final_result)
            
            # 按最终分数排序
            final_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
            print(f"✅ BGE检索完成，返回 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            print(f"❌ BGE检索失败: {e}")
            # 返回原始向量检索结果
            return vector_results[:rerank_k] if 'vector_results' in locals() else []