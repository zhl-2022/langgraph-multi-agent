# rag/stable_retriever.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict
from config import Config
import os

class StableReranker:
    """稳定版Reranker - 逐文档处理，避免批量问题"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载模型 - 简化版本"""
        try:
            print(f"🔄 加载Reranker模型从: {self.model_path}")
            
            # 使用更简单的加载方式
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 强制设置padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("✅ Reranker模型加载完成")
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            self.model = None
            self.tokenizer = None
    
    def rerank_serial(self, query: str, documents: List[str]) -> List[Dict]:
        """串行重排序 - 最稳定的方法"""
        if not documents or self.model is None:
            return []
        
        results = []
        
        for i, doc in enumerate(documents):
            try:
                # 对每个文档单独处理
                pair = [query, doc]
                
                inputs = self.tokenizer(
                    pair,
                    padding='max_length',  # 使用固定长度padding
                    truncation=True,
                    max_length=256,  # 使用较短的序列长度
                    return_tensors="pt"
                )
                
                # 移动到设备
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    score = torch.softmax(outputs.logits, dim=1)[0, 1].item()
                
                results.append({
                    'document': doc,
                    'score': score,
                    'rank': i
                })
                
            except Exception as e:
                print(f"❌ 处理文档 {i} 失败: {e}")
                results.append({
                    'document': doc,
                    'score': 0.5,
                    'rank': i
                })
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

class StableRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.reranker = StableReranker(config.RERANKER_MODEL_PATH) if config.RERANKER_MODEL_PATH else None
    
    def retrieve(self, query: str, vector_store, top_k: int = 5) -> List[Dict]:
        """稳定版检索"""
        try:
            if vector_store is None:
                return []
                
            # 向量检索
            vector_results = vector_store.similarity_search(query, k=top_k)
            
            if not vector_results or self.reranker is None:
                return vector_results[:top_k]
            
            # 重排序
            documents = [result['content'] for result in vector_results]
            reranked = self.reranker.rerank_serial(query, documents)
            
            # 合并结果
            final_results = []
            for item in reranked[:top_k]:
                idx = item['rank']
                if idx < len(vector_results):
                    final_result = vector_results[idx].copy()
                    final_result['rerank_score'] = item['score']
                    final_results.append(final_result)
            
            return final_results
            
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return []