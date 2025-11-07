# rag/retriever.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Optional
from config import Config
import os
import logging

logger = logging.getLogger(__name__)

class QwenReranker:
    """Qwen3-Reranker模型封装 - 修复版本"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载Qwen3-Reranker模型"""
        try:
            print(f"🔄 加载Qwen3-Reranker模型从: {self.model_path}")
            
            # 检查路径是否存在
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
            
            # 先加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 关键修复：正确设置padding token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    print(f"✅ 使用eos_token作为pad_token: {self.tokenizer.pad_token}")
                elif self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                    print(f"✅ 使用unk_token作为pad_token: {self.tokenizer.pad_token}")
                else:
                    # 如果都没有，添加一个特殊的pad_token
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    print("✅ 添加了新的pad_token: [PAD]")
            
            # 确保pad_token_id已设置
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else 0
            
            print(f"📋 Tokenizer配置: pad_token={self.tokenizer.pad_token}, pad_token_id={self.tokenizer.pad_token_id}")
            
            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 如果添加了新的token，需要调整模型嵌入层
            if len(self.tokenizer) != self.model.config.vocab_size:
                print("🔄 调整模型词汇表大小...")
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            print("✅ Qwen3-Reranker模型加载完成")
            
        except Exception as e:
            print(f"❌ 加载Reranker模型失败: {e}")
            self.model = None
            self.tokenizer = None
            raise
    
    def rerank_single(self, query: str, document: str) -> float:
        """单文档重排序 - 避免批量处理问题"""
        try:
            # 构建单个输入对
            pair = [query, document]
            
            # 编码单个输入
            inputs = self.tokenizer(
                pair,
                padding=True,  # 单个样本也需要padding以确保一致性
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=True
            ).to(self.model.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = torch.softmax(outputs.logits, dim=1)[0, 1].item()
            
            return score
            
        except Exception as e:
            print(f"❌ 单文档重排序失败: {e}")
            return 0.5  # 默认分数
    
    def rerank(self, query: str, documents: List[str]) -> List[Dict]:
        """对文档进行重排序 - 使用逐文档处理避免批量问题"""
        if not documents or self.model is None:
            return []
        
        try:
            # 逐文档处理，避免批量padding问题
            results = []
            for i, doc in enumerate(documents):
                score = self.rerank_single(query, doc)
                results.append({
                    'document': doc,
                    'score': score,
                    'rank': i
                })
            
            # 按分数排序
            results.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"✅ 重排序完成，处理了 {len(results)} 个文档")
            return results
            
        except Exception as e:
            print(f"❌ 重排序失败: {e}")
            # 返回原始顺序
            return [{'document': doc, 'score': 0.5, 'rank': i} for i, doc in enumerate(documents)]

class HybridRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.reranker = None
        
        # 只有在提供了Reranker模型路径时才初始化
        if config.RERANKER_MODEL_PATH and os.path.exists(config.RERANKER_MODEL_PATH):
            try:
                self.reranker = QwenReranker(config.RERANKER_MODEL_PATH)
                print("✅ Reranker初始化成功")
            except Exception as e:
                print(f"⚠️ Reranker初始化失败，将使用简化检索: {e}")
                self.reranker = None
        else:
            print("⚠️ 未配置Reranker模型路径，使用简化检索")
    
    def retrieve(self, query: str, vector_store, top_k: int = 10, rerank_k: int = 5) -> List[Dict]:
        """混合检索与重排序"""
        try:
            if vector_store is None or vector_store.collection is None:
                print("⚠️ Milvus不可用，返回空结果")
                return []
                
            # 1. 向量检索
            vector_results = vector_store.similarity_search(query, k=top_k)
            
            if not vector_results:
                return []
            
            # 如果没有reranker或者reranker失败，直接返回向量检索结果
            if self.reranker is None or self.reranker.model is None:
                print("⚠️ 使用简化检索（无Reranker）")
                return vector_results[:rerank_k]
            
            # 2. 提取文档内容用于重排序
            documents = [result['content'] for result in vector_results]
            
            # 3. 使用Qwen3-Reranker进行精排
            print("🔄 使用Qwen3-Reranker进行重排序...")
            reranked_results = self.reranker.rerank(query, documents)
            
            # 4. 合并结果
            final_results = []
            for rerank_item in reranked_results[:rerank_k]:
                original_index = rerank_item['rank']
                if original_index < len(vector_results):
                    final_result = vector_results[original_index].copy()
                    final_result['rerank_score'] = rerank_item['score']
                    final_result['final_score'] = rerank_item['score'] - final_result.get('distance', 0) * 0.1
                    final_results.append(final_result)
            
            # 按最终分数排序
            final_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
            print(f"✅ 检索完成，返回 {len(final_results)} 个重排序结果")
            return final_results
            
        except Exception as e:
            print(f"❌ 检索过程中出错: {e}")
            # 返回原始向量检索结果
            return vector_results[:rerank_k] if 'vector_results' in locals() else []