# rag/ultimate_retriever.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict
from config import Config
import os

class UltimateReranker:
    """终极稳定版Reranker - 彻底解决padding问题"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model_safely()
    
    def _load_model_safely(self):
        """安全加载模型"""
        try:
            print(f"🔄 加载Reranker模型从: {self.model_path}")
            
            # 方法1: 尝试直接加载
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"⚠️ 标准加载失败: {e}")
                # 方法2: 使用本地文件加载
                self._load_from_local_files()
            
            # 强制设置padding配置
            self._force_padding_config()
            
            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("✅ Reranker模型加载完成")
            
        except Exception as e:
            print(f"❌ 所有加载方法都失败: {e}")
            self.model = None
            self.tokenizer = None
    
    def _load_from_local_files(self):
        """从本地文件加载tokenizer"""
        try:
            # 检查必要的文件
            required_files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
            has_files = all(os.path.exists(os.path.join(self.model_path, f)) for f in required_files)
            
            if has_files:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                print("✅ 从本地文件加载tokenizer成功")
            else:
                raise FileNotFoundError("缺少必要的tokenizer文件")
                
        except Exception as e:
            print(f"❌ 本地文件加载失败: {e}")
            raise
    
    def _force_padding_config(self):
        """强制设置padding配置"""
        if self.tokenizer is None:
            return
            
        # 确保有pad_token
        if self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                # 添加新的pad_token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            print(f"✅ 设置pad_token为: {self.tokenizer.pad_token}")
        
        # 确保pad_token_id有效
        if self.tokenizer.pad_token_id is None:
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token_id = 0  # 默认值
        
        print(f"📋 最终配置 - pad_token: {self.tokenizer.pad_token}, pad_token_id: {self.tokenizer.pad_token_id}")
    
    def rerank_ultra_safe(self, query: str, documents: List[str]) -> List[Dict]:
        """超安全重排序 - 彻底避免批量问题"""
        if not documents or self.model is None or self.tokenizer is None:
            return self._create_default_results(documents)
        
        results = []
        
        for i, doc in enumerate(documents):
            try:
                score = self._score_single_pair(query, doc)
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
    
    def _score_single_pair(self, query: str, document: str) -> float:
        """为单个查询-文档对评分"""
        try:
            # 构建单个对
            text_pair = [query, document]
            
            # 使用最安全的编码方式
            inputs = self.tokenizer(
                text_pair,
                padding='max_length',      # 固定长度padding
                truncation=True,
                max_length=256,           # 较短的序列
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=True
            )
            
            # 手动检查并修复输入
            inputs = self._validate_and_fix_inputs(inputs)
            
            # 移动到设备
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=-1)
                score = scores[0, 1].item()  # 正例分数
            
            return score
            
        except Exception as e:
            print(f"❌ 评分失败: {e}")
            return 0.5
    
    def _validate_and_fix_inputs(self, inputs):
        """验证和修复输入"""
        # 确保attention_mask存在
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        # 确保token_type_ids存在（如果需要）
        if 'token_type_ids' not in inputs and hasattr(self.model.config, 'type_vocab_size'):
            seq_length = inputs['input_ids'].shape[1]
            inputs['token_type_ids'] = torch.zeros((1, seq_length), dtype=torch.long)
        
        return inputs
    
    def _create_default_results(self, documents: List[str]) -> List[Dict]:
        """创建默认结果"""
        return [{'document': doc, 'score': 0.5, 'rank': i} for i, doc in enumerate(documents)]

class UltimateRetriever:
    """终极检索器"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # 只有在路径有效时才初始化reranker
        if (config.RERANKER_MODEL_PATH and 
            os.path.exists(config.RERANKER_MODEL_PATH) and
            self._check_model_validity(config.RERANKER_MODEL_PATH)):
            
            self.reranker = UltimateReranker(config.RERANKER_MODEL_PATH)
            print("✅ 使用终极版检索器")
        else:
            self.reranker = None
            print("⚠️ 使用无Reranker的简化检索")
    
    def _check_model_validity(self, model_path: str) -> bool:
        """检查模型有效性"""
        try:
            # 简单检查是否存在必要的文件
            required = ['config.json', 'pytorch_model.bin', 'model.safetensors']
            has_required = any(os.path.exists(os.path.join(model_path, f)) for f in required)
            return has_required
        except:
            return False
    
    def retrieve(self, query: str, vector_store, top_k: int = 5) -> List[Dict]:
        """检索方法"""
        try:
            if vector_store is None:
                return []
                
            # 向量检索
            vector_results = vector_store.similarity_search(query, k=top_k)
            
            if not vector_results or self.reranker is None or self.reranker.model is None:
                return vector_results[:top_k]
            
            # 重排序
            print("🔄 使用终极版Reranker进行重排序...")
            documents = [result['content'] for result in vector_results]
            reranked = self.reranker.rerank_ultra_safe(query, documents)
            
            # 合并结果
            final_results = []
            for item in reranked[:top_k]:
                idx = item['rank']
                if idx < len(vector_results):
                    final_result = vector_results[idx].copy()
                    final_result['rerank_score'] = item['score']
                    final_result['final_score'] = item['score']
                    final_results.append(final_result)
            
            print(f"✅ 检索完成，返回 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            # 返回原始向量结果
            return vector_results[:top_k] if 'vector_results' in locals() else []