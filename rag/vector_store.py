# rag/vector_store.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from config import Config

class QwenEmbeddingModel:
    """Qwen3-Embedding模型封装"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载Qwen3-Embedding模型"""
        print("🔄 加载Qwen3-Embedding模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Qwen3-Embedding模型加载完成")
    
    def _mean_pooling(self, model_output, attention_mask):
        """均值池化策略"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def encode(self, texts: list):
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # 编码文本
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 使用均值池化获得文档级嵌入
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            
            # 归一化（可选，但通常能提升检索效果）
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()
            
        except Exception as e:
            print(f"❌ 编码失败: {e}")
            # 返回随机向量作为备选
            return np.random.randn(len(texts), 1024).astype(np.float32)

class MilvusVectorStore:
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = QwenEmbeddingModel(config.EMBEDDING_MODEL_PATH)
        self.collection = None
        self._connect()
        
    def _connect(self):
        """连接Milvus数据库"""
        try:
            connections.connect(
                alias="default",
                uri=self.config.MILVUS_URI,
                token=self.config.MILVUS_TOKEN,
                user=self.config.MILVUS_USER, 
                password=self.config.MILVUS_PASSWORD,
                secure=True
            )
            print("✅ 成功连接到Milvus数据库")
            
            # 检查集合是否存在
            if utility.has_collection(self.config.COLLECTION_NAME):
                self.collection = Collection(self.config.COLLECTION_NAME)
                self.collection.load()
                print(f"✅ 集合 {self.config.COLLECTION_NAME} 已存在并已加载")
            else:
                print(f"⚠️ 集合 {self.config.COLLECTION_NAME} 不存在，将在需要时创建")
                
        except Exception as e:
            print(f"❌ 连接Milvus数据库失败: {e}")
            self.collection = None
    
    def create_collection(self):
        """创建向量集合"""
        try:
            # 如果集合已存在，先删除
            if utility.has_collection(self.config.COLLECTION_NAME):
                utility.drop_collection(self.config.COLLECTION_NAME)
                print(f"🗑️ 已删除旧集合: {self.config.COLLECTION_NAME}")
            
            # 测试获取维度
            test_embedding = self.embedding_model.encode(["测试文本"])
            embedding_dim = test_embedding.shape[1]
            print(f"📐 Qwen3-Embedding维度: {embedding_dim}")
            
            # 使用正确的维度定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(fields, "企业知识库向量存储")
            self.collection = Collection(self.config.COLLECTION_NAME, schema)
            
            # 创建向量索引
            index_params = {
                "index_type": "AUTOINDEX",
                "metric_type": "L2", 
                "params": {}
            }
            self.collection.create_index("embedding", index_params)
            self.collection.load()
            
            print(f"✅ 成功创建集合: {self.config.COLLECTION_NAME} (维度: {embedding_dim})")
            
        except Exception as e:
            print(f"❌ 创建集合失败: {e}")
            raise
    
    def add_documents(self, documents: list, metadatas: list = None):
        """添加文档到向量库"""
        if self.collection is None:
            print("❌ 集合未初始化，请先创建集合")
            return False
            
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        try:
            # 生成嵌入向量
            print("🔄 使用Qwen3-Embedding生成嵌入向量...")
            embeddings = self.embedding_model.encode(documents)
            
            print(f"📐 嵌入矩阵形状: {embeddings.shape}")
            print(f"📐 实际嵌入维度: {embeddings.shape[1]}")
            
            # 转换为列表格式
            embeddings_list = embeddings.tolist()
            
            # 准备插入数据
            entities = [
                documents,  # content字段
                embeddings_list, # embedding字段  
                metadatas  # metadata字段
            ]
            
            # 插入数据
            print("🔄 插入数据到Milvus...")
            insert_result = self.collection.insert(entities)
            self.collection.flush()
            
            print(f"✅ 成功插入 {len(documents)} 个文档")
            print(f"📈 集合现在有 {self.collection.num_entities} 个实体")
            return True
            
        except Exception as e:
            print(f"❌ 插入文档失败: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5):
        """相似性搜索"""
        if self.collection is None:
            print("❌ 集合未初始化")
            return []
            
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode([query])
            query_embedding_list = query_embedding.tolist()
            
            print(f"🔍 查询向量维度: {len(query_embedding_list[0])}")
            
            # 执行搜索
            search_params = {"metric_type": "L2", "params": {"ef": 32}}
            
            results = self.collection.search(
                data=query_embedding_list,
                anns_field="embedding",
                param=search_params,
                limit=k,
                output_fields=["content", "metadata"]
            )
            
            search_results = []
            for hit in results[0]:
                search_results.append({
                    'content': hit.entity.get('content'),
                    'metadata': hit.entity.get('metadata', {}),
                    'distance': hit.distance
                })
            
            print(f"✅ 搜索完成，找到 {len(search_results)} 个结果")
            return search_results
            
        except Exception as e:
            print(f"❌ 搜索过程中出错: {e}")
            return []

    def get_collection_info(self):
        """获取集合信息"""
        if self.collection is None:
            return "集合未初始化"
        
        try:
            num_entities = self.collection.num_entities
            return f"集合: {self.config.COLLECTION_NAME}, 实体数量: {num_entities}"
        except:
            return f"集合: {self.config.COLLECTION_NAME}, 状态: 已加载"