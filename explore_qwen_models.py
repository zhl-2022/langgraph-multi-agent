# explore_qwen_models.py
from transformers import AutoModel, AutoTokenizer
import torch
from config import Config

def explore_qwen_embedding():
    """探索Qwen3-Embedding的正确API"""
    config = Config()
    
    print("🔍 探索Qwen3-Embedding API")
    print("=" * 50)
    
    try:
        # 加载模型
        tokenizer = AutoTokenizer.from_pretrained(
            config.EMBEDDING_MODEL_PATH,
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            config.EMBEDDING_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ 模型加载成功")
        print(f"📋 模型类型: {type(model)}")
        print(f"📋 模型类: {model.__class__}")
        
        # 检查模型的方法
        methods = [method for method in dir(model) if not method.startswith('_')]
        print(f"📋 模型方法: {methods[:10]}...")  # 只显示前10个方法
        
        # 测试文本
        texts = ["这是一个测试文本", "这是另一个测试文本"]
        
        # 尝试不同的编码方式
        print("\n🔍 尝试编码方式...")
        
        # 方式1: 直接调用模型
        try:
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            print("✅ 方式1成功 - 直接调用模型")
            print(f"📐 输出类型: {type(outputs)}")
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
                print(f"📐 嵌入形状: {embeddings.shape}")
        except Exception as e:
            print(f"❌ 方式1失败: {e}")
        
        # 方式2: 检查是否有encode方法
        try:
            if hasattr(model, 'encode'):
                embeddings = model.encode(texts)
                print("✅ 方式2成功 - 使用encode方法")
                print(f"📐 嵌入形状: {embeddings.shape}")
        except Exception as e:
            print(f"❌ 方式2失败: {e}")
            
        # 方式3: 检查是否有get_text_embeddings方法
        try:
            if hasattr(model, 'get_text_embeddings'):
                embeddings = model.get_text_embeddings(texts)
                print("✅ 方式3成功 - 使用get_text_embeddings方法")
                print(f"📐 嵌入形状: {embeddings.shape}")
        except Exception as e:
            print(f"❌ 方式3失败: {e}")
            
    except Exception as e:
        print(f"❌ 探索失败: {e}")

if __name__ == "__main__":
    explore_qwen_embedding()