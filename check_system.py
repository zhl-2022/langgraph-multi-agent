# check_system.py
from config import Config
from rag.vector_store import MilvusVectorStore
from rag.retriever import HybridRetriever
import sys

def check_system():
    """检查系统组件是否正常"""
    config = Config()
    
    print("🔍 系统组件检查")
    print("=" * 50)
    
    # 检查Milvus连接
    try:
        vector_store = MilvusVectorStore(config)
        print("✅ Milvus连接正常")
        
        # 测试搜索
        test_results = vector_store.similarity_search("测试", k=2)
        if test_results:
            print("✅ 向量搜索正常")
        else:
            print("⚠️ 向量搜索无结果（可能是集合为空）")
            
    except Exception as e:
        print(f"❌ Milvus检查失败: {e}")
        return False
    
    # 检查Reranker
    try:
        retriever = HybridRetriever(config)
        print("✅ Reranker加载正常")
        
        # 测试重排序（使用模拟数据）
        test_docs = ["这是一个测试文档", "这是另一个测试文档"]
        rerank_results = retriever.reranker.rerank("测试", test_docs)
        if rerank_results:
            print("✅ Reranker推理正常")
        else:
            print("⚠️ Reranker无结果")
            
    except Exception as e:
        print(f"❌ Reranker检查失败: {e}")
        print("⚠️ 将使用简化版检索器")
        return True  # 仍然可以继续，使用简化版
    
    print("🎉 所有组件检查完成")
    return True

if __name__ == "__main__":
    success = check_system()
    sys.exit(0 if success else 1)