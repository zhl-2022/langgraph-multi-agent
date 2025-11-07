# fix_collection.py
from pymilvus import connections, utility
from config import Config

def fix_collection():
    """修复集合维度问题"""
    config = Config()
    
    print("🔧 修复集合维度问题")
    print("=" * 50)
    
    # 连接
    connections.connect(
        alias="default",
        uri=config.MILVUS_URI,
        token=config.MILVUS_TOKEN,
        user=config.MILVUS_USER, 
        password=config.MILVUS_PASSWORD,
        secure=True
    )
    
    # 删除错误的集合
    if utility.has_collection(config.COLLECTION_NAME):
        utility.drop_collection(config.COLLECTION_NAME)
        print(f"🗑️ 已删除错误的集合: {config.COLLECTION_NAME}")
    
    print("✅ 修复完成，现在可以重新运行初始化脚本")

if __name__ == "__main__":
    fix_collection()