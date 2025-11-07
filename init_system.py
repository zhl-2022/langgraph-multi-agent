# init_system.py
from config import Config
from rag.vector_store import MilvusVectorStore
import os

def initialize_system():
    """初始化系统"""
    config = Config()

    print("🚀 开始初始化系统...")
    vector_store = MilvusVectorStore(config)
    
    # 检查集合是否存在，如果不存在则创建
    from pymilvus import utility
    if not utility.has_collection(config.COLLECTION_NAME):
        print("📁 创建向量集合...")
        vector_store.create_collection()
    else:
        print("✅ 集合已存在，直接使用")
        vector_store.collection = vector_store.collection  # 确保collection已加载

    # 添加示例数据
    sample_documents = [
        "公司主要业务包括企业软件开发和AI解决方案",
        "我们的技术栈包括Python、Java、机器学习框架",
        "客户服务流程包括需求分析、方案设计、项目实施和售后支持",
        "产品包括智能客服系统、数据分析平台和自动化工具",
        "张汇浏是公司的CEO，负责公司的战略规划和业务发展",
        "ceo的邮箱是zhanghuiliu@example.com",
        "ceo的电话是13800138000", 
        "ceo的地址是北京市海淀区",
        "ceo的年龄是30岁",
        "ceo的性别是男",
        "ceo的学历是本科",
    ]

    print("📝 添加示例数据...")
    success = vector_store.add_documents(sample_documents)
    
    if success:
        print("✅ 系统初始化完成！")
        print(f"📊 已添加 {len(sample_documents)} 条数据")
    else:
        print("❌ 数据添加失败，但系统会继续运行")

def check_data():
    """检查数据是否已存在"""
    config = Config()
    vector_store = MilvusVectorStore(config)
    
    try:
        # 尝试搜索测试数据
        results = vector_store.similarity_search("CEO", k=3)
        if results:
            print(f"✅ 数据已存在，找到 {len(results)} 条相关记录")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['content'][:50]}...")
            return True
        else:
            print("❌ 未找到数据，需要初始化")
            return False
    except Exception as e:
        print(f"❌ 检查数据时出错: {e}")
        return False

if __name__ == "__main__":
    # 先检查数据是否已存在
    if not check_data():
        # 如果数据不存在，进行初始化
        initialize_system()
    else:
        print("🎉 数据已准备就绪，无需重复初始化")