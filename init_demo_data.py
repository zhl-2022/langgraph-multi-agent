# init_demo_data.py
from config import Config
from rag.vector_store import MilvusVectorStore
import time

def initialize_demo_data():
    """初始化演示数据"""
    config = Config()
    
    print("=" * 50)
    print("🚀 开始初始化演示数据")
    print("=" * 50)
    
    # 创建演示文档
    demo_documents = [
        "公司主要业务包括企业软件定制开发和人工智能解决方案实施",
        "我们的核心技术栈包括Python、Java、Spring Boot、Vue.js和各类机器学习框架",
        "典型客户服务流程包含需求调研、方案设计、开发实施、测试验收和售后支持五个阶段",
        "智能客服系统支持多渠道接入，具备自然语言处理和情感分析能力",
        "数据分析平台可以处理结构化与非结构化数据，提供可视化报表和预测分析",
        "项目管理遵循敏捷开发原则，采用Scrum方法论进行迭代开发",
        "公司已通过ISO9001质量体系认证，拥有完善的软件开发流程规范",
        "我们的客户主要集中在金融、零售、制造和医疗健康行业",
        "团队拥有超过50名技术人员，包括前端、后端、算法和运维工程师",
        "我们提供从咨询、设计、开发到部署运维的全生命周期服务"
    ]
    
    print("📋 演示文档准备完成")
    
    vector_store = MilvusVectorStore(config)
    
    try:
        # 检查当前集合状态
        from pymilvus import utility
        collections = utility.list_collections()
        print(f"📊 现有集合: {collections}")
        
        # 创建集合（会自动检测维度）
        print("🔄 创建集合...")
        vector_store.create_collection()
        time.sleep(2)  # 等待集合创建完成
        
        # 添加数据
        print("📝 添加演示数据...")
        success = vector_store.add_documents(demo_documents)
        
        if success:
            print("🎉 演示数据初始化完成！")
            print(f"📚 已添加 {len(demo_documents)} 个知识文档")
            
            # 测试搜索功能
            print("\n🧪 测试搜索功能...")
            test_results = vector_store.similarity_search("人工智能解决方案", k=3)
            if test_results:
                print("✅ 搜索测试成功！")
                for i, result in enumerate(test_results):
                    print(f"  {i+1}. {result['content'][:50]}... (距离: {result['distance']:.4f})")
            else:
                print("⚠️ 搜索测试无结果")
                
        else:
            print("❌ 数据添加失败")
            
    except Exception as e:
        print(f"❌ 初始化演示数据时出错: {e}")
        print("⚠️ 系统将继续运行，但RAG功能可能受限")

if __name__ == "__main__":
    initialize_demo_data()