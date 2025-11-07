# test_milvus_connection.py
from pymilvus import connections, utility

def test_connection():
    try:
        # 测试连接
        connections.connect(
            alias="default",
            uri="https://in03-f2c7c2ce6a1bdff.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
            token="85afa5d1098a63edeefad6c703e22e1bf9d8aa6cacdad29adf5fe13e82df4b4d647d646530fe098f79f0dc70da223673855f2b3d",
            user="db_f2c7c2ce6a1bdff", 
            password="Ga7(rHn0xu(Uud",
            secure=True
        )
        print("✅ 连接成功！")
        
        # 列出所有集合
        collections = utility.list_collections()
        print(f"现有集合: {collections}")
        
        return True
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

if __name__ == "__main__":
    test_connection()