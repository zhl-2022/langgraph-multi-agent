# services/quick_response.py
from rag.vector_store import MilvusVectorStore
from rag.bge_retriever import BGERetriever
from rag.simple_retriever import SimpleRetriever
from agents.llm_wrapper import get_llm
from config import Config

class QuickResponseService:
    """快速响应服务 - 使用BGE检索器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = MilvusVectorStore(config)
        self.llm = get_llm()
        
        # 优先使用BGE检索器
        try:
            self.retriever = BGERetriever(config)
            print("✅ 使用BGE检索器")
        except Exception as e:
            print(f"⚠️ BGE检索器失败，使用简化版: {e}")
            self.retriever = SimpleRetriever(config)
    
    def generate_quick_response(self, query: str) -> dict:
        """生成快速响应"""
        try:
            # 检索相关知识
            if hasattr(self.retriever, 'retrieve'):
                rag_results = self.retriever.retrieve(query, self.vector_store, top_k=5, rerank_k=3)
            else:
                # 简化检索器
                rag_results = self.retriever.retrieve(query, self.vector_store, top_k=3)
            
            # 构建上下文
            context = self._build_context(rag_results)
            
            # 生成回答
            response = self._generate_answer(query, context)
            
            return {
                'type': 'quick_response',
                'answer': response,
                'sources': rag_results,
                'has_related_info': len(rag_results) > 0
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'answer': f"抱歉，处理问题时出现错误: {str(e)}",
                'sources': [],
                'has_related_info': False
            }
    
    def _build_context(self, rag_results: list) -> str:
        """构建上下文"""
        if not rag_results:
            return "没有找到相关信息。"
        
        context_parts = ["参考信息："]
        for i, result in enumerate(rag_results, 1):
            content = result['content'][:150]
            score = result.get('rerank_score', result.get('final_score', 0.5))
            context_parts.append(f"{i}. {content} (相关度: {score:.3f})")
        
        return "\n".join(context_parts)
    
    # services/quick_response.py
    # 在_generate_answer方法中大幅优化提示词

    def _generate_answer(self, query: str, context: str) -> str:
        """生成回答 - 强制简洁版本"""
        
        # 根据问题类型使用不同的提示词
        if self._is_fact_query(query):
            prompt = self._build_fact_prompt(query, context)
        else:
            prompt = self._build_general_prompt(query, context)
        
        response = self.llm.generate(prompt)
        return self._post_process_response(response)

    def _is_fact_query(self, query: str) -> bool:
        """判断是否是事实性查询"""
        fact_keywords = ['是谁', '是什么', '多少', '哪里', '什么时候', '电话', '邮箱', '地址']
        return any(keyword in query for keyword in fact_keywords)

    def _build_fact_prompt(self, query: str, context: str) -> str:
        """构建事实性查询的提示词"""
        return f"""请直接回答以下问题，只给出事实信息，不要解释和分析。

    参考信息：
    {context}

    问题：{query}

    要求：
    1. 直接给出答案，不要开头语
    2. 只包含问题相关的具体信息
    3. 如果信息充分，直接回答
    4. 如果信息不足，只说"信息不足"
    5. 回答要简洁，不超过2句话

    回答："""

    def _build_general_prompt(self, query: str, context: str) -> str:
        """构建一般性查询的提示词"""
        return f"""请基于参考信息简洁回答以下问题。

    参考信息：
    {context}

    问题：{query}

    要求：
    1. 直接回答问题核心
    2. 回答要简洁明了
    3. 不要重复信息
    4. 不要添加解释性内容
    5. 如果信息不足请说明

    回答："""

    def _post_process_response(self, text: str) -> str:
        """后处理响应 - 强制简洁"""
        # 移除常见的啰嗦开头
        text = self._remove_verbose_starts(text)
        
        # 限制回答长度
        sentences = text.split('。')
        if len(sentences) > 2:
            text = '。'.join(sentences[:2]) + '。'
        
        # 移除多余的空格和换行
        text = ' '.join(text.split())
        
        return text

    def _remove_verbose_starts(self, text: str) -> str:
        """移除啰嗦的开头"""
        verbose_starts = [
            "根据提供的信息，",
            "基于参考信息，", 
            "根据上下文，",
            "需要明确的是，",
            "需要注意的是，",
            "综上所述，",
            "通过分析，",
            "根据以上信息，"
        ]
        
        for start in verbose_starts:
            if text.startswith(start):
                text = text[len(start):]
                break
        
        return text