from .base_agent import BaseAgent
from typing import Dict, Any

class BusinessExpertAgent(BaseAgent):
    def __init__(self):
        super().__init__("业务专家", "精通业务公司和客户行业背景")
        
    def process_task(self, task: str, context: dict) -> Dict[str, Any]:
        """业务专家处理任务"""
        rag_context = context.get('rag_context', [])
        rag_info = "\n".join([doc['content'] for doc in rag_context]) if rag_context else "暂无相关信息"
        
        prompt = f"""作为业务专家，你负责处理客户业务相关的需求。

用户需求: {task}
相关知识库信息: {rag_info}

请从业务角度提供专业分析，包括：
1. 客户行业背景和市场需求分析
2. 业务流程优化建议
3. 潜在的业务风险和机会
4. 具体的业务实施方案
5. 是否需要技术专家进一步分析

请提供详细的业务分析报告："""

        analysis = self.generate_response(prompt)
        
        # 判断是否需要技术专家介入
        needs_tech = any(keyword in task.lower() for keyword in ['技术', '系统', '开发', '实现', '平台'])
        
        return {
            'role': 'business_expert',
            'analysis': analysis,
            'recommendations': '基于业务分析的建议',
            'next_step': 'technical_review' if needs_tech else 'project_planning'
        }