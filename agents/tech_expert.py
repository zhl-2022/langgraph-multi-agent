from .base_agent import BaseAgent
from typing import Dict, Any

class TechnicalExpertAgent(BaseAgent):
    def __init__(self):
        super().__init__("技术专家", "精通公司产品和技术细节")
        
    def process_task(self, task: str, context: dict) -> Dict[str, Any]:
        """技术专家处理任务"""
        rag_context = context.get('rag_context', [])
        rag_info = "\n".join([doc['content'] for doc in rag_context]) if rag_context else "暂无相关信息"
        
        prompt = f"""作为技术专家，你负责处理技术相关的需求。

用户需求: {task}
相关知识库信息: {rag_info}

请从技术角度提供专业分析，包括：
1. 技术可行性评估
2. 系统架构和解决方案设计
3. 技术栈选择建议
4. 开发周期和资源估算
5. 技术风险评估和应对措施
6. 是否需要项目经理制定详细计划

请提供详细的技术分析报告："""

        analysis = self.generate_response(prompt)
        
        return {
            'role': 'tech_expert', 
            'analysis': analysis,
            'recommendations': '基于技术分析的建议',
            'next_step': 'project_planning'
        }