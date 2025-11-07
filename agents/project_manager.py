from .base_agent import BaseAgent
from typing import Dict, Any

class ProjectManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__("项目经理", "负责内部流程推进")
        
    def process_task(self, task: str, context: dict) -> Dict[str, Any]:
        """项目经理处理任务"""
        rag_context = context.get('rag_context', [])
        rag_info = "\n".join([doc['content'] for doc in rag_context]) if rag_context else "暂无相关信息"
        
        prompt = f"""作为项目经理，你负责项目规划和执行。

用户需求: {task}
相关知识库信息: {rag_info}
之前的分析结果: {context.get('previous_results', {})}

请制定详细的项目计划，包括：
1. 项目目标和关键成果
2. 时间线和里程碑设置
3. 资源分配和团队组建
4. 风险评估和应对策略
5. 预算和成本估算
6. 质量保证措施

请提供完整的项目执行计划："""

        analysis = self.generate_response(prompt)
        
        return {
            'role': 'project_manager',
            'analysis': analysis,
            'plan': '详细的项目执行计划',
            'status': 'ready_for_execution'
        }