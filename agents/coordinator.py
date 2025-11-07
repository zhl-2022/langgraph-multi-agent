from .base_agent import BaseAgent
from typing import Dict, Any

class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("协调员", "总指挥和任务分配")
        
    def process_task(self, task: str, context: dict) -> Dict[str, Any]:
        """协调员处理任务 - 分析和分配"""
        prompt = f"""作为协调员，你需要分析用户需求并决定如何分配任务。

用户需求: {task}
当前上下文: {context.get('rag_context', [])}

请分析这个需求涉及哪些方面，并决定需要哪些专家参与。可能的参与方包括：
- 业务专家：处理客户关系、行业知识、业务流程
- 技术专家：处理产品技术细节、解决方案设计  
- 项目经理：协调资源、制定计划、推进执行

请给出：
1. 需求分析
2. 需要参与的专家类型
3. 下一步行动建议"""

        analysis = self.generate_response(prompt)
        
        # 决策逻辑
        task_lower = task.lower()
        if any(keyword in task_lower for keyword in ['客户', '业务', '行业', '市场', '销售']):
            next_agent = "business_expert"
        elif any(keyword in task_lower for keyword in ['技术', '产品', '实现', '开发', '代码']):
            next_agent = "tech_expert" 
        elif any(keyword in task_lower for keyword in ['项目', '计划', '时间', '资源', '管理']):
            next_agent = "project_manager"
        else:
            next_agent = "business_expert"  # 默认
            
        return {
            'analysis': analysis,
            'next_agent': next_agent,
            'recommendations': f"建议由{next_agent}处理此任务"
        }