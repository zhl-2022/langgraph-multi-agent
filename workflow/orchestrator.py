from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
from agents.coordinator import CoordinatorAgent
from agents.business_expert import BusinessExpertAgent  
from agents.tech_expert import TechnicalExpertAgent
from agents.project_manager import ProjectManagerAgent
from rag.retriever import HybridRetriever
from rag.vector_store import MilvusVectorStore
from config import Config
from rag.simple_retriever import SimpleRetriever
class AgentState(TypedDict):
    task: str
    current_agent: str
    context: Dict[str, Any]
    results: Dict[str, Any]
    next_step: str

class WorkflowOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.coordinator = CoordinatorAgent()
        self.business_expert = BusinessExpertAgent()
        self.tech_expert = TechnicalExpertAgent() 
        self.project_manager = ProjectManagerAgent()
        
        # 尝试使用完整检索器，如果失败则使用简化版
        try:
            from rag.retriever import HybridRetriever
            self.retriever = HybridRetriever(config)
            print("✅ 使用完整检索器（包含Reranker）")
        except Exception as e:
            print(f"⚠️ 完整检索器初始化失败: {e}，使用简化版")
            self.retriever = SimpleRetriever(config)
            
        self.vector_store = MilvusVectorStore(config)
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """构建LangGraph工作流"""
        graph = StateGraph(AgentState)
        
        # 添加节点
        graph.add_node("coordinator", self._run_coordinator)
        graph.add_node("business_expert", self._run_business_expert)
        graph.add_node("tech_expert", self._run_tech_expert)
        graph.add_node("project_manager", self._run_project_manager)
        
        # 设置入口点
        graph.set_entry_point("coordinator")
        
        # 定义条件路由
        graph.add_conditional_edges(
            "coordinator",
            self._route_from_coordinator,
            {
                "business_expert": "business_expert",
                "tech_expert": "tech_expert", 
                "project_manager": "project_manager"
            }
        )
        
        graph.add_conditional_edges(
            "business_expert",
            self._route_from_business,
            {
                "technical_review": "tech_expert",
                "project_planning": "project_manager",
                "end": END
            }
        )
        
        graph.add_conditional_edges(
            "tech_expert", 
            self._route_from_tech,
            {
                "project_planning": "project_manager",
                "end": END
            }
        )
        
        graph.add_edge("project_manager", END)
        
        return graph.compile()
    
    def _run_coordinator(self, state: AgentState) -> AgentState:
        """运行协调员Agent"""
        task = state["task"]
        
        # 初始化results字典
        if "results" not in state:
            state["results"] = {}
            
        # RAG检索
        rag_context = self.retriever.retrieve(task, self.vector_store)
        
        context = {
            "rag_context": rag_context,
            "previous_results": state.get("results", {})
        }
        
        result = self.coordinator.process_task(task, context)
        state["results"]["coordinator"] = result
        state["current_agent"] = "coordinator"
        state["context"] = context
        
        return state
        
    def _run_business_expert(self, state: AgentState) -> AgentState:
        """运行业务专家Agent"""
        task = state["task"]
        context = state["context"]
        
        result = self.business_expert.process_task(task, context)
        state["results"]["business_expert"] = result
        state["current_agent"] = "business_expert"
        state["next_step"] = result.get("next_step", "end")
        
        return state
        
    def _run_tech_expert(self, state: AgentState) -> AgentState:
        """运行技术专家Agent""" 
        task = state["task"]
        context = state["context"]
        
        result = self.tech_expert.process_task(task, context)
        state["results"]["tech_expert"] = result
        state["current_agent"] = "tech_expert"
        state["next_step"] = result.get("next_step", "end")
        
        return state
        
    def _run_project_manager(self, state: AgentState) -> AgentState:
        """运行项目经理Agent"""
        task = state["task"]
        context = state["context"]
        
        result = self.project_manager.process_task(task, context)
        state["results"]["project_manager"] = result
        state["current_agent"] = "project_manager"
        
        return state
        
    def _route_from_coordinator(self, state: AgentState) -> str:
        """从协调员路由"""
        result = state["results"]["coordinator"]
        return result["next_agent"]
        
    def _route_from_business(self, state: AgentState) -> str:
        """从业务专家路由"""
        return state.get("next_step", "end")
        
    def _route_from_tech(self, state: AgentState) -> str:
        """从技术专家路由"""
        return state.get("next_step", "end")
        
    def execute_workflow(self, task: str) -> Dict[str, Any]:
        """执行工作流"""
        initial_state = AgentState(
            task=task,
            current_agent="",
            context={},
            results={},
            next_step=""
        )
        
        final_state = self.graph.invoke(initial_state)
        return {
            "task": final_state["task"],
            "results": final_state["results"],
            "final_agent": final_state["current_agent"]
        }