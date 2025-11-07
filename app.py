# app.py
import streamlit as st
import asyncio
import json
from workflow.orchestrator import WorkflowOrchestrator
from services.quick_response import QuickResponseService
from config import Config

# 设置页面
st.set_page_config(
    page_title="多Agent协同任务系统",
    page_icon="🤖",
    layout="wide"
)

# 初始化全局组件
# app.py
# 在init_system函数中替换为简洁服务

@st.cache_resource
def init_system():
    config = Config()
    
    try:
        # 使用简洁回答服务
        from services.concise_response import ConciseResponseService
        quick_service = ConciseResponseService(config)
        
        orchestrator = None  # 延迟初始化
        
        return {
            'quick_service': quick_service,
            'orchestrator': orchestrator,
            'config': config
        }
        
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        return None
        
def init_full_workflow():
    """按需初始化完整工作流"""
    if 'full_system' not in st.session_state:
        with st.spinner("🔄 正在加载完整Agent系统..."):
            config = Config()
            st.session_state.full_system = WorkflowOrchestrator(config)
    return st.session_state.full_system

def main():
    st.title("🤖 基于LangGraph的多Agent协同任务系统")
    st.markdown("""
    **智能任务处理中枢** - 提供两种处理模式：
    - 🚀 **快速响应**：基于知识库直接回答
    - 🔍 **深度分析**：启动多Agent团队进行详细分析
    """)
    
    # 初始化系统
    with st.spinner("正在初始化系统组件，请稍候..."):
        system = init_system()
    
    # 任务输入区
    st.subheader("📝 任务输入")
    task_input = st.text_area(
        "请详细描述您的任务需求：",
        placeholder="例如：我们需要为一家零售企业设计智能客服解决方案，请分析业务需求并提供技术实施方案...",
        height=100
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 快速响应", type="primary", use_container_width=True):
            if not task_input.strip():
                st.error("请输入任务描述！")
                return
                
            process_quick_response(system['quick_service'], task_input)
    
    with col2:
        if st.button("🔍 深度分析", type="secondary", use_container_width=True):
            if not task_input.strip():
                st.error("请输入任务描述！")
                return
                
            process_deep_analysis(system, task_input)

def process_quick_response(quick_service, task: str):
    """处理快速响应"""
    with st.spinner("🔍 正在检索知识库并生成回答..."):
        try:
            result = quick_service.generate_quick_response(task)
            
            if result['type'] == 'quick_response':
                st.success("✅ 快速响应完成！")
                
                # 显示回答
                st.subheader("💬 智能回答")
                st.write(result['answer'])
                
                # 显示来源
                if result['sources']:
                    with st.expander("📚 参考来源", expanded=False):
                        for i, source in enumerate(result['sources']):
                            st.write(f"**来源 {i+1}** (相关度: {1-source.get('distance', 0):.3f})")
                            st.info(source['content'])
                            
                    # 如果有相关信息，建议深度分析
                    if result['has_related_info']:
                        st.info("💡 检测到相关信息，点击'深度分析'按钮可获得更详细的分析报告")
                else:
                    st.warning("⚠️ 未找到相关背景信息，回答基于模型的一般知识")
                    
            else:
                st.error(result['answer'])
                
        except Exception as e:
            st.error(f"快速响应处理失败: {str(e)}")

def process_deep_analysis(system, task: str):
    """处理深度分析"""
    try:
        # 按需初始化完整工作流
        orchestrator = init_full_workflow()
        
        with st.spinner("🤖 智能团队正在协同分析，请耐心等待..."):
            result = orchestrator.execute_workflow(task)
            display_deep_analysis_results(result, task)
            
    except Exception as e:
        st.error(f"深度分析处理失败: {str(e)}")

def display_deep_analysis_results(result: dict, original_task: str):
    """展示深度分析结果"""
    st.success("🎉 深度分析完成！")
    
    results = result["results"]
    
    # 显示原始任务
    with st.expander("📋 原始任务描述", expanded=False):
        st.write(original_task)
    
    # 协调员分析
    if "coordinator" in results:
        st.subheader("🎯 任务分析与分配")
        coord_result = results["coordinator"]
        st.info(coord_result["analysis"])
        st.metric("执行专家", coord_result["next_agent"].replace("_", " ").title())
    
    # 业务专家分析
    if "business_expert" in results:
        st.subheader("💼 业务专家分析")
        biz_result = results["business_expert"]
        st.write(biz_result["analysis"])
    
    # 技术专家分析
    if "tech_expert" in results:
        st.subheader("🔧 技术专家分析")
        tech_result = results["tech_expert"]
        st.write(tech_result["analysis"])
    
    # 项目经理计划
    if "project_manager" in results:
        st.subheader("📅 项目执行计划")
        pm_result = results["project_manager"]
        st.write(pm_result["analysis"])
    
    # 显示完整的处理流水
    st.subheader("🔄 任务处理流水线")
    flow_data = []
    for agent_name, agent_result in results.items():
        flow_data.append({
            "处理节点": agent_name.replace("_", " ").title(),
            "角色": get_agent_role(agent_name),
            "状态": "✅ 已完成"
        })
    
    st.table(flow_data)

def get_agent_role(agent_name: str) -> str:
    """获取Agent角色描述"""
    roles = {
        "coordinator": "总指挥与任务分配",
        "business_expert": "客户业务与行业分析",
        "tech_expert": "技术方案与产品细节",
        "project_manager": "项目规划与执行推进"
    }
    return roles.get(agent_name, "专业处理")

if __name__ == "__main__":
    main()