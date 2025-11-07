# -*- coding: utf-8 -*-
"""
File:       main.py
Time:       2025-11-06-20:42
User:       zhl
Details:       
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from workflow.orchestrator import WorkflowOrchestrator
from config import Config
import uvicorn

app = FastAPI(title="多Agent协同任务系统", version="1.0.0")


class TaskRequest(BaseModel):
    task: str
    user_id: str = "default"


class TaskResponse(BaseModel):
    task_id: str
    status: str
    results: dict
    final_output: str


# 全局实例
config = Config()
orchestrator = WorkflowOrchestrator(config)


@app.post("/api/task", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    """创建并执行新任务"""
    try:
        result = orchestrator.execute_workflow(request.task)

        # 生成最终输出
        final_output = await generate_final_output(result)

        return TaskResponse(
            task_id=f"task_{hash(request.task)}",
            status="completed",
            results=result["results"],
            final_output=final_output
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "multi-agent-system"}


async def generate_final_output(result: dict) -> str:
    """生成最终输出"""
    results = result["results"]
    output_parts = []

    if "coordinator" in results:
        output_parts.append(f"## 任务分析\n{results['coordinator']['analysis']}")

    if "business_expert" in results:
        output_parts.append(f"## 业务分析\n{results['business_expert']['analysis']}")

    if "tech_expert" in results:
        output_parts.append(f"## 技术分析\n{results['tech_expert']['analysis']}")

    if "project_manager" in results:
        output_parts.append(f"## 项目计划\n{results['project_manager']['analysis']}")

    return "\n\n".join(output_parts)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)