# -*- coding: utf-8 -*-
"""
File:       server.py
Time:       2025-11-06-22:04
User:       zhl
Details:       
"""
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.tools.business_tools import BusinessTools

app = Server("business-tools")

# 初始化工具实例
business_tools = BusinessTools()

@app.tool()
async def get_customer_info(customer_id: str):
    """根据客户ID获取详细的客户信息和历史记录"""
    return await business_tools.get_customer_info(customer_id)

@app.tool()
async def check_inventory(product_id: str):
    """根据产品ID检查实时库存情况"""
    return await business_tools.check_inventory(product_id)

@app.tool()
async def create_project_task(task_name: str, description: str, assignee: str, deadline: str):
    """创建新的项目任务并分配给指定成员"""
    project_data = {
        'name': task_name,
        'description': description,
        'assignee': assignee,
        'deadline': deadline
    }
    return await business_tools.create_project_task(project_data)

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())