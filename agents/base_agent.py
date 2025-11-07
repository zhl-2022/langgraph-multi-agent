from abc import ABC, abstractmethod
from typing import Dict, Any
from .llm_wrapper import get_llm
from rag.retriever import HybridRetriever
from config import Config

class BaseAgent(ABC):
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.llm = get_llm()
    
    def generate_response(self, prompt: str) -> str:
        """使用vLLM生成响应"""
        # 构建更清晰的系统提示
        system_prompt = f"你是一名{self.role}，名叫{self.name}。请基于你的专业领域提供详细、准确的分析和建议。\n\n"
        full_prompt = system_prompt + f"任务：{prompt}\n\n请提供你的专业分析："
        return self.llm.generate(full_prompt)
    
    @abstractmethod
    def process_task(self, task: str, context: dict) -> dict:
        """处理任务抽象方法"""
        pass