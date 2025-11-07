# agents/llm_wrapper.py
from vllm import LLM, SamplingParams
from config import Config
import torch

class vLLMWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLM(
            model=config.LLM_MODEL_PATH,
            trust_remote_code=True,
            max_model_len=config.MAX_MODEL_LEN,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            quantization="AWQ",
            dtype=torch.float16
        )
        # 优化采样参数以获得更简洁的回答
        self.sampling_params = SamplingParams(
            temperature=0.1,           # 降低温度，减少随机性
            top_p=0.9,                 # 使用nucleus sampling
            top_k=50,                  # 限制候选token数量
            repetition_penalty=1.3,    # 增加重复惩罚
            max_tokens=200,            # 大幅限制生成长度
            skip_special_tokens=True,  # 跳过特殊token
            stop=[".", "。", "!", "！", "?", "？"]  # 添加停止词
        )

    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text
        # 后处理：清理回答
        return self._clean_response(response)
    
    def _clean_response(self, text: str) -> str:
        """清理回答，移除多余内容"""
        # 移除重复的句子
        sentences = text.split('。')
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if clean_sentence and clean_sentence not in seen_sentences:
                seen_sentences.add(clean_sentence)
                # 移除常见的啰嗦开头
                if not any(phrase in clean_sentence for phrase in [
                    "根据提供的信息", "基于以下信息", "根据上下文", 
                    "需要明确的是", "需要注意的是", "综上所述"
                ]):
                    unique_sentences.append(clean_sentence)
        
        cleaned = '。'.join(unique_sentences)
        if cleaned and not cleaned.endswith('。'):
            cleaned += '。'
        
        return cleaned

# 全局实例
_config = Config()
_llm_wrapper = vLLMWrapper(_config)

def get_llm() -> vLLMWrapper:
    return _llm_wrapper