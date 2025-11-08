#!/bin/bash

# 你可以在这里修改模型的存储路径
MODEL_DIR="./models"

echo "📥 开始通过 ModelScope 下载模型文件到: $MODEL_DIR"

# 创建模型存储目录
mkdir -p $MODEL_DIR/Qwen $MODEL_DIR/BAAI

# 使用 ModelScope 下载 Qwen 系列模型
echo "⬇️  开始下载 Qwen2.5-3B-Instruct-AWQ..."
modelscope download --model Qwen/Qwen2.5-3B-Instruct-AWQ --local_dir $MODEL_DIR/Qwen/Qwen2.5-3B-Instruct-AWQ --revision master

echo "⬇️  开始下载 Qwen3-Embedding-0.6B..."
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir $MODEL_DIR/Qwen/Qwen3-Embedding-0.6B --revision master

# 使用 ModelScope 下载 BGE 模型
echo "⬇️  开始下载 BGE-Reranker-large..."
modelscope download --model BAAI/bge-reranker-large --local_dir $MODEL_DIR/BAAI/bge-reranker-large --revision master

echo "✅ 所有模型下载完成！"
