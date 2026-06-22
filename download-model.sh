#!/bin/bash
# 下载模型文件
# 用法: ./download-model.sh [量化版本]

set -e

MODEL="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
REPO="unsloth/Qwen3.6-35B-A3B-GGUF"

echo "==> 下载模型: $MODEL"
echo "    仓库: $REPO"
echo ""

# 尝试 huggingface-cli
if command -v huggingface-cli &> /dev/null; then
    echo "使用 huggingface-cli..."
    huggingface-cli download "$REPO" "$MODEL" --local-dir ./qwen3-model --resume-download
    echo ""
    echo "✅ 下载完成: $(ls -lh qwen3-model/$MODEL | awk '{print $5}')"
    exit 0
fi

# 如果没装 huggingface-cli，试 curl 直链
echo "huggingface-cli 未安装，尝试 curl..."
echo "请先安装: pip install huggingface_hub"
echo "或手动从 https://huggingface.co/$REPO 下载 $MODEL"
exit 1
