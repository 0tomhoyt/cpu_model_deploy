# cpu_model_deploy

Qwen3.6-35B-A3B MoE 模型在**鲲鹏920（ARM CPU）**上的离线部署方案。

## 内容

| 文件 | 说明 |
|------|------|
| `DEPLOY.md` | 完整部署指南（编译、运行、性能优化） |
| `download-model.sh` | 模型下载脚本 |
| `llama.cpp/` | llama.cpp 源码（MIT 协议） |
| `llama.cpp-offline.tar.gz` | 源码打包（已下载） |

## 快速开始

```bash
# 1. 下载模型（需要联网）
bash download-model.sh

# 2. 编译
cd llama.cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=ON
cmake --build . -j$(nproc)

# 3. 运行
./bin/llama-server \
  -m ../../qwen3-model/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 -t $(nproc)
```

详细步骤见 [DEPLOY.md](./DEPLOY.md)。

## 依赖

- cmake >= 3.14
- gcc/g++ >= 8
- make
- **零 Python 依赖**
