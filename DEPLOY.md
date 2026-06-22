# llama.cpp 部署方案指南 —— 鲲鹏920 + Qwen3.6-35B-A3B

> 版本：2026-06 | llama.cpp b9692 | Qwen3.6-35B-A3B-UD-Q4_K_M | 鲲鹏920 + 128GB

---

## 一、部署方案总览

### 1.1 架构图

```
┌─────────────────────────────────────────────────────┐
│  你的客户端（任何机器）                                │
│  浏览器 / curl / Python / 应用程序                     │
│      │                                               │
│      │ OpenAI 兼容 API                                │
│      ▼                                               │
├─────────────────────────────────────────────────────┤
│  鲲鹏920 服务器                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │  systemd (qwen3-api.service)                    │ │
│  │     │ 自动重启、开机自启                           │ │
│  │     ▼                                           │ │
│  │  llama-server (端口 8080)                        │ │
│  │     │ ├── -t $(nproc)  多线程推理                  │ │
│  │     │ ├── -np 4        并发请求支持                │ │
│  │     │ └── -c 8192      上下文长度                  │ │
│  │     ▼                                           │ │
│  │  Qwen3.6-35B-A3B (UD-Q4_K_M)                    │ │
│  │  21GB GGUF 文件                                  │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 1.2 文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `llama.cpp-offline.tar.gz` | 33MB | llama.cpp 源码（已下载） |
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 21GB | 模型量化文件（UD-Q4_K_M，已下载） |

### 1.3 传输到服务器

```bash
# 小文件
scp /Users/tomhoyt/Desktop/modelDeploy/llama.cpp-offline.tar.gz user@SERVER:/home/user/

# 大文件（断点续传 + 进度条）
rsync -avP --progress \
  /Users/tomhoyt/Desktop/modelDeploy/qwen3-model/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  user@SERVER:/home/user/
```

---

## 二、编译

### 2.1 前置依赖

> **零 Python 依赖，纯 C++ 项目**。只需系统开发工具。

```bash
# 检查
cmake --version     # >= 3.14
gcc  --version      # >= 8，推荐 10+
g++  --version      # >= 8
make --version      # 任意版本

# 如果缺少
# CentOS/EulerOS/麒麟
yum install -y cmake gcc gcc-c++ make
# Debian/统信UOS
apt-get install -y cmake gcc g++ make
```

依赖对比（与传统 PyTorch 方案）：

| 依赖 | llama.cpp | PyTorch 方案 |
|------|-----------|-------------|
| cmake | ⭐ 必须 | ❌ 不需要 |
| gcc/g++ | ⭐ 必须 | ❌ 不需要 |
| Python | ❌ 不需要 | ⭐ 必须 |
| PyTorch | ❌ 不需要 | ⭐ 必须 |
| transformers | ❌ 不需要 | ⭐ 必须 |
| CUDA/cuDNN | ❌ 不需要 | ⭐ 必须 |
| pip 依赖数 | 0 | 30+ |
| 总占用 | 33MB | 3GB+ |

### 2.2 基础编译（推荐）

```bash
cd /home/user
tar xzf llama.cpp-offline.tar.gz
cd llama.cpp && mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU=ON \
  -DGGML_OPENMP=ON \
  -DGGML_NATIVE=ON               # 为当前 CPU 优化（鲲鹏920自动检测NEON）

cmake --build . --config Release -j$(nproc)
```

### 2.3 启用 OpenBLAS 加速（可选，+10~20%）

```bash
# 先安装 OpenBLAS
yum install -y openblas-devel              # CentOS/EulerOS/麒麟
# apt install -y libopenblas-dev            # Debian/统信

# 重新编译
rm -rf *
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU=ON \
  -DGGML_OPENMP=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_NATIVE=ON
cmake --build . --config Release -j$(nproc)
```

### 2.4 启用 KleidiAI 加速（推荐，+20~40%）

ARM 官方提供的 KleidiAI 库，对鲲鹏920等 ARM 架构有显著加速效果。**编译时自动从 GitHub 下载源码**，不需要预装。

```bash
rm -rf *
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU=ON \
  -DGGML_OPENMP=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CPU_KLEIDIAI=ON    # 启用 KleidiAI arm 优化内核
cmake --build . --config Release -j$(nproc)
```

> ⚠️ 注意：此选项需要服务器能访问 GitHub 下载 KleidiAI v1.24.0 源码。如果服务器无外网，需要在联网机器上下载后手动放入缓存：
> ```bash
> # 在联网机器上
> wget https://github.com/ARM-software/kleidiai/releases/download/v1.24.0/kleidiai-v1.24.0-src.tar.gz
> # 传输到服务器，再编译
> ```

### 2.5 编译选项完整矩阵

| 选项 | 默认值 | 鲲鹏920推荐 | 说明 |
|------|-------|-----------|------|
| `GGML_CPU` | ON | **ON** | CPU 后端 |
| `GGML_OPENMP` | ON | **ON** | 多线程并行 |
| `GGML_NATIVE` | ON | **ON** | 为当前 CPU 优化指令集 |
| `GGML_CPU_KLEIDIAI` | OFF | **ON** | KleidiAI ARM 优化内核 ⭐ |
| `GGML_BLAS` | OFF | **ON** | BLAS 加速（需安装 OpenBLAS） |
| `GGML_BLAS_VENDOR` | Generic | **OpenBLAS** | BLAS 实现选择 |
| `GGML_LLAMAFILE` | OFF | OFF | 仅 x86 有效 |
| `GGML_LTO` | OFF | OFF | 链接时优化（可尝试 ON） |
| `LLAMA_BUILD_SERVER` | ON | ON | 编译 API 服务 |
| `LLAMA_BUILD_UI` | ON | OFF | 嵌入 Web UI（不重要） |

### 2.6 验证编译结果

```bash
./bin/llama-cli --version
# 应输出类似: version: b9692 (f3e1828)
# 注意看启动日志中:
#   ggml_cpu_init: NEON = 1      <- SIMD 已启用
#   ggml_cpu_init: SVE = 0       <- 鲲鹏920不支持SVE，正常
```

---

## 三、运行

### 3.1 交互对话

```bash
cd /home/user/llama.cpp/build

./bin/llama-cli \
  -m /home/user/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -c 8192 \
  -t $(nproc) \
  --temp 0.7
```

输入 `/exit` 退出，输入 `/help` 查看所有命令。

### 3.2 API 服务

```bash
./bin/llama-server \
  -m /home/user/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 8192 \
  -t $(nproc) \
  -np 4
```

### 3.3 systemd 服务（生产环境）

```bash
sudo tee /etc/systemd/system/qwen3-api.service << 'SERVICEEOF'
[Unit]
Description=Qwen3.6 API Service (llama.cpp)
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/llama.cpp/build
ExecStart=/home/user/llama.cpp/build/bin/llama-server \
  -m /home/user/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 8192 \
  -t $(nproc) \
  -np 4
Restart=on-failure
RestartSec=10
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo systemctl daemon-reload
sudo systemctl enable qwen3-api
sudo systemctl start qwen3-api
sudo systemctl status qwen3-api
```

---

## 四、性能优化方向

### 4.1 编译维度

| 优化 | 预期提升 | 难度 | 说明 |
|------|---------|------|------|
| **KleidiAI** ⭐ | +20~40% | 🟡 需下载依赖 | ARM 官方优化矩阵内核，鲲鹏920最大收益点 |
| **OpenBLAS** | +10~20% | 🟢 易 | 通用 BLAS 加速矩阵运算 |
| **GGML_NATIVE** | +5~10% | 🟢 自动 | 默认已启用，按当前 CPU 微调指令选择 |
| **LTO** | +5~10% | 🟢 易 | 链接时优化，编译时间略长 |
| **-march=native** | +5~15% | 🟢 cmake 自动 | 编译器按 CPU 自动选择最佳指令集 |
| **静态链接** | 轻微 | 🟢 易 | 运行时减少动态链接开销 |

### 4.2 运行参数维度

| 参数 | 建议 | 原理 |
|------|------|------|
| `-t $(nproc)` | 全部物理核心 | 并行解码，核心越多越快 |
| `-c 8192~16384` | 128GB 可设 16384 | 上下文越长内存占用越高 |
| `-b 512` | batch size 越大越高 | 连续 token 批处理，显式指定可调优 |
| `--ubatch-size 512` | 与 batch 一致 | 内部微批处理大小 |
| `--mlock` | 建议启用 | 锁定内存防止被 swap 到磁盘 |
| `--no-mmap` | 备选 | 全程加载到内存而非 mmap，有提升 |
| `-np 1~8` | 按并发需求调 | 并发请求数，越多内存占用越高 |
| `--no-kv-offload` | CPU 默认 | KV cache 放内存，无需设置 |

**推荐运行命令（最优配置）：**

```bash
./bin/llama-server \
  -m /home/user/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 16384 \
  -t $(nproc) \
  -b 512 \
  --ubatch-size 512 \
  --mlock \
  -np 4
```

### 4.3 系统维度

| 优化 | 预期提升 | 操作 |
|------|---------|------|
| **关闭 NUMA balancing** | +10~30% | `echo 0 > /proc/sys/kernel/numa_balancing` |
| **CPU 隔离 + 绑核** | +10~20% | `taskset -c 0-63 ./bin/llama-server ...` |
| **实时调度优先级** | +5~10% | `chrt -f 99 ./bin/llama-server ...` |
| **禁用 CPU 频率缩放** | +5~15% | `cpupower frequency-set -g performance` |
| **大页内存 (HugeTLB)** | +5~10% | `echo 128 > /proc/sys/vm/nr_hugepages` |
| **透明大页** | +5% | `echo always > /sys/kernel/mm/transparent_hugepage/enabled` |
| **mlock（禁止 swap）** | +5~30% | `--mlock` 参数（取决于磁盘 swap 速度） |

**一键系统调优脚本：**

```bash
#!/bin/bash
# 在运行 llama-server 前执行
# sysctl-tune.sh

echo "=== 系统性能调优 ==="

# 1. 关闭 NUMA balancing
echo 0 > /proc/sys/kernel/numa_balancing
echo "NUMA balancing: OFF"

# 2. 禁止 swap 倾向
echo 0 > /proc/sys/vm/swappiness
echo "swappiness: 0"

# 3. 大页内存
echo 128 > /proc/sys/vm/nr_hugepages
echo "HugePages: 128"

# 4. 透明大页
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo "THP: always"

# 5. 性能 governor
cpupower frequency-set -g performance 2>/dev/null || true
echo "CPU governor: performance"

echo "=== 完成 ==="
```

### 4.4 模型量化维度

| 量化格式 | 大小 | 速度 | 质量 | 内存 |
|---------|------|------|------|------|
| **UD-Q4_K_M** ✅ 已下载 | 21GB | ⭐⭐⭐ 最快 | ⭐⭐⭐ 良好 | ~25GB |
| UD-Q5_K_M | 25GB | ⭐⭐ | ⭐⭐⭐⭐ 很好 | ~30GB |
| Q8_0 | 34GB | ⭐ | ⭐⭐⭐⭐⭐ 最好 | ~40GB |
| UD-IQ2_M | 11GB | ⭐⭐⭐⭐ 极快 | ⭐⭐ 一般 | ~15GB |
| MXFP4_MOE | 20GB | ⭐⭐⭐ | ⭐⭐⭐ 良好 | ~25GB |

> 对 MoE 模型来说，更高的量化级别（Q5_K_M / Q8_0）在推理时只解码激活的 expert（~3B），差距主要体现在模型质量和内存带宽占用上。

### 4.5 各优化层次预估收益

```
性能优化收益叠加（上限预估）：

基准: llama.cpp 默认编译 + 默认参数
   │
   ├─ +0%   ─── 基础编译 (NEON auto)
   │
   ├─ +15%  ─── + OpenBLAS 加速
   │
   ├─ +30%  ─── + KleidiAI 内核
   │
   ├─ +45%  ─── + 系统调优 (绑核 + NUMA + HugePages)
   │
   └─ +55%  ─── + 最优运行参数 (mlock + batch tuning)

理论综合：基准 ~5 tok/s → 优化后 ~8 tok/s
```

### 4.6 性能诊断工具

```bash
# 1. 查看 llama-server 启动时的硬件检测输出
./bin/llama-server --help 2>&1 | grep -i "neon\|sve\|arm\|cpu"

# 2. 查看运行时性能统计（发送 /stats 请求）
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}'

# 3. 查看 token 生成速度
# 运行 llama-cli 时会显示:  XX.XX tokens per second

# 4. 系统性能监控
top -d 1              # CPU 使用率
htop                  # 实时查看各核心负载
perf stat ./bin/llama-server ...  # 性能计数器

# 5. 内存带宽测试（瓶颈诊断）
yum install -y sysstat
mpstat -P ALL 1       # 各核心使用率
```

---

## 五、故障排查

### 5.1 编译失败

| 错误 | 原因 | 解决 |
|------|------|------|
| `CMake 3.14 or higher` | cmake 版本太旧 | 升级 cmake |
| `g++: command not found` | 未安装 C++ 编译器 | `yum install -y gcc-c++` |
| `internal compiler error` | GCC 版本太低 | 升级到 GCC 10+ |
| `NEON not detected` | 交叉编译环境问题 | 确认编译在鲲鹏本机执行 |
| `KleidiAI fetch failed` | 无法下载 | 手动下载后放缓存目录 |

### 5.2 运行失败

| 错误 | 原因 | 解决 |
|------|------|------|
| `GGUF version mismatch` | llama.cpp 与 GGUF 版本不兼容 | 更新 llama.cpp 到最新版 |
| `KQH ... out of memory` | 内存不足 | 减小 `-c` 上下文长度 |
| `Failed to load model` | 模型文件损坏 | 校验文件完整性 |
| `illegal instruction` | 编译了不支持的指令集 | 重建时去掉 `-DGGML_NATIVE=ON` |
| `Address already in use` | 端口占用 | 换端口或杀死旧进程 |

### 5.3 性能调优验证

```bash
# 验证 NEON 启用
./bin/llama-server ... 2>&1 | grep NEON
# 应输出: ggml_cpu_init: NEON = 1

# 验证 OpenBLAS 启用
./bin/llama-server ... 2>&1 | grep BLAS
# 应输出: BLAS = 1 | ll = 1

# 验证 KleidiAI 启用
./bin/llama-server ... 2>&1 | grep Kleidi
# 应输出: Using KleidiAI optimized kernels

# 验证线程数
htop  # 应看到 64/128 个核心满负载
```

---

## 六、客户端调用

### Python

```python
import requests

resp = requests.post("http://服务器IP:8080/v1/chat/completions", json={
    "model": "qwen3.6",
    "messages": [
        {"role": "system", "content": "你是Qwen3.6助手"},
        {"role": "user", "content": "你好"}
    ],
    "temperature": 0.7,
    "max_tokens": 2048
})

print(resp.json()["choices"][0]["message"]["content"])
```

### cURL

```bash
curl http://服务器IP:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"你好"}],
    "stream": true
  }'  # --stream 支持流式输出
```

### OpenAI SDK

```bash
pip install openai
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://服务器IP:8080/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="qwen3.6",
    messages=[{"role": "user", "content": "你好"}]
)
```

---

## 七、参考

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) — MIT 协议
- [Qwen3 官方文档](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)
- [KleidiAI (ARM)](https://github.com/ARM-software/kleidiai) — ARM I8MM 量化内核加速
- [llama.cpp 性能对比](https://github.com/ggml-org/llama.cpp/discussions/6730)
