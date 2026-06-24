# 实习生培养计划 —— ARM CPU LLM 部署方向

> 项目载体：`cpu_model_deploy` — Qwen3.6-35B-A3B 在鲲鹏920上的离线部署
> 周期：12 周 | 难度：L2→L4（能独立完成生产级部署 + 性能调优）

---

## 一、培养目标

### L2（第1-4周）：能跑通

- [ ] 理解 LLM 基础：Transformer、MoE、KV Cache 是什么
- [ ] 理解量化：为什么模型从 70GB 变成 21GB
- [ ] 能独立在 ARM 服务器上编译 llama.cpp
- [ ] 能部署成功，跑通一次对话

### L3（第5-8周）：能优化

- [ ] 理解 ggml 架构：算子注册、图构建、dispatch
- [ ] 能做性能 profiling 并指出瓶颈
- [ ] 能选择最优的编译参数和运行参数
- [ ] 能部署成 systemd 服务

### L4（第9-12周）：能改

- [ ] 理解 attention 算子的接入方式
- [ ] 能用 perf/mpirun/rocprof 等工具做细粒度性能分析
- [ ] 能写 benchmark 脚本做 A/B 对比
- [ ] 能做出优化决策并量化收益

---

## 二、阶段详细安排

### Phase 1：基础认知 — 搞清楚你在部署什么（Week 1-2）

#### Week 1：LLM 基础

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 读 [Transformer 论文](https://arxiv.org/abs/1706.03762)，理解自注意力机制 | 画出完整的 Transformer 结构图 |
| Day 2 | 理解 Q、K、V 是什么，注意力计算 `softmax(QK^T / √d) V` | 手写伪代码实现 attention |
| Day 3 | MoE（Mixture of Experts）：总参数 vs 激活参数的概念 | 解释 Qwen3.6-35B-A3B 的命名含义 |
| Day 4 | KV Cache 是什么、为什么需要、怎么存储 | 画出 KV Cache 的生存周期 |
| Day 5 | RoPE 位置编码、FFN、LayerNorm 概览 | 总结每个组件在推理中的角色 |

**交付物：** 写一篇 1500 字以上的技术笔记，用你自己的话解释 LLM 结构。发到团队文档空间。

#### Week 2：推理与量化

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 模型推理流程：Tokenization → Embedding → Layers → LM Head | 画流程图 |
| Day 2 | 量化原理：FP16 → INT4，精度损失 vs 速度提升 | 解释 GGUF Q4_K_M 含义 |
| Day 3 | MoE 量化特有问题：Expert Load Balance、激活 Expert 的选择 | 看 Qwen3 的 MOE 配置 |
| Day 4 | 对比几种部署框架：llama.cpp vs Ollama vs vLLM | 列出对比表格 |
| Day 5 | 阅读 `DEPLOY.md`，理解整个部署步骤 | 写一份"部署步骤流程图" |

**交付物：** 用你自己的机器（或 DevBox）跑通一个最小的 llama.cpp 推理（可以换小模型如 Qwen2.5-0.5B）。

---

### Phase 2：动手部署 — 让模型跑起来（Week 3-4）

#### Week 3：环境搭建与编译

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 登录鲲鹏服务器，检查环境（uname、cmake、gcc） | 提交环境检查报告 |
| Day 2 | 编译 llama.cpp：从源码到二进制 | 记录编译过程和遇到的问题 |
| Day 3 | 模型传输：scp/rsync 区别，大文件传输策略 | 测速并优化传输方案 |
| Day 4 | 完整运行一次 `llama-cli` 交互式对话 | 截图 + 记录 tokens/s |
| Day 5 | 理解 ggml 日志输出：NEON、BLAS、线程数各代表什么 | 写一篇文章解读启动日志 |

**交付物：** 部署跑通，记录基准性能数据。

#### Week 4：服务化部署

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 启动 `llama-server`，了解 OpenAI 兼容 API | 用 curl 成功调用一次 |
| Day 2 | 用 Python 调用 API，写一个简单的聊天客户端 | 可运行的 chat.py |
| Day 3 | 配置 systemd 服务、开机自启、日志查看 | systemd 配置文件和验证 |
| Day 4 | 学习性能监控：top、htop、nvidia-smi（如有 GPU） | 记录负载数据 |
| Day 5 | 复习前 4 周内容，写一篇 2000 字以上的实操总结 | 周报 + 总结文档 |

**交付物：** 跑通 API 服务 + Python 客户端 + systemd 管理。完成 L2 阶段考核。

---

### Phase 3：性能分析 — 找到你的瓶颈（Week 5-6）

#### Week 5：Profiling 工具

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | llama.cpp 自带的 `--perf` 标志：读每层耗时 | 解读 perf 输出，标出 top3 耗时层 |
| Day 2 | Linux `perf` 工具：`perf stat`、`perf record`、`perf report` | 热力图分析 |
| Day 3 | ARM 平台的 PMU 计数器：`perf stat -e cache-misses,cycles,instructions` | 找出 cache miss 瓶颈 |
| Day 4 | 内存带宽测试：`dd`、`stream` benchmark | 确认是否是内存带宽瓶颈 |
| Day 5 | 对比不同 `-t`（线程数）的性能曲线 | 画 "线程数 vs tokens/s" 曲线图 |

**交付物：** 一份性能分析报告，指出当前部署的 top3 瓶颈。

#### Week 6：编译优化实验

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 对比 `GGML_NATIVE=ON vs OFF` | 性能差异数据 |
| Day 2 | 安装 OpenBLAS 并启用 `GGML_BLAS=ON` | 测速对比 |
| Day 3 | 启用 `GGML_CPU_KLEIDIAI=ON`（ARM 优化内核） | 测速对比 |
| Day 4 | 组合最优编译选项 | 提交"最优编译配置" |
| Day 5 | LTO、静态链接等其他选项测试 | 完整对比表 |

**交付物：** 编译选项的 A/B 测试报告，给出最优编译方案。

---

### Phase 4：系统级优化（Week 7-8）

#### Week 7：系统参数调优

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | NUMA 架构理解 + `numactl` 命令 | 画出服务器的 NUMA 拓扑 |
| Day 2 | 关 NUMA balancing + 绑核实验 | 性能提升数据 |
| Day 3 | CPU governor + 频率 scaling 的影响 | 不同 governor 下的性能对比 |
| Day 4 | HugePages + THP 实验 | 测试是否受益 |
| Day 5 | `--mlock` 和 swap 倾向调优 | 内存压力测试 |

**交付物：** 系统性优化脚本 + 效果对比报告。

#### Week 8：运行参数调优

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | `-b` batch size 扫描（64/128/256/512/1024） | 找出最优 batch size |
| Day 2 | `-c` 上下文长度对性能的影响 | 性能-内存权衡曲线 |
| Day 3 | `-np` 并发请求的压力测试 | 并发 vs 延迟曲线 |
| Day 4 | 不同量化版本的性能对比 | Q4_K_M vs Q5_K_M vs Q8_0 对比 |
| Day 5 | 总结前三周优化成果，形成最佳实践文档 | "CPU推理性能优化最佳实践" v1.0 |

**交付物：** 一份完备的优化配置文档。完成 L3 阶段考核。

---

### Phase 5：深度学习 — 理解内核（Week 9-10）

#### Week 9：ggml 源码阅读

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | ggml 数据结构：`ggml_tensor`、`ggml_context`、`ggml_cgraph` | 画内存结构图 |
| Day 2 | 算子的声明（ggml.h）→ 创建（ggml_*）→ 计算（ggml_compute_forward_*）流程 | 画调用链路 |
| Day 3 | 重点看 `ggml_mul_mat`：矩阵乘法的分发策略 | 解释不同 size 走哪个内核 |
| Day 4 | 重点看 `ggml_flash_attn_ext`：fused attention 实现 | 读代码写注释 |
| Day 5 | 重点看 `ggml_soft_max_ext`：带 mask 的 softmax | 读代码写注释 |

**交付物：** 提交一份 ggml 核心数据流和算子体系的源码注释文档。

#### Week 10：Attention 接入实验

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 阅读 `build_attn_mha` 函数（`llama-graph.cpp:2066`） | 画出 attention 的图构建流程 |
| Day 2 | 理解 `ggml_permute` 对 Q/K/V 的 layout 变换 | 写一个小程序验证 layout |
| Day 3 | 理解 flash_attn 和 非 flash_attn 两条路径的区别 | 对比分析 |
| Day 4 | 框架层面修改：在 `build_attn_mha` 中加入自定义分支 | 提交代码 diff |
| Day 5 | benchmark：自定义分支 vs 原生路径的正确性与性能 | A/B 测试报告 |

**交付物：** 完成一次从 "读代码 → 改代码 → 验证性能" 的完整实验循环。

---

### Phase 6：综合实战与总结（Week 11-12）

#### Week 11：综合挑战

从以下题目中选一个做：

**A. 编写一个 benchmark 工具**
- 自动测试所有编译选项 × 运行参数组合
- 输出最优配置推荐
- 产出：可复用的 benchmark 脚本套件

**B. 实现一个自定义算子接入**
- 在 llama.cpp 中注册一个新的 ggml op
- 产出一个端到端的 "算子开发 → 注册 → 跑通" 的 mini 工程

**C. 生产环境监控方案**
- 为 API 服务配置 Prometheus + Grafana 监控
- 监控 tokens/s、延迟、内存、并发连接数
- 产出：docker-compose 环境 + 仪表盘 json

#### Week 12：总结与汇报

| 天数 | 内容 |
|------|------|
| Day 1-3 | 整理 12 周所有交付物 |
| Day 4-5 | 写最终技术报告（含所有实验数据、结果分析） |
| Day 6-7 | 做技术分享（25分钟演讲 + 5分钟 Q&A） |
| Day 8 | 成果展示 + 代码 Review |
| Day 9-10 | 填写转正面谈材料 |

**最终交付物：** 技术报告、代码贡献、Benchmark 数据仪表盘、转正申请。

---

## 三、考核标准

| 等级 | 标准 | 对应阶段 |
|------|------|---------|
| **L2 通过** | 能独立完成部署，理解每一步在做什么 | Phase 1-2 |
| **L3 通过** | 能独立做性能分析和优化，产出数据驱动的决策 | Phase 3-4 |
| **L4 通过** | 能修改框架代码，理解核心算子的实现原理 | Phase 5-6 |

### L2 考核（Week 4 末）

现场实操题（30分钟）：
1. 在鲲鹏服务器上从零编译 llama.cpp
2. 启动 API 服务
3. 用 curl 发送一次对话请求
4. 说出当前 tokens/s
5. 解释部署过程中的三个关键决策

### L3 考核（Week 8 末）

案例分析题：
给定一台新配置的 ARM 服务器，跑模型只有 2 tok/s。要求：
1. 用 perf/htop/llama-perf 定位问题
2. 给出优化方案

### L4 考核（Week 12 末）

技术改造题：
要求修改 llama.cpp 源码，在 attention 路径中加入一个自定义操作，验证正确性和性能。

---

## 四、资源清单

### 必读文档（本仓库）

| 文档 | 用途 |
|------|------|
| `DEPLOY.md` | 部署实操参考 |
| `download-model.sh` | 模型下载脚本 |
| `llama.cpp/` | 完整源码 |

### 推荐阅读

| 材料 | 优先级 |
|------|--------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | ⭐ 必读 |
| [llama.cpp README](https://github.com/ggerganov/llama.cpp) | ⭐ 必读 |
| [llama.cpp build options](https://github.com/ggerganov/llama.cpp#build) | ⭐ 必读 |
| [Qwen3 官方文档](https://qwen.readthedocs.io/en/latest/) | ⭐ 必读 |
| [GGML 源码导读](https://github.com/ggml-org/ggml) | ⭐ 必读 |
| [ARM NEON 编程指南](https://developer.arm.com/architectures/instruction-sets/intrinsics/) | 🟡 选读 |
| [KleidiAI 文档](https://github.com/ARM-software/kleidiai) | 🟡 选读 |
| [Linux perf 教程](https://perf.wiki.kernel.org/) | 🟡 选读 |
| [HugePages 说明](https://www.kernel.org/doc/html/latest/admin-guide/mm/hugetlbpage.html) | 🟢 参考 |

### 常用命令速查

```bash
# 性能
./bin/llama-cli -m model.gguf -p "test" -n 128 --perf
perf stat -e cache-misses,instructions,cycles ./bin/llama-cli ...
htop

# 系统调优
echo 0 > /proc/sys/kernel/numa_balancing
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 内存
free -h
cat /proc/meminfo | grep HugePages
```

---

## 五、文档模板

### 每周周报模板

```markdown
# Week X 周报 - 实习生姓名

## 本周完成
- [ ] 事项 1
- [ ] 事项 2

## 关键数据
- tokens/s: X.XX
- 瓶颈: ...

## 遇到的问题
- 问题 1 + 解决方案
- 问题 2 + 解决方案

## 下周计划
- [ ] 事项 1
- [ ] 事项 2
```

### 实验报告模板

```markdown
# 实验：编译选项对性能的影响

## 实验目的
...

## 实验环境
- 机器：鲲鹏920, 128GB
- 模型：Qwen3.6-35B-A3B-UD-Q4_K_M

## 控制变量
- ...

## 对比项
| 配置 | tokens/s | 首次延迟 | 内存占用 |
|------|---------|---------|---------|
| A | X | Yms | ZGB |
| B | X | Yms | ZGB |

## 结论
...

## 复现步骤
```bash
cmake ... && make ...
./bin/llama-cli ...
```
```

---

## 六、导师职责

- **每周 1 次 1on1（30 分钟）**：检查进度、解答问题
- **每两周 1 次代码 review**：验收代码质量
- **每次实验前要求写实验设计**：不允许"试一下看结果"
- **结果必须量化**：不说"变快了"，说"从 5.2tok/s 提升到 7.8tok/s，+50%"
- **鼓励问"为什么"**：每个答案都要追到源码级

---

## 七、常用术语表

| 术语 | 说明 |
|------|------|
| MoE | Mixture of Experts，混合专家模型，Qwen3.6-35B-A3B 用的架构 |
| GGUF | llama.cpp 的模型格式，单一文件，支持量化 |
| Q4_K_M | 4-bit 量化，K 表示关键层保留精度，M 表示中等质量 |
| KleidiAI | ARM 官方提供的机器学习加速库 |
| NEON | ARM 架构的 SIMD 指令集，鲲鹏920支持 |
| NUMA | Non-Uniform Memory Access，多路服务器的内存架构 |
| HugePages | 大页内存，减少 TLB miss |
| KV Cache | 缓存 attention 中的 Key/Value 矩阵 |
