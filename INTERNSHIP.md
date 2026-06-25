# 实习生培养计划 —— ARM CPU LLM 部署与推理调优

> 项目载体：`cpu_model_deploy` — Qwen3.6-35B-A3B 在**鲲鹏920F**上的推理部署与调优
> 周期：12 周 | 难度：L2→L4

---

## 一、三大培养目标

### 🎯 目标 1：能在鲲鹏 920F 上跑通大模型

从零开始，在**鲲鹏 920F**（华为 7nm ARM 服务器芯片）上完成大模型的部署：

- 理解 920F 的架构特点（64核 ARMv8.2、NEON SIMD、DDR4 内存通道拓扑）
- 编译 llama.cpp，加载量化模型，跑通对话
- 部署为稳定的 API 服务

> 920F 是 920 系列的增强版本，主频更高（2.6GHz vs 2.0GHz），支持 DDR4-2933。推理瓶颈在内存带宽，920F 的 8 通道 DDR4 配置是关键约束。

### 🎯 目标 2：融合 Unigemm（鲲鹏数学库）做推理调优

不只是跑通，而是**把华为官方的高性能计算库用上**：

- 了解 **KML（Kunpeng Math Library）**，特别是其中的 **Unigemm**——鲲鹏上最优化过的通用矩阵乘法库
- 将 llama.cpp/ggml 的默认矩阵乘法（`ggml_mul_mat`）替换或对齐到 Unigemm 内核
- 量化收益，形成调优对比报告

> Unigemm 是华为鲲鹏数学库（KML）的核心组件之一。它针对鲲鹏的 cache 层级和 NEON I8MM 指令做了深度调优。llama.cpp 里 `ggml_mul_mat` 占了推理时间的 60-70%，如果把这部分换成 Unigemm 的路由，收益会非常直接。

### 🎯 目标 3：形成"推理调优 Agent 工作流"

把经验固化为可复用的自动化流程——**用 LLM Agent 来自动做推理调优**：

- Agent 自动跑 benchmark，搜集性能数据（tokens/s、每层延迟、cache miss、内存带宽）
- Agent 自动调参（线程数、batch size、编译选项、系统参数）并 A/B 对比
- 最终输出一份"最优配置推荐报告"

> 人工调优的问题是：每次换个模型、换个量化、换个服务器，十几个参数排列组合，试一遍要好几天。这部分完全可以用 Agent 自动化。

---

## 二、阶段详细安排

### Phase 1：基础认知 — 搞清楚你在什么芯片上跑（Week 1-2）

#### Week 1：LLM 推理基础

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 读 [Transformer 论文](https://arxiv.org/abs/1706.03762)，理解自注意力机制 | 画出完整的 Transformer 结构图 |
| Day 2 | 理解 Q、K、V 是什么，注意力计算 `softmax(QK^T / √d) V` | 手写伪代码实现 attention |
| Day 3 | MoE（Mixture of Experts）：总参数 vs 激活参数 | 解释 Qwen3.6-35B-A3B 的命名含义 |
| Day 4 | KV Cache 是什么、为什么需要、怎么存储 | 画出 KV Cache 的生存周期 |
| Day 5 | RoPE 位置编码、FFN、LayerNorm 概览 | 总结每个组件在推理中的角色 |

**交付物：** 一篇技术笔记，用自己的话解释 LLM 推理流程（1500 字以上）。

#### Week 2：鲲鹏 920F 与硬件基础

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 鲲鹏 920F 架构：ARMv8.2、64核、8通道 DDR4、cache 层级 | 画芯片架构图 |
| Day 2 | NEON SIMD 基础：向量化、寄存器、指令集（I8MM、DOTPROD） | 读几个 NEON intrinsic 示例 |
| Day 3 | 推理的瓶颈分析：为什么是内存带宽而不是算力 | 了解 roofline model |
| Day 4 | 查看服务器信息：`lscpu`、`lstopo`、`dmidecode`、`numactl -H` | 提交硬件拓扑报告 |
| Day 5 | 部署框架对比：llama.cpp vs Ollama vs vLLM on ARM | 列出对比表格 |

**交付物：** 一份"鲲鹏 920F 硬件拓扑 + 推理约束分析"报告。

---

### Phase 2：动手部署 — 跑通第一个模型（Week 3-4）

#### Week 3：环境搭建与编译

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 登录服务器，检查环境 | 提交环境检查报告 |
| Day 2 | 编译 llama.cpp（基础版） | 记录编译过程 |
| Day 3 | 模型传输 + 完整跑通一次对话 | 截图 + tokens/s |
| Day 4 | 理解 ggml 启动日志：NEON、BLAS、线程数 | 解读日志文章 |
| Day 5 | `llama-cli --perf` 看每层耗时 | 标注 top3 瓶颈 |

**交付物：** 部署跑通，记录基准性能。理解 attention 层和 FFN 层各自占比。

#### Week 4：API 服务化部署

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 启动 `llama-server`，理解 OpenAI API | curl 调用成功 |
| Day 2 | Python 客户端 + 压力测试 | chat.py + 并发测试 |
| Day 3 | systemd 服务 + 开机自启 | qwen3-api.service |
| Day 4 | 日志管理 + 监控搭建 | 日志轮转 + 监控脚本 |
| Day 5 | 阶段性总结 | 周报 + L2 考核 |

**交付物：** API 服务 + Python 客户端 + systemd 管理。完成 **目标 1** 初步验收。

---

### Phase 3：Unigemm 与矩阵计算优化（Week 5-7）

#### Week 5：理解 ggml 的矩阵乘法体系

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | ggml 数据结构：`ggml_tensor`、`ggml_context`、张量布局 | 画内存结构图 |
| Day 2 | `ggml_mul_mat` 的 dispatch 链路：ops.cpp → arm 内核 | 画调用链路 |
| Day 3 | 看 arm 内核源码：`ggml/src/ggml-cpu/arch/arm/` | 读代码写注释 |
| Day 4 | 理解量化矩阵乘法：Q4_K_M 如何在乘法时反量化 | 反量化数据流图 |
| Day 5 | 用 `--perf` 定位 `ggml_mul_mat` 在推理中的占比 | 时间占比报告 |

**交付物：** ggml 矩阵乘法体系的源码分析文档。

#### Week 6：Unigemm（鲲鹏数学库 KML）学习

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 了解 KML 整体架构：Blas、Sparse、FFT、Unigemm | 功能概览表 |
| Day 2 | 安装 KML 开发包：配置 yum 源，`yum install kml` | 安装成功 |
| Day 3 | 读 Unigemm API 文档：矩阵布局、调用约定、数据类型 | API 摘要笔记 |
| Day 4 | 写一个独立的 C 程序调用 Unigemm 做矩阵乘法并 benchmark | Unigemm vs 原生速度对比 |
| Day 5 | 理解 Unigemm 的 I8MM 内核：为什么比通用实现快 | NEON I8MM 指令分析 |

**交付物：** Unigemm 调用 demo + 独立 benchmark 数据。

#### Week 7：Unigemm 接入 llama.cpp

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 设计接入方案：在 ggml 中加一个 backend，还是替换 dispatch | 方案设计文档 |
| Day 2 | 实现方案：在 `ggml_mul_mat` dispatch 中加 Unigemm 路径 | 代码 diff |
| Day 3 | 正确性验证：Unigemm 输出 vs 原生输出一致 | 数值误差报告 |
| Day 4 | 性能对比：Unigemm vs 原生 NEON，不同批量大小 | 完整 A/B 对比表 |
| Day 5 | 调优：tiling size、线程数、缓存对齐 | 最优参数配置 |

**交付物：** 完成 **目标 2**——Unigemm 接入并量化收益。

---

### Phase 4：系统级调优 + Agent 工作流（Week 8-10）

#### Week 8：系统级调优实验

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | NUMA 拓扑 + 绑核实验 | NUMA + taskset 数据 |
| Day 2 | 关 NUMA balancing + CPU governor | 系统调优前后对比 |
| Day 3 | HugePages + THP + mlock | 内存配置实验 |
| Day 4 | `-t` 线程数扫描 | 画 "核心数 vs tokens/s" 曲线 |
| Day 5 | `-b` batch size + `-c` 上下文扫描 | 最优参数组合 |

**交付物：** 系统参数调优对比表。

#### Week 9：设计 Agent 工作流

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | Agent 整体架构设计：Benchmark → Analyze → Recommend → Implement | 架构图 |
| Day 2 | 实现 Benchmark Agent：自动跑不同参数组合 | `benchmark_agent.py` |
| Day 3 | 实现 Analyze Agent：解析 perf 数据，定位瓶颈 | `analyze_agent.py` |
| Day 4 | 实现 Recommend Agent：对比历史数据，给最优配置 | `recommend_agent.py` |
| Day 5 | 串联：一次命令完成全流程调优 | `tune.py --model model.gguf` |

**交付物：** 可运行的 Agent 调优工具 v1.0。

Agent 工作流设计：

```
输入：模型文件 + 服务器信息
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Benchmark Agent                                 │
│ ├─ 编译选项扫描（KleidiAI on/off, BLAS on/off）    │
│ ├─ 运行参数扫描（-t, -b, -c, -np, --mlock）       │
│ └─ 系统参数扫描（NUMA, HugePages, governor）       │
│     ↓                                            │
│  每轮测试输出: tokens/s, TTFT, 内存占用, CPU%      │
└───────────────────┬─────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ Analyze Agent                                   │
│ ├─ 解析 perf 结果，标注 top3 瓶颈                 │
│ ├─ roofline 分析：算力 vs 带宽                    │
│ ├─ 对比历史数据，标记异常值                        │
│ └─ 输出：瓶颈列表 + 数据可视化                      │
└───────────────────┬─────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ Recommend Agent                                 │
│ ├─ 给定约束（内存、并发要求）                      │
│ ├─ 搜索最优配置（历史数据 + 启发式规则）            │
│ └─ 输出：最优编译 + 运行 + 系统配置                │
└───────────────────┬─────────────────────────────┘
                    ▼
输出：最优配置报告（json + markdown）
```

#### Week 10：Agent 工作流迭代

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1 | 用 Agent 在自己的服务器上跑一次全流程 | 完整调优报告 |
| Day 2 | 对比 Agent 推荐配置 vs 人工调优最优配置 | 差距分析 |
| Day 3 | 优化 Agent：引入贝叶斯搜索或网格搜索策略 | 收敛速度提升 |
| Day 4 | 支持多模型：换 Qwen2.5 等模型测试泛化性 | 跨模型验证 |
| Day 5 | 撰写 Agent 使用文档 | README + 示例 |

**交付物：** Agent 工作流 v2.0 + 多模型验证报告。完成 **目标 3**。

---

### Phase 5：综合实战与总结（Week 11-12）

#### Week 11：综合实战

从以下题目中选一个做深度项目：

**A. Unigemm 完整集成到 ggml**
- 不只是 `ggml_mul_mat`，把 Unigemm 扩展到更多算子
- 做充分测试覆盖，考虑作为正式 PR 提交
- 产出：完整集成 patch + 测试报告

**B. Agent 工作流产品化**
- 加 Web UI 展示调优结果
- 加历史数据库，支持对比不同时间、不同模型的结果
- 产出：产品化的调优平台

**C. 新模型适配**
- 在鲲鹏 920F 上部署并调优另一个架构不同的模型（如 DeepSeek、Llama、ChatGLM）
- 对比 Agent 跨模型的泛化能力
- 产出：多模型调优对比报告

#### Week 12：总结与汇报

| 天数 | 内容 |
|------|------|
| Day 1-3 | 整理 12 周所有交付物 |
| Day 4-5 | 写最终技术报告（数据驱动、对比清晰） |
| Day 6-7 | 技术分享（25分钟 + 5分钟 Q&A） |
| Day 8 | 代码 Review + 成果展示 |
| Day 9-10 | 转正面谈准备 |

**最终交付物：** 技术报告 + 代码贡献 + Agent 工作流 + 转正申请。

---

## 三、考核标准

### 🎯 目标 1：在鲲鹏 920F 上跑通（Week 4 末）

现场实操（30 分钟）：
1. 从零编译 llama.cpp
2. 启动 API 服务
3. curl 发起一次对话
4. 说出当前 tokens/s
5. 解释部署过程中的三个关键决策

### 🎯 目标 2：Unigemm 调优（Week 7 末）

案例分析 + 实操：
1. 解释 ggml_mul_mat 的 dispatch 链路
2. 展示 Unigemm 接入的代码 diff
3. 给出接入前后的性能对比数据
4. 解释为什么 Unigemm 更快（从 I8MM 指令角度）

### 🎯 目标 3：Agent 工作流（Week 10 末）

演示：
1. 一条命令完成全流程调优
2. Agent 自动发现至少一个"人工没想到"的优化点
3. 输出结构化的调优报告

---

## 四、预期收益

### 推理速度（Qwen3.6-35B-A3B, UD-Q4_K_M, 鲲鹏 920F）

| 阶段 | tokens/s | 累计提升 |
|------|---------|---------|
| 原生编译 + 默认参数 | ~5 | — |
| + KleidiAI / OpenBLAS | ~7 | +40% |
| + 系统调优（NUMA + HugePages + 绑核） | ~9 | +80% |
| + 最优运行参数 | ~10 | +100% |
| **+ Unigemm（预期）** | **~12-14** | **+140~180%** |
| **+ Agent 自动搜索最优组合** | **逼近硬件上限** | **自动化 + 可复现** |

> Unigemm 的收益预期基于：矩阵乘法占推理 ~60% 时间，鲲鹏数学库相对通用 NEON 内核的 1.5-2x 加速。实际需要实验验证。

### Agent 工作流收益

| 指标 | 人工调优 | Agent 调优 |
|------|---------|-----------|
| 一次全流程调优耗时 | 2-3 天 | 2-3 小时 |
| 参数覆盖面 | 10-20 组 | 100+ 组 |
| 可复现性 | 低（人的记忆） | 高（commit history） |
| 最佳配置发现 | 依赖经验 | 数据驱动 |
| 换模型后的迁移成本 | 重新试一遍 | 换行命令 |

---

## 五、资源清单

### 本仓库

| 文件 | 用途 |
|------|------|
| `DEPLOY.md` | 部署实操 |
| `download-model.sh` | 模型下载 |
| `llama.cpp/` | 完整源码 |

### 必读资料

| 材料 | 优先级 | 对应目标 |
|------|--------|---------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | ⭐ | 目标 1 |
| [llama.cpp 源码](https://github.com/ggerganov/llama.cpp) | ⭐ | 目标 1 |
| [KML 鲲鹏数学库文档](https://www.hikunpeng.com/developer/hpc/kml) | ⭐ | 目标 2 |
| [Unigemm API 说明](https://support.huawei.com/enterprise/zh/doc/EDOC1100283144) | ⭐ | 目标 2 |
| [ARM NEON I8MM 指令集](https://developer.arm.com/architectures/instruction-sets/intrinsics/) | ⭐ | 目标 2 |
| [鲲鹏 920 架构分析](https://chipsandcheese.com/p/huaweis-kunpeng-920-and-taishan-v110) | 🟡 | 目标 1 |
| [KleidiAI 文档](https://github.com/ARM-software/kleidiai) | 🟡 | 目标 1 |
| [Linux perf 教程](https://perf.wiki.kernel.org/) | 🟡 | 目标 3 |
| [llama.cpp MoE 优化讨论](https://github.com/ggml-org/llama.cpp/issues/17936) | 🟢 | 目标 1 |

---

## 六、导师指引

### 三条目标的优先级关系

```
目标 1（跑通）── 硬性前置，第 4 周必须完成
     │
     ▼
目标 2（Unigemm）── 核心技术难点，第 5-7 周攻坚
     │  真正的价值在"能不能把华为自己的数学库用上"
     ▼
目标 3（Agent）── 工程能力，第 8-10 周构建
    把前两个目标的经验沉淀为自动化工具
```

### 导师每周检查清单

| 周次 | 检查点 |
|------|--------|
| 1-2 | 技术笔记质量？硬件拓扑报告是否完整？ |
| 3 | 编译是否顺利？是否记录了每一步的输出？ |
| 4 | 服务是否稳定？代码质量如何？ |
| 5 | ggml 源码理解到什么程度？ |
| 6 | Unigemm 调用是否成功？benchmark 方法是否合理？ |
| 7 | 接入方案设计是否稳健？收益量化是否可信？ |
| 8 | 系统调优实验是否控制变量？数据是否可复现？ |
| 9 | Agent 架构设计是否合理？ |
| 10 | Agent 是否真的发现了"人不知道"的最优配置？ |
| 11 | 综合项目的深度和质量？ |
| 12 | 技术报告的数据是否扎实？表达是否清晰？ |

---

## 七、术语表

| 术语 | 说明 |
|------|------|
| 鲲鹏 920F | 华为 7nm ARM 服务器芯片，64核 ARMv8.2，8通道 DDR4 |
| NEON | ARM 架构 SIMD 指令集，128-bit 向量处理器 |
| I8MM | ARMv8.2 的 INT8 矩阵乘累加指令，用于量化推理加速 |
| KML | Kunpeng Math Library，华为鲲鹏数学库 |
| Unigemm | KML 的核心组件，针对鲲鹏深度优化的通用矩阵乘法 |
| Unigemm | 统一矩阵乘（Universal GEMM），支持多种数据类型和布局 |
| roofline model | 性能分析模型，判断当前瓶颈在算力还是带宽 |
| Agent Workflow | 用 LLM Agent 驱动的自动化调优流水线 |
