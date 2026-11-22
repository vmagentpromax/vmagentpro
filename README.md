# VMAgentPro

**A timed GPU/CPU scheduling simulator built on the Alibaba ASI v2026 cluster trace.**
面向阿里 ASI v2026 轨迹的 **GPU + CPU 时序资源调度仿真器**：离散事件、纯 Python 标准库、可逐字段核对。

仓库地址：<https://github.com/vmagentpromax/vmagentpro>（上游基线：[mail-ecnu/VMAgent](https://github.com/mail-ecnu/VMAgent) `1ffabd6`）

---

## 30 秒跑起来

```powershell
# 方式一：双击仓库根目录的 launcher.bat（起网页并打开 http://127.0.0.1:8765/）

# 方式二：命令行跑一次实验
python -m vmagent.gpu.run --clock fast --policy spotgpu --queue-policy backfill `
  --nodes runs/hetero8d_20261101/inputs/nodes.csv `
  --tasks runs/hetero8d_20261101/inputs/jobs_F.csv `
  --measurement-start-seconds 86400 --measurement-seconds 691200 --max-sim-time 4000000 `
  --output outputs/gpu/F_spotgpu_backfill
```

不需要 numpy / pandas / torch / gym / TensorFlow；推荐 Python 3.11，最低 3.8。

## 输入与输出

| | 内容 |
|---|---|
| **输入** | `nodes.csv`（每台节点一行：`server_id, cluster_id, gpu_spec_public, gpu_count, cpu_capacity_cores[, asw_id]`）、`jobs.csv`（每个 pod 一行：GPU/CPU 请求、到达时刻、服务时长、机型白名单、优先级…）、可选 `initial_placement.csv`（`pod_id, server_id`，碎片整理基准的 t=0 布局） |
| **输出** | `summary.json`（整局指标）、`tasks.csv`（每任务）、`events.csv`（每事件，带当时账本快照）、`resource_history.csv`、`nodes_final.csv`、`run_manifest.json`（含环境戳） |

字段逐条含义、以及"哪些比率的分母是什么"见 [`README_GPU.md`](README_GPU.md) 与
`vmagentpro调研与改进方案/01_现状梳理/` 下两份权威文档。

## 有哪些策略

四条正交轴，命名策略只是轴上的点：

| 轴 | 取值 |
|---|---|
| 队列规则 | `fcfs` / `backfill`（+ SpotGPU 的 `hp_first_queue` 子开关） |
| 放置规则 | `first_fit` / `best_fit` / `worst_fit` / `random` / SpotGPU 三项字典序排序 |
| SpotGPU 开关 | `preemption`、`victim_selection`、`victim_prefix`、`hp_first_queue`、`placement_terms`、`placement_cpu_aware` |
| 碎片整理 | `none` / `one_move` / `cheap_multi` / `first_multi` × 四个"迁移期间资源怎么算"的语义开关 |

## 改完之后，怎么证明数字没动

```powershell
python -m pytest tests/ -q                 # 100 项
python tools/verify_metric_formulas.py     # 47 项：把文档里的公式当断言跑
python tools/verify_spotgpu_variants.py    # 六开关实现 == 历史手写实现
python tools/verify_golden_runs.py         # 14 组场景 F 归档结果逐字段复现（约 17 分钟）
python tools/check_doc_line_references.py --quiet   # 文档行号引用是否还有效
```

## 与上游 VMAgent 的关系

本项目 = 上游 VMAgent + 一个全新写的 GPU/CPU 时序引擎（`vmagent/gpu/`，15 个文件 2,662 行）。
删除上游残留之前，对上游已发布代码的实质改动只有 **17 行**。

**2026-11-22 起，上游那半棵树已从工作树移除**（`schedvis/` 前端、`schedgym/` 与 `vmagent/schedgym/`
的 CPU-NUMA 环境、`vmagent/{learners,modules,components,controllers,config,data,utils}` 与
`train.py`/`test.py`/`baselines.py` 的强化学习侧、`docs/source/` sphinx 文档、
`conda_env.yml`/`require_sched.txt`/`setup.py`）。理由：本项目做的是 GPU 算力调度，
那套 RL/CPU-NUMA 代码不再相交，留着只会让人误以为这里有两套可用的系统。

这不是历史重写：111 个文件都还在 git 里，随时取回——

```bash
git show 1ffabd6:schedvis/dashboard/callbacks.py     # 看原文
git checkout 1ffabd6 -- schedvis/ vmagent/config/    # 整目录取回
```

删除前已把其中**值得抄的设计**逐条留档（含路径行号与对应用途）：
`vmagentpro调研与改进方案/02_外部调研/原版VMAgent可借鉴点_删除前逐条抄下的清单.md`。

感谢 [MAIL, East China Normal University](https://mail-ecnu.cn) 开源 VMAgent；
许可见 [`LICENSE`](LICENSE)。中文详细说明见 [`README_CN.md`](README_CN.md)，
GPU 引擎完整说明见 [`README_GPU.md`](README_GPU.md)。
