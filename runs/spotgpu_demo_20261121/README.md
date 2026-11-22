# SpotGPU 抢占演示场景（小规模，几秒跑完）

生成时间：2026-11-21
构建脚本：`experiments/spotgpu_demo/build_demo_scene.py`
数据来源：`runs/hetero8d_20261101/inputs/jobs_F.csv` 的**子集**，所有字段值未经修改

## 这个场景解决什么问题

引擎从 `d20f8b1` 起就支持 HP/LP 抢占，但：

- **内置场景 A 完全没有 `priority_class`**（实测 50 个任务全为 None）→ 抢占路径永不触发；
- **场景 F 有真实优先级**（GPU 任务 HP 447 / LP 5,215 / Other 248），但它是 119,495 任务 / 30 节点，单组 25–280 秒。

结果就是：**clone 下来的人看不到这个项目最新的能力**。本场景用 29 个任务、0.003 秒墙钟把抢占完整演示出来。

## 场景构成

| 项 | 值 |
|---|---|
| 节点 | 3 台 XPU-A，每台 8 GPU / 192 核 → **24 GPU / 576 核** |
| 任务 | **29 个**：24 个 LP + 5 个 HP |
| LP 占用 | 24.0 GPU（**恰好填满池**） |
| 锚点 T | **95,999.394 秒** |
| T 时刻全 trace 的 LP 并发占用 | **49.0 GPU**（远超本池 24，说明饱和是真实重叠而非人为构造） |
| HP 到达窗 | `[95,999.394, 103,199.394]` 秒（T 后 2 小时） |
| checkpoint 间隔 | 3,600 秒 |
| 测量窗 | **[86,400, 129,600] 秒**（12 小时） |
| 仿真上限 | 400,000 秒 |

**测量窗必须从 86,400 秒起。** 本场景的 Pod 都在 87,211 秒之后到达（它们是场景 F 8 天中段的一片），最后一个约在 118,463 秒结束。若按默认把窗口设成 `[0, 43200]`，运行照样排空、抢占数照样是 5，但四个负载/预留率会全部显示 0%——数字看起来像坏了，其实是窗口没盖住活动。`tests/test_spotgpu_demo_scene.py` 里有一条测试专门盯这一点。

## 选法（以及两种被否决的做法，都实测过）

选 LP 的规则是**"在 T 时刻真的在跑"**，不是"落在某个时间窗里"：

1. 在 XPU-A 的 GPU 任务中，按时间找**第一个 HP 到达时刻 T，使得 T 时刻并发 LP 占用 ≥ 池容量**；
2. 取 T 时刻仍在运行、服务时长 ≥ 一个 checkpoint 间隔、且活过窗口末的 LP，按 GPU 降序填到池满；
3. 取 `[T, T+2h]` 内到达的 HP。

**被否决的做法 A：单一时间窗，按到达时间排序混放。** `_spotgpu_choose_action` 本身按 HP 优先排序，于是 156 个 HP 把 24 个 GPU 槽全抢了——实测 `node_lp_running = [0,0,0]`，**24 个运行任务全是 HP，抢占数 0**。引擎行为正确，是场景设计错了。

**被否决的做法 B：LP 先占一个窗、HP 后占一个窗。** 真实 LP 服务时长很短，等 HP 窗口打开时只剩 8 个 LP 还活着，且该窗口内池机型（XPU-A）的 HP 到达数为 0 → **选出 0 个任务**。

## 结果（本机实测）

| 策略 | 完成 | 拒绝 | HP 最大等待 | HP 平均等待 | LP 平均等待 | 抢占次数 | wasted_gpu_sec | 墙钟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `first_fit` | 29 | 0 | **8,352 s** | 5,739.1 s | 0.0 s | 0 | 0 | 0.003 s |
| `best_fit` | 29 | 0 | **8,352 s** | 5,739.1 s | 0.0 s | 0 | 0 | 0.003 s |
| `spotgpu` | 29 | 0 | **0 s** | **0.0 s** | **437.3 s** | **5** | **1,210.5** | 0.003 s |

**读法**：SpotGPU 把 HP 的最大等待从 8,352 秒压到 0 秒，代价是 LP 平均多等 437 秒、浪费 1,210.5 GPU·秒。这就是论文 §4.2 说的 SLO 保障与利用率之间的权衡，在这里一眼可见。

三种策略都 `end_reason=drained`、完成 29、拒绝 0，所以差异**不是**"谁跑完了"，而是"谁在等、谁被中断"。

同一窗口下的 GPU 资源预留率：`first_fit` / `best_fit` **41.12%**，`spotgpu` **41.24%**；CPU 分别 13.15% 与 13.19%。抢占让利用率略升，来自被驱逐 LP 短暂空出的份额被 HP 立即顶上。

### 六个 SpotGPU 开关预设在这份数据上的差别（2026-11-22 实测）

`spotgpu` 现在是六个正交开关的一组取值，`experiments/preemption_20261107/harness.py` 的五个实验策略就是其中的六组预设（见 `README_GPU.md` 8.1 节）。同一份 29 任务数据、同一套参数，只换预设：

| 预设 | 抢占次数 | `wasted_gpu_sec` | GPU 预留率 | HP 最大等待 | LP 平均等待 |
|---|---:|---:|---:|---:|---:|
| `legacy_spotgpu`（引擎默认） | 5 | **1,210.5** | 0.412394 | 0 s | 437.3 s |
| `paper_cost` | 5 | 1,210.5 | 0.412394 | 0 s | 437.3 s |
| `no_history` | 5 | 1,210.5 | 0.412394 | 0 s | 437.3 s |
| `random_victim` | 5 | **9,906.1** | 0.420781 | 0 s | 437.3 s |
| `paper_baseline` | 5 | 9,906.1 | 0.420781 | 0 s | 437.3 s |
| `no_preempt` | 0 | 0 | 0.411227 | **8,352 s** | 0 s |

**这份小数据把"成本感知受害者"的作用单独显示出来了**：随机挑受害者，抢占次数一样是 5、HP 一样 0 等待，但浪费的计算从 1,210.5 涨到 **9,906.1 GPU·秒（8.2 倍）**。这正是论文用合并基线测出 24% 的那件事，在这里只需要 0.003 秒就能看见。

复现：

```bash
python -c "
import sys; sys.path.insert(0,'.')
from vmagent.gpu.contract import load_nodes, load_workload
from vmagent.gpu.environment import GpuSchedEnv
from vmagent.gpu.spotgpu import VARIANTS
n=load_nodes('runs/spotgpu_demo_20261121/inputs/nodes.csv')
t=load_workload('runs/spotgpu_demo_20261121/inputs/jobs_spotgpu_demo.csv')[0]
for name in sorted(VARIANTS):
    e=GpuSchedEnv(n,list(t),clock='fast',queue_policy='backfill',seed=42,
                  measurement_window=[86400.,129600.],max_sim_time=400000.,spotgpu=VARIANTS[name])
    s=e.run('spotgpu'); print(name,s['preemptions'],round(s['wasted_gpu_sec'],1))"
```

等价性核对（开关组合 == 原子类实现，逐字段比 records 与 events）：

```bash
python tools/verify_spotgpu_variants.py --limit 40000
```

## 复跑命令

```bash
# 不抢占（HP 只能等 LP 自然结束）
python -m vmagent.gpu.run --clock fast --seed 42 --policy first_fit --queue-policy backfill ^
  --measurement-start-seconds 86400 --measurement-seconds 43200 ^
  --max-sim-time 400000 --checkpoint-interval 3600 ^
  --nodes runs/spotgpu_demo_20261121/inputs/nodes.csv ^
  --tasks runs/spotgpu_demo_20261121/inputs/jobs_spotgpu_demo.csv ^
  --output outputs/gpu/demo-first_fit

# 抢占（HP 立刻插入，LP 被驱逐后按 checkpoint 恢复）
python -m vmagent.gpu.run --clock fast --seed 42 --policy spotgpu --queue-policy backfill ^
  --measurement-start-seconds 86400 --measurement-seconds 43200 ^
  --max-sim-time 400000 --checkpoint-interval 3600 ^
  --nodes runs/spotgpu_demo_20261121/inputs/nodes.csv ^
  --tasks runs/spotgpu_demo_20261121/inputs/jobs_spotgpu_demo.csv ^
  --output outputs/gpu/demo-spotgpu
```

网页里更省事：打开前端后点“SpotGPU 抢占演示数据”，数据和上面这套参数（含 86,400 秒起点）会一起设好。

重建场景（参数可调）：

```bash
python experiments/spotgpu_demo/build_demo_scene.py --nodes 3 --hp-after-hours 2 --max-tasks 200
```

`--nodes` 改变池容量；太小会触发"找不到并发 LP 占用 ≥ 容量的 HP 到达时刻"并明确报错，不会静默产出空场景。

## 文件

```text
runs/spotgpu_demo_20261121/
├── README.md                              本文
└── inputs/
    ├── nodes.csv                          3 台 XPU-A
    ├── jobs_spotgpu_demo.csv              29 个任务（24 LP + 5 HP）
    └── construction_stats.json            锚点 T、并发占用、选中/未选计数、来源声明
```

## 使用边界

- **它是演示，不是基准。** 29 个任务、3 台节点，任何指标都不应外推到集群规模。要评价策略请用场景 F。
- **LP 被选中时加了"必须活过窗口末"的约束**，所以 LP 的服务时长分布偏长，不代表 LP 整体。
- **锚点 T 是按"并发 LP 占用 ≥ 池容量"挑的第一个 HP 到达时刻**，带有选择性；换 `--nodes` 会得到不同的 T 和不同的任务集合。
- `wasted_gpu_sec` 与 `preserved_progress` 依赖 `checkpoint_interval` 这个**模型假设**（论文里 checkpoint 间隔是部署参数，trace 未提供真实值）。改 `--checkpoint-interval` 会直接改变浪费量。
- 场景 F 里 49 个 H20 专属任务在本池会被 `gpu_model_unavailable` 拒绝，所以构建脚本按池机型做了过滤——这是为了让演示的行都在讲抢占，而不是在讲拒绝。
