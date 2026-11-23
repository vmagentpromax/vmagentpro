# S3 GPU 碎片整理实例

此实例来自真实 H20 需求，经受控重放置构造。筛选：通过；原因：全部固定条件满足。
完整规则、来源筛选偏差和复现环境见套件根目录 README.md。

- 随机种子 20261107；S1–S3 指数到达均值 10.0 秒。
- 100 台真实服务器，800 GPU；每台 8 H20 / 192 CPU 核。
- 初始 294 Pod，560 GPU（70%），6097 CPU 核；空闲卡数→机器数：{"1": 10, "2": 40, "3": 50}。
- 到达 160 个 Pod，GPU 卡数分布 {"1": 80, "2": 32, "4": 24, "8": 24}；到达结束 1575.41 秒。
- 到达 GPU 工作量 2158678.2000 卡秒；服务时长分位数 [min,Q1,median,Q3,max]：[1800.0, 1800.0, 4320.0, 7200.0, 7200.0]。
- 初始残余时长分位数：[726.544120128359, 1335.3068396976623, 2966.215964273046, 4198.607472344116, 6458.06564864142]。
- t=0 独立请求分类：{"direct": 112, "fragmentation": 48}。静态独立探测不是动态启动率。
- 8卡请求资源下界至少 2 次移动；已找到的最少移动数 3（启发式找到的上界，不是最优性证明）。
- 同步数决策空间审计：{"eight_gpu_found_plans": 40, "one_move_found_plans": 0, "one_move_cost_min_gpu_sec": null, "one_move_cost_max_gpu_sec": null, "one_move_target_hosts": 0}。不以额外多搬Pod造成的成本差异冒充选择难度。
- 忽略单机约束的同总资源乐观池：300秒启动率 99.375%，P95 等待 133.31 秒。它只作容量诊断，不能当合法策略。

## 八种基线的实际结果

|放置策略|迁移策略|P95 等待/秒|300秒启动率|0–3600秒有效 GPU 分配率|迁移次数|GPU迁移量|GPU暂停成本/卡秒|
|---|---|---:|---:|---:|---:|---:|---:|
|first_fit|none|2856.13|78.125%|71.521%|0|0|0.00|
|first_fit|one_move|1144.65|85.000%|80.081%|37|80|3762.80|
|first_fit|cheap_multi|861.19|80.000%|81.999%|71|88|2835.30|
|first_fit|first_multi|1441.87|79.375%|79.010%|80|132|5366.80|
|best_fit|none|2886.36|79.375%|70.825%|0|0|0.00|
|best_fit|one_move|1044.11|85.000%|80.857%|36|73|3363.40|
|best_fit|cheap_multi|759.17|81.250%|82.472%|69|81|2540.90|
|best_fit|first_multi|1380.97|81.250%|79.337%|80|156|7032.70|

P95 包含全部到达任务：未启动任务按 10800−arrival 记录右删失等待下界。执行截至10800秒；GPU有效分配率固定统计前3600秒；完整窗口分配率另列在 CSV。
成本 migration_gpu_pause_sec 为启动迁移时承诺的 GPU×暂停秒数，跨窗口的迁移也计入；accrued 字段为窗口内实际累计暂停卡秒。保留初始 Pod 与到达 Pod 的全部计算占用，GPU 分配率不等于实际 SM 利用率。

## 文件与运行

servers.parquet、initial_pods.parquet、arrivals.parquet、config.json 是数据输入；运行时只向策略揭示当前资源、正在运行 Pod 的需求和已经到达的等待队列。完整 arrivals 由仿真器持有。
validation.json、screening.json、baselines/、evaluator_only/ 只给评测者。迁移见证只用于独立合法性验证，不能作为策略提示或输入。

```powershell
python verify_instance.py
python reproduce_instance.py --output 'D:\一般文件\benchmark任务\S3_reproduced'
```

verify_instance.py 使用根目录冻结源摘录独立重放八套基线事件；reproduce_instance.py 从冻结需求池、随机种子和构造器重建本实例，再比较全部输入和事件。完整原始数据复现见根目录 reproduce.py。
