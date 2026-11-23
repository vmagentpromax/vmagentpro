# S2 GPU 碎片整理实例

此实例来自真实 H20 需求，经受控重放置构造。筛选：通过；原因：全部固定条件满足。
完整规则、来源筛选偏差和复现环境见套件根目录 README.md。

- 随机种子 20261108；S1–S3 指数到达均值 10.0 秒。
- 100 台真实服务器，800 GPU；每台 8 H20 / 192 CPU 核。
- 初始 293 Pod，560 GPU（70%），6282 CPU 核；空闲卡数→机器数：{"0": 40, "2": 20, "4": 20, "6": 20}。
- 到达 160 个 Pod，GPU 卡数分布 {"1": 80, "2": 32, "4": 24, "8": 24}；到达结束 1424.75 秒。
- 到达 GPU 工作量 2141692.8000 卡秒；服务时长分位数 [min,Q1,median,Q3,max]：[1800.0, 1800.0, 6047.8, 7200.0, 7200.0]。
- 初始残余时长分位数：[721.273391514663, 1285.5361540805497, 3039.027017609844, 4469.514072555664, 6444.236648190321]。
- t=0 独立请求分类：{"direct": 136, "fragmentation": 24}。静态独立探测不是动态启动率。
- 8卡请求资源下界至少 1 次移动；已找到的最少移动数 1（启发式找到的上界，不是最优性证明）。
- 同步数决策空间审计：{"eight_gpu_found_plans": 95, "one_move_found_plans": 19, "one_move_cost_min_gpu_sec": 72.8, "one_move_cost_max_gpu_sec": 246.4, "one_move_target_hosts": 19}。不以额外多搬Pod造成的成本差异冒充选择难度。
- 忽略单机约束的同总资源乐观池：300秒启动率 100.000%，P95 等待 0.00 秒。它只作容量诊断，不能当合法策略。

## 八种基线的实际结果

|放置策略|迁移策略|P95 等待/秒|300秒启动率|0–3600秒有效 GPU 分配率|迁移次数|GPU迁移量|GPU暂停成本/卡秒|
|---|---|---:|---:|---:|---:|---:|---:|
|first_fit|none|2116.39|85.000%|75.952%|0|0|0.00|
|first_fit|one_move|91.06|98.125%|84.989%|24|58|2639.40|
|first_fit|cheap_multi|151.24|96.250%|84.589%|52|62|1980.80|
|first_fit|first_multi|739.46|86.250%|82.091%|70|129|5462.70|
|best_fit|none|1775.71|85.000%|78.216%|0|0|0.00|
|best_fit|one_move|94.58|98.125%|85.062%|23|55|2447.80|
|best_fit|cheap_multi|133.76|99.375%|85.130%|41|51|1636.90|
|best_fit|first_multi|615.95|88.125%|82.956%|59|113|4846.70|

P95 包含全部到达任务：未启动任务按 10800−arrival 记录右删失等待下界。执行截至10800秒；GPU有效分配率固定统计前3600秒；完整窗口分配率另列在 CSV。
成本 migration_gpu_pause_sec 为启动迁移时承诺的 GPU×暂停秒数，跨窗口的迁移也计入；accrued 字段为窗口内实际累计暂停卡秒。保留初始 Pod 与到达 Pod 的全部计算占用，GPU 分配率不等于实际 SM 利用率。

## 文件与运行

servers.parquet、initial_pods.parquet、arrivals.parquet、config.json 是数据输入；运行时只向策略揭示当前资源、正在运行 Pod 的需求和已经到达的等待队列。完整 arrivals 由仿真器持有。
validation.json、screening.json、baselines/、evaluator_only/ 只给评测者。迁移见证只用于独立合法性验证，不能作为策略提示或输入。

```powershell
python verify_instance.py
python reproduce_instance.py --output 'D:\一般文件\benchmark任务\S2_selected_reproduced'
```

verify_instance.py 使用根目录冻结源摘录独立重放八套基线事件；reproduce_instance.py 从冻结需求池、随机种子和构造器重建本实例，再比较全部输入和事件。完整原始数据复现见根目录 reproduce.py。
