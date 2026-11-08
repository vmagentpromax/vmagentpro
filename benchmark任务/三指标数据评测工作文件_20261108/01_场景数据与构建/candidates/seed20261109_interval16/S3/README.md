# S3 GPU 碎片整理实例

此实例来自真实 H20 需求，经受控重放置构造。筛选：通过；原因：全部固定条件满足。
完整规则、来源筛选偏差和复现环境见套件根目录 README.md。

- 随机种子 20261109；S1–S3 指数到达均值 16.0 秒。
- 100 台真实服务器，800 GPU；每台 8 H20 / 192 CPU 核。
- 初始 284 Pod，560 GPU（70%），5648 CPU 核；空闲卡数→机器数：{"1": 10, "2": 40, "3": 50}。
- 到达 160 个 Pod，GPU 卡数分布 {"1": 80, "2": 32, "4": 24, "8": 24}；到达结束 2569.25 秒。
- 到达 GPU 工作量 2272435.2000 卡秒；服务时长分位数 [min,Q1,median,Q3,max]：[1800.0, 1800.0, 4320.0, 7200.0, 7200.0]。
- 初始残余时长分位数：[725.095492576298, 1361.3825090504627, 3081.9761222254474, 4748.602358859418, 6446.664141605267]。
- t=0 独立请求分类：{"direct": 112, "fragmentation": 48}。静态独立探测不是动态启动率。
- 8卡请求资源下界至少 2 次移动；已找到的最少移动数 3（启发式找到的上界，不是最优性证明）。
- 同步数决策空间审计：{"eight_gpu_found_plans": 40, "one_move_found_plans": 0, "one_move_cost_min_gpu_sec": null, "one_move_cost_max_gpu_sec": null, "one_move_target_hosts": 0}。不以额外多搬Pod造成的成本差异冒充选择难度。
- 忽略单机约束的同总资源乐观池：300秒启动率 100.000%，P95 等待 0.00 秒。它只作容量诊断，不能当合法策略。

## 八种基线的实际结果

|放置策略|迁移策略|P95 等待/秒|300秒启动率|0–3600秒有效 GPU 分配率|迁移次数|GPU迁移量|GPU暂停成本/卡秒|
|---|---|---:|---:|---:|---:|---:|---:|
|first_fit|none|2119.05|83.125%|69.127%|0|0|0.00|
|first_fit|one_move|304.39|94.375%|79.391%|28|38|1403.80|
|first_fit|cheap_multi|123.22|100.000%|79.887%|45|53|1666.10|
|first_fit|first_multi|655.73|86.250%|76.305%|80|146|6170.90|
|best_fit|none|2545.18|83.125%|68.004%|0|0|0.00|
|best_fit|one_move|332.82|94.375%|79.357%|28|34|1163.00|
|best_fit|cheap_multi|123.22|100.000%|79.911%|43|51|1603.50|
|best_fit|first_multi|259.48|96.250%|78.655%|62|123|5393.60|

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
