"""Human-readable reports and manifest; no effect on instance selection."""
from pathlib import Path
import json,hashlib,collections
import pandas as pd
from source_adapter import dump,digest

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def baseline_table(df):
    lines=['|放置策略|迁移策略|P95 等待/秒|300秒启动率|0–3600秒有效 GPU 分配率|迁移次数|GPU迁移量|GPU暂停成本/卡秒|',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in df.to_dict('records'):
        lines.append(f"|{r['placement']}|{r['migration']}|{r['p95_wait_censored_sec']:.2f}|{r['start_within_deadline_rate']:.3%}|{r['gpu_effective_allocation_rate']:.3%}|{r['migration_count']}|{r['gpu_units_moved']}|{r['migration_gpu_pause_sec']:.2f}|")
    return '\n'.join(lines)

def finalize(root):
    root=Path(root);registry=load(root/'candidate_registry.json');summary=[]
    paths=[root/r['path'] for r in registry['candidates']]+[root/r for r in registry['selected'].values()]
    wrapper='''from pathlib import Path
import sys
here=Path(__file__).resolve().parent
root=next(p for p in here.parents if (p/'source_audit').exists() and (p/'engine.py').exists())
sys.path.insert(0,str(root))
'''
    for path in paths:
        v=load(path/'validation.json');cfg=load(path/'config.json');decision=load(path/'screening.json');df=pd.read_csv(path/'baseline_results.csv')
        s=cfg['scenario'];reference=v['aggregate_capacity_relaxation']
        # Show that costs differ even at the same move count, not merely because
        # an expensive plan moves unnecessary extra Pods.
        from engine import State,plans
        servers=pd.read_parquet(path/'servers.parquet').to_dict('records')
        initial=pd.read_parquet(path/'initial_pods.parquet').to_dict('records')
        arrivals=pd.read_parquet(path/'arrivals.parquet').to_dict('records')
        state=State(servers,initial);request=next(p for p in arrivals if p['gpu_request']==8)
        space={}
        if state.classify(request)=='fragmentation':
            found=plans(state,request,cfg);one=[p for p in found if len(p['moves'])==1]
            space={'eight_gpu_found_plans':len(found),'one_move_found_plans':len(one),
              'one_move_cost_min_gpu_sec':min((p['gpu_pause_sec'] for p in one),default=None),
              'one_move_cost_max_gpu_sec':max((p['gpu_pause_sec'] for p in one),default=None),
              'one_move_target_hosts':len({p['target'] for p in one})}
            if one:
                witness=load(path/'evaluator_only'/'migration_witnesses.json')
                item=next(x for x in witness if x['gpu']==8)
                item['plans']=list({json.dumps(p,sort_keys=True):p for p in item['plans']+[one[0],one[-1]]}.values())
                dump(path/'evaluator_only'/'migration_witnesses.json',witness)
        v['same_move_count_cost_choices']=space;v['validation_status']='PASS_RESOURCE_AND_LIFECYCLE_CHECKS';v['screening_passed']=decision['passed']
        dump(path/'validation.json',v)
        readme=f'''# {s} GPU 碎片整理实例

此实例来自真实 H20 需求，经受控重放置构造。筛选：{'通过' if decision['passed'] else '未通过'}；原因：{'; '.join(decision['reasons']) or '全部固定条件满足'}。
完整规则、来源筛选偏差和复现环境见套件根目录 README.md。

- 随机种子 {cfg['seed']}；S1–S3 指数到达均值 {cfg['arrival_mean_interval_sec_S1_S3']} 秒。
- {v['servers']} 台真实服务器，{v['gpu_capacity']} GPU；每台 8 H20 / 192 CPU 核。
- 初始 {v['initial_pods']} Pod，560 GPU（70%），{v['initial_cpu']:.0f} CPU 核；空闲卡数→机器数：{json.dumps(v['free_gpu_histogram'],ensure_ascii=False)}。
- 到达 {v['arrival_requests']} 个 Pod，GPU 卡数分布 {json.dumps(v['arrival_gpu_counts'])}；到达结束 {v['arrival_end_sec']:.2f} 秒。
- 到达 GPU 工作量 {v['arrival_gpu_service_work_sec']:.4f} 卡秒；服务时长分位数 [min,Q1,median,Q3,max]：{v['duration_quantiles_sec']}。
- 初始残余时长分位数：{v['initial_remaining_sec_quantiles']}。
- t=0 独立请求分类：{json.dumps(v['frozen_request_classes'])}。静态独立探测不是动态启动率。
- 8卡请求资源下界至少 {v['eight_gpu_min_moves_resource_lower_bound']} 次移动；已找到的最少移动数 {v['probe_by_size']['8']['min_found_moves']}（启发式找到的上界，不是最优性证明）。
- 同步数决策空间审计：{json.dumps(space,ensure_ascii=False)}。不以额外多搬Pod造成的成本差异冒充选择难度。
- 忽略单机约束的同总资源乐观池：300秒启动率 {reference['start_within_deadline_rate']:.3%}，P95 等待 {reference['p95_wait_censored_sec']:.2f} 秒。它只作容量诊断，不能当合法策略。

## 八种基线的实际结果

{baseline_table(df)}

P95 包含全部到达任务：未启动任务按 10800−arrival 记录右删失等待下界。执行截至10800秒；GPU有效分配率固定统计前3600秒；完整窗口分配率另列在 CSV。
成本 migration_gpu_pause_sec 为启动迁移时承诺的 GPU×暂停秒数，跨窗口的迁移也计入；accrued 字段为窗口内实际累计暂停卡秒。保留初始 Pod 与到达 Pod 的全部计算占用，GPU 分配率不等于实际 SM 利用率。

## 文件与运行

servers.parquet、initial_pods.parquet、arrivals.parquet、config.json 是数据输入；运行时只向策略揭示当前资源、正在运行 Pod 的需求和已经到达的等待队列。完整 arrivals 由仿真器持有。
validation.json、screening.json、baselines/、evaluator_only/ 只给评测者。迁移见证只用于独立合法性验证，不能作为策略提示或输入。

```powershell
python verify_instance.py
python reproduce_instance.py --output 'D:\\一般文件\\benchmark任务\\{path.name}_reproduced'
```

verify_instance.py 使用根目录冻结源摘录独立重放八套基线事件；reproduce_instance.py 从冻结需求池、随机种子和构造器重建本实例，再比较全部输入和事件。完整原始数据复现见根目录 reproduce.py。
'''
        (path/'README.md').write_text(readme,encoding='utf-8')
        (path/'verify_instance.py').write_text(wrapper+"from verify_suite import verify\nprint(verify(root,here)['status'])\n",encoding='utf-8')
        (path/'reproduce_instance.py').write_text(wrapper+"import argparse\nfrom reproduce import instance\na=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();instance(root,here,args.output.resolve())\n",encoding='utf-8')
        if path.name.endswith('_selected'):
            for r in df.to_dict('records'):summary.append(dict(scenario=s,**r))
    pd.DataFrame(summary).to_csv(root/'selected_baseline_summary.csv',index=False)
    lines=[]
    for s,folder in registry['selected'].items():
        v=load(root/folder/'validation.json');df=pd.read_csv(root/folder/'baseline_results.csv')
        lines.append(f"|{s}|{v['initial_pods']}|{v['arrival_requests']}|{df.p95_wait_censored_sec.min():.1f}–{df.p95_wait_censored_sec.max():.1f}|{df.start_within_deadline_rate.min():.2%}–{df.start_within_deadline_rate.max():.2%}|{df.gpu_effective_allocation_rate.min():.2%}–{df.gpu_effective_allocation_rate.max():.2%}|{df.migration_count.min()}–{df.migration_count.max()}|")
    failures=[r for r in registry['candidates'] if not r['passed']]
    screening=['|场景|种子|平均间隔/秒|判定|原因|','|---|---:|---:|---|---|']
    for r in registry['candidates']:screening.append(f"|{r['scenario']}|{r['seed']}|{r['interval_sec']:.0f}|{'通过' if r['passed'] else '未通过'}|{'; '.join(r['reasons']) or '满足'}|")
    (root/'SCREENING.md').write_text('# 全部候选筛选记录\n\n'+'\n'.join(screening)+'\n',encoding='utf-8')
    selected_seed=load(root/registry['selected']['S1']/'config.json')['seed'] if registry['selected'] else None
    text=f'''# 四类 GPU 碎片整理 Benchmark（真实需求，受控布局，动态评测）

正式保存目录：`D:\\一般文件\\benchmark任务\\四类场景`。构建于用户 Windows 本地，2026-11-07。
旧 `D:\\一般文件\\benchmark任务\\S2_001` 保留原状。本版重新读取真实源文件，没有将旧版 PASS 当作验证结论。

## 最终实例与实际结果

共同选中种子 **{selected_seed}**，S1–S3 平均到达间隔10秒。共生成28个候选（每类7个），每个运行8种放置×迁移组合，共224次动态评测，另有56个低碎片布局对照和28个总资源松弛对照。
通过筛选 {len(registry['candidates'])-len(failures)} 个，未通过 {len(failures)} 个；所有候选输入、结果和失败原因均保留，不删除不利结果。四个 selected 目录是同一候选组的副本。

|场景|初始Pod数|到达Pod数|八基线P95等待范围/秒|300秒启动率范围|前3600秒有效GPU分配率|迁移次数范围|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

全部实例为100台H20服务器，共800卡/19200 CPU核，初始560卡（70%）。完整结果见 selected_baseline_summary.csv、各实例 baseline_results.csv 与 baselines/*.json。
本表显示全部基线范围，不指定单一赢家；不同指标之间存在取舍。S1 的 P95 对少量慢请求敏感，不能从 P95=0 推断每条请求都零等待；同时报告平均等待和限时启动率。

## 四类定义及控制变量

- **S1：低初始碎片、预防后续碎片**。25台空闲8卡、20台空闲2卡、55台满载。所有初始独立探测均可直接放置，8卡不可用空闲比例16.67%；后续小任务如何使用完整空机影响大任务等待。
- **S2：中碎片、成本选择**。20台空闲6卡、20台空闲4卡、20台空闲2卡、40台满载；4卡可直接放置，8卡受阻。同一受阻8卡需求存在多个逐步合法且卡秒成本不同的迁移方案。
- **S3：高碎片、多步决策**。10台空闲1卡、40台空闲2卡、50台空闲3卡，4/8卡请求初始都受阻。最大空闲3卡且最大初始Pod为4卡，故一条8卡请求无法由单次迁移腾出单机容量，资源下界至少2次；实际找到并独立重放3步方案。这个证据不宣称必须有未来信息或启发式已最优。
- **S4：大任务集中到达与竞争**。初始布局逐字段等于S2；120个到达请求，1/2/4/8卡分别12/8/40/60个，大请求比例83.33%，S1–S3为30%。四个30请求短突发覆盖相同到达时段；按一个全局合成时长系数，精确匹配S2的总GPU×服务时间。卡数总量、任务数量和CPU工作量不强行匹配，均如实保留在数据中。

S1–S3逐字段固定服务器、Pod需求、服务时长/残余时间、到达任务及顺序，仅 initial_pods.server_id 改变。S4服务器与初始Pod布局等于S2。跨类断言由独立验证器再次检查。
标签是构造定义，不是数据集官方标签；不只按初始占用率命名。不可用空闲比例按某一请求卡数计算，另报告 CPU/型号合法的直接可放置主机数。

## 数据来源与审计

原始根目录：`E:\\cluster-trace-gpu-v2026\\data`。实际检查后采用：

- `asi_opensource_server_hourly/asi_opensource_server_hourly/day=0/hour=00/part-000.parquet`
- `asi_opensource_pod_hourly/asi_opensource_pod_hourly/day=0/hour=00/part-000.parquet`
- `asi_opensource_job_execution_summary/asi_opensource_job_execution_summary/part-000.parquet`

day/hour为trace相对时间，不映射到日历。限定同一 cluster_id=`1f28e063c4e5e1bc48ff55269617dd2f`、H20、8卡192核真实服务器；按种子无放回选择100台。
Pod源池不限于选定100台原始宿主机；经ready=true且used_gpu_hours>0活动证据、整数独占GPU、真实CPU单机可容纳及源机一致性筛选，保留4185个1/2/4/8卡Pod。
不把小数GPU向上取整；H20正需求中分数排除282/5460=5.164835%，全快照为9897/88167=11.225288%。CPU-only和罕见5/6卡请求未纳入，不能声称无偏覆盖所有GPU负载。
资源容量、Pod GPU卡数/CPU核数/型号始终取原值，无扩卡、无降低CPU需求、无跨型号迁移。初始布局采用带CPU约束的重放置；贪心构造失败时调用整数约束求解，不把搜索失败冒称不可行。

source_audit/provenance.json 保存三源文件完整SHA256、文件大小、行数、schema、筛选漏斗与依赖版本；source_audit/evaluator_only/ 冻结全部可用需求、服务器快照和匹配summary行。源文件只读。
pod_hourly.state_public为Unknown，不能恢复真实瞬间运行集合；该套件为真实需求驱动的合成动态实验，而非原始集群完整重放。初始Pod均假设可迁移，不意味着原应用支持checkpoint。

## 服务时长、到达与迁移参数

summary只有执行时长，没有可可靠连接的精确提交/开始/结束时间；4188个候选活动整数Pod仅149个有匹配summary（270行，82个多行）。
若Pod有同型号同卡数自身摘要则固定种子采样一条，否则从同型号同卡数匹配摘要记录重采样。保留donor Pod ID、原始行号、duration_hours_raw与方法。
`duration_base_sec=clip(duration_hours_raw*3600*0.01,1800,7200)`，初始残余为该时长乘U(0.4,0.9)；这些缩放、截断及残余均为合成，不能解释为真实时长。S4再乘一个全局工作量匹配系数，系数在config中。
S1–S3到达采用指数间隔，固定PCG64种子。最终10秒均值是工程压力参数：该请求集初始70%占用、真实服务分布经缩放后，10/16/40秒压力网格中10秒组同时满足四类筛选；未声称原始trace提交强度为10秒。
按 `GPU工作量/到达时段` 记录瞬态需求压力，而非只用均值间隔估计负载。低碎片对照及可分割资源池用于区分碎片损失和总资源竞争；S4资源池仍有等待，真实保留了容量竞争而非掩盖它。

迁移暂停时间 `20 + 8*GPU卡数 + 0.1*CPU核数` 秒，完全合成，旨在使大Pod更贵，未根据网络/显存臆造实测带宽。没有checkpoint大小数据，不能声称真实迁移时间已校准。
目标机在源机释放前必须能同时容纳完整Pod的GPU与CPU；迁移期间源机和目标机双占用，Pod暂停计算，结束后剩余服务时间不减，完成时间顺延。单迁移引擎串行，整个计划期间暂停新任务放置，其他Pod继续计算，完成和到达照常处理。全程禁止抢占、丢弃和环形原子交换。
最多6步/计划、80次已启动迁移/仿真，所有基线预算相同；跨结束窗口也计承诺成本。成本分别报告次数、GPU卡数、承诺GPU暂停卡秒及实际累计暂停卡秒。

## 仿真规则和基线

仿真0–10800秒，任务启动后服务时长才开始消耗；等待不消耗服务。每30秒及到达/完成/迁移完成时调度，同刻先完成再到达。等待队列按到达顺序扫描，允许后续任务回填。
GPU有效分配率只统计0–3600秒的实际计算分配（包含初始Pod，剔除迁移暂停），除以800×3600；另保存全10800秒积分。它不是实际SM利用率。长窗口全部完成时总有效工作量趋于相同，因此单列前期指标。
全部请求在10800−300秒前到达；300秒内启动率分母是全部请求。P95和均值包含全部请求，未启动请求按H−arrival记录右删失下界，不剔除未完成/未启动请求。

放置策略与迁移策略做笛卡尔积，绝不把first-fit/best-fit称为迁移算法：

- 放置 `first_fit`：按server_id选首个可容纳主机；`best_fit`：最小放置后GPU余量，CPU余量和server_id依次破同分。
- 迁移 `none`：无整理，FIFO回填等待资源释放。
- 迁移 `one_move`：当前碎片阻塞时，仅找可直接完成腾挪的一次迁移，按卡秒成本排序。
- 迁移 `cheap_multi`：从当前状态枚举最多6个Pod的目标机疏散子集，目标机外的目的地用best-fit分配，选所找到方案中GPU暂停卡秒最低者。
- 迁移 `first_multi`：相同6步预算，按主机/子集枚举顺序选择第一个合法方案，代表简单整理。

这些都是在线启发式；有限搜索失败只记未找到，不证明不可整理。cheap_multi不宣称全局最优，也没有未来工作负载输入。
engine.py 不读取任何文件；public_observation显式白名单只包含当前主机、Pod资源与已到达等待请求，不给finish、残余服务时长、未来请求、donor来源或任何见证。放置函数也只检查当前空闲资源。评测者独立从当前初态搜索见证，策略每次从实时可见状态重新计算，绝不加载答案。
接入外部Agent时也必须使用该观察接口并隔离评测文件；当前Python函数边界不是操作系统级安全隔离。

## 筛选与偏差控制

最终固定网格：3个种子20261107/8/9 × 间隔10/16秒，加20261107的40秒轻负载，共7组28实例。网格中的所有结果均保留。
方向中性的公共阈值：8基线之间300秒启动率范围≥5个百分点，或P95范围≥120秒；不要求某算法胜出。
S1另要求低碎片（8卡不可用空闲≤20%），且两种无迁移放置之间也达到上述任一差异，以检验预防作用。
S2/S4另要求同一受阻8卡需求有多个合法迁移方案，已找到成本跨度≥20卡秒；S3另要求4卡空闲不可用比例≥90%、8卡单步不可能且存在合法多步方案。
S2–S4还要求低碎片对照或同放置下迁移相对无迁移的300秒启动率提高≥3个百分点，以排除纯容量不足而无整理收益的候选。
选取固定网格顺序中**第一个四类同时通过的组**，不取某基线收益最大组。其它已通过普通/困难实例仍在candidates中；未通过者也保留。SCREENING.md和candidate_registry.json逐项给出原因。
早期校准曾测20261107/16秒组：S1未达到预防差异阈值，S2–S4通过；20261108的纯贪心S1构造失败，随后增加合法整数约束求解后构造成功。见calibration_history.json；没有降低阈值以让该候选通过。
这是公开开发/诊断套件，已用基线筛选，不能冒称独立隐藏测试集或对新策略泛化的统计证明。后续正式排行应冻结策略后另留未筛选种子，并做迁移成本敏感性分析。

## 校验与复现

```powershell
python -m pip install -r requirements.txt
python test_engine.py
python verify_suite.py --root . --raw-source 'E:\\cluster-trace-gpu-v2026\\data' --report verification_report.json
python reproduce.py --instance S3_selected --output 'D:\\一般文件\\benchmark任务\\S3_reproduced'
python reproduce.py --output 'D:\\一般文件\\benchmark任务\\四类场景_reproduced'
python reproduce.py --raw-source 'E:\\cluster-trace-gpu-v2026\\data' --output 'D:\\一般文件\\benchmark任务\\四类场景_from_raw'
```

默认输出路径必须不存在，不递归删除、不覆盖旧文件。完整复现可直接用冻结source_audit；--raw-source则重新扫描三个原始Parquet。依赖同版本可比较Parquet字节；跨版本优先比较逻辑表及指标。
独立verify_suite.py不调用仿真器State或simulate来验证日志：重算每一步资源、目标机预释放容纳、暂停、服务完成、P95/启动率/有效GPU积分及迁移成本，同时核对原始需求和donor变换。manifest.json覆盖全部静态交付文件SHA256；verification_report.json与reproducibility_report.json为生成后校验回执，故不参与自引用哈希。
每个实例另提供verify_instance.py和reproduce_instance.py。所有脚本只在本地执行，不依赖云端。

## 明确未覆盖的约束

服务器没有可用显存容量字段，gpu_mem_request_trace仅原样审计，不能声称已验证显存。未建模主机内存、NUMA、拓扑、带宽、gang、多Pod依赖或迁移失败；优先级/任务类型保留元数据而不作为优先调度规则。
本套件只覆盖一个H20集群、整数独占卡和可迁移假设下的调度问题。真实工作负载需求未改，时序与运行机制是明确、可复现的合成模型。
'''
    (root/'README.md').write_text(text,encoding='utf-8')
    (root/'requirements.txt').write_text('numpy==1.24.3\npandas==2.0.3\npyarrow==11.0.0\nscipy==1.11.1\n',encoding='utf-8')
    dump(root/'calibration_history.json',{'early_run':'work/run_v2','same_thresholds_as_final':True,
      'observations':[{'seed':20261107,'interval_sec':16,'S1':'failed prevention difference','S2':'passed','S3':'passed','S4':'passed'},
      {'seed':20261108,'interval_sec':16,'S1':'greedy construction failed; not a proof of infeasibility; integer constraint fallback added'}],
      'final_grid_adaptation':'expanded to 10/16 second pressures across three seeds plus 40-second light boundary; all final results retained',
      'engine_accounting_fix':'in-flight migrations charged at start; independent accrued pause accounting; no selected trajectory in-flight at horizon'})
    excluded={'manifest.json','verification_report.json','reproducibility_report.json','deployment_receipt.json'}
    hashes={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts}
    dump(root/'manifest.json',{'algorithm':'SHA-256','sha256':hashes})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);finalize(p.parse_args().root)
