"""Write detailed, measured README.md for each generated scene."""
import csv,hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from build_scenarios import DEFAULT_OUT,read,DAY

ROOT=DEFAULT_OUT
MANIFEST=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
SCENES={
'P0_real_F8d':('F场景边界任务预热基准','以原F任务为基础，检验CPU高度竞争时的策略；将边界前已出现的任务先调度，再开始8天测量。',
 'F已有小时首见/执行摘要/出现跨度代理构造；1070个源边界presence_remaining任务在模拟t=0到达，并增加3600秒准备服务量；其余F到达时刻统一后移3600秒。',
 'HP GPU工作量占比很高，CPU offered>100%；不能用它单独评判LP牺牲是否合理。'),
'P1_steady':('持续高GPU竞争','在连续15天约105% GPU提供负载下，比较有无抢占及成本感知受害选择。',
 '从F中合格GPU模板按优先级和型号分层有放回抽样，15天内分层均匀提交；保留每条模板的资源、时长、类型和优先级。',
 '它是经过筛选的受控实验，不是阿里真实15天的到达分布；LP长任务占比被有意提高。'),
'P2_burst':('每6小时HP集中到达','与P1保持同一GPU模板工作量，单独观察HP突发时的抢占次数、LP损失和HP尾响应。',
 'P1同一随机抽样序列；HP到达时间从每6小时段压缩到前30分钟，LP仍均匀提交。',
 'HP集中会改变与测量窗边界的交集，因此提供负载需看实际清单，不能声称P1/P2精确相等。'),
'P3_overload':('LP持续过载','保持P2的HP设定，将LP目标强度从70%提高到95%，观察累计积压、反复抢占与长期公平性。',
 '沿用P2分层抽样与HP突发过程，仅增加LP模板数量，目标总GPU提供负载约130%。',
 '长期需求高于容量，稳定排队不可期待；固定截止完成率与排空后JCT应分别解释。'),
'P4_model_skew':('H20局部短缺','检验全池约105%时，H20严重拥挤、XPU-A相对空闲是否造成机型局部HP延迟。',
 '在P2方法上调整分机型抽样量：H20目标约150%、XPU-A约82.5%；任务本身机型与资源不改。',
 '不同型号只做资格约束；没有模型速度换算或跨型号替代。全池平均可能掩盖H20瓶颈。'),
'P5_cpu_contention':('CPU-only叠加竞争','在与P2完全相同GPU序列上追加CPU-only模板，检验CPU紧张时抢占GPU是否仍能保证HP。',
 'P2所有GPU任务与到达时间原样保留；额外抽样F中CPU-only联合模板，使CPU提供负载约105%。',
 'CPU-only的时长是原F出现跨度代理；不能把GPU延迟全归因于抢占受害选择。'),
'P6_real_H20_early15d':('早期H20连续窗口','作为执行摘要覆盖不足时的重建敏感性对照，观察时长代理如何影响负载和策略。',
 '集群9089…、day48–62，读取所有360个小时加前1小时；固定30台8卡H20，Pod去重后以首见小时内确定性位置提交。',
 '早期任务时长全部使用可见出现跨度，短任务被按整小时放大；不用于论文主结论。'),
'P7_real_H20_15d':('后期H20连续窗口','使用更可靠的执行摘要与真实HP/LP身份，作为不重采样的外部检查。',
 '集群9089…、day158–172，读取所有360个小时加前1小时；固定30台8卡H20。GPU服务时间以Pod+型号+GPU请求匹配的执行跨度中位数为代理。',
 '完整CPU-only流使总任务数很大，重建GPU提供负载约66%；它是现实分布检查，不是强抢占压力场景。'),
'P8_real_XPUD_15d':('后期XPU-D连续窗口','检验H20上的结论是否可直接迁移到另一型号和集群。',
 '集群2f96…、day132–146，读取所有360个小时加前1小时；固定30台8卡XPU-D，GPU时长按执行摘要中位数代理。',
 '其中1台服务器在day143有12小时cluster标签变化，型号/容量不变；固定池模型没有模拟成员迁移。'),
'P9_H20_gpu_capacity_pressure':('真实GPU序列的容量压力','在P7全部真实GPU任务与到达时间不变的前提下，只减少节点容量，形成约104%GPU负载。',
 '抽出P7全部12,869个GPU任务、按server_id排序取原30台中的前19台；不复制任务、不改GPU/CPU需求、时长、机型和优先级。',
 '剔除了CPU-only且缩小了节点池，故这是受控容量实验，不能叫原集群真实利用率。')}

def fnum(r,k): return float(r.get(k) or 0)
def pct(v): return f'{100*v:.2f}%'
def quantile(xs,q):
    if not xs:return 0
    xs.sort();return xs[max(0,math.ceil(len(xs)*q)-1)]
def source_style(name):
    if name=='P0_real_F8d':return 'F输入的严格仿真预热改造'
    if name.startswith(('P6_','P7_','P8_')):return '连续原始小时窗口重建'
    if name.startswith('P9_'):return '真实GPU序列，受控缩池'
    return '真实联合模板有放回重采样'

def methodology(name,m):
    """State exact provenance and simulation-time transformations per scene."""
    f_source='`runs/hetero8d_20261101/inputs/jobs_F.csv`（仓库内；原始 D 盘位置见该目录 README 第 6 节）'
    if name=='P0_real_F8d':
        return [f'- **原始天数：**阿里原始 `day=109–116`，连续8天；另读 `day=108/hour=23` 判断边界前已出现的 Pod。输入直接来自 F 的 {f_source}，不是新的原始扫描，也不是未修改的真实提交日志。',
            '- **取样和改动：**沿用 F 的30台节点、119,495个去重任务，不再次抽样。F 中1,070个 `presence_remaining` 边界任务在模拟0秒到达，服务时长各增加3,600秒准备量；F 中其余正时刻到达统一加3,600秒。测量 `[3600,694800)`；源任务资源请求、机型、优先级和正时刻任务之间的到达间距不改。',
            '- **到达时刻：**F 的窗口内任务最初按 `pod_id` 在源 day109–116 首见的小时内，用种子20261101生成均匀伪随机位置；本场景只整体平移这些时刻。源边界任务到达时刻是重建的模拟0秒，不是其真实提交时刻。',
            '- **执行时长：**F 的非边界 GPU 任务优先用执行摘要 `duration_hours` 中位数×3600；CPU-only 与未匹配任务用源 Pod 首末可见小时跨度 `(末小时−首小时+1)×3600`。边界任务原 `service_sec` 是窗口内剩余出现跨度代理，本场景再加3,600秒准备量；连续运行者在测量起点仍有原F估计的剩余服务，排队者会剩余更多。各来源实际数量见下方 `duration_match_status` 表。']
    if name.startswith(('P1_','P2_','P3_','P4_','P5_')):
        common=[f'- **原始天数：**任务模板来自 F 的阿里 `day=109–116`（8天）及其边界前 `day=108/hour=23`；实际模板筛选要求 F 的 `submit_time_sec>0`，因此不抽取那1,070个边界任务。不是从阿里连续15天直接回放。节点池沿用 F 的30台，不重采样节点。',
            f'- **模板过滤：**读取 {f_source}，GPU模板要求 HP/LP、`gpu_request>0`、`300≤service_sec≤86400`、`cpu_request_cores/gpu_request≤12`；LP另要求 `service_sec≥1800`。按 `(gpu_spec_public,priority_class)` 四个层分别抽样。',
            '- **有放回抽样：**对每层用 `random.Random(20261107 + 100×机型序号 + 优先级序号)`；机型序号 H20=0、XPU-A=1，优先级 HP=0、LP=1。每次从该层均匀抽一整行联合模板，累计 `gpu_request×service_sec`，达到目标 `比率×该型号GPU容量×15×86400` 后停止；最后一次可超过目标。保留该行的 GPU/CPU 请求、机型、优先级、类型、原服务时间，不把字段独立打乱。场景任务可重复使用同一个来源 Pod；`source_pod_id` 可回溯来源，任务 `pod_id` 改为场景内唯一 ID。',
            '- **合成到达：**抽样后的第 i 个任务（从0计）取 `t=(i+U_i)/N×(15×86400)`，`U_i` 为同层随机流的 `[0,1)` 均匀伪随机数；这让每层在15天内分层均匀铺开，并不是按来源 Pod 首见小时随机到达。相同层、相同种子和目标产生相同模板序列。测量从模拟第2天起，共14天；第1天任务参与预热。',
            '- **执行时长：**GPU任务直接继承抽中 F 行的 `service_sec`，没有按15天窗口重新估时，也没有按GPU型号换算速度。F 非边界 GPU 通常为 Pod+机型+GPU请求匹配的执行摘要 `duration_hours` 中位数×3600；`duration_match_status` 逐行说明。重复抽中的任务重复使用同一时长；该值是模拟服务时长代理，不是硬件SM运行时间。']
        if name=='P1_steady':
            common+=['- **本场景变量：**H20(80 GPU)与XPU-A(160 GPU)两型号的 HP 工作量目标均为容量的35%×15天，LP均为70%×15天；HP/LP都使用上述分层均匀到达，不做突发变换。']
        elif name=='P2_burst':
            common+=['- **本场景变量：**抽样模板和工作量与P1逐层相同。仅对HP到达应用 `t′=floor(t/21600)×21600 + (t mod 21600)/12`，即每6小时的HP到达压入该段前30分钟；LP时间不变。']
        elif name=='P3_overload':
            common+=['- **本场景变量：**HP目标和突发规则与P2相同；LP目标从每型号容量的70%×15天提高到95%×15天。LP因此继续有放回抽样更多模板，不是简单把原任务时间缩短。']
        elif name=='P4_model_skew':
            common+=['- **本场景变量：**沿用P2的HP突发规则和35%/70%优先级目标，但H20层目标再乘 `1.5/1.05`、XPU-A层乘 `0.825/1.05`；使H20约150%、XPU-A约82.5%总目标。每层重新按相同随机流抽到新目标，没有修改任一模板的资源值。']
        else:
            common+=['- **本场景变量：**P2的全部GPU任务及合成到达时刻原样保留。额外从F中 `gpu_request=0`、`16≤cpu_request_cores≤96`、`300≤service_sec≤86400`、原到达>0的CPU-only行有放回抽样；使用种子 `20261107+999`，累计 `cpu_request_cores×service_sec` 到 `(1.05−P2现有CPU提供负载)×5760×15×86400`，末次可超额。CPU-only 的到达也用上式15天分层均匀铺开，时长继承F的出现跨度代理。']
        common.append(f'- **来源重复量：**本场景 {m["tasks"]:,} 条任务、{m["unique_source_pods"]:,} 个不同来源 Pod；有放回抽样的重复记录是实验负载，不是原始轨迹中新出现的 Pod。')
        return common
    source=m.get('source_days',[158,172]);cluster='9089d80baea5f56a3c169801052daa1f' if name.startswith(('P6_','P7_','P9_')) else '2f9611355f71fe804da2606899ddfb36'
    common=[f'- **原始天数和文件：**Pod路径模式为 `D:/cluster-trace-gpu-v2026/asi_opensource_pod_hourly/day=<day>/hour=<hour>/part-000.parquet`，实际读取 `day={source[0]}–{source[1]}`（含端点，连续15天、360个小时）的全部小时，另读 `day={source[0]-1}/hour=23` 带入边界 Pod。节点来自相同日期的 `asi_opensource_server_hourly`；时长匹配读取 `asi_opensource_job_execution_summary/part-000.parquet`。来源集群 `{cluster}`。这是小时观测重建，不是精确提交/完成事件日志。',
        '- **节点和任务选择：**先选在上述15个每日00时快照里集群、型号、8卡和CPU规格稳定的服务器；按源首日00时各服务器 `min(HP,LP)` 的观测 `used_gpu_hours` 降序、server_id破同分，取前30台。读取所选服务器和集群的361个Pod小时分区，只保留GPU或CPU请求正值的记录；同一Pod去重成一个任务。对跨小时有不同请求/优先级/类型组合的Pod，取出现次数最多的联合形状，平手按字段排序；空节点也放不下的任务剔除。',
        '- **到达时刻：**非边界Pod取首次观测的相对小时 `h`，令 `seed=int(SHA256(pod_id+"20261107")前16位十六进制)`，再取 `U=random.Random(seed).random()`，设置 `submit_time_sec=(h+U)×3600`。这是每个Pod独立、可复现的小时内伪随机位置；不是真实提交时刻。前一小时已出现且窗口内仍可见的Pod设模拟0秒到达，不按小时内随机化。',
        '- **执行时长：**前一小时已出现者用 `service_sec=(末次相对小时+1)×3600`，即窗口内剩余可见跨度代理。其余Pod如匹配执行摘要，则按同一 `pod_id`、机型、GPU请求的正 `duration_hours` 中位数×3600；否则用 `(末次相对小时−首次相对小时+1)×3600` 可见跨度代理。P6早期窗口不使用执行摘要，全部走跨度代理；P7/P8按匹配情况分别处理，CPU-only通常走跨度代理。右边界仍可见者有截断风险；具体来源数量见下方标记表。']
    if name.startswith('P9_'):
        common[1]='- **受控修改：**从P7的重建任务表只保留全部GPU任务，完全保留P7的到达时刻、服务时长、请求、型号和优先级；删除CPU-only任务。P7的30台节点按server_id顺序取前19台，不重抽Pod，也不改任何GPU任务行。'
        common[2]='- **到达时刻：**逐行继承P7：非边界Pod在最早观测小时里按 `SHA256(pod_id+"20261107")` 派生种子确定位置，前一小时已出现者设模拟0秒。P9本身没有新的随机抽样或到达变换。'
        common[3]='- **执行时长：**逐行继承P7，不重新从执行摘要匹配。前一小时已出现者为剩余可见跨度代理，其他GPU任务以Pod+机型+GPU请求匹配的正执行跨度中位数×3600为主，缺失时用可见跨度代理；实际来源数量见标记表。'
    return common

def plain_sequence_explanation(name):
    """Explain P1–P5 in the same simple order as the actual generator."""
    if name not in ('P1_steady','P2_burst','P3_overload','P4_model_skew','P5_cpu_contention'):
        return []
    variants={
        'P1_steady':'HP 和 LP 都按上述方法大致均匀到达；每种机型的目标工作量分别是 HP 35%、LP 70%。这是后续对照的起点。',
        'P2_burst':'抽中的 GPU 任务与 P1 相同，GPU/CPU 需求和服务时长也相同。只把 HP 在每个 6 小时段内的到达时刻压到前 30 分钟；LP 到达不改。例如原来在该段第 4 小时到达，改为第 20 分钟到达。',
        'P3_overload':'保留 P2 的 HP 目标和每 6 小时突发规律，把每种机型 LP 的目标工作量从 70% 提到 95%，于是抽取更多 LP 模板。因为总条数变化，LP 重新铺开 15 天；它不是简单把 P2 的同一批 LP 到达时刻压短。',
        'P4_model_skew':'沿用 P2 的 HP 突发方法，但分机型改变抽样目标：H20 多抽、XPU-A 少抽，使目标 GPU 压力约为 H20 150%、XPU-A 82.5%。每条模板自身的型号、GPU/CPU 请求和时长不改；全池平均负载可能掩盖 H20 的局部拥挤。',
        'P5_cpu_contention':'P2 的所有 GPU 任务及其到达时刻逐条保留。另从 F 有放回抽取 31,305 条 CPU-only 任务模板，按同样的 15 天铺开方法设置新到达时刻，使总 CPU 需求约达 105%；CPU-only 的服务时长沿用 F 的可见跨度代理。'
    }
    return ['### 先用简单的话看懂任务序列', '',
        '这里的“序列”就是每个任务何时到达、需要多少 GPU/CPU、运行多久。P1–P5 不是把 F 的 8 天任务按原时间顺序复制到 15 天，也不是新的阿里真实 15 天提交日志。',
        '',
        '1. **先挑来源任务。**从 F 对应阿里 day 109–116 的 8 天任务表中，只保留合格的 HP/LP GPU 行：服务时长 5 分钟到 24 小时，CPU 核数与 GPU 张数比不超过 12；LP 还要求至少运行 30 分钟。窗口边界的 1,070 条任务不进入这个模板池。节点仍是 F 的 30 台。',
        '2. **再按机型和优先级分四组抽样。**H20-HP、H20-LP、XPU-A-HP、XPU-A-LP 各自抽。一抽就是原任务的**整行**：GPU 数、CPU 核数、型号、优先级、类型和服务时长一起保留。“有放回”表示抽过的源任务以后仍可再抽；持续抽到该组预设的 GPU×运行秒数工作量目标，最后一条可能略超目标。相同来源重复出现是合成任务，不是阿里原轨迹里真的来了多个新 Pod。',
        '3. **给抽中的任务安排新的 15 天到达时刻。**对每组 N 条任务，第 i 条（i 从 0 算）取 `t=(i+U_i)/N×15天`，其中 `U_i` 是固定种子的 0 到 1 之间随机数。可以理解成把 15 天切成 N 个小格，每格放这一组的一个任务，再在格内打散；因此整体大致均匀，**不是**沿用来源 Pod 首见小时，也不是每个任务各自在整段 15 天独立随机抽一个时刻。',
        '4. **时长不重新生成。**抽中哪条 F 行，就用它原来的 `service_sec`。同一来源被抽中两次，两条合成任务沿用同一个服务时长。GPU 任务的 F 时长通常来自执行摘要匹配后的中位数；它是模拟时长代理，不等于硬件实测的持续计算时间。',
        '5. **先预热再测量。**合成任务从模拟第 1 天开始到达；第 1 天正常调度，其未完成任务继续占资源。正式测量为第 2–15 天，共 14 天。窗口结束后停止新到达并继续排空。',
        '',f'**本场景在共同步骤上的改动：**{variants[name]}',
        '', '下面保留精确筛选条件、随机种子、抽样目标和时间公式，便于逐行复现。']

def write(name,m):
    folder=ROOT/name; nodes_file=ROOT/m.get('nodes_file','nodes.csv')
    nodes=read(nodes_file)
    gpu_cap=sum(fnum(n,'gpu_count') for n in nodes);cpu_cap=sum(fnum(n,'cpu_capacity_cores') for n in nodes)
    group=Counter();sizes=Counter();jobs=Counter();dur=Counter();origins=Counter()
    arrivals=[Counter() for _ in range(math.ceil(m['arrival_horizon_sec']/DAY))]
    day_gpu=[0.]*len(arrivals);day_cpu=[0.]*len(arrivals)
    services=defaultdict(list); n=0;sm_count=0;right=0;warm=0
    with (folder/'tasks.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            n+=1;g=fnum(r,'gpu_request');c=fnum(r,'cpu_request_cores');t=fnum(r,'submit_time_sec');s=fnum(r,'service_sec')
            priority=r.get('priority_class','Other');label=priority+' GPU' if g>0 else 'CPU-only'
            group[label]+=1
            if g>0:
                sizes[(r.get('gpu_spec_public',''),r.get('gpu_request',''))]+=1
                jobs[r.get('job_type_public','unknown')]+=1
                services[priority].append(s)
            dur[r.get('duration_match_status','unknown')]+=1
            if r.get('observed_sm_mean_not_used_by_simulator','')!='':sm_count+=1
            if str(r.get('right_boundary_observed','')).lower()=='true':right+=1
            if t==0:warm+=1
            di=min(len(arrivals)-1,int(t//DAY));arrivals[di][label]+=1
            for d in range(max(0,di),min(len(arrivals),int((t+s)//DAY)+1)):
                ov=max(0.,min(t+s,(d+1)*DAY)-max(t,d*DAY))
                day_gpu[d]+=g*ov;day_cpu[d]+=c*ov
    assert n==m['tasks']
    node_types=Counter((r['gpu_spec_public'],r['gpu_count'],r['cpu_capacity_cores']) for r in nodes)
    cfg=SCENES[name];measure=m['measurement_window_sec']
    lines=[f'# {name}：{cfg[0]}','',f'场景生成日期：2026-11-07；本说明修订：2026-11-09。场景类型：{source_style(name)}。',
        f'**实验目的：**{cfg[1]}','', '## 1. 数据来源与构造', '',cfg[2],
        '',f'- 输入：`{folder.as_posix()}/tasks.csv`；节点：`{nodes_file.as_posix()}`。',
        f'- 原始数据：`D:/cluster-trace-gpu-v2026`；详细程序：`C:/Users/trw/Desktop/论文/vmagentpro/experiments/preemption_20261107/`。',
        '- 阿里公开Pod表是按小时的观测，不是精确提交日志；执行摘要的时长是模拟服务时间代理。',
        f"- 节点来源集群：`{nodes[0]['cluster_id']}`；所有场景节点均从同一cluster选择。",
        f"- 任务CSV SHA256：`{m['sha256']}`；生成器种子及源文件哈希见根目录 `manifest.json`。",
        '', *plain_sequence_explanation(name),
        '', '### 原始天数、抽样、到达与执行时长', '',*methodology(name,m),
        '', '## 2. 节点池', '',f'- {len(nodes)}台；容量 **{gpu_cap:g} GPU / {cpu_cap:g} CPU核**。节点规格见下表。',
        '', '| GPU型号 | 单机GPU | 单机CPU核 | 台数 |', '|---|---:|---:|---:|']
    for (model,g,c),count in sorted(node_types.items()):lines.append(f'| {model} | {g} | {c} | {count} |')
    lines+=['','完整 server_id 清单：','','| 序号 | server_id | 型号 | GPU | CPU核 |','|---:|---|---|---:|---:|']
    for i,r in enumerate(nodes,1):lines.append(f"| {i} | `{r['server_id']}` | {r['gpu_spec_public']} | {r['gpu_count']} | {r['cpu_capacity_cores']} |")
    lines+=['','## 3. 时间、预热与任务', '',
        f"- 到达区间：[0,{m['arrival_horizon_sec']:,.0f})秒，共{m['arrival_horizon_sec']/DAY:g}天；测量窗口：`[{measure[0]:,.0f},{measure[1]:,.0f})`秒，共{(measure[1]-measure[0])/DAY:g}天。",
        f"- 最大仿真时刻：{m['max_sim_time_sec']:,.0f}秒（{m['max_sim_time_sec']/DAY:g}天）；到达结束后继续排空。",
        f"- {n:,}个唯一任务，GPU任务{m['gpu_tasks']:,}，CPU-only任务{n-m['gpu_tasks']:,}，模拟t=0到达{warm:,}。",
        '- 调度在空节点池从t=0启动；测量起点之前已经开始且尚未完成的任务，其实际资源预留计入窗口。源轨迹没有可直接导入的精确初始节点占用。',
        '', '| 类别 | 任务数 | 占总数 |', '|---|---:|---:|']
    for label,v in sorted(group.items()):lines.append(f'| {label} | {v:,} | {pct(v/n)} |')
    lines+=['','GPU任务请求形状：','','| 型号 | GPU请求 | 任务数 |','|---|---:|---:|']
    for (model,size),v in sizes.most_common(12):lines.append(f'| {model} | {size} | {v:,} |')
    lines+=['','GPU任务类型（前8）：','','| 类型 | 数量 |','|---|---:|']
    for job,v in jobs.most_common(8):lines.append(f'| {job} | {v:,} |')
    lines+=['','服务时间代理来源：','','| 标记 | 数量 |','|---|---:|']
    for label,v in dur.most_common():lines.append(f'| {label} | {v:,} |')
    lines+=['', '| GPU优先级 | 中位服务时长(h) | P95(h) | 最大(h) |', '|---|---:|---:|---:|']
    for pri,vals in sorted(services.items()):
        lines.append(f'| {pri} | {quantile(vals,.5)/3600:.2f} | {quantile(vals,.95)/3600:.2f} | {max(vals)/3600:.2f} |')
    lines+=['','## 4. 实际提供负载', '',
        '提供负载是任务若从提交时刻立刻运行，服务区间与测量窗口重叠的资源时间除以池容量×窗口；它可以超过100%，与设备SM利用率不同。',
        f"- GPU offered：**{pct(m['offered_gpu_ratio'])}**；CPU offered：**{pct(m['offered_cpu_ratio'])}**；背景CPU比例为0。",
        '- offered与仿真实际reserved分别代表请求压力和资源占用，抢占浪费会使两者不能直接解释成有效产出。',
        '', '| GPU型号 | 该型号GPU offered |', '|---|---:|']
    for model,val in sorted(m['per_model_offered_gpu_ratio'].items()):lines.append(f'| {model} | {pct(val)} |')
    lines+=['','| GPU优先级 | 对全池GPU容量的offered贡献 |','|---|---:|']
    for pri,val in sorted(m['priority_offered_gpu_ratio'].items()):
        if val:lines.append(f'| {pri} | {pct(val)} |')
    lines+=['','逐日到达数与**理想当日服务交集**（全池容量作分母；不等于当天实际预留）：','',
            '| 仿真日 | HP GPU到达 | LP GPU到达 | CPU-only到达 | GPU理想负载 | CPU理想负载 |',
            '|---:|---:|---:|---:|---:|---:|']
    for d,row in enumerate(arrivals):
        duration=min(DAY,m['arrival_horizon_sec']-d*DAY)
        lines.append(f"| {d+1} | {row['HP GPU']:,} | {row['LP GPU']:,} | {row['CPU-only']:,} | {pct(day_gpu[d]/(gpu_cap*duration))} | {pct(day_cpu[d]/(cpu_cap*duration))} |")
    if name=='P0_real_F8d':lines+=['','P0的第9个仿真日只有末尾1小时；该行的负载分母为1小时容量。']
    lines+=['','## 5. 适合比较的问题与限制','',cfg[1],cfg[3]]
    if name.startswith(('P1_','P2_','P3_','P4_','P5_')):
        lines+=['',f"- 本场景模板库筛选：`{MANIFEST['template_filter']}`；模板有放回，保留`source_pod_id`与`source_submit_time_sec`。",
                '- 场景强度和HP突发时刻为实验变量，不是来源Pod的真实到达律。']
    if name.startswith(('P6_','P7_','P8_')):
        qa=m.get('quality_audit',{})
        lines+=['',f"- 源天数：{m['source_days'][0]}–{m['source_days'][1]}；原始Pod分区读取{m['raw_partitions_read']}个（含前一小时）。",
                f"- 窗末最后小时仍可见的Pod数：{qa.get('right_boundary_tasks','待复核')}；右边界时长有截断风险。",
                f"- 相同Pod的GPU/CPU/型号/优先级组合在观测小时内有多个取值者：{qa.get('multiple_joint_shapes_pods','待复核')}；此处取联合作业形状出现次数最多的组合。",
                '- 原始SM均值只作来源旁证，未注入新策略的算力模型。']
        if name.startswith('P8_'):
            lines+=['- 逐小时库存复核：单台server在day143的12小时cluster标签变化，型号/容量不变；固定池未模拟成员迁移。']
    if name.startswith('P9_'):lines+=['- 与P7按GPU任务行完全相同，节点变为原30台中的前19台；因此不能与P7直接归因于抢占策略差异。']
    if name=='P0_real_F8d':lines+=['- 原F源边界任务先经过3600秒仿真预热，其他到达统一平移3600秒，随后测量完整8天；边界任务增加3600秒准备服务量，使不排队者在测量起点的剩余服务等于旧F输入。排队者可能剩余更多；这只是重建的窗口起始运行状态，不保证逐个复现阿里真实节点上的运行位置。',
        '- 本任务表已不再与旧F逐字节相同；旧P0输入和10组旧结果保存在根目录 `archive_pre_warmup_P0_20261107/`。详见准备工作目录的《十场景预热与窗口边界任务核查_20261107.md》。']
    lines+=['','## 6. 策略对比与指标', '',
        '本目录实验默认队列为backfill，测量窗口严格采用上述区间。原引擎的First/Best/Worst/Random和`legacy_spotgpu`沿用原取任务顺序；其余五个实验变体采用明确HP GPU优先搜索。`no_preempt`对`paper_cost`隔离抢占作用，`random_victim`隔离受害者选择，`paper_baseline`同时关闭共置/历史，`no_history`单独消融历史。下表只列实际完成的配置，跨原引擎与实验变体比较时须同时考虑队列时机差异。',
        '优先看HP GPU首次响应P95/P99与60秒违约率，再看LP GPU平均/P95 JCT、窗末完成率、窗口内浪费GPU·h和有效GPU工作率。CPU-only数量很大时，所有任务混合平均会掩盖GPU抢占。',
        '有效GPU工作率=测量窗口GPU占用时间减去该时间段最终丢失的进度，再除以容量×窗口；它不是SM执行利用率。检查点间隔为假设的3600秒；未建模检查点I/O、抢占启动/驱逐延迟、Standby与跨节点gang。']
    available=[]
    for policy in ('first_fit','best_fit','worst_fit','random','legacy_spotgpu',
                   'no_preempt','paper_cost','random_victim','paper_baseline','no_history'):
        out=ROOT/'outputs'/f'{name}-{policy}-s42-ck3600'
        if (out/'summary.json').exists() and (out/'gpu_priority_metrics.json').exists():
            available.append((policy,json.loads((out/'summary.json').read_text()),
                json.loads((out/'metrics.json').read_text()),
                json.loads((out/'gpu_priority_metrics.json').read_text())))
    if available:
        lines+=['','已完成的seed=42、checkpoint=3600秒首轮结果。同一场景的策略共享完全相同的任务和节点输入；真实窗口结果同样可作该重建场景内的配对比较，但小时内到达时间和部分服务时间仍为代理，单个窗口不代表总体排名。',
          '', '| 策略 | 完成/总任务 | 窗内GPU占用 | 窗内有效工作 | 浪费GPU·h | HP GPU P95响应(s) | LP GPU平均JCT(h) |',
          '|---|---:|---:|---:|---:|---:|---:|']
        for policy,s,extra,gp in available:
            hp=gp['HP_GPU']['response_p95_sec_started'];lp=gp['LP_GPU']['mean_jct_sec_completed']
            lines.append(f"| {policy} | {s['completed']:,}/{s['task_count']:,} | {pct(extra['allocated_gpu_ratio'])} | {pct(extra['useful_gpu_ratio_retrospective'])} | {extra['wasted_gpu_hours_window']:.2f} | {hp:.2f} | {lp/3600:.2f} |")
        lines+=['', '以上是仿真验证结果，完成时间与有效工作率的解释应结合HP保障及LP代价；单个随机种子不足以给出总体置信区间。']
    elif name=='P0_real_F8d':
        lines+=['','P0已改为窗口前预热，不能与旧F结果当作同一输入直接比较；旧版本见归档目录。']
    lines+=['','## 7. 运行与验收', '', 'PowerShell，在仓库根目录：', '', '```powershell',
        "Set-Location 'C:/Users/trw/Desktop/论文/vmagentpro'", "$env:PYTHONHASHSEED='0'",
        f"python experiments/preemption_20261107/run_suite.py --scenarios {name} --policies paper_cost paper_baseline --seeds 45",
        '```','',
        '使用未被占用的seed或新输出根目录；脚本拒绝覆盖既有结果。原CLI运行时使用本场景的节点CSV和`tasks.json`，并把max_sim_time设为上述值；对预热场景不要传`--measurement-seconds`，否则会把测量起点覆盖为0。',
        '验收：加载全部任务；唯一ID；每任务可在至少一台空节点满足GPU型号、GPU数及CPU核数；场景manifest中的SHA256匹配；同一场景不同策略复用完全相同输入。已生成根目录`validation.json`记录验收。',
        '', '相关文件：根目录 `manifest.json`、`comparison_all.csv`、`首轮验证结果.md`，以及 `outputs/` 中逐次 `run_manifest.json`、`summary.json`、`metrics.json`、`gpu_priority_metrics.json`。']
    (folder/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(name,'README lines',len(lines),flush=True)

def main():
    for name in sorted(MANIFEST['scenarios'],key=lambda x:int(x.split('_')[0][1:])):
        write(name,MANIFEST['scenarios'][name])
if __name__=='__main__':main()
