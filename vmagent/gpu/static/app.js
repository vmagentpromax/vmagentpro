"use strict";
const $ = id => document.getElementById(id);
const esc = text => String(text ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (x, digits=3) => x === null || x === undefined ? "—" : Number(x).toLocaleString("zh-CN", {maximumFractionDigits:digits});
const statusNames = {idle:"尚未运行",starting:"正在准备",running:"运行中",stopping:"正在停止",completed:"实验结束",cancelled:"已停止",failed:"运行失败"};
const taskNames = {not_arrived:"尚未到达",waiting:"等待中",running:"运行中",completed:"已完成",rejected:"永久不可满足"};
const eventNames = {reset:"初始化",arrival:"任务到达",start:"开始运行",complete:"完成释放",reject:"拒绝",drained:"全部处理结束",cancelled:"停止",max_sim_time:"达到时间上限",blocked:"无法继续"};
let nodesUpload = null, tasksUpload = null, lastState = null, busy = false;
async function api(path, body) {
 const response = await fetch(path, body === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
 const result = await response.json();
 if (!response.ok) throw new Error(result.error || "请求失败");
 return result;
}
function message(text, error=false) { $("message").textContent=text; $("message").className=error?"error":""; }
async function upload(kind) {
 const file = $(kind+"-file").files[0]; if(!file) return;
 busy=true; $("run").disabled=true; message("正在导入并校验 "+file.name+"…");
 try {
  if (file.size > 10*1024*1024) throw new Error("文件不能超过 10 MiB");
  const content = await file.text();
  const result = await api("/api/upload", {kind,filename:file.name,content});
  if(kind==="nodes") nodesUpload=result.upload_id; else tasksUpload=result.upload_id;
  if(kind==="tasks") {
   const data=file.name.toLowerCase().endsWith(".json")?JSON.parse(content.replace(/^\uFEFF/,"")):{};
   const window=data.measurement_window_sec || [0,120];
   $("measurement-start").value=window[0];$("measurement").value=window[1]-window[0];
  }
  $(kind+"-source").textContent=(kind==="nodes"?"服务器：":"任务：")+file.name+" · "+result.record_count+" 条记录";
  message("导入成功。点击“开始实验”使用这份数据。");
 } catch(error) { message("导入失败："+error.message,true); }
 finally { busy=false; render(lastState || {status:"idle"}); }
}
$("nodes-file").addEventListener("change",()=>upload("nodes"));
$("tasks-file").addEventListener("change",()=>upload("tasks"));
$("defaults").onclick=()=>{nodesUpload=tasksUpload=null;$("measurement-start").value=0;$("measurement").value=120;$("nodes-file").value=$("tasks-file").value="";$("nodes-source").textContent="服务器：内置 20 台 H20";$("tasks-source").textContent="任务：内置场景 A，50 个 Pod";message("已选择内置数据。");};
$("run").onclick=async()=>{
 busy=true;$("run").disabled=true;message("正在启动…");
 try {
  await api("/api/run",{nodes_upload_id:nodesUpload,tasks_upload_id:tasksUpload,policy:$("policy").value,clock:$("clock").value,queue_policy:$("queue-policy").value,seed:Number($("seed").value),measurement_start_seconds:Number($("measurement-start").value),measurement_seconds:Number($("measurement").value),max_sim_time:Number($("max-time").value),cpu_background_fraction:Number($("background").value)});
  message("实验已启动。资源状态每 0.5 秒刷新。");await poll();
 } catch(error) {message("启动失败："+error.message,true);}
 finally {busy=false;render(lastState || {status:"idle"});}
};
$("stop").onclick=async()=>{try{await api("/api/stop",{});message("已请求停止，将保留未完成任务状态。");await poll();}catch(error){message(error.message,true);}};
$("task-filter").onchange=()=>renderTasks(lastState?.tasks || []);
function chart(id, history, series, yMax, yLabel, currentTime) {
 const width=640,height=220,left=53,right=16,top=15,bottom=38;
 const xMax=Math.max(1,currentTime || 0,...history.map(r=>r.time_sec));
 const x=t=>left+(width-left-right)*t/xMax, y=v=>height-bottom-(height-top-bottom)*v/yMax;
 let content="";
 for(let i=0;i<=4;i++) {
  const v=yMax*i/4, yy=y(v); content+=`<line x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}" stroke="#e3e9f0"/><text x="${left-8}" y="${yy+4}" text-anchor="end">${fmt(v,1)}</text>`;
  const t=xMax*i/4; content+=`<text x="${x(t)}" y="${height-bottom+18}" text-anchor="middle">${fmt(t,1)}</text>`;
 }
 content+=`<text x="${left}" y="11">${esc(yLabel)}</text><text x="${width/2}" y="${height-3}" text-anchor="middle">模拟时间（秒）</text>`;
 for(const s of series) {
  let d=`M ${x(0)} ${y(0)}`;
  for(const row of history) d+=` H ${x(row.time_sec)} V ${y(s.value(row))}`;
  if(history.length) d+=` H ${x(xMax)}`;
  content+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
 }
 $(id).innerHTML=content;
}
function renderTasks(tasks) {
 const filter=$("task-filter").value;
 const allFiltered=tasks.filter(t=>filter==="all" || t.status===filter);
 const filtered=allFiltered.slice(0,200);
 $("tasks-body").innerHTML=filtered.length ? filtered.map(t=>`<tr><td class="id" title="${esc(t.pod_id)}">${esc(t.pod_id)}</td><td>${fmt(t.gpu_request)}</td><td>${fmt(t.cpu_request_cores)}</td><td>${fmt(t.submit_time_sec)}</td><td>${fmt(t.service_sec,6)}</td><td>${fmt(t.start_time_sec)}</td><td>${fmt(t.finish_time_sec)}</td><td>${fmt(t.wait_sec)}</td><td title="${esc(t.rejection_reason || "")}"><span class="pill">${taskNames[t.status] || esc(t.status)}</span></td><td>${t.duration_match_status==="ambiguous_longest_span_proxy"?"多段：最长段代理":t.duration_match_status==="single_matching_span"?"单条匹配":"输入服务时间"}</td></tr>`).join("") : `<tr><td colspan="10" class="muted">当前筛选条件下没有任务。</td></tr>`;
 if(allFiltered.length>200) $("tasks-body").insertAdjacentHTML("afterbegin",`<tr><td colspan="10" class="muted">当前筛选共 ${allFiltered.length} 条，页面显示前 200 条；完整任务请下载任务明细。</td></tr>`);
}
function render(state) {
 const previousStatus=lastState?.status;
 lastState=state;
 const active=["starting","running","stopping"].includes(state.status);
 $("run").disabled=active || busy;$("stop").disabled=!active;
 $("status").textContent=statusNames[state.status] || state.status;
 if(state.error) message("运行失败："+state.error,true);
 if(["completed","cancelled"].includes(state.status) && ["starting","running","stopping"].includes(previousStatus)) message(state.status==="completed"?"实验已结束，结果已保存，可在下方下载。":"实验已停止，未完成任务状态已保存。");
 if(!state.summary) {chart("resource-chart",[],[],100,"预留率（%）",0);chart("queue-chart",[],[],1,"等待任务（Pod）",0);return;}
 const s=state.summary;
 $("completed").textContent=fmt(s.completed,0)+" / "+fmt(s.task_count,0);$("running").textContent=s.running;$("waiting").textContent=s.waiting;$("rejected").textContent=s.rejected;
 $("run-label").textContent=`${s.node_count} 台服务器 · ${fmt(s.gpu_capacity)} GPU · ${fmt(s.cpu_capacity_cores)} 核 CPU · ${s.clock==="realtime"?"真实时间节奏":"快速模式"}${s.end_reason?" · "+(eventNames[s.end_reason] || s.end_reason)+" · 未完成 "+s.unfinished:""}`;
 $("clock-label").textContent=`模拟 ${fmt(s.sim_time_sec,2)} s / 墙钟 ${fmt(s.wall_time_sec,2)} s`;
 $("metric-caption").textContent=`测量窗 ${s.measurement_window_sec.join("～")} s；提供负载 GPU ${fmt(s.offered_gpu_ratio*100,2)}% / CPU ${fmt(s.offered_cpu_ratio*100,2)}%；资源预留 GPU ${fmt(s.reserved_gpu_ratio*100,2)}% / CPU ${fmt(s.reserved_cpu_ratio*100,2)}%。${active?"运行中的预留率按已观测时段计算。":""} 这些不是执行利用率。`;
 $("capacity-label").textContent=`${s.node_count} 台 · 每个节点独立核算 GPU 和 CPU`;
 $("nodes-body").innerHTML=state.nodes.map(n=>`<tr><td class="id">${esc(n.server_id)}</td><td>${esc(n.gpu_type)}</td><td>${fmt(n.free_gpu)} / ${fmt(n.gpu_capacity)}</td><td>${fmt(n.free_cpu_cores)} / ${fmt(n.cpu_capacity_cores)}</td><td>${fmt(n.cpu_background_cores)}</td></tr>`).join("");
 renderTasks(state.tasks);
 $("events-body").innerHTML=state.events.slice(-80).reverse().map(e=>`<tr><td>${fmt(e.time_sec,4)}</td><td>${eventNames[e.event]||esc(e.event)}</td><td class="id">${esc(e.pod_id || "—")}</td><td>${e.waiting}</td><td>${e.running}</td><td>${e.completed}</td></tr>`).join("");
 chart("resource-chart",state.history,[{color:"#2064c4",value:r=>r.gpu_reserved/s.gpu_capacity*100},{color:"#52765d",value:r=>r.cpu_reserved_cores/s.cpu_capacity_cores*100}],100,"预留率（%）",s.sim_time_sec);
 chart("queue-chart",state.history,[{color:"#617184",value:r=>r.waiting}],Math.max(1,...state.history.map(r=>r.waiting)),"等待任务（Pod）",s.sim_time_sec);
 const downloadable=["completed","cancelled"].includes(state.status);$("downloads").hidden=!downloadable;
 if(downloadable) {
  $("download-links").innerHTML=[["summary.json","运行汇总"],["tasks.csv","任务明细"],["resource_history.csv","资源曲线"],["events.csv","事件日志"],["nodes_final.csv","最终节点状态"],["run_manifest.json","输入与配置"]].map(([file,label])=>`<a href="/api/download/${state.run_id}/${file}" download>${label}</a>`).join("");
  $("output-folder").textContent="保存目录："+state.output_folder;
 }
}
async function poll(){try{render(await api("/api/state"));}catch(error){$("status").textContent="连接中断：请检查启动终端";}}
api("/api/config").then(c=>{if(!c.defaults_available) message("内置数据不存在，请分别上传服务器表与任务表。",true);}).catch(()=>message("无法连接本地服务。",true));
poll();setInterval(poll,500);
