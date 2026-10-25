"use strict";

const $ = id => document.getElementById(id);
const esc = text => String(text ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (x, digits = 3) => x === null || x === undefined ? "—" : Number(x).toLocaleString("zh-CN", {maximumFractionDigits: digits});

const statusNames = {idle:"尚未运行", starting:"正在准备", running:"运行中", stopping:"正在停止", completed:"实验结束", cancelled:"已停止", failed:"运行失败"};
const statusTone = {idle:"neutral", starting:"info", running:"active", stopping:"warn", completed:"success", cancelled:"warn", failed:"danger"};
const taskNames = {not_arrived:"尚未到达", waiting:"等待中", running:"运行中", completed:"已完成", rejected:"永久不可满足"};
const taskTone = {not_arrived:"neutral", waiting:"warn", running:"active", completed:"success", rejected:"danger"};
const eventNames = {reset:"初始化", arrival:"任务到达", start:"开始运行", complete:"完成释放", reject:"拒绝", drained:"全部处理结束", cancelled:"停止", max_sim_time:"达到时间上限", blocked:"无法继续"};
const eventTone = {reset:"neutral", arrival:"info", start:"active", complete:"success", reject:"danger", drained:"success", cancelled:"warn", max_sim_time:"warn", blocked:"warn"};
const durationNames = {
  job_median_duration: "执行汇总中位数",
  observed_presence_span: "观测出现时段",
  presence_remaining: "预热剩余时长",
  ambiguous_longest_span_proxy: "多段 · 最长段代理",
  single_matching_span: "单条匹配"
};
const idleCaption = "提供负载按理想运行区间积分，可能超过 100%；资源预留按实际分配/释放事件积分。两者都不是 SM 或 CPU 执行利用率。";

let nodesUpload = null, tasksUpload = null, lastState = null, busy = false;

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败");
  return result;
}

function message(text, error = false) {
  const el = $("message");
  el.textContent = text;
  el.className = "message" + (error ? " error" : "");
}

async function upload(kind) {
  const file = $(kind + "-file").files[0];
  if (!file) return;
  busy = true;
  $("run").disabled = true;
  message("正在导入并校验 " + file.name + "…");
  try {
    if (file.size > 10 * 1024 * 1024) throw new Error("文件不能超过 10 MiB");
    const content = await file.text();
    const result = await api("/api/upload", {kind, filename:file.name, content});
    if (kind === "nodes") nodesUpload = result.upload_id; else tasksUpload = result.upload_id;
    if (kind === "tasks") {
      const data = file.name.toLowerCase().endsWith(".json") ? JSON.parse(content.replace(/^\uFEFF/, "")) : {};
      const window = data.measurement_window_sec || [0, 120];
      $("measurement-start").value = window[0];
      $("measurement").value = window[1] - window[0];
    }
    $(kind + "-source").textContent = (kind === "nodes" ? "服务器：" : "任务：") + file.name + " · " + result.record_count + " 条记录";
    message("导入成功。点击“开始实验”使用这份数据。");
  } catch (error) {
    message("导入失败：" + error.message, true);
  } finally {
    busy = false;
    render(lastState || {status:"idle"});
  }
}

$("nodes-file").addEventListener("change", () => upload("nodes"));
$("tasks-file").addEventListener("change", () => upload("tasks"));

$("defaults").onclick = () => {
  nodesUpload = tasksUpload = null;
  $("measurement-start").value = 0;
  $("measurement").value = 120;
  $("nodes-file").value = $("tasks-file").value = "";
  $("nodes-source").textContent = "内置 20 台 H20";
  $("tasks-source").textContent = "内置场景 A，50 个 Pod";
  message("已选择内置数据。");
};

$("run").onclick = async () => {
  busy = true;
  $("run").disabled = true;
  message("正在启动…");
  try {
    await api("/api/run", {
      nodes_upload_id:nodesUpload,
      tasks_upload_id:tasksUpload,
      policy:$("policy").value,
      clock:$("clock").value,
      queue_policy:$("queue-policy").value,
      seed:Number($("seed").value),
      measurement_start_seconds:Number($("measurement-start").value),
      measurement_seconds:Number($("measurement").value),
      max_sim_time:Number($("max-time").value),
      cpu_background_fraction:Number($("background").value)
    });
    message("实验已启动。资源状态每 0.5 秒刷新。");
    await poll();
    if (window.matchMedia("(max-width: 900px)").matches) {
      document.querySelector("main").scrollIntoView({behavior:"smooth", block:"start"});
    }
  } catch (error) {
    message("启动失败：" + error.message, true);
  } finally {
    busy = false;
    render(lastState || {status:"idle"});
  }
};

$("stop").onclick = async () => {
  try {
    await api("/api/stop", {});
    message("已请求停止，将保留未完成任务状态。");
    await poll();
  } catch (error) {
    message(error.message, true);
  }
};

$("task-filter").onchange = () => renderTasks(lastState?.tasks || []);

function chart(id, history, series, yMax, yLabel, currentTime) {
  const width = 640, height = 220, left = 54, right = 18, top = 18, bottom = 38;
  const xMax = Math.max(1, currentTime || 0, ...history.map(r => r.time_sec));
  const x = t => left + (width - left - right) * t / xMax;
  const y = v => height - bottom - (height - top - bottom) * Math.min(v, yMax) / yMax;
  let content = "";
  for (let i = 0; i <= 4; i++) {
    const value = yMax * i / 4;
    const yy = y(value);
    content += `<line x1="${left}" y1="${yy}" x2="${width - right}" y2="${yy}" stroke="#e2e8f0" stroke-dasharray="4 5"/>`;
    content += `<text x="${left - 10}" y="${yy + 4}" text-anchor="end">${fmt(value, value < 10 ? 1 : 0)}</text>`;
    const tick = xMax * i / 4;
    content += `<text x="${x(tick)}" y="${height - bottom + 20}" text-anchor="middle">${fmt(tick, tick < 10 ? 2 : 1)}</text>`;
  }
  content += `<text x="${left}" y="12">${esc(yLabel)}</text>`;
  content += `<text x="${width / 2}" y="${height - 4}" text-anchor="middle">模拟时间（秒）</text>`;
  for (const item of series) {
    if (!history.length) continue;
    let d = `M ${x(0)} ${y(0)}`;
    for (const row of history) d += ` H ${x(row.time_sec)} V ${y(item.value(row))}`;
    d += ` H ${x(xMax)}`;
    if (item.fill) content += `<path d="${d} L ${x(xMax)} ${y(0)} L ${x(0)} ${y(0)} Z" fill="${item.color}" opacity=".07"/>`;
    content += `<path d="${d}" fill="none" stroke="${item.color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>`;
    const last = item.value(history[history.length - 1]);
    content += `<circle cx="${x(history[history.length - 1].time_sec)}" cy="${y(last)}" r="3.2" fill="${item.color}"/>`;
  }
  $(id).innerHTML = content;
}

function usageCell(free, total) {
  const t = Number(total) || 0, f = Number(free) || 0;
  const used = Math.max(0, t - f);
  const pct = t ? Math.min(100, used / t * 100) : 0;
  const tone = pct >= 95 ? "danger" : pct >= 70 ? "warn" : "ok";
  return `<div class="resource-cell" title="已占用 ${fmt(pct, 1)}%"><span><b>${fmt(f)}</b> / ${fmt(t)}</span><div class="bar"><i class="${tone}" style="width:${pct}%"></i></div></div>`;
}

function renderTasks(tasks) {
  const filter = $("task-filter").value;
  const allFiltered = tasks.filter(t => filter === "all" || t.status === filter);
  const filtered = allFiltered.slice(0, 200);
  $("tasks-body").innerHTML = filtered.length ? filtered.map(t => {
    const tone = taskTone[t.status] || "neutral";
    const duration = durationNames[t.duration_match_status] || "输入服务时间";
    return `<tr>
      <td class="id" title="${esc(t.pod_id)}">${esc(t.pod_id)}</td>
      <td>${fmt(t.gpu_request)}</td>
      <td>${fmt(t.cpu_request_cores)}</td>
      <td>${fmt(t.submit_time_sec)}</td>
      <td title="${fmt(t.service_sec / 3600, 3)} 小时">${fmt(t.service_sec, 3)}</td>
      <td>${fmt(t.start_time_sec)}</td>
      <td>${fmt(t.finish_time_sec)}</td>
      <td>${fmt(t.wait_sec)}</td>
      <td title="${esc(t.rejection_reason || "")}"><span class="pill ${tone}">${taskNames[t.status] || esc(t.status)}</span></td>
      <td><span class="duration-tag">${duration}</span></td>
    </tr>`;
  }).join("") : `<tr><td colspan="10" class="empty-cell">当前筛选条件下没有任务。</td></tr>`;
  if (allFiltered.length > 200) {
    $("tasks-body").insertAdjacentHTML("afterbegin", `<tr><td colspan="10" class="empty-cell">当前筛选共 ${allFiltered.length} 条，页面显示前 200 条；完整任务请下载任务明细。</td></tr>`);
  }
}

function render(state) {
  const previousStatus = lastState?.status;
  lastState = state;
  const active = ["starting", "running", "stopping"].includes(state.status);
  $("run").disabled = active || busy;
  $("stop").disabled = !active;
  $("status").textContent = statusNames[state.status] || state.status;
  $("status").className = "status " + (statusTone[state.status] || "neutral");
  document.body.dataset.status = state.status;

  if (state.error) message("运行失败：" + state.error, true);
  if (["completed", "cancelled"].includes(state.status) && ["starting", "running", "stopping"].includes(previousStatus)) {
    message(state.status === "completed" ? "实验已结束，结果已保存，可在下方下载。" : "实验已停止，未完成任务状态已保存。");
  }

  if (!state.summary) {
    ["completed", "running", "waiting", "rejected", "offered-gpu", "offered-cpu", "reserved-gpu", "reserved-cpu", "measurement-window", "gpu-current", "cpu-current", "queue-current"].forEach(id => {
      if ($(id)) $(id).textContent = "—";
    });
    $("progress-fill").style.width = "0%";
    $("progress-text").textContent = "等待实验启动";
    $("capacity-label").textContent = "等待实验启动";
    $("metric-caption").textContent = idleCaption;
    chart("resource-chart", [], [], 100, "预留率（%）", 0);
    chart("queue-chart", [], [], 1, "等待任务（Pod）", 0);
    $("downloads").hidden = true;
    return;
  }

  const s = state.summary;
  const donePct = s.task_count ? Math.round(s.completed / s.task_count * 100) : 0;
  $("completed").textContent = `${fmt(s.completed, 0)} / ${fmt(s.task_count, 0)}`;
  $("running").textContent = fmt(s.running, 0);
  $("waiting").textContent = fmt(s.waiting, 0);
  $("rejected").textContent = fmt(s.rejected, 0);
  $("progress-fill").style.width = donePct + "%";
  $("progress-text").textContent = `${donePct}% 完成 · 未完成 ${fmt(s.unfinished, 0)}`;

  $("offered-gpu").textContent = fmt(s.offered_gpu_ratio * 100, 2) + "%";
  $("offered-cpu").textContent = fmt(s.offered_cpu_ratio * 100, 2) + "%";
  $("reserved-gpu").textContent = fmt(s.reserved_gpu_ratio * 100, 2) + "%";
  $("reserved-cpu").textContent = fmt(s.reserved_cpu_ratio * 100, 2) + "%";
  $("measurement-window").textContent = `测量窗 ${fmt(s.measurement_window_sec[0], 0)}–${fmt(s.measurement_window_sec[1], 0)} s`;
  $("gpu-current").textContent = fmt(s.reserved_gpu_ratio * 100, 1) + "%";
  $("cpu-current").textContent = fmt(s.reserved_cpu_ratio * 100, 1) + "%";
  $("queue-current").textContent = fmt(s.waiting, 0);

  $("run-label").textContent = `${s.node_count} 台服务器 · ${fmt(s.gpu_capacity)} GPU · ${fmt(s.cpu_capacity_cores)} 核 CPU · ${s.clock === "realtime" ? "真实时间节奏" : "快速模式"}${s.end_reason ? " · " + (eventNames[s.end_reason] || s.end_reason) + " · 未完成 " + s.unfinished : ""}`;
  $("clock-label").textContent = `模拟 ${fmt(s.sim_time_sec, 2)} s / 墙钟 ${fmt(s.wall_time_sec, 2)} s`;
  $("metric-caption").textContent = `提供负载是“提交后立即运行”的理想需求，可能超过 100%；资源预留是实际分配随时间的积分。${active ? "运行中的预留率按已观测时段计算。" : ""} 这些都不是执行利用率。`;
  $("capacity-label").textContent = `${s.node_count} 台 · 每节点独立核算 GPU 和 CPU`;

  $("nodes-body").innerHTML = state.nodes.map(n => `<tr>
    <td class="id" title="${esc(n.server_id)}">${esc(n.server_id)}</td>
    <td><span class="model-tag">${esc(n.gpu_type)}</span></td>
    <td>${usageCell(n.free_gpu, n.gpu_capacity)}</td>
    <td>${usageCell(n.free_cpu_cores, n.cpu_capacity_cores)}</td>
    <td>${fmt(n.cpu_background_cores)}</td>
  </tr>`).join("");

  renderTasks(state.tasks);
  $("events-body").innerHTML = state.events.slice(-80).reverse().map(e => `<tr>
    <td>${fmt(e.time_sec, 4)}</td>
    <td><span class="pill ${eventTone[e.event] || "neutral"}">${eventNames[e.event] || esc(e.event)}</span></td>
    <td class="id" title="${esc(e.pod_id || "")}">${esc(e.pod_id || "—")}</td>
    <td>${e.waiting}</td>
    <td>${e.running}</td>
    <td>${e.completed}</td>
  </tr>`).join("");

  chart("resource-chart", state.history, [
    {color:"#2563eb", fill:true, value:r => r.gpu_reserved / s.gpu_capacity * 100},
    {color:"#0f9f8f", fill:true, value:r => r.cpu_reserved_cores / s.cpu_capacity_cores * 100}
  ], 100, "预留率（%）", s.sim_time_sec);
  chart("queue-chart", state.history, [
    {color:"#f59e0b", fill:true, value:r => r.waiting}
  ], Math.max(1, ...state.history.map(r => r.waiting)), "等待任务（Pod）", s.sim_time_sec);

  const downloadable = ["completed", "cancelled"].includes(state.status);
  $("downloads").hidden = !downloadable;
  if (downloadable) {
    $("download-links").innerHTML = [
      ["summary.json", "运行汇总"],
      ["tasks.csv", "任务明细"],
      ["resource_history.csv", "资源曲线"],
      ["events.csv", "事件日志"],
      ["nodes_final.csv", "最终节点状态"],
      ["run_manifest.json", "输入与配置"]
    ].map(([file, label]) => `<a href="/api/download/${state.run_id}/${file}" download>${label}</a>`).join("");
    $("output-folder").textContent = "保存目录：" + state.output_folder;
  }
}

async function poll() {
  try {
    render(await api("/api/state"));
  } catch (error) {
    $("status").textContent = "连接中断：请检查启动终端";
    $("status").className = "status danger";
  }
}

api("/api/config").then(c => {
  if (!c.defaults_available) message("内置数据不存在，请分别上传服务器表与任务表。", true);
}).catch(() => message("无法连接本地服务。", true));

poll();
setInterval(poll, 500);
