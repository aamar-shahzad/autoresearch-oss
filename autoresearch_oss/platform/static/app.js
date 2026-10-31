async function api(path, opts = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(opts.headers || {}),
  };
  const tenant = document.getElementById("tenantId")?.value?.trim();
  if (tenant) headers["X-Tenant-Id"] = tenant;
  const bearer = document.getElementById("apiBearer")?.value?.trim();
  if (bearer) headers.Authorization = `Bearer ${bearer}`;
  const r = await fetch(path, { headers, ...opts });
  const text = await r.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { raw: text };
  }
  if (!r.ok) {
    const detail = data.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
          : data.error || r.statusText || text;
    throw new Error(msg);
  }
  return data;
}

function wf() {
  return document.getElementById("workflow").value.trim();
}

function configPath() {
  return document.getElementById("configPath").value.trim();
}

function cfgBody() {
  const c = configPath();
  return c ? { config: c } : null;
}

function metric(m, k, d = 3) {
  const v = m && m[k];
  return v == null ? "—" : typeof v === "number" ? v.toFixed(d) : v;
}

function requireConfig() {
  const c = configPath();
  if (!c) throw new Error("Set config path");
  return c;
}

async function loadProduction() {
  const w = wf();
  const el = document.getElementById("production");
  const rb = document.getElementById("btnRollback");
  if (!w) {
    el.textContent = "Set workflow name";
    rb.disabled = true;
    return;
  }
  try {
    const m = await api(
      `/api/v1/models/production?workflow=${encodeURIComponent(w)}`,
    );
    el.textContent = JSON.stringify(m, null, 2);
    rb.disabled = false;
  } catch (e) {
    el.textContent = String(e.message);
    rb.disabled = true;
  }
}

async function loadModels() {
  const w = wf();
  const q = w ? `?workflow=${encodeURIComponent(w)}` : "";
  const rows = await api(`/api/v1/models${q}`);
  const tb = document.querySelector("#models tbody");
  tb.innerHTML = "";
  for (const m of rows) {
    const p = m.payload || {};
    const hm = p.holdout_metrics || {};
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${m.id}</code></td>
      <td>${p.stage || "?"}</td>
      <td>${metric(hm, "fix_rate")}</td>
      <td>${(p.recipe || "").slice(0, 40)}</td>
      <td></td>`;
    const btn = document.createElement("button");
    btn.className = "promote";
    btn.textContent = "Promote";
    btn.onclick = async () => {
      await api(`/api/v1/models/${encodeURIComponent(m.id)}/promote`, {
        method: "POST",
        body: "{}",
      });
      await refresh();
    };
    tr.lastElementChild.appendChild(btn);
    tb.appendChild(tr);
  }
}

async function loadExperiments() {
  const w = wf();
  const q = w ? `?workflow=${encodeURIComponent(w)}` : "";
  const rows = await api(`/api/v1/experiments${q}`);
  const tb = document.querySelector("#experiments tbody");
  tb.innerHTML = "";
  for (const e of rows.slice(-40)) {
    const p = e.payload || {};
    const m = p.metrics || {};
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${e.id}</code></td>
      <td>${p.phase || "?"}</td>
      <td>${metric(m, "fix_rate")}</td>
      <td>${metric(m, "worst_slice")}</td>
      <td>${(p.recipe || p.recipe_id || "").toString().slice(0, 36)}</td>`;
    tb.appendChild(tr);
  }
}

async function refresh() {
  await loadProduction();
  await loadModels();
  await loadExperiments();
}

let trainPollTimer = null;

async function pollTrainJob(jobId) {
  const el = document.getElementById("trainJob");
  const tick = async () => {
    const j = await api(`/api/v1/jobs/${jobId}`);
    el.textContent = `Train job ${jobId}: ${j.status}\n${(j.log || j.error || "").slice(-4000)}`;
    if (j.status === "running") {
      trainPollTimer = setTimeout(tick, 2000);
    } else {
      await refresh();
    }
  };
  await tick();
}

async function loadTemplates() {
  const sel = document.getElementById("initTemplate");
  const t = await api("/api/v1/templates");
  sel.innerHTML = "";
  for (const [k, label] of Object.entries(t)) {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = `${k} — ${label}`;
    sel.appendChild(o);
  }
}

document.getElementById("btnRefresh").onclick = () => refresh().catch(alert);

document.getElementById("btnRollback").onclick = async () => {
  const w = wf();
  if (!w) return;
  await api(`/api/v1/workflows/${encodeURIComponent(w)}/rollback`, {
    method: "POST",
    body: "{}",
  });
  await refresh();
};

document.getElementById("btnDoctor").onclick = async () => {
  const data = await api("/api/v1/doctor");
  document.getElementById("actionsOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnValidate").onclick = async () => {
  const c = requireConfig();
  const data = await api("/api/v1/validate", {
    method: "POST",
    body: JSON.stringify({ config: c }),
  });
  document.getElementById("actionsOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnEntitySync").onclick = async () => {
  const c = requireConfig();
  const data = await api("/api/v1/entities/sync", {
    method: "POST",
    body: JSON.stringify({ config: c, sync_cases: true }),
  });
  document.getElementById("actionsOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnEvalRun").onclick = async () => {
  const c = requireConfig();
  const data = await api("/api/v1/eval/run", {
    method: "POST",
    body: JSON.stringify({ config: c, split: "eval" }),
  });
  document.getElementById("actionsOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnMaterialize").onclick = async () => {
  const c = requireConfig();
  const types = document
    .getElementById("matTypes")
    .value.split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const data = await api("/api/v1/episodes/materialize", {
    method: "POST",
    body: JSON.stringify({ config: c, entity_types: types }),
  });
  document.getElementById("actionsOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnStudy").onclick = async () => {
  const c = requireConfig();
  const data = await api("/api/v1/studies/propose", {
    method: "POST",
    body: JSON.stringify({
      config: c,
      use_llm: document.getElementById("studyLlm").checked,
    }),
  });
  document.getElementById("actionsOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnSchedule").onclick = async () => {
  const c = requireConfig();
  const live = document.getElementById("livePath").value.trim();
  const data = await api("/api/v1/schedules", {
    method: "POST",
    body: JSON.stringify({
      action: document.getElementById("schedAction").value,
      interval: document.getElementById("schedInterval").value,
      config: c,
      live: live || undefined,
      promote: document.getElementById("trainPromote").checked,
      out: document.getElementById("trainOut").value.trim() || "artifacts",
    }),
  });
  document.getElementById("actionsOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnAudit").onclick = async () => {
  const data = await api("/api/v1/audit?limit=50");
  document.getElementById("actionsOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnTrain").onclick = async () => {
  const c = requireConfig();
  const out = document.getElementById("trainOut").value.trim() || "artifacts";
  const promote = document.getElementById("trainPromote").checked;
  const data = await api("/api/v1/train", {
    method: "POST",
    body: JSON.stringify({ config: c, out, promote }),
  });
  document.getElementById("actionsOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
  if (trainPollTimer) clearTimeout(trainPollTimer);
  await pollTrainJob(data.job_id);
};

document.getElementById("btnExport").onclick = async () => {
  const c = requireConfig();
  const name = document.getElementById("exportName").value.trim();
  if (!name) {
    alert("Set export name");
    return;
  }
  const data = await api("/api/v1/export-dataset", {
    method: "POST",
    body: JSON.stringify({ config: c, name }),
  });
  document.getElementById("actionsOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnInit").onclick = async () => {
  const path = document.getElementById("initPath").value.trim();
  const template = document.getElementById("initTemplate").value;
  if (!path) {
    alert("Set init path");
    return;
  }
  const data = await api("/api/v1/projects/init", {
    method: "POST",
    body: JSON.stringify({ path, template }),
  });
  document.getElementById("actionsOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnPredict").onclick = async () => {
  const w = wf();
  if (!w) {
    alert("Set workflow name");
    return;
  }
  const c = requireConfig();
  const inputs = {
    alarm: Number(document.getElementById("inAlarm").value),
    flow: Number(document.getElementById("inFlow").value),
    temp: Number(document.getElementById("inTemp").value),
    ambient: Number(document.getElementById("inAmbient").value),
    vibration: Number(document.getElementById("inVibration").value),
  };
  const data = await api("/api/v1/predict", {
    method: "POST",
    body: JSON.stringify({ workflow: w, inputs, config: c }),
  });
  document.getElementById("predictOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
  document.getElementById("fbRequestId").value = data.request_id || "";
};

document.getElementById("btnFeedback").onclick = async () => {
  const c = requireConfig();
  const body = {
    config: c,
    request_id: document.getElementById("fbRequestId").value.trim(),
    final_action: document.getElementById("fbAction").value.trim(),
    outcome_success: document.getElementById("fbSuccess").checked,
  };
  const slice = document.getElementById("fbSlice").value.trim();
  if (slice) body.slice = slice;
  const data = await api("/api/v1/feedback", {
    method: "POST",
    body: JSON.stringify(body),
  });
  document.getElementById("feedbackOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnMonitor").onclick = async () => {
  const live = document.getElementById("livePath").value.trim();
  const c = requireConfig();
  if (!live) {
    alert("Set live CSV path");
    return;
  }
  const data = await api(
    `/api/v1/monitor?live=${encodeURIComponent(live)}&config=${encodeURIComponent(c)}`,
  );
  document.getElementById("monitorOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

document.getElementById("btnOntology").onclick = async () => {
  const data = await api("/api/v1/ontology/export");
  document.getElementById("ontologyOut").textContent = JSON.stringify(
    data,
    null,
    2,
  );
};

loadTemplates().catch(() => {});
refresh().catch(() => {});
