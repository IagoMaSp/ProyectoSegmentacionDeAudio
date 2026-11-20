"use strict";

const CATS = [
  { key: "musica", label: "Música", color: "#2563EB" },
  { key: "voz", label: "Voz", color: "#059669" },
  { key: "ruido", label: "Ruido", color: "#D97706" },
  { key: "silencio", label: "Silencio", color: "#94A3B8" },
];
const COLOR = Object.fromEntries(CATS.map((c) => [c.key, c.color]));
const LABEL = Object.fromEntries(CATS.map((c) => [c.key, c.label]));
const INK = "#171717", MUTED = "#52524E", FAINT = "#8A8983", RULE_SOFT = "#D6D5CE", RULE_FAINT = "#E8E7E0";
const MAX_BYTES = 200 * 1024 * 1024;
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => Number(v).toFixed(d).replace(".", ",");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch { /* respuesta vacia */ }
  if (!res.ok) throw new Error((data && data.error) || `Error ${res.status}`);
  return data;
}

/* ======================================================================
   Graficos en canvas (sin bibliotecas)
   ====================================================================== */

function prepCanvas(cv) {
  if (!cv.dataset.h) cv.dataset.h = cv.getAttribute("height") || "150";
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(50, cv.clientWidth);
  const h = Number(cv.dataset.h);
  cv.style.height = h + "px";
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

function shortNum(v) {
  const a = Math.abs(v);
  if (a >= 1000) return (v / 1000).toString().replace(".", ",") + "k";
  return Number(v.toFixed(3)).toString().replace(".", ",");
}

/** Lineas sobre ejes lineales o logaritmicos. */
function plotLines(cv, o) {
  const { ctx, w, h } = prepCanvas(cv);
  const m = { l: o.ml ?? 46, r: 10, t: 8, b: 22 };
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  const xs = o.series.flatMap((s) => s.x);
  const ys = o.series.flatMap((s) => s.y).filter((v) => Number.isFinite(v));
  let x0 = o.xmin ?? Math.min(...xs), x1 = o.xmax ?? Math.max(...xs);
  let y0 = o.ymin ?? Math.min(...ys), y1 = o.ymax ?? Math.max(...ys);
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const lx = (v) => (o.xlog ? Math.log10(Math.max(v, 1e-9)) : v);
  const X = (v) => m.l + ((lx(v) - lx(x0)) / (lx(x1) - lx(x0))) * pw;
  const Y = (v) => m.t + (1 - (Math.min(Math.max(v, y0), y1) - y0) / (y1 - y0)) * ph;

  ctx.font = "10px ui-monospace, Menlo, monospace";
  ctx.fillStyle = MUTED;
  ctx.strokeStyle = RULE_FAINT;
  ctx.lineWidth = 1;
  const yt = o.yticks || niceTicks(y0, y1, 4);
  for (const v of yt) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + pw, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText((o.yfmt || shortNum)(v), m.l - 5, y);
  }
  let xt = o.xticks;
  if (!xt) {
    if (o.xlog) {
      xt = [];
      for (let p = Math.floor(Math.log10(x0)); p <= Math.ceil(Math.log10(x1)); p++) {
        for (const k of [1, 2, 5]) { const v = k * 10 ** p; if (v >= x0 && v <= x1) xt.push(v); }
      }
    } else xt = niceTicks(x0, x1, 6);
  }
  for (const v of xt) {
    const x = Math.round(X(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ph); ctx.stroke();
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText((o.xfmt || shortNum)(v), x, m.t + ph + 5);
  }
  for (const vl of o.vlines || []) {
    ctx.save();
    ctx.strokeStyle = vl.color || FAINT; ctx.setLineDash(vl.dash || [3, 3]);
    const x = Math.round(X(vl.x)) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ph); ctx.stroke();
    if (vl.label) { ctx.fillStyle = vl.color || MUTED; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(vl.label, x + 4, m.t + 2); }
    ctx.restore();
  }
  for (const hl of o.hlines || []) {
    ctx.save(); ctx.strokeStyle = hl.color || FAINT; ctx.setLineDash(hl.dash || [3, 3]);
    const y = Math.round(Y(hl.y)) + 0.5;
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + pw, y); ctx.stroke(); ctx.restore();
  }
  ctx.save();
  ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
  for (const s of o.series) {
    ctx.strokeStyle = s.color || INK; ctx.lineWidth = s.width || 1.3; ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      if (!Number.isFinite(s.y[i])) { pen = false; continue; }
      const px = X(s.x[i]), py = Y(s.y[i]);
      if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; }
    }
    ctx.stroke();
    if (s.dots) {
      ctx.fillStyle = s.color || INK;
      for (let i = 0; i < s.x.length; i++) if (Number.isFinite(s.y[i])) ctx.fillRect(X(s.x[i]) - 2.5, Y(s.y[i]) - 2.5, 5, 5);
    }
  }
  ctx.restore();
  ctx.strokeStyle = INK; ctx.setLineDash([]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(m.l + 0.5, m.t); ctx.lineTo(m.l + 0.5, m.t + ph + 0.5); ctx.lineTo(m.l + pw, m.t + ph + 0.5); ctx.stroke();
  for (const mk of o.markers || []) {
    ctx.fillStyle = mk.color || INK;
    ctx.beginPath(); ctx.arc(X(mk.x), Y(mk.y), 4, 0, 2 * Math.PI); ctx.fill();
    if (mk.label) { ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText(mk.label, X(mk.x) + 6, Y(mk.y) - 2); }
  }
}

/** Mapa de grises (tinta sobre marfil). cols: array de columnas (tiempo) de valores. */
function grayRGB(t) {
  const v = Math.round(251 - Math.min(Math.max(t, 0), 1) * 228);
  return [v, v, Math.round(v * 0.985)];
}

function plotHeat(cv, cols, vmin, vmax, o = {}) {
  const { ctx, w, h } = prepCanvas(cv);
  const m = { l: o.ml ?? 46, r: 10, t: 6, b: o.xlabels ? 20 : 6 };
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  if (!cols.length) return;
  const nT = cols.length, nF = cols[0].length;
  const img = ctx.createImageData(nT, nF);
  for (let t = 0; t < nT; t++) {
    for (let f = 0; f < nF; f++) {
      const [r, g, b] = grayRGB((cols[t][f] - vmin) / (vmax - vmin));
      const idx = ((nF - 1 - f) * nT + t) * 4;
      img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
    }
  }
  const tmp = document.createElement("canvas");
  tmp.width = nT; tmp.height = nF;
  tmp.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = nT < pw;
  ctx.drawImage(tmp, m.l, m.t, pw, ph);
  ctx.strokeStyle = INK; ctx.lineWidth = 1;
  ctx.strokeRect(m.l + 0.5, m.t + 0.5, pw - 1, ph - 1);
  ctx.font = "10px ui-monospace, Menlo, monospace"; ctx.fillStyle = MUTED;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  const fmax = o.fmax || 8000;
  for (const f of niceTicks(0, fmax, 4)) {
    const y = m.t + (1 - f / fmax) * ph;
    ctx.fillText(shortNum(f), m.l - 5, y);
  }
  if (o.xlabels) {
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const [frac, txt] of o.xlabels) {
      ctx.textAlign = frac <= 0 ? "left" : frac >= 1 ? "right" : "center";
      ctx.fillText(txt, m.l + frac * pw, m.t + ph + 4);
    }
  }
}

function plotEnvelope(cv, env, color = INK) {
  const { ctx, w, h } = prepCanvas(cv);
  if (!env || !env.max.length) return;
  const n = env.max.length;
  const peak = Math.max(1e-6, ...env.max.map(Math.abs), ...env.min.map(Math.abs));
  ctx.fillStyle = color;
  const mid = h / 2;
  for (let i = 0; i < n; i++) {
    const x = (i / n) * w;
    const y1 = mid - (env.max[i] / peak) * (mid - 3);
    const y2 = mid - (env.min[i] / peak) * (mid - 3);
    ctx.fillRect(x, y1, Math.max(1, w / n), Math.max(1, y2 - y1));
  }
  ctx.strokeStyle = RULE_SOFT; ctx.beginPath(); ctx.moveTo(0, mid + 0.5); ctx.lineTo(w, mid + 0.5); ctx.stroke();
}

function probRows(container, probs, ghost) {
  container.innerHTML = "";
  for (const c of CATS) {
    const p = probs[c.key] ?? 0;
    const row = document.createElement("div");
    row.className = "prob-row";
    const g = ghost ? `<span class="prob-ghost" style="left:calc(${(ghost[c.key] ?? 0) * 100}% - 1px)"></span>` : "";
    row.innerHTML = `<span class="cat-cell"><span class="swatch" style="background:${c.color}"></span>${c.label}</span>
      <span class="prob-track"><span class="prob-fill" style="display:block;width:${(p * 100).toFixed(1)}%;background:${c.color}"></span>${g}</span>
      <span class="mono num" style="text-align:right">${(p * 100).toFixed(1)}%</span>`;
    container.appendChild(row);
  }
}

/* ======================================================================
   Pestanas y estado del modelo
   ====================================================================== */

const loaded = {};
function showView(name) {
  for (const t of document.querySelectorAll(".tab")) t.setAttribute("aria-selected", String(t.dataset.view === name));
  for (const v of document.querySelectorAll("main > .view")) v.hidden = v.id !== `view-${name}`;
  if (name === "filtros" && !loaded.filtros) { loaded.filtros = true; loadFilters(); }
  if (name === "aliasing" && !loaded.aliasing) { loaded.aliasing = true; loadAliasing(); }
  if (name === "metricas") loadMetrics();
  if (name === "archivo" && currentData) redrawTimeline();
  if (name === "filtros" && filterData) drawFilters();
}
for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => showView(t.dataset.view));

let serverInfo = null;
getJSON("/api/estado").then((info) => {
  serverInfo = info;
  const p = info.parametros;
  $("specs").textContent = `16 kHz mono · ventana de ${fmt(p.ventana_s, 0)} s cada ${fmt(p.salto_s, 1)} s · ` +
    (info.modelo_cargado ? `${info.descripcion}, ${info.n_caracteristicas} características` : "sin modelo");
  if (!info.modelo_cargado) {
    $("model-warning").hidden = false;
    $("model-warning").textContent = `${info.error} Las pestañas Acondicionamiento y Aliasing funcionan igual.`;
  }
}).catch(() => {
  $("model-warning").hidden = false;
  $("model-warning").textContent = "No se pudo conectar con el servidor local. ¿Está corriendo python -m src.server?";
});

/* ======================================================================
   Archivo: carga y segmentacion
   ====================================================================== */

let currentData = null, activeFilter = null, objectUrl = null, openSeg = null, currentName = "audio";

function formatBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${fmt(b / 1024, 1)} KB`;
  return `${fmt(b / 1048576, 2)} MB`;
}
function setSelectedFile(file) {
  $("file-label").textContent = file.name;
  $("file-meta").textContent = `${formatBytes(file.size)} · listo para procesar`;
}
$("file").addEventListener("change", () => { const f = $("file").files[0]; if (f) setSelectedFile(f); });
const dz = $("dropzone");
["dragenter", "dragover"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("is-dragover"); }));
["dragleave", "drop"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("is-dragover"); }));
dz.addEventListener("drop", (ev) => {
  const f = ev.dataTransfer?.files?.[0];
  if (!f) return;
  const dt = new DataTransfer(); dt.items.add(f); $("file").files = dt.files; setSelectedFile(f);
});

function setStatus(html, isError = false) {
  $("status").className = "status-line" + (isError ? " is-error" : "");
  $("status").innerHTML = html;
}
function setLoading(on) {
  $("submit-btn").disabled = on;
  $("btn-text").innerHTML = on ? '<span class="spinner" aria-hidden="true"></span>Procesando…' : "Procesar audio";
}

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("file").files[0];
  if (!file) return;
  if (file.size > MAX_BYTES) return setStatus(`El archivo pesa ${formatBytes(file.size)} y supera el límite de 200 MB.`, true);
  const q = new URLSearchParams({
    salto: $("p-salto").value, suavizado: $("p-suavizado").value,
    penalidad: $("p-penalidad").value, minimo: $("p-minimo").value,
  });
  setStatus("Decodificando, remuestreando a 16 kHz, filtrando y clasificando ventanas…");
  $("results").hidden = true; $("empty-state").hidden = true; $("why-panel").hidden = true;
  setLoading(true);
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  $("player").src = objectUrl;
  currentName = file.name.replace(/\.[^.]+$/, "") || "audio";
  try {
    const data = await getJSON(`/api/segmentar?${q}`, {
      method: "POST", body: file, headers: { "X-Nombre-Archivo": encodeURIComponent(file.name) },
    });
    currentData = data; activeFilter = null; openSeg = null;
    render(data);
    enableAB(data.id);
  } catch (err) {
    setStatus(esc(err.message || "No se pudo conectar con el servidor."), true);
    $("empty-state").hidden = false;
  } finally {
    setLoading(false);
  }
});

function render(d) {
  const smoothingName = { viterbi: `Viterbi (penalidad ${fmt(d.penalty, 1)})`, mediana: "mediana", ninguno: "sin suavizado" }[d.smoothing];
  setStatus(
    `<span class="num"><strong>${fmt(d.duration, 2)} s</strong> · fs original ${d.fs_original} Hz · ` +
    `${d.windows.start.length} ventanas · ${d.segments.length} segmentos</span>` +
    `<span class="num muted">${smoothingName} · duración mínima ${fmt(d.min_duration, 1)} s · ${fmt(d.segundos_proceso, 2)} s de cálculo</span>`
  );
  $("cell-label").textContent = `${fmt(d.hop, 2)} s`;
  renderSummary();
  renderTable();
  $("results").hidden = false;
  $("now-playing").hidden = true;
  requestAnimationFrame(redrawTimeline);
}

function redrawTimeline() {
  if (!currentData) return;
  renderRuler(currentData.duration);
  plotEnvelope($("wave-canvas"), currentData.envolvente, "#3F3F3B");
  renderBar();
  drawProbTrack();
  $("playhead").classList.add("is-visible");
  updatePlayback();
}

function renderSummary() {
  const d = currentData;
  const totals = Object.fromEntries(CATS.map((c) => [c.key, 0]));
  for (const s of d.segments) totals[s.label] += s.end_time - s.start_time;
  const strip = $("summary-strip");
  strip.innerHTML = "";
  for (const c of CATS) {
    const sec = totals[c.key], pct = d.duration > 0 ? (sec / d.duration) * 100 : 0;
    const b = document.createElement("button");
    b.type = "button";
    b.className = "stat-btn" + (activeFilter === c.key ? " is-selected" : "");
    b.setAttribute("aria-pressed", String(activeFilter === c.key));
    b.innerHTML = `<span class="stat-head"><span class="stat-label"><span class="swatch" style="background:${c.color}"></span>${c.label}</span>
      <span>${activeFilter === c.key ? "Filtrado" : "Filtrar"}</span></span>
      <span class="stat-value num">${fmt(pct, 1)}%<span class="stat-seconds">${fmt(sec, 1)} s</span></span>`;
    b.onclick = () => { activeFilter = activeFilter === c.key ? null : c.key; renderSummary(); renderBar(); renderTable(); };
    strip.appendChild(b);
  }
}

function renderRuler(duration) {
  const r = $("ruler");
  r.innerHTML = "";
  for (let i = 0; i <= 6; i++) {
    const s = document.createElement("span");
    s.textContent = `${fmt((duration * i) / 6, 1)} s`;
    r.appendChild(s);
  }
}

function segDesc(s) {
  return `${fmt(s.start_time)}–${fmt(s.end_time)} s · ${LABEL[s.label]} · confianza ${fmt(s.confidence * 100, 0)}%`;
}

function renderBar() {
  const d = currentData, bar = $("bar");
  bar.innerHTML = "";
  d.segments.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "strata-seg" + (activeFilter && activeFilter !== s.label ? " is-dimmed" : "") + (openSeg === i ? " is-open" : "");
    b.style.width = `${((s.end_time - s.start_time) / d.duration) * 100}%`;
    b.style.background = COLOR[s.label];
    b.title = segDesc(s);
    b.setAttribute("aria-label", `Abrir ${segDesc(s)}`);
    b.onmouseenter = () => { $("hover-readout").textContent = segDesc(s); };
    b.onmouseleave = () => { $("hover-readout").textContent = "Hacé clic en un bloque para escucharlo y ver el detalle"; };
    b.onclick = () => openSegment(i, true);
    bar.appendChild(b);
  });
}

/** Probabilidades por celda como bandas apiladas (antes del suavizado). */
function drawProbTrack() {
  const cv = $("prob-canvas");
  const { ctx, w, h } = prepCanvas(cv);
  const cells = currentData.cells, n = cells.raw.length, hop = currentData.hop, dur = currentData.duration;
  for (let k = 0; k < n; k++) {
    const x0 = (cells.t[k] / dur) * w;
    const x1 = (Math.min(cells.t[k] + hop, dur) / dur) * w;
    let y = h;
    CATS.forEach((c, j) => {
      const hh = cells.raw[k][j] * h;
      ctx.fillStyle = c.color;
      ctx.fillRect(x0, y - hh, Math.max(1, x1 - x0 + 0.5), hh);
      y -= hh;
    });
  }
  ctx.strokeStyle = "rgba(251,251,248,.8)"; ctx.setLineDash([2, 3]);
  ctx.beginPath(); ctx.moveTo(0, h / 2 + 0.5); ctx.lineTo(w, h / 2 + 0.5); ctx.stroke();
}

function renderTable() {
  const d = currentData, tbody = $("table").querySelector("tbody");
  tbody.innerHTML = "";
  const rows = d.segments.map((s, i) => [s, i]).filter(([s]) => !activeFilter || s.label === activeFilter);
  $("clear-filter").hidden = !activeFilter;
  $("table-heading").textContent = activeFilter ? `Intervalos de ${LABEL[activeFilter].toLowerCase()} (${rows.length})` : `Intervalos detectados (${rows.length})`;
  for (const [s, i] of rows) {
    const tr = tbody.insertRow();
    tr.dataset.idx = i; tr.dataset.start = s.start_time; tr.dataset.end = s.end_time; tr.tabIndex = 0;
    if (openSeg === i) tr.classList.add("is-open");
    tr.innerHTML = `<td class="num">${fmt(s.start_time)}</td><td class="num">${fmt(s.end_time)}</td>
      <td class="num">${fmt(s.end_time - s.start_time, 1)} s</td>
      <td><span class="cat-cell"><span class="swatch" style="background:${COLOR[s.label]}"></span>${LABEL[s.label]}</span></td>
      <td class="num"><span class="conf-bar"><i style="width:${(s.confidence * 100).toFixed(0)}%"></i></span>${fmt(s.confidence * 100, 0)}%</td>
      <td class="r muted">Ver por qué</td>`;
    tr.onclick = () => openSegment(i, true);
    tr.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openSegment(i, true); } };
  }
}
$("clear-filter").addEventListener("click", () => { activeFilter = null; renderSummary(); renderBar(); renderTable(); });

function seekAndPlay(t) {
  const p = $("player");
  p.currentTime = t;
  p.play().catch(() => {});
}

function updatePlayback() {
  if (!currentData) return;
  const t = $("player").currentTime || 0;
  $("playhead").style.left = `${Math.min(100, Math.max(0, (t / currentData.duration) * 100))}%`;
  const seg = currentData.segments.find((s) => t >= s.start_time && t < s.end_time);
  if (seg) {
    $("now-playing").hidden = false;
    $("now-swatch").style.background = COLOR[seg.label];
    $("now-text").textContent = `${LABEL[seg.label]} (${fmt(seg.start_time)}–${fmt(seg.end_time)} s)`;
  } else $("now-playing").hidden = true;
  for (const r of $("table").querySelectorAll("tbody tr")) r.classList.toggle("is-playing", t >= +r.dataset.start && t < +r.dataset.end);
}
$("player").addEventListener("timeupdate", updatePlayback);
$("player").addEventListener("seeked", updatePlayback);
window.addEventListener("resize", () => {
  if (!$("view-archivo").hidden && currentData) { redrawTimeline(); if (whyData) drawWhy(); }
  if (!$("view-filtros").hidden && filterData) drawFilters();
});

/* ---------- panel "por que" ---------- */

let whyData = null;
async function openSegment(i, play) {
  const s = currentData.segments[i];
  openSeg = i;
  renderBar(); renderTable();
  if (play) seekAndPlay(s.start_time);
  const panel = $("why-panel");
  panel.hidden = false;
  $("why-title").textContent = `Detalle: ${segDesc(s)}`;
  $("why-verdict").textContent = "Calculando…";
  try {
    whyData = await getJSON(`/api/explicar?id=${currentData.id}&inicio=${s.start_time}&fin=${s.end_time}`);
    whyData._seg = s;
    drawWhy();
  } catch (err) {
    $("why-verdict").textContent = err.message;
  }
}
$("why-close").addEventListener("click", () => { $("why-panel").hidden = true; openSeg = null; renderBar(); renderTable(); });

function drawWhy() {
  const d = whyData, s = d._seg;
  plotEnvelope($("why-wave"), d.envolvente, COLOR[s.label]);
  const dur = d.fin - d.inicio;
  plotHeat($("why-spec"), d.espectrograma.db, -100, -20, {
    fmax: d.espectrograma.fmax,
    xlabels: [[0, `${fmt(d.inicio, 1)} s`], [0.5, `${fmt(d.inicio + dur / 2, 1)} s`], [1, `${fmt(d.fin, 1)} s`]],
  });
  const curve = (id, arr, o = {}) => {
    const x = arr.map((_, k) => d.inicio + (k / Math.max(1, arr.length - 1)) * dur);
    plotLines($(id), { series: [{ x, y: arr, color: COLOR[s.label], width: 1.1 }], ml: 40, ...o });
  };
  curve("why-rms", d.curvas.rms, { ymin: Math.max(-100, Math.min(...d.curvas.rms)), ymax: 0 });
  curve("why-zcr", d.curvas.zcr, { ymin: 0 });
  curve("why-cent", d.curvas.centroide, { ymin: 0 });
  curve("why-flat", d.curvas.planitud, { ymin: 0, ymax: 1 });

  const alt = d.alternativa;
  $("why-verdict").innerHTML = `Clasificado como <b style="color:${COLOR[d.clase]}">${LABEL[d.clase].toLowerCase()}</b>` +
    ` con ${fmt(d.probabilidades[d.clase] * 100, 0)}% de probabilidad media sobre ${d.ventanas.inicio.length} ventana(s);` +
    ` la segunda opción fue ${LABEL[alt].toLowerCase()} con ${fmt(d.probabilidades[alt] * 100, 0)}%.`;
  probRows($("why-probs"), d.probabilidades);
  drawPCA();
  $("why-traits-title").textContent = `Rasgos que más lo acercan a ${LABEL[d.clase].toLowerCase()} frente a ${LABEL[alt].toLowerCase()}`;
  $("why-traits").innerHTML = d.rasgos.map((r) =>
    `<tr><td class="mono">${esc(r.nombre)}</td><td class="r num">${fmt(r.z, 2)}</td><td class="r num">${fmt(r.z_clase, 2)}</td><td class="r num">${fmt(r.z_alternativa, 2)}</td></tr>`
  ).join("");
}

function drawPCA() {
  const d = whyData, cv = $("why-pca");
  const { ctx, w, h } = prepCanvas(cv);
  const pts = d.pca.nube;
  const xs = pts.map((p) => p[0]).concat([d.pca.punto[0]]);
  const ys = pts.map((p) => p[1]).concat([d.pca.punto[1]]);
  const q = (arr, f) => { const s = [...arr].sort((a, b) => a - b); return s[Math.floor(f * (s.length - 1))]; };
  let x0 = q(xs, 0.01), x1 = q(xs, 0.99), y0 = q(ys, 0.01), y1 = q(ys, 0.99);
  x0 = Math.min(x0, d.pca.punto[0]); x1 = Math.max(x1, d.pca.punto[0]);
  y0 = Math.min(y0, d.pca.punto[1]); y1 = Math.max(y1, d.pca.punto[1]);
  const pad = 0.06, m = 8;
  const X = (v) => m + ((v - x0) / (x1 - x0) * (1 - 2 * pad) + pad) * (w - 2 * m);
  const Y = (v) => m + (1 - ((v - y0) / (y1 - y0) * (1 - 2 * pad) + pad)) * (h - 2 * m);
  ctx.strokeStyle = RULE_FAINT; ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  ctx.globalAlpha = 0.32;
  for (const p of pts) { ctx.fillStyle = CATS[p[2]].color; ctx.fillRect(X(p[0]) - 1.5, Y(p[1]) - 1.5, 3, 3); }
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1.4;
  for (const v of d.pca.vecinos) {
    ctx.strokeStyle = COLOR[v.label];
    ctx.beginPath(); ctx.arc(X(v.x), Y(v.y), 4.5, 0, 2 * Math.PI); ctx.stroke();
  }
  if (d.pca.ventanas.length > 1) {
    ctx.strokeStyle = MUTED; ctx.lineWidth = 1;
    ctx.beginPath();
    d.pca.ventanas.forEach((p, k) => (k ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
    ctx.stroke();
  }
  const [px, py] = [X(d.pca.punto[0]), Y(d.pca.punto[1])];
  ctx.strokeStyle = INK; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px - 7, py); ctx.lineTo(px + 7, py); ctx.moveTo(px, py - 7); ctx.lineTo(px, py + 7); ctx.stroke();
  const counts = {};
  for (const v of d.pca.vecinos) counts[v.label] = (counts[v.label] || 0) + 1;
  const vec = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${n} de ${LABEL[k].toLowerCase()}`).join(", ");
  const varTxt = d.pca.varianza.map((v) => fmt(v * 100, 0) + "%").join(" + ");
  $("why-pca-cap").textContent = `Cruz: media del segmento. Círculos: los 10 vecinos más cercanos en el espacio de 64 dimensiones (${vec}). Puntos: muestra de entrenamiento. Varianza explicada ${varTxt}.`;
}

/* ---------- exportar ---------- */

function download(content, name, mime) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const exportable = () => currentData && {
  archivo: currentName, duracion: currentData.duration, salto: currentData.hop, suavizado: currentData.smoothing,
  penalidad: currentData.penalty, duracion_minima: currentData.min_duration, modelo: currentData.modelo,
  segmentos: currentData.segments.map(({ label_id, ...s }) => s),
};
$("export-json").addEventListener("click", () => {
  if (currentData) download(JSON.stringify(exportable(), null, 2), `${currentName}_segmentos.json`, "application/json");
});
$("export-csv").addEventListener("click", () => {
  if (!currentData) return;
  const lines = ["inicio_s,fin_s,duracion_s,categoria,confianza," + CATS.map((c) => `p_${c.key}`).join(",")];
  for (const s of currentData.segments) {
    lines.push([s.start_time, s.end_time, (s.end_time - s.start_time).toFixed(3), s.label, s.confidence,
      ...CATS.map((c) => s.probabilities[c.key])].join(","));
  }
  download(lines.join("\n") + "\n", `${currentName}_segmentos.csv`, "text/csv;charset=utf-8");
});
function srtTime(t) {
  const ms = Math.round(t * 1000);
  const p = (v, n = 2) => String(v).padStart(n, "0");
  return `${p(Math.floor(ms / 3600000))}:${p(Math.floor(ms / 60000) % 60)}:${p(Math.floor(ms / 1000) % 60)},${p(ms % 1000, 3)}`;
}
$("export-srt").addEventListener("click", () => {
  if (!currentData) return;
  const blocks = currentData.segments.map((s, i) =>
    `${i + 1}\n${srtTime(s.start_time)} --> ${srtTime(s.end_time)}\n[${LABEL[s.label]}] ${fmt(s.confidence * 100, 0)}%\n`);
  download(blocks.join("\n"), `${currentName}_segmentos.srt`, "text/plain;charset=utf-8");
});

/* ======================================================================
   En vivo
   ====================================================================== */

const live = { ctx: null, node: null, stream: null, session: null, queue: [], queued: 0, busy: false, cols: [], decisions: [], fs: 0 };
const LIVE_COLS = 1000; // 10 s a 100 columnas por segundo

$("live-toggle").addEventListener("click", () => (live.ctx ? stopLive() : startLive()));
probRows($("live-probs"), {});

async function startLive() {
  try {
    if (serverInfo && !serverInfo.modelo_cargado) throw new Error("El servidor no tiene modelo cargado.");
    live.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    live.ctx = new AudioContext();
    live.fs = live.ctx.sampleRate;
    await live.ctx.audioWorklet.addModule("/captura.js");
    const src = live.ctx.createMediaStreamSource(live.stream);
    live.node = new AudioWorkletNode(live.ctx, "captura");
    live.node.port.onmessage = (e) => { live.queue.push(e.data); live.queued += e.data.length; if (live.queued >= live.fs / 4) flushLive(); };
    src.connect(live.node);
    live.session = Math.random().toString(36).slice(2);
    live.cols = []; live.decisions = [];
    $("live-toggle").textContent = "Detener";
    $("live-info").textContent = `placa a ${live.fs} Hz → remuestreo a 16 kHz en el servidor`;
  } catch (err) {
    $("live-info").textContent = err.message || "No se pudo abrir el micrófono.";
    stopLive();
  }
}

function stopLive() {
  if (live.stream) live.stream.getTracks().forEach((t) => t.stop());
  if (live.ctx) live.ctx.close();
  if (live.session) fetch(`/api/vivo/fin?sesion=${live.session}`, { method: "POST" }).catch(() => {});
  Object.assign(live, { ctx: null, node: null, stream: null, session: null, queue: [], queued: 0, busy: false });
  $("live-toggle").textContent = "Iniciar micrófono";
}

async function flushLive() {
  if (live.busy || !live.session) return;
  const buf = new Float32Array(live.queued);
  let o = 0;
  for (const c of live.queue) { buf.set(c, o); o += c.length; }
  live.queue = []; live.queued = 0; live.busy = true;
  try {
    const r = await getJSON(`/api/vivo?sesion=${live.session}&fs=${live.fs}`, { method: "POST", body: buf.buffer });
    for (const c of r.columns) live.cols.push(c);
    if (live.cols.length > LIVE_COLS) live.cols.splice(0, live.cols.length - LIVE_COLS);
    for (const d of r.decisions) live.decisions.push(d);
    if (live.decisions.length > 60) live.decisions.splice(0, live.decisions.length - 60);
    drawLive(r);
  } catch (err) {
    $("live-info").textContent = err.message;
  } finally {
    live.busy = false;
    if (live.queued >= live.fs / 4) flushLive();
  }
}

function drawLive(r) {
  const cols = live.cols.length < LIVE_COLS ? Array(LIVE_COLS - live.cols.length).fill(new Array(128).fill(-140)).concat(live.cols) : live.cols;
  plotHeat($("live-spec"), cols, -100, -20, { fmax: 8000, xlabels: [[0, "−10 s"], [0.5, "−5 s"], [1, "ahora"]] });
  const { ctx, w, h } = prepCanvas($("live-strip"));
  const n = 60;
  live.decisions.forEach((d, k) => {
    const x = ((n - live.decisions.length + k) / n) * w;
    ctx.fillStyle = COLOR[d.label];
    ctx.fillRect(x, 0, w / n + 0.5, h);
  });
  ctx.strokeStyle = INK; ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  if (r.level_db != null) {
    $("live-level").textContent = `${fmt(r.level_db, 1)} dBFS`;
    $("live-meter").style.width = `${Math.min(100, Math.max(0, (r.level_db + 90) / 90 * 100))}%`;
  }
  const last = live.decisions[live.decisions.length - 1];
  if (last) {
    const el = $("live-label");
    el.style.background = COLOR[last.label];
    el.firstChild.nodeValue = LABEL[last.label];
    $("live-sub").textContent = last.label_crudo !== last.label ? `crudo: ${LABEL[last.label_crudo].toLowerCase()}` : `t = ${fmt(last.t, 0)} s`;
    probRows($("live-probs"), last.suavizado, last.crudo);
  }
}

/* ======================================================================
   Acondicionamiento
   ====================================================================== */

let filterData = null, pzZoom = true;
async function loadFilters() {
  try {
    filterData = await getJSON("/api/filtros");
    const p = filterData.parametros;
    $("f-summary").textContent = `${p.secciones} secciones de 2.º orden · fs ${p.fs} Hz · ${filterData.estable ? "estable" : "INESTABLE"}`;
    $("f-params").innerHTML = [
      ["Pasa-altos", `Butterworth, orden ${p.hp_orden}`], ["Frecuencia de corte", `${p.hp_fc} Hz (−3 dB)`],
      ["Pendiente", `${6 * p.hp_orden} dB/octava`], ["Notch", `${p.notch_f0} Hz, Q = ${p.notch_q}`],
      ["Ancho de la muesca", `${fmt(p.notch_bw, 1)} Hz (f0/Q)`], ["Radio máximo de polos", `${filterData.radio_max_polos}`],
      ["Aplicación offline", "sosfiltfilt, fase nula, |H|²"], ["Aplicación en vivo", "sosfilt ×2 con estado, |H|²"],
      ["Anti-alias 48→16 kHz", `FIR Kaiser, ${filterData.antialias.coeficientes} coef., ${filterData.antialias.atenuacion_db} dB`],
    ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    drawFilters();
  } catch (err) {
    $("f-summary").textContent = err.message;
  }
}

function drawFilters() {
  const d = filterData;
  if (!d || $("view-filtros").hidden) return;
  plotLines($("f-mag"), {
    xlog: true, xmin: 5, xmax: 8000, ymin: -90, ymax: 5,
    series: [
      { x: d.f, y: d.hp_db, color: COLOR.musica },
      { x: d.f, y: d.notch_db, color: COLOR.ruido },
      { x: d.f, y: d.total_filtfilt_db, color: FAINT, dash: [5, 3] },
      { x: d.f, y: d.total_db, color: INK, width: 1.6 },
    ],
    vlines: [{ x: d.parametros.hp_fc, label: `${d.parametros.hp_fc} Hz` }, { x: 50, label: "50 Hz", color: COLOR.ruido }],
    hlines: [{ y: -3 }],
  });
  plotLines($("f-phase"), { xlog: true, xmin: 5, xmax: 8000, series: [{ x: d.f, y: d.fase_grados, color: INK }] });
  plotLines($("f-gd"), { xlog: true, xmin: 20, xmax: 8000, ymin: 0, ymax: 40, series: [{ x: d.f, y: d.retardo_ms, color: INK }] });
  drawPZ();
  drawSpectrum();
  plotLines($("f-aa"), {
    xmin: 0, xmax: 12000, ymin: -120, ymax: 5,
    series: [{ x: d.antialias.f, y: d.antialias.db, color: INK }],
    vlines: [{ x: 8000, label: "Nyquist 8 kHz" }, { x: 7200 }], hlines: [{ y: -80 }],
  });
  $("aa-cap").textContent = "FIR anti-alias para 48 → 16 kHz: banda de paso hasta 7,2 kHz, rechazo ≥ 80 dB desde 8 kHz.";
}

function drawPZ() {
  const d = filterData, { ctx, w, h } = prepCanvas($("f-pz"));
  const size = Math.min(w, h);
  const [cx, cy, r] = pzZoom ? [0.985, 0, 0.035] : [0, 0, 1.15];
  const ox = (w - size) / 2;
  const X = (v) => ox + ((v - (cx - r)) / (2 * r)) * size;
  const Y = (v) => ((cy + r - v) / (2 * r)) * size;
  ctx.strokeStyle = RULE_FAINT;
  ctx.beginPath(); ctx.moveTo(X(cx - r), Y(0)); ctx.lineTo(X(cx + r), Y(0)); ctx.moveTo(X(0), Y(cy - r)); ctx.lineTo(X(0), Y(cy + r)); ctx.stroke();
  ctx.strokeStyle = FAINT; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = 0; k <= 2000; k++) {
    const a = (k / 2000) * 2 * Math.PI;
    const px = X(Math.cos(a)), py = Y(Math.sin(a));
    k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  }
  ctx.stroke();
  ctx.lineWidth = 1.5;
  for (const [re, im] of d.ceros) { ctx.strokeStyle = COLOR.musica; ctx.beginPath(); ctx.arc(X(re), Y(im), 5, 0, 2 * Math.PI); ctx.stroke(); }
  for (const [re, im] of d.polos) {
    ctx.strokeStyle = COLOR.ruido; const px = X(re), py = Y(im);
    ctx.beginPath(); ctx.moveTo(px - 5, py - 5); ctx.lineTo(px + 5, py + 5); ctx.moveTo(px + 5, py - 5); ctx.lineTo(px - 5, py + 5); ctx.stroke();
  }
  ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.strokeRect(ox + 0.5, 0.5, size - 1, size - 1);
  $("pz-cap").textContent = pzZoom
    ? "Detalle cerca de z = 1: el pasa-altos tiene un cero de orden 4 en z = 1 (rechazo total de continua) y sus cuatro polos sobre un arco; el notch pone sus ceros exactamente sobre la circunferencia en e^(±jω0), ω0 = 2π·50/16000, con los polos apenas adentro. Círculos: ceros; cruces: polos."
    : "Plano z completo. Todo se concentra cerca de z = 1 porque 70 Hz es apenas el 0,9 % de fs.";
}
$("pz-zoom").addEventListener("click", () => {
  pzZoom = !pzZoom;
  $("pz-zoom").setAttribute("aria-pressed", String(pzZoom));
  if (filterData) drawPZ();
});

let specData = null;
function enableAB(id) {
  $("ab-orig").disabled = false; $("ab-cond").disabled = false; $("ab-dif").disabled = false;
  $("ab-note").textContent =
    "A es tu audio tal como entra al sistema (decodificado, en mono y remuestreado a 16 kHz). B es lo mismo después del pasa-altos " +
    "y el notch, que es lo que ve el clasificador. El filtro solo toca lo que está por debajo de ~70 Hz: viento, golpes al micrófono, " +
    "continua y zumbido de red a 50 Hz. Voz y música quedan intactas, así que en una grabación limpia A y B suenan casi iguales; " +
    "la diferencia se escucha con auriculares en grabaciones con viento o zumbido. A − B reproduce solo lo que el filtro quitó, " +
    "amplificado para que se oiga (la ganancia aplicada figura en el resumen de abajo); si ahí casi no hay nada, es porque tu grabación " +
    "no tenía graves ni zumbido que sacar. Al cambiar entre versiones se conserva la posición.";
  setAB("original", id);
  loadSpectrum(id);
}

async function loadSpectrum(id) {
  specData = null;
  try {
    specData = await getJSON(`/api/espectro?id=${id}`);
  } catch (err) {
    $("spec-fig").hidden = true; $("spec-resumen").hidden = true; $("spec-legend").hidden = true;
    return;
  }
  const r = specData.resumen;
  $("spec-resumen").innerHTML = [
    ["Energía total quitada", `${fmt(r.energia_quitada_pct, 2)} %`],
    [`Energía bajo ${r.hp_fc} Hz`, `${fmt(r.graves_antes_pct, 2)} % → ${fmt(r.graves_despues_pct, 2)} % del total`],
    [`Pico en ${r.red_f0} Hz (red)`, `${fmt(r.red_antes_db, 1)} → ${fmt(r.red_despues_db, 1)} dB (${fmt(r.red_despues_db - r.red_antes_db, 1)} dB)`],
    ["Ganancia aplicada a A − B", `+${fmt(r.diferencia_ganancia_db, 1)} dB`],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("spec-cap").textContent =
    "Densidad espectral de potencia [dB] contra frecuencia [Hz] (Welch, ventanas de 0,5 s). Por encima de ~150 Hz las dos curvas " +
    "se superponen: el filtro no cambia esa parte. La separación en graves sigue la respuesta del pasa-altos de arriba, y en 50 Hz " +
    "aparece la muesca si había zumbido de red.";
  $("spec-fig").hidden = false; $("spec-resumen").hidden = false; $("spec-legend").hidden = false;
  drawSpectrum();
}

function drawSpectrum() {
  const d = specData;
  if (!d || $("view-filtros").hidden) return;
  const all = d.original_db.concat(d.acondicionado_db).filter(Number.isFinite);
  const top = Math.ceil(Math.max(...all) / 10) * 10 + 5;
  plotLines($("f-spec"), {
    xlog: true, xmin: 5, xmax: 8000, ymin: top - 100, ymax: top,
    series: [
      { x: d.f, y: d.original_db, color: FAINT, width: 1.4 },
      { x: d.f, y: d.acondicionado_db, color: INK, width: 1.4 },
    ],
    vlines: [{ x: d.resumen.hp_fc, label: `${d.resumen.hp_fc} Hz` }, { x: d.resumen.red_f0, label: `${d.resumen.red_f0} Hz`, color: COLOR.ruido }],
  });
}
function setAB(version, id = currentData && currentData.id) {
  const p = $("ab-player"), t = p.currentTime || 0, playing = !p.paused;
  p.src = `/api/audio?id=${id}&version=${version}`;
  p.addEventListener("loadedmetadata", () => { p.currentTime = t; if (playing) p.play().catch(() => {}); }, { once: true });
  $("ab-orig").setAttribute("aria-pressed", String(version === "original"));
  $("ab-cond").setAttribute("aria-pressed", String(version === "acondicionado"));
  $("ab-dif").setAttribute("aria-pressed", String(version === "diferencia"));
}
$("ab-orig").addEventListener("click", () => setAB("original"));
$("ab-cond").addEventListener("click", () => setAB("acondicionado"));
$("ab-dif").addEventListener("click", () => setAB("diferencia"));

/* ======================================================================
   Aliasing
   ====================================================================== */

let alTimer = null;
function aliasQuery() {
  return new URLSearchParams({ modo: $("al-modo").value, tono: $("al-tono").value, fs: $("al-fs").value });
}
async function loadAliasing() {
  const q = aliasQuery();
  $("al-tono-val").textContent = `${$("al-tono").value} Hz`;
  $("al-tono-wrap").hidden = q.get("modo") !== "tono";
  try {
    const d = await getJSON(`/api/aliasing?${q}`);
    const nyq = d.nyquist;
    if (d.modo === "tono") {
      $("al-tone-figs").hidden = false; $("al-sweep-figs").hidden = true;
      const above = d.tono > nyq;
      $("al-explain").textContent = above
        ? `El tono de ${d.tono} Hz supera la nueva Nyquist (${nyq} Hz). Sin filtro se pliega a |f − k·fs| = ${fmt(d.alias_previsto, 0)} Hz, una frecuencia que no existía en la señal. Con el anti-alias se elimina antes de diezmar.`
        : `El tono de ${d.tono} Hz está por debajo de la nueva Nyquist (${nyq} Hz): los dos métodos lo conservan. Subilo por encima de ${nyq} Hz para ver el alias.`;
      const e = d.espectros;
      plotLines($("al-orig"), { xmin: 0, xmax: 24000, ymin: -120, ymax: 5, series: [{ x: e.original.f, y: e.original.db, color: INK }], vlines: [{ x: nyq, color: COLOR.ruido, label: `fs/2 = ${nyq}` }] });
      const mk = above ? [{ x: d.alias_previsto, y: 0, color: COLOR.ruido, label: `alias ${fmt(d.alias_previsto, 0)} Hz` }] : [];
      plotLines($("al-sin"), { xmin: 0, xmax: nyq, ymin: -120, ymax: 5, series: [{ x: e.sin.f, y: e.sin.db, color: COLOR.ruido }], markers: mk });
      plotLines($("al-con"), { xmin: 0, xmax: nyq, ymin: -120, ymax: 5, series: [{ x: e.con.f, y: e.con.db, color: COLOR.musica }] });
    } else {
      $("al-tone-figs").hidden = true; $("al-sweep-figs").hidden = false;
      $("al-explain").textContent = `Barrido lineal de 100 Hz a 20 kHz en 4 s, llevado de 48 kHz a ${d.fs_out} Hz. Sin filtro, la parte del barrido por encima de ${nyq} Hz vuelve reflejada como un tono que baja y sube: eso es el aliasing que se escucha.`;
      const xl = [[0, "0 s"], [0.5, "2 s"], [1, "4 s"]];
      plotHeat($("al-sg-sin"), d.espectrogramas.sin.db, -110, -20, { fmax: nyq, xlabels: xl });
      plotHeat($("al-sg-con"), d.espectrogramas.con.db, -110, -20, { fmax: nyq, xlabels: xl });
    }
  } catch (err) {
    $("al-explain").textContent = err.message;
  }
}
for (const id of ["al-modo", "al-fs"]) $(id).addEventListener("change", loadAliasing);
$("al-tono").addEventListener("input", () => {
  $("al-tono-val").textContent = `${$("al-tono").value} Hz`;
  clearTimeout(alTimer); alTimer = setTimeout(loadAliasing, 120);
});
for (const b of document.querySelectorAll("[data-al]")) {
  b.addEventListener("click", () => {
    const q = aliasQuery(); q.set("variante", b.dataset.al);
    const p = $("al-player"); p.src = `/api/aliasing.wav?${q}`; p.play().catch(() => {});
  });
}

/* ======================================================================
   Metricas
   ====================================================================== */

const MODEL_LABEL = { knn: "k-NN", mlp: "MLP", svm: "SVM RBF", rf: "Random Forest", gb: "Gradient Boosting" };
const pct = (v) => (v == null ? "—" : `${fmt(v * 100, 1)}%`);

function confusionTable(conf) {
  const head = `<tr><th>real \\ pred.</th>${CATS.map((c) => `<th>${c.label}</th>`).join("")}<th>recall</th></tr>`;
  const rows = conf.map((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0) || 1;
    const cells = row.map((v, j) => {
      const f = v / tot;
      const bg = `rgba(23,23,23,${(f * 0.85).toFixed(3)})`;
      return `<td style="background:${bg};color:${f > 0.5 ? "#FBFBF8" : INK}">${fmt(f * 100, 1)}%<br><span style="opacity:.7">${v}</span></td>`;
    }).join("");
    return `<tr><th><span class="cat-cell"><span class="swatch" style="background:${CATS[i].color}"></span>${CATS[i].label}</span></th>${cells}<td>${fmt((row[i] / tot) * 100, 1)}%</td></tr>`;
  }).join("");
  return `<table class="cm">${head}${rows}</table>`;
}

function curveChart(id, block, xlabel) {
  const cv = document.getElementById(id);
  if (!cv || !block) return;
  const x = block.x;
  plotLines(cv, {
    xmin: Math.min(...x) - 1, xmax: Math.max(...x) + 1, ymin: 0, ymax: 1,
    yfmt: (v) => `${Math.round(v * 100)}%`,
    xticks: x, xfmt: (v) => `${v} dB`,
    series: [
      { x, y: block.anterior_curva, color: FAINT, dash: [5, 3], dots: true },
      { x, y: block.nuevo_curva, color: INK, dots: true, width: 1.6 },
    ],
  });
}

async function loadMetrics() {
  const root = $("view-metricas");
  let m;
  try { m = await getJSON("/api/metricas"); } catch (err) { root.innerHTML = `<p class="status-line is-error">${esc(err.message)}</p>`; return; }
  const t = m.entrenamiento, legacy = m.base_anterior, rob = m.robustez, real = m.reales;
  let html = "";
  if (!t) {
    html += `<section class="panel"><div class="empty-box">Todavía no hay métricas. Generarlas con <code>python -m src.build_features</code> y <code>python -m src.train</code>.</div></section>`;
    root.innerHTML = html; return;
  }
  const p = t.protocolo, best = t.prueba.modelo;
  html += `<section class="panel"><div class="panel-head"><h2 class="section-heading">Protocolo</h2></div><div class="panel-body"><p class="note">
    Validación cruzada de ${p.pliegues_cv.length} pliegues disjuntos por archivo y, en música, por artista (${p.muestras_desarrollo} ventanas de 3 s),
    y pliegue ${p.pliegue_prueba} reservado como prueba final (${p.muestras_prueba} ventanas). Criterio de selección: ${p.criterio}.
    El modelo guardado es el que se evaluó en la prueba.</p></div></section>`;

  const rows = Object.entries(t.modelos_cv).map(([k, v]) => `<tr class="${k === best ? "is-best" : ""}">
    <td>${MODEL_LABEL[k] || k}${k === best ? " (elegido)" : ""}</td>
    <td class="r num">${fmt(v.f1_macro_media, 4)} ± ${fmt(v.f1_macro_desvio, 4)}</td>
    <td class="r num">${pct(v.accuracy_media)}</td>
    ${CATS.map((c) => `<td class="r num">${pct(v.recall_media[c.key])}</td>`).join("")}
    <td class="r num muted">${fmt(v.segundos, 0)} s</td></tr>`).join("");
  html += `<section class="panel"><div class="panel-head"><h2 class="section-heading">Comparación de clasificadores (CV)</h2></div>
    <div class="table-scroll"><table><thead><tr><th>Modelo</th><th class="r">F1 macro</th><th class="r">Accuracy</th>
    ${CATS.map((c) => `<th class="r">Recall ${c.label.toLowerCase()}</th>`).join("")}<th class="r">Tiempo</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;

  const tp = t.prueba;
  let cmp = "";
  if (legacy) {
    cmp = `<p class="note" style="margin-top:.75rem">Versión anterior (18 características, bloques fijos de 3 s, sin acondicionamiento) sobre su propia partición:
      accuracy ${pct(legacy.accuracy)}, F1 macro ${fmt(legacy.f1_macro, 4)}, recall de ruido ${pct(legacy.recall.ruido)}.
      No son directamente comparables (otro conjunto de prueba, sin augmentation); la comparación justa está en las curvas de robustez.</p>`;
  }
  html += `<section class="panel"><div class="panel-head"><h2 class="section-heading">Prueba final · ${MODEL_LABEL[best]}</h2>
    <span class="mono muted">accuracy ${pct(tp.accuracy)} · F1 macro ${fmt(tp.f1_macro, 4)}</span></div>
    <div class="two-col"><div>${confusionTable(tp.confusion)}${cmp}</div>
    <div><h3 class="sub-heading">Accuracy por tipo de muestra</h3><table class="compact"><thead><tr><th>Tipo</th><th class="r">n</th><th class="r">Accuracy</th></tr></thead><tbody>
    ${Object.entries(tp.accuracy_por_tipo).sort((a, b) => a[1].accuracy - b[1].accuracy).map(([k, v]) =>
      `<tr><td class="mono">${esc(k)}</td><td class="r num">${v.n}</td><td class="r num">${pct(v.accuracy)}</td></tr>`).join("")}
    </tbody></table></div></div></section>`;

  if (t.ablation) {
    html += `<section class="panel"><div class="panel-head"><h2 class="section-heading">Ablation de características (${MODEL_LABEL[best]}, CV)</h2></div>
      <div class="table-scroll"><table><thead><tr><th>Grupos</th><th class="r">Dimensión</th><th class="r">F1 macro</th><th class="r">Accuracy</th><th class="r">Recall ruido</th></tr></thead><tbody>
      ${t.ablation.map((a) => `<tr><td>${a.grupos.join(" + ")}</td><td class="r num">${a.n}</td><td class="r num">${fmt(a.f1_macro_media, 4)} ± ${fmt(a.f1_macro_desvio, 4)}</td>
      <td class="r num">${pct(a.accuracy_media)}</td><td class="r num">${pct(a.recall_ruido_media)}</td></tr>`).join("")}
      </tbody></table></div></section>`;
  }

  if (rob) {
    const table = (blk, head) => `<table class="compact"><thead><tr><th>${head}</th><th class="r">Nuevo</th>${blk.anterior ? '<th class="r">Anterior</th>' : ""}</tr></thead><tbody>
      ${blk.etiquetas.map((e, i) => `<tr><td class="mono">${esc(e)}</td><td class="r num">${pct(blk.nuevo[i])}</td>${blk.anterior ? `<td class="r num">${pct(blk.anterior[i])}</td>` : ""}</tr>`).join("")}</tbody></table>`;
    html += `<section class="panel"><div class="panel-head"><h2 class="section-heading">Robustez (pliegue de prueba, ${rob.n_por_condicion / 4} recortes por clase)</h2>
      <div class="legend-inline"><span class="legend-chip"><span class="swatch" style="background:${INK}"></span>Sistema nuevo</span>
      <span class="legend-chip"><span class="swatch" style="background:${FAINT}"></span>Versión anterior</span></div></div>
      <div class="two-col"><div class="fig"><canvas id="rob-snr" height="200"></canvas><span class="fig-caption">Accuracy de voz y música contra SNR [dB] con ruido del pliegue de prueba</span></div>
      <div>${table(rob.snr, "SNR")}</div></div>
      <div class="two-col" style="border-top:1px solid var(--rule)"><div>${table(rob.fs, "Frecuencia de muestreo")}</div><div>${table(rob.codec, "Códec")}</div></div></section>`;
  }
  if (rob && rob.segmentacion) {
    const v = rob.segmentacion.variantes;
    html += `<section class="panel"><div class="panel-head"><h2 class="section-heading">Segmentación: suavizado sobre ${rob.segmentacion.n_secuencias} secuencias de audio continuo del pliegue de prueba</h2></div>
      <div class="table-scroll"><table><thead><tr><th>Variante</th><th class="r">Acierto por celda</th><th class="r">Segmentos predichos / reales</th><th class="r">Error de borde (mediana)</th><th class="r">Error de borde (media)</th></tr></thead><tbody>
      ${Object.entries(v).map(([k, r]) => `<tr><td class="mono">${esc(k)}</td><td class="r num">${pct(r.accuracy_celdas)}</td>
      <td class="r num">${r.segmentos_predichos} / ${r.segmentos_reales}</td><td class="r num">${fmt(r.error_borde_mediana_s, 2)} s</td><td class="r num">${fmt(r.error_borde_medio_s, 2)} s</td></tr>`).join("")}
      </tbody></table></div></section>`;
  }
  if (real) {
    html += `<section class="panel"><div class="panel-head"><h2 class="section-heading">Grabaciones reales (data/real)</h2>
      <span class="mono muted">${real.archivos.length} archivos · acierto por celda ${pct(real.accuracy)} (sin suavizado ${pct(real.accuracy_sin_suavizado)}, versión anterior ${pct(real.accuracy_anterior)}) · F1 macro ${fmt(real.f1_macro, 3)}</span></div>
      <div class="two-col"><div>${confusionTable(real.confusion)}</div><div><table class="compact"><thead><tr><th>Archivo</th><th>Clase</th><th class="r">Acierto</th><th class="r">Anterior</th></tr></thead><tbody>
      ${real.archivos.map((a) => `<tr><td class="mono">${esc(a.archivo)}</td><td>${esc(a.clases.join(", "))}</td><td class="r num">${pct(a.accuracy)}</td><td class="r num">${pct(a.accuracy_anterior)}</td></tr>`).join("")}</tbody></table></div></div></section>`;
  } else {
    html += `<section class="panel"><div class="empty-box">Sin evaluación con grabaciones propias todavía: ver <code>data/real/LEEME.md</code> y correr <code>python -m scripts.eval_real</code>.</div></section>`;
  }
  root.innerHTML = html;
  if (rob) curveChart("rob-snr", rob.snr);
}
