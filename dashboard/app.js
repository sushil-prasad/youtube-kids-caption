const STEPS = [
  "UPLOAD",
  "QUEUED",
  "EXTRACTING_AUDIO",
  "TRANSCRIBING",
  "ALIGNING",
  "PUNCTUATING",
  "DETECTING_SPEAKERS",
  "DETECTING_SOUNDS",
  "SAFETY_ANALYSIS",
  "CORRECTING",
  "SEGMENTING",
  "GENERATING_SRT",
  "READY_FOR_REVIEW",
  "REVIEWED",
  "EXPORTED",
];

const BUSY = new Set(STEPS.slice(0, STEPS.indexOf("READY_FOR_REVIEW")));

const state = {
  jobId: null,
  job: null,
  captions: [],
  selected: null,
  duration: 0,
  previewUrl: null,
  pollToken: 0,
  uploadAbort: null,
};

const $ = (sel) => document.querySelector(sel);

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("is-active", item === tab));
    document.querySelectorAll(".panel").forEach((panel) => {
      panel.classList.toggle("is-active", panel.id === `panel-${tab.dataset.tab}`);
    });
  });
});

$("#upload-form").addEventListener("submit", (event) => event.preventDefault());

$("#upload-file").addEventListener("click", () => $("#file").click());

$("#try-sample").addEventListener("click", async () => {
  const status = $("#upload-status");
  status.hidden = false;
  status.textContent = "Loading sample clip…";
  const response = await fetch("/sample.mp4");
  if (!response.ok) {
    status.textContent = "Sample clip sample.mp4 is missing.";
    return;
  }
  const blob = await response.blob();
  const file = new File([blob], "sample.mp4", { type: blob.type || "video/mp4" });
  previewLocalFile(file);
  startUpload(file);
});

$("#cancel-upload").addEventListener("click", cancelUpload);

$("#file").addEventListener("change", () => {
  syncCancelButton();
  const file = $("#file").files[0];
  if (!file) {
    clearLocalPreview();
    return;
  }
  previewLocalFile(file);
  startUpload(file);
});

function syncCancelButton() {
  const busy = Boolean(state.job && BUSY.has(state.job.status));
  $("#cancel-upload").disabled = !($("#file").files[0] || state.previewUrl || busy || state.uploadAbort);
}

async function cancelUpload() {
  if (state.uploadAbort) state.uploadAbort.abort();
  state.uploadAbort = null;
  state.pollToken += 1;
  const jobId = state.jobId;
  $("#file").value = "";
  clearLocalPreview();
  state.jobId = null;
  state.job = null;
  state.captions = [];
  state.selected = null;
  $("#caption-list").innerHTML = "";
  $("#timeline").innerHTML = "";
  const status = $("#upload-status");
  status.hidden = false;
  status.textContent = "Upload cancelled.";
  renderStatus();
  syncCancelButton();
  if (jobId) {
    await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  }
}

async function startUpload(file) {
  const status = $("#upload-status");
  status.hidden = false;
  status.textContent = "Uploading a copy of the video…";
  if (state.uploadAbort) state.uploadAbort.abort();
  state.uploadAbort = new AbortController();
  syncCancelButton();
  const body = new FormData();
  body.append("file", file);
  let response;
  try {
    response = await fetch("/api/upload", { method: "POST", body, signal: state.uploadAbort.signal });
  } catch (error) {
    if (error && error.name === "AbortError") return;
    status.textContent = "Upload failed";
    state.uploadAbort = null;
    syncCancelButton();
    return;
  }
  state.uploadAbort = null;
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    const busyId = detail && typeof detail === "object" ? detail.busy_job_id : null;
    status.textContent =
      (detail && detail.message) || (typeof detail === "string" ? detail : "Upload failed");
    if (response.status === 409 && busyId) {
      state.jobId = busyId;
      showCaptionSkeleton();
      pollJob();
    }
    syncCancelButton();
    return;
  }
  $("#file").value = "";
  state.jobId = data.job_id;
  state.captions = [];
  state.selected = null;
  $("#caption-list").innerHTML = "";
  $("#timeline").innerHTML = "";
  status.textContent = `Job ${state.jobId} queued.`;
  showCaptionSkeleton();
  syncCancelButton();
  pollJob();
}

function showFirstFrame() {
  const player = $("#player");
  player.pause();
  const paint = () => player.pause();
  player.addEventListener("seeked", paint, { once: true });
  player.currentTime = 0.001;
}

function previewLocalFile(file) {
  const player = $("#player");
  clearLocalPreview(false);
  state.previewUrl = URL.createObjectURL(file);
  player.pause();
  player.removeAttribute("poster");
  player.src = state.previewUrl;
  player.setAttribute("data-src", state.previewUrl);
  $("#player-frame").hidden = false;
  player.load();
  player.addEventListener("loadeddata", showFirstFrame, { once: true });
  syncCancelButton();
}

function clearLocalPreview(hide = true) {
  const player = $("#player");
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }
  if (!hide) return;
  player.pause();
  player.removeAttribute("src");
  player.removeAttribute("data-src");
  player.removeAttribute("poster");
  player.load();
  $("#player-frame").hidden = true;
}

$("#mark-reviewed").addEventListener("click", async () => {
  if (!state.jobId) return;
  await fetch(`/api/jobs/${state.jobId}/review`, { method: "POST", headers: jsonHeaders(), body: "{}" });
  pollJob();
});

document.querySelectorAll("[data-export]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!state.jobId) return;
    window.location = `/api/jobs/${state.jobId}/export/${button.dataset.export}`;
  });
});

$("#player").addEventListener("timeupdate", () => {
  const time = $("#player").currentTime;
  const caption = state.captions.find((item) => time >= item.start && time < item.end);
  if (caption) selectCaption(caption.index, false);
});

$("#vocab-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await fetch("/api/vocabulary", {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify({
      term: $("#vocab-term").value,
      category: $("#vocab-category").value,
    }),
  });
  $("#vocab-term").value = "";
  loadVocabulary();
});

$("#vocab-export").addEventListener("click", async () => {
  const data = await (await fetch("/api/vocabulary/export")).json();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "vocabulary.json";
  link.click();
  URL.revokeObjectURL(url);
});

$("#vocab-import-btn").addEventListener("click", () => {
  $("#vocab-import").click();
});

$("#vocab-import").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const payload = JSON.parse(await file.text());
  await fetch("/api/vocabulary/import", {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  event.target.value = "";
  loadVocabulary();
});

$("#settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const body = {
    safety_mode: form.get("safety_mode"),
    unknown_profanity: form.get("unknown_profanity"),
    enable_sound_events: form.get("enable_sound_events") === "true",
  };
  await fetch("/api/settings", { method: "PUT", headers: jsonHeaders(), body: JSON.stringify(body) });
  $("#settings-saved").hidden = false;
});

function jsonHeaders() {
  return { "Content-Type": "application/json" };
}

function confidenceColor(band) {
  if (band === "low") return "var(--low)";
  if (band === "medium") return "var(--medium)";
  return "var(--high)";
}

function formatTime(seconds) {
  const value = Math.max(0, seconds || 0);
  const minutes = Math.floor(value / 60);
  const secs = Math.floor(value % 60);
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function attachVideo(jobId) {
  if (state.previewUrl) return;
  const player = $("#player");
  const stamp = encodeURIComponent(jobId);
  const src = `/api/jobs/${jobId}/video?v=${stamp}`;
  const poster = `/api/jobs/${jobId}/thumbnail?v=${stamp}`;
  $("#player-frame").hidden = false;
  if (player.getAttribute("data-src") === src) return;
  player.pause();
  player.removeAttribute("src");
  player.poster = poster;
  player.src = src;
  player.setAttribute("data-src", src);
  player.load();
  player.addEventListener("loadeddata", showFirstFrame, { once: true });
}

async function pollJob() {
  if (!state.jobId) return;
  const token = state.pollToken;
  const response = await fetch(`/api/jobs/${state.jobId}`);
  if (!response.ok || token !== state.pollToken) return;
  state.job = await response.json();
  renderStatus();
  syncCancelButton();
  if (state.job.has_video) attachVideo(state.jobId);
  if (state.job.has_captions) await loadCaptions();
  else if (BUSY.has(state.job.status)) showCaptionSkeleton();
  else if (!$("#caption-list").querySelector(".cue:not(.skeleton)")) {
    $("#caption-list").innerHTML = "";
    $("#timeline").innerHTML = "";
    $("#timeline").classList.remove("is-skeleton");
  }
  if (token !== state.pollToken) return;
  if (BUSY.has(state.job.status)) {
    setTimeout(pollJob, 1500);
  }
}

function renderStatus() {
  const job = state.job;
  $("#job-status").textContent = job
    ? `${job.status.replaceAll("_", " ")}${job.error ? ` — ${job.error}` : ""}`
    : "No job yet";
  $("#status-steps").innerHTML = STEPS.map((step) => {
    const current = job && job.status === step;
    const done = job && STEPS.indexOf(job.status) > STEPS.indexOf(step);
    return `<li class="${current ? "is-current" : ""} ${done ? "is-done" : ""}">${step.replaceAll("_", " ")}</li>`;
  }).join("");
  const quality = job && job.quality;
  const box = $("#quality");
  if (!quality) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.innerHTML = [
    ["Overall confidence", `${quality.overall_confidence}%`],
    ["Words", quality.words],
    ["Low-confidence", quality.low_confidence_words],
    ["Potential profanity", quality.potential_profanity],
    ["Auto-corrected", quality.automatically_corrected],
    ["Censored", quality.censored],
    ["Sound events", quality.sound_events],
    ["Reading-speed warnings", quality.reading_speed_warnings],
  ]
    .map(([label, value]) => `<p>${label}<strong>${value}</strong></p>`)
    .join("");
}

async function loadCaptions() {
  const data = await (await fetch(`/api/jobs/${state.jobId}/captions`)).json();
  state.captions = data.captions || [];
  state.duration = Math.max(...state.captions.map((item) => item.end), state.job.duration || 0, 1);
  renderCaptions();
}

function showCaptionSkeleton() {
  const list = $("#caption-list");
  if (list.querySelector(".cue.skeleton")) return;
  $("#timeline").classList.add("is-skeleton");
  $("#timeline").innerHTML = [
    [6, 16],
    [28, 22],
    [56, 14],
    [76, 18],
  ]
    .map(([left, width]) => `<span style="left:${left}%;width:${width}%"></span>`)
    .join("");
  list.setAttribute("aria-busy", "true");
  list.innerHTML = Array.from({ length: 7 }, () => {
    return `<article class="cue skeleton" aria-hidden="true">
      <span class="cue-bar"></span>
      <div class="cue-body">
        <span class="skeleton-line skeleton-time"></span>
        <span class="skeleton-line skeleton-text"></span>
        <span class="skeleton-line skeleton-text short"></span>
      </div>
    </article>`;
  }).join("");
}

function renderCaptions() {
  $("#timeline").classList.remove("is-skeleton");
  $("#caption-list").removeAttribute("aria-busy");
  const duration = state.duration || 1;
  $("#timeline").innerHTML = state.captions
    .map((caption) => {
      const left = (caption.start / duration) * 100;
      const width = Math.max(0.6, ((caption.end - caption.start) / duration) * 100);
      const band = caption.confidence_band || "high";
      const color = confidenceColor(band);
      return `<span data-index="${caption.index}" style="left:${left}%;width:${width}%;background:${color}"></span>`;
    })
    .join("");
  $("#timeline").querySelectorAll("span").forEach((mark) => {
    mark.addEventListener("click", () => selectCaption(Number(mark.dataset.index), true));
  });
  $("#caption-list").innerHTML = state.captions
    .map((caption) => {
      const band = caption.confidence_band || "high";
      const active = state.selected === caption.index ? "is-active" : "";
      const color = confidenceColor(band);
      return `<article class="cue ${band} ${active}" data-index="${caption.index}">
        <span class="cue-bar" style="background:${color}" aria-hidden="true"></span>
        <div class="cue-body">
          <time>${formatTime(caption.start)} → ${formatTime(caption.end)}</time>
          <textarea>${escapeHtml(caption.text)}</textarea>
        </div>
      </article>`;
    })
    .join("");
  $("#caption-list").querySelectorAll(".cue").forEach((cue) => {
    const index = Number(cue.dataset.index);
    cue.addEventListener("click", () => selectCaption(index, true));
    cue.querySelector("textarea").addEventListener("change", (event) => saveCaption(index, event.target.value));
  });
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function selectCaption(index, seek) {
  state.selected = index;
  document.querySelectorAll(".cue").forEach((cue) => {
    cue.classList.toggle("is-active", Number(cue.dataset.index) === index);
  });
  const caption = state.captions.find((item) => item.index === index);
  if (!caption) return;
  if (seek) $("#player").currentTime = caption.start;
  renderSafety(caption);
}

function renderSafety(caption) {
  const panel = $("#safety-panel");
  const flag = (caption.safety || []).find((item) => item.needs_review);
  if (!flag) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  panel.innerHTML = `
    <h3>⚠ Potentially unsafe word</h3>
    <p><q>${escapeHtml(caption.text)}</q></p>
    <p>Suggested correction: <strong>${escapeHtml(flag.suggested_text || flag.replacement)}</strong></p>
    <div class="actions">
      <button type="button" data-act="accept">Accept</button>
      <button type="button" data-act="keep_censored">Keep censored</button>
      <button type="button" data-act="edit">Edit</button>
    </div>`;
  panel.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", async () => {
      const action = button.dataset.act;
      const payload = { index: caption.index, action };
      if (action === "edit") {
        payload.text = prompt("Edit caption", caption.text) || caption.text;
      }
      await fetch(`/api/jobs/${state.jobId}/safety`, {
        method: "POST",
        headers: jsonHeaders(),
        body: JSON.stringify(payload),
      });
      await loadCaptions();
      selectCaption(caption.index, false);
    });
  });
}

async function saveCaption(index, text) {
  const captions = state.captions.map((item) => ({
    index: item.index,
    start: item.start,
    end: item.end,
    text: item.index === index ? text : item.text,
    speaker: item.speaker,
    confidence_band: item.confidence_band,
    mean_confidence: item.mean_confidence,
    flags: item.flags,
  }));
  await fetch(`/api/jobs/${state.jobId}/captions`, {
    method: "PUT",
    headers: jsonHeaders(),
    body: JSON.stringify({ captions }),
  });
  await loadCaptions();
}

async function loadVocabulary() {
  const data = await (await fetch("/api/vocabulary")).json();
  const labels = {
    character_names: "Character names",
    people: "People",
    games: "Brands",
    toys: "Toys",
    locations: "Places",
    fictional: "Made-up words",
    brands: "Brands",
    phrases: "Phrases",
    other: "Other",
  };
  const sections = Object.entries(data.grouped || {}).filter(([, terms]) => terms.length);
  $("#vocab-groups").innerHTML =
    `<p class="hint">Click × to remove a word, including built-in names.</p>` +
    sections
      .map(
        ([category, terms]) =>
          `<section><h3>${labels[category] || category}</h3>${terms
            .map((term) => renderVocabChip(term))
            .join("")}</section>`
      )
      .join("");
  $("#vocab-groups").querySelectorAll(".chip-remove").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.preventDefault();
      const term = button.dataset.term;
      if (!term) return;
      const response = await fetch(`/api/vocabulary/${encodeURIComponent(term)}`, { method: "DELETE" });
      if (!response.ok) return;
      loadVocabulary();
    });
  });
}

function renderVocabChip(term) {
  return `<span class="chip is-custom">${escapeHtml(term)}<button type="button" class="chip-remove" data-term="${escapeHtml(term)}" aria-label="Remove ${escapeHtml(term)}">×</button></span>`;
}

async function loadSettings() {
  const data = await (await fetch("/api/settings")).json();
  for (const [name, value] of Object.entries(data)) {
    const encoded = typeof value === "boolean" ? String(value) : value;
    const input = document.querySelector(`[name="${name}"][value="${encoded}"]`);
    if (input) input.checked = true;
  }
}

loadVocabulary();
loadSettings();
restoreActiveJob();

async function restoreActiveJob() {
  const response = await fetch("/api/jobs");
  if (!response.ok) return;
  const data = await response.json();
  if (!data.busy_job_id) return;
  state.jobId = data.busy_job_id;
  showCaptionSkeleton();
  pollJob();
}
