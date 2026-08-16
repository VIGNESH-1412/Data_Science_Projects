/* VitalSign AI — ICU Mortality Risk Predictor — Frontend Logic */

// ---------- Floating heart-beat animation ----------
(function () {
  const layer = document.getElementById("floatLayer");
  if (!layer) return;
  const glyphs = ["♡", "♥", "+", "·"];
  for (let i = 0; i < 18; i++) {
    const el = document.createElement("span");
    el.className = "beat";
    el.textContent = glyphs[i % glyphs.length];
    el.style.left = Math.random() * 100 + "%";
    el.style.bottom = -10 - Math.random() * 30 + "%";
    el.style.fontSize = 14 + Math.random() * 22 + "px";
    el.style.animationDuration = 8 + Math.random() * 10 + "s";
    el.style.animationDelay = Math.random() * 10 + "s";
    layer.appendChild(el);
  }
  const yrEl = document.getElementById("yr");
  if (yrEl) yrEl.textContent = new Date().getFullYear();
})();

// ---------- API URL (auto-detect Vercel vs localhost) ----------
const IS_VERCEL =
  window.location.hostname &&
  !window.location.hostname.includes("localhost") &&
  !window.location.hostname.includes("127.0.0.1");
const API = IS_VERCEL
  ? window.location.origin + "/api"
  : "http://localhost:5000/api";
const apiUrlEl = document.getElementById("apiUrl");
if (apiUrlEl) apiUrlEl.textContent = API;

// ---------- Validation ----------
function validateField(el) {
  const errEl = el.parentElement.querySelector(".err");
  el.classList.remove("invalid");
  errEl.textContent = "";
  if (el.disabled) return true;
  const val = el.value.trim();
  if (el.required && val === "") {
    el.classList.add("invalid");
    errEl.textContent = "Required.";
    return false;
  }
  if (val !== "" && el.type === "number") {
    const n = Number(val);
    if (Number.isNaN(n)) {
      el.classList.add("invalid");
      errEl.textContent = "Must be a number.";
      return false;
    }
    if (el.min !== "" && n < Number(el.min)) {
      el.classList.add("invalid");
      errEl.textContent = `Min ${el.min}.`;
      return false;
    }
    if (el.max !== "" && n > Number(el.max)) {
      el.classList.add("invalid");
      errEl.textContent = `Max ${el.max}.`;
      return false;
    }
  }
  return true;
}

function validateForm(form) {
  let ok = true;
  form.querySelectorAll("input,select").forEach((el) => {
    if (!validateField(el)) ok = false;
  });
  return ok;
}

const form = document.getElementById("predictForm");
if (form) {
  form.addEventListener("input", (e) => {
    if (e.target.matches("input,select")) validateField(e.target);
  });
}

const resetBtn = document.getElementById("resetBtn");
if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    form.reset();
    document.getElementById("result").classList.remove("show");
    form
      .querySelectorAll(".invalid")
      .forEach((el) => el.classList.remove("invalid"));
    form.querySelectorAll(".err").forEach((el) => (el.textContent = ""));
  });
}

if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const resBox = document.getElementById("result");
    if (!validateForm(form)) {
      resBox.classList.add("show");
      resBox.innerHTML =
        '<strong style="color:var(--danger)">Please correct the highlighted fields before submitting.</strong>';
      return;
    }
    const fd = new FormData(form);
    const payload = {};
    fd.forEach((v, k) => (payload[k] = v));
    resBox.classList.add("show");
    resBox.innerHTML = "Scoring…";
    try {
      const r = await fetch(API + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "Prediction failed");
      const pct = Math.round(data.probability * 1000) / 10;
      resBox.innerHTML = `
        <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
          <span class="risk-badge risk-${data.risk_band}">● ${data.risk_band} risk</span>
          <span style="font-size:28px;font-weight:600">${pct}%</span>
          <span style="color:var(--muted)">${data.label}</span>
        </div>
        <div class="bar"><span id="bar"></span></div>
        <p style="color:var(--muted);font-size:13px;margin-top:10px">Decision threshold: ${data.threshold}. Values above are classified as high-risk.</p>`;
      requestAnimationFrame(() =>
        document.getElementById("bar").style.setProperty("width", pct + "%")
      );
    } catch (err) {
      resBox.innerHTML = `<strong style="color:var(--danger)">Error:</strong> ${err.message}. Confirm the API is running at <code>${API}</code>.`;
    }
  });
}

// ---------- Batch CSV ----------
let lastBatch = null;
const csvBtn = document.getElementById("csvBtn");
if (csvBtn) {
  csvBtn.addEventListener("click", async () => {
    const f = document.getElementById("csvFile").files[0];
    const out = document.getElementById("batchOut");
    if (!f) {
      out.innerHTML =
        '<p style="color:var(--danger)">Choose a CSV file first.</p>';
      return;
    }
    out.innerHTML = "Uploading…";
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await fetch(API + "/predict_batch", {
        method: "POST",
        body: fd,
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "Batch failed");
      lastBatch = data.results;
      document.getElementById("csvDownload").disabled = false;
      const rows = data.results
        .map(
          (x) =>
            `<tr><td>${x.row}</td><td>${(x.probability * 100).toFixed(1)}%</td>
       <td><span class="risk-badge risk-${x.risk_band}">${x.risk_band}</span></td>
       <td>${x.label}</td></tr>`
        )
        .join("");
      out.innerHTML = `<p style="margin-top:12px;color:var(--muted)">${data.count} rows scored (threshold ${data.threshold}).</p>
      <div style="overflow:auto"><table><thead><tr><th>#</th><th>Probability</th><th>Band</th><th>Label</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    } catch (err) {
      out.innerHTML = `<p style="color:var(--danger)">Error: ${err.message}</p>`;
    }
  });
}

const csvDownload = document.getElementById("csvDownload");
if (csvDownload) {
  csvDownload.addEventListener("click", () => {
    if (!lastBatch) return;
    const header = "row,probability,prediction,risk_band,label";
    const csv = [header]
      .concat(
        lastBatch.map((x) =>
          [x.row, x.probability, x.prediction, x.risk_band, `"${x.label}"`].join(
            ","
          )
        )
      )
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "vitalsign_ai_predictions.csv";
    a.click();
  });
}
