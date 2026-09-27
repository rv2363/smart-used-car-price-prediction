/* Smart Car Valuator - frontend
   All numbers shown come from the /predict API; nothing here invents values. */
(function () {
  "use strict";

  const { options: OPTIONS, thisYear: THIS_YEAR } = window.SCV;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // ---------- formatting ----------
  const inr = (v) => "₹" + Math.round(Number(v) || 0).toLocaleString("en-IN");
  function short(v) {
    v = Number(v) || 0;
    if (v >= 1e7) return "₹" + (v / 1e7).toFixed(2) + " Cr";
    if (v >= 1e5) return "₹" + (v / 1e5).toFixed(2) + "L";
    return inr(v);
  }
  const km = (v) => Math.round(Number(v) || 0).toLocaleString("en-IN") + " km";
  const title = (s) => String(s || "").replace(/\b\w/g, (c) => c.toUpperCase());
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const clamp = (v, a, b) => Math.min(Math.max(v, a), b);

  // ---------- nav ----------
  const nav = $(".nav");
  $("#navToggle").addEventListener("click", () => {
    const open = nav.classList.toggle("is-open");
    $("#navToggle").setAttribute("aria-expanded", String(open));
    $("#navToggle").setAttribute("aria-label", open ? "Close menu" : "Open menu");
  });
  $$("#navMenu a").forEach((a) => a.addEventListener("click", () => {
    nav.classList.remove("is-open");
    $("#navToggle").setAttribute("aria-expanded", "false");
  }));

  // ---------- dependent make -> model -> variant selects ----------
  function fillSelect(select, values, placeholder) {
    select.innerHTML = `<option value="">${esc(placeholder)}</option>` +
      values.map((v) => `<option value="${esc(v)}">${esc(title(v))}</option>`).join("");
    select.disabled = values.length === 0;
  }
  function wireCarSelects(root) {
    const make = $("[name=make]", root), model = $("[name=model]", root), variant = $("[name=variant]", root);
    make.addEventListener("change", () => {
      fillSelect(model, OPTIONS.make_models[make.value] || [], make.value ? "Select model" : "Select brand first");
      if (variant) fillSelect(variant, [], "Select model first");
    });
    model.addEventListener("change", () => {
      if (variant) fillSelect(variant, OPTIONS.variants[make.value + "|" + model.value] || [], "Not sure");
    });
  }
  function setCarValues(root, data) {
    const set = (name, v) => { const f = $(`[name=${name}]`, root); if (f && v !== undefined && v !== null) f.value = v; };
    set("make", data.make); $("[name=make]", root).dispatchEvent(new Event("change"));
    set("model", data.model); $("[name=model]", root).dispatchEvent(new Event("change"));
    ["variant", "yr_mfr", "kms_run", "fuel_type", "transmission", "body_type", "total_owners", "city", "asking_price", "new_price"]
      .forEach((n) => set(n, data[n] ?? ""));
  }

  // ---------- API ----------
  function formToObject(form) {
    const data = {};
    new FormData(form).forEach((v, k) => { data[k] = String(v).trim(); });
    return data;
  }
  async function postJSON(url, body) {
    let res;
    try {
      res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch (e) {
      throw { errors: ["Could not reach the server. Check your connection and try again. (A sleeping free server can take up to a minute to wake up.)"] };
    }
    let json = null;
    try { json = await res.json(); } catch (e) { /* non-JSON */ }
    if (!res.ok || !json || !json.success) {
      throw { errors: (json && (json.errors || [json.error])) || ["Something went wrong. Please try again."] };
    }
    return json;
  }
  function showErrors(box, errors) {
    if (!errors || !errors.length) { box.hidden = true; return; }
    box.innerHTML = errors.length === 1 ? esc(errors[0])
      : "Please fix the following:<ul>" + errors.map((e) => `<li>${esc(e)}</li>`).join("") + "</ul>";
    box.hidden = false;
  }
  function setLoading(btn, on, text) {
    btn.disabled = on;
    btn.classList.toggle("is-loading", on);
    const label = $(".btn__label", btn);
    if (on) { btn.dataset.label = label.textContent; label.textContent = text; }
    else if (btn.dataset.label) { label.textContent = btn.dataset.label; }
  }
  // Client-side check mirrors the server, so most mistakes are caught instantly.
  function clientValidate(form, requiredNames) {
    const errors = [];
    $$(".field", form).forEach((f) => f.classList.remove("is-invalid"));
    const mark = (name, msg) => { const el = $(`[name=${name}]`, form); if (el) el.closest(".field").classList.add("is-invalid"); errors.push(msg); };
    const labels = { make: "Brand", model: "Model", yr_mfr: "Manufacturing year", kms_run: "Kilometers driven", fuel_type: "Fuel type", transmission: "Transmission" };
    requiredNames.forEach((n) => { const el = $(`[name=${n}]`, form); if (el && !el.value.trim()) mark(n, `${labels[n]} is required.`); });
    const yr = $("[name=yr_mfr]", form);
    if (yr && yr.value && (+yr.value < 1990 || +yr.value > THIS_YEAR || !Number.isInteger(+yr.value))) mark("yr_mfr", `Manufacturing year must be between 1990 and ${THIS_YEAR}.`);
    const k = $("[name=kms_run]", form);
    if (k && k.value && (+k.value < 0 || +k.value > 1000000)) mark("kms_run", "Kilometers driven must be between 0 and 10,00,000.");
    ["asking_price", "new_price"].forEach((n) => {
      const el = $(`[name=${n}]`, form);
      if (el && el.value && (+el.value < 10000 || +el.value > 50000000)) mark(n, `${n === "asking_price" ? "Asking price" : "Original price"} must be between ₹10,000 and ₹5,00,00,000.`);
    });
    return errors;
  }
  const REQUIRED = ["make", "model", "yr_mfr", "kms_run", "fuel_type", "transmission"];

  // ---------- transmission lock for electric ----------
  function wireFuelLock(root) {
    const fuel = $("[name=fuel_type]", root), trans = $("[name=transmission]", root);
    if (!fuel || !trans) return;
    fuel.addEventListener("change", () => {
      const manual = $("option[value=manual]", trans);
      if (fuel.value === "electric") { trans.value = "automatic"; if (manual) manual.disabled = true; }
      else if (manual) manual.disabled = false;
    });
  }

  // =========================================================
  // VALUATION
  // =========================================================
  const form = $("#valuationForm");
  wireCarSelects(form);
  wireFuelLock(form);
  let lastInputs = null, lastResult = null;

  form.addEventListener("reset", () => {
    setTimeout(() => {
      fillSelect($("#model"), [], "Select brand first");
      fillSelect($("#variant"), [], "Select model first");
      showErrors($("#formError"), []);
      $$(".field", form).forEach((f) => f.classList.remove("is-invalid"));
    });
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errors = clientValidate(form, REQUIRED);
    showErrors($("#formError"), errors);
    if (errors.length) { $("#formError").scrollIntoView({ block: "center", behavior: "smooth" }); return; }

    const btn = $("#submitBtn");
    setLoading(btn, true, "Calculating valuation");
    try {
      const inputs = formToObject(form);
      const result = await postJSON("/predict", inputs);
      lastInputs = inputs; lastResult = result;
      renderResult(result);
      $("#results").hidden = false;
      $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
      syncEmiPrice(result);
    } catch (err) {
      showErrors($("#formError"), err.errors);
    } finally {
      setLoading(btn, false);
    }
  });

  function positionRange(bar, low, high, price, asking) {
    // Scale: show the band with some breathing room, and the asking price if given.
    let min = low * 0.8, max = high * 1.12;
    if (asking) { min = Math.min(min, asking * 0.95); max = Math.max(max, asking * 1.05); }
    const pct = (v) => clamp(((v - min) / (max - min)) * 100, 0, 100);
    $(".range-bar__band", bar).style.left = pct(low) + "%";
    $(".range-bar__band", bar).style.right = (100 - pct(high)) + "%";
    $(".range-bar__marker", bar).style.left = pct(price) + "%";
    const a = $(".range-bar__asking", bar);
    if (a) {
      a.hidden = !asking;
      if (asking) {
        const p = pct(asking);
        a.style.left = p + "%";
        a.dataset.label = "Asking " + short(asking);
        // keep the label inside the card near the edges
        a.style.setProperty("--label-shift", p > 75 ? "calc(-100% + 10px)" : p < 25 ? "-10px" : "-50%");
      }
      bar.classList.toggle("has-asking", !!asking);
    }
  }

  function renderResult(r) {
    const c = r.car;
    $("#rPrice").textContent = inr(r.predicted_price);
    $("#rCar").textContent = `${c.yr_mfr} ${c.make} ${c.model} ${c.variant}`.trim() + ` · ${km(c.kms_run)}`;
    $("#rLow").textContent = short(r.range_low);
    $("#rHigh").textContent = short(r.range_high);
    positionRange($("#rRangeBar"), r.range_low, r.range_high, r.predicted_price, c.asking_price);

    // status + recommendation
    const status = $("#rStatus");
    status.dataset.status = r.status || "NONE";
    status.textContent = r.status ? title(r.status.toLowerCase()) : "Add an asking price for a verdict";
    $("#rStatusExplain").textContent = r.status_explanation;
    $(".rec").dataset.code = r.recommendation.code;
    $("#rRecCode").textContent = title(r.recommendation.code.toLowerCase());
    $("#rRecReason").textContent = r.recommendation.reason;

    $("#rScrap").hidden = !r.scrap_message;
    $("#rScrap").textContent = r.scrap_message || "";
    $("#rNotes").innerHTML = r.notes.map((n) => `<li>${esc(n)}</li>`).join("");

    renderBars(r);
    renderFacts(r);
    renderImportance(r.feature_importance);
    renderTrend(r.price_trend, c.yr_mfr);
    renderFuture(r);
    renderMarketPosition(r.market_position);
    renderSimilar(r);
  }

  // Estimated price difference = (predicted - reference) / reference x 100, computed server-side.
  function diffText(diff, refName) {
    if (!diff) return "";
    if (diff.direction === "equal") return `Estimated price difference: same as ${refName}`;
    const sign = diff.pct > 0 ? "+" : "−";
    return `Estimated price difference: ${sign}${Math.abs(diff.pct).toFixed(1)}% ${diff.direction} ${refName} (${sign}${short(Math.abs(diff.amount))})`;
  }

  function renderBars(r) {
    const c = r.car;
    const rows = [];
    if (c.new_price) rows.push({ cls: "new", label: "Original price (new)", v: c.new_price, diff: r.vs_new, ref: "original price" });
    if (c.asking_price) rows.push({ cls: "asking", label: "Seller's asking price", v: c.asking_price, diff: r.vs_asking, ref: "asking price" });
    rows.push({ cls: "market", label: "Estimated market value", v: r.predicted_price });
    const max = Math.max(r.range_high, ...rows.map((x) => x.v)) * 1.05;
    $("#rBars").innerHTML = rows.map((x) => {
      const band = x.cls === "market"
        ? `<span class="cbar__band" style="left:${(r.range_low / max) * 100}%;width:${((r.range_high - r.range_low) / max) * 100}%"></span>` : "";
      const d = x.diff ? `<p class="diff-line">${esc(diffText(x.diff, x.ref))}</p>` : "";
      return `<div class="cbar cbar--${x.cls}">
        <div class="cbar__head"><span>${esc(x.label)}</span><strong>${inr(x.v)}</strong></div>
        <div class="cbar__track"><span class="cbar__fill" style="width:0"></span>${band}</div>${d}</div>`;
    }).join("");
    requestAnimationFrame(() => $$("#rBars .cbar__fill").forEach((el, i) => { el.style.width = (rows[i].v / max) * 100 + "%"; }));
    if (!c.new_price && !c.asking_price) {
      $("#rBars").insertAdjacentHTML("beforeend", `<p class="diff-line">Estimated price difference: add original/asking price to compare.</p>`);
    }
  }

  function renderFacts(r) {
    const c = r.car;
    const facts = [
      ["Vehicle age", `${r.vehicle_age} year${r.vehicle_age === 1 ? "" : "s"}`],
      ["Kilometers driven", km(c.kms_run)],
      ["Fuel type", c.fuel_type],
      ["Transmission", c.transmission],
      ["City", c.city],
      ["Owners", String(c.total_owners)],
    ];
    $("#rFacts").innerHTML = facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
  }

  function renderImportance(list) {
    const top = list;  // every model input, as measured - nothing hidden or rescaled
    const max = Math.max(...top.map((d) => d.importance)) || 1;
    $("#rImportance").innerHTML = top.map((d) => `
      <div class="imp__row"><span class="imp__name">${esc(d.feature)}</span>
      <span class="imp__track"><span class="imp__fill" style="width:0"></span></span>
      <span class="imp__val">${d.importance < 1 ? d.importance.toFixed(1) : d.importance.toFixed(0)}%</span></div>`).join("");
    requestAnimationFrame(() => $$("#rImportance .imp__fill").forEach((el, i) => { el.style.width = (top[i].importance / max) * 100 + "%"; }));
  }

  function renderTrend(points, carYear) {
    const W = 520, H = 220, P = { l: 8, r: 8, t: 28, b: 30 };
    const xs = points.map((_, i) => P.l + (i * (W - P.l - P.r)) / (points.length - 1));
    const vals = points.map((p) => p.price);
    const lo = Math.min(...vals) * 0.9, hi = Math.max(...vals) * 1.05;
    const y = (v) => P.t + (1 - (v - lo) / (hi - lo || 1)) * (H - P.t - P.b);
    const line = points.map((p, i) => `${i ? "L" : "M"}${xs[i].toFixed(1)},${y(p.price).toFixed(1)}`).join("");
    const area = `${line}L${xs[xs.length - 1]},${H - P.b}L${xs[0]},${H - P.b}Z`;
    const dots = points.map((p, i) => {
      const own = p.year === carYear;
      return `<circle cx="${xs[i]}" cy="${y(p.price)}" r="${own ? 6 : 3.5}" fill="${own ? "#5B2EE0" : "#fff"}" stroke="#5B2EE0" stroke-width="2"/>
        <text x="${xs[i]}" y="${H - 8}" text-anchor="middle">${p.year}</text>
        ${own || i === 0 || i === points.length - 1 ? `<text class="pt-label" x="${clamp(xs[i], 30, W - 30)}" y="${y(p.price) - 12}" text-anchor="middle">${short(p.price)}</text>` : ""}`;
    }).join("");
    $("#rTrend").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Estimated price by manufacturing year">
      <defs><linearGradient id="tg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5B2EE0" stop-opacity=".22"/><stop offset="1" stop-color="#5B2EE0" stop-opacity="0"/></linearGradient></defs>
      <path d="${area}" fill="url(#tg)"/><path d="${line}" fill="none" stroke="#5B2EE0" stroke-width="2.5" stroke-linejoin="round"/>${dots}</svg>`;
  }

  let futureSel = 3;
  function renderFuture(r) {
    $("#rFutureNote").textContent = `Assumes the car keeps being driven about ${km(r.km_per_year)} a year (its usage so far) and applies the current valuation model to its future age and kilometers.`;
    $("#futureChips").innerHTML = r.future_values.map((f) =>
      `<button type="button" class="chip" role="tab" aria-selected="${f.years === futureSel}" data-years="${f.years}">${f.years} year${f.years > 1 ? "s" : ""}</button>`).join("");
    const show = () => {
      const f = r.future_values.find((x) => x.years === futureSel) || r.future_values[0];
      $("#rFuture").innerHTML = `
        <div class="future__box"><span>Value in ${f.years} year${f.years > 1 ? "s" : ""}</span><strong>${short(f.price)}</strong><small>${f.change_pct > 0 ? "+" : ""}${f.change_pct}% vs today</small></div>
        <div class="future__box"><span>Assumed then</span><strong>${f.age} yrs</strong><small style="color:var(--muted)">${km(f.kms)} driven</small></div>`;
      $$("#futureChips .chip").forEach((b) => b.setAttribute("aria-selected", String(+b.dataset.years === futureSel)));
    };
    $$("#futureChips .chip").forEach((b) => b.addEventListener("click", () => { futureSel = +b.dataset.years; show(); }));
    show();
  }

  function renderMarketPosition(mp) {
    const box = $("#rMarketPos");
    if (!mp.available) { box.innerHTML = `<p class="mpos__empty">${esc(mp.message)}</p>`; return; }
    const lo = Math.min(mp.low, mp.prediction), hi = Math.max(mp.high, mp.prediction);
    const pad = (hi - lo) * 0.12 || hi * 0.1;
    const min = lo - pad, max = hi + pad;
    const pct = (v) => ((v - min) / (max - min)) * 100;
    const p = pct(mp.prediction);
    const shift = p > 80 ? "calc(-100% + 8px)" : p < 20 ? "-8px" : "-50%";
    box.innerHTML = `
      <div class="mpos">
        <div class="mpos__scale" role="img" aria-label="Prediction ${short(mp.prediction)} compared with ${mp.count} historical listings from ${short(mp.low)} to ${short(mp.high)}">
          <span class="mpos__track"></span>
          <span class="mpos__band" style="left:${pct(mp.low)}%;width:${pct(mp.high) - pct(mp.low)}%"></span>
          ${mp.prices.map((v) => `<span class="mpos__dot" style="left:${pct(v)}%" title="${inr(v)}"></span>`).join("")}
          <span class="mpos__pred" style="left:${p}%;--label-shift:${shift}" data-label="${short(mp.prediction)}"></span>
        </div>
        <dl class="mpos__legend">
          <div><dt><i class="mpos__key mpos__key--pred"></i>Current prediction (estimated market value)</dt><dd>${short(mp.prediction)}</dd></div>
          <div><dt><i class="mpos__key mpos__key--comp"></i>Historical comparable listings (${mp.count})</dt><dd>${short(mp.low)} – ${short(mp.high)}</dd></div>
          <div><dt>Median comparable listing</dt><dd>${short(mp.median)}</dd></div>
        </dl>
      </div>`;
  }

  function renderSimilar(r) {
    const list = r.similar_listings;
    $("#rSimilarSub").textContent = r.similar_match === "make"
      ? `Fewer than 3 listings of this model exist, so these are other ${r.car.make} models. Based on historical listings from ${r.data_period}.`
      : `Same model at a similar age and kilometers. Based on historical listings from ${r.data_period}.`;
    const body = $("#rSimilar tbody");
    body.innerHTML = list.length
      ? list.map((s) => `<tr><td>${esc(s.name)} <small>${esc(s.variant)}</small></td><td>${s.age} yrs</td><td>${km(s.kms)}</td><td>${esc(s.city)}</td><td class="num">${inr(s.price)}</td></tr>`).join("")
      : `<tr><td colspan="5">No listings of this model in the data.</td></tr>`;
  }

  // ---------- PDF report ----------
  $("#downloadBtn").addEventListener("click", async () => {
    if (!lastInputs) return;
    const btn = $("#downloadBtn");
    setLoading(btn, true, "Preparing report");
    try {
      const res = await fetch("/report", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...lastInputs, health: healthAnswers() }),
      });
      if (!res.ok) throw new Error();
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const cd = res.headers.get("Content-Disposition") || "";
      a.href = url; a.download = (cd.match(/filename="?([^"]+)"?/) || [])[1] || "valuation-report.pdf";
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    } catch (e) {
      alert("The report could not be created. Please try again.");
    } finally {
      setLoading(btn, false);
    }
  });

  // =========================================================
  // COMPARE CARS
  // =========================================================
  const makes = Object.keys(OPTIONS.make_models).sort();
  const optionList = (vals) => vals.map((v) => `<option value="${esc(v)}">${esc(title(v))}</option>`).join("");
  function compareFormHTML(slot) {
    return `
      <div class="compare-form__head">
        <h3><span class="car-tag">${slot}</span>Car ${slot}</h3>
        ${slot === "A" ? `<button type="button" class="link-btn" data-fill-from-valuation disabled>Use my valued car</button>`
                       : `<button type="button" class="link-btn" data-clear>Clear</button>`}
      </div>
      <div class="grid">
        <label class="field">Make *<select name="make" required><option value="">Select brand</option>${optionList(makes)}</select></label>
        <label class="field">Model *<select name="model" required disabled><option value="">Select brand first</option></select></label>
        <label class="field">Variant<select name="variant" disabled><option value="">Select model first</option></select></label>
        <label class="field">Body type<select name="body_type"><option value="">Not sure</option>${optionList(OPTIONS.body_type)}</select></label>
        <label class="field">Year *<input type="number" name="yr_mfr" min="1990" max="${THIS_YEAR}" placeholder="e.g. 2018" inputmode="numeric" required></label>
        <label class="field">KM driven *<input type="number" name="kms_run" min="0" placeholder="e.g. 45000" inputmode="numeric" required></label>
        <label class="field">Fuel *<select name="fuel_type" required><option value="">Select fuel</option>${optionList(OPTIONS.fuel_type)}</select></label>
        <label class="field">Transmission *<select name="transmission" required><option value="">Select</option>${optionList(OPTIONS.transmission)}</select></label>
        <label class="field">Owners<select name="total_owners"><option value="1">1st owner</option><option value="2">2nd owner</option><option value="3">3rd owner</option><option value="4">4 or more</option></select></label>
        <label class="field">City<select name="city">${optionList(OPTIONS.city)}<option value="">Other city</option></select></label>
        <label class="field field--full">Asking price (₹)<input type="number" name="asking_price" min="10000" placeholder="Optional, for the verdict" inputmode="numeric"></label>
        <label class="field field--full">Original price when new (₹)<input type="number" name="new_price" min="10000" placeholder="Optional" inputmode="numeric"></label>
      </div>`;
  }
  const cmpA = $("#compareA"), cmpB = $("#compareB");
  [cmpA, cmpB].forEach((f) => {
    f.innerHTML = compareFormHTML(f.dataset.slot);
    wireCarSelects(f); wireFuelLock(f);
    f.addEventListener("submit", (e) => { e.preventDefault(); runCompare(); });
  });
  const fillBtn = $("[data-fill-from-valuation]", cmpA);
  fillBtn.addEventListener("click", () => { if (lastInputs) setCarValues(cmpA, lastInputs); });
  $("[data-clear]", cmpB).addEventListener("click", () => resetForm(cmpB));
  function resetForm(f) {
    f.reset();
    fillSelect($("[name=model]", f), [], "Select brand first");
    fillSelect($("[name=variant]", f), [], "Select model first");
    $$(".field", f).forEach((x) => x.classList.remove("is-invalid"));
  }

  $("#useInCompareBtn").addEventListener("click", () => {
    if (lastInputs) setCarValues(cmpA, lastInputs);
    $("#compare").scrollIntoView({ behavior: "smooth" });
  });

  $("#compareBtn").addEventListener("click", runCompare);
  $("#compareAnotherBtn").addEventListener("click", () => {
    resetForm(cmpB);
    $("#compareResult").hidden = true;
    cmpB.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  async function runCompare() {
    const box = $("#compareError");
    const errs = [
      ...clientValidate(cmpA, REQUIRED).map((e) => "Car A: " + e),
      ...clientValidate(cmpB, REQUIRED).map((e) => "Car B: " + e),
    ];
    showErrors(box, errs);
    if (errs.length) return;
    const btn = $("#compareBtn");
    setLoading(btn, true, "Comparing");
    try {
      const [a, b] = await Promise.all([
        postJSON("/predict", formToObject(cmpA)).catch((e) => { throw { errors: e.errors.map((x) => "Car A: " + x) }; }),
        postJSON("/predict", formToObject(cmpB)).catch((e) => { throw { errors: e.errors.map((x) => "Car B: " + x) }; }),
      ]);
      renderCompare(a, b);
      $("#compareResult").hidden = false;
      $("#compareResult").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showErrors(box, err.errors);
    } finally {
      setLoading(btn, false);
    }
  }

  function renderCompare(a, b) {
    const better = (va, vb, lowerIsBetter) => {
      if (va == null || vb == null || va === vb) return ["", ""];
      const aWins = lowerIsBetter ? va < vb : va > vb;
      return aWins ? ["is-better", ""] : ["", "is-better"];
    };
    const rangeCell = (r) => `<div class="mini-range">${short(r.range_low)} – ${short(r.range_high)}
      <div class="range-bar"><span class="range-bar__band"></span><span class="range-bar__marker"></span></div></div>`;
    const ratio = (r) => (r.car.asking_price ? r.car.asking_price / r.predicted_price : null);
    const rows = [
      ["Make", a.car.make, b.car.make],
      ["Model", a.car.model, b.car.model],
      ["Year", a.car.yr_mfr, b.car.yr_mfr, better(a.car.yr_mfr, b.car.yr_mfr, false)],
      ["KM driven", km(a.car.kms_run), km(b.car.kms_run), better(a.car.kms_run, b.car.kms_run, true)],
      ["Fuel", a.car.fuel_type, b.car.fuel_type],
      ["Transmission", a.car.transmission, b.car.transmission],
      ["Owners", a.car.total_owners, b.car.total_owners, better(a.car.total_owners, b.car.total_owners, true)],
      ["Original price", a.car.new_price ? inr(a.car.new_price) : "–", b.car.new_price ? inr(b.car.new_price) : "–"],
      ["Asking price", a.car.asking_price ? inr(a.car.asking_price) : "–", b.car.asking_price ? inr(b.car.asking_price) : "–"],
      ["Estimated market price", `<strong>${inr(a.predicted_price)}</strong>`, `<strong>${inr(b.predicted_price)}</strong>`],
      ["Estimated market range", rangeCell(a), rangeCell(b), null, true],
      ["Asking vs market value", gapText(a), gapText(b), better(ratio(a), ratio(b), true), false, true],
      ["Valuation status", a.status ? title(a.status.toLowerCase()) : "No asking price", b.status ? title(b.status.toLowerCase()) : "No asking price"],
      ["Recommendation", title(a.recommendation.code.toLowerCase()), title(b.recommendation.code.toLowerCase())],
    ];
    function gapText(r) {
      const x = ratio(r);
      if (x == null) return "No asking price";
      const pct = Math.abs((x - 1) * 100).toFixed(0);
      return x === 1 ? "Equal to market value" : `${pct}% ${x > 1 ? "above" : "below"} market value`;
    }
    const carName = (r) => `${r.car.yr_mfr} ${r.car.make} ${r.car.model}`;
    $("#compareTable").innerHTML = `
      <thead><tr><th></th><th><span class="car-tag">A</span>${esc(carName(a))}</th><th><span class="car-tag" style="background:var(--violet-deep)">B</span>${esc(carName(b))}</th></tr></thead>
      <tbody>${rows.map(([label, va, vb, cls, raw, key]) => {
        const [ca, cb] = cls || ["", ""];
        const cell = (v) => (raw || String(v).startsWith("<") ? v : esc(v));
        return `<tr${key ? ' class="is-key"' : ""}><th scope="row">${label}</th><td class="${ca}">${cell(va)}</td><td class="${cb}">${cell(vb)}</td></tr>`;
      }).join("")}</tbody>`;
    const bars = $$("#compareTable .range-bar");
    positionRange(bars[0], a.range_low, a.range_high, a.predicted_price, null);
    positionRange(bars[1], b.range_low, b.range_high, b.predicted_price, null);

    // Plain-language summary based only on returned numbers.
    const ra = ratio(a), rb = ratio(b);
    let summary;
    if (ra != null && rb != null) {
      const winner = ra < rb ? "A" : "B";
      const lo = Math.min(ra, rb), hi = Math.max(ra, rb);
      const d = (x) => `${Math.abs((x - 1) * 100).toFixed(0)}% ${x <= 1 ? "below" : "above"}`;
      summary = ra === rb
        ? "Both asking prices sit at the same distance from their estimated market values."
        : lo > 1
          ? `Both cars are priced above their estimated market value, but Car ${winner} is closer (${d(lo)} vs ${d(hi)}), so it is the better starting point for negotiation.`
          : `Car ${winner} is the better deal: its asking price is ${d(lo)} its estimated market value, compared with ${d(hi)} for the other car.`;
    } else {
      const hi = a.predicted_price >= b.predicted_price ? "A" : "B";
      summary = `Car ${hi} has the higher estimated market value (difference ${short(Math.abs(a.predicted_price - b.predicted_price))}). Add both asking prices to see which is the better deal.`;
    }
    $("#compareSummary").textContent = summary + " These are estimates; inspect both cars before deciding.";
  }

  // =========================================================
  // HEALTH CHECK
  // =========================================================
  function healthAnswers() {
    const out = {};
    $$("#healthForm input:checked").forEach((i) => { out[i.dataset.group] = i.value; });
    return out;
  }
  function updateHealth() {
    let earned = 0, possible = 0, answered = 0;
    $$("#healthForm .health-item").forEach((fs) => {
      const inputs = $$("input", fs);
      const checked = inputs.find((i) => i.checked);
      if (checked) {
        answered++;
        earned += +checked.dataset.points;
        possible += Math.max(...inputs.map((i) => +i.dataset.points));
      }
    });
    const total = $$("#healthForm .health-item").length;
    const box = $("#healthScore");
    if (!answered) {
      box.dataset.band = "none";
      $("#healthNum").textContent = "–";
      $("#healthBand").textContent = "Answer a few items";
      return;
    }
    const score = Math.round((earned / possible) * 100);
    const band = score >= 80 ? "Excellent" : score >= 60 ? "Good" : "Needs attention";
    box.dataset.band = band;
    $("#healthNum").textContent = score;
    $("#healthBand").textContent = `${band} · ${answered}/${total} answered`;
  }
  $("#healthForm").addEventListener("change", updateHealth);
  $("#healthReset").addEventListener("click", () => { $$("#healthForm input").forEach((i) => { i.checked = false; }); updateHealth(); });

  // =========================================================
  // EMI
  // =========================================================
  const emi = { price: $("#emiPrice"), down: $("#emiDown"), rate: $("#emiRate"), tenure: $("#emiTenure") };
  function calcEmi() {
    const price = Math.max(+emi.price.value || 0, 0);
    const down = clamp(+emi.down.value || 0, 0, price);
    const loan = price - down;
    const n = +emi.tenure.value;
    const r = Math.max(+emi.rate.value || 0, 0) / 12 / 100;
    const monthly = loan === 0 ? 0 : r === 0 ? loan / n : (loan * r * Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1);
    const total = monthly * n;
    const interest = Math.max(total - loan, 0);
    $("#emiMonthly").textContent = inr(monthly);
    $("#emiLoan").textContent = inr(loan);
    $("#emiInterest").textContent = inr(interest);
    $("#emiTotal").textContent = inr(total);
    const sum = loan + interest || 1;
    $(".split-bar__a").style.width = (loan / sum) * 100 + "%";
    $(".split-bar__b").style.width = (interest / sum) * 100 + "%";
  }
  Object.values(emi).forEach((el) => el.addEventListener("input", calcEmi));
  $("#emiForm").addEventListener("submit", (e) => e.preventDefault());
  function syncEmiPrice(r) {
    const price = r.car.asking_price || r.predicted_price;
    emi.price.value = Math.round(price);
    emi.down.value = Math.round(price * 0.2 / 1000) * 1000;
    calcEmi();
  }
  calcEmi();

  // =========================================================
  // CHECKLIST
  // =========================================================
  const boxes = $$(".check__box");
  function updateChecklist() {
    const done = boxes.filter((b) => b.checked).length;
    $("#checkCount").textContent = `Checklist completed: ${done}/${boxes.length}`;
    $("#checkFill").style.width = (done / boxes.length) * 100 + "%";
  }
  boxes.forEach((b) => b.addEventListener("change", updateChecklist));
  updateChecklist();

  // Enable "Use my valued car" once a valuation exists.
  const observer = new MutationObserver(() => { fillBtn.disabled = !lastInputs; });
  observer.observe($("#results"), { attributes: true, attributeFilter: ["hidden"] });
})();
