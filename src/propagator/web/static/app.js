(() => {
  "use strict";

  const ACTION_LABELS = {
    canadair: "Canadair",
    helicopter: "Helicopter",
    waterline_action: "Waterline",
    heavy_action: "Heavy vehicles",
  };
  const ACTION_COLORS = {
    canadair: "#2b6cb0",
    helicopter: "#3182ce",
    waterline_action: "#0e7490",
    heavy_action: "#7c5a24",
  };

  const state = {
    mode: "center",
    center: null,
    ignition: null,
    radiusKm: 15,
    circle: null,
    centerMarker: null,
    ignitionMarker: null,
    jobId: null,
    pollTimer: null,
    bounds: null,
    frameTimes: [],
    statsByTime: {},
    isoLayer: null,
    overlayLayer: null,
    actions: [],
    drawingLayer: null,
    drawingPoints: null,
    _fitted: false,
  };

  const map = L.map("map").setView([42.5, 12.5], 6);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 18,
  }).addTo(map);

  const $ = (id) => document.getElementById(id);

  function bindRange(id, labelId, formatter) {
    const input = $(id);
    const label = $(labelId);
    const update = () => {
      label.textContent = formatter(input.value);
    };
    input.addEventListener("input", update);
    update();
    return input;
  }

  const radiusInput = bindRange("radius", "radius-value", (v) => `${v} km`);
  const cellsizeInput = bindRange("cellsize", "cellsize-value", (v) => `${v} m`);
  const windDirInput = bindRange("wind-dir", "wind-dir-value", (v) => `${v}°`);
  const windSpeedInput = bindRange("wind-speed", "wind-speed-value", (v) => `${v} km/h`);
  const moistureInput = bindRange("moisture", "moisture-value", (v) => `${v}%`);
  const durationInput = bindRange("duration", "duration-value", (v) => `${v} h`);
  const resolutionInput = bindRange("resolution", "resolution-value", (v) => `${v} h`);
  const realizationsInput = bindRange("realizations", "realizations-value", (v) => v);
  const actionTimeInput = bindRange("action-time", "action-time-value", (v) => `${v} h`);
  const spottingInput = $("do-spotting");
  const isochroneInput = $("isochrone-thresholds");

  radiusInput.addEventListener("input", () => {
    state.radiusKm = Number(radiusInput.value);
    if (state.circle) {
      state.circle.setRadius(state.radiusKm * 1000);
    }
    checkReady();
  });

  function setMode(mode) {
    state.mode = mode;
    $("pick-center").classList.toggle("active", mode === "center");
    $("pick-ignition").classList.toggle("active", mode === "ignition");
    $("draw-action-line").classList.toggle("active", mode === "draw-action");
  }

  $("pick-center").addEventListener("click", () => setMode("center"));
  $("pick-ignition").addEventListener("click", () => setMode("ignition"));

  map.on("click", (e) => {
    const { lat, lng } = e.latlng;
    if (state.mode === "center") {
      state.center = { lat, lon: lng };
      if (state.centerMarker) {
        state.centerMarker.setLatLng(e.latlng);
      } else {
        state.centerMarker = L.marker(e.latlng).addTo(map);
      }
      if (state.circle) {
        state.circle.setLatLng(e.latlng);
      } else {
        state.circle = L.circle(e.latlng, {
          radius: state.radiusKm * 1000,
          color: "#c4471e",
          fillOpacity: 0.05,
        }).addTo(map);
      }
      setMode("ignition");
    } else if (state.mode === "ignition") {
      state.ignition = { lat, lon: lng };
      if (state.ignitionMarker) {
        state.ignitionMarker.setLatLng(e.latlng);
      } else {
        state.ignitionMarker = L.marker(e.latlng, {
          icon: L.divIcon({ className: "", html: "★", iconSize: [16, 16] }),
        }).addTo(map);
      }
    } else if (state.mode === "draw-action") {
      state.drawingPoints.push([lat, lng]);
      state.drawingLayer.setLatLngs(state.drawingPoints);
      $("finish-action-line").disabled = state.drawingPoints.length < 2;
    }
    checkReady();
  });

  function checkReady() {
    $("run-button").disabled = !(state.center && state.ignition);
  }

  // --- firefighting actions: draw a line, queue it as an action ---

  $("draw-action-line").addEventListener("click", () => {
    if (state.drawingLayer) map.removeLayer(state.drawingLayer);
    state.drawingPoints = [];
    state.drawingLayer = L.polyline([], { color: "#9c6a0e", dashArray: "5 5", weight: 2 }).addTo(map);
    $("finish-action-line").disabled = true;
    setMode("draw-action");
  });

  $("finish-action-line").addEventListener("click", () => {
    if (!state.drawingPoints || state.drawingPoints.length < 2) return;
    const actionType = $("action-type").value;
    const timeH = Number(actionTimeInput.value);
    const line = state.drawingPoints.slice();
    const layer = L.polyline(line, {
      color: ACTION_COLORS[actionType] || "#9c6a0e",
      weight: 3,
    }).addTo(map);
    const id = `action-${Date.now()}`;
    state.actions.push({ id, action_type: actionType, time_h: timeH, line, layer });
    renderActionsList();

    map.removeLayer(state.drawingLayer);
    state.drawingLayer = null;
    state.drawingPoints = null;
    $("finish-action-line").disabled = true;
    setMode(state.ignition ? "ignition" : "center");
  });

  function renderActionsList() {
    const ul = $("actions-list");
    ul.innerHTML = "";
    for (const action of state.actions) {
      const li = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = `${ACTION_LABELS[action.action_type]} — ${action.time_h} h`;
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.textContent = "Remove";
      removeBtn.title = "Removes this action from the simulation";
      removeBtn.addEventListener("click", () => {
        map.removeLayer(action.layer);
        state.actions = state.actions.filter((a) => a.id !== action.id);
        renderActionsList();
      });
      li.appendChild(label);
      li.appendChild(removeBtn);
      ul.appendChild(li);
    }
  }

  function parseIsochroneThresholds() {
    const parts = isochroneInput.value
      .split(",")
      .map((s) => Number(s.trim()))
      .filter((v) => Number.isFinite(v) && v > 0 && v < 1);
    return parts.length > 0 ? parts : [0.5, 0.75, 0.9];
  }

  function buildRequest() {
    return {
      center_lat: state.center.lat,
      center_lon: state.center.lon,
      radius_km: Number(radiusInput.value),
      cellsize: Number(cellsizeInput.value),
      ignition_lat: state.ignition.lat,
      ignition_lon: state.ignition.lon,
      wind_dir: Number(windDirInput.value),
      wind_speed: Number(windSpeedInput.value),
      moisture: Number(moistureInput.value),
      realizations: Number(realizationsInput.value),
      do_spotting: spottingInput.checked,
      time_limit_h: Number(durationInput.value),
      time_resolution_h: Number(resolutionInput.value),
      isochrone_thresholds: parseIsochroneThresholds(),
      actions: state.actions.map((a) => ({
        action_type: a.action_type,
        time_h: a.time_h,
        line: a.line,
      })),
    };
  }

  function formatErrorDetail(detail, fallback) {
    if (!detail) return fallback;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // FastAPI/Pydantic validation errors: [{loc, msg, type}, ...]
      return detail
        .map((e) => {
          if (e && typeof e === "object") {
            const field = Array.isArray(e.loc) ? e.loc.join(".") : e.loc;
            return field ? `${field}: ${e.msg}` : e.msg || JSON.stringify(e);
          }
          return String(e);
        })
        .join("; ");
    }
    return JSON.stringify(detail);
  }

  async function startSimulation() {
    $("run-button").disabled = true;
    $("warning-banner").classList.remove("visible");
    $("result-panel").classList.remove("visible");
    $("status-panel").classList.add("visible");
    $("status-text").textContent = "Starting…";
    $("progress-fill").style.width = "0%";
    clearResults();

    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildRequest()),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      $("status-text").textContent = `Error: ${formatErrorDetail(body.detail, res.statusText)}`;
      $("run-button").disabled = false;
      return;
    }
    const { job_id } = await res.json();
    state.jobId = job_id;
    state.pollTimer = setInterval(poll, 1500);
  }

  $("run-button").addEventListener("click", startSimulation);

  async function poll() {
    const res = await fetch(`/api/simulate/${state.jobId}`);
    if (!res.ok) return;
    const job = await res.json();

    if (job.warning) {
      $("warning-banner").textContent = job.warning;
      $("warning-banner").classList.add("visible");
    }

    const pct = job.time_limit_s > 0 ? Math.min(100, (100 * job.current_time_s) / job.time_limit_s) : 0;
    $("progress-fill").style.width = `${pct}%`;

    const phaseLabels = {
      pending: "Queued…",
      preparing_data: "Downloading DEM and land cover…",
      running: `Simulating: ${(job.current_time_s / 3600).toFixed(1)} / ${(job.time_limit_s / 3600).toFixed(1)} h`,
      done: "Done.",
      failed: `Error: ${job.error || "unknown"}`,
      cancelled: "Cancelled.",
    };
    $("status-text").textContent = phaseLabels[job.status] || job.status;

    if (job.status === "done") {
      clearInterval(state.pollTimer);
      $("run-button").disabled = false;
      await loadResults();
    } else if (job.status === "failed" || job.status === "cancelled") {
      clearInterval(state.pollTimer);
      $("run-button").disabled = false;
    }
  }

  function clearResults() {
    if (state.overlayLayer) {
      map.removeLayer(state.overlayLayer);
      state.overlayLayer = null;
    }
    if (state.isoLayer) {
      map.removeLayer(state.isoLayer);
      state.isoLayer = null;
    }
    state.frameTimes = [];
    state.statsByTime = {};
    state.bounds = null;
    state._fitted = false;
  }

  async function loadResults() {
    const res = await fetch(`/api/simulate/${state.jobId}/frames`);
    const data = await res.json();
    state.bounds = data.bounds_wgs84; // [west, south, east, north]
    state.frameTimes = data.frame_times_s;
    state.statsByTime = {};
    for (const s of data.stats_history) {
      state.statsByTime[s.time_s] = s;
    }

    $("result-panel").classList.add("visible");
    $("stat-frame-count").textContent = state.frameTimes.length;

    const slider = $("frame-slider");
    slider.min = 0;
    slider.max = Math.max(0, state.frameTimes.length - 1);
    slider.value = slider.max;
    slider.oninput = () => showFrame(state.frameTimes[Number(slider.value)]);

    drawChart();
    if (state.frameTimes.length > 0) {
      showFrame(state.frameTimes[state.frameTimes.length - 1]);
    }
  }

  async function showFrame(timeS) {
    $("frame-value").textContent = `${(timeS / 3600).toFixed(1)} h`;

    if (state.overlayLayer) map.removeLayer(state.overlayLayer);
    if (state.isoLayer) map.removeLayer(state.isoLayer);

    if (state.bounds) {
      const [west, south, east, north] = state.bounds;
      const leafletBounds = [
        [south, west],
        [north, east],
      ];
      state.overlayLayer = L.imageOverlay(
        `/api/simulate/${state.jobId}/frame/${timeS}/image.png`,
        leafletBounds,
        { opacity: 0.85 }
      ).addTo(map);
      if (!state._fitted) {
        map.fitBounds(leafletBounds, { maxZoom: 13 });
        state._fitted = true;
      }
    }

    const res = await fetch(`/api/simulate/${state.jobId}/frame/${timeS}`);
    const frame = await res.json();
    const group = L.layerGroup();
    for (const iso of frame.isochrones) {
      for (const line of iso.coordinates) {
        const latlngs = line.map(([lon, lat]) => [lat, lon]);
        L.polyline(latlngs, {
          color: "#1b2320",
          weight: iso.threshold >= 0.9 ? 2.5 : 1.4,
          opacity: 0.85,
        }).addTo(group);
      }
    }
    state.isoLayer = group.addTo(map);

    const stats = state.statsByTime[timeS];
    if (stats) {
      $("stat-area-mean").textContent = (stats.area_mean / 10000).toFixed(1);
      $("stat-area-50").textContent = (stats.area_50 / 10000).toFixed(1);
      $("stat-n-active").textContent = stats.n_active;
    }
    highlightChartPoint(timeS);
  }

  function drawChart() {
    const svg = $("chart");
    svg.innerHTML = "";
    const times = state.frameTimes;
    if (times.length === 0) return;

    const values = times.map((t) => state.statsByTime[t].area_mean / 10000);
    const maxV = Math.max(...values) * 1.15 || 1;
    const W = 320, H = 140, padL = 30, padR = 8, padT = 8, padB = 18;
    const xf = (i) => padL + (i * (W - padL - padR)) / Math.max(1, times.length - 1);
    const yf = (v) => padT + (H - padT - padB) * (1 - v / maxV);

    const pts = values.map((v, i) => `${xf(i).toFixed(1)},${yf(v).toFixed(1)}`).join(" ");
    const areaPath = `M${xf(0)},${H - padB} L${pts.split(" ").join(" L")} L${xf(values.length - 1)},${H - padB} Z`;

    const ns = "http://www.w3.org/2000/svg";
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", areaPath);
    path.setAttribute("fill", "rgba(196,71,30,0.12)");
    svg.appendChild(path);

    const polyline = document.createElementNS(ns, "polyline");
    polyline.setAttribute("points", pts);
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", "#c4471e");
    polyline.setAttribute("stroke-width", "2");
    svg.appendChild(polyline);

    svg.dataset.xf = JSON.stringify(times.map((_, i) => xf(i)));
    svg.dataset.yf = JSON.stringify(values.map((v) => yf(v)));
  }

  function highlightChartPoint(timeS) {
    const svg = $("chart");
    const idx = state.frameTimes.indexOf(timeS);
    if (idx < 0 || !svg.dataset.xf) return;
    const xs = JSON.parse(svg.dataset.xf);
    const ys = JSON.parse(svg.dataset.yf);
    let dot = svg.querySelector("circle.marker");
    if (!dot) {
      dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("class", "marker");
      dot.setAttribute("r", "3.5");
      dot.setAttribute("fill", "#9c6a0e");
      svg.appendChild(dot);
    }
    dot.setAttribute("cx", xs[idx]);
    dot.setAttribute("cy", ys[idx]);
  }
})();
