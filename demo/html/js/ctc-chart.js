/* ── Sweep CTC Chart.js visualisation ─────────────────────────── */

var CtcChart = (function () {
  "use strict";

  var chart = null;      /* Chart.js instance */
  var canvasId = null;
  var rawData = null;
  var animTimer = null;

  /* Friendly labels for each series key */
  var LABELS = {
    sweep_ctc:    "4-SSD Aggregate",
    sweep_ctc_01: "SSD #1 (run A)",
    sweep_ctc_02: "SSD #2 (run A)",
    sweep_ctc_03: "SSD #3 (run A)",
    sweep_ctc_04: "SSD #4 (run A)",
    sweep_ctc_e1: "SSD #1 (run B)",
    sweep_ctc_e2: "SSD #2 (run B)",
    sweep_ctc_e3: "SSD #3 (run B)",
    sweep_ctc_e4: "SSD #4 (run B)",
  };

  /* Colour palette */
  var COLORS = [
    "#5470c6", "#91cc75", "#fac858", "#ee6666",
    "#73c0de", "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc"
  ];

  var LIVE_COLOR = "#00e5ff";
  var IDEAL_COLOR = "#f567ba";
  var X_AXIS_MIN = 0;
  var X_AXIS_MAX = 2;

  /* ── Helpers ────────────────────────────────────────────────── */

  function sortedKeys(obj) {
    return Object.keys(obj).sort(function (a, b) {
      if (a === "sweep_ctc") return -1;
      if (b === "sweep_ctc") return 1;
      return a < b ? -1 : a > b ? 1 : 0;
    });
  }

  function destroyChart() {
    if (chart) {
      chart.destroy();
      chart = null;
    }
  }

  function idealSpeedup(ctc) {
    if (ctc <= 1) {
      return 1 + ctc;
    }
    return 1 + (1 / ctc);
  }

  function buildIdealPoints() {
    var points = [];
    var samples = 100;
    for (var index = 0; index <= samples; index += 1) {
      var ctc = X_AXIS_MIN + ((X_AXIS_MAX - X_AXIS_MIN) * index) / samples;
      points.push({ x: ctc, y: idealSpeedup(ctc) });
    }
    return points;
  }

  function buildIdealDataset() {
    return {
      label: "Ideal Speedup",
      data: buildIdealPoints(),
      borderColor: IDEAL_COLOR,
      backgroundColor: IDEAL_COLOR,
      borderWidth: 2,
      borderDash: [8, 6],
      pointRadius: 0,
      pointHoverRadius: 0,
      showLine: true,
      fill: false,
      tension: 0,
    };
  }

  /** Shared Chart.js options */
  function baseOptions(titleText) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 400,
        easing: "easeOutCubic",
      },
      plugins: {
        title: {
          display: true,
          text: titleText,
          color: "#ccc",
          font: { size: 16 },
        },
        legend: {
          labels: { color: "#aaa", font: { size: 11 } },
        },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              var p = ctx.parsed;
              return ctx.dataset.label + ": CTC=" + p.x.toFixed(4) +
                     ", Speedup=" + p.y.toFixed(4) + "×";
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          min: X_AXIS_MIN,
          max: X_AXIS_MAX,
          title: { display: true, text: "CTC Ratio", color: "#aaa", font: { size: 13 } },
          ticks: { color: "#aaa" },
          grid: { color: "#333" },
        },
        y: {
          type: "linear",
          title: { display: true, text: "Speedup (×)", color: "#aaa", font: { size: 13 } },
          ticks: { color: "#aaa" },
          grid: { color: "#333" },
        },
      },
    };
  }

  /* ── Public API ─────────────────────────────────────────────── */

  function init(domId) {
    canvasId = domId;
  }

  /** Load saved JSON data and render with progressive animation. */
  function load(url) {
    fetch(url + "?t=" + Date.now())
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then(function (json) {
        rawData = json;
        renderAnimated();
      })
      .catch(function (err) {
        console.error("CTC chart data load failed:", err);
      });
  }

  /** Render all saved data at once. */
  function renderFull() {
    if (!rawData) return;
    stopAnimation();
    destroyChart();

    var keys = sortedKeys(rawData);
    var datasets = keys.map(function (k, i) {
      var isAgg = k === "sweep_ctc";
      return {
        label: LABELS[k] || k,
        data: rawData[k].map(function (r) { return { x: r.ctc, y: r.speedup }; }),
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: COLORS[i % COLORS.length],
        borderWidth: isAgg ? 3 : 1.5,
        pointRadius: isAgg ? 4 : 2,
        pointStyle: isAgg ? "rectRot" : "circle",
        showLine: true,
        fill: false,
        tension: 0.3,
      };
    });
    datasets.push(buildIdealDataset());

    chart = new Chart(document.getElementById(canvasId), {
      type: "scatter",
      data: { datasets: datasets },
      options: baseOptions("Sweep CTC \u2014 Speedup vs. CTC Ratio"),
    });
  }

  /** Render saved data with a progressive left-to-right animation. */
  function renderAnimated() {
    if (!rawData) return;
    stopAnimation();
    destroyChart();

    var keys = sortedKeys(rawData);
    var maxLen = 0;
    keys.forEach(function (k) {
      maxLen = Math.max(maxLen, rawData[k].length);
    });

    /* Create chart with empty datasets. */
    var datasets = keys.map(function (k, i) {
      var isAgg = k === "sweep_ctc";
      return {
        label: LABELS[k] || k,
        data: [],
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: COLORS[i % COLORS.length],
        borderWidth: isAgg ? 3 : 1.5,
        pointRadius: isAgg ? 4 : 2,
        pointStyle: isAgg ? "rectRot" : "circle",
        showLine: true,
        fill: false,
        tension: 0.3,
      };
    });
    datasets.push(buildIdealDataset());

    chart = new Chart(document.getElementById(canvasId), {
      type: "scatter",
      data: { datasets: datasets },
      options: baseOptions("Sweep CTC \u2014 Speedup vs. CTC Ratio"),
    });

    var step = 0;
    var STEP_SIZE = 3;
    var INTERVAL_MS = 60;

    function tick() {
      step += STEP_SIZE;
      if (step > maxLen) step = maxLen;

      keys.forEach(function (k, i) {
        var rows = rawData[k];
        var subset = rows.slice(0, step).map(function (r) {
          return { x: r.ctc, y: r.speedup };
        });
        chart.data.datasets[i].data = subset;
      });
      chart.update("none"); /* skip per-point animation, rely on interval */

      if (step >= maxLen) {
        stopAnimation();
        chart.update(); /* final smooth render */
      }
    }

    tick();
    animTimer = setInterval(tick, INTERVAL_MS);
  }

  function stopAnimation() {
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
    }
  }

  /* ── Live / streaming mode ──────────────────────────────────── */

  /** Clear live data and prepare chart for streaming points. */
  function resetLive() {
    stopAnimation();
    destroyChart();

    chart = new Chart(document.getElementById(canvasId), {
      type: "scatter",
      data: {
        datasets: [{
          label: "Live CTC Sweep",
          data: [],
          borderColor: LIVE_COLOR,
          backgroundColor: "rgba(0,229,255,0.25)",
          borderWidth: 2.5,
          pointRadius: 5,
          pointBackgroundColor: LIVE_COLOR,
          showLine: true,
          fill: true,
          tension: 0.3,
        }, buildIdealDataset()],
      },
      options: baseOptions("Live CTC Sweep \u2014 Speedup vs. CTC Ratio"),
    });
  }

  /** Append a single data point from a WebSocket message. */
  function addPoint(row) {
    var ctc     = parseFloat(row.ctc);
    var speedup = parseFloat(row.speedup);
    if (isNaN(ctc) || isNaN(speedup)) return;
    if (!chart) return;
    chart.data.datasets[0].data.push({ x: ctc, y: speedup });
    chart.update();
  }

  return {
    init: init,
    load: load,
    renderFull: renderFull,
    renderAnimated: renderAnimated,
    stopAnimation: stopAnimation,
    resetLive: resetLive,
    addPoint: addPoint,
  };
})();
