/* ── Sweep CTC Chart.js visualisation ─────────────────────────── */

var CtcChart = (function () {
  "use strict";

  var chart = null;      /* Chart.js instance */
  var canvasId = null;
  var rawData = null;
  var animTimer = null;
  var VISIBLE_KEYS = ["sweep_ctc_02"];

  /* Friendly labels for each series key */
  var LABELS = {
    sweep_ctc_02: "Async I/O",
  };

  /* Colour palette */
  var COLORS = [
    "#7bf2c9", "#65c9ff", "#ffd280", "#9ec5ff"
  ];

  var LIVE_COLOR = "#65c9ff";
  var LIVE_FILL_COLOR = "rgba(101, 201, 255, 0.22)";
  var BASELINE_COLOR = "#9aa5b5";
  var IDEAL_COLOR = "#ffd280";
  var X_AXIS_MIN = 0;
  var X_AXIS_MAX = 2;

  /* ── Helpers ────────────────────────────────────────────────── */

  function sortedKeys(obj) {
    return VISIBLE_KEYS.filter(function (key) {
      return Object.prototype.hasOwnProperty.call(obj, key);
    });
  }

  function pickVisibleSeries(data) {
    return VISIBLE_KEYS.reduce(function (accumulator, key) {
      if (Object.prototype.hasOwnProperty.call(data, key)) {
        accumulator[key] = data[key];
      }
      return accumulator;
    }, {});
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

  function buildSyncBaselineDataset() {
    return {
      label: "Sync I/O Baseline",
      data: [
        { x: X_AXIS_MIN, y: 1 },
        { x: X_AXIS_MAX, y: 1 },
      ],
      borderColor: BASELINE_COLOR,
      backgroundColor: BASELINE_COLOR,
      borderWidth: 2,
      borderDash: [4, 4],
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
          min: 0.8,
          max: 2.0,
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
        rawData = pickVisibleSeries(json);
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
    datasets.push(buildSyncBaselineDataset(), buildIdealDataset());

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
    datasets.push(buildSyncBaselineDataset(), buildIdealDataset());

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
          label: "Async I/O",
          data: [],
          borderColor: LIVE_COLOR,
          backgroundColor: LIVE_FILL_COLOR,
          borderWidth: 2.5,
          pointRadius: 5,
          pointBackgroundColor: LIVE_COLOR,
          showLine: true,
          fill: true,
          tension: 0.3,
        }, buildSyncBaselineDataset(), buildIdealDataset()],
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
