/* ── Graph test breakdown chart ───────────────────────────────── */

var GraphBreakdownChart = (function () {
  "use strict";

  var chart = null;
  var canvasId = null;
  var resultsByTest = {};

  var TEST_ORDER = [
    "bfs-g20-1",
    "bfs-g20-2",
    "bfs-u20-1",
    "bfs-u20-2",
    "pr-g19-1",
    "pr-g19-2",
    "pr-u19-1",
    "pr-u19-2"
  ];

  function init(domId) {
    canvasId = domId;
  }

  function destroyChart() {
    if (chart) {
      chart.destroy();
      chart = null;
    }
  }

  function reset() {
    resultsByTest = {};
    destroyChart();
  }

  function addResult(result) {
    if (!result || !result.test) {
      return;
    }
    resultsByTest[result.test] = result;
    render();
  }

  function load(url) {
    fetch(url + "?t=" + Date.now())
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then(function (rows) {
        reset();
        rows.forEach(function (row) {
          resultsByTest[row.test] = row;
        });
        render();
      })
      .catch(function (err) {
        console.error("Graph breakdown load failed:", err);
      });
  }

  function orderedResults() {
    return Object.keys(resultsByTest)
      .sort(function (left, right) {
        var leftIndex = TEST_ORDER.indexOf(left);
        var rightIndex = TEST_ORDER.indexOf(right);

        if (leftIndex === -1 && rightIndex === -1) {
          return left.localeCompare(right);
        }
        if (leftIndex === -1) return 1;
        if (rightIndex === -1) return -1;
        return leftIndex - rightIndex;
      })
      .map(function (key) {
        return resultsByTest[key];
      });
  }

  function parseNumber(value) {
    return Number.parseFloat(value);
  }

  function shortLabel(testName) {
    return testName.toUpperCase();
  }

  function computeBreakdown(cold, warm, gpu) {
    var kernel = 1;
    var cache = Math.max((warm - gpu) / gpu, 0);
    var io = Math.max((cold - warm) / gpu, 0);
    return {
      kernel: kernel,
      cache: cache,
      io: io,
      total: kernel + cache + io
    };
  }

  function buildChartModel() {
    var rows = orderedResults();
    var labels = [];
    var kernelData = [];
    var cacheData = [];
    var ioData = [];
    var maxTotal = 1;

    rows.forEach(function (row) {
      var gpu = parseNumber(row.gpu);
      var bam = computeBreakdown(parseNumber(row["bam-cold"]), parseNumber(row["bam-warm"]), gpu);
      var agile = computeBreakdown(parseNumber(row["agile-cold"]), parseNumber(row["agile-warm"]), gpu);
      var testLabel = shortLabel(row.test);

      labels.push(["BaM", testLabel]);
      kernelData.push(bam.kernel);
      cacheData.push(bam.cache);
      ioData.push(bam.io);
      maxTotal = Math.max(maxTotal, bam.total);

      labels.push(["AGILE", testLabel]);
      kernelData.push(agile.kernel);
      cacheData.push(agile.cache);
      ioData.push(agile.io);
      maxTotal = Math.max(maxTotal, agile.total);
    });

    return {
      labels: labels,
      kernelData: kernelData,
      cacheData: cacheData,
      ioData: ioData,
      maxTotal: maxTotal
    };
  }

  function logTickLabel(value) {
    var allowed = [0.5, 1, 2, 4, 8, 16, 32, 64, 128];
    for (var i = 0; i < allowed.length; i += 1) {
      if (Math.abs(value - allowed[i]) < 0.0001) {
        return allowed[i];
      }
    }
    return "";
  }

  function render() {
    if (!canvasId) {
      return;
    }

    var model = buildChartModel();
    if (!model.labels.length) {
      destroyChart();
      return;
    }

    destroyChart();
    chart = new Chart(document.getElementById(canvasId), {
      type: "bar",
      data: {
        labels: model.labels,
        datasets: [
          {
            label: "Kernel",
            data: model.kernelData,
            backgroundColor: "#4b82e6",
            stack: "time"
          },
          {
            label: "Cache API",
            data: model.cacheData,
            backgroundColor: "#b17a3b",
            stack: "time"
          },
          {
            label: "I/O API",
            data: model.ioData,
            backgroundColor: "#ffb000",
            stack: "time"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 350,
          easing: "easeOutCubic"
        },
        layout: {
          padding: {
            top: 8,
            right: 12,
            bottom: 0,
            left: 8
          }
        },
        plugins: {
          title: {
            display: true,
            text: "Execution Time Breakdown Across Graph Tests",
            color: "#ccc",
            font: { size: 16 }
          },
          legend: {
            position: "top",
            labels: {
              color: "#aaa",
              boxWidth: 20,
              font: { size: 11 }
            }
          },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ": " + ctx.parsed.y.toFixed(3) + "x";
              }
            }
          }
        },
        scales: {
          x: {
            stacked: true,
            ticks: {
              color: "#aaa",
              maxRotation: 0,
              autoSkip: false,
              font: { size: 11 }
            },
            grid: {
              color: "rgba(120, 120, 120, 0.18)"
            },
            title: {
              display: true,
              text: "Graph Tests",
              color: "#aaa",
              font: { size: 13 }
            }
          },
          y: {
            type: "logarithmic",
            min: 0.5,
            max: Math.max(128, Math.pow(2, Math.ceil(Math.log2(model.maxTotal)))),
            stacked: true,
            ticks: {
              color: "#aaa",
              callback: function (value) {
                return logTickLabel(value);
              }
            },
            grid: {
              color: "rgba(120, 120, 120, 0.18)"
            },
            title: {
              display: true,
              text: ["Time Breakdown", "Normalized to Kernel"],
              color: "#aaa",
              font: { size: 13 }
            }
          }
        }
      }
    });
  }

  return {
    init: init,
    reset: reset,
    load: load,
    addResult: addResult
  };
})();