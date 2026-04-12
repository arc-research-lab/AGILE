/* ── DLRM comparison charts ───────────────────────────────────── */

var DlrmCharts = (function () {
  "use strict";

  var chart = null;
  var canvasId = null;
  var rawData = null;
  var dataUrl = null;
  var COLORS = {
    bam: "#4b82e6",
    sync: "#ffb703",
    async: "#b98144"
  };

  var LABEL_PLUGIN = {
    id: "dlrmValueLabels",
    afterDatasetsDraw: function (chart) {
      var ctx = chart.ctx;
      ctx.save();
      ctx.font = "12px IBM Plex Sans, Segoe UI, sans-serif";
      ctx.fillStyle = "#2f2c28";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";

      chart.data.datasets.forEach(function (dataset, datasetIndex) {
        var meta = chart.getDatasetMeta(datasetIndex);
        if (meta.hidden) {
          return;
        }
        meta.data.forEach(function (bar, index) {
          var value = dataset.data[index];
          if (typeof value !== "number") {
            return;
          }
          ctx.fillText(value.toFixed(2).replace(/\.00$/, ""), bar.x, bar.y - 6);
        });
      });
      ctx.restore();
    }
  };

  var CHART_DEFS = {
    config: {
      title: "DLRM Config Comparison",
      xAxisLabel: "Recommendation Models",
      yMin: 0.5,
      yMax: 2.0,
      key: "configs"
    },
    batch: {
      title: "DLRM Batch Size Sweep",
      xAxisLabel: "Batch Size",
      yMin: 0.95,
      yMax: 1.95,
      key: "batchSizes"
    },
    queue: {
      title: "DLRM NVMe Queue Pair Sweep",
      xAxisLabel: "#IO Queue Pairs",
      yMin: 0.8,
      yMax: 1.6,
      key: "queuePairs"
    },
    cache: {
      title: "DLRM Software Cache Sweep",
      xAxisLabel: "Software Cache Size (MB)",
      yMin: 0.8,
      yMax: 1.9,
      key: "cacheSizes"
    }
  };

  function init(id, url) {
    canvasId = id;
    dataUrl = url;
  }

  function destroyAll() {
    if (chart) {
      chart.destroy();
      chart = null;
    }
  }

  function baseOptions(title, xAxisLabel, yMin, yMax) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 900,
        easing: "easeOutQuart"
      },
      transitions: {
        active: {
          animation: {
            duration: 250
          }
        }
      },
      layout: {
        padding: {
          top: 22,
          right: 16,
          bottom: 4,
          left: 8
        }
      },
      plugins: {
        title: {
          display: true,
          text: title,
          color: "#616161",
          font: {
            size: 16,
            weight: "600"
          },
          padding: {
            bottom: 10
          }
        },
        legend: {
          position: "top",
          labels: {
            color: "#4a4a4a",
            boxWidth: 18,
            boxHeight: 18,
            font: {
              size: 11,
              weight: "600"
            }
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
          grid: {
            color: "rgba(128, 128, 128, 0.16)"
          },
          ticks: {
            color: "#505050",
            maxRotation: 0,
            autoSkip: false,
            font: {
              size: 11
            }
          },
          title: {
            display: true,
            text: xAxisLabel,
            color: "#505050",
            font: {
              size: 13,
              weight: "600"
            }
          }
        },
        y: {
          min: yMin,
          max: yMax,
          grid: {
            color: "rgba(128, 128, 128, 0.16)"
          },
          ticks: {
            color: "#505050",
            stepSize: 0.2
          },
          title: {
            display: true,
            text: ["Speedup Normalized", "to BaM Baseline"],
            color: "#505050",
            font: {
              size: 13,
              weight: "600"
            }
          }
        }
      }
    };
  }

  function makeDataset(label, key, color, rows) {
    return {
      label: label,
      data: rows.map(function (row) { return row[key]; }),
      backgroundColor: color,
      borderColor: color,
      borderWidth: 0,
      categoryPercentage: 0.68,
      barPercentage: 0.68
    };
  }

  function renderChart(title, xAxisLabel, rows, yMin, yMax) {
    var element = document.getElementById(canvasId);
    if (!element) {
      return;
    }

    destroyAll();
    chart = new Chart(element, {
      type: "bar",
      data: {
        labels: rows.map(function (row) { return row.label; }),
        datasets: [
          makeDataset("BaM", "bam", COLORS.bam, rows),
          makeDataset("AGILE (sync)", "agileSync", COLORS.sync, rows),
          makeDataset("AGILE (async)", "agileAsync", COLORS.async, rows)
        ]
      },
      options: baseOptions(title, xAxisLabel, yMin, yMax),
      plugins: [LABEL_PLUGIN]
    });
  }

  function ensureDataLoaded() {
    if (rawData) {
      return Promise.resolve(rawData);
    }

    return fetch(dataUrl + "?t=" + Date.now())
      .then(function (res) {
        if (!res.ok) {
          throw new Error("HTTP " + res.status);
        }
        return res.json();
      })
      .then(function (data) {
        rawData = data;
        return rawData;
      });
  }

  function show(chartKey) {
    var def = CHART_DEFS[chartKey];
    if (!def) {
      return Promise.reject(new Error("Unknown DLRM chart: " + chartKey));
    }

    return ensureDataLoaded()
      .then(function (data) {
        renderChart(def.title, def.xAxisLabel, data[def.key], def.yMin, def.yMax);
      })
      .catch(function (err) {
        console.error("Failed to load DLRM charts:", err);
      });
  }

  return {
    init: init,
    show: show,
    destroyAll: destroyAll
  };
})();
