/* ── Graph test breakdown chart ───────────────────────────────── */

var GraphBreakdownChart = (function () {
  "use strict";

  var chart = null;
  var canvasId = null;
  var resultsByTest = {};
  var GROUP_SEPARATOR_PLUGIN = {
    id: "groupSeparator",
    afterDatasetsDraw: function (chartInstance, _args, pluginOptions) {
      var boundaries = pluginOptions && Array.isArray(pluginOptions.boundaries)
        ? pluginOptions.boundaries
        : [];
      var groupLabels = pluginOptions && Array.isArray(pluginOptions.groupLabels)
        ? pluginOptions.groupLabels
        : [];
      var xScale = chartInstance.scales.x;
      var chartArea = chartInstance.chartArea;
      var ctx = chartInstance.ctx;

      if (!xScale || !chartArea) {
        return;
      }

      ctx.save();
      if (boundaries.length) {
        ctx.strokeStyle = "rgba(123, 242, 201, 0.5)";
        ctx.lineWidth = 2;
        ctx.setLineDash([7, 4]);

        boundaries.forEach(function (boundaryIndex) {
          var left = xScale.getPixelForValue(boundaryIndex);
          var right = xScale.getPixelForValue(boundaryIndex + 1);

          if (!Number.isFinite(left) || !Number.isFinite(right)) {
            return;
          }

          var x = (left + right) / 2;
          ctx.beginPath();
          ctx.moveTo(x, chartArea.top + 4);
          ctx.lineTo(x, chartArea.bottom);
          ctx.stroke();
        });
      }

      if (groupLabels.length) {
        ctx.setLineDash([]);
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = "rgba(237, 245, 255, 0.72)";
        ctx.font = "600 10px 'IBM Plex Mono', monospace";

        groupLabels.forEach(function (label, index) {
          var left = xScale.getPixelForValue(index * 2);
          var right = xScale.getPixelForValue(index * 2 + 1);
          var lines = Array.isArray(label) ? label : [label];
          var baseY = chartArea.bottom + 18;

          if (!Number.isFinite(left) || !Number.isFinite(right)) {
            return;
          }

          lines.forEach(function (line, lineIndex) {
            ctx.fillText(line, (left + right) / 2, baseY + lineIndex * 12);
          });
        });
      }

      ctx.restore();
    }
  };

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
  var POWER_TICKS = [1, 2, 4, 8, 16, 32, 64];

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
    var parts = String(testName || "").split("-");
    var app = parts[0] ? parts[0].toUpperCase() : "";
    var family = parts[1] || "";
    var familyLabel = "";

    if (family.charAt(0) === "g") {
      familyLabel = "Kronecker Graph";
    } else if (family.charAt(0) === "u") {
      familyLabel = "Uniform Random Graph";
    } else {
      return String(testName || "").toUpperCase();
    }

    return [app, familyLabel];
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
    var groupBoundaries = [];
    var groupLabels = [];
    var maxTotal = 1;

    rows.forEach(function (row, rowIndex) {
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
      groupLabels.push(testLabel);

      if (rowIndex < rows.length - 1) {
        groupBoundaries.push(labels.length - 1);
      }
    });

    return {
      labels: labels,
      kernelData: kernelData,
      cacheData: cacheData,
      ioData: ioData,
      groupBoundaries: groupBoundaries,
      groupLabels: groupLabels,
      maxTotal: maxTotal
    };
  }

  function logTickLabel(value) {
    var allowed = [1, 2, 4, 8, 16, 32, 64, 128];
    for (var i = 0; i < allowed.length; i += 1) {
      if (Math.abs(value - allowed[i]) < 0.0001) {
        return String(allowed[i]);
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
      plugins: [GROUP_SEPARATOR_PLUGIN],
      data: {
        labels: model.labels,
        datasets: [
          {
            label: "Kernel",
            data: model.kernelData,
            backgroundColor: "#65c9ff",
            stack: "time"
          },
          {
            label: "Cache API",
            data: model.cacheData,
            backgroundColor: "#7bf2c9",
            stack: "time"
          },
          {
            label: "I/O API",
            data: model.ioData,
            backgroundColor: "#ffd280",
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
            bottom: 50,
            left: 8
          }
        },
        plugins: {
          groupSeparator: {
            boundaries: model.groupBoundaries,
            groupLabels: model.groupLabels
          },
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
            mode: "index",
            intersect: false,
            callbacks: {
              title: function (items) {
                if (!items || !items.length) {
                  return "";
                }

                var label = items[0].label;
                return Array.isArray(label) ? label.join(" ") : label;
              },
              label: function (ctx) {
                var dataIndex = ctx.dataIndex;
                var datasetIndex = ctx.datasetIndex;
                var datasetLabel = ctx.dataset.label;
                var barLabel = ctx.chart.data.labels[dataIndex];
                var accumulated = ctx.chart.data.datasets.slice(0, datasetIndex + 1).reduce(function (sum, dataset) {
                  return sum + Number(dataset.data[dataIndex] || 0);
                }, 0);

                var suffix = "";
                var isAgileBar = Array.isArray(barLabel) && barLabel[0] === "AGILE";
                var isComparedSegment = datasetLabel === "Cache API" || datasetLabel === "I/O API";

                if (isAgileBar && isComparedSegment && dataIndex > 0) {
                  var bamAccumulated = ctx.chart.data.datasets.slice(0, datasetIndex + 1).reduce(function (sum, dataset) {
                    return sum + Number(dataset.data[dataIndex - 1] || 0);
                  }, 0);

                  if (accumulated > 0) {
                    suffix = " (" + (bamAccumulated / accumulated).toFixed(2) + "x over BaM)";
                  }
                }

                return datasetLabel + ": " + accumulated.toFixed(2) + suffix;
              },
              footer: function (items) {
                if (!items || !items.length) {
                  return "";
                }

                var dataIndex = items[0].dataIndex;
                var total = items[0].chart.data.datasets.reduce(function (sum, dataset) {
                  var value = Number(dataset.data[dataIndex] || 0);
                  return sum + value;
                }, 0);

                return "Total: " + total.toFixed(3) + "";
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
              callback: function (_value, index) {
                var label = model.labels[index];
                return Array.isArray(label) ? label[0] : label;
              },
              font: { size: 11 }
            },
            grid: {
              color: "rgba(120, 120, 120, 0.18)"
            },
            title: {
              display: false
            }
          },
          y: {
            type: "logarithmic",
            min: 0.5,
            max: Math.max(128, Math.pow(2, Math.ceil(Math.log2(model.maxTotal)))),
            afterBuildTicks: function (axis) {
              axis.ticks = POWER_TICKS
                .filter(function (value) {
                  return value >= axis.min && value <= axis.max;
                })
                .map(function (value) {
                  return { value: value };
                });
            },
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