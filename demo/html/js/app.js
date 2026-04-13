const storageKeys = {
  host: "websocket-demo-host",
  port: "websocket-demo-port",
};

const $serverHostInput = $("#server-host");
const $serverPortInput = $("#server-port");
const $targetUrl = $("#target-url");
const $statusDot = $("#status-dot");
const $brandStatusDot = $("#brand-status-dot");
const $statusText = $("#status-text");
const $messageInput = $("#message");
const $connectButton = $("#connect");
const $sendButton = $("#send");
const $log = $("#log");

let socket = null;
/* Track which example is currently running so we can disable its button. */
let runningExample = null;

function syncBrandStatusDot(connected) {
  if (!$brandStatusDot.length) {
    return;
  }

  if (connected) {
    $brandStatusDot
      .removeClass("disconnected")
      .addClass("connected")
      .css({
        background: "#7bf2c9",
        boxShadow: "0 0 0 0.25rem rgba(123, 242, 201, 0.16)",
      });
  } else {
    $brandStatusDot
      .removeClass("connected")
      .addClass("disconnected")
      .css({
        background: "rgba(255, 210, 128, 0.78)",
        boxShadow: "0 0 0 0.25rem rgba(255, 210, 128, 0.14)",
      });
  }
}

function loadSavedTarget() {
  const savedHost = localStorage.getItem(storageKeys.host);
  const savedPort = localStorage.getItem(storageKeys.port);

  if (savedHost) {
    $serverHostInput.val(savedHost);
  }
  if (savedPort) {
    $serverPortInput.val(savedPort);
  }
}

function saveTarget() {
  localStorage.setItem(storageKeys.host, $serverHostInput.val().trim());
  localStorage.setItem(storageKeys.port, $serverPortInput.val().trim());
}

function getWebSocketUrl() {
  const host = $serverHostInput.val().trim() || "127.0.0.1";
  const port = $serverPortInput.val().trim() || "9002";
  return `ws://${host}:${port}`;
}

function refreshTargetUrl() {
  saveTarget();
  $targetUrl.text(getWebSocketUrl());
}

function addLogEntry(type, text) {
  if (!$log.length) {
    if (type === "meta") {
      console.info(text);
    } else if (type === "server") {
      console.debug(text);
    } else {
      console.log(text);
    }
    return;
  }

  const $entry = $("<div>")
    .addClass(`entry ${type}`)
    .text(text);
  $log.append($entry);
  $log.scrollTop($log[0].scrollHeight);
}


/* ── Handle structured JSON from the server ─────────────────────── */

function handleServerMessage(raw) {
  let msg;
  try {
    msg = JSON.parse(raw);
  } catch (_) {
    /* Plain text fallback. */
    addLogEntry("server", `Server: ${raw}`);
    return;
  }

  switch (msg.type) {
    case "welcome":
      addLogEntry("meta", "Connected to AGILE demo server");
      if (Array.isArray(msg.examples) && msg.examples.length) {
        
      }
      break;

    case "examples":
      if (Array.isArray(msg.examples)) {
        
      }
      break;

    case "start":
      runningExample = msg.example;
      addLogEntry("meta", msg.data || `Starting ${msg.example}...`);
      break;

    case "output":
      addLogEntry("server", msg.data);
      /* Try to detect CTC sweep data inside output messages. */
      if (typeof msg.data === "string" && msg.data.indexOf("compute_itr") !== -1) {
        try {
          var ctcObj = JSON.parse(msg.data);
          if (ctcObj.ctc !== undefined) {
            CtcChart.addPoint(ctcObj);
          }
        } catch (_e) { /* not JSON, ignore */ }
      }
      break;

    case "done":
      addLogEntry("meta", msg.data || `${msg.example} finished.`);
      runningExample = null;
      $("#graph-btn").prop("disabled", false);
      $("#ctc-run").prop("disabled", false);
      $("#reg-report-btn").prop("disabled", false);
      $("#reg-report-load-btn").prop("disabled", false);
      break;

    case "error":
      addLogEntry("meta", `Error: ${msg.data}`);
      runningExample = null;
      $("#graph-btn").prop("disabled", false);
      $("#ctc-run").prop("disabled", false);
      $("#reg-report-btn").prop("disabled", false);
      $("#reg-report-load-btn").prop("disabled", false);
      break;

    case "bench-result":
      if (typeof msg.data === "object" && msg.data !== null) {
        addLogEntry("server", JSON.stringify(msg.data));
        if (msg.example === "ctc-sweep") {
          CtcChart.addPoint(msg.data);
        } else if (msg.example === "sweep") {
          GraphBreakdownChart.addResult(msg.data);
        }
      }
      break;

    case "reg-report":
      if (Array.isArray(msg.data)) {
        renderRegReportTable(msg.data);
        addLogEntry("meta", "Register report: " + msg.data.length + " kernels");
      }
      break;

    case "echo":
      addLogEntry("server", `Echo: ${msg.data}`);
      break;

    default:
      /* Detect bare CTC sweep result (no type field, has compute_itr). */
      if (msg.compute_itr !== undefined && msg.ctc !== undefined) {
        addLogEntry("server", raw);
        CtcChart.addPoint(msg);
        break;
      }
      if (msg.test !== undefined && msg.gpu !== undefined) {
        addLogEntry("server", raw);
        GraphBreakdownChart.addResult(msg);
        break;
      }
      addLogEntry("server", raw);
  }
}

function updateStatus(connected) {
  if (connected) {
    $statusDot.removeClass("disconnected").addClass("connected");
    syncBrandStatusDot(true);
    $statusText.text("Connected");
  } else {
    $statusDot.removeClass("connected").addClass("disconnected");
    syncBrandStatusDot(false);
    $statusText.text("Disconnected");
  }
}

/* ── Connection management ──────────────────────────────────────── */

function connect() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
  }

  const url = getWebSocketUrl();
  socket = new WebSocket(url);
  addLogEntry("meta", `Connecting to ${url} ...`);



  socket.addEventListener("open", () => {
    updateStatus(true);
    addLogEntry("meta", `Connection opened to ${url}`);
  });

  socket.addEventListener("message", (event) => {
    handleServerMessage(event.data);
  });

  socket.addEventListener("close", () => {
    updateStatus(false);
    runningExample = null;
    addLogEntry("meta", "Connection closed");
  });

  socket.addEventListener("error", () => {
    addLogEntry("meta", `WebSocket error while connecting to ${url}`);
  });
}

/* ── Send actions ───────────────────────────────────────────────── */

/* Build extra CLI args from the BFS configuration panel. */
function getBfsArgs(exampleId) {
  const info   = $("#bfs-info").val().trim();
  const start  = $("#bfs-start").val().trim();
  const output = $("#bfs-output").val().trim();

  let args = "";
  if (info)   args += " -i " + info;
  if (start)  args += " -s " + start;
  if (output) args += " -o " + output;

  if (exampleId === "bfs") {
    const dev = $("#bfs-agile-dev").val().trim();
    if (dev) args += " -d " + dev;
  } else if (exampleId === "bfs-bam") {
    const dev = $("#bfs-bam-dev").val().trim();
    if (dev) args += " -d " + dev;
  }
  return args.trim();
}

function runExample(exampleId) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    addLogEntry("meta", "Connect to the server first");
    return;
  }
  let extra = "";
  if (exampleId === "bfs" || exampleId === "bfs-bam") {
    extra = getBfsArgs(exampleId);
  }
  const payload = JSON.stringify({ action: "run", example: exampleId, args: extra });
  socket.send(payload);
  addLogEntry("client", `Run: ${exampleId}` + (extra ? ` (${extra})` : ""));
}

function sendMessage() {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    addLogEntry("meta", "Connect to the server first");
    return;
  }

  const message = $messageInput.val().trim();
  if (!message) {
    return;
  }

  socket.send(message);
  addLogEntry("client", `Client: ${message}`);
}

/* ── Event bindings ────────────────────────────────────────────── */

$connectButton.on("click", connect);
$sendButton.on("click", sendMessage);
$serverHostInput.on("input", refreshTargetUrl);
$serverPortInput.on("input", refreshTargetUrl);
$messageInput.on("keydown", (event) => {
  if (event.key === "Enter") {
    sendMessage();
  }
});


loadSavedTarget();
refreshTargetUrl();
updateStatus(false);
addLogEntry("meta", "Press Connect after starting the C++ server");
DlrmCharts.init("dlrm-chart-canvas", "data/dlrm_charts.json");

function setActiveDlrmButton(activeId) {
  $(".dlrm-controls .btn").removeClass("is-active");
  $(activeId).addClass("is-active");
}

function showDlrmChart(chartKey, buttonId) {
  $("#dlrm-chart-wrap").show();
  setActiveDlrmButton(buttonId);
  DlrmCharts.show(chartKey);
}

/* ── Register report ────────────────────────────────────────── */

function runRegReport() {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    addLogEntry("meta", "Connect to the server first");
    return;
  }
  $("#reg-report-table tbody").empty();
  $("#reg-report-table").removeClass("has-rows");
  $("#reg-report-btn").prop("disabled", true);
  $("#reg-report-load-btn").prop("disabled", true);
  socket.send(JSON.stringify({ action: "reg-report" }));
}

function loadSavedRegReport() {
  addLogEntry("meta", "Loading saved register report...");
  $("#reg-report-table tbody").empty();
  $("#reg-report-table").removeClass("has-rows");
  $("#reg-report-load-btn").prop("disabled", true);

  fetch("data/reg_report_data.json?t=" + Date.now())
    .then((res) => {
      if (!res.ok) throw new Error("HTTP " + res.status);
      return res.json();
    })
    .then((rows) => {
      if (!Array.isArray(rows) || rows.length === 0) {
        addLogEntry("meta", "No saved register report found.");
        $("#reg-report-table").removeClass("has-rows");
        return;
      }

      renderRegReportTable(rows);
      addLogEntry("meta", "Loaded saved register report.");
    })
    .catch((err) => {
      addLogEntry("meta", "Failed to load saved register report: " + err.message);
    })
    .finally(() => {
      $("#reg-report-load-btn").prop("disabled", false);
    });
}

function renderRegReportTable(rows) {
  var $table = $("#reg-report-table");
  var $tbody = $table.find("tbody").empty();
  rows.forEach(function (r) {
    $tbody.append(
      "<tr>" +
        "<td>" + $('<span>').text(r.label).html() + "</td>" +
        "<td><code>" + $('<span>').text(r.kernel).html() + "</code></td>" +
        "<td>" + r.regs + "</td>" +
        "<td>" + r.cmem + "</td>" +
      "</tr>"
    );
  });
  $table.toggleClass("has-rows", rows.length > 0);
  $("#reg-report-wrap").show();
  $("#reg-report-btn").prop("disabled", false);
}

$("#reg-report-btn").on("click", function () {
  runRegReport();
});

$("#reg-report-load-btn").on("click", function () {
  loadSavedRegReport();
});

$("#dlrm-config-btn").on("click", function () {
  showDlrmChart("config", "#dlrm-config-btn");
});

$("#dlrm-batch-btn").on("click", function () {
  showDlrmChart("batch", "#dlrm-batch-btn");
});

$("#dlrm-queue-btn").on("click", function () {
  showDlrmChart("queue", "#dlrm-queue-btn");
});

$("#dlrm-cache-btn").on("click", function () {
  showDlrmChart("cache", "#dlrm-cache-btn");
});

/* ── Sweep results ─────────────────────────────────────────────── */

function loadSweepResults() {
  addLogEntry("meta", "Loading sweep results...");
  fetch("data/sweep_results.json?t=" + Date.now())
    .then((res) => {
      if (!res.ok) throw new Error("HTTP " + res.status);
      return res.json();
    })
    .then((results) => {
      if (!Array.isArray(results) || results.length === 0) {
        addLogEntry("meta", "No sweep results found.");
        return;
      }

      $("#graph-chart-wrap").removeClass("is-hidden").show();
      GraphBreakdownChart.init("graph-breakdown-chart");
      GraphBreakdownChart.reset();

      results.forEach((r) => {
        addLogEntry("server", JSON.stringify(r));
        GraphBreakdownChart.addResult(r);
      });
      addLogEntry("meta", `Loaded ${results.length} sweep results.`);
    })
    .catch((err) => {
      addLogEntry("meta", "Failed to load sweep results: " + err.message);
    });
}

function runSweep(args) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    addLogEntry("meta", "Connect to the server first");
    return;
  }

  $("#graph-chart-wrap").removeClass("is-hidden").show();
  GraphBreakdownChart.init("graph-breakdown-chart");
  GraphBreakdownChart.reset();

  const payload = JSON.stringify({ action: "sweep", args: args || "" });
  socket.send(payload);
  $("#graph-btn").prop("disabled", true);
  addLogEntry("client", `Run: sweep` + (args ? ` (${args})` : ""));
}

$("#graph-btn").on("click", function () {
  runSweep("");
});
$("#sweep-results-btn").on("click", function () {
  loadSweepResults();
});

/* ── CTC sweep ─────────────────────────────────────────────────── */

function runCtcSweep() {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    addLogEntry("meta", "Connect to the server first");
    return;
  }

  /* Show chart and reset for live streaming */
  $("#ctc-chart-wrap").removeClass("is-hidden").show();
  CtcChart.init("ctc-chart");
  CtcChart.resetLive();

  console.log("Initialized CTC chart for live updates.");
  const payload = JSON.stringify({ action: "ctc-sweep" , bdf : "0000:e2:00.0"});
  socket.send(payload);
  $("#ctc-run").prop("disabled", true);
  addLogEntry("client", "Run: ctc-sweep");
}

$("#ctc-run").on("click", function () {
  runCtcSweep();
});

function loadSavedCtcSweep() {
  $("#ctc-chart-wrap").removeClass("is-hidden").show();
  CtcChart.init("ctc-chart");
  CtcChart.load("data/sweep_ctc_data.json");
  addLogEntry("meta", "Loaded saved ctc-sweep data");
}

/* ── CTC Chart (Chart.js) – load saved data ────────────────────── */

$("#ctc-chart-btn").on("click", function () {
  loadSavedCtcSweep();
});


