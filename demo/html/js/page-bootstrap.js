(function () {
    var dlrmConfigButton = document.getElementById("dlrm-config-btn");

    if (window.GraphBreakdownChart) {
        GraphBreakdownChart.init("graph-breakdown-chart");
        GraphBreakdownChart.load("data/sweep_results.json");
    }

    if (window.CtcChart) {
        CtcChart.init("ctc-chart");
        CtcChart.load("data/sweep_ctc_data.json");
    }

    if (dlrmConfigButton) {
        dlrmConfigButton.click();
    }
})();