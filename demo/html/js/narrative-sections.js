(function () {
    var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var animeLib = window.anime;
    var memoryChart = document.getElementById("memoryWallChart");
    var memoryWallSection = document.getElementById("memoryWallSection");
    var memoryTrendModel = memoryChart ? memoryChart.querySelector(".memory-trend-model") : null;
    var memoryTrendMemory = memoryChart ? memoryChart.querySelector(".memory-trend-memory") : null;
    var systemFigure = document.getElementById("systemFlowFigure");   /* SVG overlay for flows */
    var systemScene  = document.getElementById("systemFlowScene");     /* outer scene container */
    var systemSection = document.getElementById("systemStorySection");
    var syncModelSection = document.getElementById("syncModelSection");
    var asyncModelSection = document.getElementById("asyncModelSection");
    var systemReplayButton = document.getElementById("systemReplay");
    var syncModelReplayButton = document.getElementById("syncModelReplay");
    var asyncModelReplayButton = document.getElementById("asyncModelReplay");
    var systemNote = document.getElementById("systemStoryNote");
    var systemCards = systemSection ? Array.from(systemSection.querySelectorAll(".io-case-card")) : [];
    var systemStepButtons = Array.from(document.querySelectorAll(".system-step"));
    var systemNotes = [
        "Multiple GPUs are provisioned to serve a large model, but capacity pressure means the system is memory-bound before it is fully compute-bound.",
        "With ZeRO-Infinity style offload, the GPU still waits for CPU orchestration: it synchronizes first, then the CPU issues DRAM and SSD transfers."
    ];
    var scrollytellingSection = document.getElementById("scrollytelling");
    var memoryChartPlayed = false;
    var systemStoryPlayed = false;
    var syncModelPlayed = false;
    var asyncModelPlayed = false;
    var systemTimeline = null;
    var syncModelTimeline = null;
    var asyncModelTimeline = null;
    var memoryWallSnapTimer = null;
    var isMemoryWallSnapping = false;
    var systemStorySnapTimer = null;
    var isSystemStorySnapping = false;
    var syncModelSnapTimer = null;
    var isSyncModelSnapping = false;
    var asyncModelSnapTimer = null;
    var isAsyncModelSnapping = false;
    var systemAnimationTimers = [];
    /* outline/bus: no SVG outlines in new scene — empty array keeps JS logic intact */
    var outlineElements = [];
    /* 3D HTML component nodes (queried from the scene container, not the SVG) */
    var sceneRoot = systemScene || document.getElementById("systemStorySection");
    var gpuNodes  = sceneRoot ? Array.from(sceneRoot.querySelectorAll(".gpu-node"))  : [];
    var cpuNodes  = sceneRoot ? Array.from(sceneRoot.querySelectorAll(".cpu-node"))  : [];
    var dramNodes = sceneRoot ? Array.from(sceneRoot.querySelectorAll(".dram-node")) : [];
    var ssdNodes  = sceneRoot ? Array.from(sceneRoot.querySelectorAll(".ssd-node"))  : [];
    /* SVG flow elements are still inside the SVG overlay */
    var flowElements = systemFigure ? Array.from(systemFigure.querySelectorAll(".system-flow"))       : [];
    var flowLabels   = systemFigure ? Array.from(systemFigure.querySelectorAll(".system-flow-label")) : [];
    var captions     = systemFigure ? Array.from(systemFigure.querySelectorAll(".system-caption"))    : [];
    /* CPU pulse ring is now an HTML element inside the cpu-node */
    var cpuPulse = sceneRoot ? sceneRoot.querySelector(".hw-pulse") : null;
    /* Util fill bars are HTML elements inside gpu-node cards */
    var softUtilBars = sceneRoot ? Array.from(sceneRoot.querySelectorAll(".hw-util-fill")) : [];
    var syncCodeLines = syncModelSection ? Array.from(syncModelSection.querySelectorAll(".sync-code-line")) : [];
    var syncBlockingChip = syncModelSection ? syncModelSection.querySelector(".sync-code-chip-block") : null;
    var syncCodeWait = document.getElementById("syncCodeWait");
    var syncBatchRows = syncModelSection ? Array.from(syncModelSection.querySelectorAll(".sync-batch-row")) : [];
    var syncTimelineBlocks = syncModelSection ? Array.from(syncModelSection.querySelectorAll(".sync-block")) : [];
    var syncTimelineProgress = syncModelSection ? Array.from(syncModelSection.querySelectorAll(".sync-block-progress")) : [];
    var syncTimelineNote = document.getElementById("syncTimelineNote");
    var asyncCodeLines = asyncModelSection ? Array.from(asyncModelSection.querySelectorAll(".async-code-line")) : [];
    var asyncPrefetchChip = asyncModelSection ? asyncModelSection.querySelector(".async-code-chip-prefetch") : null;
    var asyncOverlapChip = asyncModelSection ? asyncModelSection.querySelector(".async-code-chip-overlap") : null;
    var asyncCodeLive = document.getElementById("asyncCodeLive");
    var asyncBatchRows = asyncModelSection ? Array.from(asyncModelSection.querySelectorAll(".async-batch-row")) : [];
    var asyncTimelineBlocks = asyncModelSection ? Array.from(asyncModelSection.querySelectorAll(".async-block")) : [];
    var asyncTimelineProgress = asyncModelSection ? Array.from(asyncModelSection.querySelectorAll(".async-block-progress")) : [];
    var asyncTimelineNote = document.getElementById("asyncTimelineNote");
    var asyncBufferPills = asyncModelSection ? Array.from(asyncModelSection.querySelectorAll(".async-buffer-pill")) : [];
    var memoryWallLayout = {
        xMin: 118,
        xMax: 780,
        yMin: 466,
        yMax: 86,
        yearMin: 2015.9,
        yearMax: 2022.2,
        valueMin: 0.01,
        valueMax: 10000
    };
    var memoryWallData = {
        cnn: [
            { year: 2015.94, value: 0.025, labelDx: -28, labelDy: -10 },
            { year: 2016.14, value: 0.084, labelDx: -62, labelDy: -18 },
            { year: 2016.90, value: 0.083, labelDx: -34, labelDy: -18 },
            { year: 2016.95, value: 0.027, labelDx: -20, labelDy: -10 }
        ],
        models: [
            { year: 2017.92, value: 0.065, labelDx: -40, labelDy: -20 },
            { year: 2018.45, value: 0.117, labelDx: -20, labelDy: -18 },
            { year: 2018.78, value: 0.34, labelDx: -20, labelDy: -18 },
            { year: 2019.12, value: 2.5, labelDx: -20, labelDy: -18 },
            { year: 2019.62, value: 15, labelDx: -60, labelDy: -18 },
            { year: 2020.10, value: 30, labelDx: -78, labelDy: -18 },
            { year: 2020.35, value: 400, labelDx: -22, labelDy: -18 },
            { year: 2020.46, value: 1600, labelDx: -24, labelDy: -18 },
            { year: 2021.03, value: 4600, labelDx: -60, labelDy: -18 },
            { year: 2021.68, value: 1600, labelDx: -60, labelDy: 20 }
        ],
        memory: [
            { year: 2016.42, memoryGb: 12, plotValue: 12 / 4, labelDx: -48, labelDy: -18 },
            { year: 2017.35, memoryGb: 16, plotValue: 16 / 4, labelDx: -56, labelDy: 26 },
            { year: 2017.45, memoryGb: 32, plotValue: 32 / 4, labelDx: -36, labelDy: -18 },
            { year: 2018.35, memoryGb: 32, plotValue: 32 / 4, labelDx: -54, labelDy: 20 },
            { year: 2020.90, memoryGb: 80, plotValue: 80 / 4, labelDx: -20, labelDy: -18 },
            { year: 2020.35, memoryGb: 40, plotValue: 40 / 4, labelDx: -26, labelDy: 28 },
            { year: 2022.20, memoryGb: 80, plotValue: 80 / 4, labelDx: -42, labelDy: -18 }
        ],
        outlier: [
            { year: 2020.44, value: 20000, labelDx: -92, labelDy: -18 }
        ]
    };

    function scaleMemoryWallX(year) {
        var range = memoryWallLayout.yearMax - memoryWallLayout.yearMin;
        return memoryWallLayout.xMin + ((year - memoryWallLayout.yearMin) / range) * (memoryWallLayout.xMax - memoryWallLayout.xMin);
    }

    function scaleMemoryWallY(value) {
        var logMin = Math.log10(memoryWallLayout.valueMin);
        var logMax = Math.log10(memoryWallLayout.valueMax);
        var ratio = (Math.log10(value) - logMin) / (logMax - logMin);
        return memoryWallLayout.yMin - ratio * (memoryWallLayout.yMin - memoryWallLayout.yMax);
    }

    function buildSmoothPath(samples) {
        if (!samples.length) {
            return "";
        }

        if (samples.length === 1) {
            return "M " + samples[0].x + " " + samples[0].y;
        }

        var path = "M " + samples[0].x + " " + samples[0].y;
        var index;

        for (index = 1; index < samples.length - 1; index += 1) {
            var midX = (samples[index].x + samples[index + 1].x) / 2;
            var midY = (samples[index].y + samples[index + 1].y) / 2;
            path += " Q " + samples[index].x + " " + samples[index].y + " " + midX + " " + midY;
        }

        var lastControl = samples[samples.length - 2];
        var lastPoint = samples[samples.length - 1];
        path += " Q " + lastControl.x + " " + lastControl.y + " " + lastPoint.x + " " + lastPoint.y;

        return path;
    }

    function buildLinearPath(samples) {
        if (!samples.length) {
            return "";
        }

        if (samples.length === 1) {
            return "M " + samples[0].x + " " + samples[0].y;
        }

        var xSum = 0;
        var ySum = 0;
        var xySum = 0;
        var xxSum = 0;
        var minX = samples[0].x;
        var maxX = samples[0].x;

        samples.forEach(function (sample) {
            xSum += sample.x;
            ySum += sample.y;
            xySum += sample.x * sample.y;
            xxSum += sample.x * sample.x;
            minX = Math.min(minX, sample.x);
            maxX = Math.max(maxX, sample.x);
        });

        var count = samples.length;
        var denominator = count * xxSum - xSum * xSum;
        if (Math.abs(denominator) < 1e-9) {
            return "M " + minX + " " + samples[0].y + " L " + maxX + " " + samples[samples.length - 1].y;
        }

        var slope = (count * xySum - xSum * ySum) / denominator;
        var intercept = (ySum - slope * xSum) / count;
        var startY = slope * minX + intercept;
        var endY = slope * maxX + intercept;

        return "M " + minX + " " + startY + " L " + maxX + " " + endY;
    }

    function positionMemorySeries(points, labels, data, valueAccessor) {
        data.forEach(function (entry, index) {
            var value = valueAccessor(entry);
            var x = scaleMemoryWallX(entry.year);
            var y = scaleMemoryWallY(value);
            var point = points[index];
            var label = labels[index];

            if (point) {
                point.setAttribute("cx", x.toFixed(1));
                point.setAttribute("cy", y.toFixed(1));
            }

            if (label) {
                label.setAttribute("x", (x + (entry.labelDx || 0)).toFixed(1));
                label.setAttribute("y", (y + (entry.labelDy || 0)).toFixed(1));
            }
        });
    }

    function renderMemoryWallChart() {
        if (!memoryChart) {
            return;
        }

        positionMemorySeries(
            Array.from(memoryChart.querySelectorAll(".memory-series-cnn .memory-point")),
            Array.from(memoryChart.querySelectorAll(".memory-series-cnn .memory-label")),
            memoryWallData.cnn,
            function (entry) { return entry.value; }
        );

        positionMemorySeries(
            Array.from(memoryChart.querySelectorAll(".memory-series-models .memory-point")),
            Array.from(memoryChart.querySelectorAll(".memory-series-models .memory-label")),
            memoryWallData.models,
            function (entry) { return entry.value; }
        );

        positionMemorySeries(
            Array.from(memoryChart.querySelectorAll(".memory-series-memory .memory-point")),
            Array.from(memoryChart.querySelectorAll(".memory-series-memory .memory-label")),
            memoryWallData.memory,
            function (entry) { return entry.plotValue; }
        );

        positionMemorySeries(
            Array.from(memoryChart.querySelectorAll(".memory-series-outlier .memory-point")),
            Array.from(memoryChart.querySelectorAll(".memory-series-outlier .memory-label")),
            memoryWallData.outlier,
            function (entry) { return entry.value; }
        );

        if (memoryTrendModel) {
            memoryTrendModel.setAttribute("d", buildLinearPath(memoryWallData.models.map(function (entry) {
                return {
                    x: scaleMemoryWallX(entry.year),
                    y: scaleMemoryWallY(entry.value)
                };
            })));
            memoryTrendModel.dataset.strokeReady = "false";
        }

        if (memoryTrendMemory) {
            memoryTrendMemory.setAttribute("d", buildLinearPath(memoryWallData.memory.map(function (entry) {
                return {
                    x: scaleMemoryWallX(entry.year),
                    y: scaleMemoryWallY(entry.plotValue)
                };
            })));
            memoryTrendMemory.dataset.strokeReady = "false";
        }
    }

    function prepareStroke(node) {
        if (!node || node.dataset.strokeReady === "true" || typeof node.getTotalLength !== "function") {
            return;
        }

        var length = node.getTotalLength();
        node.dataset.strokeLength = String(length);
        node.style.strokeDasharray = length + "px";
        node.style.strokeDashoffset = length + "px";
        node.dataset.strokeReady = "true";
    }

    function resetStroke(node, visible) {
        if (!node) {
            return;
        }

        prepareStroke(node);

        if (node.dataset.strokeLength) {
            node.style.strokeDashoffset = visible ? "0px" : node.dataset.strokeLength + "px";
        }
    }

    function getCenteredScrollTop(section) {
        var sectionRect = section.getBoundingClientRect();
        var absoluteTop = window.scrollY + sectionRect.top;
        return Math.max(0, absoluteTop - Math.max(window.innerHeight - section.offsetHeight, 0) / 2);
    }

    function maybeSnapMemoryWall() {
        if (reducedMotion || !memoryWallSection || !systemSection || isMemoryWallSnapping) {
            return;
        }

        var memoryRect = memoryWallSection.getBoundingClientRect();
        var systemRect = systemSection.getBoundingClientRect();
        var viewportHeight = window.innerHeight;
        var visibleTop = Math.max(memoryRect.top, 0);
        var visibleBottom = Math.min(memoryRect.bottom, viewportHeight);
        var visibleHeight = Math.max(visibleBottom - visibleTop, 0);
        var visibleRatio = visibleHeight / Math.min(memoryWallSection.offsetHeight, viewportHeight);

        if (visibleRatio < 0.55 || systemRect.top < viewportHeight * 0.2) {
            return;
        }

        var targetTop = getCenteredScrollTop(memoryWallSection);
        if (Math.abs(window.scrollY - targetTop) < 18) {
            return;
        }

        isMemoryWallSnapping = true;
        window.scrollTo({
            top: targetTop,
            behavior: "smooth"
        });

        window.setTimeout(function () {
            isMemoryWallSnapping = false;
        }, 420);
    }

    function queueMemoryWallSnap() {
        if (reducedMotion || !memoryWallSection) {
            return;
        }

        window.clearTimeout(memoryWallSnapTimer);
        memoryWallSnapTimer = window.setTimeout(maybeSnapMemoryWall, 140);
    }

    function maybeSnapSystemStory() {
        if (reducedMotion || !systemSection || isSystemStorySnapping) {
            return;
        }

        var systemRect = systemSection.getBoundingClientRect();
        var scrollyRect = scrollytellingSection ? scrollytellingSection.getBoundingClientRect() : { top: Infinity };
        var viewportHeight = window.innerHeight;
        var visibleTop = Math.max(systemRect.top, 0);
        var visibleBottom = Math.min(systemRect.bottom, viewportHeight);
        var visibleHeight = Math.max(visibleBottom - visibleTop, 0);
        var visibleRatio = visibleHeight / Math.min(systemSection.offsetHeight, viewportHeight);

        if (visibleRatio < 0.55 || scrollyRect.top < viewportHeight * 0.2) {
            return;
        }

        var targetTop = getCenteredScrollTop(systemSection);
        if (Math.abs(window.scrollY - targetTop) < 18) {
            return;
        }

        isSystemStorySnapping = true;
        window.scrollTo({
            top: targetTop,
            behavior: "smooth"
        });

        window.setTimeout(function () {
            isSystemStorySnapping = false;
        }, 420);
    }

    function queueSystemStorySnap() {
        if (reducedMotion || !systemSection) {
            return;
        }

        window.clearTimeout(systemStorySnapTimer);
        systemStorySnapTimer = window.setTimeout(maybeSnapSystemStory, 140);
    }

    function maybeSnapSyncModel() {
        if (reducedMotion || !syncModelSection || isSyncModelSnapping) {
            return;
        }

        var syncRect = syncModelSection.getBoundingClientRect();
        var asyncRect = asyncModelSection ? asyncModelSection.getBoundingClientRect() : { top: Infinity };
        var viewportHeight = window.innerHeight;
        var visibleTop = Math.max(syncRect.top, 0);
        var visibleBottom = Math.min(syncRect.bottom, viewportHeight);
        var visibleHeight = Math.max(visibleBottom - visibleTop, 0);
        var visibleRatio = visibleHeight / Math.min(syncModelSection.offsetHeight, viewportHeight);

        if (visibleRatio < 0.55 || asyncRect.top < viewportHeight * 0.2) {
            return;
        }

        var targetTop = getCenteredScrollTop(syncModelSection);
        if (Math.abs(window.scrollY - targetTop) < 18) {
            return;
        }

        isSyncModelSnapping = true;
        window.scrollTo({
            top: targetTop,
            behavior: "smooth"
        });

        window.setTimeout(function () {
            isSyncModelSnapping = false;
        }, 420);
    }

    function queueSyncModelSnap() {
        if (reducedMotion || !syncModelSection) {
            return;
        }

        window.clearTimeout(syncModelSnapTimer);
        syncModelSnapTimer = window.setTimeout(maybeSnapSyncModel, 140);
    }

    function maybeSnapAsyncModel() {
        if (reducedMotion || !asyncModelSection || isAsyncModelSnapping) {
            return;
        }

        var asyncRect = asyncModelSection.getBoundingClientRect();
        var scrollyRect = scrollytellingSection ? scrollytellingSection.getBoundingClientRect() : { top: Infinity };
        var viewportHeight = window.innerHeight;
        var visibleTop = Math.max(asyncRect.top, 0);
        var visibleBottom = Math.min(asyncRect.bottom, viewportHeight);
        var visibleHeight = Math.max(visibleBottom - visibleTop, 0);
        var visibleRatio = visibleHeight / Math.min(asyncModelSection.offsetHeight, viewportHeight);

        if (visibleRatio < 0.55 || scrollyRect.top < viewportHeight * 0.2) {
            return;
        }

        var targetTop = getCenteredScrollTop(asyncModelSection);
        if (Math.abs(window.scrollY - targetTop) < 18) {
            return;
        }

        isAsyncModelSnapping = true;
        window.scrollTo({
            top: targetTop,
            behavior: "smooth"
        });

        window.setTimeout(function () {
            isAsyncModelSnapping = false;
        }, 420);
    }

    function queueAsyncModelSnap() {
        if (reducedMotion || !asyncModelSection) {
            return;
        }

        window.clearTimeout(asyncModelSnapTimer);
        asyncModelSnapTimer = window.setTimeout(maybeSnapAsyncModel, 140);
    }

    function setSystemStep(stage) {
        systemStepButtons.forEach(function (button, index) {
            var isActive = index === stage;
            button.classList.toggle("is-active", isActive);
            button.setAttribute("aria-selected", isActive ? "true" : "false");
        });

        if (systemNote) {
            systemNote.textContent = systemNotes[stage] || "";
        }
    }

    function setNodesActive(nodes, isActive) {
        nodes.forEach(function (node) {
            node.classList.toggle("is-active", isActive);
        });
    }

    function setSystemCaption(stage) {
        captions.forEach(function (caption, index) {
            caption.classList.toggle("is-visible", index === stage);
        });
    }

    function setSystemFlowLabels(visible) {
        flowLabels.forEach(function (label) {
            label.classList.toggle("is-visible", visible);
        });
    }

    function clearSystemAnimationTimers() {
        systemAnimationTimers.forEach(function (timerId) {
            window.clearTimeout(timerId);
        });
        systemAnimationTimers = [];
    }

    function resetSystemFigure() {
        if (!systemSection) {
            return;
        }

        if (systemTimeline) {
            systemTimeline.pause();
            systemTimeline = null;
        }

        clearSystemAnimationTimers();

        systemCards.forEach(function (card) {
            card.classList.remove("is-animating", "is-static");
        });

        /* outlineElements is empty for the 3D scene — no-op */
        outlineElements.forEach(function (node) {
            resetStroke(node, false);
        });

        /* reset SVG flow paths */
        flowElements.forEach(function (node) {
            resetStroke(node, false);
            node.style.opacity = "0";
        });

        setNodesActive(gpuNodes, false);
        setNodesActive(cpuNodes, false);
        setNodesActive(dramNodes, false);
        setNodesActive(ssdNodes, false);
        setSystemFlowLabels(false);
        setSystemCaption(-1);

        if (cpuPulse) {
            cpuPulse.style.opacity = "0";
            cpuPulse.style.transform = "translate(-50%, -50%) scale(0.9)";
        }

        /* util bars: reset CSS custom property via inline style width */
        softUtilBars.forEach(function (bar) {
            bar.style.opacity = "0.18";
            bar.style.transform = "scaleX(0.45)";
            bar.style.transformOrigin = "left center";
        });
    }

    function showMemoryChartStatic() {
        if (!memoryChart) {
            return;
        }

        Array.from(memoryChart.querySelectorAll(".memory-trend")).forEach(function (node) {
            resetStroke(node, true);
            node.style.opacity = "0.88";
        });

        Array.from(memoryChart.querySelectorAll(".memory-point")).forEach(function (node) {
            node.style.opacity = "1";
            node.style.transform = "scale(1)";
        });

        Array.from(memoryChart.querySelectorAll(".memory-label, .memory-callout")).forEach(function (node) {
            node.style.opacity = "1";
            node.style.transform = "translateY(0px)";
        });
    }

    function playMemoryChart(force) {
        if (!memoryChart || (memoryChartPlayed && !force)) {
            return;
        }

        memoryChartPlayed = true;

        if (reducedMotion || !animeLib) {
            showMemoryChartStatic();
            return;
        }

        var trends = Array.from(memoryChart.querySelectorAll(".memory-trend"));
        var points = Array.from(memoryChart.querySelectorAll(".memory-point"));
        var labels = Array.from(memoryChart.querySelectorAll(".memory-label, .memory-callout"));

        trends.forEach(function (node) {
            resetStroke(node, false);
            node.style.opacity = "0.88";
        });

        points.forEach(function (node) {
            node.style.opacity = "0";
            node.style.transform = "scale(0.4)";
        });

        labels.forEach(function (node) {
            node.style.opacity = "0";
            node.style.transform = "translateY(14px)";
        });

        animeLib.timeline({ easing: "easeOutQuart" })
            .add({
                targets: trends,
                strokeDashoffset: function (node) {
                    return [animeLib.setDashoffset(node), 0];
                },
                duration: 1300,
                delay: animeLib.stagger(180)
            })
            .add({
                targets: points,
                opacity: [0, 1],
                scale: [0.4, 1],
                duration: 520,
                delay: animeLib.stagger(26, { from: "center" }),
                easing: "easeOutBack"
            }, "-=760")
            .add({
                targets: labels,
                opacity: [0, 1],
                translateY: [14, 0],
                duration: 460,
                delay: animeLib.stagger(14)
            }, "-=560");
    }

    function showSystemStageStatic(stage) {
        resetSystemFigure();
        systemCards.forEach(function (card) {
            card.classList.add("is-static");
        });
    }

    function playSystemStage(stage) {
        if (reducedMotion || !animeLib) {
            showSystemStageStatic(stage);
            return;
        }

        resetSystemFigure();

        var tl = animeLib.timeline({ easing: "easeOutQuart" });
        systemTimeline = tl;

        tl.add({
            targets: gpuNodes,
            translateY: [8, 0],
            opacity: [0, 1],
            duration: 420,
            delay: animeLib.stagger(100),
            begin: function () {
                setNodesActive(gpuNodes, true);
                setSystemStep(stage === 0 ? 0 : 1);
                setSystemCaption(stage === 0 ? 0 : 1);
            }
        })
        .add({
            targets: softUtilBars,
            scaleX: [0.2, 1],
            opacity: [0.2, 1],
            duration: 500,
            delay: animeLib.stagger(80),
            easing: "easeOutBack"
        }, "-=280");

        if (stage === 0) {
            return;
        }

        tl.add({
            targets: cpuNodes.concat(dramNodes, ssdNodes),
            translateY: [8, 0],
            opacity: [0, 1],
            duration: 440,
            delay: animeLib.stagger(120),
            begin: function () {
                setNodesActive(cpuNodes, true);
                setNodesActive(dramNodes, true);
                setNodesActive(ssdNodes, true);
            }
        }, "+=120")
        .add({
            targets: cpuPulse,
            opacity: [0, 0.7, 0.16],
            scale: [0.8, 1.1, 1],
            duration: 1000,
            easing: "easeOutSine"
        }, "-=280")
        .add({
            targets: flowElements,
            opacity: [0.3, 1],
            strokeDashoffset: function (node) {
                return [animeLib.setDashoffset(node), 0];
            },
            duration: 600,
            delay: animeLib.stagger(160)
        }, "-=700")
        .add({
            targets: flowLabels,
            opacity: [0, 1],
            translateY: [10, 0],
            duration: 340,
            delay: animeLib.stagger(90),
            begin: function () {
                setSystemFlowLabels(true);
            }
        }, "-=360");
    }

    function playSystemStory(force) {
        /* Delegated to io-step-animation.js when active */
        if (window.__ioStepAnimationActive) {
            systemStoryPlayed = true;
            return;
        }
        if (!systemCards.length || (systemStoryPlayed && !force)) {
            return;
        }

        systemStoryPlayed = true;

        if (reducedMotion) {
            showSystemStageStatic(1);
            return;
        }

        resetSystemFigure();

        systemCards.forEach(function (card, index) {
            var timerId = window.setTimeout(function () {
                card.classList.add("is-animating");
            }, index * 220);
            systemAnimationTimers.push(timerId);
        });
    }

    function setSyncCodeState(activeIndexes, blockedIndex, currentIndex) {
        syncCodeLines.forEach(function (line, index) {
            line.classList.toggle("is-active", activeIndexes.indexOf(index) !== -1);
            line.classList.toggle("is-blocked", index === blockedIndex);
            line.classList.toggle("is-current", index === currentIndex);
        });
    }

    function setSyncBatchState(currentBatch, currentBlock) {
        syncBatchRows.forEach(function (row, rowIndex) {
            row.classList.toggle("is-current", rowIndex === currentBatch);
        });

        syncTimelineBlocks.forEach(function (block, blockIndex) {
            block.classList.toggle("is-current", blockIndex === currentBlock);
        });

        if (syncCodeWait) {
            syncCodeWait.classList.toggle("is-live", false);
        }
    }

    function setSyncStatus(message, isLive) {
        if (!syncCodeWait) {
            return;
        }

        syncCodeWait.textContent = message;
        syncCodeWait.classList.toggle("is-live", Boolean(isLive));
    }

    function resetSyncProgress() {
        syncTimelineProgress.forEach(function (fill) {
            fill.style.transform = "scaleX(0)";
        });
    }

    function resetSyncModel() {
        if (!syncModelSection) {
            return;
        }

        if (syncModelTimeline) {
            syncModelTimeline.pause();
            syncModelTimeline = null;
        }

        setSyncCodeState([0, 1, 3, 4], 1, 1);
        setSyncBatchState(0, 0);

        syncCodeLines.forEach(function (line) {
            line.style.opacity = "1";
            line.style.transform = "translateY(0px)";
        });

        if (syncBlockingChip) {
            syncBlockingChip.style.opacity = "1";
            syncBlockingChip.style.transform = "translateX(0px)";
        }

        if (syncCodeWait) {
            syncCodeWait.textContent = "blocked on read";
            syncCodeWait.style.opacity = "1";
            syncCodeWait.style.transform = "translateY(0px)";
            syncCodeWait.classList.remove("is-live");
        }

        syncBatchRows.forEach(function (row) {
            row.classList.add("is-active");
            row.style.opacity = "1";
            row.style.transform = "translateY(0px)";
        });

        syncTimelineBlocks.forEach(function (block) {
            block.style.opacity = "1";
            block.style.transform = "scaleX(1)";
            block.style.transformOrigin = "left center";
        });

        resetSyncProgress();

        if (syncTimelineNote) {
            syncTimelineNote.style.opacity = "1";
            syncTimelineNote.style.transform = "translateY(0px)";
        }
    }

    function showSyncModelStatic() {
        resetSyncModel();
        setSyncCodeState([0, 1, 3, 4], 1, 4);
        setSyncBatchState(2, 5);

        syncCodeLines.forEach(function (line) {
            line.style.opacity = "1";
            line.style.transform = "translateY(0px)";
        });

        if (syncBlockingChip) {
            syncBlockingChip.style.opacity = "1";
            syncBlockingChip.style.transform = "translateX(0px)";
        }

        if (syncCodeWait) {
            syncCodeWait.style.opacity = "1";
            syncCodeWait.style.transform = "translateY(0px)";
        }

        setSyncStatus("batch retired", true);

        syncBatchRows.forEach(function (row) {
            row.classList.add("is-active");
            row.style.opacity = "1";
            row.style.transform = "translateY(0px)";
        });

        syncTimelineBlocks.forEach(function (block) {
            block.style.opacity = "1";
            block.style.transform = "scaleX(1)";
        });

        syncTimelineProgress.forEach(function (fill) {
            fill.style.transform = "scaleX(1)";
        });

        if (syncTimelineNote) {
            syncTimelineNote.style.opacity = "1";
            syncTimelineNote.style.transform = "translateY(0px)";
        }
    }

    function playSyncModel(force) {
        if (!syncModelSection || (syncModelPlayed && !force)) {
            return;
        }

        syncModelPlayed = true;

        if (reducedMotion || !animeLib) {
            showSyncModelStatic();
            return;
        }

        var readDuration = 1400;
        var waitDuration = 420;
        var computeDuration = 1100;
        var commitDuration = 240;

        resetSyncModel();

        function activateRead(batchIndex) {
            setSyncCodeState([0, 1], 1, 1);
            setSyncBatchState(batchIndex, batchIndex * 2);
            setSyncStatus("blocked on read", false);
        }

        function activateWait(batchIndex) {
            setSyncCodeState([0, 1, 2], 1, 2);
            setSyncBatchState(batchIndex, batchIndex * 2);
            setSyncStatus("waiting for DMA", false);
        }

        function activateCompute(batchIndex) {
            setSyncCodeState([0, 1, 3], -1, 3);
            setSyncBatchState(batchIndex, (batchIndex * 2) + 1);
            setSyncStatus("kernel running", true);
        }

        function activateCommit(batchIndex) {
            setSyncCodeState([0, 1, 3, 4], -1, 4);
            setSyncBatchState(batchIndex, (batchIndex * 2) + 1);
            setSyncStatus("commit complete", true);
        }

        syncModelTimeline = animeLib.timeline({ easing: "easeOutQuart" })
            .add({
                targets: syncTimelineProgress[0],
                scaleX: [0, 1],
                duration: readDuration,
                easing: "linear",
                begin: function () {
                    activateRead(0);
                },
                complete: function () {
                    activateWait(0);
                }
            })
            .add({
                duration: waitDuration,
                begin: function () {
                    activateCompute(0);
                }
            })
            .add({
                targets: syncTimelineProgress[1],
                scaleX: [0, 1],
                duration: computeDuration,
                easing: "linear"
            })
            .add({
                duration: commitDuration,
                begin: function () {
                    activateCommit(0);
                }
            })
            .add({
                targets: syncTimelineProgress[2],
                scaleX: [0, 1],
                duration: readDuration,
                easing: "linear",
                begin: function () {
                    activateRead(1);
                },
                complete: function () {
                    activateWait(1);
                }
            }, "-=40")
            .add({
                duration: waitDuration,
                begin: function () {
                    activateCompute(1);
                }
            })
            .add({
                targets: syncTimelineProgress[3],
                scaleX: [0, 1],
                duration: computeDuration,
                easing: "linear"
            })
            .add({
                duration: commitDuration,
                begin: function () {
                    activateCommit(1);
                }
            })
            .add({
                targets: syncTimelineProgress[4],
                scaleX: [0, 1],
                duration: readDuration,
                easing: "linear",
                begin: function () {
                    activateRead(2);
                },
                complete: function () {
                    activateWait(2);
                }
            }, "-=40")
            .add({
                duration: waitDuration,
                begin: function () {
                    activateCompute(2);
                }
            })
            .add({
                targets: syncTimelineProgress[5],
                scaleX: [0, 1],
                duration: computeDuration,
                easing: "linear"
            })
            .add({
                duration: commitDuration,
                begin: function () {
                    activateCommit(2);
                }
            })
            .add({
                targets: syncTimelineNote,
                opacity: [0, 1],
                translateY: [14, 0],
                duration: 320
            }, "-=80");
    }

    function setAsyncCodeState(activeIndexes, waitingIndex, currentIndex) {
        asyncCodeLines.forEach(function (line, index) {
            line.classList.toggle("is-active", activeIndexes.indexOf(index) !== -1);
            line.classList.toggle("is-waiting", index === waitingIndex);
            line.classList.toggle("is-current", index === currentIndex);
        });
    }

    function setAsyncActivity(activeRows, activeBlocks) {
        asyncBatchRows.forEach(function (row, index) {
            row.classList.toggle("is-current", activeRows.indexOf(index) !== -1);
        });

        asyncTimelineBlocks.forEach(function (block, index) {
            block.classList.toggle("is-current", activeBlocks.indexOf(index) !== -1);
        });
    }

    function setAsyncBuffers(currentBuffer, fillingBuffer) {
        asyncBufferPills.forEach(function (pill, index) {
            pill.classList.toggle("is-current", index === currentBuffer);
            pill.classList.toggle("is-filling", index === fillingBuffer);
        });
    }

    function setAsyncStatus(message, isCompute) {
        if (!asyncCodeLive) {
            return;
        }

        asyncCodeLive.textContent = message;
        asyncCodeLive.classList.toggle("is-compute", Boolean(isCompute));
    }

    function resetAsyncProgress() {
        asyncTimelineProgress.forEach(function (fill) {
            fill.style.transform = "scaleX(0)";
        });
    }

    function resetAsyncModel() {
        if (!asyncModelSection) {
            return;
        }

        if (asyncModelTimeline) {
            asyncModelTimeline.pause();
            asyncModelTimeline = null;
        }

        setAsyncCodeState([0, 1, 2, 3, 4, 5], -1, 1);
        setAsyncActivity([0], [0]);
        setAsyncBuffers(-1, 0);

        asyncCodeLines.forEach(function (line) {
            line.style.opacity = "1";
            line.style.transform = "translateY(0px)";
        });

        [asyncPrefetchChip, asyncOverlapChip].forEach(function (chip) {
            if (!chip) {
                return;
            }

            chip.style.opacity = "1";
            chip.style.transform = "translateX(0px)";
        });

        if (asyncCodeLive) {
            asyncCodeLive.textContent = "ready to prefetch next batch";
            asyncCodeLive.style.opacity = "1";
            asyncCodeLive.style.transform = "translateY(0px)";
            asyncCodeLive.classList.remove("is-compute");
        }

        asyncBatchRows.forEach(function (row) {
            row.classList.add("is-active");
            row.style.opacity = "1";
            row.style.transform = "translateY(0px)";
        });

        asyncTimelineBlocks.forEach(function (block) {
            block.style.opacity = "1";
            block.style.transform = "scaleX(1)";
            block.style.transformOrigin = "left center";
        });

        resetAsyncProgress();

        if (asyncTimelineNote) {
            asyncTimelineNote.style.opacity = "1";
            asyncTimelineNote.style.transform = "translateY(0px)";
        }
    }

    function showAsyncModelStatic() {
        resetAsyncModel();
        setAsyncCodeState([0, 1, 2, 3, 4, 5, 6, 7], -1, 7);
        setAsyncActivity([2, 3], [5, 7]);
        setAsyncBuffers(1, 0);

        asyncCodeLines.forEach(function (line) {
            line.style.opacity = "1";
            line.style.transform = "translateY(0px)";
        });

        [asyncPrefetchChip, asyncOverlapChip].forEach(function (chip) {
            if (!chip) {
                return;
            }

            chip.style.opacity = "1";
            chip.style.transform = "translateX(0px)";
        });

        if (asyncCodeLive) {
            asyncCodeLive.style.opacity = "1";
            asyncCodeLive.style.transform = "translateY(0px)";
        }

        setAsyncStatus("compute overlaps with the next DMA", true);

        asyncBatchRows.forEach(function (row) {
            row.classList.add("is-active");
            row.style.opacity = "1";
            row.style.transform = "translateY(0px)";
        });

        asyncTimelineBlocks.forEach(function (block) {
            block.style.opacity = "1";
            block.style.transform = "scaleX(1)";
        });

        asyncTimelineProgress.forEach(function (fill) {
            fill.style.transform = "scaleX(1)";
        });

        if (asyncTimelineNote) {
            asyncTimelineNote.style.opacity = "1";
            asyncTimelineNote.style.transform = "translateY(0px)";
        }
    }

    function playAsyncModel(force) {
        if (!asyncModelSection || (asyncModelPlayed && !force)) {
            return;
        }

        asyncModelPlayed = true;

        if (reducedMotion || !animeLib) {
            showAsyncModelStatic();
            return;
        }

        var bootstrapIssueDuration = 700;
        var prefetchIssueDuration = 520;
        var readDuration = 1700;
        var computeDuration = 1550;
        var waitDuration = 420;
        var settleDuration = 180;

        resetAsyncModel();

        function activateBootstrap() {
            setAsyncCodeState([0, 1, 2], -1, 1);
            setAsyncActivity([0], [0]);
            setAsyncBuffers(-1, 0);
            setAsyncStatus("blocking read fills the first buffer", false);
        }

        function activatePrefetch(batchIndex, nextBatch, currentBuffer, nextBuffer) {
            setAsyncCodeState([0, 1, 2, 3, 4], -1, 4);
            setAsyncActivity([batchIndex, nextBatch], [nextBatch * 2]);
            setAsyncBuffers(currentBuffer, nextBuffer);
            setAsyncStatus("queueing async prefetch", false);
        }

        function activateOverlap(currentBatch, nextBatch, currentBuffer, nextBuffer) {
            setAsyncCodeState([0, 1, 2, 3, 4, 5], -1, 5);
            setAsyncActivity([currentBatch, nextBatch], [(currentBatch * 2) + 1, nextBatch * 2]);
            setAsyncBuffers(currentBuffer, nextBuffer);
            setAsyncStatus("kernel and DMA overlap", true);
        }

        function activateWait(batchIndex, currentBuffer) {
            setAsyncCodeState([0, 1, 2, 3, 4, 5, 6], 6, 6);
            setAsyncActivity([batchIndex], [(batchIndex * 2) + 1]);
            setAsyncBuffers(currentBuffer, -1);
            setAsyncStatus("swap only when next buffer is needed", false);
        }

        function activateCommit(batchIndex, currentBuffer) {
            setAsyncCodeState([0, 1, 2, 3, 4, 5, 7], -1, 7);
            setAsyncActivity([batchIndex], [(batchIndex * 2) + 1]);
            setAsyncBuffers(currentBuffer, -1);
            setAsyncStatus("retire batch and keep pipeline full", true);
        }

        asyncModelTimeline = animeLib.timeline({ easing: "easeOutQuart" })
            .add({
                duration: bootstrapIssueDuration,
                begin: function () {
                    activateBootstrap();
                }
            })
            .add({
                targets: asyncTimelineProgress[0],
                scaleX: [0, 1],
                duration: readDuration,
                easing: "linear"
            })
            .add({
                duration: prefetchIssueDuration,
                begin: function () {
                    activatePrefetch(0, 1, 0, 1);
                }
            })
            .add({
                duration: 1,
                begin: function () {
                    activateOverlap(0, 1, 0, 1);
                }
            })
            .add({
                targets: asyncTimelineProgress[1],
                scaleX: [0, 1],
                duration: computeDuration,
                easing: "linear"
            })
            .add({
                targets: asyncTimelineProgress[2],
                scaleX: [0, 1],
                duration: readDuration,
                easing: "linear"
            }, "-=1550")
            .add({
                duration: waitDuration,
                begin: function () {
                    activateWait(0, 0);
                }
            })
            .add({
                duration: settleDuration,
                begin: function () {
                    activateCommit(0, 0);
                }
            })
            .add({
                duration: prefetchIssueDuration,
                begin: function () {
                    activatePrefetch(1, 2, 1, 0);
                }
            }, "+=0")
            .add({
                duration: 1,
                begin: function () {
                    activateOverlap(1, 2, 1, 0);
                }
            })
            .add({
                targets: asyncTimelineProgress[3],
                scaleX: [0, 1],
                duration: computeDuration,
                easing: "linear"
            })
            .add({
                targets: asyncTimelineProgress[4],
                scaleX: [0, 1],
                duration: readDuration,
                easing: "linear"
            }, "-=1550")
            .add({
                duration: waitDuration,
                begin: function () {
                    activateWait(1, 1);
                }
            })
            .add({
                duration: settleDuration,
                begin: function () {
                    activateCommit(1, 1);
                }
            })
            .add({
                duration: prefetchIssueDuration,
                begin: function () {
                    activatePrefetch(2, 3, 0, 1);
                }
            }, "+=0")
            .add({
                duration: 1,
                begin: function () {
                    activateOverlap(2, 3, 0, 1);
                }
            })
            .add({
                targets: asyncTimelineProgress[5],
                scaleX: [0, 1],
                duration: computeDuration,
                easing: "linear"
            })
            .add({
                targets: asyncTimelineProgress[6],
                scaleX: [0, 1],
                duration: readDuration,
                easing: "linear"
            }, "-=1550")
            .add({
                duration: waitDuration,
                begin: function () {
                    activateWait(2, 0);
                }
            })
            .add({
                duration: settleDuration,
                begin: function () {
                    activateCommit(2, 0);
                }
            })
            .add({
                duration: 1,
                begin: function () {
                    activateOverlap(3, 3, 1, -1);
                    setAsyncActivity([3], [7]);
                    setAsyncBuffers(1, -1);
                    setAsyncStatus("final batch computes with no stall", true);
                }
            }, "+=0")
            .add({
                targets: asyncTimelineProgress[7],
                scaleX: [0, 1],
                duration: computeDuration,
                easing: "linear"
            })
            .add({
                duration: settleDuration,
                begin: function () {
                    activateCommit(3, 1);
                }
            })
            .add({
                targets: asyncTimelineNote,
                opacity: [0, 1],
                translateY: [14, 0],
                duration: 320
            }, "-=60");
    }

    systemStepButtons.forEach(function (button) {
        button.addEventListener("click", function () {
            var stage = Number(button.getAttribute("data-stage"));
            playSystemStage(stage);
        });
    });

    if (systemReplayButton) {
        systemReplayButton.addEventListener("click", function () {
            playSystemStory(true);
        });
    }

    if (syncModelReplayButton) {
        syncModelReplayButton.addEventListener("click", function () {
            playSyncModel(true);
        });
    }

    if (asyncModelReplayButton) {
        asyncModelReplayButton.addEventListener("click", function () {
            playAsyncModel(true);
        });
    }

    if (memoryWallSection && !reducedMotion) {
        window.addEventListener("scroll", queueMemoryWallSnap, { passive: true });
        window.addEventListener("resize", queueMemoryWallSnap);
    }

    if (systemSection && !reducedMotion) {
        window.addEventListener("scroll", queueSystemStorySnap, { passive: true });
        window.addEventListener("resize", queueSystemStorySnap);
    }

    if (syncModelSection && !reducedMotion) {
        window.addEventListener("scroll", queueSyncModelSnap, { passive: true });
        window.addEventListener("resize", queueSyncModelSnap);
    }

    if (asyncModelSection && !reducedMotion) {
        window.addEventListener("scroll", queueAsyncModelSnap, { passive: true });
        window.addEventListener("resize", queueAsyncModelSnap);
    }

    if (memoryChart) {
        renderMemoryWallChart();
        Array.from(memoryChart.querySelectorAll(".memory-trend")).forEach(function (node) {
            prepareStroke(node);
        });
        Array.from(memoryChart.querySelectorAll(".memory-label, .memory-callout")).forEach(function (node) {
            node.style.transform = "translateY(14px)";
        });
    }

    if (systemSection) {
        resetSystemFigure();
    }

    if (syncModelSection) {
        resetSyncModel();
    }

    if (asyncModelSection) {
        resetAsyncModel();
    }

    if (!("IntersectionObserver" in window)) {
        playMemoryChart();
        playSystemStory();
        queueMemoryWallSnap();
        return;
    }

    var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (!entry.isIntersecting) {
                return;
            }

            if (entry.target === memoryChart) {
                playMemoryChart();
            }

            if (entry.target === systemSection) {
                playSystemStory();
            }
        });
    }, {
        threshold: 0.42
    });

    if (memoryChart) {
        observer.observe(memoryChart);
    }

    if (systemSection) {
        observer.observe(systemSection);
    }

})();
