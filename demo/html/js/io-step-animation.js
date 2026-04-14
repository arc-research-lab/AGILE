/**
 * Step-by-step IO animation controller for the GPU-Centric I/O section.
 *
 * For each of the three "case" cards it:
 *   1. Removes the old flow-path elements and static muted/primary classes.
 *   2. Cycles through steps, highlighting only the active components.
 *   3. Draws flow-path lines with a stroke-dashoffset draw-in, then shows a
 *      flowing "marching ants" animation to convey data/message transfer.
 *   4. Loops the animation after a hold period.
 *
 * Depends on the CSS rules in index.css (IO Step Animation section).
 */
(function () {
    "use strict";

    /* ── timing constants (ms) ──────────────────────────────────── */
    var STEP_DURATION      = 2800;
    var DRAW_DURATION      = 850;
    var FLOW_STAGGER       = 350;   /* delay between multiple flows in one step */
    var PAUSE_BETWEEN      = 500;
    var HOLD_AFTER         = 3200;

    /* ── marker colours per flow class ──────────────────────────── */
    var MARKER_COLORS = {
        "is-control":     "#97a8ff",
        "is-data":        "#7ff4d7",
        "is-bounce":      "#ffad97",
        "is-gpu-control": "#ffd57c"
    };

    /* ── node identifier → CSS class on the <g> element ─────────── */
    var NODE_CLS = {
        gpu:  "io-node-gpu",
        cpu:  "io-node-cpu",
        dram: "io-node-dram",
        ssd:  "io-node-ssd"
    };

    /* ── per-card step definitions ──────────────────────────────── */
    /*
     * Edge midpoints (from SVG rects):
     *   DRAM  center(80,51)   right(126,51)  bottom(80,74)
     *   CPU   center(209,51)  right(245,51)  bottom(209,87)  left(173,51)
     *   NVMe  center(348,146) left(304,146)  top(348,120)    bottom(348,172)
     *   GPU   center(118,223) top(118,198)   right(166,223)
     */
    var CONFIGS = [
        /* Card 0 – Without GPUDirect Storage -------------------- */
        {
            steps: [
                {
                    /* Step 1: GPU-CPU sync  (GPU top → CPU bottom) */
                    activeNodes: ["gpu", "cpu"],
                    flows: [
                        { d: "M 118 198 V 138 H 209 V 87", cls: "is-control" }
                    ]
                },
                {
                    /* Step 2: CPU reads data from NVMe to DRAM
                       CPU right → NVMe top,  NVMe left → DRAM bottom */
                    activeNodes: ["cpu", "ssd", "dram"],
                    flows: [
                        { d: "M 245 51 H 348 V 120",  cls: "is-bounce" },
                        { d: "M 304 146 H 80 V 74",   cls: "is-bounce" }
                    ]
                },
                {
                    /* Step 3: CPU copies data from DRAM to GPU */
                    activeNodes: ["cpu", "dram", "gpu"],
                    flows: [
                        { d: "M 80 74 V 140 H 118 V 198", cls: "is-bounce" }
                    ]
                }
            ]
        },
        /* Card 1 – With GPUDirect Storage ----------------------- */
        {
            steps: [
                {
                    /* Step 1: GPU-CPU sync  (GPU top → CPU bottom) */
                    activeNodes: ["gpu", "cpu"],
                    flows: [
                        { d: "M 118 198 V 138 H 209 V 87", cls: "is-control" }
                    ]
                },
                {
                    /* Step 2: CPU issues NVMe read command */
                    activeNodes: ["cpu", "ssd"],
                    flows: [
                        { d: "M 245 51 H 348 V 120", cls: "is-control" }
                    ]
                },
                {
                    /* Step 3: NVMe DMA data directly to GPU */
                    activeNodes: ["ssd", "gpu"],
                    flows: [
                        { d: "M 304 146 H 236 V 223 H 166", cls: "is-data" }
                    ]
                }
            ]
        },
        /* Card 2 – GPU-Centric Storage -------------------------- */
        {
            steps: [
                {
                    /* Step 1: GPU issues NVMe read command */
                    activeNodes: ["gpu", "ssd"],
                    flows: [
                        { d: "M 166 210 H 235 V 146 H 304", cls: "is-gpu-control" }
                    ]
                },
                {
                    /* Step 2: NVMe DMA data directly to GPU */
                    activeNodes: ["ssd", "gpu"],
                    flows: [
                        { d: "M 348 172 V 236 H 166", cls: "is-data" }
                    ]
                },
                {
                    /* Step 3: No CPU or DRAM — show both control + data paths together */
                    activeNodes: ["gpu", "ssd"],
                    flows: [
                        { d: "M 166 210 H 235 V 146 H 304", cls: "is-gpu-control" },
                        { d: "M 348 172 V 236 H 166",       cls: "is-data" }
                    ]
                }
            ]
        }
    ];

    var svgNS = "http://www.w3.org/2000/svg";

    /* ── query the three cards ──────────────────────────────────── */
    var section = document.getElementById("systemStorySection");
    if (!section) return;

    var cards = Array.from(section.querySelectorAll(".io-case-card"));
    if (!cards.length) return;

    /* signal to narrative-sections.js that the new controller is active */
    window.__ioStepAnimationActive = true;

    /* ── per-card runtime state ─────────────────────────────────── */
    var states = [];

    cards.forEach(function (card, idx) {
        var cfg = CONFIGS[idx];
        if (!cfg) return;

        var svg = card.querySelector(".io-case-svg");
        if (!svg) return;

        /* remove old animated flow paths */
        Array.from(svg.querySelectorAll(".io-flow-path")).forEach(function (p) {
            p.remove();
        });

        /* strip old static visual classes – JS will control all state */
        Array.from(svg.querySelectorAll(".io-node")).forEach(function (n) {
            n.classList.remove("is-muted", "is-primary");
        });
        card.classList.remove("is-animating", "is-static");

        /* collect node refs */
        var nodes = {};
        Object.keys(NODE_CLS).forEach(function (key) {
            nodes[key] = svg.querySelector("." + NODE_CLS[key]);
        });

        /* show nodes at full opacity/position on load (no auto-play) */
        Object.keys(nodes).forEach(function (k) {
            var n = nodes[k];
            if (n) {
                n.style.opacity = "1";
                n.style.transform = "translateY(0) scale(1)";
            }
        });

        /* grab step chips */
        var chips = Array.from(card.querySelectorAll(".io-step-chip"));

        /* show chips at full opacity on load */
        chips.forEach(function (c) {
            c.style.opacity = "1";
            c.style.transform = "translateY(0)";
        });

        /* create SVG <defs> with markers unique to this card */
        var defs = svg.querySelector("defs");
        if (!defs) {
            defs = document.createElementNS(svgNS, "defs");
            svg.insertBefore(defs, svg.firstChild);
        }
        /* clear old markers */
        while (defs.firstChild) defs.removeChild(defs.firstChild);

        var markers = {};
        Object.keys(MARKER_COLORS).forEach(function (cls) {
            var id = "io-step-mk-" + idx + "-" + cls.replace("is-", "");
            var mk = document.createElementNS(svgNS, "marker");
            mk.setAttribute("id", id);
            mk.setAttribute("markerWidth",  "5");
            mk.setAttribute("markerHeight", "4");
            mk.setAttribute("refX", "4.2");
            mk.setAttribute("refY", "2.2");
            mk.setAttribute("orient", "auto");
            var arrow = document.createElementNS(svgNS, "path");
            arrow.setAttribute("d", "M 0.4 0.4 L 4 2 L 0.4 3.6");
            arrow.setAttribute("fill", "none");
            arrow.setAttribute("stroke", MARKER_COLORS[cls]);
            arrow.setAttribute("stroke-width", "1");
            arrow.setAttribute("stroke-linejoin", "round");
            mk.appendChild(arrow);
            defs.appendChild(mk);
            markers[cls] = id;
        });

        states.push({
            card:       card,
            svg:        svg,
            cfg:        cfg,
            nodes:      nodes,
            chips:      chips,
            markers:    markers,
            flowPaths:  [],
            timers:     [],
            playing:    false,
            played:     false
        });
    });

    /* ── helpers ─────────────────────────────────────────────────── */

    function clearTimers(st) {
        st.timers.forEach(clearTimeout);
        st.timers = [];
    }

    function removeFlows(st) {
        st.flowPaths.forEach(function (p) { p.remove(); });
        st.flowPaths = [];
    }

    function resetCard(st) {
        clearTimers(st);
        removeFlows(st);

        Object.keys(st.nodes).forEach(function (k) {
            var n = st.nodes[k];
            if (n) n.classList.remove("is-step-active", "is-step-blink", "is-step-dimmed");
        });
        st.chips.forEach(function (c) {
            c.classList.remove("is-step-active");
        });
        /* keep takeaway highlighted if already revealed */
        if (st.takeawayRevealed && st.chips[st.cfg.steps.length]) {
            st.chips[st.cfg.steps.length].classList.add("is-step-active");
        }
        st.card.classList.remove("is-step-playing");
        st.playing = false;
    }

    function createFlow(st, flowDef) {
        var p = document.createElementNS(svgNS, "path");
        p.setAttribute("d", flowDef.d);
        p.classList.add("io-step-flow", flowDef.cls);
        if (st.markers[flowDef.cls]) {
            p.setAttribute("marker-end", "url(#" + st.markers[flowDef.cls] + ")");
        }
        /* insert before first node so lines render behind components */
        var firstNode = st.svg.querySelector(".io-node");
        if (firstNode) {
            st.svg.insertBefore(p, firstNode);
        } else {
            st.svg.appendChild(p);
        }
        st.flowPaths.push(p);
        return p;
    }

    function drawIn(path, dur) {
        var len = path.getTotalLength();
        /* reset to hidden */
        path.style.transition      = "none";
        path.style.strokeDasharray = len + " " + len;
        path.style.strokeDashoffset = len;
        path.style.opacity         = "0.95";
        path.classList.add("is-drawing");
        /* force reflow before starting transition */
        void path.getBoundingClientRect();
        path.style.transition = "stroke-dashoffset " + dur +
                                "ms cubic-bezier(0.22, 1, 0.36, 1)";
        path.style.strokeDashoffset = "0";
    }

    function switchToFlowing(path) {
        path.classList.remove("is-drawing");
        path.style.transition       = "none";
        path.style.strokeDasharray  = "8 14";
        path.style.strokeDashoffset = "0";
        path.classList.add("is-flowing");
    }

    /* ── step activation ────────────────────────────────────────── */

    function activateStep(st, stepIdx) {
        var step = st.cfg.steps[stepIdx];
        if (!step) return;

        /* fade-out old flows */
        st.flowPaths.forEach(function (p) {
            p.classList.remove("is-drawing", "is-flowing");
            p.style.transition = "opacity 0.35s ease";
            p.style.opacity    = "0";
        });
        /* remove them after the fade completes */
        var oldPaths = st.flowPaths.slice();
        var removeTimer = setTimeout(function () {
            oldPaths.forEach(function (p) { p.remove(); });
        }, 380);
        st.timers.push(removeTimer);
        st.flowPaths = [];

        /* update nodes: dim all, then activate */
        Object.keys(st.nodes).forEach(function (k) {
            var n = st.nodes[k];
            if (n) {
                n.classList.remove("is-step-active", "is-step-blink", "is-step-dimmed");
            }
        });
        step.activeNodes.forEach(function (k) {
            var n = st.nodes[k];
            if (n) n.classList.add("is-step-active");
        });

        /* blink then dim specific nodes */
        if (step.blinkThenDim) {
            step.blinkThenDim.forEach(function (k) {
                var n = st.nodes[k];
                if (!n) return;
                n.classList.add("is-step-blink");
                var dimTid = setTimeout(function () {
                    n.classList.remove("is-step-blink");
                    n.classList.add("is-step-dimmed");
                }, 1000);
                st.timers.push(dimTid);
            });
        }

        /* update chips: dim all step chips, but preserve takeaway if revealed */
        st.chips.forEach(function (c, ci) {
            if (st.takeawayRevealed && ci === st.cfg.steps.length) return;
            c.classList.remove("is-step-active");
        });
        if (st.chips[stepIdx]) st.chips[stepIdx].classList.add("is-step-active");

        /* create + animate new flows with stagger */
        step.flows.forEach(function (fDef, fi) {
            var delay = fi * FLOW_STAGGER;
            var tid = setTimeout(function () {
                var p = createFlow(st, fDef);
                drawIn(p, DRAW_DURATION);
                /* switch to flowing after draw finishes */
                var flowTid = setTimeout(function () {
                    switchToFlowing(p);
                }, DRAW_DURATION + 30);
                st.timers.push(flowTid);
            }, delay);
            st.timers.push(tid);
        });
    }

    /* ── card playback loop ─────────────────────────────────────── */

    function playCard(st) {
        if (st.playing) return;
        st.playing = true;
        st.played  = true;
        st.card.classList.add("is-step-playing");

        /* ensure nodes are positioned (override the base translateY/scale) */
        Object.keys(st.nodes).forEach(function (k) {
            var n = st.nodes[k];
            if (n) {
                n.style.transform = "translateY(0) scale(1)";
            }
        });

        var total = st.cfg.steps.length;

        function run(i) {
            if (i >= total) {
                /* activate takeaway chip (the one after all steps) */
                if (st.chips[total]) {
                    st.chips[total].classList.add("is-step-active");
                    /* keep takeaway permanently highlighted */
                    st.takeawayRevealed = true;
                }
                /* hold final state, then loop */
                var hold = setTimeout(function () {
                    resetCard(st);
                    playCard(st);
                }, HOLD_AFTER);
                st.timers.push(hold);
                return;
            }
            activateStep(st, i);
            var next = setTimeout(function () { run(i + 1); },
                                  STEP_DURATION + PAUSE_BETWEEN);
            st.timers.push(next);
        }

        run(0);
    }

    /* ── public API for other scripts ───────────────────────────── */

    window.__ioStepPlay = function () {
        states.forEach(function (st) { if (!st.played) playCard(st); });
    };
    window.__ioStepReplay = function () {
        states.forEach(function (st) { resetCard(st); playCard(st); });
    };

    /* ── toggle buttons: RUN / STOP ─────────────────────────────── */

    function updateBtn(btn, playing) {
        btn.textContent = playing ? "STOP" : "RUN";
    }

    states.forEach(function (st) {
        var btn = st.card.querySelector(".io-replay-btn");
        if (!btn) return;
        st.btn = btn;
        btn.addEventListener("click", function (e) {
            e.stopPropagation();
            if (st.playing) {
                /* full stop: clear takeaway highlight too */
                st.takeawayRevealed = false;
                resetCard(st);
                updateBtn(btn, false);
            } else {
                st.takeawayRevealed = false;
                resetCard(st);
                playCard(st);
                updateBtn(btn, true);
            }
        });
    });

    /* Override resetCard to also update the button text,
       but only when the user explicitly stops (not during loop resets) */
    var _origReset = resetCard;
    resetCard = function (st) {
        _origReset(st);
    };
})();
