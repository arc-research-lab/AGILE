(function () {
    const scrollytelling = document.getElementById("scrollytelling");
    const panels = Array.from(document.querySelectorAll("[data-panel]"));
    const dots = Array.from(document.querySelectorAll(".stage-dot"));
    const progressBar = document.getElementById("progressBar");
    const connectionToggle = document.getElementById("connectionToggle");
    const connectionFlyout = document.getElementById("connectionFlyout");
    const connectionBackdrop = document.getElementById("connectionBackdrop");
    const connectionClose = document.getElementById("connectionClose");
    const hostInput = document.getElementById("server-host");
    const portInput = document.getElementById("server-port");
    const targetUrl = document.getElementById("target-url");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const panelCount = panels.length;
    const snapDelayMs = 140;
    const snapThresholdPx = 6;
    const storageKeys = {
        host: "websocket-demo-host",
        port: "websocket-demo-port"
    };

    if (!scrollytelling || !progressBar || !connectionToggle || !connectionFlyout || !connectionBackdrop || !connectionClose || !hostInput || !portInput || !targetUrl) {
        return;
    }

    scrollytelling.style.setProperty("--panel-count", String(panelCount));

    let ticking = false;
    let snapTimer = null;
    let activeIndex = 0;
    let isAutoSnapping = false;

    function setConnectionPanelOpen(isOpen) {
        connectionFlyout.classList.toggle("is-open", isOpen);
        connectionBackdrop.classList.toggle("is-open", isOpen);
        connectionFlyout.setAttribute("aria-hidden", isOpen ? "false" : "true");
        connectionToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");

        if (isOpen) {
            hostInput.focus();
        }
    }

    function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }

    function getWebSocketUrl() {
        const host = hostInput.value.trim() || "127.0.0.1";
        const port = portInput.value.trim() || "9002";
        return "ws://" + host + ":" + port;
    }

    function refreshTargetUrl() {
        localStorage.setItem(storageKeys.host, hostInput.value.trim());
        localStorage.setItem(storageKeys.port, portInput.value.trim());
        targetUrl.textContent = getWebSocketUrl();
    }

    function loadSavedTarget() {
        const savedHost = localStorage.getItem(storageKeys.host);
        const savedPort = localStorage.getItem(storageKeys.port);

        if (savedHost) {
            hostInput.value = savedHost;
        }

        if (savedPort) {
            portInput.value = savedPort;
        }
    }

    function scrollToPanel(index) {
        const travel = Math.max(scrollytelling.offsetHeight - window.innerHeight, 1);
        const targetProgress = panelCount > 1 ? index / (panelCount - 1) : 0;
        const targetTop = window.scrollY + scrollytelling.getBoundingClientRect().top + targetProgress * travel;
        const clampedIndex = clamp(index, 0, panelCount - 1);

        activeIndex = clampedIndex;
        isAutoSnapping = true;

        window.scrollTo({
            top: targetTop,
            behavior: reducedMotion ? "auto" : "smooth"
        });

        window.clearTimeout(snapTimer);
        snapTimer = window.setTimeout(function () {
            isAutoSnapping = false;
            updatePanels();
        }, reducedMotion ? 0 : 260);
    }

    function getTargetTopForPanel(index) {
        const travel = Math.max(scrollytelling.offsetHeight - window.innerHeight, 1);
        const targetProgress = panelCount > 1 ? index / (panelCount - 1) : 0;
        return window.scrollY + scrollytelling.getBoundingClientRect().top + targetProgress * travel;
    }

    function isStageActive() {
        const rect = scrollytelling.getBoundingClientRect();
        return rect.top <= 0 && rect.bottom >= window.innerHeight;
    }

    function snapToNearestPanel() {
        if (isAutoSnapping || !isStageActive()) {
            return;
        }

        const targetTop = getTargetTopForPanel(activeIndex);
        if (Math.abs(window.scrollY - targetTop) <= snapThresholdPx) {
            return;
        }

        scrollToPanel(activeIndex);
    }

    function updatePanels() {
        const rect = scrollytelling.getBoundingClientRect();
        const travel = Math.max(scrollytelling.offsetHeight - window.innerHeight, 1);
        const progress = clamp((-rect.top) / travel, 0, 1);
        const scaled = progress * Math.max(panelCount - 1, 1);
        activeIndex = Math.round(scaled);

        progressBar.style.transform = "scaleX(" + progress.toFixed(4) + ")";

        panels.forEach(function (panel, index) {
            const delta = scaled - index;
            const distance = Math.abs(delta);
            const opacity = reducedMotion ? (index === activeIndex ? 1 : 0) : clamp(1 - distance * 0.9, 0, 1);
            const shift = reducedMotion ? 0 : delta * -70;
            const scale = reducedMotion ? 1 : 1 - Math.min(distance, 1) * 0.09;
            const blur = reducedMotion ? 0 : Math.min(distance, 1) * 14;
            const tilt = reducedMotion ? 0 : delta * -2.4;
            const artShift = reducedMotion ? 0 : delta * -28;

            panel.style.setProperty("--panel-opacity", opacity.toFixed(4));
            panel.style.setProperty("--panel-shift", shift.toFixed(2) + "px");
            panel.style.setProperty("--panel-scale", scale.toFixed(4));
            panel.style.setProperty("--panel-blur", blur.toFixed(2) + "px");
            panel.style.setProperty("--panel-tilt", tilt.toFixed(2) + "deg");
            panel.style.setProperty("--art-shift", artShift.toFixed(2) + "px");
            panel.style.zIndex = String(panelCount - Math.round(distance * 10));
            panel.classList.toggle("is-active", index === activeIndex);
        });

        dots.forEach(function (dot, index) {
            dot.classList.toggle("is-active", index === activeIndex);
            dot.setAttribute("aria-current", index === activeIndex ? "true" : "false");
        });

        ticking = false;
    }

    function requestUpdate() {
        if (!ticking) {
            window.requestAnimationFrame(updatePanels);
            ticking = true;
        }

        if (!isAutoSnapping && isStageActive()) {
            window.clearTimeout(snapTimer);
            snapTimer = window.setTimeout(snapToNearestPanel, snapDelayMs);
        } else if (!isStageActive()) {
            window.clearTimeout(snapTimer);
        }
    }

    function runPaperTitleAnimation() {
        var paperTitle = document.getElementById("paperTitle");
        var paperTitleLogo = document.getElementById("paperTitleLogo");
        var blinkTimers = [];
        if (!paperTitle) {
            return;
        }

        var slotOrder = ["A", "G", "I", "L", "E"];
        var letterColors = {
            A: "#7bf2c9",
            G: "#65c9ff",
            I: "#ffd280",
            L: "#ff9b8e",
            E: "#b8a6ff"
        };

        function revealPaperLogo() {
            if (!paperTitleLogo) {
                return;
            }

            window.setTimeout(function () {
                paperTitleLogo.classList.add("is-visible");
            }, reducedMotion ? 0 : 180);
        }

        function stopIdleBlinking() {
            blinkTimers.forEach(function (timerId) {
                window.clearTimeout(timerId);
            });
            blinkTimers = [];

            Array.from(paperTitle.querySelectorAll(".paper-word-initial.is-blinking, .paper-acronym-letter.is-blinking")).forEach(function (node) {
                node.classList.remove("is-blinking");
            });
        }

        function scheduleIdleBlinking() {
            stopIdleBlinking();

            var blinkTargets = slotOrder.reduce(function (targets, key) {
                var source = sources[key];
                var slot = paperTitle.querySelector('.paper-acronym-letter[data-slot="' + key + '"]');

                if (source) {
                    targets.push(source);
                }

                if (slot) {
                    targets.push(slot);
                }

                return targets;
            }, []);

            if (!blinkTargets.length) {
                return;
            }

            function queueNextBlink() {
                var nextDelay = 900 + Math.random() * 1800;
                var timerId = window.setTimeout(function () {
                    var target = blinkTargets[Math.floor(Math.random() * blinkTargets.length)];

                    target.classList.remove("is-blinking");
                    void target.offsetWidth;
                    target.classList.add("is-blinking");

                    var cleanupId = window.setTimeout(function () {
                        target.classList.remove("is-blinking");
                    }, 620);
                    blinkTimers.push(cleanupId);
                    queueNextBlink();
                }, nextDelay);

                blinkTimers.push(timerId);
            }

            queueNextBlink();
        }

        if (reducedMotion) {
            slotOrder.forEach(function (key) {
                var reducedSource = paperTitle.querySelector('.paper-word[data-key="' + key + '"] .paper-word-initial');
                var reducedSlot = paperTitle.querySelector('.paper-acronym-letter[data-slot="' + key + '"]');
                var reducedColor = letterColors[key] || "#65c9ff";

                if (reducedSource) {
                    reducedSource.style.setProperty("--paper-accent", reducedColor);
                    reducedSource.classList.add("is-highlighted");
                }

                if (reducedSlot) {
                    reducedSlot.style.setProperty("--paper-accent", reducedColor);
                    reducedSlot.classList.add("is-visible");
                }
            });

            paperTitle.classList.add("is-animated");
            revealPaperLogo();
            return;
        }

        var highlightDelay = 180;
        var holdBeforeFlight = 720;
        var launchOffsets = [0, 170, 90, 260, 140];
        var durations = [1380, 1820, 1540, 2060, 1680];
        var sources = {};

        Array.from(paperTitle.querySelectorAll(".paper-word[data-key]")).forEach(function (word) {
            var key = word.getAttribute("data-key");
            var initial = word.querySelector(".paper-word-initial");
            if (key && initial) {
                sources[key] = initial;
            }
        });

        var slots = slotOrder.map(function (key) {
            return paperTitle.querySelector('.paper-acronym-letter[data-slot="' + key + '"]');
        });

        if (slotOrder.some(function (key) { return !sources[key]; }) || slots.some(function (slot) { return !slot; })) {
            paperTitle.classList.add("is-animated");
            revealPaperLogo();
            return;
        }

        paperTitle.classList.add("is-animating");

        var completed = 0;

        function finishAnimation() {
            paperTitle.classList.remove("is-animating");
            paperTitle.classList.add("is-animated");
            revealPaperLogo();
            scheduleIdleBlinking();
        }

        slotOrder.forEach(function (key, index) {
            var source = sources[key];
            var slot = slots[index];
            var accent = letterColors[key] || "#65c9ff";
            var launchTime = highlightDelay * index + holdBeforeFlight + launchOffsets[index];

            source.style.setProperty("--paper-accent", accent);
            slot.style.setProperty("--paper-accent", accent);

            window.setTimeout(function () {
                source.classList.add("is-highlighted");
            }, highlightDelay * index);

            window.setTimeout(function () {
                var sourceRect = source.getBoundingClientRect();
                var slotRect = slot.getBoundingClientRect();
                var computed = window.getComputedStyle(source);
                var flyer = document.createElement("span");

                source.classList.add("is-source-active");

                flyer.className = "paper-fly-letter";
                flyer.textContent = key;
                flyer.style.setProperty("--paper-accent", accent);
                flyer.style.left = sourceRect.left + "px";
                flyer.style.top = sourceRect.top + "px";
                flyer.style.fontFamily = computed.fontFamily;
                flyer.style.fontSize = computed.fontSize;
                flyer.style.fontWeight = computed.fontWeight;
                flyer.style.lineHeight = computed.lineHeight;
                flyer.style.letterSpacing = computed.letterSpacing;
                document.body.appendChild(flyer);

                requestAnimationFrame(function () {
                    flyer.style.transform = "translate(" + (slotRect.left - sourceRect.left) + "px, " + (slotRect.top - sourceRect.top) + "px) scale(1.08)";
                    flyer.style.opacity = "1";
                    flyer.style.transition = "transform " + durations[index] + "ms cubic-bezier(0.2, 0.9, 0.2, 1), opacity 220ms ease";
                });

                window.setTimeout(function () {
                    slot.classList.add("is-visible");
                    source.classList.remove("is-source-active");
                    flyer.remove();
                    completed += 1;
                    if (completed === slotOrder.length) {
                        finishAnimation();
                    }
                }, durations[index] + 60);
            }, launchTime);
        });
    }

    function initRelatedWorksDialog() {
        var nodes = Array.from(document.querySelectorAll(".related-node"));
        var map = document.getElementById("relatedWorksMap");
        var dialog = document.getElementById("relatedWorksDialog");
        var title = document.getElementById("relatedDialogTitle");
        var subtitle = document.getElementById("relatedDialogSubtitle");
        var features = document.getElementById("relatedDialogFeatures");
        var shortcomings = document.getElementById("relatedDialogShortcomings");
        var shortcomingsSection = document.getElementById("relatedDialogShortcomingsSection");
        var defaultNode = document.querySelector(".related-node.is-default") || nodes[0];
        var activeNode = null;

        if (!map || !dialog || !title || !subtitle || !features || !shortcomings || !shortcomingsSection || !nodes.length) {
            return;
        }

        function fillList(target, value) {
            target.textContent = "";
            var items = value.split("|").filter(Boolean);

            items.forEach(function (item) {
                var listItem = document.createElement("li");
                listItem.textContent = item;
                target.appendChild(listItem);
            });

            return items.length > 0;
        }

        function setDialogVisible(isVisible) {
            dialog.classList.toggle("is-visible", isVisible);
        }

        function clearActiveNode() {
            activeNode = null;
            setDialogVisible(false);
            nodes.forEach(function (item) {
                item.classList.remove("is-active");
            });
        }

        function positionDialog(node) {
            if (window.innerWidth <= 640) {
                dialog.style.removeProperty("left");
                dialog.style.removeProperty("top");
                dialog.style.removeProperty("max-width");
                return;
            }

            var mapRect = map.getBoundingClientRect();
            var nodeRect = node.getBoundingClientRect();
            var dialogRect = dialog.getBoundingClientRect();
            var horizontalGap = 18;
            var verticalPadding = 18;
            var minLeft = 18;
            var maxLeft = Math.max(minLeft, mapRect.width - dialogRect.width - 18);
            var preferredLeft = nodeRect.right - mapRect.left + horizontalGap;

            if (preferredLeft > maxLeft) {
                preferredLeft = nodeRect.left - mapRect.left - dialogRect.width - horizontalGap;
            }

            var top = nodeRect.top - mapRect.top + (nodeRect.height - dialogRect.height) / 2;
            var clampedTop = Math.min(Math.max(top, verticalPadding), Math.max(verticalPadding, mapRect.height - dialogRect.height - verticalPadding));
            var clampedLeft = Math.min(Math.max(preferredLeft, minLeft), maxLeft);

            dialog.style.left = clampedLeft + "px";
            dialog.style.top = clampedTop + "px";
            dialog.style.maxWidth = Math.min(320, Math.max(240, mapRect.width * 0.34)) + "px";
        }

        function activate(node) {
            activeNode = node;
            setDialogVisible(true);

            nodes.forEach(function (item) {
                item.classList.toggle("is-active", item === node);
            });

            title.textContent = node.dataset.title || "";
            subtitle.textContent = node.dataset.subtitle || "";
            fillList(features, node.dataset.features || "");
            shortcomingsSection.hidden = !fillList(shortcomings, node.dataset.shortcomings || "");
            positionDialog(node);
        }

        nodes.forEach(function (node) {
            node.addEventListener("mouseenter", function () {
                activate(node);
            });

            node.addEventListener("mouseleave", function (event) {
                var nextTarget = event.relatedTarget;

                if (nextTarget && nextTarget.closest && nextTarget.closest(".related-node")) {
                    return;
                }

                if (activeNode === node) {
                    clearActiveNode();
                }
            });

            node.addEventListener("focus", function () {
                activate(node);
            });

            node.addEventListener("blur", function (event) {
                var nextTarget = event.relatedTarget;

                if (nextTarget && nextTarget.closest && nextTarget.closest(".related-node")) {
                    return;
                }

                if (activeNode === node) {
                    clearActiveNode();
                }
            });

            node.addEventListener("click", function () {
                activate(node);
            });
        });

        window.addEventListener("resize", function () {
            if (activeNode) {
                positionDialog(activeNode);
            }
        });

        if (defaultNode) {
            title.textContent = defaultNode.dataset.title || "";
            subtitle.textContent = defaultNode.dataset.subtitle || "";
            fillList(features, defaultNode.dataset.features || "");
            shortcomingsSection.hidden = !fillList(shortcomings, defaultNode.dataset.shortcomings || "");
            activeNode = defaultNode;
            positionDialog(defaultNode);
        }
        setDialogVisible(false);
    }

    window.addEventListener("scroll", requestUpdate, { passive: true });
    window.addEventListener("resize", requestUpdate);

    dots.forEach(function (dot, index) {
        dot.addEventListener("click", function () {
            scrollToPanel(index);
        });
    });

    connectionToggle.addEventListener("click", function () {
        setConnectionPanelOpen(!connectionFlyout.classList.contains("is-open"));
    });

    connectionClose.addEventListener("click", function () {
        setConnectionPanelOpen(false);
        connectionToggle.focus();
    });

    connectionBackdrop.addEventListener("click", function () {
        setConnectionPanelOpen(false);
    });

    window.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && connectionFlyout.classList.contains("is-open")) {
            setConnectionPanelOpen(false);
            connectionToggle.focus();
        }
    });

    loadSavedTarget();
    refreshTargetUrl();
    hostInput.addEventListener("input", refreshTargetUrl);
    portInput.addEventListener("input", refreshTargetUrl);
    updatePanels();
    initRelatedWorksDialog();
    window.requestAnimationFrame(function () {
        window.requestAnimationFrame(runPaperTitleAnimation);
    });
})();