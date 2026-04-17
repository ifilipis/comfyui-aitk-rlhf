import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const LOG_CLASS_NAME = "AITKLog";
const STATUS_WIDGET_NAME = "AITK Log Status";
const FALLBACK_LOG_WIDGET_NAME = "AITK Live Logs";
const MAX_LOG_TEXT_CHARS = 16000;

function clampInt(value, minValue, maxValue, fallback) {
	const num = Number.parseInt(value, 10);
	if (Number.isNaN(num)) return fallback;
	return Math.max(minValue, Math.min(maxValue, num));
}

function getWidgetValue(node, widgetName, fallback) {
	if (!node?.widgets) return fallback;
	const widget = node.widgets.find((w) => w?.name === widgetName);
	if (!widget) return fallback;
	return widget.value ?? fallback;
}

function setReadonlyWidget(widget) {
	if (!widget?.inputEl) return;
	widget.inputEl.readOnly = true;
	widget.inputEl.style.opacity = 0.7;
}

function makeLogSink(node) {
	if (typeof node.addDOMWidget === "function") {
		const wrapper = document.createElement("div");
		wrapper.style.width = "100%";
		wrapper.style.maxHeight = "220px";
		wrapper.style.overflow = "auto";
		wrapper.style.background = "rgba(0, 0, 0, 0.08)";
		wrapper.style.border = "1px solid rgba(255, 255, 255, 0.16)";
		wrapper.style.borderRadius = "6px";
		wrapper.style.padding = "6px";

		const pre = document.createElement("pre");
		pre.style.margin = "0";
		pre.style.whiteSpace = "pre-wrap";
		pre.style.wordBreak = "break-word";
		pre.style.fontSize = "11px";
		pre.style.lineHeight = "1.3";
		pre.textContent = "Waiting for logs...";
		wrapper.appendChild(pre);

		node.addDOMWidget(FALLBACK_LOG_WIDGET_NAME, FALLBACK_LOG_WIDGET_NAME, wrapper, {
			serialize: false,
		});

		return {
			setText(text) {
				pre.textContent = text;
			},
		};
	}

	const fallback = node.addWidget(
		"text",
		FALLBACK_LOG_WIDGET_NAME,
		"Waiting for logs...",
		() => {},
		{ serialize: false }
	);
	setReadonlyWidget(fallback);
	return {
		setText(text) {
			fallback.value = text;
		},
	};
}

function setNodeStatus(node, text) {
	if (!node?.widgets) return;
	const statusWidget = node.widgets.find((w) => w?.name === STATUS_WIDGET_NAME);
	if (!statusWidget) return;
	statusWidget.value = text;
	app.graph.setDirtyCanvas(true, false);
}

function formatLogLines(entries) {
	if (!Array.isArray(entries) || entries.length === 0) {
		return "No log entries yet.";
	}
	const lines = entries.map((entry) => {
		const ts = entry?.timestamp || "-";
		const level = String(entry?.level || "info").toUpperCase();
		const event = entry?.event ? `[${entry.event}] ` : "";
		const message = entry?.message || "";
		return `${ts} ${level} ${event}${message}`;
	});
	const text = lines.join("\n");
	if (text.length <= MAX_LOG_TEXT_CHARS) {
		return text;
	}
	return text.slice(text.length - MAX_LOG_TEXT_CHARS);
}

async function fetchLogs(node, limit) {
	const params = new URLSearchParams();
	params.set("node_id", String(node.id));
	params.set("limit", String(limit));

	const response = await api.fetchApi(`/aitk_rlhf/logs?${params.toString()}`, {
		method: "GET",
	});

	let payload = {};
	try {
		payload = await response.json();
	} catch {
		payload = {};
	}

	if (!response.ok) {
		const message = payload?.error || `Request failed (${response.status})`;
		throw new Error(message);
	}

	return payload;
}

function computeStatus(payload) {
	const sessionId = payload?.session_id || "unknown-session";
	const summary = payload?.session_summary || null;
	const step = summary?.step_count;
	const cached = summary?.cached_candidates;
	const active = payload?.session_active !== false;

	if (step !== undefined && cached !== undefined) {
		return `${sessionId} | step ${step} | cached ${cached} | ${active ? "active" : "inactive"}`;
	}
	if (payload?.session_error) {
		return `${sessionId} | ${payload.session_error}`;
	}
	return `${sessionId} | ${active ? "active" : "inactive"}`;
}

function startPolling(node, sink) {
	const controller = {
		stopped: false,
		timer: null,
	};
	node.__aitkLogController = controller;

	const scheduleNext = () => {
		if (controller.stopped) return;
		const delay = clampInt(
			getWidgetValue(node, "poll_interval_ms", 1500),
			250,
			60000,
			1500
		);
		controller.timer = setTimeout(runPoll, delay);
	};

	const runPoll = async () => {
		if (controller.stopped) return;
		try {
			if (node?.id == null) {
				setNodeStatus(node, "Waiting for node id...");
				scheduleNext();
				return;
			}
			const limit = clampInt(
				getWidgetValue(node, "tail_entries", 60),
				1,
				1000,
				60
			);
			const payload = await fetchLogs(node, limit);
			setNodeStatus(node, computeStatus(payload));
			sink.setText(formatLogLines(payload?.entries));
		} catch (error) {
			const message = error?.message || String(error);
			setNodeStatus(node, `Log polling error: ${message}`);
			sink.setText(`Log polling failed:\n${message}`);
			console.error("[AITK Log] polling failed:", error);
		} finally {
			app.graph.setDirtyCanvas(true, false);
			scheduleNext();
		}
	};

	runPoll();
}

function installLogWidgets(node) {
	if (node.__aitkLogWidgetsInstalled) return;
	node.__aitkLogWidgetsInstalled = true;

	const statusWidget = node.addWidget(
		"text",
		STATUS_WIDGET_NAME,
		"Initializing log poller...",
		() => {},
		{ serialize: false }
	);
	setReadonlyWidget(statusWidget);

	const sink = makeLogSink(node);
	startPolling(node, sink);

	const originalOnRemoved = node.onRemoved;
	node.onRemoved = function () {
		const controller = this.__aitkLogController;
		if (controller) {
			controller.stopped = true;
			if (controller.timer) {
				clearTimeout(controller.timer);
			}
		}
		if (typeof originalOnRemoved === "function") {
			return originalOnRemoved.apply(this, arguments);
		}
		return undefined;
	};
}

app.registerExtension({
	name: "AITK.RLHF.LiveLog",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== LOG_CLASS_NAME) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const result = onNodeCreated?.apply(this, arguments);
			installLogWidgets(this);
			return result;
		};
	},
});

