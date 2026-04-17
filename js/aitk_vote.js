import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const VOTE_CLASS_NAME = "AITKVote";
const STATUS_WIDGET_NAME = "AITK Vote Status";

async function postVote(nodeId, vote) {
	const response = await api.fetchApi("/aitk_rlhf/vote", {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			node_id: String(nodeId),
			vote,
		}),
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

function setNodeStatus(node, text) {
	if (!node.widgets) return;
	const statusWidget = node.widgets.find((w) => w?.name === STATUS_WIDGET_NAME);
	if (!statusWidget) return;
	statusWidget.value = text;
	app.graph.setDirtyCanvas(true, false);
}

function installVoteControls(node) {
	if (node.__aitkVoteControlsInstalled) return;
	node.__aitkVoteControlsInstalled = true;

	const statusWidget = node.addWidget(
		"text",
		STATUS_WIDGET_NAME,
		"Ready",
		() => {},
		{ serialize: false }
	);
	if (statusWidget?.inputEl) {
		statusWidget.inputEl.readOnly = true;
		statusWidget.inputEl.style.opacity = 0.65;
	}

	const doVote = async (vote) => {
		setNodeStatus(node, `Running ${vote}...`);
		try {
			const result = await postVote(node.id, vote);
			if (vote === "manual_checkpoint") {
				const step = result?.checkpoint?.step_count ?? "unknown";
				setNodeStatus(node, `Checkpoint saved at step ${step}`);
			} else {
				const step = result?.step_count ?? "unknown";
				setNodeStatus(node, `${vote} applied | step ${step}`);
			}
		} catch (error) {
			const message = error?.message || String(error);
			setNodeStatus(node, `Error: ${message}`);
			console.error("[AITK Vote] action failed:", error);
			alert(`[AITK Vote] ${message}`);
		}
	};

	node.addWidget("button", "Upvote", "Upvote", () => doVote("upvote"));
	node.addWidget("button", "Downvote", "Downvote", () => doVote("downvote"));
	node.addWidget("button", "Skip", "Skip", () => doVote("skip"));
	node.addWidget(
		"button",
		"Manual Checkpoint",
		"Manual Checkpoint",
		() => doVote("manual_checkpoint")
	);
}

app.registerExtension({
	name: "AITK.RLHF.VoteControls",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== VOTE_CLASS_NAME) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const result = onNodeCreated?.apply(this, arguments);
			installVoteControls(this);
			return result;
		};
	},
});

