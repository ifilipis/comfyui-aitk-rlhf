import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
	modelArchs,
	groupedModelOptions,
	quantizationOptions,
	defaultQtype,
} from "./aitk_ai_toolkit_options.js";
import { defaultJobConfig } from "./aitk_ai_toolkit_jobconfig.js";

const SESSION_CLASS_NAME = "AITKRLHFSession";
const STYLE_ID = "aitk-session-node-style";
const MIN_WIDTH = 1020;
const MIN_HEIGHT = 760;

function ensureStyle() {
	if (document.getElementById(STYLE_ID)) return;
	const style = document.createElement("style");
	style.id = STYLE_ID;
	style.textContent = `
		.aitk-session-root {
			color: #e5e7eb;
			font-family: Inter, ui-sans-serif, system-ui, sans-serif;
			font-size: 12px;
			line-height: 1.4;
			width: 100%;
			box-sizing: border-box;
		}
		.aitk-session-status {
			padding: 8px 10px;
			margin-bottom: 12px;
			border-radius: 8px;
			border: 1px solid rgba(148, 163, 184, 0.25);
			background: rgba(15, 23, 42, 0.7);
			color: #cbd5e1;
		}
		.aitk-session-grid {
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
			gap: 12px;
			align-items: start;
		}
		.aitk-session-card {
			background: #111827;
			border: 1px solid rgba(255, 255, 255, 0.06);
			border-radius: 10px;
			padding: 12px;
			box-sizing: border-box;
		}
		.aitk-session-card h3 {
			margin: 0 0 10px 0;
			font-size: 15px;
			font-weight: 700;
			letter-spacing: 0.06em;
			text-transform: uppercase;
			color: #6b7280;
		}
		.aitk-session-field {
			margin-bottom: 10px;
		}
		.aitk-session-field:last-child {
			margin-bottom: 0;
		}
		.aitk-session-label {
			display: block;
			margin-bottom: 4px;
			font-size: 11px;
			color: #d1d5db;
		}
		.aitk-session-input,
		.aitk-session-select {
			width: 100%;
			box-sizing: border-box;
			padding: 7px 9px;
			border-radius: 4px;
			border: 1px solid #374151;
			background: #1f2937;
			color: #f3f4f6;
			font-size: 12px;
		}
		.aitk-session-input[readonly] {
			opacity: 0.7;
			cursor: default;
		}
		.aitk-session-row {
			display: grid;
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: 8px;
		}
		.aitk-session-checkbox {
			display: flex;
			align-items: center;
			gap: 8px;
			margin-bottom: 8px;
			color: #d1d5db;
		}
		.aitk-session-checkbox:last-child {
			margin-bottom: 0;
		}
		.aitk-session-checkbox input {
			margin: 0;
		}
		.aitk-session-slider-value {
			float: right;
			color: #94a3b8;
		}
		.aitk-session-muted {
			font-size: 11px;
			color: #94a3b8;
		}
		.aitk-session-loading,
		.aitk-session-error {
			padding: 14px;
			border-radius: 10px;
			background: #111827;
			border: 1px solid rgba(255, 255, 255, 0.08);
		}
		.aitk-session-error {
			color: #fecaca;
			border-color: rgba(248, 113, 113, 0.35);
		}
	`;
	document.head.appendChild(style);
}

function deepClone(value) {
	return JSON.parse(JSON.stringify(value));
}

function getWidget(node, name) {
	return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function hideWidget(widget) {
	if (!widget || widget.__aitkHidden) return;
	widget.__aitkHidden = true;
	widget.computeSize = () => [0, -4];
	widget.type = "converted-widget";
	if (widget.inputEl) {
		widget.inputEl.style.display = "none";
	}
}

function getProcess(state) {
	return state?.job_config?.config?.process?.[0] || null;
}

function setByPath(obj, path, value) {
	const parts = String(path || "").split(".").filter(Boolean);
	if (!parts.length) return;
	let cursor = obj;
	for (let i = 0; i < parts.length - 1; i += 1) {
		const part = parts[i];
		if (part.includes("[") || part.includes("]")) return;
		if (!cursor[part] || typeof cursor[part] !== "object" || Array.isArray(cursor[part])) {
			cursor[part] = {};
		}
		cursor = cursor[part];
	}
	cursor[parts[parts.length - 1]] = value;
}

function getByPath(obj, path, fallback = undefined) {
	const parts = String(path || "").split(".").filter(Boolean);
	let cursor = obj;
	for (const part of parts) {
		if (cursor == null || typeof cursor !== "object") return fallback;
		cursor = cursor[part];
	}
	return cursor == null ? fallback : cursor;
}

function escapeHtml(value) {
	return String(value ?? "")
		.replaceAll("&", "&amp;")
		.replaceAll("<", "&lt;")
		.replaceAll(">", "&gt;")
		.replaceAll('"', "&quot;")
		.replaceAll("'", "&#39;");
}

function groupedOptionsToHtml(groups, selectedValue) {
	return groups
		.map(
			(group) => `
				<optgroup label="${escapeHtml(group.label)}">
					${(group.options || [])
						.map(
							(option) => `
								<option value="${escapeHtml(option.value)}" ${option.value === selectedValue ? "selected" : ""}>
									${escapeHtml(option.label)}
								</option>
							`,
						)
						.join("")}
				</optgroup>
			`,
		)
		.join("");
}

function optionsToHtml(options, selectedValue) {
	return (options || [])
		.map(
			(option) => `
				<option value="${escapeHtml(option.value)}" ${option.value === selectedValue ? "selected" : ""}>
					${escapeHtml(option.label)}
				</option>
			`,
		)
		.join("");
}

function buildTransformerQuantOptions(schema, modelArch) {
	const quantizationOptions = schema?.quantization_options || [];
	const adapters = modelArch?.accuracyRecoveryAdapters || {};
	const adapterEntries = Object.entries(adapters);
	if (!adapterEntries.length) {
		return [{ label: "Standard", options: quantizationOptions }];
	}
	return [
		{
			label: "Standard",
			options: quantizationOptions.slice(0, 2),
		},
		{
			label: "Accuracy Recovery Adapters",
			options: adapterEntries.map(([label, value]) => ({ label, value })),
		},
		{
			label: "Additional Quantization Options",
			options: quantizationOptions.slice(2),
		},
	];
}

function createDefaultState(schema) {
	const state = {
		job_config: deepClone(defaultJobConfig),
		runtime: {
			device: "cuda",
			dtype: "fp16",
			checkpoint_root:
				"/content/comfyui-aitk-rlhf/third_party/ai-toolkit/output/aitk_flow_grpo",
			resume: true,
		},
		grpo: {
			clip_range: 1e-4,
			adv_clip_max: 5.0,
			beta: 0.0,
			noise_level: 0.7,
			sde_type: "sde",
			timestep_fraction: 1.0,
		},
	};
	const process = getProcess(state);
	process.save.save_every = 25;
	process.save.max_step_saves_to_keep = 4;
	process.save.dtype = "bf16";
	process.train.optimizer = "adamw";
	process.model.quantize = false;
	process.model.quantize_te = false;
	process.model.qtype = defaultQtype;
	process.model.qtype_te = defaultQtype;
	process.model.layer_offloading = false;
	process.model.layer_offloading_transformer_percent = 1.0;
	process.model.layer_offloading_text_encoder_percent = 1.0;
	process.model.assistant_lora_path = "";
	process.model.accuracy_recovery_adapter = "";
	process.model.model_kwargs = process.model.model_kwargs || {};
	process.configName = undefined;
	const initialArch = schema.default_model_arch;
	applyPresetDefaults(state, schema, process.model.arch, initialArch);
	state.job_config.config.name = "aitk-session-1";
	return state;
}

function normalizeState(schema, state) {
	const nextState = deepClone(state || createDefaultState(schema));
	const process = getProcess(nextState);
	if (!process) return createDefaultState(schema);
	const currentArch = getByPath(process, "model.arch", schema.default_model_arch);
	if (!schema.model_archs.find((item) => item.name === currentArch)) {
		setByPath(process, "model.arch", schema.default_model_arch);
	}
	return nextState;
}

function applyPresetDefaults(state, schema, oldArch, newArch) {
	const process = getProcess(state);
	if (!process) return;
	const byName = new Map((schema.model_archs || []).map((item) => [item.name, item]));
	const oldPreset = byName.get(oldArch);
	const newPreset = byName.get(newArch);

	const applyDefaults = (defaults, index) => {
		for (const [path, pair] of Object.entries(defaults || {})) {
			if (
				path.startsWith("datasets[") ||
				path.startsWith("sample.") ||
				path.startsWith("slider.")
			) {
				continue;
			}
			let value = pair;
			if (Array.isArray(pair)) {
				value = pair[index];
			}
			setByPath(process, path, deepClone(value));
		}
	};

	if (oldPreset) {
		applyDefaults(oldPreset.defaults, 1);
	}
	if (newPreset) {
		applyDefaults(newPreset.defaults, 0);
	}

	setByPath(process, "model.arch", newArch);

	if (!(newPreset?.additionalSections || []).includes("model.low_vram")) {
		setByPath(process, "model.low_vram", false);
	}
	if (!(newPreset?.additionalSections || []).includes("model.layer_offloading")) {
		setByPath(process, "model.layer_offloading", false);
		setByPath(process, "model.layer_offloading_transformer_percent", 1.0);
		setByPath(process, "model.layer_offloading_text_encoder_percent", 1.0);
	}
	if (!(newPreset?.additionalSections || []).includes("model.assistant_lora_path")) {
		setByPath(process, "model.assistant_lora_path", "");
	}
	if (!(newPreset?.additionalSections || []).includes("model.qie.match_target_res")) {
		setByPath(process, "model.model_kwargs.match_target_res", false);
	}
	if (!(newPreset?.additionalSections || []).includes("model.multistage")) {
		setByPath(process, "model.model_kwargs.train_high_noise", false);
		setByPath(process, "model.model_kwargs.train_low_noise", false);
		setByPath(process, "train.switch_boundary_every", 1);
	}
}

function buildSchema() {
	const normalizedModelArchs = (modelArchs || []).map((item) => ({
		name: item.name,
		label: item.label,
		group: item.group,
		defaults: item.defaults || {},
		disableSections: item.disableSections || [],
		additionalSections: item.additionalSections || [],
		accuracyRecoveryAdapters: item.accuracyRecoveryAdapters || {},
	}));
	const defaultModelArch = normalizedModelArchs.some((item) => item.name === "sd15")
		? "sd15"
		: normalizedModelArchs[0]?.name || "sd15";
	return {
		default_state: null,
		default_model_arch: defaultModelArch,
		model_archs: normalizedModelArchs,
		grouped_model_options: groupedModelOptions || [],
		quantization_options: quantizationOptions || [],
		default_qtype: defaultQtype,
		network_type_options: [
			{ value: "lora", label: "LoRA" },
			{ value: "lokr", label: "LoKr" },
		],
		dtype_options: [
			{ value: "bf16", label: "BF16" },
			{ value: "fp16", label: "FP16" },
			{ value: "fp32", label: "FP32" },
		],
		sde_type_options: [
			{ value: "sde", label: "SDE" },
			{ value: "cps", label: "CPS" },
		],
	};
}

function syncNodeState(node) {
	const controller = node.__aitkSessionController;
	if (!controller) return;
	const sessionWidget = controller.widgets.sessionId;
	const configWidget = controller.widgets.configJson;
	const forceResetWidget = controller.widgets.forceReset;
	const sessionName = getByPath(controller.state, "job_config.config.name", "aitk-session-1");
	if (sessionWidget) sessionWidget.value = sessionName;
	if (configWidget) configWidget.value = JSON.stringify(controller.state);
	if (forceResetWidget) forceResetWidget.value = !!controller.forceReset;
	app.graph.setDirtyCanvas(true, false);
}

function resizeNode(node, root) {
	const width = Math.max(MIN_WIDTH, Math.ceil(root.scrollWidth + 24));
	const height = Math.max(MIN_HEIGHT, Math.ceil(root.scrollHeight + 48));
	node.setSize([width, height]);
	app.graph.setDirtyCanvas(true, false);
}

function render(node) {
	const controller = node.__aitkSessionController;
	if (!controller) return;
	const { root, schema, state, statusMessage, errorMessage } = controller;
	if (!schema) {
		root.innerHTML = `<div class="aitk-session-loading">Loading AI Toolkit session schema...</div>`;
		resizeNode(node, root);
		return;
	}

	const process = getProcess(state);
	if (!process) {
		root.innerHTML = `<div class="aitk-session-error">Invalid session state.</div>`;
		resizeNode(node, root);
		return;
	}

	const modelArchName = getByPath(process, "model.arch", schema.default_model_arch);
	const modelArch = schema.model_archs.find((item) => item.name === modelArchName) || schema.model_archs[0];
	const additionalSections = new Set(modelArch?.additionalSections || []);
	const disableSections = new Set(modelArch?.disableSections || []);
	const transformerQuantOptions = buildTransformerQuantOptions(schema, modelArch);
	const transformerQuantValue = getByPath(process, "model.quantize", false)
		? getByPath(process, "model.qtype", schema.default_qtype)
		: "";
	const textEncoderQuantValue = getByPath(process, "model.quantize_te", false)
		? getByPath(process, "model.qtype_te", schema.default_qtype)
		: "";

	root.innerHTML = `
		<div class="aitk-session-root">
			<div class="aitk-session-status">${escapeHtml(errorMessage || statusMessage || "AI Toolkit session configurator")}</div>
			<div class="aitk-session-grid">
				<section class="aitk-session-card">
					<h3>Job</h3>
					<div class="aitk-session-field">
						<label class="aitk-session-label">Training Name</label>
						<input class="aitk-session-input" data-text-path="job_config.config.name" value="${escapeHtml(
							getByPath(state, "job_config.config.name", "aitk-session-1"),
						)}" />
					</div>
					<label class="aitk-session-checkbox">
						<input type="checkbox" data-bool-path="runtime.resume" ${getByPath(state, "runtime.resume", true) ? "checked" : ""} />
						<span>Resume</span>
					</label>
					<label class="aitk-session-checkbox">
						<input type="checkbox" data-force-reset ${controller.forceReset ? "checked" : ""} />
						<span>Force Reset</span>
					</label>
				</section>

				<section class="aitk-session-card">
					<h3>Model</h3>
					<div class="aitk-session-field">
						<label class="aitk-session-label">Model Architecture</label>
						<select class="aitk-session-select" data-model-arch>
							${groupedOptionsToHtml(schema.grouped_model_options, modelArchName)}
						</select>
					</div>
					<div class="aitk-session-field">
						<label class="aitk-session-label">Name or Path</label>
						<input class="aitk-session-input" readonly value="${escapeHtml(
							getByPath(process, "model.name_or_path", ""),
						)}" />
					</div>
					${
						getByPath(process, "model.extras_name_or_path", "")
							? `
								<div class="aitk-session-field">
									<label class="aitk-session-label">Extras Name or Path</label>
									<input class="aitk-session-input" readonly value="${escapeHtml(
										getByPath(process, "model.extras_name_or_path", ""),
									)}" />
								</div>
							`
							: ""
					}
					${
						additionalSections.has("model.assistant_lora_path")
							? `
								<div class="aitk-session-field">
									<label class="aitk-session-label">Training Adapter Path</label>
									<input class="aitk-session-input" readonly value="${escapeHtml(
										getByPath(process, "model.assistant_lora_path", ""),
									)}" />
								</div>
							`
							: ""
					}
					${
						additionalSections.has("model.low_vram")
							? `
								<label class="aitk-session-checkbox">
									<input type="checkbox" data-bool-path="job_config.config.process.0.model.low_vram" ${getByPath(process, "model.low_vram", false) ? "checked" : ""} />
									<span>Low VRAM</span>
								</label>
							`
							: ""
					}
					${
						additionalSections.has("model.qie.match_target_res")
							? `
								<label class="aitk-session-checkbox">
									<input type="checkbox" data-bool-path="job_config.config.process.0.model.model_kwargs.match_target_res" ${getByPath(process, "model.model_kwargs.match_target_res", false) ? "checked" : ""} />
									<span>Match Target Res</span>
								</label>
							`
							: ""
					}
					${
						additionalSections.has("model.layer_offloading")
							? `
								<label class="aitk-session-checkbox">
									<input type="checkbox" data-bool-path="job_config.config.process.0.model.layer_offloading" ${getByPath(process, "model.layer_offloading", false) ? "checked" : ""} />
									<span>Layer Offloading</span>
								</label>
								${
									getByPath(process, "model.layer_offloading", false)
										? `
											<div class="aitk-session-field">
												<label class="aitk-session-label">
													Transformer Offload %
													<span class="aitk-session-slider-value">${Math.round(
														100 * getByPath(process, "model.layer_offloading_transformer_percent", 1),
													)}%</span>
												</label>
												<input type="range" min="0" max="100" step="1" value="${Math.round(
													100 * getByPath(process, "model.layer_offloading_transformer_percent", 1),
												)}" data-percent-path="job_config.config.process.0.model.layer_offloading_transformer_percent" />
											</div>
											<div class="aitk-session-field">
												<label class="aitk-session-label">
													Text Encoder Offload %
													<span class="aitk-session-slider-value">${Math.round(
														100 * getByPath(process, "model.layer_offloading_text_encoder_percent", 1),
													)}%</span>
												</label>
												<input type="range" min="0" max="100" step="1" value="${Math.round(
													100 * getByPath(process, "model.layer_offloading_text_encoder_percent", 1),
												)}" data-percent-path="job_config.config.process.0.model.layer_offloading_text_encoder_percent" />
											</div>
										`
										: ""
								}
							`
							: ""
					}
				</section>

				${
					disableSections.has("model.quantize")
						? ""
						: `
							<section class="aitk-session-card">
								<h3>Quantization</h3>
								<div class="aitk-session-field">
									<label class="aitk-session-label">Transformer</label>
									<select class="aitk-session-select" data-transformer-quant>
										${groupedOptionsToHtml(transformerQuantOptions, transformerQuantValue)}
									</select>
								</div>
								<div class="aitk-session-field">
									<label class="aitk-session-label">Text Encoder</label>
									<select class="aitk-session-select" data-text-encoder-quant>
										${optionsToHtml(schema.quantization_options, textEncoderQuantValue)}
									</select>
								</div>
							</section>
						`
				}

				<section class="aitk-session-card">
					<h3>Target</h3>
					<div class="aitk-session-field">
						<label class="aitk-session-label">Target Type</label>
						<select class="aitk-session-select" data-select-path="job_config.config.process.0.network.type">
							${optionsToHtml(
								schema.network_type_options,
								getByPath(process, "network.type", "lora"),
							)}
						</select>
					</div>
					${
						getByPath(process, "network.type", "lora") === "lokr"
							? `
								<div class="aitk-session-field">
									<label class="aitk-session-label">LoKr Factor</label>
									<select class="aitk-session-select" data-select-path="job_config.config.process.0.network.lokr_factor">
										${optionsToHtml(
											[
												{ value: "-1", label: "Auto" },
												{ value: "4", label: "4" },
												{ value: "8", label: "8" },
												{ value: "16", label: "16" },
												{ value: "32", label: "32" },
											],
											String(getByPath(process, "network.lokr_factor", -1)),
										)}
									</select>
								</div>
							`
							: `
								<div class="aitk-session-row">
									<div class="aitk-session-field">
										<label class="aitk-session-label">Linear Rank</label>
										<input type="number" min="1" max="1024" step="1" class="aitk-session-input" data-int-path="job_config.config.process.0.network.linear" value="${escapeHtml(
											getByPath(process, "network.linear", 32),
										)}" />
									</div>
									${
										disableSections.has("network.conv")
											? ""
											: `
												<div class="aitk-session-field">
													<label class="aitk-session-label">Conv Rank</label>
													<input type="number" min="0" max="1024" step="1" class="aitk-session-input" data-int-path="job_config.config.process.0.network.conv" value="${escapeHtml(
														getByPath(process, "network.conv", 16),
													)}" />
												</div>
											`
									}
								</div>
							`
					}
				</section>

				${
					additionalSections.has("model.multistage")
						? `
							<section class="aitk-session-card">
								<h3>Multistage</h3>
								<label class="aitk-session-checkbox">
									<input type="checkbox" data-bool-path="job_config.config.process.0.model.model_kwargs.train_high_noise" ${getByPath(process, "model.model_kwargs.train_high_noise", false) ? "checked" : ""} />
									<span>High Noise</span>
								</label>
								<label class="aitk-session-checkbox">
									<input type="checkbox" data-bool-path="job_config.config.process.0.model.model_kwargs.train_low_noise" ${getByPath(process, "model.model_kwargs.train_low_noise", false) ? "checked" : ""} />
									<span>Low Noise</span>
								</label>
								<div class="aitk-session-field">
									<label class="aitk-session-label">Switch Every</label>
									<input type="number" min="1" step="1" class="aitk-session-input" data-int-path="job_config.config.process.0.train.switch_boundary_every" value="${escapeHtml(
										getByPath(process, "train.switch_boundary_every", 1),
									)}" />
								</div>
							</section>
						`
						: ""
				}

				<section class="aitk-session-card">
					<h3>Save</h3>
					<div class="aitk-session-field">
						<label class="aitk-session-label">Data Type</label>
						<select class="aitk-session-select" data-save-dtype>
							${optionsToHtml(schema.dtype_options, getByPath(process, "save.dtype", "bf16"))}
						</select>
					</div>
					<div class="aitk-session-field">
						<label class="aitk-session-label">Save Every</label>
						<input type="number" min="1" step="1" class="aitk-session-input" data-int-path="job_config.config.process.0.save.save_every" value="${escapeHtml(
							getByPath(process, "save.save_every", 25),
						)}" />
					</div>
					<div class="aitk-session-field">
						<label class="aitk-session-label">Max Step Saves to Keep</label>
						<input type="number" min="1" step="1" class="aitk-session-input" data-int-path="job_config.config.process.0.save.max_step_saves_to_keep" value="${escapeHtml(
							getByPath(process, "save.max_step_saves_to_keep", 4),
						)}" />
					</div>
				</section>

				<section class="aitk-session-card">
					<h3>GRPO</h3>
					<div class="aitk-session-row">
						<div class="aitk-session-field">
							<label class="aitk-session-label">Clip Range</label>
							<input type="number" min="0.00000001" step="0.000001" class="aitk-session-input" data-float-path="grpo.clip_range" value="${escapeHtml(
								getByPath(state, "grpo.clip_range", 0.0001),
							)}" />
						</div>
						<div class="aitk-session-field">
							<label class="aitk-session-label">Adv Clip Max</label>
							<input type="number" min="0.1" step="0.1" class="aitk-session-input" data-float-path="grpo.adv_clip_max" value="${escapeHtml(
								getByPath(state, "grpo.adv_clip_max", 5.0),
							)}" />
						</div>
					</div>
					<div class="aitk-session-row">
						<div class="aitk-session-field">
							<label class="aitk-session-label">Beta</label>
							<input type="number" min="0" step="0.001" class="aitk-session-input" data-float-path="grpo.beta" value="${escapeHtml(
								getByPath(state, "grpo.beta", 0.0),
							)}" />
						</div>
						<div class="aitk-session-field">
							<label class="aitk-session-label">Noise Level</label>
							<input type="number" min="0" max="1" step="0.001" class="aitk-session-input" data-float-path="grpo.noise_level" value="${escapeHtml(
								getByPath(state, "grpo.noise_level", 0.7),
							)}" />
						</div>
					</div>
					<div class="aitk-session-row">
						<div class="aitk-session-field">
							<label class="aitk-session-label">SDE Type</label>
							<select class="aitk-session-select" data-select-path="grpo.sde_type">
								${optionsToHtml(schema.sde_type_options, getByPath(state, "grpo.sde_type", "sde"))}
							</select>
						</div>
						<div class="aitk-session-field">
							<label class="aitk-session-label">Timestep Fraction</label>
							<input type="number" min="0.01" max="1" step="0.01" class="aitk-session-input" data-float-path="grpo.timestep_fraction" value="${escapeHtml(
								getByPath(state, "grpo.timestep_fraction", 1.0),
							)}" />
						</div>
					</div>
				</section>
			</div>
			<div class="aitk-session-muted" style="margin-top: 10px;">
				Model endpoints are derived from the selected AI Toolkit architecture and are not user-editable in this node.
			</div>
		</div>
	`;

	root.querySelector("[data-model-arch]")?.addEventListener("change", (event) => {
		const nextArch = event.target.value;
		const currentArch = getByPath(getProcess(controller.state), "model.arch", schema.default_model_arch);
		applyPresetDefaults(controller.state, schema, currentArch, nextArch);
		syncNodeState(node);
		render(node);
	});

	root.querySelector("[data-transformer-quant]")?.addEventListener("change", (event) => {
		const value = event.target.value;
		setByPath(process, "model.quantize", value !== "");
		setByPath(process, "model.qtype", value || schema.default_qtype);
		setByPath(
			process,
			"model.accuracy_recovery_adapter",
			value.includes("|") ? value.split("|").slice(1).join("|") : "",
		);
		syncNodeState(node);
		render(node);
	});

	root.querySelector("[data-text-encoder-quant]")?.addEventListener("change", (event) => {
		const value = event.target.value;
		setByPath(process, "model.quantize_te", value !== "");
		setByPath(process, "model.qtype_te", value || schema.default_qtype);
		syncNodeState(node);
		render(node);
	});

	root.querySelector("[data-save-dtype]")?.addEventListener("change", (event) => {
		const value = event.target.value;
		setByPath(process, "save.dtype", value);
		setByPath(controller.state, "runtime.dtype", value);
		syncNodeState(node);
		render(node);
	});

	root.querySelector("[data-force-reset]")?.addEventListener("change", (event) => {
		controller.forceReset = !!event.target.checked;
		syncNodeState(node);
	});

	for (const input of root.querySelectorAll("[data-text-path]")) {
		input.addEventListener("input", (event) => {
			setByPath(controller.state, event.target.dataset.textPath, event.target.value);
			syncNodeState(node);
		});
	}

	for (const input of root.querySelectorAll("[data-int-path]")) {
		input.addEventListener("change", (event) => {
			const value = Number.parseInt(event.target.value, 10);
			if (!Number.isNaN(value)) {
				setByPath(controller.state, event.target.dataset.intPath, value);
				if (event.target.dataset.intPath === "job_config.config.process.0.network.linear") {
					setByPath(controller.state, "job_config.config.process.0.network.linear_alpha", value);
				}
				if (event.target.dataset.intPath === "job_config.config.process.0.network.conv") {
					setByPath(controller.state, "job_config.config.process.0.network.conv_alpha", value);
				}
				syncNodeState(node);
				render(node);
			}
		});
	}

	for (const input of root.querySelectorAll("[data-float-path]")) {
		input.addEventListener("change", (event) => {
			const value = Number.parseFloat(event.target.value);
			if (!Number.isNaN(value)) {
				setByPath(controller.state, event.target.dataset.floatPath, value);
				syncNodeState(node);
			}
		});
	}

	for (const input of root.querySelectorAll("[data-select-path]")) {
		input.addEventListener("change", (event) => {
			const value = event.target.value;
			const path = event.target.dataset.selectPath;
			setByPath(controller.state, path, path.endsWith("lokr_factor") ? Number.parseInt(value, 10) : value);
			syncNodeState(node);
			render(node);
		});
	}

	for (const input of root.querySelectorAll("[data-bool-path]")) {
		input.addEventListener("change", (event) => {
			setByPath(controller.state, event.target.dataset.boolPath, !!event.target.checked);
			syncNodeState(node);
			render(node);
		});
	}

	for (const input of root.querySelectorAll("[data-percent-path]")) {
		input.addEventListener("input", (event) => {
			const value = Number.parseFloat(event.target.value);
			if (!Number.isNaN(value)) {
				setByPath(controller.state, event.target.dataset.percentPath, value / 100);
				syncNodeState(node);
				render(node);
			}
		});
	}

	resizeNode(node, root);
}

async function installSessionUi(node) {
	if (node.__aitkSessionUiInstalled) return;
	node.__aitkSessionUiInstalled = true;
	ensureStyle();

	const widgets = {
		sessionId: getWidget(node, "session_id"),
		configJson: getWidget(node, "config_json"),
		forceReset: getWidget(node, "force_reset"),
	};
	hideWidget(widgets.sessionId);
	hideWidget(widgets.configJson);
	hideWidget(widgets.forceReset);

	const root = document.createElement("div");
	root.style.width = "100%";
	root.style.boxSizing = "border-box";
	node.addDOMWidget("aitk_session_ui", "aitk_session_ui", root, { serialize: false });

	let parsedState = null;
	try {
		parsedState = widgets.configJson?.value ? JSON.parse(widgets.configJson.value) : null;
	} catch {
		parsedState = null;
	}

	node.__aitkSessionController = {
		root,
		schema: null,
		state: parsedState,
		forceReset: !!widgets.forceReset?.value,
		statusMessage: "Loading AI Toolkit session schema...",
		errorMessage: "",
		widgets,
	};
	render(node);

	try {
		const schema = buildSchema();
		schema.default_state = createDefaultState(schema);
		const controller = node.__aitkSessionController;
		if (!controller) return;
		controller.schema = schema;
		controller.state = normalizeState(schema, controller.state || schema.default_state);
		controller.statusMessage = "AI Toolkit session UI ready";
		controller.errorMessage = "";
		syncNodeState(node);
		render(node);
	} catch (error) {
		const controller = node.__aitkSessionController;
		if (!controller) return;
		controller.errorMessage = error?.message || String(error);
		render(node);
	}
}

app.registerExtension({
	name: "AITK.RLHF.SessionUI",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== SESSION_CLASS_NAME) return;
		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const result = onNodeCreated?.apply(this, arguments);
			installSessionUi(this);
			return result;
		};
	},
});
