from __future__ import annotations

import ast
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aiohttp import web
from PIL import Image

import folder_paths
from server import PromptServer

_NODE_ROOT = Path(__file__).resolve().parent
_THIRD_PARTY_AITK_ROOT = _NODE_ROOT / "third_party" / "ai-toolkit"
_PROMPT_SERVER_INSTANCE = getattr(PromptServer, "instance", None)


def _ensure_aitk_path() -> None:
    ai_toolkit_root = Path(
        os.environ.get("AITK_ROOT", str(_THIRD_PARTY_AITK_ROOT))
    ).resolve()
    if ai_toolkit_root.exists() and str(ai_toolkit_root) not in sys.path:
        sys.path.insert(0, str(ai_toolkit_root))


_ensure_aitk_path()

try:
    from toolkit.online_flow_grpo import (  # type: ignore
        GRPOConfig,
        LoRAConfigSpec,
        OptimizerConfig,
        SessionConfig,
        SessionError,
        get_online_flow_grpo_manager,
    )
except Exception as import_error:  # pragma: no cover
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


def _raise_if_unavailable() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import AI Toolkit online Flow-GRPO module. "
            "Set AITK_ROOT to your ai-toolkit fork path and ensure dependencies are installed. "
            f"Original error: {_IMPORT_ERROR}"
        )


_AITK_UI_OPTIONS_PATH = Path(
    os.environ.get(
        "AITK_UI_MODEL_OPTIONS_PATH",
        str(_THIRD_PARTY_AITK_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"),
    )
).resolve()
_MODEL_PRESET_LOCK = threading.Lock()
_MODEL_PRESET_CACHE: dict[str, dict[str, Any]] | None = None
_TRISTATE_BOOL = ["auto", "true", "false"]
_KNOWN_JS_IDENTIFIERS: dict[str, Any] = {
    "defaultNameOrPath": "",
    "defaultLinearRank": 32,
}


def _tristate_to_optional_bool(value: str) -> bool | None:
    normalized = str(value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _extract_balanced(text: str, start_idx: int, open_char: str, close_char: str) -> tuple[str, int]:
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != open_char:
        raise ValueError("Invalid balanced-expression start index.")

    depth = 0
    quote = ""
    in_string = False
    escaped = False

    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_string = False
            continue

        if ch in {"'", '"', "`"}:
            in_string = True
            quote = ch
            continue

        if ch == open_char:
            depth += 1
            continue
        if ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1], idx + 1

    raise ValueError("Unbalanced expression.")


def _first_js_list_item(list_literal: str) -> str:
    content = list_literal.strip()
    if not content.startswith("[") or not content.endswith("]"):
        return content

    inner = content[1:-1].strip()
    if not inner:
        return ""

    depth_curly = 0
    depth_square = 0
    depth_paren = 0
    quote = ""
    in_string = False
    escaped = False

    for idx, ch in enumerate(inner):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_string = False
            continue

        if ch in {"'", '"', "`"}:
            in_string = True
            quote = ch
            continue

        if ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "," and depth_curly == 0 and depth_square == 0 and depth_paren == 0:
            return inner[:idx].strip()

    return inner.strip()


def _split_js_list_items(list_literal: str) -> list[str]:
    content = list_literal.strip()
    if not content.startswith("[") or not content.endswith("]"):
        return [content]

    inner = content[1:-1].strip()
    if not inner:
        return []

    items: list[str] = []
    depth_curly = 0
    depth_square = 0
    depth_paren = 0
    quote = ""
    in_string = False
    escaped = False
    start = 0

    for idx, ch in enumerate(inner):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_string = False
            continue

        if ch in {"'", '"', "`"}:
            in_string = True
            quote = ch
            continue

        if ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "," and depth_curly == 0 and depth_square == 0 and depth_paren == 0:
            item = inner[start:idx].strip()
            if item:
                items.append(item)
            start = idx + 1

    tail = inner[start:].strip()
    if tail:
        items.append(tail)
    return items


def _parse_js_literal(value: str) -> Any:
    text = str(value or "").strip()
    if text == "":
        return None
    if text in _KNOWN_JS_IDENTIFIERS:
        return _KNOWN_JS_IDENTIFIERS[text]
    if text in {"undefined", "null"}:
        return None
    if text == "true":
        return True
    if text == "false":
        return False

    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        try:
            return ast.literal_eval(text)
        except Exception:
            return text[1:-1]

    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        pass

    python_text = text
    python_text = re.sub(r"//.*", "", python_text)
    python_text = python_text.replace("undefined", "None")
    python_text = python_text.replace("null", "None")
    python_text = python_text.replace("true", "True")
    python_text = python_text.replace("false", "False")
    python_text = re.sub(r"([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', python_text)
    python_text = re.sub(r",(\s*[}\]])", r"\1", python_text)

    try:
        return ast.literal_eval(python_text)
    except Exception:
        return text


def _extract_model_default_pairs(defaults_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    prefix = "'config.process[0]."
    cursor = 0
    while True:
        key_start = defaults_text.find(prefix, cursor)
        if key_start < 0:
            break
        key_start += len(prefix)
        key_end = defaults_text.find("'", key_start)
        if key_end < 0:
            break
        key = defaults_text[key_start:key_end].strip()
        if not key:
            cursor = key_end + 1
            continue

        colon_idx = defaults_text.find(":", key_end)
        if colon_idx < 0:
            break
        list_start = defaults_text.find("[", colon_idx)
        if list_start < 0:
            break

        try:
            list_literal, next_idx = _extract_balanced(defaults_text, list_start, "[", "]")
        except ValueError:
            cursor = list_start + 1
            continue

        out[key] = [_parse_js_literal(item) for item in _split_js_list_items(list_literal)]
        cursor = next_idx

    return out


def _extract_model_defaults(defaults_text: str) -> dict[str, Any]:
    selected_defaults: dict[str, Any] = {}
    for key, value in _extract_model_default_pairs(defaults_text).items():
        if isinstance(value, list) and value:
            selected_defaults[key] = value[0]
        else:
            selected_defaults[key] = value
    return selected_defaults


def _extract_js_key_literal(block: str, key: str, open_char: str, close_char: str) -> Any:
    marker = f"{key}:"
    marker_idx = block.find(marker)
    if marker_idx < 0:
        return None
    literal_start = block.find(open_char, marker_idx)
    if literal_start < 0:
        return None
    try:
        literal_text, _ = _extract_balanced(block, literal_start, open_char, close_char)
    except ValueError:
        return None
    return _parse_js_literal(literal_text)


def _load_aitk_ui_model_presets() -> dict[str, dict[str, Any]]:
    global _MODEL_PRESET_CACHE
    with _MODEL_PRESET_LOCK:
        if _MODEL_PRESET_CACHE is not None:
            return _MODEL_PRESET_CACHE

        presets: dict[str, dict[str, Any]] = {
            "custom": {"label": "Custom", "arch": "sd15"},
        }

        if not _AITK_UI_OPTIONS_PATH.exists():
            _MODEL_PRESET_CACHE = presets
            return presets

        try:
            content = _AITK_UI_OPTIONS_PATH.read_text(encoding="utf-8")
            marker = "export const modelArchs: ModelArch[] = ["
            marker_idx = content.find(marker)
            if marker_idx < 0:
                _MODEL_PRESET_CACHE = presets
                return presets

            array_start = content.find("[", marker_idx + len(marker) - 1)
            model_archs_array, _ = _extract_balanced(content, array_start, "[", "]")
            array_body = model_archs_array[1:-1]

            idx = 0
            while idx < len(array_body):
                if array_body[idx] != "{":
                    idx += 1
                    continue
                block, next_idx = _extract_balanced(array_body, idx, "{", "}")
                idx = next_idx

                name_match = re.search(r"\bname\s*:\s*'([^']+)'", block)
                if not name_match:
                    continue
                name = name_match.group(1).strip()
                label_match = re.search(r"\blabel\s*:\s*'([^']+)'", block)
                label = label_match.group(1).strip() if label_match else name
                group_match = re.search(r"\bgroup\s*:\s*'([^']+)'", block)
                group = group_match.group(1).strip() if group_match else "image"

                preset: dict[str, Any] = {
                    "label": label,
                    "group": group,
                    "arch": name,
                }

                disable_sections = _extract_js_key_literal(block, "disableSections", "[", "]")
                if isinstance(disable_sections, list):
                    preset["disableSections"] = disable_sections

                additional_sections = _extract_js_key_literal(block, "additionalSections", "[", "]")
                if isinstance(additional_sections, list):
                    preset["additionalSections"] = additional_sections

                accuracy_recovery_adapters = _extract_js_key_literal(
                    block,
                    "accuracyRecoveryAdapters",
                    "{",
                    "}",
                )
                if isinstance(accuracy_recovery_adapters, dict):
                    preset["accuracyRecoveryAdapters"] = accuracy_recovery_adapters

                defaults_idx = block.find("defaults:")
                if defaults_idx >= 0:
                    defaults_start = block.find("{", defaults_idx)
                    if defaults_start >= 0:
                        defaults_obj, _ = _extract_balanced(block, defaults_start, "{", "}")
                        default_pairs = _extract_model_default_pairs(defaults_obj)
                        preset["defaults"] = default_pairs
                        for key, value in _extract_model_defaults(defaults_obj).items():
                            preset[key] = value
                            if key.startswith("model."):
                                preset[key.split("model.", 1)[1]] = value

                if "name_or_path" not in preset:
                    preset["name_or_path"] = ""
                if "qtype" not in preset:
                    preset["qtype"] = "qfloat8"
                if "qtype_te" not in preset:
                    preset["qtype_te"] = "qfloat8"

                presets[name] = preset
        except Exception:
            pass

        _MODEL_PRESET_CACHE = presets
        return presets


def _model_preset_names() -> list[str]:
    return list(_load_aitk_ui_model_presets().keys())


_NODE_STATE_LOCK = threading.RLock()
_NODE_RUNTIME_STATE: dict[str, dict[str, Any]] = {}
_SESSION_LOG_LOCK = threading.RLock()
_SESSION_LOGS: dict[str, list[dict[str, Any]]] = {}
_SESSION_LOG_SEQ = 0
_MAX_SESSION_LOG_ENTRIES = int(os.environ.get("AITK_RLHF_MAX_LOG_ENTRIES", "1000"))


def _set_node_state(
    node_id: str,
    *,
    session_id: str,
    candidate_id: str | None,
    **extra: Any,
) -> None:
    with _NODE_STATE_LOCK:
        state = {
            "session_id": session_id,
            "candidate_id": candidate_id,
        }
        if extra:
            state.update(extra)
        _NODE_RUNTIME_STATE[str(node_id)] = state


def _get_node_state(node_id: str) -> dict[str, Any] | None:
    with _NODE_STATE_LOCK:
        return _NODE_RUNTIME_STATE.get(str(node_id))


def _now_iso_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _append_session_log(
    session_id: str,
    *,
    event: str,
    message: str,
    level: str = "info",
    node_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    normalized_session = str(session_id or "").strip()
    if not normalized_session:
        return

    global _SESSION_LOG_SEQ
    with _SESSION_LOG_LOCK:
        _SESSION_LOG_SEQ += 1
        entry = {
            "seq": _SESSION_LOG_SEQ,
            "timestamp": _now_iso_utc(),
            "timestamp_ms": int(time.time() * 1000),
            "event": str(event),
            "level": str(level),
            "message": str(message),
            "node_id": str(node_id) if node_id is not None else None,
            "payload": payload or {},
        }
        log_entries = _SESSION_LOGS.setdefault(normalized_session, [])
        log_entries.append(entry)
        if len(log_entries) > _MAX_SESSION_LOG_ENTRIES:
            del log_entries[: len(log_entries) - _MAX_SESSION_LOG_ENTRIES]


def _get_session_logs(session_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
    normalized_session = str(session_id or "").strip()
    if not normalized_session:
        return []
    safe_limit = max(1, min(int(limit), _MAX_SESSION_LOG_ENTRIES))
    with _SESSION_LOG_LOCK:
        entries = _SESSION_LOGS.get(normalized_session, [])
        if not entries:
            return []
        return entries[-safe_limit:]


def _get_safe_session_summary(session_id: str) -> tuple[dict[str, Any] | None, str | None]:
    if _IMPORT_ERROR is not None:
        return None, "AI Toolkit online Flow-GRPO module unavailable."

    try:
        manager = get_online_flow_grpo_manager()
        session = manager.get_session(session_id)
        return session.summary(), None
    except SessionError as error:
        return None, str(error)
    except Exception as error:  # pragma: no cover
        return None, f"Unexpected error: {error}"


def _image_bhwc_to_bchw(images: torch.Tensor) -> torch.Tensor:
    # Comfy IMAGE tensors are [B, H, W, C] in [0, 1].
    return images.permute(0, 3, 1, 2).contiguous()


def _image_bchw_to_bhwc(images: torch.Tensor) -> torch.Tensor:
    # AI Toolkit generation returns [B, C, H, W] in [0, 1].
    return images.permute(0, 2, 3, 1).contiguous()


def _save_temp_images(images: torch.Tensor, filename_prefix: str = "AITKVotePreview") -> list[dict[str, str]]:
    temp_dir = folder_paths.get_temp_directory()
    if len(images.shape) != 4:
        raise ValueError("Expected image tensor with 4 dimensions [B,H,W,C].")

    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix,
        temp_dir,
        images[0].shape[1],
        images[0].shape[0],
    )
    results: list[dict[str, str]] = []

    for batch_number, image in enumerate(images):
        array = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(array)
        filename_with_batch = filename.replace("%batch_num%", str(batch_number))
        output_name = f"{filename_with_batch}_{counter:05}_.png"
        pil_image.save(os.path.join(full_output_folder, output_name), compress_level=1)
        results.append(
            {
                "filename": output_name,
                "subfolder": subfolder,
                "type": "temp",
            }
        )
        counter += 1

    return results


def _deep_copy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value) if value is not None else default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _set_nested_process_value(target: dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in str(path or "").split(".") if part]
    if not parts:
        return
    if any("[" in part or "]" in part for part in parts):
        return
    cursor: Any = target
    for part in parts[:-1]:
        next_cursor = cursor.get(part)
        if not isinstance(next_cursor, dict):
            next_cursor = {}
            cursor[part] = next_cursor
        cursor = next_cursor
    cursor[parts[-1]] = value


def _default_model_arch_name() -> str:
    preset_names = [name for name in _model_preset_names() if name != "custom"]
    if "sd15" in preset_names:
        return "sd15"
    return preset_names[0] if preset_names else "sd15"


def _apply_preset_defaults_to_process(process: dict[str, Any], preset_name: str) -> None:
    presets = _load_aitk_ui_model_presets()
    preset = presets.get(preset_name, {})
    defaults = preset.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    for path, pair in defaults.items():
        if not isinstance(path, str) or not path:
            continue
        if path.startswith("datasets[") or path.startswith("sample.") or path.startswith("slider."):
            continue
        selected_value = pair[0] if isinstance(pair, list) and len(pair) > 0 else pair
        _set_nested_process_value(process, path, selected_value)

    model = process.setdefault("model", {})
    if isinstance(model, dict):
        model["arch"] = preset_name
        model.setdefault("name_or_path", preset.get("name_or_path", ""))
        model.setdefault("qtype", preset.get("qtype", "qfloat8"))
        model.setdefault("qtype_te", preset.get("qtype_te", "qfloat8"))
        model.setdefault("model_kwargs", {})
        model.setdefault("low_vram", False)


def _default_session_ui_state() -> dict[str, Any]:
    default_arch = _default_model_arch_name()
    state: dict[str, Any] = {
        "job_config": {
            "job": "extension",
            "config": {
                "name": "aitk-session-1",
                "process": [
                    {
                        "type": "diffusion_trainer",
                        "network": {
                            "type": "lora",
                            "linear": 32,
                            "linear_alpha": 32,
                            "conv": 16,
                            "conv_alpha": 16,
                            "lokr_full_rank": True,
                            "lokr_factor": -1,
                            "dropout": 0.0,
                            "network_kwargs": {
                                "ignore_if_contains": [],
                            },
                        },
                        "save": {
                            "dtype": "bf16",
                            "save_every": 25,
                            "max_step_saves_to_keep": 4,
                            "save_format": "diffusers",
                            "push_to_hub": False,
                        },
                        "train": {
                            "optimizer": "adamw",
                            "lr": 1e-4,
                            "optimizer_params": {
                                "weight_decay": 1e-4,
                            },
                            "switch_boundary_every": 1,
                        },
                        "model": {
                            "arch": default_arch,
                            "name_or_path": "",
                            "extras_name_or_path": "",
                            "quantize": False,
                            "qtype": "qfloat8",
                            "quantize_te": False,
                            "qtype_te": "qfloat8",
                            "low_vram": False,
                            "layer_offloading": False,
                            "layer_offloading_transformer_percent": 1.0,
                            "layer_offloading_text_encoder_percent": 1.0,
                            "assistant_lora_path": "",
                            "accuracy_recovery_adapter": "",
                            "model_kwargs": {},
                        },
                    }
                ],
            },
            "meta": {
                "name": "[name]",
                "version": "1.0",
            },
        },
        "runtime": {
            "device": "cuda",
            "dtype": "fp16",
            "checkpoint_root": str(_THIRD_PARTY_AITK_ROOT / "output" / "aitk_flow_grpo"),
            "resume": True,
        },
        "grpo": {
            "clip_range": 1e-4,
            "adv_clip_max": 5.0,
            "beta": 0.0,
            "noise_level": 0.7,
            "sde_type": "sde",
            "timestep_fraction": 1.0,
        },
    }
    process = state["job_config"]["config"]["process"][0]
    _apply_preset_defaults_to_process(process, default_arch)
    return state


def _session_ui_schema() -> dict[str, Any]:
    presets = _load_aitk_ui_model_presets()
    model_archs: list[dict[str, Any]] = []
    for name, preset in presets.items():
        if name == "custom":
            continue
        model_archs.append(
            {
                "name": name,
                "label": preset.get("label", name),
                "group": preset.get("group", "image"),
                "defaults": preset.get("defaults", {}),
                "disableSections": preset.get("disableSections", []),
                "additionalSections": preset.get("additionalSections", []),
                "accuracyRecoveryAdapters": preset.get("accuracyRecoveryAdapters", {}),
            }
        )

    model_archs.sort(key=lambda item: str(item.get("label", "")).lower())
    grouped: dict[str, list[dict[str, str]]] = {}
    for item in model_archs:
        group = str(item.get("group", "image"))
        grouped.setdefault(group, []).append({"value": item["name"], "label": item["label"]})

    grouped_model_options = [
        {"label": group, "options": options}
        for group, options in sorted(grouped.items(), key=lambda pair: pair[0].lower())
    ]

    return {
        "default_state": _default_session_ui_state(),
        "default_model_arch": _default_model_arch_name(),
        "model_archs": model_archs,
        "grouped_model_options": grouped_model_options,
        "quantization_options": [
            {"value": "", "label": "- NONE -"},
            {"value": "qfloat8", "label": "float8 (default)"},
            {"value": "uint7", "label": "7 bit"},
            {"value": "uint6", "label": "6 bit"},
            {"value": "uint5", "label": "5 bit"},
            {"value": "uint4", "label": "4 bit"},
            {"value": "uint3", "label": "3 bit"},
            {"value": "uint2", "label": "2 bit"},
        ],
        "default_qtype": "qfloat8",
        "network_type_options": [
            {"value": "lora", "label": "LoRA"},
            {"value": "lokr", "label": "LoKr"},
        ],
        "dtype_options": [
            {"value": "bf16", "label": "BF16"},
            {"value": "fp16", "label": "FP16"},
            {"value": "fp32", "label": "FP32"},
        ],
        "optimizer_options": [
            {"value": "adamw", "label": "AdamW"},
        ],
        "sde_type_options": [
            {"value": "sde", "label": "SDE"},
            {"value": "cps", "label": "CPS"},
        ],
    }


def _build_session_config_from_ui_state(session_id: str, ui_state: dict[str, Any]) -> SessionConfig:
    job_config = ui_state.get("job_config", {}) if isinstance(ui_state, dict) else {}
    runtime = ui_state.get("runtime", {}) if isinstance(ui_state, dict) else {}
    grpo_state = ui_state.get("grpo", {}) if isinstance(ui_state, dict) else {}

    process_list = (((job_config.get("config") or {}).get("process")) or []) if isinstance(job_config, dict) else []
    process = process_list[0] if process_list else {}
    if not isinstance(process, dict):
        process = {}

    model = process.get("model", {})
    if not isinstance(model, dict):
        model = {}
    network = process.get("network", {})
    if not isinstance(network, dict):
        network = {}
    save = process.get("save", {})
    if not isinstance(save, dict):
        save = {}
    train = process.get("train", {})
    if not isinstance(train, dict):
        train = {}
    optimizer_params = train.get("optimizer_params", {})
    if not isinstance(optimizer_params, dict):
        optimizer_params = {}
    model_kwargs = model.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        model_kwargs = {}

    model_arch = str(model.get("arch", _default_model_arch_name()) or _default_model_arch_name()).strip()
    model_name = str(model.get("name_or_path", "") or "").strip()
    if not model_name:
        preset = _load_aitk_ui_model_presets().get(model_arch, {})
        model_name = str(preset.get("name_or_path", "") or "").strip()
    if not model_name:
        raise ValueError("Model name is required for the selected model architecture.")

    model_config_overrides: dict[str, Any] = {
        "arch": model_arch,
        "name_or_path": model_name,
        "quantize": _coerce_bool(model.get("quantize"), False),
        "quantize_te": _coerce_bool(model.get("quantize_te"), False),
        "qtype": str(model.get("qtype", "qfloat8") or "qfloat8"),
        "qtype_te": str(model.get("qtype_te", "qfloat8") or "qfloat8"),
        "low_vram": _coerce_bool(model.get("low_vram"), False),
        "layer_offloading": _coerce_bool(model.get("layer_offloading"), False),
    }
    extras_name_or_path = str(model.get("extras_name_or_path", "") or "").strip()
    if extras_name_or_path:
        model_config_overrides["extras_name_or_path"] = extras_name_or_path
    assistant_lora_path = str(model.get("assistant_lora_path", "") or "").strip()
    if assistant_lora_path:
        model_config_overrides["assistant_lora_path"] = assistant_lora_path
    accuracy_recovery_adapter = str(model.get("accuracy_recovery_adapter", "") or "").strip()
    if accuracy_recovery_adapter:
        model_config_overrides["accuracy_recovery_adapter"] = accuracy_recovery_adapter
    if model_config_overrides["layer_offloading"]:
        model_config_overrides["layer_offloading_transformer_percent"] = _coerce_float(
            model.get("layer_offloading_transformer_percent"),
            1.0,
        )
        model_config_overrides["layer_offloading_text_encoder_percent"] = _coerce_float(
            model.get("layer_offloading_text_encoder_percent"),
            1.0,
        )

    return SessionConfig(
        session_id=session_id,
        model_arch=model_arch,
        model_name=model_name,
        model_extras_name_or_path=extras_name_or_path or None,
        model_kwargs=_deep_copy_jsonable(model_kwargs),
        model_paths={},
        model_config_overrides=model_config_overrides,
        device=str(runtime.get("device", "cuda") or "cuda"),
        dtype=str(runtime.get("dtype", save.get("dtype", "fp16")) or "fp16"),  # type: ignore[arg-type]
        seed=0,
        checkpoint_root=str(
            runtime.get("checkpoint_root", str(_THIRD_PARTY_AITK_ROOT / "output" / "aitk_flow_grpo"))
            or str(_THIRD_PARTY_AITK_ROOT / "output" / "aitk_flow_grpo")
        ),
        checkpoint_interval_steps=_coerce_int(save.get("save_every"), 25),
        checkpoint_dtype=str(save.get("dtype", "fp16") or "fp16"),  # type: ignore[arg-type]
        max_checkpoints=_coerce_int(save.get("max_step_saves_to_keep"), 4),
        resume=_coerce_bool(runtime.get("resume"), True),
        lora=LoRAConfigSpec(
            enabled=True,
            network_type=str(network.get("type", "lora") or "lora"),  # type: ignore[arg-type]
            rank=_coerce_int(network.get("linear"), 32),
            alpha=_coerce_int(network.get("linear_alpha", network.get("linear")), 32),
            conv_rank=(
                _coerce_int(network.get("conv"), 0)
                if network.get("conv") is not None
                else None
            ),
            conv_alpha=(
                _coerce_float(network.get("conv_alpha"), 0.0)
                if network.get("conv_alpha") is not None
                else None
            ),
            dropout=_coerce_float(network.get("dropout"), 0.0),
            lokr_full_rank=_coerce_bool(network.get("lokr_full_rank"), True),
            lokr_factor=_coerce_int(network.get("lokr_factor"), -1),
            network_kwargs=_deep_copy_jsonable(network.get("network_kwargs", {})),
            lora_path=None,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            learning_rate=_coerce_float(train.get("lr"), 1e-4),
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_weight_decay=_coerce_float(optimizer_params.get("weight_decay"), 1e-4),
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
        ),
        grpo=GRPOConfig(
            clip_range=_coerce_float(grpo_state.get("clip_range"), 1e-4),
            adv_clip_max=_coerce_float(grpo_state.get("adv_clip_max"), 5.0),
            beta=_coerce_float(grpo_state.get("beta"), 0.0),
            noise_level=_coerce_float(grpo_state.get("noise_level"), 0.7),
            sde_type=str(grpo_state.get("sde_type", "sde") or "sde"),  # type: ignore[arg-type]
            timestep_fraction=_coerce_float(grpo_state.get("timestep_fraction"), 1.0),
        ),
    )


async def aitk_rlhf_vote_route(request):
    _raise_if_unavailable()
    body = await request.json()
    node_id = str(body.get("node_id", "")).strip()
    vote = str(body.get("vote", "")).strip().lower()

    if not node_id:
        return web.json_response({"error": "Missing node_id."}, status=400)
    if vote not in {"upvote", "downvote", "skip", "manual_checkpoint"}:
        return web.json_response({"error": f"Unsupported vote '{vote}'."}, status=400)

    state = _get_node_state(node_id)
    if state is None:
        return web.json_response(
            {"error": "No active vote state for this node. Generate a candidate first."},
            status=404,
        )

    session_id = state.get("session_id")
    candidate_id = state.get("candidate_id")
    if not session_id:
        return web.json_response({"error": "Missing session_id for node."}, status=500)

    _append_session_log(
        session_id,
        event="vote_request",
        message=f"Vote action requested: {vote}",
        node_id=node_id,
        payload={"candidate_id": candidate_id},
    )

    manager = get_online_flow_grpo_manager()
    try:
        session = manager.get_session(session_id)
        if vote == "manual_checkpoint":
            checkpoint_state = session.save_checkpoint()
            _append_session_log(
                session_id,
                event="manual_checkpoint",
                message=f"Manual checkpoint saved at step {checkpoint_state.get('step_count', 'unknown')}.",
                node_id=node_id,
                payload={"candidate_id": candidate_id},
            )
            return web.json_response(
                {
                    "ok": True,
                    "action": vote,
                    "session_id": session_id,
                    "candidate_id": candidate_id,
                    "checkpoint": checkpoint_state,
                }
            )

        if not candidate_id:
            _append_session_log(
                session_id,
                event="vote_error",
                level="error",
                message=f"Vote '{vote}' rejected: no candidate is staged for node {node_id}.",
                node_id=node_id,
            )
            return web.json_response(
                {"error": "No candidate is registered for this node."},
                status=400,
            )

        result = session.vote_step(
            candidate_id=candidate_id,
            vote=vote,
            consume_candidate=True,
        )
        _set_node_state(node_id, session_id=session_id, candidate_id=None)
        _append_session_log(
            session_id,
            event="vote_applied",
            message=(
                f"Vote '{vote}' applied to candidate {candidate_id}; "
                f"step={result.get('step_count', 'unknown')}"
            ),
            node_id=node_id,
            payload={
                "candidate_id": candidate_id,
                "vote": vote,
                "step_count": result.get("step_count"),
                "trained": result.get("trained"),
            },
        )
        return web.json_response({"ok": True, **result})
    except SessionError as error:
        _append_session_log(
            session_id,
            event="vote_error",
            level="error",
            message=f"Vote failed: {error}",
            node_id=node_id,
            payload={"candidate_id": candidate_id, "vote": vote},
        )
        return web.json_response({"error": str(error)}, status=400)
    except Exception as error:  # pragma: no cover
        _append_session_log(
            session_id,
            event="vote_error",
            level="error",
            message=f"Unexpected vote error: {error}",
            node_id=node_id,
            payload={"candidate_id": candidate_id, "vote": vote},
        )
        return web.json_response({"error": f"Unexpected error: {error}"}, status=500)


async def aitk_rlhf_logs_route(request):
    node_id = str(request.rel_url.query.get("node_id", "")).strip()
    session_id = str(request.rel_url.query.get("session_id", "")).strip()
    limit_raw = request.rel_url.query.get("limit", "100")

    try:
        limit = max(1, min(int(limit_raw), _MAX_SESSION_LOG_ENTRIES))
    except ValueError:
        limit = 100

    node_state = None
    if node_id:
        node_state = _get_node_state(node_id)
        if node_state is not None and not session_id:
            session_id = str(node_state.get("session_id", "")).strip()

    if not session_id:
        return web.json_response(
            {
                "error": (
                    "Missing session_id. Provide session_id directly or pass node_id "
                    "for a node that has been bound to a session."
                )
            },
            status=400,
        )

    logs = _get_session_logs(session_id, limit=limit)
    summary, session_error = _get_safe_session_summary(session_id)

    response: dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "node_id": node_id or None,
        "node_state": node_state,
        "entries": logs,
        "entry_count": len(logs),
        "session_summary": summary,
        "session_active": summary is not None and session_error is None,
    }
    if session_error is not None:
        response["session_error"] = session_error

    return web.json_response(response)


async def aitk_rlhf_ui_schema_route(request):
    return web.json_response({"ok": True, "schema": _session_ui_schema()})


if _PROMPT_SERVER_INSTANCE is not None:
    _PROMPT_SERVER_INSTANCE.routes.post("/aitk_rlhf/vote")(aitk_rlhf_vote_route)
    _PROMPT_SERVER_INSTANCE.routes.get("/aitk_rlhf/logs")(aitk_rlhf_logs_route)
    _PROMPT_SERVER_INSTANCE.routes.get("/aitk_rlhf/ui_schema")(aitk_rlhf_ui_schema_route)


class AITKRLHFSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": "aitk-session-1"}),
                "config_json": (
                    "STRING",
                    {
                        "default": json.dumps(_default_session_ui_state(), ensure_ascii=True),
                        "multiline": True,
                    },
                ),
                "force_reset": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AITK_SESSION", "STRING")
    RETURN_NAMES = ("session", "status")
    FUNCTION = "start_session"
    CATEGORY = "AITK/RLHF"

    def start_session(
        self,
        session_id: str,
        config_json: str,
        force_reset: bool,
    ):
        _raise_if_unavailable()
        manager = get_online_flow_grpo_manager()
        normalized_session_id = session_id.strip()

        try:
            ui_state = json.loads(config_json) if str(config_json or "").strip() else _default_session_ui_state()
        except Exception as error:
            raise ValueError(f"Invalid session UI state JSON: {error}") from error

        if not normalized_session_id:
            job_name = (
                (((ui_state.get("job_config") or {}).get("config") or {}).get("name"))
                if isinstance(ui_state, dict)
                else None
            )
            normalized_session_id = str(job_name or "aitk-session-1").strip()

        config = _build_session_config_from_ui_state(normalized_session_id, ui_state)
        try:
            session = manager.create_or_get_session(config, force_reset=bool(force_reset))
            summary = session.summary()
            summary_arch = summary.get("model_arch", summary.get("model_family", "unknown"))
            status = (
                f"Session '{summary['session_id']}' ready | "
                f"arch={summary_arch} | "
                f"step_count={summary['step_count']} | "
                f"cached_candidates={summary['cached_candidates']}"
            )
            _append_session_log(
                normalized_session_id,
                event="session_ready",
                message=status,
                payload={
                    "force_reset": bool(force_reset),
                    "resume": bool(config.resume),
                    "model_arch": config.model_arch,
                    "model_name": config.model_name,
                },
            )
            return (normalized_session_id, status)
        except Exception as error:
            _append_session_log(
                normalized_session_id,
                event="session_error",
                level="error",
                message=f"Session creation failed: {error}",
                payload={"force_reset": bool(force_reset), "resume": bool(config.resume)},
            )
            raise


class AITKGenerateCandidate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": ("AITK_SESSION",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "neg_prompt": ("STRING", {"multiline": True, "default": ""}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 250}),
                "sampler": (["flow_grpo_sde"], {"default": "flow_grpo_sde"}),
                "scheduler": (
                    ["flow_match_euler_discrete"],
                    {"default": "flow_match_euler_discrete"},
                ),
            },
            "optional": {
                "reference_images": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "AITK_METADATA", "STRING")
    RETURN_NAMES = ("image", "generation_metadata", "status")
    FUNCTION = "generate_candidate"
    CATEGORY = "AITK/RLHF"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Candidate generation is stateful and must not be cache-reused.
        return float("nan")

    def generate_candidate(
        self,
        session: str,
        seed: int,
        prompt: str,
        neg_prompt: str,
        cfg: float,
        steps: int,
        sampler: str,
        scheduler: str,
        reference_images: torch.Tensor | None = None,
        unique_id: str | None = None,
    ):
        _raise_if_unavailable()
        node_id = str(unique_id) if unique_id is not None else None
        try:
            manager = get_online_flow_grpo_manager()
            live_session = manager.get_session(session)

            images_bchw, metadata = live_session.generate_candidate(
                prompt=prompt,
                negative_prompt=neg_prompt,
                seed=int(seed),
                cfg=float(cfg),
                steps=int(steps),
                sampler=sampler,
                scheduler=scheduler,
                reference_images=_image_bhwc_to_bchw(reference_images).to(live_session.device)
                if reference_images is not None
                else None,
            )

            images_bhwc = _image_bchw_to_bhwc(images_bchw).float().clamp(0.0, 1.0)
            metadata_json = json.dumps(metadata, ensure_ascii=True)
            status = (
                f"Generated candidate {metadata['candidate_id']} | "
                f"session={metadata['session_id']} | "
                f"steps={metadata['steps']}"
            )
            if node_id is not None:
                _set_node_state(
                    node_id,
                    session_id=session,
                    candidate_id=metadata["candidate_id"],
                )
            _append_session_log(
                session,
                event="candidate_generated",
                message=status,
                node_id=node_id,
                payload={
                    "candidate_id": metadata["candidate_id"],
                    "seed": int(seed),
                    "steps": int(steps),
                    "cfg": float(cfg),
                },
            )
            return (images_bhwc, metadata_json, status)
        except Exception as error:
            _append_session_log(
                session,
                event="candidate_error",
                level="error",
                message=f"Candidate generation failed: {error}",
                node_id=node_id,
                payload={"seed": int(seed), "steps": int(steps), "cfg": float(cfg)},
            )
            raise


class AITKVote:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": ("AITK_SESSION",),
                "image": ("IMAGE",),
                "generation_metadata": ("AITK_METADATA", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AITK_SESSION", "STRING")
    RETURN_NAMES = ("session", "status")
    FUNCTION = "register_vote_candidate"
    OUTPUT_NODE = True
    CATEGORY = "AITK/RLHF"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Vote staging is stateful and must not be cache-reused.
        return float("nan")

    def register_vote_candidate(
        self,
        session: str,
        image: torch.Tensor,
        generation_metadata: str,
        unique_id: str | None = None,
    ):
        _raise_if_unavailable()
        node_id = str(unique_id) if unique_id is not None else "unknown-node"
        try:
            metadata = (
                generation_metadata
                if isinstance(generation_metadata, dict)
                else json.loads(generation_metadata)
            )
            candidate_id = metadata.get("candidate_id")
            session_id = metadata.get("session_id", session)

            _set_node_state(node_id, session_id=session_id, candidate_id=candidate_id)
            _append_session_log(
                session_id,
                event="candidate_staged",
                message=f"Candidate {candidate_id} staged for voting on node {node_id}.",
                node_id=node_id,
                payload={
                    "candidate_id": candidate_id,
                    "batch_size": int(image.shape[0]) if hasattr(image, "shape") else None,
                },
            )

            previews = _save_temp_images(image, filename_prefix=f"AITKVote_{node_id}")
            status = (
                f"Candidate ready: {candidate_id}. "
                "Use node buttons for Upvote, Downvote, Skip, or Manual Checkpoint."
            )
            return {
                "ui": {
                    "images": previews,
                    "text": [status],
                },
                "result": (session, status),
            }
        except Exception as error:
            _append_session_log(
                session,
                event="candidate_stage_error",
                level="error",
                message=f"Failed to stage candidate for vote: {error}",
                node_id=node_id,
            )
            raise


class AITKLog:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": ("AITK_SESSION",),
                "tail_entries": ("INT", {"default": 60, "min": 1, "max": 1000}),
                "poll_interval_ms": ("INT", {"default": 1500, "min": 250, "max": 60000}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AITK_SESSION", "STRING")
    RETURN_NAMES = ("session", "status")
    FUNCTION = "bind_log_session"
    OUTPUT_NODE = True
    HAS_INTERMEDIATE_OUTPUT = True
    CATEGORY = "AITK/RLHF"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Keep this node executing so binding and status stay in sync with live state.
        return float("nan")

    def bind_log_session(
        self,
        session: str,
        tail_entries: int,
        poll_interval_ms: int,
        unique_id: str | None = None,
    ):
        node_id = str(unique_id) if unique_id is not None else "unknown-node"
        existing_state = _get_node_state(node_id) or {}
        candidate_id = existing_state.get("candidate_id")
        _set_node_state(
            node_id,
            session_id=session,
            candidate_id=candidate_id,
            tail_entries=int(tail_entries),
            poll_interval_ms=int(poll_interval_ms),
        )

        summary, session_error = _get_safe_session_summary(session)
        initial_logs = _get_session_logs(session, limit=max(1, min(int(tail_entries), 10)))

        if summary is not None:
            status = (
                f"AITK Log bound to '{session}' | "
                f"step_count={summary.get('step_count', 'unknown')} | "
                f"cached_candidates={summary.get('cached_candidates', 'unknown')}"
            )
        elif session_error is not None:
            status = f"AITK Log bound to '{session}' | session inactive: {session_error}"
        else:
            status = f"AITK Log bound to '{session}'"

        _append_session_log(
            session,
            event="log_bound",
            message=f"AITK Log node bound on {node_id}.",
            node_id=node_id,
            payload={
                "tail_entries": int(tail_entries),
                "poll_interval_ms": int(poll_interval_ms),
            },
        )

        preview_lines = [
            f"[{entry['timestamp']}] {entry['level'].upper()} {entry['message']}"
            for entry in initial_logs
        ]
        if not preview_lines:
            preview_lines = ["No log entries yet. Generate and vote to populate live logs."]

        return {
            "ui": {
                "text": [status, "\n".join(preview_lines)],
            },
            "result": (session, status),
        }


NODE_CLASS_MAPPINGS = {
    "AITKRLHFSession": AITKRLHFSession,
    "AITKGenerateCandidate": AITKGenerateCandidate,
    "AITKVote": AITKVote,
    "AITKLog": AITKLog,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AITKRLHFSession": "AITK RLHF Session",
    "AITKGenerateCandidate": "AITK Generate Candidate",
    "AITKVote": "AITK Vote",
    "AITKLog": "AITK Log",
}
