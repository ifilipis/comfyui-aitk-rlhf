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


def _ensure_aitk_path() -> None:
    ai_toolkit_root = Path(
        os.environ.get("AITK_ROOT", "/content/ai-toolkit-fork")
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
        "/content/ai-toolkit-fork/ui/src/app/jobs/new/options.ts",
    )
).resolve()
_MODEL_PRESET_LOCK = threading.Lock()
_MODEL_PRESET_CACHE: dict[str, dict[str, Any]] | None = None
_TRISTATE_BOOL = ["auto", "true", "false"]


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


def _parse_js_literal(value: str) -> Any:
    text = str(value or "").strip()
    if text == "":
        return None
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


def _extract_model_defaults(defaults_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    prefix = "'config.process[0].model."
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

        first_item_literal = _first_js_list_item(list_literal)
        out[key] = _parse_js_literal(first_item_literal)
        cursor = next_idx

    return out


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

            array_start = content.find("[", marker_idx)
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

                preset: dict[str, Any] = {"label": label, "arch": name}

                defaults_idx = block.find("defaults:")
                if defaults_idx >= 0:
                    defaults_start = block.find("{", defaults_idx)
                    if defaults_start >= 0:
                        defaults_obj, _ = _extract_balanced(block, defaults_start, "{", "}")
                        defaults = _extract_model_defaults(defaults_obj)
                        for key, value in defaults.items():
                            preset[key] = value

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


@PromptServer.instance.routes.post("/aitk_rlhf/vote")
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


@PromptServer.instance.routes.get("/aitk_rlhf/logs")
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


class AITKRLHFSession:
    @classmethod
    def INPUT_TYPES(cls):
        model_preset_options = _model_preset_names()
        if not model_preset_options:
            model_preset_options = ["custom"]
        default_model_preset = "sd15" if "sd15" in model_preset_options else model_preset_options[0]

        return {
            "required": {
                "session_id": ("STRING", {"default": "aitk-session-1"}),
                "model_preset": (model_preset_options, {"default": default_model_preset}),
                "use_preset_defaults": ("BOOLEAN", {"default": True}),
                "model_arch": ("STRING", {"default": "sd15"}),
                "model_name": (
                    "STRING",
                    {"default": "stable-diffusion-v1-5/stable-diffusion-v1-5"},
                ),
                "model_extras_name_or_path": ("STRING", {"default": ""}),
                "model_quantize": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_quantize_te": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_qtype": ("STRING", {"default": ""}),
                "model_qtype_te": ("STRING", {"default": ""}),
                "model_low_vram": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_layer_offloading": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_layer_offloading_transformer_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "model_layer_offloading_text_encoder_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "model_attn_masking": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_split_model_over_gpus": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_assistant_lora_path": ("STRING", {"default": ""}),
                "model_accuracy_recovery_adapter": ("STRING", {"default": ""}),
                "model_match_target_res": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_train_high_noise": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_train_low_noise": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_do_random_inpainting": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_random_blur_mask": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_random_dialate_mask": (_TRISTATE_BOOL, {"default": "auto"}),
                "model_invert_inpaint_mask_chance": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "model_inpaint_dropout": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "model_control_dropout": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "model_inpaint_random_chance": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "device": ("STRING", {"default": "cuda"}),
                "dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "checkpoint_root": (
                    "STRING",
                    {"default": "/content/ai-toolkit-fork/output/aitk_flow_grpo"},
                ),
                "checkpoint_interval_steps": (
                    "INT",
                    {"default": 25, "min": 1, "max": 1000000},
                ),
                "resume": ("BOOLEAN", {"default": True}),
                "force_reset": ("BOOLEAN", {"default": False}),
                "lora_rank": ("INT", {"default": 32, "min": 1, "max": 1024}),
                "lora_alpha": ("INT", {"default": 64, "min": 1, "max": 4096}),
                "lora_dropout": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "lora_path": ("STRING", {"default": ""}),
                "learning_rate": (
                    "FLOAT",
                    {"default": 1e-4, "min": 1e-7, "max": 1.0, "step": 1e-6},
                ),
                "adam_beta1": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 0.9999, "step": 0.0001},
                ),
                "adam_beta2": (
                    "FLOAT",
                    {"default": 0.999, "min": 0.0, "max": 0.999999, "step": 0.000001},
                ),
                "adam_weight_decay": (
                    "FLOAT",
                    {"default": 1e-4, "min": 0.0, "max": 1.0, "step": 1e-6},
                ),
                "adam_epsilon": (
                    "FLOAT",
                    {"default": 1e-8, "min": 1e-12, "max": 1e-2, "step": 1e-10},
                ),
                "max_grad_norm": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01},
                ),
                "grpo_clip_range": (
                    "FLOAT",
                    {"default": 1e-4, "min": 1e-8, "max": 1.0, "step": 1e-6},
                ),
                "grpo_adv_clip_max": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.1, "max": 1000.0, "step": 0.1},
                ),
                "grpo_beta": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "grpo_noise_level": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "grpo_sde_type": (["sde", "cps"], {"default": "sde"}),
                "grpo_timestep_fraction": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("AITK_SESSION", "STRING")
    RETURN_NAMES = ("session", "status")
    FUNCTION = "start_session"
    CATEGORY = "AITK/RLHF"

    def start_session(
        self,
        session_id: str,
        model_preset: str,
        use_preset_defaults: bool,
        model_arch: str,
        model_name: str,
        model_extras_name_or_path: str,
        model_quantize: str,
        model_quantize_te: str,
        model_qtype: str,
        model_qtype_te: str,
        model_low_vram: str,
        model_layer_offloading: str,
        model_layer_offloading_transformer_percent: float,
        model_layer_offloading_text_encoder_percent: float,
        model_attn_masking: str,
        model_split_model_over_gpus: str,
        model_assistant_lora_path: str,
        model_accuracy_recovery_adapter: str,
        model_match_target_res: str,
        model_train_high_noise: str,
        model_train_low_noise: str,
        model_do_random_inpainting: str,
        model_random_blur_mask: str,
        model_random_dialate_mask: str,
        model_invert_inpaint_mask_chance: float,
        model_inpaint_dropout: float,
        model_control_dropout: float,
        model_inpaint_random_chance: float,
        device: str,
        dtype: str,
        seed: int,
        checkpoint_root: str,
        checkpoint_interval_steps: int,
        resume: bool,
        force_reset: bool,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_path: str,
        learning_rate: float,
        adam_beta1: float,
        adam_beta2: float,
        adam_weight_decay: float,
        adam_epsilon: float,
        max_grad_norm: float,
        grpo_clip_range: float,
        grpo_adv_clip_max: float,
        grpo_beta: float,
        grpo_noise_level: float,
        grpo_sde_type: str,
        grpo_timestep_fraction: float,
    ):
        _raise_if_unavailable()
        manager = get_online_flow_grpo_manager()
        normalized_session_id = session_id.strip()
        model_config_overrides: dict[str, Any] = {}
        model_kwargs: dict[str, Any] = {}
        model_paths: dict[str, str] = {}

        effective_model_arch = model_arch.strip()
        effective_model_name = model_name.strip()
        effective_model_extras = model_extras_name_or_path.strip() or None

        presets = _load_aitk_ui_model_presets()
        preset = presets.get(model_preset, {})
        if bool(use_preset_defaults) and model_preset != "custom":
            preset_arch = preset.get("arch")
            if isinstance(preset_arch, str) and preset_arch.strip():
                effective_model_arch = preset_arch.strip()
            preset_name = preset.get("name_or_path")
            if isinstance(preset_name, str) and preset_name.strip():
                effective_model_name = preset_name.strip()
            preset_extras = preset.get("extras_name_or_path")
            if isinstance(preset_extras, str) and preset_extras.strip():
                effective_model_extras = preset_extras.strip()
            preset_model_kwargs = preset.get("model_kwargs")
            if isinstance(preset_model_kwargs, dict):
                model_kwargs.update(preset_model_kwargs)
            preset_model_paths = preset.get("model_paths")
            if isinstance(preset_model_paths, dict):
                model_paths.update({str(k): str(v) for k, v in preset_model_paths.items()})

            for key in (
                "quantize",
                "quantize_te",
                "qtype",
                "qtype_te",
                "low_vram",
                "layer_offloading",
                "attn_masking",
                "split_model_over_gpus",
                "assistant_lora_path",
                "accuracy_recovery_adapter",
            ):
                if key in preset and preset[key] is not None:
                    model_config_overrides[key] = preset[key]

        if not effective_model_name:
            raise ValueError("Model name is required. Set model_name or select a preset with defaults enabled.")

        model_config_overrides["arch"] = effective_model_arch
        model_config_overrides["name_or_path"] = effective_model_name
        if effective_model_extras:
            model_config_overrides["extras_name_or_path"] = effective_model_extras

        def apply_tristate(target: dict[str, Any], key: str, mode: str) -> None:
            parsed = _tristate_to_optional_bool(mode)
            if parsed is not None:
                target[key] = parsed

        apply_tristate(model_config_overrides, "quantize", model_quantize)
        apply_tristate(model_config_overrides, "quantize_te", model_quantize_te)
        apply_tristate(model_config_overrides, "low_vram", model_low_vram)
        apply_tristate(model_config_overrides, "layer_offloading", model_layer_offloading)
        apply_tristate(model_config_overrides, "attn_masking", model_attn_masking)
        apply_tristate(model_config_overrides, "split_model_over_gpus", model_split_model_over_gpus)

        if model_qtype.strip():
            model_config_overrides["qtype"] = model_qtype.strip()
        if model_qtype_te.strip():
            model_config_overrides["qtype_te"] = model_qtype_te.strip()
        if model_assistant_lora_path.strip():
            model_config_overrides["assistant_lora_path"] = model_assistant_lora_path.strip()
        if model_accuracy_recovery_adapter.strip():
            model_config_overrides["accuracy_recovery_adapter"] = model_accuracy_recovery_adapter.strip()

        if bool(model_config_overrides.get("layer_offloading", False)):
            model_config_overrides["layer_offloading_transformer_percent"] = float(
                model_layer_offloading_transformer_percent
            )
            model_config_overrides["layer_offloading_text_encoder_percent"] = float(
                model_layer_offloading_text_encoder_percent
            )

        apply_tristate(model_kwargs, "match_target_res", model_match_target_res)
        apply_tristate(model_kwargs, "train_high_noise", model_train_high_noise)
        apply_tristate(model_kwargs, "train_low_noise", model_train_low_noise)
        apply_tristate(model_kwargs, "do_random_inpainting", model_do_random_inpainting)
        apply_tristate(model_kwargs, "random_blur_mask", model_random_blur_mask)
        apply_tristate(model_kwargs, "random_dialate_mask", model_random_dialate_mask)

        if float(model_invert_inpaint_mask_chance) >= 0.0:
            model_kwargs["invert_inpaint_mask_chance"] = float(model_invert_inpaint_mask_chance)
        if float(model_inpaint_dropout) >= 0.0:
            model_kwargs["inpaint_dropout"] = float(model_inpaint_dropout)
        if float(model_control_dropout) >= 0.0:
            model_kwargs["control_dropout"] = float(model_control_dropout)
        if float(model_inpaint_random_chance) >= 0.0:
            model_kwargs["inpaint_random_chance"] = float(model_inpaint_random_chance)

        config = SessionConfig(
            session_id=normalized_session_id,
            model_arch=effective_model_arch,
            model_name=effective_model_name,
            model_extras_name_or_path=effective_model_extras or None,
            model_kwargs=model_kwargs,
            model_paths=model_paths,
            model_config_overrides=model_config_overrides,
            device=device.strip(),
            dtype=dtype,  # type: ignore[arg-type]
            seed=int(seed),
            checkpoint_root=checkpoint_root.strip(),
            checkpoint_interval_steps=int(checkpoint_interval_steps),
            resume=bool(resume),
            lora=LoRAConfigSpec(
                enabled=True,
                rank=int(lora_rank),
                alpha=int(lora_alpha),
                dropout=float(lora_dropout),
                lora_path=lora_path.strip() or None,
            ),
            optimizer=OptimizerConfig(
                optimizer="adamw",
                learning_rate=float(learning_rate),
                adam_beta1=float(adam_beta1),
                adam_beta2=float(adam_beta2),
                adam_weight_decay=float(adam_weight_decay),
                adam_epsilon=float(adam_epsilon),
                max_grad_norm=float(max_grad_norm),
            ),
            grpo=GRPOConfig(
                clip_range=float(grpo_clip_range),
                adv_clip_max=float(grpo_adv_clip_max),
                beta=float(grpo_beta),
                noise_level=float(grpo_noise_level),
                sde_type=grpo_sde_type,  # type: ignore[arg-type]
                timestep_fraction=float(grpo_timestep_fraction),
            ),
        )
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
                    "resume": bool(resume),
                    "model_preset": model_preset,
                    "model_arch": effective_model_arch,
                    "model_name": effective_model_name,
                },
            )
            return (session_id, status)
        except Exception as error:
            _append_session_log(
                normalized_session_id,
                event="session_error",
                level="error",
                message=f"Session creation failed: {error}",
                payload={"force_reset": bool(force_reset), "resume": bool(resume)},
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
