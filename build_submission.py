"""Validator and Dual-Slot Packager for Kaggle Gemma 4 Developer Agent (ADR-0010).

Builds and strictly validates both Final Submission slots for the $10,000 Small Model Track:
- Slot 1 (`submission_pure_adk.zip` / `submission.zip`): Zero-Adapter Pure ADK Bundle with
  `eval_config.yaml`, `thinking_budget: 1280` inside `max_output_tokens: 12288`, CLI-invoked
  `AST Skeletal Subgraph` (`localize_and_test.py --mode ast-skeleton`), 2-strike `safe-patch`
  fallback, and Konwinski-style `Selective Patch Skipping` (`R_max = 2`).
- Slot 2 (`submission_adapter_e4b.zip`, built when `adapters_e4b/main_lora/` is present):
  Adapter-Mounted ADK Bundle declaring `adapter: main_lora` in `agent.yaml` and bundling
  genuine pretrained `google/gemma-4-E4B-it` PEFT LoRA weights verified for exact 42-layer
  `language_model` shapes, zero vision/audio tower leakage, finite weights, and non-zero learned delta.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import zipfile
from typing import Any

import yaml

TARGET_SMALL_MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
GLOBAL_LAYER_INDICES: frozenset[int] = frozenset({5, 11, 17, 23, 29, 35, 41})
MAX_ARCHIVE_BYTES = 3 * 1024 * 1024 * 1024  # 3 GiB competition ceiling

VALID_HARNESS_TOOLS: frozenset[str] = frozenset({
    "run_command",
    "submit_patch",
    "get_status",
    "read_file",
    "edit_file",
    "write_file",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
})

ALLOWED_AGENT_KEYS_PURE: frozenset[str] = frozenset({
    "agent_class",
    "name",
    "model",
    "description",
    "instruction",
    "generate_content_config",
    "tools",
    "sub_agents",
    "skills",
})

ALLOWED_AGENT_KEYS_ADAPTER: frozenset[str] = ALLOWED_AGENT_KEYS_PURE | {"adapter"}

ALLOWED_GENERATE_CONTENT_KEYS: frozenset[str] = frozenset({
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
    "thinking_config",
})

REQUIRED_RELATIVE_FILES: tuple[str, ...] = (
    "agent.yaml",
    "eval_config.yaml",
    "configs/sampling.yaml",
    "prompts/system.md",
    "prompts/analyzer.md",
    "prompts/verifier.md",
    "sub_agents/code_analyzer.yaml",
    "sub_agents/patch_verifier.yaml",
    "skills/repo_navigation/SKILL.md",
    "skills/repo_navigation/scripts/localize_and_test.py",
    "skills/repo_navigation/resources/swe_playbook.md",
)


def load_yaml_with_includes(yaml_path: pathlib.Path, bundle_root: pathlib.Path) -> Any:
    """Load a YAML file while resolving `!include` relative to `yaml_path.parent` within `bundle_root`."""
    raw_text = yaml_path.read_text(encoding="utf-8")
    root_resolved = bundle_root.resolve()

    class IncludeLoader(yaml.SafeLoader):
        pass

    def _include_constructor(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> Any:
        rel_target = loader.construct_scalar(node).strip()
        if rel_target.startswith(("/", "\\")):
            raise ValueError(f"Absolute path forbidden in !include: {rel_target}")
        cand_local = (yaml_path.parent / rel_target).resolve()
        cand_root = (root_resolved / rel_target).resolve()
        if cand_local.exists():
            target_path = cand_local
        elif cand_root.exists():
            target_path = cand_root
        else:
            raise FileNotFoundError(f"!include target '{rel_target}' not found from {yaml_path}")

        try:
            target_path.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"Forbidden path traversal outside bundle_root in !include '{rel_target}' from {yaml_path}"
            ) from exc

        if target_path.suffix in (".yaml", ".yml"):
            return load_yaml_with_includes(target_path, bundle_root)
        return target_path.read_text(encoding="utf-8")

    IncludeLoader.add_constructor("!include", _include_constructor)
    return yaml.load(raw_text, Loader=IncludeLoader)


def validate_eval_config(eval_cfg_path: pathlib.Path) -> dict[str, Any]:
    """Validate root `eval_config.yaml` anti-timeout envelope for 120-task 12h wall-clock safety."""
    cfg = yaml.safe_load(eval_cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("evaluation"), dict):
        raise ValueError("eval_config.yaml must contain top-level 'evaluation' mapping")
    ev = cfg["evaluation"]
    for key in ("timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns"):
        if key not in ev:
            raise ValueError(f"eval_config.yaml missing evaluation.{key}")
    if not (30 <= int(ev["timeout_seconds"]) <= 300):
        raise ValueError(f"timeout_seconds out of safe [30, 300] range: {ev['timeout_seconds']}")
    if not (1.0 <= float(ev["max_time_minutes"]) <= 6.0):
        raise ValueError(f"max_time_minutes must be in [1.0, 6.0] for 120-task 12h safety: {ev['max_time_minutes']}")
    if not (15 <= int(ev["max_tool_calls"]) <= 60):
        raise ValueError(f"max_tool_calls out of safe [15, 60] range: {ev['max_tool_calls']}")
    return ev


def validate_skill_markdown(skill_md_path: pathlib.Path) -> dict[str, Any]:
    """Validate YAML frontmatter (`name`, `description`) and body of a SKILL.md file."""
    text = skill_md_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{skill_md_path} must start with YAML frontmatter '---'")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{skill_md_path} is missing closing YAML frontmatter '---'")
    frontmatter = yaml.safe_load(parts[1])
    if not isinstance(frontmatter, dict):
        raise ValueError(f"{skill_md_path} frontmatter did not parse as a YAML mapping")
    for req_key in ("name", "description"):
        if not frontmatter.get(req_key) or not isinstance(frontmatter[req_key], str):
            raise ValueError(f"{skill_md_path} frontmatter missing non-empty string '{req_key}'")
    body = parts[2].strip()
    if len(body) < 50:
        raise ValueError(f"{skill_md_path} markdown body is too short")
    return frontmatter


def validate_agent_node(node: dict[str, Any], label: str, allow_adapter: bool = False) -> None:
    """Validate an ADK LlmAgent configuration dictionary."""
    if not isinstance(node, dict):
        raise ValueError(f"{label} did not resolve to a dict")
    allowed = ALLOWED_AGENT_KEYS_ADAPTER if allow_adapter else ALLOWED_AGENT_KEYS_PURE
    extra_keys = set(node.keys()) - allowed
    if extra_keys:
        raise ValueError(f"{label} contains unsupported ADK LlmAgent keys: {sorted(extra_keys)}")

    if node.get("agent_class") != "LlmAgent":
        raise ValueError(f"{label} must declare agent_class: LlmAgent")

    model = node.get("model")
    if model != TARGET_SMALL_MODEL:
        raise ValueError(
            f"{label} model is '{model}', but must be '{TARGET_SMALL_MODEL}' "
            f"for the $10,000 Small Model Bonus"
        )
    instruction = node.get("instruction")
    if not isinstance(instruction, str) or len(instruction.strip()) < 40:
        raise ValueError(f"{label} has missing or empty resolved instruction prompt")

    gen_cfg = node.get("generate_content_config")
    if not isinstance(gen_cfg, dict) or "temperature" not in gen_cfg:
        raise ValueError(f"{label} has missing or invalid generate_content_config")
    extra_gen_keys = set(gen_cfg.keys()) - ALLOWED_GENERATE_CONTENT_KEYS
    if extra_gen_keys:
        raise ValueError(f"{label} generate_content_config has unsupported keys: {sorted(extra_gen_keys)}")

    max_out = int(gen_cfg.get("max_output_tokens", 0))
    think_cfg = gen_cfg.get("thinking_config")
    if not isinstance(think_cfg, dict) or "thinking_budget" not in think_cfg:
        raise ValueError(f"{label} must set thinking_config.thinking_budget to prevent vLLM <tool_call> truncation")
    think_budget = int(think_cfg["thinking_budget"])
    if max_out < 8192 or think_budget > 2048 or (max_out - think_budget) < 4096:
        raise ValueError(
            f"{label} insufficient tool-call headroom: max_output_tokens={max_out}, thinking_budget={think_budget}"
        )

    tools = node.get("tools", [])
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{label} must declare a non-empty list of tools")
    unknown_tools = set(tools) - VALID_HARNESS_TOOLS
    if unknown_tools:
        raise ValueError(f"{label} declares unsupported tools: {unknown_tools}")


def validate_cli_helper_functionality(helper_script: pathlib.Path) -> dict[str, bool]:
    """Execute live smoke tests on `localize_and_test.py` (`ast-skeleton` & `safe-patch`)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        sample_py = tmp_path / "pkg_mod.py"
        sample_py.write_text(
            "import os\n\n"
            "def unrelated_helper(x: int) -> int:\n"
            "    a = x + 1\n"
            "    b = a * 2\n"
            "    return b\n\n"
            "def caller_entry(payload: dict) -> str:\n"
            "    return format_token(payload.get('key'))\n\n"
            "def format_token(val: str | None) -> str:\n"
            "    return val.strip().lower()\n",
            encoding="utf-8",
        )
        proc_skel = subprocess.run(
            [
                sys.executable,
                str(helper_script),
                "--mode",
                "ast-skeleton",
                "--repo-root",
                str(tmp_path),
                "--target",
                "pkg_mod.py",
                "--symbol",
                "format_token",
                "--callers",
                "caller_entry",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        skel_out = json.loads(proc_skel.stdout)
        if not skel_out.get("ast_parsed") or "AST-elided" not in skel_out.get("skeleton", ""):
            raise ValueError("localize_and_test.py --mode ast-skeleton failed to elide unrelated AST functions")

        # Verify ast.parse gate rejects invalid syntax and accepts valid syntax
        bad_payload = json.dumps({
            "path": "pkg_mod.py",
            "old_string": "    return val.strip().lower()",
            "new_string": "    if val is None:\n        return (\n    return val.strip().lower()",
        })
        proc_bad = subprocess.run(
            [sys.executable, str(helper_script), "--mode", "safe-patch", "--repo-root", str(tmp_path)],
            input=bad_payload,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_bad.returncode == 0:
            raise ValueError("localize_and_test.py --mode safe-patch failed to reject SyntaxError patch")

        good_payload = json.dumps({
            "path": "pkg_mod.py",
            "old_string": "    return val.strip().lower()",
            "new_string": "    if val is None:\n        return ''\n    return val.strip().lower()",
        })
        proc_good = subprocess.run(
            [sys.executable, str(helper_script), "--mode", "safe-patch", "--repo-root", str(tmp_path)],
            input=good_payload,
            capture_output=True,
            text=True,
            check=True,
        )
        good_out = json.loads(proc_good.stdout)
        if not good_out.get("applied") or not good_out.get("syntax_verified"):
            raise ValueError("localize_and_test.py --mode safe-patch failed valid patch test")

    return {
        "ast_skeleton_pruning_verified": True,
        "safe_patch_syntax_gate_verified": True,
    }


def validate_prompt_hardening(src_root: pathlib.Path) -> dict[str, bool]:
    """Verify ADR-0005, ADR-0008 & ADR-0010 prompt hardening rules."""
    if (src_root / "sub_agents" / "prompts").exists():
        raise ValueError("Redundant dead directory submission_src/sub_agents/prompts must be deleted")

    sys_text = (src_root / "prompts" / "system.md").read_text(encoding="utf-8")
    ana_text = (src_root / "prompts" / "analyzer.md").read_text(encoding="utf-8")
    ver_text = (src_root / "prompts" / "verifier.md").read_text(encoding="utf-8")
    skill_text = (src_root / "skills" / "repo_navigation" / "SKILL.md").read_text(encoding="utf-8")

    for name, content in (
        ("prompts/system.md", sys_text),
        ("prompts/analyzer.md", ana_text),
        ("prompts/verifier.md", ver_text),
        ("skills/repo_navigation/SKILL.md", skill_text),
    ):
        if "python skills/repo_navigation/scripts/localize_and_test.py" in content:
            raise ValueError(
                f"{name} still instructs running unsafe relative 'python skills/repo_navigation/scripts/localize_and_test.py'"
            )

    for name, content in (
        ("prompts/system.md", sys_text),
        ("prompts/analyzer.md", ana_text),
        ("skills/repo_navigation/SKILL.md", skill_text),
    ):
        if 'grep -rn --include="*.py"' not in content or "head -n 40" not in content:
            raise ValueError(f"Missing bounded 'grep -rn --include=\"*.py\" ... | head -n 40' rule in {name}")
        if "ast-skeleton" not in content:
            raise ValueError(f"Missing CLI 'ast-skeleton' invocation guidance in {name}")

    for name, content in (
        ("prompts/analyzer.md", ana_text),
        ("prompts/verifier.md", ver_text),
    ):
        if 'transfer_to_agent(agent_name="gemma4_graph_swe_orchestrator")' not in content:
            raise ValueError(f"Missing mandatory ADK handoff in {name}")

    for name, content in (
        ("prompts/system.md", sys_text),
        ("prompts/verifier.md", ver_text),
        ("skills/repo_navigation/SKILL.md", skill_text),
    ):
        if "git checkout --" not in content:
            raise ValueError(f"Missing 'git checkout --' clean rollback guard in {name}")

    if "safe-patch" not in sys_text or "safe-patch" not in skill_text:
        raise ValueError("Missing 2-strike 'safe-patch' fallback rule in system.md / SKILL.md")
    if "Selective Patch Skipping" not in sys_text:
        raise ValueError("Missing Konwinski-style Selective Patch Skipping rule in system.md")
    if "never delegate to `code_analyzer` a second time" not in sys_text:
        raise ValueError("Missing anti-ping-pong rule in system.md")
    if "at most 3 tool calls" not in ana_text.lower() and "<= 3" not in ana_text:
        raise ValueError("Missing <= 3 tool calls bound in analyzer.md")
    if "pytest" not in ver_text or "git diff" not in ver_text:
        raise ValueError("Missing mandatory pytest + git diff checks in verifier.md")

    return {
        "no_dead_sub_agent_prompts_dir": True,
        "no_unsafe_relative_script_calls": True,
        "bounded_python_grep_fallback_present": True,
        "cli_ast_skeleton_wired": True,
        "two_strike_safe_patch_wired": True,
        "selective_patch_skipping_wired": True,
        "anti_ping_pong_wired": True,
        "mandatory_adk_handoff_transfer_to_agent": True,
        "git_checkout_rollback_guard_present": True,
    }


def _load_safetensors_PurePy_or_torch(weights_path: pathlib.Path) -> list[tuple[str, tuple[int, ...], bool, float]]:
    """Return list of `(name, shape, all_finite, l2_norm)` via lightweight binary `.safetensors` parsing (zero CUDA DLL overhead)."""
    import math
    import struct

    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]

    with weights_path.open("rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len).decode("utf-8"))
        data_base = 8 + header_len
        out: list[tuple[str, tuple[int, ...], bool, float]] = []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shape = tuple(int(d) for d in meta["shape"])
            dtype = meta["dtype"]
            start_off, end_off = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
            fh.seek(data_base + start_off)
            raw = fh.read(end_off - start_off)
            numel = 1
            for d in shape:
                numel *= d
            if dtype == "F32":
                if np is not None:
                    arr = np.frombuffer(raw, dtype="<f4")
                    finite = bool(np.isfinite(arr).all())
                    norm = float(np.linalg.norm(arr.astype(np.float64))) if finite else float("nan")
                else:
                    vals = struct.unpack(f"<{numel}f", raw)
                    finite = all(math.isfinite(v) for v in vals)
                    norm = math.sqrt(sum(v * v for v in vals)) if finite else float("nan")
            else:
                raise ValueError(f"Expected F32 LoRA tensor, got {dtype} on {name}")
            out.append((name, shape, finite, norm))
        return out


def validate_e4b_peft_adapter(adapter_dir: pathlib.Path) -> dict[str, Any]:
    """Validate genuine `google/gemma-4-E4B-it` PEFT LoRA checkpoint shapes, finiteness, and learned delta."""
    cfg_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json or adapter_model.safetensors in {adapter_dir}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("peft_type") != "LORA" or cfg.get("task_type") != "CAUSAL_LM":
        raise ValueError(f"Invalid PEFT config types: {cfg.get('peft_type')} / {cfg.get('task_type')}")
    r = int(cfg["r"])
    tensor_summaries = _load_safetensors_PurePy_or_torch(weights_path)
    if not tensor_summaries:
        raise ValueError("Empty adapter_model.safetensors")

    layer_pat = re.compile(
        r"^base_model\.model\.model\.language_model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.lora_(A|B)\.weight$"
    )
    seen_layers: set[int] = set()
    nonzero_b_count = 0
    total_params = 0

    for name, shape, is_finite, l2_norm in tensor_summaries:
        if any(bad in name for bad in ("vision_tower", "audio_tower", "embed_vision", "embed_audio")):
            raise ValueError(f"Forbidden multimodal tower LoRA tensor in text agent adapter: {name}")
        m = layer_pat.match(name)
        if not m:
            raise ValueError(f"Unexpected LoRA tensor key '{name}' (does not match Gemma 4 E4B language_model)")
        if not is_finite:
            raise ValueError(f"Non-finite values in LoRA tensor {name}")
        layer_idx = int(m.group(1))
        if not (0 <= layer_idx < 42):
            raise ValueError(f"Out-of-range layer index {layer_idx} in {name}")
        seen_layers.add(layer_idx)
        proj_full = m.group(2)
        ab = m.group(5)
        is_global = layer_idx in GLOBAL_LAYER_INDICES

        expected_in_out: dict[str, tuple[int, int]] = {
            "self_attn.q_proj": (2560, 4096 if is_global else 2048),
            "self_attn.k_proj": (2560, 1024 if is_global else 512),
            "self_attn.v_proj": (2560, 1024 if is_global else 512),
            "self_attn.o_proj": (4096 if is_global else 2048, 2560),
            "mlp.gate_proj": (2560, 10240),
            "mlp.up_proj": (2560, 10240),
            "mlp.down_proj": (10240, 2560),
        }
        in_f, out_f = expected_in_out[proj_full]
        exp_shape = (r, in_f) if ab == "A" else (out_f, r)
        if shape != exp_shape:
            raise ValueError(f"Shape mismatch on {name}: got {shape}, expected {exp_shape}")
        if ab == "B" and l2_norm > 1e-7:
            nonzero_b_count += 1
        numel = 1
        for d in shape:
            numel *= d
        total_params += numel

    if nonzero_b_count == 0:
        raise ValueError("All lora_B tensors are zero — adapter has zero learned update!")

    return {
        "base_model_name_or_path": cfg.get("base_model_name_or_path", ""),
        "rank": r,
        "lora_alpha": cfg.get("lora_alpha"),
        "adapted_layers_count": len(seen_layers),
        "tensor_count": len(tensor_summaries),
        "nonzero_lora_b_tensors": nonzero_b_count,
        "trainable_params": total_params,
        "safetensors_bytes": weights_path.stat().st_size,
    }


def validate_and_build_submission(
    src_dir: pathlib.Path,
    out_zip: pathlib.Path,
    validate_only: bool = False,
    adapter_dir: pathlib.Path | None = None,
    adapter_name: str = "main_lora",
) -> dict[str, Any]:
    src_root = src_dir.resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src_root}")

    if (src_root / "adapters").exists():
        raise ValueError(f"submission_src must remain adapter-free; pass external --adapter-dir for Slot 2")

    for rel in REQUIRED_RELATIVE_FILES:
        p = src_root / rel
        if not p.exists():
            raise FileNotFoundError(f"Required submission file missing: {rel}")

    eval_cfg_report = validate_eval_config(src_root / "eval_config.yaml")
    cli_report = validate_cli_helper_functionality(
        src_root / "skills" / "repo_navigation" / "scripts" / "localize_and_test.py"
    )
    hardening_report = validate_prompt_hardening(src_root)

    all_files: list[pathlib.Path] = []
    for p in sorted(src_root.rglob("*")):
        if p.is_symlink():
            raise ValueError(f"Symlinks are forbidden in submission archive: {p}")
        if p.is_file():
            rel_posix = p.relative_to(src_root).as_posix()
            if ".." in rel_posix.split("/"):
                raise ValueError(f"Path traversal segment '..' found in {rel_posix}")
            if "__pycache__" in rel_posix or rel_posix.endswith(".pyc"):
                continue
            all_files.append(p)
            if p.suffix == ".py":
                ast.parse(p.read_text(encoding="utf-8"), filename=rel_posix)

    root_cfg = load_yaml_with_includes(src_root / "agent.yaml", src_root)
    validate_agent_node(root_cfg, "root(agent.yaml)", allow_adapter=False)

    sub_agents = root_cfg.get("sub_agents", [])
    if not isinstance(sub_agents, list) or len(sub_agents) < 2:
        raise ValueError("agent.yaml must include at least 2 resolved sub_agents")
    for idx, sub_cfg in enumerate(sub_agents):
        validate_agent_node(sub_cfg, f"sub_agent[{idx}:{sub_cfg.get('name', 'unnamed')}]", allow_adapter=False)

    skills = root_cfg.get("skills", [])
    skill_reports: list[dict[str, Any]] = []
    for skill_rel in skills:
        if ".." in pathlib.PurePosixPath(skill_rel).parts:
            raise ValueError(f"Forbidden '..' in skill path: {skill_rel}")
        skill_md = src_root / skill_rel / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"Missing SKILL.md for declared skill: {skill_rel}")
        fm = validate_skill_markdown(skill_md)
        skill_reports.append({"path": skill_rel, "name": fm["name"]})

    adapter_report: dict[str, Any] | None = None
    if adapter_dir is not None:
        adapter_report = validate_e4b_peft_adapter(adapter_dir)

    if not validate_only or not out_zip.exists():
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in all_files:
                arcname = file_path.relative_to(src_root).as_posix()
                if adapter_dir is not None and arcname == "agent.yaml":
                    raw_yaml = file_path.read_text(encoding="utf-8")
                    injected = raw_yaml.replace(
                        f"model: {TARGET_SMALL_MODEL}\n",
                        f"model: {TARGET_SMALL_MODEL}\nadapter: {adapter_name}\n",
                        1,
                    )
                    zf.writestr(arcname, injected)
                else:
                    zf.write(file_path, arcname=arcname)

            if adapter_dir is not None:
                for af in ("adapter_config.json", "adapter_model.safetensors"):
                    zf.write(adapter_dir / af, arcname=f"adapters/{adapter_name}/{af}")

    zip_bytes = out_zip.stat().st_size
    if zip_bytes >= MAX_ARCHIVE_BYTES:
        raise ValueError(f"Archive {out_zip} exceeds 3 GiB limit: {zip_bytes}")

    with zipfile.ZipFile(out_zip, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            raise RuntimeError(f"Corrupted entry in zip archive: {bad_file}")
        archived_names = zf.namelist()
        if adapter_dir is not None:
            for sub_y in ("sub_agents/code_analyzer.yaml", "sub_agents/patch_verifier.yaml"):
                if "adapter:" in zf.read(sub_y).decode("utf-8"):
                    raise RuntimeError(f"Read-only subagent {sub_y} must not mount edit_file LoRA")
        if adapter_dir is None and any(name.startswith("adapters/") for name in archived_names):
            raise RuntimeError(f"Pure ADK archive {out_zip} contains forbidden adapters/ entries")
        if adapter_dir is not None:
            with tempfile.TemporaryDirectory() as verify_tmp:
                v_root = pathlib.Path(verify_tmp)
                zf.extractall(v_root)
                v_cfg = load_yaml_with_includes(v_root / "agent.yaml", v_root)
                validate_agent_node(v_cfg, "adapter_root(agent.yaml)", allow_adapter=True)
                if v_cfg.get("adapter") != adapter_name:
                    raise RuntimeError("Injected agent.yaml missing adapter field")

    return {
        "zip_path": str(out_zip.resolve()),
        "zip_size_bytes": zip_bytes,
        "slot_mode": "adapter_mounted_adk" if adapter_dir is not None else "zero_adapter_pure_adk",
        "root_agent_name": root_cfg["name"],
        "agent_class": root_cfg["agent_class"],
        "uniform_model": TARGET_SMALL_MODEL,
        "eval_config": eval_cfg_report,
        "sampling_config": root_cfg["generate_content_config"],
        "sub_agents": [sa["name"] for sa in sub_agents],
        "skills": skill_reports,
        "cli_helper_checks": cli_report,
        "prompt_hardening": hardening_report,
        "adapter_validation": adapter_report,
        "archived_files_count": len(archived_names),
        "archived_files": archived_names,
        "all_checks_passed": True,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate and build Gemma 4 Dual-Slot ADK submission archives")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--adapter-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory containing trained E4B adapter_config.json & adapter_model.safetensors for Slot 2",
    )
    args, _ = parser.parse_known_args()

    base_dir = pathlib.Path(__file__).resolve().parent
    src_dir = base_dir / "submission_src"
    default_adapter_dir = args.adapter_dir or (base_dir / "adapters_e4b" / "main_lora")

    pure_report = validate_and_build_submission(
        src_dir, base_dir / "submission_pure_adk.zip", validate_only=args.validate_only, adapter_dir=None
    )
    full_report = validate_and_build_submission(
        src_dir, base_dir / "submission.zip", validate_only=args.validate_only, adapter_dir=None
    )
    summary: dict[str, Any] = {
        "slot1_pure_adk": pure_report,
        "primary_submission": full_report,
    }
    if default_adapter_dir.exists():
        slot2_report = validate_and_build_submission(
            src_dir,
            base_dir / "submission_adapter_e4b.zip",
            validate_only=args.validate_only,
            adapter_dir=default_adapter_dir,
        )
        summary["slot2_adapter_e4b"] = slot2_report

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
