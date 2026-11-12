#!/usr/bin/env python3
"""Deterministic repository navigation, AST skeletal subgraph pruning, and safe patch helper for Gemma 4 E4B Agent."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any


def extract_symbols_from_text(text: str) -> dict[str, list[str]]:
    """Extract candidate file paths, dotted Class.method symbols, camelCase/snake_case symbols, and exceptions."""
    files = sorted(set(re.findall(r"[\w./\\-]+\.py\b", text)))
    exceptions = sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning)\b", text)))
    dotted = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", text))
    backticked = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", text))
    camel_or_snake = set(
        tok
        for tok in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,60}\b", text)
        if "_" in tok or (any(c.isupper() for c in tok[1:]) and any(c.islower() for c in tok))
    )
    all_idents = sorted(dotted | backticked | camel_or_snake)
    return {
        "candidate_files": files[:15],
        "exceptions": exceptions[:10],
        "identifiers": all_idents[:40],
    }


def _collect_call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                names.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                names.add(child.func.attr)
    return names


def build_ast_skeletal_subgraph(
    file_path: pathlib.Path,
    target_symbol: str,
    caller_symbols: list[str] | None = None,
    max_target_lines: int = 90,
) -> dict[str, Any]:
    """Fold unrelated AST sibling definitions while retaining imports, target body, and 1-hop caller contracts.

    Packs the active fault neighborhood and caller invariants within Gemma 4's W_slide=512 local window.
    """
    source = file_path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    total_lines = len(lines)
    raw_approx_tokens = max(1, int(round(len(re.findall(r"\w+|[^\w\s]", source)) * 1.12)))

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        snippet = "\n".join(f"{idx + 1:4d}: {ln}" for idx, ln in enumerate(lines[:max_target_lines]))
        return {
            "file": file_path.as_posix(),
            "target_symbol": target_symbol,
            "ast_parsed": False,
            "raw_lines": total_lines,
            "skeletal_lines": min(total_lines, max_target_lines),
            "raw_approx_tokens": raw_approx_tokens,
            "skeletal_approx_tokens": max(1, int(round(len(re.findall(r"\w+|[^\w\s]", snippet)) * 1.12))),
            "skeleton": snippet,
        }

    target_leaf = target_symbol.split(".")[-1].strip()
    caller_set = {c.split(".")[-1].strip() for c in (caller_symbols or []) if c.strip()}
    out_blocks: list[str] = []
    matched_spans: list[tuple[int, int]] = []

    def _sig_header(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
        start = node.lineno - 1
        end = min(len(lines), (node.body[0].lineno - 1) if node.body else node.lineno)
        header_lines = [f"{ln_no + 1:4d}: {lines[ln_no]}" for ln_no in range(start, max(start + 1, end))]
        end_line = getattr(node, "end_lineno", node.lineno)
        header_lines.append(f"      ...  # [AST-elided L{end + 1}-L{end_line}]")
        return "\n".join(header_lines)

    def _full_span(node: ast.AST, cap_lines: int) -> str:
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        matched_spans.append((start + 1, end))
        sliced_end = min(end, start + cap_lines)
        rendered = [f"{ln_no + 1:4d}: {lines[ln_no]}" for ln_no in range(start, sliced_end)]
        if sliced_end < end:
            rendered.append(f"      ...  # [truncated L{sliced_end + 1}-L{end}]")
        return "\n".join(rendered)

    for top in tree.body:
        if isinstance(top, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            s_ln = top.lineno - 1
            e_ln = min(getattr(top, "end_lineno", top.lineno), s_ln + 4)
            out_blocks.append("\n".join(f"{i + 1:4d}: {lines[i]}" for i in range(s_ln, e_ln)))
        elif isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = _collect_call_names(top)
            if top.name == target_leaf or target_leaf in top.name:
                out_blocks.append(_full_span(top, max_target_lines))
            elif top.name in caller_set or (target_leaf and target_leaf in calls):
                out_blocks.append(_full_span(top, 28))
            else:
                out_blocks.append(_sig_header(top))
        elif isinstance(top, ast.ClassDef):
            cls_match = top.name == target_leaf or top.name in target_symbol.split(".")
            method_blocks: list[str] = []
            any_inner_keep = False
            for item in top.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    calls = _collect_call_names(item)
                    qual = f"{top.name}.{item.name}"
                    if item.name == target_leaf or qual == target_symbol or (cls_match and item.name == "__init__"):
                        method_blocks.append(_full_span(item, max_target_lines))
                        any_inner_keep = True
                    elif item.name in caller_set or (target_leaf and target_leaf in calls):
                        method_blocks.append(_full_span(item, 24))
                        any_inner_keep = True
                    else:
                        method_blocks.append(_sig_header(item))
                elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                    s_ln = item.lineno - 1
                    method_blocks.append(f"{s_ln + 1:4d}: {lines[s_ln]}")
            if cls_match or any_inner_keep:
                cls_hdr_end = (top.body[0].lineno - 1) if top.body else top.lineno
                cls_hdr = "\n".join(f"{i + 1:4d}: {lines[i]}" for i in range(top.lineno - 1, max(top.lineno, cls_hdr_end)))
                out_blocks.append(cls_hdr + ("\n" + "\n".join(method_blocks) if method_blocks else ""))
            else:
                out_blocks.append(_sig_header(top))

    skeleton_text = "\n".join(out_blocks)
    skeletal_tokens = max(1, int(round(len(re.findall(r"\w+|[^\w\s]", skeleton_text)) * 1.12)))
    return {
        "file": file_path.as_posix(),
        "target_symbol": target_symbol,
        "ast_parsed": True,
        "matched_spans": matched_spans,
        "raw_lines": total_lines,
        "skeletal_lines": len(skeleton_text.splitlines()),
        "raw_approx_tokens": raw_approx_tokens,
        "skeletal_approx_tokens": skeletal_tokens,
        "fits_w_slide_512": skeletal_tokens <= 512,
        "skeleton": skeleton_text,
    }


def read_exact_span(file_path: pathlib.Path, start_line: int, end_line: int) -> dict[str, Any]:
    """Return both line-numbered view and exact raw substring for lossless `edit_file` matching."""
    source = file_path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines(keepends=True)
    s_idx = max(0, start_line - 1)
    e_idx = min(len(lines), max(s_idx + 1, end_line))
    raw_exact = "".join(lines[s_idx:e_idx])
    numbered = "".join(f"{i + 1:4d}| {lines[i]}" for i in range(s_idx, e_idx))
    return {
        "file": file_path.as_posix(),
        "start_line": s_idx + 1,
        "end_line": e_idx,
        "numbered_view": numbered,
        "exact_old_string": raw_exact,
    }


def apply_safe_patch(file_path: pathlib.Path, old_string: str, new_string: str) -> dict[str, Any]:
    """Apply an exact string edit gated by `ast.parse()` before writing to disk."""
    original = file_path.read_text(encoding="utf-8")
    occurrences = original.count(old_string)
    if occurrences == 0:
        # Attempt normalized trailing-space match per line if exact match failed on trailing whitespace
        norm_orig_lines = [ln.rstrip() for ln in original.splitlines()]
        norm_old_lines = [ln.rstrip() for ln in old_string.splitlines()]
        n_old = len(norm_old_lines)
        matches = [
            i for i in range(len(norm_orig_lines) - n_old + 1)
            if norm_orig_lines[i : i + n_old] == norm_old_lines
        ]
        if len(matches) != 1:
            return {
                "applied": False,
                "reason": f"old_string not found uniquely (exact={occurrences}, line_normalized={len(matches)})",
            }
        orig_lines_keep = original.splitlines(keepends=True)
        m_idx = matches[0]
        trailing_nl = "\n" if not new_string.endswith("\n") and orig_lines_keep[m_idx + n_old - 1].endswith("\n") else ""
        candidate = "".join(orig_lines_keep[:m_idx]) + new_string + trailing_nl + "".join(orig_lines_keep[m_idx + n_old :])
    elif occurrences > 1:
        return {
            "applied": False,
            "reason": f"old_string matched {occurrences} times; include more surrounding context lines",
        }
    else:
        candidate = original.replace(old_string, new_string, 1)

    if file_path.suffix == ".py":
        try:
            ast.parse(candidate, filename=str(file_path))
        except SyntaxError as exc:
            return {
                "applied": False,
                "reason": f"Rejected by ast.parse gate: SyntaxError at line {exc.lineno}: {exc.msg}",
            }

    file_path.write_text(candidate, encoding="utf-8")
    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(),
            candidate.splitlines(),
            fromfile=f"a/{file_path.as_posix()}",
            tofile=f"b/{file_path.as_posix()}",
            lineterm="",
        )
    )
    return {
        "applied": True,
        "syntax_verified": True,
        "diff_preview": "\n".join(diff_lines[:80]),
    }


def find_related_tests(repo_root: pathlib.Path, target_file: str) -> list[str]:
    """Locate the closest pytest/unittest files corresponding to `target_file`."""
    t_path = pathlib.Path(target_file)
    stem = t_path.stem
    candidates: list[tuple[int, str]] = []
    for p in repo_root.rglob("*.py"):
        rel = p.relative_to(repo_root).as_posix()
        if "/.git/" in f"/{rel}" or "__pycache__" in rel:
            continue
        name = p.name
        if not (name.startswith("test_") or name.endswith("_test.py") or "tests/" in rel):
            continue
        score = 0
        if name in (f"test_{stem}.py", f"{stem}_test.py"):
            score += 100
        elif stem in name:
            score += 50
        common = len(set(t_path.parts) & set(p.relative_to(repo_root).parts))
        score += common * 5
        if score > 0:
            candidates.append((score, rel))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [rel for _, rel in candidates[:8]]


def check_diff_hygiene(repo_root: pathlib.Path) -> dict[str, Any]:
    """Validate git diff for syntax validity of touched Python files and absence of conflict markers."""
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    touched: list[str] = []
    syntax_errors: list[str] = []
    untracked_files: list[str] = []
    conflict_markers: list[str] = []

    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        rel_path = line[3:].strip()
        if status == "??":
            untracked_files.append(rel_path)
        else:
            touched.append(rel_path)
        full_p = repo_root / rel_path
        if full_p.exists() and full_p.is_file():
            content = full_p.read_text(encoding="utf-8", errors="replace")
            if any(marker in content for marker in ("<<<<<<< ", "=======\n", ">>>>>>> ")):
                conflict_markers.append(rel_path)
            if full_p.suffix == ".py":
                try:
                    ast.parse(content, filename=rel_path)
                except SyntaxError as exc:
                    syntax_errors.append(f"{rel_path}:{exc.lineno}: {exc.msg}")

    return {
        "clean_syntax": len(syntax_errors) == 0 and len(conflict_markers) == 0,
        "modified_files": touched,
        "untracked_files": untracked_files,
        "syntax_errors": syntax_errors,
        "conflict_markers": conflict_markers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo navigation & AST skeletal helper for Gemma 4 E4B")
    parser.add_argument(
        "--mode",
        choices=("extract-symbols", "ast-skeleton", "read-span", "safe-patch", "find-tests", "check-diff"),
        required=True,
    )
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--symbol", type=str, default="")
    parser.add_argument("--callers", type=str, default="")
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--end-line", type=int, default=80)
    parser.add_argument("--patch-json", type=pathlib.Path, default=None)
    parser.add_argument("--text", type=str, default="")
    args = parser.parse_args()

    if args.mode == "extract-symbols":
        payload = args.text or sys.stdin.read()
        print(json.dumps(extract_symbols_from_text(payload), indent=2))
    elif args.mode == "ast-skeleton":
        caller_list = [c.strip() for c in args.callers.split(",") if c.strip()]
        res = build_ast_skeletal_subgraph(args.repo_root / args.target, args.symbol, caller_list)
        print(json.dumps(res, indent=2))
    elif args.mode == "read-span":
        res = read_exact_span(args.repo_root / args.target, args.start_line, args.end_line)
        print(json.dumps(res, indent=2))
    elif args.mode == "safe-patch":
        raw_payload = args.patch_json.read_text(encoding="utf-8") if args.patch_json else sys.stdin.read()
        spec = json.loads(raw_payload)
        res = apply_safe_patch(
            args.repo_root / spec["path"],
            spec["old_string"],
            spec["new_string"],
        )
        print(json.dumps(res, indent=2))
        if not res.get("applied"):
            sys.exit(1)
    elif args.mode == "find-tests":
        print(json.dumps({"tests": find_related_tests(args.repo_root, args.target)}, indent=2))
    elif args.mode == "check-diff":
        report = check_diff_hygiene(args.repo_root)
        print(json.dumps(report, indent=2))
        if not report["clean_syntax"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
