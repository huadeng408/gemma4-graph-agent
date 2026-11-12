---
name: repo_navigation
description: Graph-guided fault localization, CLI AST skeletal subgraph pruning, 2-strike safe-patch fallback, Selective Patch Skipping rollback, and targeted pytest verification workflows inside /testbed for Gemma 4 E4B developer agents.
version: 2.0.0
---

# Repository Navigation, AST Skeletal Pruning & Safe Patching Skill

Use this skill whenever resolving a repository task in the Gemma 4 Developer Agent harness (`/testbed`). Because `gemma-4-e4b-it-qat-w4a16-ct` uses 35 local sliding-window layers (`W_slide = 512` tokens) interleaved 5:1 with 7 global full-attention layers, packing the target function and its 1-hop caller contracts inside `<= 512` tokens via AST skeletal subgraph pruning maximizes direct local-attention utilization and tool-call precision.

## 1. Graph-First Context Budgeting, 1-Turn Fast-Fallback & Anti-Ping-Pong
- **Step 1 (Semantic Seed)**: Call `search_similar_code` with the most specific exception, class.method, or domain identifier.
- **Fast-Fallback Rule**: If `search_similar_code` returns empty or irrelevant symbols, **immediately fall back within 1 turn** to `run_command` executing `grep -rn --include="*.py" "<symbol>" . | head -n 40` inside `/testbed`.
- **Step 2 (1-Hop & Steiner Subgraph)**: Call `get_code_neighbors` (`CALLS`, `CALLED_BY`, `INHERITS`, `TESTED_BY`) and `get_code_subgraph` when bridging two interacting symbols.
- **Sub-Agent Budget & Handoff**: Bound `code_analyzer` to `<= 3` tool calls returning a `<= 400`-token Localization Report via `transfer_to_agent(agent_name="gemma4_graph_swe_orchestrator")`. Never re-delegate to `code_analyzer` once its report is received.

## 2. Dynamic CLI Commands from `/testbed` (`localize_and_test.py`)
Never assume `skills/` is inside `/testbed`; always discover `localize_and_test.py` dynamically:
- **AST Skeletal Subgraph Pruning (`<= 512` tokens)**:
  `HELPER=$(find / -maxdepth 5 -name "localize_and_test.py" 2>/dev/null | head -n 1); [ -n "$HELPER" ] && python "$HELPER" --mode ast-skeleton --repo-root /testbed --target <rel_path.py> --symbol <TargetSymbol> --callers <Caller1,Caller2>`
- **Exact Indentation Span Reader (before `edit_file`)**:
  `HELPER=$(find / -maxdepth 5 -name "localize_and_test.py" 2>/dev/null | head -n 1); [ -n "$HELPER" ] && python "$HELPER" --mode read-span --repo-root /testbed --target <rel_path.py> --start-line <S> --end-line <E>`
- **2-Strike `ast.parse`-Gated Safe Patch Fallback (if `edit_file` fails twice)**:
  Pipe `{"path": "<rel_path.py>", "old_string": "...", "new_string": "..."}` to `python "$HELPER" --mode safe-patch --repo-root /testbed` and inspect `git diff`.
- **Bounded Python-Only Grep & Test Discovery**:
  `grep -rn --include="*.py" "<symbol>" . | head -n 40`
  `find . -maxdepth 4 -name "*test*.py" | grep -i "<module_stem>" | head -n 20`
- **Targeted Reproduction & 1-Hop Caller Verification**:
  `pytest <path/to/test_file.py> -k "<test_keyword>" -q`
- **Clean Rollback & Selective Patch Skipping (`R_max = 2`)**:
  Run `git checkout -- <file>` immediately on any `SyntaxError`; if 2 repair retries fail to clear regressions, run `git checkout -- .` so zero broken patches are submitted.
- **Pre-Submission AST & Diff Hygiene Audit**:
  `git status --short && git diff && python -c "import ast, pathlib, subprocess; files=[p for p in subprocess.check_output(['git','diff','--name-only'],text=True).splitlines() if p.endswith('.py') and pathlib.Path(p).exists()]; [ast.parse(pathlib.Path(f).read_text()) for f in files]; print('SYNTAX_OK', files)"`

## 3. Reference Playbook
Consult `resources/swe_playbook.md` for canonical minimal-diff patterns (boundary guards, None-propagation, off-by-one indexing, mutable defaults, and caller/callee return-contract preservation).
