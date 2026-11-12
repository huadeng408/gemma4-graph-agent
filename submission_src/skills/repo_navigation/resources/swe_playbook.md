# SWE Patch Synthesis & Graph Reasoning Playbook (`gemma-4-e4b-it-qat-w4a16-ct`)

## 1. Why Graph-First Retrieval Outperforms Naive File Reading on 4B Models
Small-footprint models (`gemma-4-e4b-it-qat-w4a16-ct`) use 35 local sliding-window layers (`W_slide = 512`) and 7 global attention layers. They experience severe attention dilution when >16k tokens of unrelated repository files enter the prompt window. Always restrict context using the 3-step funnel (with bounded 1-turn `grep -rn` fallback if graph search returns empty):
1. `search_similar_code(query)` -> Top 3–5 semantic anchors (or immediate 1-turn fallback to `grep -rn --include="*.py" "<symbol>" . | head -n 40` in `/testbed`).
2. `get_code_neighbors(symbol_id)` -> Immediate callers, callees, base classes, and unit tests.
3. `get_code_subgraph(symbol_ids)` -> Exact structural bridge between entry point and failure site.

## 2. High-Frequency Defect Archetypes & Surgical Fixes
1. **None / Empty Container Boundary**:
   - Distinguish `if val is None:` from `if not val:` when `0`, `0.0`, `False`, or `""` are valid domain inputs.
2. **Silent State Mutation Across Callers**:
   - Check `get_code_neighbors` callers before mutating an input dict/list in-place; copy or return a new structure if callers reuse the reference.
3. **Subclasses & Method Resolution Order (MRO)**:
   - When modifying a base class method or signature, inspect `INHERITS` edges in `get_code_neighbors` to ensure overridden methods accept compatible keyword/positional arguments.
4. **Clean Rollback on Broken Edits**:
   - If an edit causes a `SyntaxError` or breaks existing tests, run `git checkout -- <file>` immediately before applying a clean replacement.

## 3. Pre-Submission Gate (`patch_verifier`) & ADK Return Handoff
Never call `submit_patch` until:
- The failing reproduction test passes via `run_command` (`pytest <target_test_file> -q`).
- Neighboring 1-hop caller unit tests identified via `get_code_neighbors` pass.
- `git status --short && git diff` and `python -c "import ast, pathlib, subprocess; files=[p for p in subprocess.check_output(['git','diff','--name-only'],text=True).splitlines() if p.endswith('.py') and pathlib.Path(p).exists()]; [ast.parse(pathlib.Path(f).read_text()) for f in files]; print('SYNTAX_OK', files)"` confirm valid AST syntax and zero unintended untracked files inside `/testbed`.
- Both `code_analyzer` and `patch_verifier` must immediately call `transfer_to_agent(agent_name="gemma4_graph_swe_orchestrator")` after emitting their structured output.
