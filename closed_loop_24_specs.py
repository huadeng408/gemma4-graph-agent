"""Pure 24-Task 3-Tier Multi-Module Sandbox Specifications (Zero Hardcoded Candidate Patches).

Contains ONLY the buggy repository files (`bridge_code`, `caller_code`, optional `runtime_dispatch_code`),
natural-language bug report (`issue`), and ground-truth `pytest` suites (`test_target`, `test_caller`).
All candidate patches MUST be synthesized live by `google/gemma-4-E4B-it`.
"""

from __future__ import annotations

from typing import Any

CLOSED_LOOP_24_SPECS: list[dict[str, Any]] = [{'task_id': 'cl_fastapi_01',
  'repo': 'fastapi/fastapi',
  'tier': 'tier2_hop1_caller_bridge',
  'issue': "Bug in `normalize_route_path`: mounting a sub-router with an empty endpoint path `''` and "
           "prefix `'/api/'` produces a trailing slash `'/api/'` instead of `'/api'`.",
  'bridge_func': 'normalize_route_path',
  'bridge_code': 'def normalize_route_path(prefix: str, path: str) -> str:\n'
                 '    """Combine router prefix and endpoint path."""\n'
                 "    clean_prefix = '/' + prefix.strip('/') if prefix.strip('/') else ''\n"
                 "    clean_path = '/' + path.lstrip('/') if path else '/'\n"
                 '    return clean_prefix + clean_path\n',
  'caller_code': 'from pkg.bridge import normalize_route_path\n'
                 '\n'
                 'def register_subrouter(routes: list[str], prefix: str, path: str, trailing_ok: bool = '
                 'False) -> dict[str, object]:\n'
                 '    res = normalize_route_path(prefix, path, trailing_ok=trailing_ok)\n'
                 '    if isinstance(res, tuple):\n'
                 '        full_path, has_prefix = res\n'
                 '    else:\n'
                 '        full_path = res\n'
                 "        has_prefix = bool(prefix.strip('/'))\n"
                 "    if not trailing_ok and full_path != '/' and full_path.endswith('/'):\n"
                 "        raise ValueError(f'Unexpected trailing slash: {full_path}')\n"
                 "    return {'full_path': full_path, 'has_prefix': has_prefix, 'count': len(routes) + "
                 '1}\n',
  'test_target': 'from pkg.caller import register_subrouter\n'
                 '\n'
                 'def test_empty_subroute_strips_trailing_slash():\n'
                 "    res = register_subrouter(['/health'], '/api/', '')\n"
                 "    assert res['full_path'] == '/api'\n"
                 "    assert res['has_prefix'] is True\n",
  'test_caller': 'from pkg.caller import register_subrouter\n'
                 'from pkg.bridge import normalize_route_path\n'
                 '\n'
                 'def test_caller_tuple_contract_and_root_preserved():\n'
                 "    res_root = register_subrouter([], '', '')\n"
                 "    assert res_root['full_path'] == '/'\n"
                 "    assert res_root['has_prefix'] is False\n"
                 "    res = register_subrouter([], '/v1', '/users/')\n"
                 "    assert res['full_path'] == '/v1/users'\n"
                 "    assert res['has_prefix'] is True\n"
                 "    assert normalize_route_path('/v1', '/items/', trailing_ok=True) in ('/v1/items/', "
                 "('/v1/items/', True))\n"},
 {'task_id': 'cl_fastapi_02',
  'repo': 'fastapi/fastapi',
  'tier': 'tier2_hop1_caller_bridge',
  'issue': "Bug in `build_dep_cache_key`: when `scopes` is passed as `['write', 'read', 'write']` vs "
           "`['read', 'write']`, cache keys differ because `scopes` are not deduplicated and sorted.",
  'bridge_func': 'build_dep_cache_key',
  'bridge_code': 'def build_dep_cache_key(call_name: str, scopes: list[str]) -> tuple:\n'
                 '    """Construct dependency cache key tuple."""\n'
                 '    scopes.sort()\n'
                 '    return (call_name, tuple(scopes), True)\n',
  'caller_code': 'from pkg.bridge import build_dep_cache_key\n'
                 '\n'
                 'def resolve_dependency_tree(call_name: str, scopes: list[str], use_cache: bool = True) '
                 '-> dict[str, object]:\n'
                 '    before = list(scopes)\n'
                 '    key = build_dep_cache_key(call_name, scopes, use_cache=use_cache)\n'
                 '    if scopes != before:\n'
                 "        raise RuntimeError('Caller scopes list was mutated in-place!')\n"
                 '    if len(key) != 3 or not isinstance(key[1], tuple):\n'
                 "        raise TypeError('Expected 3-tuple (str, tuple, bool)')\n"
                 '    if not use_cache and key != (call_name, (), False):\n'
                 "        raise ValueError('Uncached call must return empty scope tuple and False')\n"
                 "    return {'key': key, 'scopes_preserved': scopes == before}\n",
  'test_target': 'from pkg.bridge import build_dep_cache_key\n'
                 '\n'
                 'def test_scope_order_canonicalized_and_deduped():\n'
                 "    k1 = build_dep_cache_key('get_user', ['write', 'read', 'write'])\n"
                 "    k2 = build_dep_cache_key('get_user', ['read', 'write'])\n"
                 "    assert k1 == k2 == ('get_user', ('read', 'write'), True)\n",
  'test_caller': 'from pkg.caller import resolve_dependency_tree\n'
                 '\n'
                 'def test_caller_scopes_not_mutated_and_no_cache_mode():\n'
                 "    raw = ['z_scope', 'a_scope']\n"
                 "    res = resolve_dependency_tree('auth', raw, use_cache=True)\n"
                 "    assert raw == ['z_scope', 'a_scope']\n"
                 "    assert res['scopes_preserved'] is True\n"
                 "    nocache = resolve_dependency_tree('auth', raw, use_cache=False)\n"
                 "    assert nocache['key'] == ('auth', (), False)\n"},
 {'task_id': 'cl_fastapi_03',
  'repo': 'fastapi/fastapi',
  'tier': 'tier1_single_module_contract',
  'issue': "Issue in `merge_response_headers`: merging `{'x-rate-limit': '10'}` into `{'X-Rate-Limit': "
           "'100', 'Content-Type': 'application/json'}` leaves duplicate case-insensitive keys instead of "
           'normalizing all keys to lowercase.',
  'bridge_func': 'merge_response_headers',
  'bridge_code': 'def merge_response_headers(defaults: dict[str, str], overrides: dict[str, str] | None = '
                 'None) -> dict[str, str]:\n'
                 '    """Merge override headers into defaults without mutating input."""\n'
                 '    defaults.update(overrides or {})\n'
                 '    return defaults\n',
  'caller_code': 'from pkg.bridge import merge_response_headers\n'
                 '\n'
                 'def finalize_endpoint_response(base_headers: dict[str, str], custom: dict[str, str] | '
                 'None) -> tuple[dict[str, str], dict[str, str]]:\n'
                 '    snap = dict(base_headers)\n'
                 '    out = merge_response_headers(base_headers, custom)\n'
                 '    if base_headers != snap:\n'
                 "        raise RuntimeError('base_headers was mutated in-place')\n"
                 '    return out, base_headers\n',
  'test_target': 'from pkg.bridge import merge_response_headers\n'
                 '\n'
                 'def test_case_insensitive_header_override():\n'
                 "    out = merge_response_headers({'X-Rate-Limit': '100', 'Content-Type': "
                 "'application/json'}, {'x-rate-limit': '10'})\n"
                 "    assert out == {'x-rate-limit': '10', 'content-type': 'application/json'}\n",
  'test_caller': 'from pkg.caller import finalize_endpoint_response\n'
                 '\n'
                 'def test_caller_base_headers_unmutated_and_none_overrides():\n'
                 "    base = {'X-Trace': 'abc'}\n"
                 '    out, orig = finalize_endpoint_response(base, None)\n'
                 "    assert orig == {'X-Trace': 'abc'}\n"
                 "    assert out == {'x-trace': 'abc'}\n"},
 {'task_id': 'cl_fastapi_04',
  'repo': 'fastapi/fastapi',
  'tier': 'tier1_single_module_contract',
  'issue': "Bug in `coerce_query_flag`: passing string `'0'`, `'false'`, or `'no'` evaluates to `True` "
           "because `bool('0')` is truthy.",
  'bridge_func': 'coerce_query_flag',
  'bridge_code': 'def coerce_query_flag(val: object, default: bool = False) -> tuple[bool, str]:\n'
                 '    """Coerce query flag to (bool_val, source_tag) where None uses (default, '
                 '\'default\')."""\n'
                 '    if val is None:\n'
                 "        return (default, 'default')\n"
                 "    return (bool(val), 'explicit')\n",
  'caller_code': 'from pkg.bridge import coerce_query_flag\n'
                 '\n'
                 'def parse_pagination_params(raw_flag: object, default: bool = False) -> dict[str, '
                 'object]:\n'
                 '    flag, source = coerce_query_flag(raw_flag, default=default)\n'
                 "    if source not in ('default', 'explicit'):\n"
                 "        raise ValueError('Invalid source tag')\n"
                 "    return {'include_deleted': flag, 'source': source}\n",
  'test_target': 'from pkg.caller import parse_pagination_params\n'
                 '\n'
                 'def test_string_false_and_zero_coerced_to_false():\n'
                 "    assert parse_pagination_params('0')['include_deleted'] is False\n"
                 "    assert parse_pagination_params('false')['include_deleted'] is False\n"
                 "    assert parse_pagination_params('FALSE')['include_deleted'] is False\n"
                 "    assert parse_pagination_params('no')['include_deleted'] is False\n"
                 "    assert parse_pagination_params('1')['include_deleted'] is True\n"
                 "    assert parse_pagination_params('true')['include_deleted'] is True\n",
  'test_caller': 'from pkg.caller import parse_pagination_params\n'
                 '\n'
                 'def test_none_default_and_source_tag_contract():\n'
                 '    res = parse_pagination_params(None, default=True)\n'
                 "    assert res == {'include_deleted': True, 'source': 'default'}\n"
                 '    res2 = parse_pagination_params(0, default=True)\n'
                 "    assert res2 == {'include_deleted': False, 'source': 'explicit'}\n"},
 {'task_id': 'cl_fastapi_05',
  'repo': 'fastapi/fastapi',
  'tier': 'tier3_multihop_dynamic_dispatch',
  'issue': 'Bug in `classify_status_body_policy`: HTTP status codes `204` and `304` must suppress response '
           'body (`allow_body=False`).',
  'bridge_func': 'classify_status_body_policy',
  'bridge_code': 'def classify_status_body_policy(status_code: int) -> tuple[int, bool]:\n'
                 '    """Return (status_code, allow_body)."""\n'
                 '    return (status_code, status_code != 204)\n',
  'caller_code': 'from pkg.bridge import classify_status_body_policy\n'
                 '\n'
                 'def build_route_response(status_code: int | str, payload: str) -> dict[str, object]:\n'
                 '    code, allow_body = classify_status_body_policy(status_code)\n'
                 '    if not isinstance(code, int):\n'
                 "        raise TypeError('code must be coerced to int')\n"
                 '    if (100 <= code < 200 or code in (204, 205, 304)) and allow_body:\n'
                 "        raise ValueError(f'RFC 9110 forbids body for status {code}')\n"
                 "    return {'status': code, 'body': payload if allow_body else None, 'allow_body': "
                 'allow_body}\n',
  'test_target': 'from pkg.bridge import classify_status_body_policy\n'
                 '\n'
                 'def test_204_and_304_suppressed():\n'
                 '    assert classify_status_body_policy(204) == (204, False)\n'
                 '    assert classify_status_body_policy(304) == (304, False)\n',
  'test_caller': 'from pkg.caller import build_route_response\n'
                 '\n'
                 'def test_1xx_205_and_str_status_codes_supported():\n'
                 "    for sc in (100, 101, 204, 205, 304, '204', '205'):\n"
                 "        res = build_route_response(sc, 'hello')\n"
                 "        assert res['allow_body'] is False\n"
                 "        assert res['body'] is None\n"
                 "    assert build_route_response('200', 'ok') == {'status': 200, 'body': 'ok', "
                 "'allow_body': True}\n"},
 {'task_id': 'cl_fastapi_06',
  'repo': 'fastapi/fastapi',
  'tier': 'tier3_multihop_dynamic_dispatch',
  'failure_category': 'dynamic_getattr_indirection',
  'failure_description': "2-hop `getattr(PluginHookRegistry, f'on_{phase}')` dispatch in "
                         '`pkg/runtime_dispatch.py` lacks a static `ast.Call` edge from `bridge.py`, so '
                         "static 1-hop subgraph pruning misses the registry's required `phase_token` "
                         'plan field during startup/shutdown dispatch.',
  'issue': 'Bug in `dispatch_lifespan_event`: duplicate event names in `events` list are executed multiple '
           'times instead of deduplicating while preserving registration order.',
  'bridge_func': 'dispatch_lifespan_event',
  'bridge_code': 'def dispatch_lifespan_event(phase: str, events: list[str]) -> dict[str, object]:\n'
                 '    """Normalize lifespan phase and deduplicate event hooks."""\n'
                 "    return {'phase': phase.strip(), 'events': list(events)}\n",
  'caller_code': 'from pkg.bridge import dispatch_lifespan_event\n'
                 'from pkg.runtime_dispatch import invoke_dynamic_phase_registry\n'
                 '\n'
                 'def run_app_lifespan(phase: str, events: list[str]) -> dict[str, object]:\n'
                 '    plan = dispatch_lifespan_event(phase, events)\n'
                 '    return invoke_dynamic_phase_registry(plan)\n',
  'runtime_dispatch_code': 'class PluginHookRegistry:\n'
                           '    def on_startup(self, plan: dict[str, object]) -> dict[str, object]:\n'
                           "        ev = list(plan['events'])\n"
                           '        if plan.get(\'phase_token\') != f"startup:{len(ev)}":\n'
                           "            raise RuntimeError('Missing dynamic registry phase_token required "
                           "by on_startup')\n"
                           "        return {'phase': 'startup', 'events': ev, 'verified': True}\n"
                           '\n'
                           '    def on_shutdown(self, plan: dict[str, object]) -> dict[str, object]:\n'
                           "        ev = list(plan['events'])\n"
                           '        if plan.get(\'phase_token\') != f"shutdown:{len(ev)}":\n'
                           "            raise RuntimeError('Missing dynamic registry phase_token required "
                           "by on_shutdown')\n"
                           "        return {'phase': 'shutdown', 'events': ev, 'verified': True}\n"
                           '\n'
                           'def invoke_dynamic_phase_registry(plan: dict[str, object]) -> dict[str, '
                           'object]:\n'
                           '    reg = PluginHookRegistry()\n'
                           '    handler = getattr(reg, f"on_{plan[\'phase\'].lower()}")\n'
                           '    return handler(plan)\n',
  'test_target': 'from pkg.bridge import dispatch_lifespan_event\n'
                 '\n'
                 'def test_lifespan_events_deduplicated_in_order():\n'
                 "    res = dispatch_lifespan_event('startup', ['db', 'cache', 'db', 'metrics'])\n"
                 "    assert res['events'] == ['db', 'cache', 'metrics']\n",
  'test_caller': 'from pkg.caller import run_app_lifespan\n'
                 '\n'
                 'def test_dynamic_getattr_registry_token_handshake():\n'
                 "    res = run_app_lifespan('startup', ['db', 'cache', 'db'])\n"
                 "    assert res == {'phase': 'startup', 'events': ['db', 'cache'], 'verified': True}\n"},
 {'task_id': 'cl_rich_07',
  'repo': 'Textualize/rich',
  'tier': 'tier2_hop1_caller_bridge',
  'issue': 'Bug in `slice_style_spans`: slicing a line to `[start:end]` shifts and clamps style spans `(s, '
           'e, style)`, but spans that lie completely outside `[start, end]` (where `clamped_s >= '
           'clamped_e`) are still included in the returned list.',
  'bridge_func': 'slice_style_spans',
  'bridge_code': 'def slice_style_spans(spans: list[tuple[int, int, str]], start: int, end: int) -> '
                 'list[tuple[int, int, str]]:\n'
                 '    """Shift and clamp (span_start, span_end, style) to [0, end - start]."""\n'
                 '    out: list[tuple[int, int, str]] = []\n'
                 '    width = max(0, end - start)\n'
                 '    for s, e, style in spans:\n'
                 '        ns = max(0, s - start)\n'
                 '        ne = min(width, e - start)\n'
                 '        out.append((ns, ne, style))\n'
                 '    return out\n',
  'caller_code': 'from pkg.bridge import slice_style_spans\n'
                 '\n'
                 'def render_clipped_line(text: str, spans: list[tuple[int, int, str]], start: int = 0, '
                 'end: int | None = None) -> dict[str, object]:\n'
                 '    orig = list(spans)\n'
                 '    clipped_spans = slice_style_spans(spans, start, end)\n'
                 '    if spans != orig:\n'
                 "        raise RuntimeError('Input spans mutated')\n"
                 '    if any(not st or ns >= ne for ns, ne, st in clipped_spans):\n'
                 "        raise ValueError('Empty style or zero-length span leaked into clipped_spans')\n"
                 "    return {'text': text[start:end], 'spans': clipped_spans}\n",
  'test_target': 'from pkg.bridge import slice_style_spans\n'
                 '\n'
                 'def test_out_of_range_and_zero_length_spans_dropped():\n'
                 "    spans = [(0, 3, 'bold'), (3, 8, 'red'), (8, 12, 'italic')]\n"
                 "    assert slice_style_spans(spans, 3, 8) == [(0, 5, 'red')]\n",
  'test_caller': 'from pkg.caller import render_clipped_line\n'
                 '\n'
                 'def test_end_none_and_empty_style_filtered():\n'
                 "    spans = [(1, 6, 'cyan'), (2, 5, ''), (4, 10, 'underline')]\n"
                 "    res = render_clipped_line('0123456789', spans, 2, None)\n"
                 "    assert res['spans'] == [(0, 4, 'cyan'), (2, 8, 'underline')]\n"},
 {'task_id': 'cl_rich_08',
  'repo': 'Textualize/rich',
  'tier': 'tier2_hop1_caller_bridge',
  'issue': 'Bug in `clamp_column_width`: when `min_width` exceeds `max_width`, `clamp_column_width` '
           'returns `min_width` instead of capping strictly at `max_width`.',
  'bridge_func': 'clamp_column_width',
  'bridge_code': 'def clamp_column_width(content_len: int, min_width: int | None, max_width: int | None) '
                 '-> tuple[int, bool]:\n'
                 '    """Clamp column width and return (width, was_truncated)."""\n'
                 '    w = content_len\n'
                 '    if max_width is not None:\n'
                 '        w = min(w, max_width)\n'
                 '    if min_width is not None:\n'
                 '        w = max(w, min_width)\n'
                 '    return (w, w < content_len)\n',
  'caller_code': 'from pkg.bridge import clamp_column_width\n'
                 '\n'
                 'def layout_table_row(cells: list[str], min_w: int | None, max_w: int | None, pad: int = '
                 '0) -> list[tuple[int, bool]]:\n'
                 '    return [clamp_column_width(len(c), min_w, max_w, pad=pad) for c in cells]\n',
  'test_target': 'from pkg.bridge import clamp_column_width\n'
                 '\n'
                 'def test_max_width_takes_precedence_over_min_width():\n'
                 '    assert clamp_column_width(10, 8, 5) == (5, True)\n'
                 '    assert clamp_column_width(2, 8, 5) == (5, False)\n',
  'test_caller': 'from pkg.caller import layout_table_row\n'
                 '\n'
                 'def test_caller_pad_keyword_and_bounds():\n'
                 "    assert layout_table_row(['abc'], min_w=2, max_w=10, pad=2) == [(5, False)]\n"
                 "    assert layout_table_row(['abcdefgh'], min_w=None, max_w=6, pad=2) == [(6, True)]\n"},
 {'task_id': 'cl_rich_09',
  'repo': 'Textualize/rich',
  'tier': 'tier1_single_module_contract',
  'issue': 'Bug in `escape_markup_tags`: `escape_markup_tags` escapes every `[` including already escaped '
           '`\\[` sequences, turning `\\[INFO]` into `\\\\[INFO]`.',
  'bridge_func': 'escape_markup_tags',
  'bridge_code': 'import re\n'
                 '\n'
                 'def escape_markup_tags(text: str | None) -> tuple[str, int]:\n'
                 '    r"""Escape unescaped opening brackets `[` as `\\[` and return (escaped_text, '
                 'num_newly_escaped)."""\n'
                 '    if text is None:\n'
                 "        return ('', 0)\n"
                 "    count = text.count('[')\n"
                 "    return (text.replace('[', r'\\['), count)\n",
  'caller_code': 'from pkg.bridge import escape_markup_tags\n'
                 '\n'
                 'def format_log_record(prefix: str, msg: str | None) -> dict[str, object]:\n'
                 '    escaped, n_esc = escape_markup_tags(msg)\n'
                 "    return {'formatted': f'{prefix}: {escaped}', 'escaped_count': n_esc}\n",
  'test_target': 'from pkg.bridge import escape_markup_tags\n'
                 '\n'
                 'def test_already_escaped_brackets_not_double_escaped():\n'
                 "    esc, cnt = escape_markup_tags(r'already \\[bold] and raw [red]')\n"
                 "    assert esc == r'already \\[bold] and raw \\[red]'\n"
                 '    assert cnt == 1\n',
  'test_caller': 'from pkg.caller import format_log_record\n'
                 '\n'
                 'def test_none_msg_and_leading_brackets():\n'
                 "    assert format_log_record('APP', None) == {'formatted': 'APP: ', 'escaped_count': 0}\n"
                 "    res = format_log_record('APP', '[a][b]')\n"
                 "    assert res == {'formatted': r'APP: \\[a]\\[b]', 'escaped_count': 2}\n"},
 {'task_id': 'cl_rich_10',
  'repo': 'Textualize/rich',
  'tier': 'tier1_single_module_contract',
  'issue': 'Bug in `pkg/caller.py::render_progress_summary`: when `total=0` or `completed > total`, '
           '`compute_progress_ratio` raises `ZeroDivisionError` or exceeds `100.0`.',
  'bridge_func': 'compute_progress_ratio',
  'bridge_code': 'def compute_progress_ratio(completed: float, total: float | None) -> tuple[float, '
                 'bool]:\n'
                 '    """Return (percentage_0_to_100, is_indeterminate)."""\n'
                 '    if total is None:\n'
                 '        return (0.0, True)\n'
                 '    pct = (completed / total) * 100.0\n'
                 '    return (round(pct, 2), False)\n',
  'caller_code': 'from pkg.bridge import compute_progress_ratio\n'
                 '\n'
                 'def render_progress_summary(task_name: str, completed: float, total: float | None) -> '
                 'dict[str, object]:\n'
                 '    pct, indeterminate = compute_progress_ratio(completed, total)\n'
                 '    if not (0.0 <= pct <= 100.0):\n'
                 "        raise ValueError(f'Percentage out of [0, 100] bounds: {pct}')\n"
                 "    return {'task': task_name, 'pct': pct, 'indeterminate': indeterminate}\n",
  'test_target': 'from pkg.caller import render_progress_summary\n'
                 '\n'
                 'def test_zero_total_and_overflow_clamped():\n'
                 "    r_zero = render_progress_summary('download', 0, 0)\n"
                 "    assert r_zero['pct'] == 0.0 and r_zero['indeterminate'] is False\n"
                 "    r_over = render_progress_summary('download', 150, 100)\n"
                 "    assert r_over['pct'] == 100.0\n"
                 "    r_neg = render_progress_summary('download', -5, 100)\n"
                 "    assert r_neg['pct'] == 0.0\n",
  'test_caller': 'from pkg.caller import render_progress_summary\n'
                 '\n'
                 'def test_indeterminate_none_and_fractional_progress():\n'
                 "    assert render_progress_summary('scan', 10, None) == {'task': 'scan', 'pct': 0.0, "
                 "'indeterminate': True}\n"
                 "    assert render_progress_summary('scan', 1, 4) == {'task': 'scan', 'pct': 25.0, "
                 "'indeterminate': False}\n"},
 {'task_id': 'cl_rich_11',
  'repo': 'Textualize/rich',
  'tier': 'tier3_multihop_dynamic_dispatch',
  'issue': 'Bug in `pkg/caller.py::measure_visible_cell_len`: `strip_ansi_codes` removes `\\x1b[...m` SGR '
           'codes but leaves cursor movement sequences like `\\x1b[2K` and `\\x1b[1A` in the string.',
  'bridge_func': 'strip_ansi_codes',
  'bridge_code': 'import re\n'
                 '\n'
                 'def strip_ansi_codes(text: str) -> tuple[str, int]:\n'
                 '    """Strip all CSI ANSI escape sequences `\\x1b[...<letter>` and return (clean_text, '
                 'visible_len)."""\n'
                 "    clean = re.sub(r'\\x1b\\[[0-9;]*m', '', text)\n"
                 '    return (clean, len(clean))\n',
  'caller_code': 'from pkg.bridge import strip_ansi_codes\n'
                 '\n'
                 'def measure_visible_cell_len(line: str) -> dict[str, object]:\n'
                 '    clean, vlen = strip_ansi_codes(line)\n'
                 "    return {'clean': clean, 'visible_len': vlen}\n",
  'test_target': 'from pkg.caller import measure_visible_cell_len\n'
                 '\n'
                 'def test_strips_non_sgr_csi_sequences():\n'
                 "    res = measure_visible_cell_len('\\x1b[2K\\x1b[32mReady\\x1b[0m\\x1b[1A')\n"
                 "    assert res == {'clean': 'Ready', 'visible_len': 5}\n",
  'test_caller': 'from pkg.caller import measure_visible_cell_len\n'
                 '\n'
                 'def test_plain_text_unchanged():\n'
                 "    assert measure_visible_cell_len('plain_[text]') == {'clean': 'plain_[text]', "
                 "'visible_len': 12}\n"},
 {'task_id': 'cl_rich_12',
  'repo': 'Textualize/rich',
  'tier': 'tier3_multihop_dynamic_dispatch',
  'failure_category': 'diamond_mro_super_side_effect',
  'failure_description': '2-hop diamond MRO inheritance chain (`ConsolePanel(PaddingMixin, BorderMixin, '
                         'BaseRenderable)`) in `pkg/runtime_dispatch.py` mutates a shared '
                         '`__mro_sync__` sentinel across cooperative `super().render_tokens()` calls 2 hops above '
                         '`bridge.py::coalesce_adjacent_segments`.',
  'issue': 'Bug in `coalesce_adjacent_segments`: adjacent `(text, style)` tuples with identical `style` '
           'are not merged into a single tuple.',
  'bridge_func': 'coalesce_adjacent_segments',
  'bridge_code': 'def coalesce_adjacent_segments(segments: list[tuple[str, str]]) -> list[tuple[str, '
                 'str]]:\n'
                 '    """Merge adjacent (text, style) segments sharing the same style."""\n'
                 '    return list(segments)\n',
  'caller_code': 'from pkg.bridge import coalesce_adjacent_segments\n'
                 'from pkg.runtime_dispatch import render_diamond_panel\n'
                 '\n'
                 'def build_panel_line(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:\n'
                 '    merged = coalesce_adjacent_segments(segments)\n'
                 '    return render_diamond_panel(merged)\n',
  'runtime_dispatch_code': 'class BaseRenderable:\n'
                           '    _mro_epoch: int = 0\n'
                           '    def render_tokens(self, segs: list[tuple[str, str]]) -> list[tuple[str, '
                           'str]]:\n'
                           '        return list(segs)\n'
                           '\n'
                           'class BorderMixin(BaseRenderable):\n'
                           '    def render_tokens(self, segs: list[tuple[str, str]]) -> list[tuple[str, '
                           'str]]:\n'
                           '        out = super().render_tokens(segs)\n'
                           "        if not out or out[0][1] != '__mro_sync__':\n"
                           "            raise RuntimeError('Diamond MRO BorderMixin expected __mro_sync__ "
                           "sentinel from cooperative super()')\n"
                           '        return out[1:]\n'
                           '\n'
                           'class PaddingMixin(BaseRenderable):\n'
                           '    def render_tokens(self, segs: list[tuple[str, str]]) -> list[tuple[str, '
                           'str]]:\n'
                           '        return super().render_tokens(segs)\n'
                           '\n'
                           'class ConsolePanel(PaddingMixin, BorderMixin):\n'
                           '    pass\n'
                           '\n'
                           'def render_diamond_panel(segs: list[tuple[str, str]]) -> list[tuple[str, '
                           'str]]:\n'
                           '    return ConsolePanel().render_tokens(segs)\n',
  'test_target': 'from pkg.bridge import coalesce_adjacent_segments\n'
                 '\n'
                 'def test_adjacent_identical_styles_merged():\n'
                 "    raw = [('Hello ', 'bold'), ('World', 'bold'), ('!', 'red')]\n"
                 "    assert coalesce_adjacent_segments(raw) == [('Hello World', 'bold'), ('!', 'red')]\n",
  'test_caller': 'from pkg.caller import build_panel_line\n'
                 '\n'
                 'def test_diamond_mro_panel_render():\n'
                 "    raw = [('Hello ', 'bold'), ('World', 'bold')]\n"
                 "    assert build_panel_line(raw) == [('Hello World', 'bold')]\n"},
 {'task_id': 'cl_requests_13',
  'repo': 'psf/requests',
  'tier': 'tier2_hop1_caller_bridge',
  'issue': 'Security regression in `pkg/caller.py::prepare_redirect_request`: redirecting from '
           '`https://example.com:8443/v1` to `https://example.com/v1` (different port) or downgrading '
           '`https://example.com/a` to `http://example.com/b` fails to strip the `Authorization` header.',
  'bridge_func': 'should_strip_auth_on_redirect',
  'bridge_code': 'from urllib.parse import urlparse\n'
                 '\n'
                 'def should_strip_auth_on_redirect(old_url: str, new_url: str) -> bool:\n'
                 '    """Return True if Authorization header must be removed across redirect."""\n'
                 '    p_old = urlparse(old_url)\n'
                 '    p_new = urlparse(new_url)\n'
                 '    return p_old.hostname != p_new.hostname\n',
  'caller_code': 'from pkg.bridge import should_strip_auth_on_redirect\n'
                 '\n'
                 'def prepare_redirect_request(old_url: str, new_url: str, headers: dict[str, str]) -> '
                 'dict[str, str]:\n'
                 '    out = dict(headers)\n'
                 '    if should_strip_auth_on_redirect(old_url, new_url):\n'
                 "        out.pop('Authorization', None)\n"
                 '    return out\n',
  'test_target': 'from pkg.caller import prepare_redirect_request\n'
                 '\n'
                 'def test_port_change_and_https_to_http_downgrade_strip_auth():\n'
                 "    h = {'Authorization': 'Bearer secret', 'Accept': '*/*'}\n"
                 "    r1 = prepare_redirect_request('https://api.com:8443/a', 'https://api.com/b', h)\n"
                 "    assert 'Authorization' not in r1\n"
                 "    r2 = prepare_redirect_request('https://api.com/a', 'http://api.com/b', h)\n"
                 "    assert 'Authorization' not in r2\n",
  'test_caller': 'from pkg.caller import prepare_redirect_request\n'
                 '\n'
                 'def test_same_origin_and_http_to_https_upgrade_on_standard_ports_preserve_auth():\n'
                 "    h = {'Authorization': 'Bearer secret', 'Accept': '*/*'}\n"
                 "    r_same = prepare_redirect_request('https://api.com:443/a', 'https://api.com/b', h)\n"
                 "    assert r_same.get('Authorization') == 'Bearer secret'\n"
                 "    r_upg = prepare_redirect_request('http://api.com:80/a', 'https://api.com:443/b', h)\n"
                 "    assert r_upg.get('Authorization') == 'Bearer secret'\n"},
 {'task_id': 'cl_requests_14',
  'repo': 'psf/requests',
  'tier': 'tier1_single_module_contract',
  'issue': "Bug in `pkg/caller.py::select_request_cookies`: cookie with domain `'example.com'` matches "
           "`'badexample.com'` because `request_host.endswith(cookie_domain)` does not check for a dot "
           'boundary.',
  'bridge_func': 'cookie_domain_matches',
  'bridge_code': 'def cookie_domain_matches(cookie_domain: str, request_host: str) -> bool:\n'
                 '    """Return True if request_host matches cookie_domain per RFC 6265 domain-match."""\n'
                 "    cd = cookie_domain.lstrip('.').lower()\n"
                 '    rh = request_host.lower()\n'
                 '    return rh == cd or rh.endswith(cd)\n',
  'caller_code': 'from pkg.bridge import cookie_domain_matches\n'
                 '\n'
                 'def select_request_cookies(jar: list[tuple[str, str, str]], host: str) -> dict[str, '
                 'str]:\n'
                 '    return {name: val for dom, name, val in jar if cookie_domain_matches(dom, host)}\n',
  'test_target': 'from pkg.caller import select_request_cookies\n'
                 '\n'
                 'def test_superstring_host_does_not_match_cookie_domain():\n'
                 "    jar = [('.example.com', 'sid', '123'), ('example.com', 'theme', 'dark')]\n"
                 "    assert select_request_cookies(jar, 'badexample.com') == {}\n"
                 "    assert select_request_cookies(jar, 'sub.example.com') == {'sid': '123', 'theme': "
                 "'dark'}\n",
  'test_caller': 'from pkg.caller import select_request_cookies\n'
                 '\n'
                 'def test_exact_domain_and_case_insensitivity():\n'
                 "    jar = [('.Example.COM', 'sid', '123')]\n"
                 "    assert select_request_cookies(jar, 'EXAMPLE.COM') == {'sid': '123'}\n"},
 {'task_id': 'cl_requests_15',
  'repo': 'psf/requests',
  'tier': 'tier3_multihop_dynamic_dispatch',
  'issue': 'Bug in `pkg/caller.py::dispatch_response_hooks`: when a hook callback returns `None` '
           '(observing the response without replacing it), `apply_hook_chain` overwrites the response '
           'object with `None`.',
  'bridge_func': 'apply_hook_chain',
  'bridge_code': 'from typing import Callable, Any\n'
                 '\n'
                 'def apply_hook_chain(hooks: list[Callable[[Any], Any]] | None, payload: Any) -> '
                 'tuple[Any, int]:\n'
                 '    """Run hook callbacks sequentially; keep current payload if callback returns '
                 'None."""\n'
                 '    if not hooks:\n'
                 '        return (payload, 0)\n'
                 '    cur = payload\n'
                 '    for fn in hooks:\n'
                 '        cur = fn(cur)\n'
                 '    return (cur, len(hooks))\n',
  'caller_code': 'from pkg.bridge import apply_hook_chain\n'
                 '\n'
                 'def dispatch_response_hooks(hooks: list | None, response: dict[str, object]) -> '
                 'dict[str, object]:\n'
                 '    final_resp, ran_count = apply_hook_chain(hooks, response)\n'
                 '    if not isinstance(final_resp, dict):\n'
                 "        raise TypeError('Response became non-dict after hooks')\n"
                 "    return {'response': final_resp, 'hooks_ran': ran_count}\n",
  'test_target': 'from pkg.caller import dispatch_response_hooks\n'
                 '\n'
                 'def test_none_returning_hook_preserves_response():\n'
                 '    seen = []\n'
                 '    def log_hook(r):\n'
                 "        seen.append(r['status'])\n"
                 '        return None\n'
                 '    def mutate_hook(r):\n'
                 "        return {**r, 'hooked': True}\n"
                 "    res = dispatch_response_hooks([log_hook, mutate_hook, log_hook], {'status': 200})\n"
                 "    assert res['response'] == {'status': 200, 'hooked': True}\n"
                 "    assert res['hooks_ran'] == 3\n",
  'test_caller': 'from pkg.caller import dispatch_response_hooks\n'
                 '\n'
                 'def test_empty_or_none_hooks():\n'
                 "    assert dispatch_response_hooks(None, {'status': 204}) == {'response': {'status': "
                 "204}, 'hooks_ran': 0}\n"},
 {'task_id': 'cl_requests_16',
  'repo': 'psf/requests',
  'tier': 'tier1_single_module_contract',
  'issue': 'Bug in `pkg/caller.py::build_prepared_query`: query parameters whose value is `None` (e.g. '
           "`{'a': '1', 'b': None, 'c': ['x', None, 'y']}`) are serialized as `'b=None'` instead of being "
           'omitted.',
  'bridge_func': 'encode_query_pairs',
  'bridge_code': 'def encode_query_pairs(params: dict[str, object] | list[tuple[str, object]]) -> '
                 'list[tuple[str, str]]:\n'
                 '    """Flatten params into (key, str_val) pairs, omitting entries where value is '
                 'None."""\n'
                 '    items = params.items() if isinstance(params, dict) else params\n'
                 '    pairs: list[tuple[str, str]] = []\n'
                 '    for k, v in items:\n'
                 '        if isinstance(v, (list, tuple)):\n'
                 '            for item in v:\n'
                 '                pairs.append((str(k), str(item)))\n'
                 '        else:\n'
                 '            pairs.append((str(k), str(v)))\n'
                 '    return pairs\n',
  'caller_code': 'from pkg.bridge import encode_query_pairs\n'
                 '\n'
                 'def build_prepared_query(params: dict[str, object] | list[tuple[str, object]]) -> str:\n'
                 '    pairs = encode_query_pairs(params)\n'
                 "    return '&'.join(f'{k}={v}' for k, v in pairs)\n",
  'test_target': 'from pkg.caller import build_prepared_query\n'
                 '\n'
                 'def test_none_values_omitted_while_zero_and_empty_string_kept():\n'
                 "    q = build_prepared_query({'a': 0, 'b': None, 'c': ['x', None, ''], 'd': False})\n"
                 "    assert q == 'a=0&c=x&c=&d=False'\n",
  'test_caller': 'from pkg.caller import build_prepared_query\n'
                 '\n'
                 'def test_list_of_tuples_input_supported():\n'
                 "    q = build_prepared_query([('tag', 'py'), ('tag', None), ('page', 2)])\n"
                 "    assert q == 'tag=py&page=2'\n"},
 {'task_id': 'cl_requests_17',
  'repo': 'psf/requests',
  'tier': 'tier2_hop1_caller_bridge',
  'issue': 'Bug in `pkg/caller.py::compute_retry_sleep`: on attempt 1 (`attempt=1`), exponential backoff '
           '`backoff_factor * (2 ** (attempt - 1))` should apply, and capped at `max_backoff=60.0`, but '
           '`compute_backoff_delay` uses `2 ** attempt` and ignores `max_backoff`.',
  'bridge_func': 'compute_backoff_delay',
  'bridge_code': 'def compute_backoff_delay(attempt: int, backoff_factor: float, max_backoff: float = '
                 '60.0) -> tuple[float, bool]:\n'
                 '    """Return (delay_seconds, was_capped) for 1-indexed retry attempt."""\n'
                 '    if attempt <= 1:\n'
                 '        return (0.0, False)\n'
                 '    delay = backoff_factor * (2 ** attempt)\n'
                 '    return (float(delay), False)\n',
  'caller_code': 'from pkg.bridge import compute_backoff_delay\n'
                 '\n'
                 'def compute_retry_sleep(attempt: int, backoff_factor: float, max_backoff: float = 60.0) '
                 '-> dict[str, object]:\n'
                 '    delay, capped = compute_backoff_delay(attempt, backoff_factor, '
                 'max_backoff=max_backoff)\n'
                 "    return {'delay': delay, 'capped': capped}\n",
  'test_target': 'from pkg.caller import compute_retry_sleep\n'
                 '\n'
                 'def test_standard_urllib3_backoff_schedule_and_cap():\n'
                 "    assert compute_retry_sleep(1, 0.5, max_backoff=10.0) == {'delay': 0.5, 'capped': "
                 'False}\n'
                 "    assert compute_retry_sleep(2, 0.5, max_backoff=10.0) == {'delay': 1.0, 'capped': "
                 'False}\n'
                 "    assert compute_retry_sleep(3, 0.5, max_backoff=10.0) == {'delay': 2.0, 'capped': "
                 'False}\n'
                 "    assert compute_retry_sleep(6, 1.0, max_backoff=10.0) == {'delay': 10.0, 'capped': "
                 'True}\n',
  'test_caller': 'from pkg.caller import compute_retry_sleep\n'
                 '\n'
                 'def test_non_positive_attempt_returns_zero():\n'
                 "    assert compute_retry_sleep(0, 1.0, max_backoff=10.0) == {'delay': 0.0, 'capped': "
                 'False}\n'},
 {'task_id': 'cl_requests_18',
  'repo': 'psf/requests',
  'tier': 'tier3_multihop_dynamic_dispatch',
  'issue': "Bug in `resolve_adapter_mount`: mounting adapters for `'https://'` and "
           "`'https://api.example.com/v1'` selects `'https://'` because mounts are iterated in insertion "
           "order instead of longest matching prefix, and prefix `'https://api.example.com/v1'` "
           "erroneously matches `'https://api.example.com/v10/users'`.",
  'bridge_func': 'resolve_adapter_mount',
  'bridge_code': 'def resolve_adapter_mount(url: str, mounts: dict[str, str]) -> tuple[str, str]:\n'
                 '    """Return (matched_prefix, adapter_tag) for url from mounts table."""\n'
                 '    for prefix, adapter in mounts.items():\n'
                 '        if url.startswith(prefix):\n'
                 '            return (prefix, adapter)\n'
                 "    raise KeyError(f'No adapter found for {url}')\n",
  'caller_code': 'from pkg.bridge import resolve_adapter_mount\n'
                 '\n'
                 'def dispatch_mounted_adapter(url: str, mounts: dict[str, str]) -> dict[str, str]:\n'
                 '    prefix, adapter = resolve_adapter_mount(url.strip(), mounts)\n'
                 "    return {'prefix': prefix, 'adapter': adapter}\n",
  'test_target': 'from pkg.caller import dispatch_mounted_adapter\n'
                 '\n'
                 'def test_longest_prefix_on_segment_boundary():\n'
                 "    mounts = {'https://': 'default_tls', 'https://api.example.com/v1': 'v1_custom'}\n"
                 "    r1 = dispatch_mounted_adapter('https://api.example.com/v1/users', mounts)\n"
                 "    assert r1 == {'prefix': 'https://api.example.com/v1', 'adapter': 'v1_custom'}\n"
                 "    r2 = dispatch_mounted_adapter('https://api.example.com/v10/users', mounts)\n"
                 "    assert r2 == {'prefix': 'https://', 'adapter': 'default_tls'}\n",
  'test_caller': 'from pkg.caller import dispatch_mounted_adapter\n'
                 '\n'
                 'def test_case_insensitive_scheme_and_exact_prefix():\n'
                 "    mounts = {'https://': 'default_tls', 'https://api.example.com/v1': 'v1_custom'}\n"
                 "    r3 = dispatch_mounted_adapter('HTTPS://api.example.com/v1', mounts)\n"
                 "    assert r3 == {'prefix': 'https://api.example.com/v1', 'adapter': 'v1_custom'}\n"},
 {'task_id': 'cl_httpx_19',
  'repo': 'encode/httpx',
  'tier': 'tier2_hop1_caller_bridge',
  'issue': 'Bug in `pkg/caller.py::build_client_request_url`: joining '
           "`base_url='https://api.example.com/v1'` with relative URL `'/users'` via "
           "`urllib.parse.urljoin` drops the `'/v1'` base path and produces "
           "`'https://api.example.com/users'` instead of `'https://api.example.com/v1/users'`.",
  'bridge_func': 'join_client_base_url',
  'bridge_code': 'from urllib.parse import urljoin\n'
                 '\n'
                 'def join_client_base_url(base_url: str, target_url: str) -> tuple[str, bool]:\n'
                 '    """Join client base_url and target_url preserving base_url path prefix."""\n'
                 "    if target_url.startswith(('http://', 'https://')):\n"
                 '        return (target_url, True)\n'
                 '    return (urljoin(base_url, target_url), False)\n',
  'caller_code': 'from pkg.bridge import join_client_base_url\n'
                 '\n'
                 'def build_client_request_url(base_url: str, target_url: str) -> dict[str, object]:\n'
                 '    url, is_abs = join_client_base_url(base_url, target_url)\n'
                 "    return {'url': url, 'is_absolute': is_abs}\n",
  'test_target': 'from pkg.caller import build_client_request_url\n'
                 '\n'
                 'def test_base_url_subpath_preserved_on_leading_slash_target():\n'
                 "    r1 = build_client_request_url('https://api.example.com/v1', '/users')\n"
                 "    assert r1 == {'url': 'https://api.example.com/v1/users', 'is_absolute': False}\n"
                 "    r2 = build_client_request_url('https://api.example.com/v1/', 'users/42')\n"
                 "    assert r2 == {'url': 'https://api.example.com/v1/users/42', 'is_absolute': False}\n",
  'test_caller': 'from pkg.caller import build_client_request_url\n'
                 '\n'
                 'def test_absolute_target_and_empty_target():\n'
                 "    r_abs = build_client_request_url('https://api.example.com/v1', "
                 "'https://other.org/ping')\n"
                 "    assert r_abs == {'url': 'https://other.org/ping', 'is_absolute': True}\n"
                 "    r_empty = build_client_request_url('https://api.example.com/v1/', '')\n"
                 "    assert r_empty == {'url': 'https://api.example.com/v1/', 'is_absolute': False}\n"},
 {'task_id': 'cl_httpx_20',
  'repo': 'encode/httpx',
  'tier': 'tier2_hop1_caller_bridge',
  'issue': 'Bug in `pkg/caller.py::serialize_wire_headers`: folding multi-value header tuples `(name, '
           "value)` must fold duplicate case-insensitive keys with `', '`, EXCEPT `'set-cookie'` which "
           'must remain separate entries in the output list.',
  'bridge_func': 'fold_multivalue_headers',
  'bridge_code': 'def fold_multivalue_headers(raw_headers: list[tuple[str, str]]) -> list[tuple[str, '
                 'str]]:\n'
                 '    """Lowercase header names and fold duplicates with \', \', keeping set-cookie '
                 'separate."""\n'
                 '    folded: dict[str, str] = {}\n'
                 '    for k, v in raw_headers:\n'
                 '        lk = k.lower()\n'
                 '        if lk in folded:\n'
                 "            folded[lk] = f'{folded[lk]}, {v}'\n"
                 '        else:\n'
                 '            folded[lk] = v\n'
                 '    return list(folded.items())\n',
  'caller_code': 'from pkg.bridge import fold_multivalue_headers\n'
                 '\n'
                 'def serialize_wire_headers(raw_headers: list[tuple[str, str]]) -> list[tuple[str, '
                 'str]]:\n'
                 '    snap = list(raw_headers)\n'
                 '    res = fold_multivalue_headers(raw_headers)\n'
                 '    if raw_headers != snap:\n'
                 "        raise RuntimeError('raw_headers mutated')\n"
                 '    return res\n',
  'test_target': 'from pkg.caller import serialize_wire_headers\n'
                 '\n'
                 'def test_set_cookie_not_folded_while_accept_is_folded():\n'
                 "    raw = [('Accept', 'text/html'), ('accept', 'application/xhtml+xml'), ('Set-Cookie', "
                 "'a=1'), ('set-cookie', 'b=2')]\n"
                 '    res = serialize_wire_headers(raw)\n'
                 "    assert res == [('accept', 'text/html, application/xhtml+xml'), ('set-cookie', "
                 "'a=1'), ('set-cookie', 'b=2')]\n",
  'test_caller': 'from pkg.caller import serialize_wire_headers\n'
                 '\n'
                 'def test_input_unmutated_and_order_preserved():\n'
                 "    raw = [('X-First', '1'), ('X-Second', '2'), ('x-first', '3')]\n"
                 '    res = serialize_wire_headers(raw)\n'
                 "    assert res == [('x-first', '1, 3'), ('x-second', '2')]\n"},
 {'task_id': 'cl_httpx_21',
  'repo': 'encode/httpx',
  'tier': 'tier3_multihop_dynamic_dispatch',
  'issue': 'Bug in `pkg/caller.py::configure_transport_timeout`: passing a 2-tuple `(connect, read)` to '
           '`normalize_timeout_config` raises `ValueError: too many values to unpack` or fails to apply '
           '`default_timeout` to `write` and `pool`.',
  'bridge_func': 'normalize_timeout_config',
  'bridge_code': 'def normalize_timeout_config(timeout: float | tuple | None, default_timeout: float = '
                 '5.0) -> dict[str, float | None]:\n'
                 '    """Normalize scalar, 2-tuple (connect, read), or 4-tuple into {connect, read, write, '
                 'pool}."""\n'
                 '    if timeout is None or isinstance(timeout, (int, float)):\n'
                 '        val = float(timeout) if timeout is not None else None\n'
                 "        return {'connect': val, 'read': val, 'write': val, 'pool': val}\n"
                 '    c, r, w, p = timeout\n'
                 "    return {'connect': float(c), 'read': float(r), 'write': float(w), 'pool': "
                 'float(p)}\n',
  'caller_code': 'from pkg.bridge import normalize_timeout_config\n'
                 '\n'
                 'def configure_transport_timeout(timeout: float | tuple | None, default_timeout: float = '
                 '5.0) -> dict[str, float | None]:\n'
                 '    cfg = normalize_timeout_config(timeout, default_timeout=default_timeout)\n'
                 "    if set(cfg.keys()) != {'connect', 'read', 'write', 'pool'}:\n"
                 "        raise KeyError('Missing timeout keys')\n"
                 '    return cfg\n',
  'test_target': 'from pkg.caller import configure_transport_timeout\n'
                 '\n'
                 'def test_two_tuple_and_four_tuple_with_none_elements():\n'
                 '    r2 = configure_transport_timeout((2.0, 10.0), default_timeout=5.0)\n'
                 "    assert r2 == {'connect': 2.0, 'read': 10.0, 'write': 5.0, 'pool': 5.0}\n"
                 '    r4 = configure_transport_timeout((1.0, None, 3.0, 4.0), default_timeout=5.0)\n'
                 "    assert r4 == {'connect': 1.0, 'read': None, 'write': 3.0, 'pool': 4.0}\n",
  'test_caller': 'from pkg.caller import configure_transport_timeout\n'
                 '\n'
                 'def test_scalar_and_none_timeout():\n'
                 "    assert configure_transport_timeout(3.5) == {'connect': 3.5, 'read': 3.5, 'write': "
                 "3.5, 'pool': 3.5}\n"
                 "    assert configure_transport_timeout(None) == {'connect': None, 'read': None, 'write': "
                 "None, 'pool': None}\n"},
 {'task_id': 'cl_httpx_22',
  'repo': 'encode/httpx',
  'tier': 'tier1_single_module_contract',
  'issue': 'Bug in `pkg/caller.py::stream_request_body`: when `chunk_size <= 0`, `iter_byte_chunks` enters '
           "an infinite loop or raises `ValueError` instead of raising `ValueError('chunk_size must be "
           "positive')`, and empty payload `b''` must return `([], 0)`.",
  'bridge_func': 'iter_byte_chunks',
  'bridge_code': 'def iter_byte_chunks(data: bytes, chunk_size: int | None) -> tuple[list[bytes], int]:\n'
                 '    """Split bytes into chunks of chunk_size and return (chunks_list, total_bytes)."""\n'
                 '    if chunk_size is None:\n'
                 '        return ([data], len(data))\n'
                 '    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]\n'
                 '    return (chunks, len(data))\n',
  'caller_code': 'from pkg.bridge import iter_byte_chunks\n'
                 '\n'
                 'def stream_request_body(data: bytes, chunk_size: int | None = 4) -> dict[str, object]:\n'
                 '    chunks, total = iter_byte_chunks(data, chunk_size)\n'
                 "    return {'chunks': chunks, 'total_bytes': total, 'num_chunks': len(chunks)}\n",
  'test_target': 'import pytest\n'
                 'from pkg.caller import stream_request_body\n'
                 '\n'
                 'def test_empty_bytes_and_invalid_chunk_size():\n'
                 "    assert stream_request_body(b'', chunk_size=4) == {'chunks': [], 'total_bytes': 0, "
                 "'num_chunks': 0}\n"
                 "    assert stream_request_body(b'', chunk_size=None) == {'chunks': [], 'total_bytes': 0, "
                 "'num_chunks': 0}\n"
                 '    with pytest.raises(ValueError):\n'
                 "        stream_request_body(b'abcdef', chunk_size=0)\n"
                 '    with pytest.raises(ValueError):\n'
                 "        stream_request_body(b'abcdef', chunk_size=-2)\n",
  'test_caller': 'from pkg.caller import stream_request_body\n'
                 '\n'
                 'def test_normal_chunk_splitting_and_none_chunk_size():\n'
                 "    res = stream_request_body(b'abcdef', chunk_size=2)\n"
                 "    assert res == {'chunks': [b'ab', b'cd', b'ef'], 'total_bytes': 6, 'num_chunks': 3}\n"
                 "    res_all = stream_request_body(b'abcdef', chunk_size=None)\n"
                 "    assert res_all == {'chunks': [b'abcdef'], 'total_bytes': 6, 'num_chunks': 1}\n"},
 {'task_id': 'cl_httpx_23',
  'repo': 'encode/httpx',
  'tier': 'tier1_single_module_contract',
  'issue': 'Bug in `pkg/caller.py::canonicalize_origin`: default ports (`http://example.com:80` and '
           '`https://example.com:443`) are not stripped from netloc, and uppercase hostnames are not '
           'lowercased.',
  'bridge_func': 'normalize_origin_tuple',
  'bridge_code': 'from urllib.parse import urlparse\n'
                 '\n'
                 'def normalize_origin_tuple(url: str) -> tuple[str, str, int]:\n'
                 '    """Return (canonical_origin_str, scheme, effective_port) for HTTP/HTTPS URL."""\n'
                 '    p = urlparse(url)\n'
                 '    scheme = p.scheme.lower()\n'
                 "    host = p.hostname or ''\n"
                 "    port = p.port or (443 if scheme == 'https' else 80)\n"
                 "    return (f'{scheme}://{host}:{port}', scheme, port)\n",
  'caller_code': 'from pkg.bridge import normalize_origin_tuple\n'
                 '\n'
                 'def canonicalize_origin(url: str) -> dict[str, object]:\n'
                 '    origin, scheme, port = normalize_origin_tuple(url)\n'
                 '    if not isinstance(port, int):\n'
                 "        raise TypeError('effective_port must be int')\n"
                 "    return {'origin': origin, 'scheme': scheme, 'port': port}\n",
  'test_target': 'from pkg.caller import canonicalize_origin\n'
                 '\n'
                 'def test_default_ports_stripped_and_host_lowercased():\n'
                 "    r1 = canonicalize_origin('HTTPS://Example.COM:443/path')\n"
                 "    assert r1 == {'origin': 'https://example.com', 'scheme': 'https', 'port': 443}\n"
                 "    r2 = canonicalize_origin('http://Example.COM:80/path')\n"
                 "    assert r2 == {'origin': 'http://example.com', 'scheme': 'http', 'port': 80}\n",
  'test_caller': 'from pkg.caller import canonicalize_origin\n'
                 '\n'
                 'def test_custom_port_retained_in_origin():\n'
                 "    r3 = canonicalize_origin('https://Example.COM:8443/path')\n"
                 "    assert r3 == {'origin': 'https://example.com:8443', 'scheme': 'https', 'port': "
                 '8443}\n'},
 {'task_id': 'cl_httpx_24',
  'repo': 'encode/httpx',
  'tier': 'tier3_multihop_dynamic_dispatch',
  'failure_category': 'decorator_closure_state_leak',
  'failure_description': 'A 2-hop `@transactional_pool_guard` context-manager decorator in '
                         '`pkg/runtime_dispatch.py` caches `_closure_epoch` inside a closure cell '
                         'across idempotent releases; patching `bridge.py` alone cannot reset the '
                         'decorator closure state in `runtime_dispatch.py`.',
  'issue': 'Bug in `release_pool_lease`: releasing a connection lease id that is already closed raises '
           '`KeyError` instead of returning `(lease_id, False)`.',
  'bridge_func': 'release_pool_lease',
  'bridge_code': 'def release_pool_lease(active_leases: set[str], lease_id: str) -> tuple[str, bool]:\n'
                 '    """Remove lease_id from active_leases and return (lease_id, was_present)."""\n'
                 '    active_leases.remove(lease_id)\n'
                 '    return (lease_id, True)\n',
  'caller_code': 'from pkg.bridge import release_pool_lease\n'
                 'from pkg.runtime_dispatch import transactional_pool_guard\n'
                 '\n'
                 '@transactional_pool_guard\n'
                 'def close_client_lease(active_leases: set[str], lease_id: str) -> dict[str, object]:\n'
                 '    lid, released = release_pool_lease(active_leases, lease_id)\n'
                 "    return {'lease_id': lid, 'released': released, 'remaining': len(active_leases)}\n",
  'runtime_dispatch_code': 'import functools\n'
                           '\n'
                           'def transactional_pool_guard(fn):\n'
                           "    _closure_epoch = {'rollback_cleared': False}\n"
                           '    @functools.wraps(fn)\n'
                           '    def _wrapped(active_leases: set[str], lease_id: str):\n'
                           '        res = fn(active_leases, lease_id)\n'
                           "        if not res['released'] and not _closure_epoch['rollback_cleared']:\n"
                           "            raise RuntimeError('Closure guard _closure_epoch was not reset "
                           "across idempotent release')\n"
                           '        return res\n'
                           '    return _wrapped\n',
  'test_target': 'from pkg.bridge import release_pool_lease\n'
                 '\n'
                 'def test_idempotent_lease_release():\n'
                 "    pool = {'conn_1', 'conn_2'}\n"
                 "    assert release_pool_lease(pool, 'conn_1') == ('conn_1', True)\n"
                 "    assert release_pool_lease(pool, 'conn_1') == ('conn_1', False)\n",
  'test_caller': 'from pkg.caller import close_client_lease\n'
                 '\n'
                 'def test_transactional_decorator_closure_idempotent_release():\n'
                 "    pool = {'conn_1'}\n"
                 "    assert close_client_lease(pool, 'conn_1')['released'] is True\n"
                 "    assert close_client_lease(pool, 'conn_1')['released'] is False\n"}]


def get_closed_loop_24_specs() -> list[dict[str, Any]]:
    return [dict(s) for s in CLOSED_LOOP_24_SPECS]


def generate_distractor_module_text(spec: dict[str, Any]) -> str:
    blocks: list[str] = [
        '"""Legacy compatibility utilities and deprecated helpers (DO NOT USE for active router/client paths)."""\n'
    ]
    fn_name = spec["bridge_func"]
    for i in range(1, 28):
        blocks.append(
            f"def legacy_{fn_name}_v{i}(payload: object, strict: bool = False, fallback: str = 'default_{i}') -> dict[str, object]:\n"
            f'    """Deprecated compatibility wrapper #{i} around {fn_name} for old v0.{i} schema migration."""\n'
            f"    state = {{'step': {i}, 'strict': strict, 'fallback': fallback, 'raw': str(payload)[:64]}}\n"
            f"    if payload is None:\n"
            f"        return {{'status': 'empty_{i}', 'state': state}}\n"
            f"    for idx in range(3):\n"
            f"        state[f'audit_{{idx}}'] = f'checked_{{idx}}_{{fallback}}'\n"
            f"    return {{'status': 'ok_{i}', 'state': state}}\n"
        )
    return "\n".join(blocks)
