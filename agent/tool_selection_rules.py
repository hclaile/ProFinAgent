import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple


def _safe_int(v: Any, default: int) -> int:
    try:
        x = int(v)
        return x
    except Exception:
        return default


def _merge_strategy(base: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (child or {}).items():
        if k == "inherits":
            continue
        out[k] = v
    return out


def load_tool_selection_strategy(
    config_path: Optional[str] = None,
    env: Optional[str] = None,
    cache: Optional[Dict[str, Any]] = None,
    debug_print: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    dbg = debug_print or (lambda *a, **k: None)

    if not config_path:
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../configs/must_have_rules.json')
        )

    # default (config agent)
    fallback = {
        "per_type_quota": 1,
        "min_total_tools": 1,
        "max_total_tools": 16,
        "must_have_rules": [],
    }

    try:
        mtime = os.path.getmtime(config_path)
    except Exception:
        dbg(f"[WARN] must_have_rules config does not exist read: {config_path}, use")
        return dict(fallback)

    # update cache: cache save {mtime, raw}
    if cache is not None:
        if cache.get("mtime") == mtime and isinstance(cache.get("raw"), dict):
            raw = cache["raw"]
        else:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                cache["mtime"] = mtime
                cache["raw"] = raw
                dbg(f"[INFO] must_have_rules: {config_path}")
            except Exception as e:
                dbg(f"[WARN] read must_have_rules failed: {e}, use")
                return dict(fallback)
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            dbg(f"[WARN] read must_have_rules failed: {e}, use")
            return dict(fallback)

    if not isinstance(raw, dict):
        dbg('[WARN] must_have_rules.json is not dict structure, use')
        return dict(fallback)

    default_env = raw.get("default_env") or "default"
    env_from = raw.get("env_from") or {}
    env_var = env_from.get("env_var") or "RULES_ENV"
    # compatible history environment variable, config
    legacy_env_vars = ["ProFinAgent_RULES_ENV"]
    allowed = env_from.get("allowed")
    if not isinstance(allowed, list) or not allowed:
        allowed = ["default", "benchmark", "online"]

    env_from_config = os.getenv(env_var, "")
    if not env_from_config:
        for legacy_name in legacy_env_vars:
            env_from_config = os.getenv(legacy_name, "")
            if env_from_config:
                dbg(
                    f"[INFO] history environment variable {legacy_name}={env_from_config},"
                    f"{env_var}"
                )
                break

    selected_env = env or env_from_config or default_env or "default"
    if selected_env not in allowed:
        dbg(f"[WARN] must_have_rules env='{selected_env}' allowed={allowed}, default_env='{default_env}'")
        selected_env = default_env if default_env in allowed else "default"

    strategies = raw.get("strategies") or {}
    if not isinstance(strategies, dict):
        dbg('[WARN] must_have_rules.json strategies is not dict, use')
        return dict(fallback)

    #
    def _resolve_env(e: str, depth: int = 0) -> Dict[str, Any]:
        if depth > 5:
            return {}
        s = strategies.get(e) or {}
        if not isinstance(s, dict):
            return {}
        parent = s.get("inherits")
        if parent and isinstance(parent, str):
            base = _resolve_env(parent, depth + 1)
            return _merge_strategy(base, s)
        return dict(s)

    strategy = _resolve_env(selected_env) or _resolve_env(default_env) or {}

    per_type_quota = _safe_int(strategy.get("per_type_quota"), fallback["per_type_quota"])
    min_total_tools = _safe_int(strategy.get("min_total_tools"), fallback["min_total_tools"])
    max_total_tools = _safe_int(strategy.get("max_total_tools"), fallback["max_total_tools"])

    # normalize must_have_rules
    must_have_rules: List[Tuple[List[str], List[str]]] = []
    raw_rules = strategy.get("must_have_rules") or []
    if isinstance(raw_rules, list):
        for item in raw_rules:
            if not isinstance(item, dict):
                continue
            patterns = item.get("patterns") or []
            tools = item.get("tools") or []
            if not isinstance(patterns, list) or not isinstance(tools, list):
                continue
            patterns = [str(p) for p in patterns if p is not None and str(p).strip()]
            tools = [str(t) for t in tools if t is not None and str(t).strip()]
            if patterns and tools:
                must_have_rules.append((patterns, tools))

    return {
        "per_type_quota": per_type_quota,
        "min_total_tools": min_total_tools,
        "max_total_tools": max_total_tools,
        "must_have_rules": must_have_rules,
        "env": selected_env,
        "config_path": config_path,
    }

