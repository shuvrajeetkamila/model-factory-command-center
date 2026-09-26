#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_factory.py — The Master Orchestrator ("The Brain")
=======================================================

Standalone automation layer that sits OUTSIDE the nine core repositories of
the Model Editing & Governance Factory. It executes the full production loop
as sequential subprocess stages, parses the causal evaluation artifacts
WITHOUT modifying any repository, and on failure dynamically retunes external
hyperparameters and loops back — a self-correcting factory.

Default pipeline (all paths configurable in ``factory_config.json``):

    [1] dedicated-feature-crosscoder   scripts/train_dfc.py --synthetic ...
              │  (writes dfc_final.pt + final_metrics.json)
    [2] circuit-bridge-plugin          circuit_bridge.py --checkpoint ... --out-dir ...
              │  (writes latest_patch.npy / .json exchange artifacts)
    [3] causal-functional-effect-test  demo/run_equylapta7_4.py
              │  (writes results/e7_4_results.json in ITS repo — read-only for us)
    [4] EVALUATION GATE ── threshold_met == false  ──►  FALLBACK LOOP:
              │                 retune crosscoder hyperparameters per config,
              │                 loop back to stage [1], up to max_retries.
              ▼ threshold_met == true
    [5] SUCCESS GATES (run once, in order):
          vault_register  (optional; e.g. vault_db.py register / circuit-vault CLI)
          audit_router    (ai-audit-router-toolkit text_router.py route ...)

Mechanical pass/fail contract discovered by workspace inspection
-----------------------------------------------------------------
``causal-functional-effect-test/demo/run_equylapta7_4.py`` writes
``<repo>/results/e7_4_results.json`` containing::

    {
      "milestone": "EQUYLAPTA E7.4",
      "scientific_level": "LEVEL B: REPRESENTATIONAL_ALIGNMENT_ONLY",
      "predeclared_agreement_threshold_pct": 90.0,
      "observed_peak_functional_agreement_pct": 46.36,
      "threshold_met": false,                     ◄── the mechanical equivalent
      "source_causal_results": {..., "causal_drop": 24.0},   of THRESHOLD_NOT_MET
      ...
    }

The gate below treats ``threshold_met: false`` (or a missing/failed numeric
comparison ``observed >= predeclared``, or an explicit status string such as
"THRESHOLD_NOT_MET"/"FAILED") as NOT MET, and ``threshold_met: true`` (or
"THRESHOLD_MET"/"PASSED") as MET.

Typical usage
-------------
::

    python run_factory.py --config factory_config.json           # full run
    python run_factory.py --dry-run                              # print resolved commands
    python run_factory.py --max-retries 5 --start-stage bridge   # overrides
    python run_factory.py --demo                                 # self-contained loop proof

License: dual-licensed MIT OR Apache-2.0 (see LICENSE, LICENSE-MIT,
LICENSE-APACHE).  Copyright 2026 Shuvrajeet Kamila.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

__version__ = "1.0.0"

LOG = logging.getLogger("run_factory")

TERMINAL_STATUSES = (
    "FACTORY_SUCCESS",
    "THRESHOLD_NOT_MET_EXHAUSTED",
    "STAGE_FAILURE",
    "CONFIG_ERROR",
)

MET_STRINGS = {"THRESHOLD_MET", "PASSED", "PASS", "SUCCESS", "MET"}
NOT_MET_STRINGS = {"THRESHOLD_NOT_MET", "FAILED", "FAIL", "NOT_MET", "THRESHOLD_FAILED"}


class FactoryError(RuntimeError):
    """Fatal orchestrator failure."""


class ConfigError(FactoryError):
    """Malformed factory configuration."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG_NAME = "factory_config.json"

_CONFIG_REQUIRED_KEYS = ("stages", "evaluation")


def load_config(path: str) -> Dict[str, Any]:
    """Load the factory config (JSON natively; YAML when PyYAML is available)."""
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path!r}")
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise ConfigError(
                f"{path!r} is YAML but PyYAML is not installed. Either "
                "`pip install pyyaml` or use the JSON config (factory_config.json)."
            ) from exc
        cfg = yaml.safe_load(text)
    else:
        try:
            cfg = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Config {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config {path!r} must deserialize to an object/dict")
    missing = [k for k in _CONFIG_REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"Config {path!r} is missing required keys: {missing}")
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    stages = cfg.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ConfigError("'stages' must be a non-empty list")
    ids = set()
    for st in stages:
        if not isinstance(st, dict) or "id" not in st:
            raise ConfigError(f"Every stage needs an 'id': {st!r}")
        sid = str(st["id"])
        if sid in ids:
            raise ConfigError(f"Duplicate stage id {sid!r}")
        ids.add(sid)
        cmd = st.get("command")
        if not st.get("success_gate", False) and (not isinstance(cmd, list) or not cmd):
            raise ConfigError(f"Stage {sid!r}: 'command' must be a non-empty list")
        if cmd is not None and not all(isinstance(tok, str) for tok in cmd):
            raise ConfigError(f"Stage {sid!r}: 'command' tokens must all be strings")
    restart = cfg.get("restart_stage")
    if restart is not None and restart not in ids:
        raise ConfigError(f"'restart_stage' {restart!r} does not match any stage id")
    ev = cfg.get("evaluation")
    if not isinstance(ev, dict) or "results_json" not in ev:
        raise ConfigError("'evaluation' must be an object containing 'results_json'")


# --------------------------------------------------------------------------- #
# Placeholder resolution
# --------------------------------------------------------------------------- #

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def resolve_workspace_root(cfg: Dict[str, Any], base_dir: str) -> str:
    """Resolve `workspace_root` to an absolute path.

    Absolute values are used verbatim; relative values are anchored to the
    config file's directory (`base_dir`) — never to the shell's CWD — so the
    pipeline behaves identically regardless of where it is invoked from.
    The shipped config uses `"workspace_root": ".."`, i.e. the parent folder of
    this Command Center repository, where the 12 sibling repos live.
    """
    raw = cfg.get("workspace_root", base_dir)
    raw = str(raw)
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    return os.path.abspath(os.path.join(base_dir, raw))


def build_context(cfg: Dict[str, Any], base_dir: str, run_id: str, run_dir: str,
                  attempt: int) -> Dict[str, str]:
    ctx: Dict[str, str] = {
        "python": sys.executable or "python3",
        "workspace": resolve_workspace_root(cfg, base_dir),
        "plugin_dir": os.path.abspath(base_dir),
        "run_id": run_id,
        "run_dir": os.path.abspath(run_dir),
        "attempt": str(attempt),
    }
    repos = cfg.get("repos") or {}
    if not isinstance(repos, dict):
        raise ConfigError("'repos' must be an object mapping names to paths")
    for name, rel in repos.items():
        ctx[f"repo_{name}"] = os.path.abspath(os.path.join(ctx["workspace"], str(rel)))
    return ctx


def resolve_token(token: str, ctx: Dict[str, str], stage_id: str) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in ctx:
            raise ConfigError(
                f"Stage {stage_id!r}: unknown placeholder {{{key}}} — available: "
                f"{sorted(ctx)}")
        return ctx[key]
    return _PLACEHOLDER_RE.sub(sub, token)


def resolve_command(cmd: Sequence[str], ctx: Dict[str, str], stage_id: str) -> List[str]:
    return [resolve_token(tok, ctx, stage_id) for tok in cmd]


# --------------------------------------------------------------------------- #
# Hyperparameter fallback tuning
# --------------------------------------------------------------------------- #


def _parse_number(token: str) -> Tuple[float, bool]:
    """Parse a CLI token as a number; returns (value, is_int_style)."""
    cleaned = token.replace("_", "")
    was_int = "." not in cleaned and "e" not in cleaned.lower()
    return float(cleaned), was_int


def _format_number(value: float, int_style: bool) -> str:
    if int_style:
        return str(int(round(value)))
    return repr(round(value, 10)).rstrip("0").rstrip(".") if value != int(value) else f"{value:.6g}"


def apply_adjustments(command: List[str], adjustments: Sequence[Dict[str, Any]],
                      stage_id: str) -> Tuple[List[str], List[str]]:
    """Rewrite numeric CLI flags of a resolved command per adjustment rules.

    Each adjustment: ``{"flag": "--total-tokens", "op": "multiply"|"add"|"set",
    "value": <num>, "min": <num>?, "max": <num>?, "round_int": bool?}``.
    Supports both ``--flag 123`` and ``--flag=123`` token styles. Returns the
    new command and a human-readable change log.
    """
    cmd = list(command)
    changes: List[str] = []
    for adj in adjustments:
        flag = str(adj.get("flag", ""))
        op = str(adj.get("op", "multiply")).lower()
        if not flag.startswith("-"):
            raise ConfigError(f"Adjustment flag must start with '-': {flag!r}")
        if op not in ("multiply", "add", "set"):
            raise ConfigError(f"Unknown adjustment op {op!r} for {flag}")

        hit = False
        for i, tok in enumerate(cmd):
            target_idx, current_token = -1, None
            if tok == flag and i + 1 < len(cmd):
                target_idx, current_token = i + 1, cmd[i + 1]
            elif tok.startswith(flag + "="):
                target_idx, current_token = i, tok.split("=", 1)[1]
            if target_idx < 0:
                continue
            try:
                value, int_style = _parse_number(current_token)
            except ValueError:
                LOG.warning("Stage %s: flag %s value %r is not numeric — skipped",
                            stage_id, flag, current_token)
                continue
            new_value = value
            if op == "multiply":
                new_value = value * float(adj.get("value", 1.0))
            elif op == "add":
                new_value = value + float(adj.get("value", 0.0))
            else:
                new_value = float(adj.get("value", value))
            if "min" in adj:
                new_value = max(new_value, float(adj["min"]))
            if "max" in adj:
                new_value = min(new_value, float(adj["max"]))
            if adj.get("round_int", int_style):
                int_style = True
            new_token = _format_number(new_value, int_style)
            if target_idx == i:
                cmd[i] = f"{flag}={new_token}"
            else:
                cmd[target_idx] = new_token
            changes.append(f"{stage_id}: {flag} {current_token} -> {new_token} ({op} {adj.get('value')})")
            hit = True
            break
        if not hit:
            LOG.warning("Stage %s: adjustment flag %s not present in command — ignored",
                        stage_id, flag)
    return cmd, changes


# --------------------------------------------------------------------------- #
# Evaluation gate
# --------------------------------------------------------------------------- #


def evaluate_results(results_path: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the causal-test results JSON and decide PASS / THRESHOLD_NOT_MET.

    Never raises on bad data: returns ``{"gate": "ERROR", ...}`` so the caller
    can treat a corrupt/missing artifact as a failed attempt (defensive spec).
    """
    out: Dict[str, Any] = {
        "gate": "ERROR",
        "results_path": os.path.abspath(results_path),
        "reason": "",
        "metrics": {},
        "raw_status": None,
    }
    if not os.path.isfile(results_path):
        out["reason"] = f"results file not found: {results_path!r}"
        return out
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        out["reason"] = f"results file unreadable/corrupt: {exc}"
        return out
    if not isinstance(data, dict):
        out["reason"] = "results JSON is not an object"
        return out

    # --- metric extraction (paths configurable, defaults match e7_4_results.json)
    metric_paths = criteria.get("metric_paths") or {
        "observed_agreement_pct": "observed_peak_functional_agreement_pct",
        "predeclared_threshold_pct": "predeclared_agreement_threshold_pct",
        "causal_drop_pp": "source_causal_results.causal_drop",
        "scientific_level": "scientific_level",
        "milestone": "milestone",
    }
    for alias, dotted in metric_paths.items():
        node: Any = data
        try:
            for part in str(dotted).split("."):
                node = node[part] if not isinstance(node, list) else node[int(part)]
            out["metrics"][alias] = node
        except (KeyError, IndexError, TypeError, ValueError):
            out["metrics"][alias] = None

    # --- decision ladder -----------------------------------------------------
    status_key = criteria.get("status_key")            # e.g. "status"
    bool_key = criteria.get("pass_key", "threshold_met")
    observed_key = criteria.get("observed_key", "observed_peak_functional_agreement_pct")
    threshold_key = criteria.get("threshold_key", "predeclared_agreement_threshold_pct")
    fixed_threshold = criteria.get("fixed_threshold")

    def _dig(dotted: str) -> Any:
        node: Any = data
        try:
            for part in str(dotted).split("."):
                node = node[part] if not isinstance(node, list) else node[int(part)]
            return node
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    # 1) explicit status string
    if status_key:
        raw = _dig(status_key)
        if isinstance(raw, str):
            out["raw_status"] = raw
            up = raw.strip().upper()
            if up in MET_STRINGS:
                out["gate"] = "PASS"
                out["reason"] = f"{status_key}={raw!r}"
                return out
            if up in NOT_MET_STRINGS:
                out["gate"] = "THRESHOLD_NOT_MET"
                out["reason"] = f"{status_key}={raw!r}"
                return out

    # 2) boolean pass key (e7_4_results.json: "threshold_met")
    bool_val = _dig(bool_key)
    if isinstance(bool_val, bool):
        out["gate"] = "PASS" if bool_val else "THRESHOLD_NOT_MET"
        out["reason"] = f"{bool_key}={bool_val}"
        return out

    # 3) numeric comparison observed >= threshold
    observed = _dig(observed_key)
    threshold = fixed_threshold if fixed_threshold is not None else _dig(threshold_key)
    if isinstance(observed, (int, float)) and isinstance(threshold, (int, float)):
        met = float(observed) >= float(threshold)
        out["gate"] = "PASS" if met else "THRESHOLD_NOT_MET"
        out["reason"] = (f"{observed_key}={observed} vs {threshold_key or 'fixed_threshold'}"
                         f"={threshold}")
        return out

    out["gate"] = "ERROR"
    out["reason"] = ("no decidable pass criteria found in results JSON "
                     f"(looked for {status_key!r}, {bool_key!r}, {observed_key!r}>={threshold_key!r})")
    return out


def build_summary_text(run_id: str, attempt: int, evaluation: Dict[str, Any],
                       patch_paths: Optional[Dict[str, str]] = None) -> str:
    m = evaluation.get("metrics", {})
    observed = m.get("observed_agreement_pct")
    threshold = m.get("predeclared_threshold_pct")
    causal = m.get("causal_drop_pp")
    level = m.get("scientific_level") or "UNKNOWN"
    parts = [
        f"Factory run {run_id} attempt {attempt}: transplant PASSED the causal gate.",
        f"Functional agreement {observed}% >= {threshold}% predeclared threshold."
        if observed is not None and threshold is not None else "Causal gate passed.",
        f"Causal drop {causal} pp." if causal is not None else "",
        f"Scientific level: {level}.",
        f"Patch: {patch_paths.get('steering_npy')}" if patch_paths else "",
        "Routing validated model into the production audit pipeline.",
    ]
    return " ".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Stage execution
# --------------------------------------------------------------------------- #


def run_stage(stage: Dict[str, Any], ctx: Dict[str, str], run_dir: str,
              attempt: int, dry_run: bool = False) -> Dict[str, Any]:
    """Execute one stage as a subprocess with live streaming into a log file."""
    sid = str(stage["id"])
    record: Dict[str, Any] = {
        "stage": sid,
        "attempt": attempt,
        "started_at": _utc_now_iso(),
        "status": "SKIPPED",
        "returncode": None,
        "duration_sec": 0.0,
        "log_file": None,
        "changes": [],
        "expect_files_ok": None,
        "error": None,
    }
    command = resolve_command(stage.get("command") or [], ctx, sid)
    if stage.get("adjustments") and ctx.get("_apply_adjustments") == "1":
        command, changes = apply_adjustments(command, stage["adjustments"], sid)
        record["changes"] = changes

    cwd = stage.get("cwd")
    cwd = resolve_token(str(cwd), ctx, sid) if cwd else ctx["workspace"]
    if not os.path.isdir(cwd):
        record.update(status="FAILED", error=f"cwd does not exist: {cwd!r}")
        return record

    timeout = float(stage.get("timeout_sec", 3600))
    record["command"] = command
    record["cwd"] = cwd
    printable = " ".join(shlex.quote(t) for t in command)
    LOG.info("─ stage [%s] %s", sid, printable)
    LOG.info("  cwd=%s timeout=%.0fs", cwd, timeout)
    if dry_run:
        record["status"] = "DRY_RUN"
        return record

    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"stage_{sid}_attempt{attempt}.log")
    record["log_file"] = log_path
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (stage.get("env") or {}).items()})

    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                command, cwd=cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(f"    │ {line}" if line.endswith("\n") else f"    │ {line}\n")
                sys.stdout.flush()
                lf.write(line)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                record.update(status="TIMEOUT", error=f"stage exceeded {timeout:.0f}s")
                record["duration_sec"] = round(time.time() - t0, 3)
                return record
    except FileNotFoundError as exc:
        record.update(status="FAILED", error=f"executable not found: {exc}")
        record["duration_sec"] = round(time.time() - t0, 3)
        return record
    except OSError as exc:
        record.update(status="FAILED", error=f"subprocess crash: {exc}")
        record["duration_sec"] = round(time.time() - t0, 3)
        return record

    record["returncode"] = rc
    record["duration_sec"] = round(time.time() - t0, 3)

    if rc != 0:
        record.update(status="FAILED", error=f"exit code {rc} (see {log_path})")
    else:
        expect = stage.get("expect_files") or []
        missing = []
        for rel in expect:
            p = resolve_token(str(rel), ctx, sid)
            p_abs = p if os.path.isabs(p) else os.path.join(cwd, p)
            if not os.path.exists(p_abs):
                missing.append(p_abs)
        record["expect_files_ok"] = not missing
        if missing:
            record.update(status="FAILED",
                          error=f"expected artifact(s) missing after success: {missing}")
        else:
            record["status"] = "OK"
    return record


# --------------------------------------------------------------------------- #
# The factory loop
# --------------------------------------------------------------------------- #


def run_factory(cfg: Dict[str, Any], base_dir: str, dry_run: bool = False,
                max_retries: Optional[int] = None, start_stage: Optional[str] = None,
                demo_root: Optional[str] = None) -> Dict[str, Any]:
    validate_config(cfg)
    runs_root = cfg.get("runs_dir", os.path.join("runs", "factory"))
    if not os.path.isabs(runs_root):
        runs_root = os.path.join(demo_root or base_dir, runs_root)
    run_id = datetime.now().strftime("factory-%Y%m%d-%H%M%S")
    run_dir = os.path.join(runs_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    retries_cfg = cfg.get("max_retries", 3)
    max_attempts = int(retries_cfg if max_retries is None else max_retries) + 1
    if max_attempts < 1:
        raise ConfigError("max_retries must be >= 0")

    stages: List[Dict[str, Any]] = [s for s in cfg["stages"] if s.get("enabled", True)]
    pipeline = [s for s in stages if not s.get("success_gate", False)]
    gates = [s for s in stages if s.get("success_gate", False)]
    restart_stage = cfg.get("restart_stage") or (pipeline[0]["id"] if pipeline else None)
    evaluation_cfg: Dict[str, Any] = dict(cfg.get("evaluation") or {})
    results_rel = str(evaluation_cfg.get("results_json"))
    results_path = results_rel if os.path.isabs(results_rel) else \
        os.path.abspath(os.path.join(resolve_workspace_root(cfg, base_dir),
                                     results_rel))
    # results_json may itself contain {repo_*} placeholders — resolved per attempt below.

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": os.path.abspath(run_dir),
        "started_at": _utc_now_iso(),
        "finished_at": None,
        "status": None,
        "max_attempts": max_attempts,
        "dry_run": dry_run,
        "attempts": [],
        "final_evaluation": None,
        "gate_stage_records": [],
        "config_path": os.path.abspath(cfg.get("_config_path", "<inline>")),
    }

    ctx_base = build_context(cfg, base_dir, run_id, run_dir, attempt=1)
    results_path_tpl = results_rel

    attempt = 1
    start_idx = 0
    if start_stage:
        ids = [s["id"] for s in pipeline]
        if start_stage not in ids:
            raise ConfigError(f"--start-stage {start_stage!r} not among pipeline stages {ids}")
        start_idx = ids.index(start_stage)

    evaluation: Dict[str, Any] = {"gate": "ERROR", "reason": "loop never reached evaluation"}
    while attempt <= max_attempts:
        LOG.info("═" * 68)
        LOG.info("FACTORY ATTEMPT %d/%d  (run_id=%s)", attempt, max_attempts, run_id)
        LOG.info("═" * 68)
        ctx = build_context(cfg, base_dir, run_id, run_dir, attempt)
        if attempt > 1:
            ctx["_apply_adjustments"] = "1"
        attempt_record: Dict[str, Any] = {"attempt": attempt, "stage_records": [],
                                          "config_snapshot": None, "evaluation": None}

        # snapshot the effective config for this attempt
        snap_path = os.path.join(run_dir, f"config_attempt_{attempt}.json")
        snapshot = json.loads(json.dumps(cfg, default=str))
        snapshot.pop("_config_path", None)
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        attempt_record["config_snapshot"] = snap_path

        failed_hard = False
        for stage in pipeline[start_idx:] if attempt == 1 else pipeline:
            rec = run_stage(stage, ctx, run_dir, attempt, dry_run=dry_run)
            attempt_record["stage_records"].append(rec)
            if rec["status"] in ("FAILED", "TIMEOUT"):
                if stage.get("optional", False):
                    LOG.warning("Optional stage [%s] failed (%s) — continuing.",
                                stage["id"], rec.get("error"))
                    continue
                stage_retry = int(stage.get("stage_retry", 0))
                retried_ok = False
                for extra in range(stage_retry):
                    LOG.warning("Stage [%s] failed — intra-stage retry %d/%d",
                                stage["id"], extra + 1, stage_retry)
                    rec2 = run_stage(stage, ctx, run_dir, attempt, dry_run=dry_run)
                    attempt_record["stage_records"].append(rec2)
                    if rec2["status"] == "OK":
                        retried_ok = True
                        break
                if not retried_ok:
                    failed_hard = True
                    break
        start_idx = 0  # subsequent attempts always restart from restart_stage/top

        if failed_hard:
            evaluation = {"gate": "ERROR", "reason": "a pipeline stage failed hard"}
            attempt_record["evaluation"] = evaluation
            summary["attempts"].append(attempt_record)
            summary["status"] = "STAGE_FAILURE"
            break

        if dry_run:
            attempt_record["evaluation"] = {"gate": "DRY_RUN"}
            summary["attempts"].append(attempt_record)
            attempt += 1
            continue

        # --- evaluation gate --------------------------------------------------
        results_path = resolve_token(results_path_tpl, ctx, str(pipeline[-1]["id"]))
        if not os.path.isabs(results_path):
            results_path = os.path.abspath(os.path.join(ctx["workspace"], results_path))
        evaluation = evaluate_results(results_path, evaluation_cfg)
        attempt_record["evaluation"] = evaluation
        summary["attempts"].append(attempt_record)

        if evaluation["gate"] == "PASS":
            LOG.info("✅ EVALUATION GATE: PASS (%s)", evaluation.get("reason"))
            summary["final_evaluation"] = evaluation
            break
        elif evaluation["gate"] == "THRESHOLD_NOT_MET":
            LOG.error("❌ EVALUATION GATE: THRESHOLD_NOT_MET (%s)", evaluation.get("reason"))
            LOG.error("   Host-learning control bypassed the transplant or the "
                      "predeclared functional-agreement target was missed.")
            if attempt >= max_attempts:
                LOG.error("   Retry budget exhausted (%d attempts).", max_attempts)
                summary["final_evaluation"] = evaluation
                summary["status"] = "THRESHOLD_NOT_MET_EXHAUSTED"
                break
            LOG.warning("   ⟲ FALLBACK: retuning hyperparameters and looping back to "
                        "stage [%s] (attempt %d/%d)…", restart_stage, attempt + 1, max_attempts)
            attempt += 1
            continue
        else:  # ERROR (missing/corrupt results artifact)
            LOG.error("⚠ EVALUATION GATE: ERROR — %s", evaluation.get("reason"))
            if attempt >= max_attempts:
                summary["final_evaluation"] = evaluation
                summary["status"] = "STAGE_FAILURE"
                break
            attempt += 1
            continue

    # --- success gates ---------------------------------------------------------
    if summary["status"] is None and evaluation.get("gate") == "PASS" and not dry_run:
        patch_paths = _collect_patch_paths(cfg, ctx_base, run_dir)
        summary_text = build_summary_text(run_id, attempt, evaluation, patch_paths)
        gate_ctx = dict(ctx_base)
        gate_ctx.update({
            "summary_text": summary_text,
            "results_json": results_path,
            "attempt": str(attempt),
        })
        if patch_paths:
            gate_ctx["patch_npy"] = patch_paths.get("steering_npy", "")
            gate_ctx["patch_json"] = patch_paths.get("manifest_json", "")
        else:
            # Defensive defaults so success-gate commands referencing these
            # placeholders still resolve (the gate stage then fails softly on
            # the empty path instead of crashing the run with a ConfigError).
            gate_ctx.setdefault("patch_npy", "")
            gate_ctx.setdefault("patch_json", "")
        all_ok = True
        for stage in gates:
            rec = run_stage(stage, gate_ctx, run_dir, attempt, dry_run=dry_run)
            summary["gate_stage_records"].append(rec)
            if rec["status"] not in ("OK", "DRY_RUN") and not stage.get("optional", False):
                all_ok = False
        summary["status"] = "FACTORY_SUCCESS" if all_ok else "STAGE_FAILURE"
        summary["summary_text"] = summary_text
    elif summary["status"] is None and dry_run:
        # Dry-run also plans the success-gate commands so every command line in
        # the config is verified before a real run (placeholder resolution,
        # flag spelling, argument order). Gate placeholders that only exist
        # after a successful attempt are defaulted defensively.
        patch_paths = _collect_patch_paths(cfg, ctx_base, run_dir)
        gate_ctx = dict(ctx_base)
        gate_ctx.update({
            "summary_text": "(dry-run summary text placeholder)",
            "results_json": resolve_token(results_path_tpl, ctx_base, "dry_run"),
            "attempt": str(max(attempt - 1, 1)),
            "patch_npy": (patch_paths or {}).get("steering_npy", ""),
            "patch_json": (patch_paths or {}).get("manifest_json", ""),
        })
        for stage in gates:
            rec = run_stage(stage, gate_ctx, run_dir, attempt, dry_run=True)
            summary["gate_stage_records"].append(rec)
        summary["status"] = "DRY_RUN"

    summary["finished_at"] = _utc_now_iso()
    if summary["final_evaluation"] is None:
        summary["final_evaluation"] = evaluation
    out_path = os.path.join(run_dir, "factory_run.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    LOG.info("Run summary written → %s", out_path)
    summary["_summary_path"] = out_path
    return summary


def _collect_patch_paths(cfg: Dict[str, Any], ctx: Dict[str, str],
                         run_dir: str) -> Optional[Dict[str, str]]:
    """Locate the bridge manifest (if the bridge stage ran) for the summary text."""
    bridge_out = cfg.get("bridge", {}) or {}
    out_dir = bridge_out.get("out_dir", os.path.join("{run_dir}", "patches"))
    name = bridge_out.get("patch_name", "latest_patch")
    try:
        out_dir_res = resolve_token(str(out_dir), ctx, "bridge")
    except ConfigError:
        out_dir_res = os.path.join(run_dir, "patches")
    manifest = os.path.join(out_dir_res, f"{name}.json")
    npy = os.path.join(out_dir_res, f"{name}.npy")
    if os.path.isfile(manifest):
        return {"manifest_json": manifest, "steering_npy": npy}
    return None


# --------------------------------------------------------------------------- #
# Demo mode — proves the fallback loop end-to-end with stub stages (no repos)
# --------------------------------------------------------------------------- #

_DEMO_CROSSCODER = textwrap.dedent("""
    import json, os, sys
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "dfc_final.pt"), "wb").write(b"demo-stub-checkpoint")
    json.dump({"loss/recon": 0.01, "latents/dead_frac_S_A": 0.1},
              open(os.path.join(out, "final_metrics.json"), "w"))
    print("[demo-crosscoder] trained stub DFC; args:", " ".join(sys.argv[2:]))
""")

_DEMO_BRIDGE = textwrap.dedent("""
    import json, os, sys
    import numpy as np
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    vec = np.arange(16, dtype=np.float32) / 16.0
    np.save(os.path.join(out, "latest_patch.npy"), vec)
    json.dump({"schema": "circuit-bridge/1", "patch_id": "cb1-demo0000000000",
               "direction": "A2B", "shapes": {"steering": [16]}},
              open(os.path.join(out, "latest_patch.json"), "w"), indent=2)
    print("[demo-bridge] wrote stub patch to", out)
""")

_DEMO_CAUSAL = textwrap.dedent("""
    import json, os, sys
    results_dir, counter = sys.argv[1], sys.argv[2]
    os.makedirs(results_dir, exist_ok=True)
    n = 0
    if os.path.isfile(counter):
        n = int(open(counter).read() or 0)
    n += 1
    open(counter, "w").write(str(n))
    passed = n >= 2   # first attempt fails, second passes → proves fallback loop
    payload = {
        "milestone": "EQUYLAPTA E7.4 (demo)",
        "scientific_level": "LEVEL A: CAUSALLY_LOAD_BEARING" if passed
                            else "LEVEL B: REPRESENTATIONAL_ALIGNMENT_ONLY",
        "predeclared_agreement_threshold_pct": 90.0,
        "observed_peak_functional_agreement_pct": 93.1 if passed else 46.36,
        "threshold_met": passed,
        "source_causal_results": {"baseline": {"mean": 48.0}, "ablation": {"mean": 24.0},
                                  "causal_drop": 24.0},
    }
    with open(os.path.join(results_dir, "e7_4_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("[demo-causal] attempt", n, "->", "PASS" if passed else "THRESHOLD_NOT_MET")
""")

_DEMO_ROUTER = textwrap.dedent("""
    import json, os, sys
    text, out = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"routed": True, "expert": "technical", "text": text},
              open(out, "w"), indent=2)
    print("[demo-router] audit routing OK for:", text[:80], "...")
""")


def build_demo_config(demo_root: str) -> Dict[str, Any]:
    py = sys.executable or "python3"
    results_dir = os.path.join(demo_root, "causal-results")
    counter = os.path.join(demo_root, "attempt_counter.txt")
    dfc_out = os.path.join(demo_root, "dfc_run")
    patch_out = os.path.join(demo_root, "patches")
    receipt = os.path.join(demo_root, "audit_receipt.json")
    cfg = {
        "factory_name": "demo-factory",
        "workspace_root": demo_root,
        "runs_dir": os.path.join(demo_root, "runs"),
        "max_retries": 3,
        "stages": [
            {"id": "crosscoder", "name": "DFC training (stub)",
             "command": [py, "-c", _DEMO_CROSSCODER, dfc_out,
                         "--total-tokens", "1000000", "--alignment-coeff", "0.25"],
             "adjustments": [
                 {"flag": "--alignment-coeff", "op": "multiply", "value": 2.0, "max": 4.0},
                 {"flag": "--total-tokens", "op": "multiply", "value": 2},
             ],
             "timeout_sec": 120,
             "expect_files": [os.path.join(dfc_out, "dfc_final.pt")]},
            {"id": "bridge", "name": "Circuit bridge (stub)",
             "command": [py, "-c", _DEMO_BRIDGE, patch_out],
             "timeout_sec": 120,
             "expect_files": [os.path.join(patch_out, "latest_patch.npy")]},
            {"id": "causal_test", "name": "Causal functional effect test (stub)",
             "command": [py, "-c", _DEMO_CAUSAL, results_dir, counter],
             "timeout_sec": 120,
             "expect_files": [os.path.join(results_dir, "e7_4_results.json")]},
            {"id": "audit_router", "name": "Audit routing (stub)", "success_gate": True,
             "command": [py, "-c", _DEMO_ROUTER, "{summary_text}", receipt],
             "timeout_sec": 120,
             "expect_files": [receipt]},
        ],
        "evaluation": {
            "results_json": os.path.join(results_dir, "e7_4_results.json"),
            "pass_key": "threshold_met",
            "observed_key": "observed_peak_functional_agreement_pct",
            "threshold_key": "predeclared_agreement_threshold_pct",
        },
        "bridge": {"out_dir": patch_out, "patch_name": "latest_patch"},
        "_config_path": "<demo>",
    }
    return cfg


def run_demo() -> int:
    import tempfile
    demo_root = tempfile.mkdtemp(prefix="factory_orchestrator_demo_")
    LOG.info("Demo root: %s", demo_root)
    cfg = build_demo_config(demo_root)
    summary = run_factory(cfg, base_dir=demo_root)

    ok = (
        summary["status"] == "FACTORY_SUCCESS"
        and len(summary["attempts"]) == 2
        and summary["attempts"][0]["evaluation"]["gate"] == "THRESHOLD_NOT_MET"
        and summary["attempts"][1]["evaluation"]["gate"] == "PASS"
        and any(rec["status"] == "OK" for rec in summary["gate_stage_records"])
        and any(ch for rec in summary["attempts"][1]["stage_records"]
                for ch in rec.get("changes", []))
    )
    print("\n" + "=" * 72)
    print("FACTORY ORCHESTRATOR DEMO:", "PASS" if ok else "FAIL")
    print(f"  status          : {summary['status']}")
    print(f"  attempts        : {len(summary['attempts'])} "
          f"(1st={summary['attempts'][0]['evaluation']['gate']}, "
          f"2nd={summary['attempts'][1]['evaluation']['gate']})")
    print(f"  fallback tuning : {summary['attempts'][1]['stage_records'][0].get('changes')}")
    print(f"  run summary     : {summary.get('_summary_path')}")
    print("=" * 72)
    if not ok:
        LOG.error("Demo self-check failed — inspect %s", demo_root)
        return 1
    shutil.rmtree(demo_root, ignore_errors=True)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_factory.py",
        description="Master sequential pipeline + fallback runner for the Model "
                    "Editing & Governance Factory (crosscoder → bridge → causal "
                    "test → audit router).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         DEFAULT_CONFIG_NAME),
                    help="Factory configuration file (JSON; YAML if PyYAML installed).")
    ap.add_argument("--max-retries", type=int, default=None,
                    help="Override config max_retries (total attempts = retries + 1).")
    ap.add_argument("--start-stage", type=str, default=None,
                    help="Skip pipeline stages before this stage id (first attempt only).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve and print every stage command without executing.")
    ap.add_argument("--demo", action="store_true",
                    help="Run a fully self-contained demo of the fallback loop "
                         "(stub stages, temporary directory, no repos required).")
    ap.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    ap.add_argument("--version", action="version", version=f"run_factory {__version__}")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[factory] %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    try:
        if args.demo:
            return run_demo()
        cfg = load_config(args.config)
        cfg["_config_path"] = args.config
        base_dir = os.path.dirname(os.path.abspath(args.config))
        summary = run_factory(cfg, base_dir=base_dir, dry_run=args.dry_run,
                              max_retries=args.max_retries, start_stage=args.start_stage)
        status = summary["status"]
        print("\n" + "=" * 72)
        print(f"FACTORY RUN COMPLETE: {status}")
        print(f"  run_id : {summary['run_id']}")
        print(f"  run_dir: {summary['run_dir']}")
        print("=" * 72)
        return 0 if status in ("FACTORY_SUCCESS", "DRY_RUN") else 1
    except ConfigError as exc:
        LOG.error("CONFIG ERROR: %s", exc)
        return 2
    except FactoryError as exc:
        LOG.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOG.error("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Unexpected failure: %s", exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
