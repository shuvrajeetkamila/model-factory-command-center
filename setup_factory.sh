#!/usr/bin/env bash
# =============================================================================
# setup_factory.sh — Master Workspace Installer
# Model Editing & Governance Factory · Repository #13 (Command Center)
# -----------------------------------------------------------------------------
# One-click automated installer that:
#   1. Verifies system prerequisites (git, python3) with friendly error advice.
#   2. Builds the neural_factory_workspace/ parent directory.
#   3. Clones all 12 shuvrajeetkamila factory repositories as siblings.
#   4. Creates the local conveyor-belt pipeline exchange directories.
#   5. Seeds a default factory_config.json if one is not present.
#   6. Runs the out-of-the-box demo verification sweep and prints a summary.
#
# Usage:
#   bash setup_factory.sh [--with-deps] [--no-tests]
#
#   --with-deps   also install python requirements (numpy / optional pyyaml)
#                 with pip, if pip is available on this machine.
#   --no-tests    assemble the workspace but skip the demo verification sweep.
#
# Dual-licensed under the MIT License OR the Apache License 2.0, at your option.
# Copyright 2026 Shuvrajeet Kamila.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Error trap — report the failing line, then exit cleanly.
# ---------------------------------------------------------------------------
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]:-$0}")"
trap 'echo "[setup] ERROR: command failed at line $LINENO (exit $?)."' ERR

# ---------------------------------------------------------------------------
# Terminal cosmetics (plain ASCII, safe on all terminals).
# ---------------------------------------------------------------------------
rule()   { printf '%s\n' "============================================================================="; }
banner() { rule; echo "[setup] $1"; rule; }
info()   { echo "[setup]   $1"; }
ok()     { echo "[setup]   [OK] $1"; }
warn()   { echo "[setup]   [WARN] $1"; }
fail()   { echo "[setup]   [FAIL] $1"; }

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
WITH_DEPS=0
RUN_TESTS=1
for arg in "$@"; do
    case "$arg" in
        --with-deps) WITH_DEPS=1 ;;
        --no-tests)  RUN_TESTS=0 ;;
        -h|--help)
            sed -n '7,20p' "${BASH_SOURCE[0]:-$0}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "[setup] ERROR: unknown option '$arg' (try: bash $SCRIPT_NAME --help)"
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 0. PREREQUISITE CHECKS
# ---------------------------------------------------------------------------
banner "PHASE 0 · PREREQUISITE VERIFICATION"

if ! command -v git >/dev/null 2>&1; then
    fail "git is not installed (or not on PATH)."
    echo
    echo "    How to fix it:"
    echo "      Ubuntu/Debian : sudo apt-get update && sudo apt-get install -y git"
    echo "      macOS         : xcode-select --install"
    echo "      Windows       : https://git-scm.com/downloads"
    echo
    exit 10
fi
ok "git found: $(git --version)"

if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 is not installed (or not on PATH)."
    echo
    echo "    How to fix it:"
    echo "      Ubuntu/Debian : sudo apt-get update && sudo apt-get install -y python3"
    echo "      macOS         : brew install python3   (or: xcode-select --install)"
    echo "      Windows       : https://www.python.org/downloads/  (tick 'Add to PATH')"
    echo
    exit 11
fi
ok "python3 found: $(python3 --version 2>&1)"

PY_NUMPY_OK=0
if python3 -c "import numpy" >/dev/null 2>&1; then
    PY_NUMPY_OK=1
    ok "numpy found: $(python3 -c 'import numpy; print(numpy.__version__)' 2>&1)"
else
    warn "numpy is NOT installed — the vault/bridge demos need it."
fi

# ---------------------------------------------------------------------------
# 1. WORKSPACE INITIALIZATION
# ---------------------------------------------------------------------------
banner "PHASE 1 · WORKSPACE INITIALIZATION"

# Decide which directory is the workspace:
#   a) already inside a folder named neural_factory_workspace  -> use it
#   b) the script's parent folder is neural_factory_workspace  -> use it
#   c) otherwise create ./neural_factory_workspace and use that
WORKSPACE=""
case "$PWD" in
    */neural_factory_workspace) WORKSPACE="$PWD" ;;
    *)
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
        if [ "$(basename "$(dirname "$SCRIPT_DIR")")" = "neural_factory_workspace" ]; then
            WORKSPACE="$(dirname "$SCRIPT_DIR")"
        else
            WORKSPACE="$PWD/neural_factory_workspace"
        fi
        ;;
esac

mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
ok "workspace directory: $WORKSPACE"

# ---------------------------------------------------------------------------
# 2. THE 12-REPOSITORY CLONE SEQUENCE (all links verified live, HTTP 200)
# ---------------------------------------------------------------------------
banner "PHASE 2 · CLONING THE 12 FACTORY REPOSITORIES"

GITHUB_ACCOUNT="shuvrajeetkamila"
REPOS=(
    "capability-genome"
    "pattern-genome"
    "rosetta-stone"
    "dedicated-feature-crosscoder"
    "circuit-bridge-plugin"
    "weight-graft-inference"
    "causal-functional-effect-test"
    "circuit-vault"
    "vault-db-registry"
    "rsd-framework"
    "ai-audit-router-toolkit"
    "factory-orchestrator"
)

CLONED=0
SKIPPED=0
FAILED=0
for repo in "${REPOS[@]}"; do
    url="https://github.com/${GITHUB_ACCOUNT}/${repo}"
    if [ -d "$repo" ]; then
        info "skip (already present): $repo"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    info "cloning $url ..."
    if git clone --depth 1 "$url" "$repo" >/dev/null 2>&1; then
        ok "cloned: $repo"
        CLONED=$((CLONED + 1))
    elif git clone "$url" "$repo" >/dev/null 2>&1; then   # fallback: full history
        ok "cloned (full history): $repo"
        CLONED=$((CLONED + 1))
    else
        fail "could not clone: $url"
        FAILED=$((FAILED + 1))
    fi
done
info "clone summary: $CLONED new, $SKIPPED skipped, $FAILED failed"

if [ "$FAILED" -gt 0 ]; then
    warn "some repositories could not be cloned — check your internet connection,"
    warn "then re-run: bash $SCRIPT_NAME   (existing folders are skipped safely)."
fi

# ---------------------------------------------------------------------------
# 3. LOCAL CONVEYOR-BELT PIPELINE CREATION
# ---------------------------------------------------------------------------
banner "PHASE 3 · BUILDING THE DATA-EXCHANGE PIPELINES"

PIPELINE_DIRS=(
    "fusionlab_data/patches"
    "vault_store/artifacts"
    "causal-functional-effect-test/results"
)
for d in "${PIPELINE_DIRS[@]}"; do
    mkdir -p "$d"
    ok "pipeline ready: $d"
done

# ---------------------------------------------------------------------------
# 4. OPTIONAL DEPENDENCY INSTALLATION
# ---------------------------------------------------------------------------
banner "PHASE 4 · PYTHON DEPENDENCIES"

if [ "$WITH_DEPS" -eq 1 ]; then
    if python3 -m pip --version >/dev/null 2>&1; then
        info "installing numpy (and optional pyyaml) via pip ..."
        if python3 -m pip install --user "numpy>=1.24.0" "PyYAML>=6.0"; then
            ok "dependencies installed"
            PY_NUMPY_OK=1
        else
            warn "pip install failed — install numpy manually:  python3 -m pip install numpy"
        fi
    else
        warn "pip is not available on this machine — skipping dependency installation."
        warn "install numpy manually if the vault/bridge demos fail."
    fi
else
    info "skipped (default). Re-run with --with-deps to install:  bash $SCRIPT_NAME --with-deps"
    info "runtime needs only: numpy>=1.24.0   (PyYAML is optional)"
fi

# ---------------------------------------------------------------------------
# 5. CONFIGURATION SEEDING
# ---------------------------------------------------------------------------
banner "PHASE 5 · CONFIGURATION SEEDING"

CONFIG_FILE="factory_config.json"
if [ -f "$CONFIG_FILE" ]; then
    ok "existing $CONFIG_FILE kept untouched (no overwrite)."
else
    info "seeding default $CONFIG_FILE ..."
    cat > "$CONFIG_FILE" <<'FACTORY_CONFIG_EOF'
{
  "_comment": [
    "Default factory configuration — assumes this plugin folder and the nine",
    "core repositories are siblings under one workspace root:",
    "  workspace/",
    "    dedicated-feature-crosscoder/   causal-functional-effect-test/",
    "    ai-audit-router-toolkit/        circuit-vault/",
    "    circuit-bridge-plugin/          vault-db-registry/",
    "    factory-orchestrator/  <-- this repo (config lives here)",
    "Adjust 'workspace_root' and 'repos' if your layout differs.",
    "Placeholders available in commands/cwd/paths: {python} {workspace}",
    "{plugin_dir} {run_id} {run_dir} {attempt} {repo_<name>} and, in",
    "success-gate stages only, {summary_text} {results_json} {patch_npy} {patch_json}."
  ],
  "factory_name": "model-editing-governance-factory",
  "workspace_root": ".",
  "runs_dir": "runs/factory",
  "max_retries": 3,
  "restart_stage": "crosscoder",
  "repos": {
    "crosscoder": "dedicated-feature-crosscoder",
    "causal_test": "causal-functional-effect-test",
    "audit_router": "ai-audit-router-toolkit",
    "circuit_vault": "circuit-vault",
    "bridge_plugin": "circuit-bridge-plugin",
    "vault_registry": "vault-db-registry"
  },
  "stages": [
    {
      "id": "crosscoder",
      "name": "Stage 1 — DFC feature isolation (synthetic profile)",
      "cwd": "{repo_crosscoder}",
      "command": [
        "{python}",
        "scripts/train_dfc.py",
        "--synthetic",
        "--synthetic-samples",
        "65536",
        "--total-tokens",
        "1000000",
        "--batch-tokens",
        "512",
        "--alignment-coeff",
        "0.5",
        "--n-latents-shared",
        "160",
        "--top-k-shared",
        "10",
        "--output-dir",
        "{run_dir}/dfc"
      ],
      "adjustments": [
        {
          "flag": "--alignment-coeff",
          "op": "multiply",
          "value": 1.5,
          "max": 4.0
        },
        {
          "flag": "--total-tokens",
          "op": "multiply",
          "value": 2
        },
        {
          "flag": "--synthetic-samples",
          "op": "multiply",
          "value": 1.5,
          "max": 1048576
        }
      ],
      "timeout_sec": 3600,
      "expect_files": [
        "{run_dir}/dfc/dfc_final.pt",
        "{run_dir}/dfc/final_metrics.json"
      ]
    },
    {
      "id": "bridge",
      "name": "Stage 2 — circuit bridge (crosscoder → graft patch)",
      "cwd": "{repo_bridge_plugin}",
      "command": [
        "{python}",
        "circuit_bridge.py",
        "--checkpoint",
        "{run_dir}/dfc/dfc_final.pt",
        "--direction",
        "A2B",
        "--scale",
        "1.0",
        "--top-n",
        "4",
        "--out-dir",
        "{run_dir}/patches",
        "--patch-name",
        "latest_patch"
      ],
      "timeout_sec": 900,
      "expect_files": [
        "{run_dir}/patches/latest_patch.npy",
        "{run_dir}/patches/latest_patch.json"
      ]
    },
    {
      "id": "causal_test",
      "name": "Stage 3 — causal functional effect test (EQUYLAPTA E7.4)",
      "cwd": "{repo_causal_test}",
      "command": [
        "{python}",
        "demo/run_equylapta7_4.py"
      ],
      "timeout_sec": 1800,
      "expect_files": [
        "results/e7_4_results.json"
      ]
    },
    {
      "id": "vault_register",
      "name": "Success gate — index the validated patch into the vault registry",
      "success_gate": true,
      "enabled": false,
      "optional": true,
      "cwd": "{repo_vault_registry}",
      "command": [
        "{python}",
        "vault_db.py",
        "--db",
        "{run_dir}/vault_registry.db",
        "register",
        "--circuit-file",
        "{patch_npy}",
        "--source-model",
        "dfc-synthetic-A",
        "--target-model",
        "dfc-synthetic-B",
        "--layer-indices",
        "1",
        "--feature-type",
        "STEERING_VECTOR",
        "--capability-tags",
        "crosscoder-isolated,shared-I_S",
        "--test-results",
        "{results_json}",
        "--status",
        "VERIFIED"
      ],
      "timeout_sec": 300
    },
    {
      "id": "audit_router",
      "name": "Success gate — route the validated transplant through the audit pipeline",
      "success_gate": true,
      "cwd": "{repo_audit_router}",
      "command": [
        "{python}",
        "text_router.py",
        "route",
        "--config",
        "examples/router_config.json",
        "--text",
        "{summary_text}"
      ],
      "timeout_sec": 300
    }
  ],
  "evaluation": {
    "_comment": "results_json lives in the causal-test repo (it writes there itself); we only READ it.",
    "results_json": "{repo_causal_test}/results/e7_4_results.json",
    "status_key": null,
    "pass_key": "threshold_met",
    "observed_key": "observed_peak_functional_agreement_pct",
    "threshold_key": "predeclared_agreement_threshold_pct",
    "metric_paths": {
      "observed_agreement_pct": "observed_peak_functional_agreement_pct",
      "predeclared_threshold_pct": "predeclared_agreement_threshold_pct",
      "causal_drop_pp": "source_causal_results.causal_drop",
      "scientific_level": "scientific_level",
      "milestone": "milestone"
    }
  },
  "bridge": {
    "out_dir": "{run_dir}/patches",
    "patch_name": "latest_patch"
  }
}
FACTORY_CONFIG_EOF
    ok "seeded default $CONFIG_FILE"
fi

# ---------------------------------------------------------------------------
# 6. OUT-OF-THE-BOX DEMO VERIFICATION SWEEP
# ---------------------------------------------------------------------------
banner "PHASE 6 · DEMO VERIFICATION SWEEP"

VAULT_RC=0
FACTORY_RC=0
BRAIN_RC=0

if [ "$RUN_TESTS" -eq 1 ]; then
    echo
    info "test 1/3: SQLite indexing memory layer  (vault-db-registry)"
    if python3 vault-db-registry/vault_db.py demo; then
        ok "vault_db.py demo: PASS"
    else
        VAULT_RC=$?
        fail "vault_db.py demo: FAIL (exit $VAULT_RC)"
    fi

    echo
    info "test 2/3: automated pipeline retry loops  (factory-orchestrator)"
    if python3 factory-orchestrator/run_factory.py --demo; then
        ok "run_factory.py --demo: PASS"
    else
        FACTORY_RC=$?
        fail "run_factory.py --demo: FAIL (exit $FACTORY_RC)"
    fi

    echo
    if [ -f "model-factory-command-center/run_factory.py" ]; then
        info "test 3/3: command-center brain self-test  (model-factory-command-center)"
        if python3 model-factory-command-center/run_factory.py --demo; then
            ok "command-center run_factory.py --demo: PASS"
        else
            BRAIN_RC=$?
            fail "command-center run_factory.py --demo: FAIL (exit $BRAIN_RC)"
        fi
    else
        info "test 3/3: skipped — model-factory-command-center/ not inside this workspace."
        info "          (that is fine: tests 1 and 2 cover the factory engines)"
    fi
else
    warn "demo sweep skipped (--no-tests)."
fi

# ---------------------------------------------------------------------------
# FINAL SUMMARY REPORT
# ---------------------------------------------------------------------------
echo
banner "SETUP SUMMARY"
info "workspace   : $WORKSPACE"
info "repositories: 12 requested · $CLONED cloned · $SKIPPED already present · $FAILED failed"
info "pipelines   : fusionlab_data/patches · vault_store/artifacts · causal-functional-effect-test/results"
info "config      : $CONFIG_FILE"
if [ "$RUN_TESTS" -eq 1 ]; then
    info "demo sweep  : vault_db=$([ "$VAULT_RC" -eq 0 ] && echo PASS || echo FAIL) · factory=$([ "$FACTORY_RC" -eq 0 ] && echo PASS || echo FAIL) · brain=$([ "$BRAIN_RC" -eq 0 ] && echo PASS || echo SKIP/FAIL)"
fi
rule

if [ "$FAILED" -gt 0 ] || [ "$VAULT_RC" -ne 0 ] || [ "$FACTORY_RC" -ne 0 ] || [ "$BRAIN_RC" -ne 0 ]; then
    echo "[setup] Finished WITH WARNINGS — see the [FAIL] lines above."
    exit 1
fi

echo "[setup] Neural Factory workspace is READY. Open the Command Center README for usage."
exit 0
