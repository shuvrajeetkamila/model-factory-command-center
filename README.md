# 🕸️ The Automated Model Splicing & Alignment Governance Factory

### Central Command Center (`model-factory-command-center`) — Repository #13

Welcome to the central control dashboard for an automated, architecture-agnostic
**Neural Factory for Locate-and-Edit Model Surgery**. This ecosystem completely
rejects standard industry brute-force methods like Fine-Tuning, LoRA, Knowledge
Distillation, and Model Merging. Instead, it treats transformer weight matrices
as physical neural anatomy — mechanically slicing out targeted cognitive circuits
from a source model and grafting them directly into a host model **in under 3
minutes**, running entirely via **NumPy on a single CPU core**.

This repository is the master control console that connects the other **12**
repositories of the ecosystem: topology map, one-click workspace installer, the
self-correcting orchestration brain, and the baseline pipeline configuration —
all in one GitHub-ready page.

---

## 🗺️ Master Factory Topology & Pipeline Flow

The complete total-system data flowchart. A model enters at the top; a verified,
governed, deployable graft leaves at the bottom:

```
                     ┌─────────────────────────────┐
                     │   [ CONFIGURATION MANIFEST ]│   factory_config.json
                     │   run_factory.py (THE BRAIN)│   max_retries = 3
                     └──────────────┬──────────────┘
                                    │
                                    ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ PHASE 1 · 🧬 SCANNING   ──► capability-genome + pattern-genome           │
   │            read the anatomy: capabilities, patterns, interventions       │
   └──────────────┬───────────────────────────────────────────────────────────┘
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ PHASE 2 · 🔍 DIFFING    ──► dedicated-feature-crosscoder ──► rosetta-stone│
   │            isolate the feature direction (Dual Feature Crosscoder)       │
   └──────────────┬───────────────────────────────────────────────────────────┘
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ PHASE 3 · 🔀 MIDDLEWARE ──► circuit_bridge.py (The Surgical Scalpel)      │
   │            trace feature indices → weight rows/cols → graft-ready patch  │
   └──────────────┬───────────────────────────────────────────────────────────┘
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ PHASE 4 · 🛠️ SURGERY    ──► weight-graft-inference                        │
   │            splice the circuit into the host model (rank-1 weight graft)  │
   └──────────────┬───────────────────────────────────────────────────────────┘
                  ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ PHASE 5 · ⚖️ AUDITING   ──► causal-functional-effect-test                 │
   │            measure functional agreement + causal dependence              │
   └──────────────┬───────────────────────────────────────────────────────────┘
                  ▼
         ┌────────┴────────────────────────────────┐
         ▼ (If THRESHOLD_NOT_MET)                  ▼ (If PASSED)
   run_factory.py                            rsd-framework
   (Auto-Tweak & Retry:                      (Recursive Stability Check)
    multiply/add config flags)                     │
         │                                        ▼
         │              ┌────────────────────────────────────────────┐
         │              │ PHASE 6 · 🗄️ STORAGE & DEPLOYMENT           │
         │              │  circuit-vault + vault_db.py (SQLite Memory)│
         │              │  ──► ai-audit-router-toolkit ──► DEPLOY    │
         │              └───────────────────┬────────────────────────┘
         └────◄── [ RETRY LOOP ] ◄──────────┘
```

**The 5 core sequential operational layers:** Scanning → Diffing → Middleware →
Surgery → Auditing — followed by Storage & Deployment. The retry loop makes the
whole line self-correcting: a failed surgery is retuned and re-run automatically,
up to a maximum retry limit of 3.

---

## ⚡ NumPy CPU Surgical Layout Benchmarks

The factory's unique value proposition, stated plainly:

| Axis | Legacy industry method | This factory |
|---|---|---|
| Method | Fine-tuning / LoRA / distillation / merging | **Locate-and-Edit model surgery** |
| Compute | Multi-GPU clusters, hours-to-days | **1 CPU core, minutes** |
| Runtime | Gradient loops over billions of tokens | **Full surgery sweep in under 3 minutes** |
| Dependencies | Heavy training frameworks (PyTorch/JAX stacks) | **Pure NumPy** (+ `torch` only to read `.pt` checkpoints) |
| Model layout | Arbitrary | **2–8 layer synthetic transformers** (`d_model = 64`, `vocab = 141`) |
| Mechanism | Update every weight statistically | **Mechanically slice one circuit, graft it** |

Surgical layout constants (verified from the repository artifacts):

* Model family: `fusionlab_data/models/*-{2,4,6,8}L.npz` — 2, 4, 6 and 8 layer
  variants of the synthetic transformer, hidden width 64, vocabulary 141.
* Benchmark target: a complete Crosscoder → Bridge → Causal-Test sweep finishes
  **in under 3 minutes on a single CPU core**, with no GPU present or required.
* Grunt work avoided: a rank-1 weight graft is one outer product of two NumPy
  vectors — there is no optimizer state, no backward pass, no fine-tuning loop.

Current scientific milestone in the ecosystem: **EQUYLAPTA E7.4 — LEVEL B
(Representational Alignment Only)**, predeclared agreement threshold 90.0%,
reported honestly in
[`causal-functional-effect-test/results/e7_4_results.json`](https://github.com/shuvrajeetkamila/causal-functional-effect-test).

---

## 🗃️ The 12-Repository Master Links Registry

Our automated factory line is built out of 12 distinct modular instruments.
Click any link to visit the independent codebase repository.

### 🧬 Layer 1 · The Genomics Layer (Reading & Mapping)

| Repository | Role |
|---|---|
| [shuvrajeetkamila/capability-genome](https://github.com/shuvrajeetkamila/capability-genome) | Records capability-level genome entries (`GenomeRecord`: capability, intervention, delta, status) describing what a circuit *does*. |
| [shuvrajeetkamila/pattern-genome](https://github.com/shuvrajeetkamila/pattern-genome) | Pattern-level genome registries mapping recurring structural motifs across models. |
| [shuvrajeetkamila/rosetta-stone](https://github.com/shuvrajeetkamila/rosetta-stone) | The translation dictionary between the two genomes — feature vocabulary alignment. |

### 🔬 Layer 2 · The Diffing & Middleware Layer (Isolation & Extraction)

| Repository | Role |
|---|---|
| [shuvrajeetkamila/dedicated-feature-crosscoder](https://github.com/shuvrajeetkamila/dedicated-feature-crosscoder) | Dual Feature Crosscoder (DFC): trains `enc_S_{A,B}` / `dec_S_{A,B}` dictionaries and isolates the shared feature direction `I_S`. |
| [shuvrajeetkamila/circuit-bridge-plugin](https://github.com/shuvrajeetkamila/circuit-bridge-plugin) | The Surgical Scalpel middleware: parses DFC outputs, traces indices to weight rows/cols, exports graft-ready patches. |

### ✂️ Layer 3 · The Surgical Layer (Transplanting & Grafting)

| Repository | Role |
|---|---|
| [shuvrajeetkamila/weight-graft-inference](https://github.com/shuvrajeetkamila/weight-graft-inference) | The host-side runtime: `ActivationPatcher`, `apply_weight_graft(op="rank1", u, v, alpha)`, `attach_from_vault`. |
| [shuvrajeetkamila/causal-functional-effect-test](https://github.com/shuvrajeetkamila/causal-functional-effect-test) | The measuring stick: baseline vs ablation vs causal drop, functional-agreement gate (`threshold_met`). |

### 🛡️ Layer 4 · The Vault & Governance Layer (Memory & Deployment)

| Repository | Role |
|---|---|
| [shuvrajeetkamila/circuit-vault](https://github.com/shuvrajeetkamila/circuit-vault) | The cold vault: `cv1-*` circuit artifacts (`.npy`) with canonical SHA-256 content hashing. |
| [shuvrajeetkamila/vault-db-registry](https://github.com/shuvrajeetkamila/vault-db-registry) | The intelligent SQLite memory index (`vr1-*`): searchable registry of circuits, tags and scores. |
| [shuvrajeetkamila/rsd-framework](https://github.com/shuvrajeetkamila/rsd-framework) | Recursive Self-Distillation framework — long-horizon stability checks on the grafted model. |
| [shuvrajeetkamila/ai-audit-router-toolkit](https://github.com/shuvrajeetkamila/ai-audit-router-toolkit) | The outer governance layer: claim validation, license registry, audit-signature routing to production. |
| [shuvrajeetkamila/factory-orchestrator](https://github.com/shuvrajeetkamila/factory-orchestrator) | The standalone orchestration engine package: sequential loop + hyperparameter fallback tuning. |

> ✅ **Link health check:** all 12 URLs above were verified live (HTTP 200) while
> building this release.

---

## 📦 System Plugins — Descriptive Reference

> ### 🔀 `circuit-bridge-plugin` — The Surgical Scalpel
> A standalone NumPy middleware engine that consumes dedicated-feature-crosscoder
> checkpoint outputs (`dfc_final.pt` or state `.npz`), validates the 2–8 layer
> graft contract, replays JumpReLU/TopK activations to select isolated features,
> traces feature indices back to concrete weight rows and columns, extracts the
> circuit slice, and reformats it into the weight-graft-inference patch shape —
> exporting `latest_patch.{npy,pt}` plus a full manifest with shape/layer logs.

> ### 🗄️ `vault-db-registry` — The Intelligent Memory
> A self-contained SQLite metadata indexer (`vault_registry.db`, no external
> database server) that turns a circuit-vault store plus loose bridge patches
> into a searchable library. Schema per the engineering specification
> (`circuit_id, file_path, source_model, target_model, layer_indices,
> capability_tags, functional_agreement_score, causal_dependence_score,
> timestamp`), with `register_circuit()` / `search_vault()` APIs and a CLI
> answering queries like *"causal score > 0.85 tagged 'induction'"* in
> milliseconds. Integrity is tamper-evident via canonical content hashing.

> ### 🧠 `factory-orchestrator` — The Self-Correcting Brain
> A zero-dependency subprocess orchestrator that runs the pipeline stages
> sequentially (Crosscoder → Bridge → Causal-Test), streams every stage log
> live, intercepts the boolean `threshold_met` from the causal test's results
> file, and on `THRESHOLD_NOT_MET` performs algebraic config flag rewrites
> (`multiply`/`add`, with safety clamps) and re-runs the surgery — up to a
> maximum retry limit of 3 — before firing the vault and audit success gates.

> ### 🎛️ `model-factory-command-center` — This Repository
> The administrative dashboard: master topology map (this page), the one-click
> workspace installer (`setup_factory.sh`), the central nervous system brain
> (`run_factory.py`), and the baseline relative-path configuration
> (`factory_config.json`) wiring the five sequential pipeline stages together.

---

## 🧰 Dashboard File Map

```
model-factory-command-center/        (Repository #13 — unzip target)
├── README.md                ← this high-impact homepage & 12-repo map
├── run_factory.py           ← the central nervous system brain (stdlib only)
├── factory_config.json      ← baseline 5-stage relative workspace configuration
├── setup_factory.sh         ← one-click master workspace installer
├── requirements.txt         ← runtime requirements (numpy; optional pyyaml)
├── .gitignore               ← isolates bytecode, .env, runtime *.db/*.sqlite
├── LICENSE                  ← combined dual-license notice (MIT OR Apache-2.0)
├── LICENSE-MIT              ← complete MIT License text
└── LICENSE-APACHE           ← complete Apache License 2.0 text
```

---

## 🚀 One-Click Workspace Setup (`setup_factory.sh`)

Turn a brand-new computer into a fully functional Neural Factory:

```bash
# 1. Get this repository (after uploading the zip to GitHub):
git clone https://github.com/shuvrajeetkamila/model-factory-command-center
cd model-factory-command-center

# 2. Run the master installer:
bash setup_factory.sh

# Optional: also pip-install numpy + pyyaml automatically:
bash setup_factory.sh --with-deps
```

The script verifies `git` and `python3` first (with friendly fix-it advice if
missing), then:

1. creates the clean parent workspace `neural_factory_workspace/` and enters it;
2. clones all **12** repositories from `shuvrajeetkamila` as siblings (existing
   folders are skipped safely, so it is re-runnable);
3. builds the conveyor-belt exchange directories:
   `fusionlab_data/patches/`, `vault_store/artifacts/`,
   `causal-functional-effect-test/results/`;
4. seeds a default `factory_config.json` if none exists;
5. runs the out-of-the-box verification sweep —
   `python3 vault-db-registry/vault_db.py demo` and
   `python3 factory-orchestrator/run_factory.py --demo` — and prints a final
   pass/fail summary report.

---

## ⚙️ Baseline Configuration (`factory_config.json`)

Relative workspace path profile mapping the **five sequential pipeline stages**
and the adjustment tuning limits:

| # | Stage | Command | Fallback tuning on `THRESHOLD_NOT_MET` |
|---|---|---|---|
| 1 | `crosscoder` | `scripts/train_dfc.py --synthetic …` | `--alignment-coeff ×1.5 (max 4.0)` · `--total-tokens ×2` · `--synthetic-samples ×1.5 (max 1 048 576)` |
| 2 | `bridge` | `circuit_bridge.py --checkpoint … --out-dir …` | — (pure extraction) |
| 3 | `causal_test` | `demo/run_equylapta7_4.py` | — (the measuring stick) |
| 4 | `vault_register` *(success gate, optional)* | `vault_db.py register --circuit-file …` | fired only after a PASS |
| 5 | `audit_router` *(success gate)* | `text_router.py route --text …` | fired only after a PASS |

Key settings: `workspace_root: ".."` (the parent folder of this repo, i.e. the
workspace holding the 12 siblings), `max_retries: 3`,
`restart_stage: "crosscoder"`, and the evaluation contract
(`pass_key: "threshold_met"`, `observed_key: "observed_peak_functional_agreement_pct"`,
`threshold_key: "predeclared_agreement_threshold_pct"`).

---

## 🧠 The Brain in Action (`run_factory.py`)

Standard Python libraries only — no external dependencies.

```bash
# Full factory run:
python3 run_factory.py --config factory_config.json

# Inspect every resolved command without executing anything:
python3 run_factory.py --dry-run

# Self-contained proof of the fallback loop (no repos needed):
python3 run_factory.py --demo

# Override the retry budget / resume mid-pipeline:
python3 run_factory.py --config factory_config.json --max-retries 5 --start-stage bridge
```

Programmatic API:

```python
from run_factory import run_factory, load_config, evaluate_results

cfg = load_config("factory_config.json")
cfg["_config_path"] = "factory_config.json"
summary = run_factory(cfg, base_dir=".", max_retries=3)
print(summary["status"], summary["run_dir"])
```

Every run leaves a complete audit trail in `runs/factory/<run_id>/`:
`factory_run.json` (attempt history, tuning changes, gate evaluations), live
stage logs, and per-attempt config snapshots.

---

## 🔒 Licensing

Every component of this ecosystem is **dual-licensed**:

* **MIT License** — see [`LICENSE-MIT`](LICENSE-MIT)
* **Apache License 2.0** — see [`LICENSE-APACHE`](LICENSE-APACHE)
* Combined notice — see [`LICENSE`](LICENSE)

You may choose either license, at your option.
**Copyright 2026 Shuvrajeet Kamila.**
