# SetupX
<p align="center">
  <img width="300" src="https://github.com/user-attachments/assets/c6684028-50bd-4205-828f-5e22d3b30f73" />
</p>


**Experience-Driven Automated Environment Configuration with LLM Agents**


## Overview

**SetupX** is an LLM-powered multi-agent system that automatically configures software repository build environments inside Docker containers. Given a GitHub repository URL, SetupX spins up a container, iteratively installs dependencies, resolves errors, and configures the environment until the project's test suite can be executed successfully.

Unlike prior approaches that start each configuration from scratch, SetupX features three mutually reinforcing mechanisms:

- **XPU (eXPerience Unit) Knowledge System** — A vector database (PostgreSQL + pgvector) that stores transferable configuration experiences. Successful fixes are extracted, deduplicated, and reused across repositories via two-layer semantic retrieval.
- **Speculative Execution** — Docker container snapshots enable safe trial-and-rollback of past fixes, addressing the inherently irreversible nature of environment configuration.
- **Adversarial Verification** — A Prosecutor–Judge pipeline structurally separates configuration and verification roles, preventing self-confirmation bias.

## Architecture

SetupX orchestrates repository configuration through three sequential phases:

**Phase 1: Setup with In-Loop Verification**
- Speculative Setup Agent (ReAct loop):
  - Observe environment state
  - Retriever Agent → XPU two-layer retrieval
  - LLM decision → Action selection
  - Docker execution (with snapshot/rollback)
  - Verifier Agent → test suite verification

**Phase 2: Adversarial Verification**
- Prosecutor Agent → investigate & file charges
- Judge Agent → verify each charge independently
- Verdict: guilty / not_guilty

**Phase 3: Experience Extraction**
- Extract transferable XPU from agent trajectory
- Deduplicate & ingest into XPU library     


### Key Components

| Component | Description |
|-----------|-------------|
| **SetupAgent** | Main orchestrator. Runs a ReAct loop with 6 action types: `SHELL_COMMAND`, `TRY_XPU_SUGGESTION`, `SET_ENV`, `ROLLBACK_ENV`, `VERIFY`, `FINISH`. |
| **RetrieverAgent** | Sub-agent for XPU knowledge retrieval. Layer 1: vector coarse filtering (pgvector cosine similarity, top-N). Layer 2: LLM re-ranking for precise matching. Also performs delayed audit of previously used XPUs. |
| **VerifierAgent** | Read-only sub-agent that runs the project's test suite (`pytest`) and distinguishes setup-induced failures from inherent project issues. |
| **ProsecutorAgent** | Adversarial investigator. Has container access for evidence gathering, files charges with concrete evidence. |
| **JudgeAgent** | Independent adjudicator. Verifies each charge with 1–2 targeted commands. Renders final verdict. |
| **EnvironmentManager** | Docker container lifecycle management: create, execute, snapshot (`docker commit`), rollback (stack-based LIFO). |
| **XPU Vector Store** | PostgreSQL + pgvector backend. Stores embeddings via `text-embedding-3-small`, supports composite scoring with telemetry-based tier boosting. |

---

The rest of this README is the **minimum runnable distribution guide**: how to
install, configure, and run SetupX on one repository at a time, plus the
600-entry warm XPU store used in the paper (`data/xpu_warm.jsonl`).

---

## 1. Requirements

- Python 3.10+
- A working Docker daemon on the host 
- Network access to one OpenAI-compatible LLM endpoint


---

## 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Configure

```bash
cp .env.example .env
```

Open `.env` and set, at minimum:

```
OPENAI_API_KEY=<your key>
OPENAI_BASE_URL=<your endpoint>     # e.g. https://api.openai.com/v1
OPENAI_MODEL=<chat model name>      # e.g. gpt-4o-mini
```

Every other field has a usable default (Docker base `python:3.10`, XPU
experience store disabled, dummy ARK placeholders). Override only what you
need.

---

## 4. Run

### Single-repo mode (default)

```bash
./scripts/run.sh <git-repo-url>
```

Equivalent direct invocation:

```bash
python -m src.main <git-repo-url>
```

The agent will:

1. Spin up a fresh Docker container (`python:3.10` by default).
2. Iteratively reason → execute shell commands → roll back on failure, until
   `VERIFY` passes or the timeout fires.
3. Run a Phase 2 review (prosecutor + judge) on the verifier transcript.
4. Write the result to `log/result_<repo-slug>_<timestamp>.json` and a full
   trace to `log/<timestamp>.log`.

### Multi-repo / non-atomic family mode

When a target repo depends on a host or component repos in the same family,
pass the family roster in `--meta-json`:

```bash
python -m src.main <target-repo-url> \
    --meta-json '{"repository":"<org/repo>",
                  "primary_repos":["<org/host>","<org/repo>"],
                  "component_repos":["<org/plugin-a>","<org/plugin-b>"]}'
```

In this mode Phase 2 is skipped by design (the task is environment setup
only, not PR-level review).

### Useful flags

| Flag | Purpose | Default |
|---|---|---|
| `--no-xpu` | Force-disable the experience store for this run | off |
| `--phase1-timeout <sec>` | Phase 1 wall-clock budget | `1800` |
| `--max-steps <n>` | Hard upper bound on agent iterations | `9999` |
| `--output-dir <dir>` | Where to write the result JSON | `log` |
| `--meta-json <json>` | Multi-repo family roster (see above) | unset |

### Keep the container after the run

```bash
OURSYS_KEEP_CONTAINER=1 ./scripts/run.sh <repo-url>
```

The container is left intact so you can `docker exec -it <id> bash` and
inspect what the agent built.

---

## 5. Output

```
log/
├── result_<repo-slug>_<timestamp>.json   # summary: phase1 + phase2 verdicts
└── <timestamp>.log                       # full trace
```

`result_*.json` schema (abridged):

```jsonc
{
  "repo": "<git-repo-url>",
  "phase1": {
    "completed": true,
    "step_count": 14,
    "reason": "FINISH triggered after VERIFY passed"
  },
  "phase2": {
    "success": true,
    "reason": "judge ruled in favor"
  },
  "container_id": "<docker-id>"
}
```

---

## 6. Optional — populate the experience store

The agent can consult an offline experience knowledge base (XPU) during
Phase 1. **It is off by default** and
the agent works fine without it — if you do not need it, skip this section.

### 6.1 Bring up Postgres + pgvector

Any Postgres ≥ 14 with the `pgvector` extension works. The fastest way:

```bash
docker run -d --name xpu-pg \
    -e POSTGRES_PASSWORD=changeme \
    -p 5433:5432 \
    pgvector/pgvector:pg16
```

Then point `.env` at it:

```
XPU_ENABLED=true
XPU_VECTOR_ENABLED=true
XPU_TABLE=xpu_entries
dns=postgresql://postgres:changeme@localhost:5433/postgres

EMBEDDING_API_KEY=<key for an OpenAI-compatible embedding endpoint>
EMBEDDING_BASE_URL=<endpoint base URL>
EMBEDDING_MODEL=text-embedding-3-small
```

The table and IVFFlat index are created automatically on first connection.
You do not need to run a separate migration.

### 6.2 Import a JSONL of experience entries

Each line is one entry with this shape:

```jsonc
{
  "id":         "unique-string",
  "signals":    {
    "applicability": { /* match conditions: language, OS, python, tools */ },
    "regex":         ["..."],
    "keywords":     ["..."],
    "situation_triggers": ["..."]
  },
  "advice_nl":  ["natural-language hints"],
  "atoms":      [{"name": "shell", "args": {"cmd": "..."}}],
  "telemetry":  {"hits": 0, "successes": 0, "failures": 0}
}
```

Bulk-import:

```bash
python scripts/import_xpu_jsonl.py path/to/entries.jsonl
python scripts/import_xpu_jsonl.py path/to/entries.jsonl --clear   # truncate first
```

### 6.3 Reproduce the paper's warm XPU store

`data/xpu_warm.jsonl` ships the 600-entry warm XPU store used in the paper's
with-xpu experiments. Each line follows the 5-field schema above; telemetry
counters (`hits` / `successes` / `failures`) are the real values accumulated
during the experiments, not zeroed out.

To reproduce the warm store from scratch:

```bash
python scripts/import_xpu_jsonl.py data/xpu_warm.jsonl --clear
```

This embeds each entry's text via `EMBEDDING_MODEL` and upserts into
`XPU_TABLE`. `--clear` ensures the resulting table contains exactly the 600
warm entries with no leftovers from prior runs. Already-present `id`s are
updated in place rather than duplicated.

### 6.4 Maintenance helpers

```bash
python scripts/export_xpu.py -o backup.jsonl --full   # dump table to JSONL
python scripts/reset_db.py                            # drop the XPU table
```

`scripts/inflate_xpu_db.py` is an optional research utility that synthesises
noise entries on top of the existing table (context perturbation,
cross-grafting, generalisation blur, cross-language drift), for stress-testing
retrieval against a partially-noisy store. It is not needed for normal use.

```bash
python scripts/inflate_xpu_db.py --target 2000       # synthesise noise up to N rows
```

All synthesised rows carry id prefixes `noise_ctx_*`, `noise_graft_*`,
`noise_vague_*`, `noise_lang_*`, and can be removed with a single
`DELETE FROM xpu_entries WHERE id LIKE 'noise_%';`.

---

## 7. Layout

```
.
├── .env.example          # config template
├── README.md             # this file
├── requirements.txt      # python deps
├── scripts/
│   ├── run.sh                # one-line wrapper around `python -m src.main`
│   ├── import_xpu_jsonl.py   # bulk-import experiences from JSONL
│   ├── export_xpu.py         # dump experiences to JSONL
│   ├── reset_db.py           # drop the XPU table
│   └── inflate_xpu_db.py     # (optional) synthesise noise entries for stress testing
├── src/
│   ├── main.py                # CLI entry point; orchestrates the 3 phases
│   ├── agent.py               # Phase 1 main loop (speculative exec + rollback)
│   ├── llm_engine.py          # LLM call + JSON action parsing
│   ├── retriever_agent.py     # Two-tier experience retrieval (vector + LLM rerank)
│   ├── environment_manager.py # Docker container lifecycle + snapshots
│   ├── verifier_agent.py      # Phase 1 verify gate
│   ├── prosecutor_agent.py    # Phase 2 prosecutor
│   ├── judge_agent.py         # Phase 2 judge
│   ├── task_meta.py           # Multi-repo family meta rendering
│   ├── models.py / config.py / logger.py / xpu_client.py
│   └── xpu/                   # Experience-store extraction & vector index
├── experiment/                # Cross-system comparison harness
│   ├── run_cli_benchmark.py      # Batch driver; dispatches per-tool launchers,
│   │                             #   then takes over the container and runs the
│   │                             #   project's VerifierAgent + Phase 2 judge.
│   ├── {claude_code,opencode,qwen_code}/
│   │                             # Per-CLI Docker wrappers (Dockerfile + Python
│   │                             #   launcher + TS in-container runner).
│   ├── ours/run_benchmark_ours.py   # Wrapper that batches our own agent.
│   ├── configs/tools.example.json   # Copy to tools.json and edit before running.
│   ├── prompts/repo_setup_task.txt  # Unified task prompt shared by all systems.
│   ├── phase2_pipeline.py / summarize_results.py / trajectory_parser.py
│   └── README.md / CLI_TOOLS_ARCHITECTURE.md  # Harness docs.
└── eval/                      # Re-adjudication of public baselines
    ├── eval_envbench.py / eval_envbench_33.py    # envBench outputs
    ├── eval_execagent.py / eval_execagent_full.py # ExecAgent outputs
    ├── eval_r2r.py                                # Repo2Run outputs
    └── results/                                   # Adjudicated JSON
```

---

## 8. Troubleshooting

- **`docker.errors.DockerException`** — Docker daemon is not running, or the
  current user is not in the `docker` group.
- **HTTP 401 / 403 from the LLM** — re-check `OPENAI_API_KEY` and
  `OPENAI_BASE_URL` in `.env`.
- **Want to keep the container** — `OURSYS_KEEP_CONTAINER=1 ./scripts/run.sh
  ...` (see §4 above).
- **Want to disable the experience store** — pass `--no-xpu`, or set
  `XPU_ENABLED=false` in `.env` (the default).
- **Postgres errors during XPU import** — confirm the `dns` connection string
  in `.env` is reachable, the `pgvector` extension is installed, and your
  embedding endpoint is responsive (the importer needs it to embed every row).

---

