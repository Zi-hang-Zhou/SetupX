# Setup Agent — Minimum Runnable Distribution

This package configures a runnable Docker environment for an arbitrary Git
repository: given a repo URL, the agent inspects the project, runs shell
commands inside a sandboxed container until installation / tests succeed, and
emits a result JSON.

This is the **minimum standalone build**. It does not ship benchmark URL
lists, experience-store dumps, run logs, or experiment scaffolding — only the
code needed to configure one repository at a time.

---

## 1. Requirements

- Python 3.10+
- A working Docker daemon on the host (`docker ps` must succeed without
  `sudo`)
- Network access to one OpenAI-compatible LLM endpoint

That is the entire dependency surface.

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

## 6. Layout

```
.
├── .env.example          # config template
├── README.md             # this file
├── requirements.txt      # python deps
├── scripts/
│   └── run.sh            # one-line wrapper around `python -m src.main`
└── src/
    ├── main.py                # CLI entry point; orchestrates the 3 phases
    ├── agent.py               # Phase 1 main loop (speculative exec + rollback)
    ├── llm_engine.py          # LLM call + JSON action parsing
    ├── retriever_agent.py     # Two-tier experience retrieval (vector + LLM rerank)
    ├── environment_manager.py # Docker container lifecycle + snapshots
    ├── verifier_agent.py      # Phase 1 verify gate
    ├── prosecutor_agent.py    # Phase 2 prosecutor
    ├── judge_agent.py         # Phase 2 judge
    ├── task_meta.py           # Multi-repo family meta rendering
    ├── models.py / config.py / logger.py / xpu_client.py
    └── xpu/                   # Experience-store extraction & vector index
```

---

## 7. Troubleshooting

- **`docker.errors.DockerException`** — Docker daemon is not running, or the
  current user is not in the `docker` group.
- **HTTP 401 / 403 from the LLM** — re-check `OPENAI_API_KEY` and
  `OPENAI_BASE_URL` in `.env`.
- **Want to keep the container** — `OURSYS_KEEP_CONTAINER=1 ./scripts/run.sh
  ...` (see §4 above).
- **Want to disable the experience store** — pass `--no-xpu`, or set
  `XPU_ENABLED=false` in `.env` (the default).

---

## 8. What is NOT in this distribution

To keep the package minimal and self-contained, the following are
intentionally excluded:

- Benchmark repository URL lists / family spec JSONs
- Experience-store dumps (`xpu_*.jsonl`)
- Run logs, trajectories, and experiment outputs
- Benchmark orchestration scripts (parallel runners, telemetry)
- Paper drafts and reference PDFs

The goal of this package is single-repo reproducibility, not benchmark
replay.
