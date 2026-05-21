"""Inflate an XPU table with synthetic noise entries.

Optional utility for stress-testing retrieval against a large, partially-noisy
knowledge base. Reads whatever entries already live in the target table (see
`XPU_TABLE` / `dns` in `.env`) and synthesises additional entries via four
strategies:

  1. Context perturbation  (Python version, OS, tool-name noise)
  2. Cross-grafting        (signals from A, advice from B, atoms from C)
  3. Generalisation blur   (specific advice -> vague templates)
  4. Cross-language drift  (Node / Rust / Go / Ruby snippets)

Synthesised rows are tagged with id prefixes `noise_ctx_*`, `noise_graft_*`,
`noise_vague_*`, `noise_lang_*`, so they can be filtered or removed later
(e.g. `DELETE FROM xpu_entries WHERE id LIKE 'noise_%';`).

Usage:
    python scripts/inflate_xpu_db.py --target 2000

The script reads these env vars (all already defined in `.env.example`):
  dns          pgvector connection string
  XPU_TABLE    target table name (default: xpu_entries)
  EMBEDDING_*  embedding endpoint used for the synthesised text

The script writes:
  Up to (target - current_count) new rows into XPU_TABLE. Existing rows are
  not modified. Re-running is idempotent up to the target count.

This is a research utility — production runs do NOT need it.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from src.xpu.xpu_vector_store import XpuVectorStore, text_to_embedding  # noqa: E402


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------

def gen_context_perturbations(entry: dict, n: int = 2) -> list[dict]:
    """Strategy 1: clone an entry and perturb its applicability (python / os / tools)."""
    variants = []
    py_versions = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]
    os_options = [["linux"], ["linux", "macos"], ["linux", "windows"]]
    tool_noise = ["conda", "poetry", "pipenv", "pdm", "hatch", "uv", "mamba"]

    for _ in range(n):
        v = copy.deepcopy(entry)
        signals = v.setdefault("signals", {})
        ctx = signals.setdefault("applicability", {})
        if isinstance(ctx, dict):
            ctx["python"] = [random.choice(py_versions)]
            ctx["os"] = random.choice(os_options)
            if random.random() > 0.5:
                existing_tools = ctx.get("tools", [])
                ctx["tools"] = existing_tools + [random.choice(tool_noise)]
        v["id"] = f"noise_ctx_{int(time.time())}_{os.urandom(3).hex()}"
        v["telemetry"] = {"hits": 0}
        variants.append(v)
    return variants


def gen_cross_graft(entries: list[dict], n: int = 100) -> list[dict]:
    """Strategy 2: chimera entries — signals from B (with A's applicability), advice from B, atoms from {A,B,C}."""
    variants = []
    for _ in range(n):
        a, b, c = random.sample(entries, 3)
        signals_b = copy.deepcopy(b["signals"]) or {}
        a_app = (a.get("signals") or {}).get("applicability", {})
        signals_b["applicability"] = copy.deepcopy(a_app)
        v = {
            "id": f"noise_graft_{int(time.time())}_{os.urandom(3).hex()}",
            "signals": signals_b,
            "advice_nl": copy.deepcopy(c["advice_nl"]),
            "atoms": copy.deepcopy(random.choice([a, b, c])["atoms"]),
            "telemetry": {"hits": 0},
        }
        variants.append(v)
        time.sleep(0.001)  # avoid id collisions
    return variants


def gen_vague_entries(entries: list[dict], n: int = 100) -> list[dict]:
    """Strategy 3: replace specific advice with generic templates."""
    vague_advice = [
        ["Check that the Python version satisfies the project's requirements",
         "Ensure the run uses a compatible Python version"],
        ["Install the project's dependencies",
         "Use pip install to fetch any missing dependencies"],
        ["Check whether the system has the required build tools",
         "Install build-essential or an equivalent toolchain"],
        ["Ensure the virtual environment is activated",
         "Create and activate a venv or conda environment"],
        ["Check that PATH includes the necessary directories",
         "Confirm the executables are reachable via PATH"],
        ["Run pip install -e . to install the project",
         "Install the project in editable mode before testing"],
        ["Install the test dependencies",
         "Check requirements-test.txt or pyproject.toml for test extras"],
        ["Check the configuration files",
         "Confirm env vars and configs match the test environment"],
        ["Upgrade pip and setuptools",
         "Run pip install --upgrade pip setuptools wheel"],
        ["Check network connectivity",
         "Confirm pip can reach PyPI"],
        ["Resolve dependency version conflicts",
         "Try pip install --force-reinstall to break ties"],
        ["Install system-level dependencies",
         "Use apt-get to fetch missing system libraries"],
    ]
    vague_signals_kw = [
        ["install failure", "dependency issue", "environment problem"],
        ["ModuleNotFoundError", "import error"],
        ["pip install", "package management"],
        ["build error", "compilation failure", "gcc"],
        ["Permission denied", "permission issue"],
        ["version conflict", "incompatible", "requires"],
    ]

    variants = []
    for _ in range(n):
        base = random.choice(entries)
        v = copy.deepcopy(base)
        v["id"] = f"noise_vague_{int(time.time())}_{os.urandom(3).hex()}"
        v["advice_nl"] = random.choice(vague_advice)
        v["signals"]["keywords"] = random.choice(vague_signals_kw)
        v["atoms"] = []  # vague advice carries no concrete action
        v["telemetry"] = {"hits": 0}
        variants.append(v)
        time.sleep(0.001)
    return variants


def gen_cross_lang_entries(n: int = 100) -> list[dict]:
    """Strategy 4: irrelevant non-Python ecosystem entries."""
    cross_lang = [
        {
            "signals": {
                "applicability": {"lang": "javascript", "os": ["linux"], "tools": ["npm", "node"]},
                "keywords": ["npm install", "node_modules", "package.json"],
                "regex": [],
                "situation_triggers": ["npm install fails"],
            },
            "advice_nl": [
                "When npm install reports ERESOLVE, retry with --legacy-peer-deps."
            ],
            "atoms": [{"name": "shell", "args": {"command": "npm install --legacy-peer-deps"}}],
        },
        {
            "signals": {
                "applicability": {"lang": "javascript", "os": ["linux"], "tools": ["yarn"]},
                "keywords": ["yarn install", "yarn.lock"],
                "regex": [],
                "situation_triggers": ["yarn install fails"],
            },
            "advice_nl": [
                "If yarn install fails on engines mismatch, retry with --ignore-engines."
            ],
            "atoms": [{"name": "shell", "args": {"command": "yarn install --ignore-engines"}}],
        },
        {
            "signals": {
                "applicability": {"lang": "rust", "os": ["linux"], "tools": ["cargo"]},
                "keywords": ["cargo build", "rustc", "Cargo.toml"],
                "regex": [],
                "situation_triggers": ["Rust compilation fails"],
            },
            "advice_nl": [
                "If cargo build fails, refresh the Rust toolchain with rustup update."
            ],
            "atoms": [{"name": "shell", "args": {"command": "rustup update stable"}}],
        },
        {
            "signals": {
                "applicability": {"lang": "go", "os": ["linux"], "tools": ["go"]},
                "keywords": ["go build", "go mod", "go.sum"],
                "regex": [],
                "situation_triggers": ["go mod download fails"],
            },
            "advice_nl": [
                "If go mod download fails, set GOPROXY to a closer mirror."
            ],
            "atoms": [{"name": "set_env", "args": {"key": "GOPROXY", "value": "https://goproxy.io,direct"}}],
        },
        {
            "signals": {
                "applicability": {"lang": "ruby", "os": ["linux"], "tools": ["bundler", "gem"]},
                "keywords": ["bundle install", "Gemfile", "gem install"],
                "regex": [],
                "situation_triggers": ["bundle install fails"],
            },
            "advice_nl": [
                "If bundle install fails on native extensions, apt-get install ruby-dev libsqlite3-dev first."
            ],
            "atoms": [{"name": "apt_install", "args": {"packages": ["ruby-dev", "libsqlite3-dev"]}}],
        },
    ]

    variants = []
    for _ in range(n):
        base = copy.deepcopy(random.choice(cross_lang))
        base["id"] = f"noise_lang_{int(time.time())}_{os.urandom(3).hex()}"
        base["telemetry"] = {"hits": 0}
        variants.append(base)
        time.sleep(0.001)
    return variants


# ---------------------------------------------------------------------------
# DB helpers (use the same connection pool as the rest of the codebase)
# ---------------------------------------------------------------------------

def load_originals(store: XpuVectorStore) -> list[dict]:
    """Pull existing non-noise rows so they can be used as seed material."""
    table = store._table
    conn = store._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, signals, advice_nl, atoms, telemetry "
                f"FROM {table} WHERE id NOT LIKE 'noise_%'"
            )
            rows = cur.fetchall()
    finally:
        store._put_conn(conn)

    entries = []
    for r in rows:
        entries.append({
            "id": r[0],
            "signals": r[1],
            "advice_nl": r[2],
            "atoms": r[3],
            "telemetry": r[4],
        })
    return entries


def current_count(store: XpuVectorStore) -> int:
    table = store._table
    conn = store._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            n = cur.fetchone()[0]
    finally:
        store._put_conn(conn)
    return n


def compute_and_insert(store: XpuVectorStore, variants: list[dict], batch_label: str) -> int:
    """Embed each variant and upsert it into the target table."""
    table = store._table
    conn = store._get_conn()
    inserted = 0
    try:
        with conn.cursor() as cur:
            for i, v in enumerate(variants):
                # Compose embedding text the same way the rest of the codebase does.
                advice_text = (
                    " ".join(v["advice_nl"]) if isinstance(v["advice_nl"], list)
                    else str(v["advice_nl"])
                )
                signals = v.get("signals", {}) or {}
                keywords = " ".join(signals.get("keywords", []))
                triggers = " ".join(signals.get("situation_triggers", []))
                embed_text = f"{advice_text} {keywords} {triggers}".strip()

                try:
                    embedding = text_to_embedding(embed_text)
                except Exception as e:
                    print(f"  [{batch_label}] embedding failed ({i + 1}): {e}", file=sys.stderr)
                    continue

                embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

                try:
                    cur.execute(
                        f"""
                        INSERT INTO {table}
                            (id, signals, advice_nl, atoms, embedding, telemetry)
                        VALUES (%s, %s, %s, %s, %s::vector, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            v["id"],
                            json.dumps(v["signals"], ensure_ascii=False),
                            json.dumps(v["advice_nl"], ensure_ascii=False),
                            json.dumps(v["atoms"], ensure_ascii=False),
                            embedding_str,
                            json.dumps(v["telemetry"], ensure_ascii=False),
                        ),
                    )
                    inserted += 1
                except Exception as e:
                    print(f"  [{batch_label}] insert failed ({i + 1}): {e}", file=sys.stderr)
                    conn.rollback()
                    continue

                if (i + 1) % 50 == 0:
                    conn.commit()
                    print(f"  [{batch_label}] progress: {i + 1}/{len(variants)}")

            conn.commit()
    finally:
        store._put_conn(conn)
    return inserted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Inflate an XPU table with synthetic noise entries.")
    parser.add_argument("--target", type=int, default=2000, help="target total row count (default: 2000)")
    args = parser.parse_args()

    store = XpuVectorStore()
    try:
        print(f"target: {args.target}")
        cur_count = current_count(store)
        print(f"current: {cur_count}")

        originals = load_originals(store)
        print(f"seed originals: {len(originals)}")
        if not originals:
            print("table holds no non-noise rows; cannot synthesise from empty seed.", file=sys.stderr)
            return 1

        need = args.target - cur_count
        if need <= 0:
            print("target already reached; nothing to do.")
            return 0

        # Mixture: 35% context perturbation, 25% cross-graft, 25% vague, remainder cross-language.
        n_ctx = int(need * 0.35)
        n_graft = int(need * 0.25)
        n_vague = int(need * 0.25)
        n_lang = need - n_ctx - n_graft - n_vague

        print(
            f"\nplan: context-perturb {n_ctx} + cross-graft {n_graft} + "
            f"vague {n_vague} + cross-language {n_lang} = {need}"
        )

        # 1. Context perturbation
        print(f"\n[1/4] context perturbation ({n_ctx})...")
        ctx_variants = []
        per_entry = max(1, n_ctx // len(originals))
        sampled = random.sample(originals, min(len(originals), n_ctx // max(per_entry, 1)))
        for e in sampled:
            ctx_variants.extend(gen_context_perturbations(e, per_entry))
        ctx_variants = ctx_variants[:n_ctx]
        inserted = compute_and_insert(store, ctx_variants, "ctx")
        print(f"  inserted {inserted}, table now {current_count(store)}")

        # 2. Cross-graft
        print(f"\n[2/4] cross-graft ({n_graft})...")
        graft_variants = gen_cross_graft(originals, n_graft)
        inserted = compute_and_insert(store, graft_variants, "graft")
        print(f"  inserted {inserted}, table now {current_count(store)}")

        # 3. Generalisation blur
        print(f"\n[3/4] generalisation blur ({n_vague})...")
        vague_variants = gen_vague_entries(originals, n_vague)
        inserted = compute_and_insert(store, vague_variants, "vague")
        print(f"  inserted {inserted}, table now {current_count(store)}")

        # 4. Cross-language drift
        print(f"\n[4/4] cross-language drift ({n_lang})...")
        lang_variants = gen_cross_lang_entries(n_lang)
        inserted = compute_and_insert(store, lang_variants, "lang")
        print(f"  inserted {inserted}, table now {current_count(store)}")

        print(f"\ndone. final size: {current_count(store)}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
