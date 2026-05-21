"""Bulk-import an XPU experience JSONL into the local pgvector store.

Usage:
    python scripts/import_xpu_jsonl.py <jsonl_path> [--clear]

    --clear   Truncate the target table before importing.

Each JSONL line is one experience entry with this schema:

    {
      "id":         "unique-string",
      "signals":    {... "applicability": {...} ...},
      "advice_nl":  [...],
      "atoms":      [{"name": "...", "args": {...}}, ...],
      "telemetry":  {...}    // optional
    }

The target table name comes from `XPU_TABLE` (env), default `xpu_entries`.
The pgvector connection string comes from `dns` (env). The embedding model
comes from `EMBEDDING_*` (env). All of these live in `.env` already.

The table and IVFFlat index are created on first connection (see
`src/xpu/xpu_vector_store.py::create_xpu_table`); you do not need a separate
schema migration step.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append(str(Path(__file__).parent.parent))

from src.xpu.xpu_vector_store import XpuVectorStore, build_xpu_text, text_to_embedding
from src.xpu.xpu_adapter import XpuEntry, XpuAtom


def import_jsonl(jsonl_path: str, clear: bool = False) -> None:
    path = Path(jsonl_path)
    if not path.exists():
        print(f"file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    lines = path.read_text(encoding="utf-8").splitlines()
    total = len([l for l in lines if l.strip()])
    print(f"{total} entries to import...")

    store = XpuVectorStore()
    table = store._table
    print(f"target table: {table}")

    if clear:
        conn = store._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table};")
                conn.commit()
            print(f"cleared table {table}")
        finally:
            store._put_conn(conn)

    ok, fail = 0, 0

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[{i}] JSON parse failed, skipped: {e}")
            fail += 1
            continue

        atoms = [
            XpuAtom(name=a["name"], args=a.get("args", {}))
            for a in raw.get("atoms", [])
        ]

        entry = XpuEntry(
            id=raw["id"],
            signals=raw.get("signals", {}) or {},
            advice_nl=raw.get("advice_nl", []),
            atoms=atoms,
            telemetry=raw.get("telemetry", {}),
        )

        try:
            text = build_xpu_text(entry)
            embedding = text_to_embedding(text)

            conn = store._get_conn()
            try:
                with conn.cursor() as cur:
                    embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                    cur.execute(
                        f"""
                        INSERT INTO {table} (id, signals, advice_nl, atoms, embedding, telemetry)
                        VALUES (%s, %s, %s, %s, %s::vector, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            signals = EXCLUDED.signals,
                            advice_nl = EXCLUDED.advice_nl,
                            atoms = EXCLUDED.atoms,
                            embedding = EXCLUDED.embedding,
                            telemetry = EXCLUDED.telemetry;
                        """,
                        (
                            entry.id,
                            json.dumps(entry.signals),
                            json.dumps(entry.advice_nl),
                            json.dumps([{"name": a.name, "args": a.args} for a in entry.atoms]),
                            embedding_str,
                            json.dumps(entry.telemetry),
                        ),
                    )
                    conn.commit()
            finally:
                store._put_conn(conn)

            print(f"[{i}/{total}] OK {entry.id}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{total}] FAIL {entry.id}: {e}")
            fail += 1

    store.close()
    print(f"\ndone: {ok} ok, {fail} failed")


if __name__ == "__main__":
    clear = "--clear" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--clear"]
    if not args:
        print("usage: python scripts/import_xpu_jsonl.py <jsonl_path> [--clear]", file=sys.stderr)
        sys.exit(2)
    import_jsonl(args[0], clear=clear)
