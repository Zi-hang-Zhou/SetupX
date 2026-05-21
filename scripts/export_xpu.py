#!/usr/bin/env python3
"""Export the XPU experience store from PostgreSQL to a JSONL file.

Default: incremental append — only entries whose `id` is not already in the
output file are written. Pass `--full` to rewrite the file from scratch.

Usage:
    python scripts/export_xpu.py                           # append to xpu.jsonl
    python scripts/export_xpu.py -o backup.jsonl --full    # full dump
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.xpu.xpu_vector_store import XpuVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export XPU experience entries from PostgreSQL to JSONL")
    parser.add_argument("-o", "--output", default="xpu.jsonl", help="output file path (default: xpu.jsonl)")
    parser.add_argument("--full", action="store_true", help="full export (overwrite); default is incremental append")
    return parser.parse_args()


def load_existing_ids(path: Path) -> set[str]:
    """Read existing IDs from the JSONL output file."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def export_from_db(store: XpuVectorStore, exclude_ids: set[str]) -> list[dict]:
    """Pull rows from the XPU table, skipping ones already in `exclude_ids`."""
    table = store._table
    conn = store._get_conn()
    try:
        with conn.cursor() as cur:
            if exclude_ids:
                cur.execute(
                    f"""
                    SELECT id, signals, advice_nl, atoms, telemetry
                    FROM {table}
                    WHERE id != ALL(%s)
                    ORDER BY created_at;
                    """,
                    (list(exclude_ids),),
                )
            else:
                cur.execute(
                    f"""
                    SELECT id, signals, advice_nl, atoms, telemetry
                    FROM {table}
                    ORDER BY created_at;
                    """
                )
            rows = cur.fetchall()
    finally:
        store._put_conn(conn)

    return [
        {
            "id": r[0],
            "signals": r[1] or {},
            "advice_nl": r[2] or [],
            "atoms": r[3] or [],
            "telemetry": r[4] or {},
        }
        for r in rows
    ]


def main() -> int:
    args = parse_args()
    output = Path(args.output)

    store = XpuVectorStore()

    if args.full:
        exclude_ids: set[str] = set()
        mode = "w"
    else:
        exclude_ids = load_existing_ids(output)
        mode = "a"
        print(f"{len(exclude_ids)} existing entries; appending only new ones...")

    entries = export_from_db(store, exclude_ids)
    store.close()

    if not entries:
        print("nothing new; file unchanged")
        return 0

    with open(output, mode, encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = len(exclude_ids) + len(entries) if not args.full else len(entries)
    print(f"wrote {len(entries)} entries to {output} ({total} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
