#!/usr/bin/env python3
"""Online XPU extractor: per-repo trajectory -> XPU -> store, reusing the offline pipeline.

Offline (run_xpu_pipeline.py):
  1. convert_tracks() : track.json -> jsonl
  2. extract_xpu_from_trajs_mvp.py : LLM extraction
  3. extract_xpu_to_v1.py : filter valid experiences
  4. index_xpu_to_vector_db.py : store in vector DB

Online: run the same flow on a single repo.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path

from ..logger import get_logger

logger = get_logger("xpu.online_extractor")


def online_extract_and_store(repo_name: str, output_dir: str, sha: str = "HEAD") -> dict:
    """Online entry point — reuses offline pipeline scripts.

    Args:
        repo_name: e.g. "owner/repo".
        output_dir: directory holding track.json.
        sha: commit SHA.
    """
    result = {
        "repo": repo_name,
        "extracted": False,
        "stored": False,
        "xpu_id": None,
        "reason": None
    }

    track_path = Path(output_dir) / "track.json"
    if not track_path.exists():
        result["reason"] = "track.json not found"
        return result

    logger.info(f"online XPU extraction: {repo_name}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="xpu_online_"))

    try:
        # Step 1: convert track.json -> jsonl
        safe_name = repo_name.replace('/', '__')
        jsonl_name = f"{safe_name}@{sha}.jsonl"
        traj_dir = tmp_dir / "trajs"
        traj_dir.mkdir()
        jsonl_path = traj_dir / jsonl_name

        with open(track_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for step in data:
                f.write(json.dumps(step, ensure_ascii=False) + "\n")

        # Step 2: LLM extraction
        extracted_file = tmp_dir / "extracted.jsonl"

        from .extract_xpu_from_trajs_mvp import extract_xpu_from_trajs
        extract_xpu_from_trajs(jsonl_path, extracted_file)

        # Step 3: keep entries marked as valid xpu
        xpu_obj = None
        if extracted_file.exists():
            with open(extracted_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get('llm_decision') == 'xpu':
                        xpu_obj = entry.get('xpu')
                        break

        if not xpu_obj:
            result["reason"] = entry.get('llm_reason', 'LLM skipped') if 'entry' in dir() else "no extraction"
            return result

        result["extracted"] = True
        result["xpu_id"] = xpu_obj.get("id")

        local_xpu_path = Path(output_dir) / "extracted_xpu.json"
        with open(local_xpu_path, 'w', encoding='utf-8') as f:
            json.dump(xpu_obj, f, ensure_ascii=False, indent=2)

        # Step 4: index into vector DB
        dns = os.environ.get("dns")
        if not dns:
            result["reason"] = "extraction ok but missing db connection (dns)"
            return result

        try:
            from .xpu_adapter import XpuEntry, XpuAtom
            from .xpu_vector_store import XpuVectorStore, build_xpu_text, text_to_embedding
            from .xpu_dedup import dedup_and_store

            atoms = [XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                     for a in xpu_obj.get("atoms", [])]
            entry = XpuEntry(
                id=xpu_obj.get("id"),
                signals=xpu_obj.get("signals", {}) or {},
                advice_nl=xpu_obj.get("advice_nl", []),
                atoms=atoms
            )

            text = build_xpu_text(entry)
            embedding = text_to_embedding(text)

            store = XpuVectorStore()

            dedup_result = dedup_and_store(store, entry, embedding, use_llm=True)
            result["stored"] = True
            result["xpu_id"] = dedup_result["xpu_id"]
            result["reason"] = dedup_result["reason"]
            logger.info(f"[Dedup] {dedup_result['action']}: {dedup_result['reason']}")

            store.close()

        except Exception as e:
            result["reason"] = f"extraction ok but store failed: {e}"
            logger.error(f"store failed: {e}")

    except Exception as e:
        result["reason"] = f"exception: {str(e)}"
        logger.error(f"online extraction failed: {e}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Online XPU extraction & storage")
    parser.add_argument("--repo", required=True, help="repo name (owner/repo)")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--sha", default="HEAD", help="commit SHA")

    args = parser.parse_args()
    result = online_extract_and_store(args.repo, args.output_dir, args.sha)
    print(json.dumps(result, ensure_ascii=False, indent=2))
