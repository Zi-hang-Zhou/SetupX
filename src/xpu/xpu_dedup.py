"""XPU dedup & smart-merge.

When a new XPU's embedding cosine similarity to an existing one is >= 0.85,
ask the LLM whether the two address the same problem; if so, merge advice_nl.

Decision flow:
  embedding -> search similar
    - no similar (< 0.85) -> insert (action="new")
    - same id              -> overwrite (action="new")
    - different id, similar (>= 0.85) -> LLM judgment
        - "different problem"   -> insert (action="different_inserted")
        - "same problem"        -> LLM merges advice_nl
            - merge changed     -> update existing (action="merged")
            - no change         -> mark duplicate (action="duplicate")

Each dedup costs 2 LLM calls (judgment + merge).
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .extract_xpu_from_trajs_mvp import (
    openai_compatible_chat_completions,
    parse_llm_json,
    load_llm_config_from_env,
    get_env_or_raise,
)
from ..logger import get_logger

logger = get_logger("xpu.dedup")

MERGE_SIMILARITY_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _call_llm_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    cfg = load_llm_config_from_env()
    api_key = get_env_or_raise(cfg["api_key_env_var"])
    base_url = os.environ.get(cfg["base_url_env_var"]) or "https://api.openai.com/v1"

    raw = openai_compatible_chat_completions(
        model=cfg["llm_model"],
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        timeout_sec=cfg["timeout_sec"],
        response_format_json=True,
    )
    content = raw["choices"][0]["message"]["content"]
    return parse_llm_json(content)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_judgment_prompt(
    existing_entry: Dict[str, Any],
    new_entry_dict: Dict[str, Any],
) -> List[Dict[str, str]]:
    system_text = (
        "You are a senior expert in Python environment configuration."
        "\nYou are given two environment experiences (XPUs); decide whether they"
        "\nare addressing essentially the same problem."
        "\nNote: even when the concrete package names or versions differ, treat them as"
        "\nthe same problem so long as the root cause and the fix idea are the same."
        "\nYour answer must be a strict JSON object, with no extra text."
    )

    user_payload = {
        "task": "decide whether the two experiences address the same problem",
        "existing_experience": {
            "id": existing_entry.get("id"),
            "signals": existing_entry.get("signals"),
            "advice_nl": existing_entry.get("advice_nl"),
        },
        "new_experience": {
            "id": new_entry_dict.get("id"),
            "signals": new_entry_dict.get("signals"),
            "advice_nl": new_entry_dict.get("advice_nl"),
        },
        "output_requirement": (
            "Output JSON: {\"same_problem\": true/false, \"reason\": \"brief justification\"}. "
            "same_problem=true means the two experiences essentially solve the same class of problem; "
            "same_problem=false means they look textually similar but are actually different problems."
        ),
    }

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def _build_merge_prompt(
    existing_advice: List[str],
    new_advice: List[str],
) -> List[Dict[str, str]]:
    system_text = (
        "You are a senior expert in Python environment configuration."
        "\nYou are given two groups of environment-configuration suggestions (advice_nl);"
        "\nthey come from two experiences that address the same problem."
        "\nMerge them intelligently into a single improved suggestion list."
        "\nMerge requirements:"
        "\n1. drop duplicates and items that say the same thing;"
        "\n2. keep every distinct insight and fix;"
        "\n3. if two suggestions can be combined into one more complete statement, do so;"
        "\n4. keep the final list to 1-7 items;"
        "\n5. write the suggestions in English."
        "\nYour answer must be a strict JSON object, with no extra text."
    )

    user_payload = {
        "task": "merge two suggestion groups",
        "existing_advice_nl": existing_advice,
        "new_advice_nl": new_advice,
        "output_requirement": (
            "Output JSON: {\"merged_advice_nl\": [\"suggestion 1\", \"suggestion 2\", ...], "
            "\"merge_summary\": \"brief description of what was merged\"}"
        ),
    }

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


# ---------------------------------------------------------------------------
# Core dedup logic
# ---------------------------------------------------------------------------

def judge_and_merge(
    existing_entry: Dict[str, Any],
    new_entry_dict: Dict[str, Any],
) -> Tuple[str, Optional[List[str]]]:
    """LLM judges identity then optionally merges advice_nl.

    Returns (action, merged_advice):
      - action in {"different", "same_merged", "same_no_change"}
      - merged_advice is non-None only when action == "same_merged".
    """
    judgment_messages = _build_judgment_prompt(existing_entry, new_entry_dict)
    judgment_result = _call_llm_json(judgment_messages)

    same_problem = judgment_result.get("same_problem", False)
    judgment_reason = judgment_result.get("reason", "")

    logger.info(
        "[Dedup] LLM judgment: same_problem=%s, reason=%s",
        same_problem, judgment_reason,
    )

    if not same_problem:
        return "different", None

    existing_advice = existing_entry.get("advice_nl") or []
    new_advice = new_entry_dict.get("advice_nl") or []

    merge_messages = _build_merge_prompt(existing_advice, new_advice)
    merge_result = _call_llm_json(merge_messages)

    merged_advice = merge_result.get("merged_advice_nl")
    merge_summary = merge_result.get("merge_summary", "")

    logger.info("[Dedup] LLM merge done: %s", merge_summary)

    if not merged_advice or not isinstance(merged_advice, list):
        logger.warning("[Dedup] invalid LLM merge result, falling back to simple append")
        merged_advice = _simple_merge(existing_advice, new_advice)

    if merged_advice is None or set(merged_advice) == set(existing_advice):
        return "same_no_change", None

    return "same_merged", merged_advice


def _simple_merge(
    existing_advice: List[str],
    new_advice: List[str],
) -> Optional[List[str]]:
    """Fallback: append new items not already in existing list."""
    merged = list(existing_advice)
    added = 0
    for adv in new_advice:
        if adv not in merged:
            merged.append(adv)
            added += 1
    return merged if added > 0 else None


def dedup_and_store(
    store,
    entry,
    embedding: List[float],
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Top-level dedup-and-store orchestration (shared by online & offline pipelines).

    Returns: {"action": str, "xpu_id": str, "reason": str}
    action in {"new", "different_inserted", "merged", "duplicate"}
    """
    from .xpu_adapter import XpuContext

    applicability = entry.signals.get("applicability", {}) or {}
    ctx_lang = applicability.get("lang", "python")
    similar_entries = store.search(
        query_embedding=embedding,
        ctx=XpuContext(lang=ctx_lang),
        k=3,
        min_similarity=MERGE_SIMILARITY_THRESHOLD,
    )

    if not similar_entries:
        store.upsert_entry(entry, embedding)
        return {
            "action": "new",
            "xpu_id": entry.id,
            "reason": "extracted and stored (new entry)",
        }

    existing = similar_entries[0]
    existing_id = existing["id"]
    similarity = existing.get("similarity", 0)

    if existing_id == entry.id:
        store.upsert_entry(entry, embedding)
        return {
            "action": "new",
            "xpu_id": entry.id,
            "reason": f"overwrote existing entry {entry.id}",
        }

    new_entry_dict = {
        "id": entry.id,
        "signals": entry.signals,
        "advice_nl": entry.advice_nl,
    }

    if use_llm:
        try:
            action, merged_advice = judge_and_merge(existing, new_entry_dict)
        except Exception as e:
            logger.warning("[Dedup] LLM call failed, falling back to simple append: %s", e)
            action = "same_merged"
            merged_advice = _simple_merge(
                existing.get("advice_nl") or [],
                entry.advice_nl or [],
            )
    else:
        action = "same_merged"
        merged_advice = _simple_merge(
            existing.get("advice_nl") or [],
            entry.advice_nl or [],
        )

    if action == "different":
        store.upsert_entry(entry, embedding)
        return {
            "action": "different_inserted",
            "xpu_id": entry.id,
            "reason": (
                f"LLM judged different from {existing_id} (sim={similarity:.3f}); "
                f"inserted as new entry"
            ),
        }
    elif action == "same_merged" and merged_advice is not None:
        store.update_advice(existing_id, merged_advice)
        return {
            "action": "merged",
            "xpu_id": existing_id,
            "reason": f"smart-merged into existing {existing_id} (sim={similarity:.3f})",
        }
    else:  # same_no_change
        return {
            "action": "duplicate",
            "xpu_id": existing_id,
            "reason": f"duplicate: existing {existing_id} fully covers (sim={similarity:.3f})",
        }
