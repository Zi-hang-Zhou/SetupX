"""Retriever Agent — XPU knowledge-retrieval sub-agent.

Acts as a sub-agent of the Setup Agent. It retrieves the most appropriate XPU
experiences and runs the refinement / delayed audit in its own context.

Core design (modeled after coding-agent file-search patterns):
  1. Isolated context: retrieval does not pollute the main agent's context;
     the main agent only sees the final suggestions.
  2. Two-layer retrieval:
     - Layer 1: vector recall (pgvector cosine, Top-N candidates).
     - Layer 2: LLM refinement (pick K most relevant from N candidates).
  3. Delayed audit: each call also audits the XPUs recommended last time.
  4. Soft filtering: no hard filter on negative telemetry at the DB level; the LLM
     judges dynamically using telemetry as a signal.

Two interfaces:
  - DB side: read/write XPU entries + telemetry via XpuVectorStore.
  - Main-agent side: take the current situation, return refined XPU suggestions.
"""

import json
import os
import re

from .logger import get_logger
from .models import XPUSuggestion

logger = get_logger("retriever_agent")


# =============================================================================
# Audit verdict structure
# =============================================================================

class AuditVerdict:
    """Result of auditing a previously recommended XPU.

    On every retrieve() call, the Retriever Agent audits the XPUs recommended
    on the previous call. The audit produces a verdict (success/failure/neutral)
    plus a continuous score.

    Attributes:
        xpu_id: ID of the audited XPU.
        verdict: success / failure / neutral.
        score: continuous in [-1.0, 1.0].
        reason: LLM-provided rationale.
    """

    def __init__(self, xpu_id: str, verdict: str, score: float, reason: str):
        self.xpu_id = xpu_id
        self.verdict = verdict
        self.score = score
        self.reason = reason

    def __repr__(self):
        return f"AuditVerdict({self.xpu_id}, {self.verdict}, score={self.score:.2f})"


# =============================================================================
# Last-XPU record (for delayed audit)
# =============================================================================

class LastXPURecord:
    """Records the most recent XPU recommendation, used by the delayed audit.

    Created automatically when retrieve() returns suggestions; does not depend
    on the main agent calling anything explicitly. Whether the main agent then
    chooses TRY_XPU_SUGGESTION or SHELL_COMMAND (consulting the XPU), the
    next retrieve() call can still audit the effect.

    Attributes:
        xpu_ids: IDs of the XPUs recommended last time.
        descriptions: per-XPU description (advice_nl or retriever reason).
        situation: the situation at recommendation time.
        state_before: the error/state at recommendation time.
        step_index: len(history) at recommendation time (anchor for subsequent steps).
    """

    def __init__(
        self,
        xpu_ids: list[str],
        descriptions: list[str],
        situation: str,
        state_before: str,
        step_index: int,
    ):
        self.xpu_ids = xpu_ids
        self.descriptions = descriptions
        self.situation = situation
        self.state_before = state_before
        self.step_index = step_index


# =============================================================================
# Retriever Agent core
# =============================================================================

# LLM refinement prompt
REFINE_PROMPT = """You are an XPU experience-retrieval assistant. Given a list of candidate XPU experiences, your task is to pick the Top-K that best match the current deployment situation.

## Current deployment situation
{situation}

## Candidate XPU experiences (total: {n_candidates}, ordered by vector similarity)
{candidates_text}

## Selection rules
1. Exact match first: XPUs whose advice_nl directly addresses the current problem rank highest.
2. Telemetry as reference: pay attention to each XPU's historical hit / success / failure counts,
   but do not discard one merely because it has many failures — judge whether the previous failure
   scenarios are similar to the current one. If they are not, the XPU may still be effective.
3. Drop the irrelevant: if an XPU's advice is completely unrelated to the current problem, do not
   pick it.
4. Pick at most {k}.

You must reply in JSON:
{{
  "selected": [
    {{
      "xpu_id": "...",
      "relevance_reason": "why this XPU matches the current situation",
      "confidence": 0.85
    }}
  ]
}}
"""

# LLM audit prompt
AUDIT_PROMPT = """You are an XPU experience-audit assistant. Your task: judge whether the XPU experiences recommended to the Agent last time helped the deployment.

Note: after an XPU suggestion is recommended, the Agent may have either executed the suggested command directly (TRY_XPU_SUGGESTION) or merely consulted the idea and generated its own command (SHELL_COMMAND). From the subsequent steps you must judge whether the Agent adopted the XPU's idea and whether that idea was effective.

## XPUs recommended last time
{xpu_list}

## Situation when recommended (the problem the Agent faced)
{situation}

## Subsequent steps after the recommendation (what the Agent did next)
{subsequent_steps}

## Decision rules (judge each XPU separately)
- success: the Agent adopted the XPU's idea (possibly with a different command but the same approach), and subsequent steps show the problem was solved or clearly improved.
- failure: the Agent adopted the XPU's idea, but the problem was not solved or new problems were introduced.
- neutral: cannot tell whether the Agent adopted the suggestion, or the suggestion is unrelated to the subsequent steps.

## Decision examples

### Example 1: success (Agent generated its own command from the idea; problem solved)
- XPU advice: "Docker container does not pre-install Python3; install it via the package manager."
- Situation: "bash: python3: command not found"
- Subsequent steps:
  [SHELL_COMMAND] apt-get update && apt-get install -y python3 python3-pip → exit=0
  [SHELL_COMMAND] python3 --version → exit=0: Python 3.10.12
→ Verdict: success, score=0.90
→ Reason: the Agent adopted the "install Python3 via package manager" idea, apt-get install succeeded, python3 is available.

### Example 2: success (same idea, different command, problem improved)
- XPU advice: "Install Poetry to manage project dependencies."
- Situation: "poetry: command not found"
- Subsequent steps:
  [SHELL_COMMAND] pip3 install poetry → exit=0
  [SHELL_COMMAND] poetry install → exit=0
→ Verdict: success, score=0.75
→ Reason: XPU suggested installing Poetry; the Agent installed it via pip3 (not the curl method the XPU mentioned), but the idea is the same and the problem is solved.

### Example 3: failure (suggestion adopted but introduces new problems)
- XPU advice: "Downgrade setuptools to fix the compatibility issue."
- Situation: "AttributeError: module 'setuptools' has no attribute 'setup'"
- Subsequent steps:
  [SHELL_COMMAND] pip install setuptools==58.0.0 → exit=0
  [SHELL_COMMAND] pip install -e . → exit=1: ERROR: Could not build wheels
→ Verdict: failure, score=-0.60
→ Reason: the Agent downgraded setuptools as suggested, but the project can no longer be built, introducing a new problem.

### Example 4: neutral (suggestion unrelated to subsequent actions)
- XPU advice: "Install Redis as the cache backend."
- Situation: "ConnectionRefusedError: Redis server not available"
- Subsequent steps:
  [SHELL_COMMAND] pip install pytest-cov → exit=0
  [SHELL_COMMAND] pytest tests/ -x → exit=1: ImportError: No module named flask
→ Verdict: neutral, score=0.0
→ Reason: the Agent did not install Redis; subsequent steps are completely unrelated to the XPU suggestion.

### Example 5: neutral (too few subsequent steps to judge)
- XPU advice: "Install the system-level dependency libxml2-dev via apt."
- Situation: "error: command 'gcc' failed"
- Subsequent steps:
  [SET_ENV] PATH=/usr/local/bin:$PATH
→ Verdict: neutral, score=0.0
→ Reason: only one environment-variable step follows; cannot tell whether the XPU was adopted or effective.

### Example 6: success but limited help (score in 0.2–0.5)
- XPU advice: "The project uses tox to run tests; install tox first."
- Situation: "tox: command not found"
- Subsequent steps:
  [SHELL_COMMAND] pip install tox → exit=0
  [SHELL_COMMAND] tox -e py310 → exit=1: ERROR: missing dependency numpy
→ Verdict: success, score=0.35
→ Reason: the Agent adopted the install-tox suggestion, tox installed successfully, but subsequent runs still hit other dependency issues; the suggestion only solved part of the problem.

### Example 7: failure but with little harm (score in -0.5 to -0.2)
- XPU advice: "Install dependencies via pip install -r requirements.txt."
- Situation: "ModuleNotFoundError: No module named 'yaml'"
- Subsequent steps:
  [SHELL_COMMAND] pip install -r requirements.txt → exit=1: ERROR: No matching distribution found for some-internal-pkg
  [SHELL_COMMAND] pip install pyyaml → exit=0
→ Verdict: failure, score=-0.30
→ Reason: the Agent tried the suggested requirements.txt install but it failed; in the end the Agent solved it with a separate pip install. The suggestion did not help, but caused no serious harm.

## Scoring rubric (continuous interval -1.0 to 1.0, no gaps)

score is a continuous value representing how much the XPU helped the current deployment:

| Score range | verdict | Meaning |
|---|---|---|
| 0.8 to 1.0 | success | Suggestion fully adopted; problem completely solved |
| 0.5 to 0.8 | success | Same idea; problem clearly improved (command may differ) |
| 0.2 to 0.5 | success | Partially adopted or indirectly helpful; some positive effect |
| -0.2 to 0.2 | neutral | Cannot tell whether adopted; or unrelated to subsequent steps |
| -0.5 to -0.2 | failure | Adopted but ineffective; problem unsolved |
| -0.8 to -0.5 | failure | Suggestion ineffective; wasted steps |
| -1.0 to -0.8 | failure | Suggestion directly caused new serious problems |

Note: the verdict field should be consistent with score, but the system trusts the verdict string ("success" / "failure" / "neutral") for telemetry updates; the numeric score is recorded for analysis only.

You must reply in JSON:
{{
  "verdicts": [
    {{
      "xpu_id": "...",
      "verdict": "success | failure | neutral",
      "score": 0.8,
      "reason": "brief reason"
    }}
  ]
}}
"""


class RetrieverAgent:
    """XPU knowledge-retrieval sub-agent.

    Runs in an isolated context; does not pollute the main agent's prompt.
    The main agent only sees the final XPUSuggestion list.

    Architecture:
      RetrieverAgent
      |-- XpuVectorStore: vector DB (Layer-1 recall).
      |-- LLMClientBase:  LLM client (Layer-2 refinement + audit).
      `-- LastXPURecord:  previous-recommendation record (delayed audit).
    """

    def __init__(self, vector_store, llm_client):
        """Init.

        Args:
            vector_store: XpuVectorStore instance.
            llm_client: LLMClientBase instance for refinement and audit.
        """
        self._store = vector_store
        self._llm = llm_client
        # Last recommendation, kept for delayed audit.
        self._last_xpu_record: LastXPURecord | None = None
        logger.info("RetrieverAgent initialized")

    # =========================================================================
    # Public: retrieve XPU suggestions
    # =========================================================================

    def retrieve(
        self,
        situation: str,
        exclude_ids: list[str] | None = None,
        full_history: list[dict] | None = None,
        k: int = 3,
        n_candidates: int = 10,
    ) -> list[XPUSuggestion]:
        """Retrieve the most relevant XPU suggestions (two-layer + delayed audit).

        Pipeline:
        1. If a previous recommendation exists, run the delayed audit using
           the anchor inside full_history.
        2. Layer 1: vector recall, Top-N.
        3. Layer 2: LLM refinement, pick K from N (skipped in direct mode).
        4. Auto-record this round's recommendation for the next audit.

        Args:
            situation: current deployment situation (what was done / what is being done / current error).
            exclude_ids: XPU IDs already tried this run.
            full_history: full main-agent history, used to slice subsequent steps for the delayed audit.
            k: final number of suggestions (default 3).
            n_candidates: Layer-1 candidate pool size (default 10).

        Returns:
            Refined list of XPUSuggestion (at most k).
        """
        history = full_history or []

        # === Step 0: delayed audit on the previous recommendation ===
        if self._last_xpu_record:
            self._do_delayed_audit(history)

        # RETRIEVER_MODE=direct: pure vector Top-K, skip LLM refinement.
        direct_mode = os.environ.get("RETRIEVER_MODE") == "direct"

        # === Step 1: Layer-1 vector retrieval ===
        search_k = k if direct_mode else n_candidates
        logger.info(f"[Layer 1] vector {'direct retrieval' if direct_mode else 'recall'}, N={search_k}")
        try:
            from .xpu.xpu_vector_store import text_to_embedding
            embedding = text_to_embedding(situation)
            candidates = self._store.search(
                embedding,
                k=search_k,
                exclude_ids=exclude_ids,
            )
        except Exception as e:
            logger.warning(f"vector retrieval failed: {e}")
            return []

        if not candidates:
            logger.info("[Layer 1] no candidate XPU found")
            return []

        logger.info(f"[Layer 1] {len(candidates)} candidates")

        # === Step 2: Layer-2 LLM refinement (skipped in direct mode) ===
        if direct_mode:
            logger.info("[direct mode] skipping LLM refinement; using vector results directly")
            suggestions = self._candidates_to_suggestions(candidates[:k])
        else:
            suggestions = self._refine_with_llm(situation, candidates, k)

        # Bulk-update hits counter for every returned XPU.
        if suggestions:
            try:
                self._store.increment_telemetry(
                    [s.id for s in suggestions], "hits"
                )
            except Exception as e:
                logger.warning(f"failed to update hits counter: {e}")

        # === Step 3: record this round for the next audit ===
        if suggestions:
            self._last_xpu_record = LastXPURecord(
                xpu_ids=[s.id for s in suggestions],
                descriptions=[s.description for s in suggestions],
                situation=situation,
                state_before=situation,  # situation already contains the current error info
                step_index=len(history),  # anchor: current history length
            )
            logger.info(
                f"recorded recommendation: ids={[s.id for s in suggestions]}, "
                f"anchor step_index={len(history)}, awaiting next audit"
            )

        logger.info(f"[Layer 2] returning {len(suggestions)} suggestions")
        return suggestions

    # =========================================================================
    # Internal: LLM refinement
    # =========================================================================

    def _refine_with_llm(
        self,
        situation: str,
        candidates: list[dict],
        k: int,
    ) -> list[XPUSuggestion]:
        """Layer 2: refine candidate XPUs via a single LLM call.

        Pack all candidates into one prompt; one LLM call returns the picks.

        Args:
            situation: current deployment situation.
            candidates: Layer-1 candidate list.
            k: number to keep.

        Returns:
            Refined list of XPUSuggestion.
        """
        from .xpu.xpu_adapter import XpuAtom, render_atom_to_commands

        # Build candidate text
        candidate_lines = []
        candidate_map = {}  # xpu_id -> raw candidate
        for i, c in enumerate(candidates):
            telemetry = c.get("telemetry") or {}
            hits = telemetry.get("hits", 0)
            successes = telemetry.get("successes", 0)
            failures = telemetry.get("failures", 0)

            advice = c.get("advice_nl") or []
            advice_text = " | ".join(advice) if isinstance(advice, list) else str(advice)

            similarity = c.get("similarity", 0)
            composite = c.get("composite_score", similarity)
            tier = c.get("tier", "normal")
            tier_label = {"golden": "* golden", "normal": "normal", "cold": "v cold"}.get(tier, tier)

            line = (
                f"[{i+1}] ID: {c['id']}  tier: {tier_label}\n"
                f"    advice: {advice_text}\n"
                f"    similarity: {similarity:.3f}, composite: {composite:.3f}\n"
                f"    history: hits={hits}, successes={successes}, failures={failures}"
            )
            candidate_lines.append(line)
            candidate_map[c["id"]] = c

        candidates_text = "\n\n".join(candidate_lines)

        prompt = REFINE_PROMPT.format(
            situation=situation,
            n_candidates=len(candidates),
            candidates_text=candidates_text,
            k=k,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Pick the most relevant XPU suggestions from the candidate list."},
        ]

        logger.info(f"[Layer 2] calling LLM to refine {len(candidates)} candidates")

        try:
            response = self._llm.chat(messages, json_mode=True)
            data = self._parse_json_response(response)
        except Exception as e:
            logger.warning(f"LLM refinement failed; falling back to recall result: {e}")
            return self._candidates_to_suggestions(candidates[:k])

        selected = data.get("selected", [])
        if not selected:
            logger.warning("LLM picked nothing; falling back to recall result")
            return self._candidates_to_suggestions(candidates[:k])

        # Build XPUSuggestion in LLM-pick order.
        suggestions = []
        for item in selected[:k]:
            xpu_id = item.get("xpu_id", "")
            if xpu_id not in candidate_map:
                continue

            c = candidate_map[xpu_id]
            reason = item.get("relevance_reason", "")
            confidence = float(item.get("confidence", c.get("composite_score", 0.5)))

            # Render atoms into executable commands.
            atoms = c.get("atoms") or []
            commands = []
            for a in atoms:
                atom = XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                commands.extend(render_atom_to_commands(atom))

            advice = c.get("advice_nl") or []
            advice_text = "\n".join(advice)
            if reason:
                description = f"[experience] {advice_text}\n[match reason] {reason}"
            else:
                description = advice_text

            suggestions.append(XPUSuggestion(
                id=xpu_id,
                description=description,
                commands=commands,
                confidence=confidence,
                source="retriever_agent",
            ))

        return suggestions

    # =========================================================================
    # Internal: delayed audit
    # =========================================================================

    def _do_delayed_audit(self, full_history: list[dict]) -> None:
        """Audit the previous round's XPU recommendation.

        Slice the subsequent steps from the anchor (step_index) — at most 5 steps
        — and ask the LLM to judge each recommended XPU's effectiveness.

        Args:
            full_history: full main-agent history.
        """
        record = self._last_xpu_record
        if not record:
            return

        logger.info(f"[audit] auditing XPU: {record.xpu_ids}")

        # From the anchor, take up to 5 subsequent steps.
        subsequent_entries = full_history[record.step_index : record.step_index + 5]

        if not subsequent_entries:
            logger.info("[audit] no subsequent steps after recommendation; skip audit")
            self._last_xpu_record = None
            return

        # Compress each step to one summary line.
        subsequent = []
        for entry in subsequent_entries:
            action = entry.get("action", {})
            result = entry.get("result", {})
            action_type = action.get("action_type", "unknown")
            cmd = action.get("content", {}).get("command", "")
            exit_code = result.get("exit_code", "?")
            stdout = (result.get("stdout") or "")[:150]
            subsequent.append(f"[{action_type}] {cmd} -> exit={exit_code}: {stdout}")

        # Build XPU list text.
        xpu_lines = []
        for xpu_id, desc in zip(record.xpu_ids, record.descriptions):
            xpu_lines.append(f"- ID: {xpu_id}\n  advice: {desc[:300]}")
        xpu_list_text = "\n".join(xpu_lines)

        prompt = AUDIT_PROMPT.format(
            xpu_list=xpu_list_text,
            situation=record.situation[:500],
            subsequent_steps="\n".join(subsequent),
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Judge each XPU suggestion separately."},
        ]

        try:
            response = self._llm.chat(messages, json_mode=True)
            data = self._parse_json_response(response)

            verdicts_data = data.get("verdicts", [])
            for v in verdicts_data:
                verdict = AuditVerdict(
                    xpu_id=v.get("xpu_id", ""),
                    verdict=v.get("verdict", "neutral"),
                    score=float(v.get("score", 0.0)),
                    reason=v.get("reason", ""),
                )
                logger.info(f"[audit] {verdict}")
                self._update_telemetry_from_audit(verdict)

        except Exception as e:
            logger.warning(f"[audit] LLM audit failed: {e}")

        # Clear record to avoid re-auditing.
        self._last_xpu_record = None

    def _update_telemetry_from_audit(self, verdict: AuditVerdict) -> None:
        """Update XPU telemetry based on the verdict string (not the numeric score).

        Rules:
        - verdict == "success" -> successes +=1
        - verdict == "failure" -> failures +=1
        - others (incl. "neutral" and unknown) -> no telemetry update

        Args:
            verdict: LLM audit result.
        """
        if not verdict.xpu_id:
            return
        v = (verdict.verdict or "").strip().lower()
        try:
            if v == "success":
                self._store.increment_telemetry([verdict.xpu_id], "successes")
                logger.info(
                    f"[audit] XPU {verdict.xpu_id}: verdict=success "
                    f"(score={verdict.score:.2f}) -> successes +1"
                )
            elif v == "failure":
                self._store.increment_telemetry([verdict.xpu_id], "failures")
                logger.info(
                    f"[audit] XPU {verdict.xpu_id}: verdict=failure "
                    f"(score={verdict.score:.2f}) -> failures +1"
                )
            else:
                logger.info(
                    f"[audit] XPU {verdict.xpu_id}: verdict={v or 'neutral'} "
                    f"(score={verdict.score:.2f}) -> no telemetry update"
                )
        except Exception as e:
            logger.warning(f"[audit] telemetry update failed: {e}")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _candidates_to_suggestions(self, candidates: list[dict]) -> list[XPUSuggestion]:
        """Convert raw candidates to XPUSuggestion (fallback path).

        Used when LLM refinement fails — fall back to Layer-1 results directly.
        """
        from .xpu.xpu_adapter import XpuAtom, render_atom_to_commands

        suggestions = []
        for c in candidates:
            atoms = c.get("atoms") or []
            commands = []
            for a in atoms:
                atom = XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                commands.extend(render_atom_to_commands(atom))

            advice = c.get("advice_nl") or []
            similarity = float(c.get("similarity", 0.5))
            composite = float(c.get("composite_score", similarity))

            suggestions.append(XPUSuggestion(
                id=c["id"],
                description="\n".join(advice) if isinstance(advice, list) else str(advice),
                commands=commands,
                confidence=composite,
                source="vector_db_fallback",
            ))

        return suggestions

    def _parse_json_response(self, response: str) -> dict:
        """Parse the LLM JSON response (tolerates markdown code fences)."""
        # Strip <think>...</think> blocks (qwen-style).
        if "<think>" in response:
            response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"could not parse LLM response as JSON: {response[:200]}")

    def close(self, full_history: list[dict] | None = None) -> None:
        """Release resources; run a final audit if one is still pending.

        Args:
            full_history: full main-agent history, for the final audit.
        """
        if self._last_xpu_record:
            if full_history:
                logger.info("[close] running final audit...")
                self._do_delayed_audit(full_history)
            else:
                logger.info("[close] pending audit record exists but no history; skip")
                self._last_xpu_record = None
