"""Judge sub-agent (per-charge verification).

Verifies each prosecutor charge with targeted commands, then issues a verdict.
- No open-ended exploration (that's the prosecutor's job).
- Runs at most a few verification commands per charge.
- May dismiss unreasonable charges.
"""

import json
import re

from .llm_engine import ARKClient, OpenAICompatibleClient
from .config import get_config
from .logger import get_logger
from .models import ProsecutionResult

logger = get_logger("judge")

# Max verification commands per charge.
MAX_VERIFY_PER_CHARGE = 2

SYSTEM_PROMPT = """\
You are the Judge. Your job is to **verify the prosecutor's charges one by one** and then issue a verdict.

You are not the prosecutor — you do not conduct open-ended investigation. Your responsibility:
for each charge raised by the prosecutor, execute 1–2 verification commands to confirm whether that charge holds.

## Trial procedure

You will receive the prosecutor's list of charges. For each charge:

1. **Read the charge and its evidence.**
2. **Design one verification command** that reproduces the prosecutor's finding (e.g., if the prosecutor says `import X` fails, you also run `import X`).
3. **Decide whether the charge holds based on the verification result.**

## Decision criteria

**Charge upheld**: your verification result matches the prosecutor's (the dependency is indeed not importable, compilation indeed fails, ...).
**Charge dismissed**:
- your verification result contradicts the prosecutor's (the dependency is in fact importable);
- the prosecutor misjudged the project type (e.g., demanded Python dependencies on a C++/Java/JS project);
- the dependency is only in optional extras, not a core dependency;
- the failure is due to an external service / network / test-logic bug, not a Setup Agent dereliction;
- the prosecutor used the wrong environment (e.g., system python3 instead of the project's venv/conda).

## Final verdict

- ≥1 charge upheld after your verification → **guilty**
- All charges dismissed → **not_guilty**

## Output format (each step must emit one valid JSON object)

For each charge: verify, then decide. After all charges have been processed, emit the final verdict:

{"thought": "verifying charge N: ...", "action": "exec_run", "args": {"command": "verification command"}}
{"thought": "all charges have been verified", "action": "verdict", "args": {
  "verdict": "guilty or not_guilty",
  "reasoning": "Charge 1 upheld/dismissed (reason); Charge 2 ...; overall verdict",
  "charges_review": [
    {"charge_index": 1, "upheld": true, "reason": "verification confirmed that dependency X is not importable"},
    {"charge_index": 2, "upheld": false, "reason": "dependency is in fact importable; the prosecutor used system python instead of the venv"}
  ]
}}

## Hard constraints

- **Install no packages**, modify no environment — read-only forensics.
- **No open-ended exploration**: verify only the prosecutor's specific charges; do not look for new problems on your own.
- **Independent judgment**: a prosecutor's claim of guilt does not entail guilt; your verification result is the basis.
"""


class JudgeAgent:
    """Judge: per-charge verification with limited container access."""

    def __init__(
        self,
        setup_history: list[dict],
        verify_messages: list[dict],
        prosecution: ProsecutionResult,
        env: "EnvironmentManager | None" = None,
    ):
        self._setup_history = setup_history
        self._verify_messages = verify_messages
        self._prosecution = prosecution
        self._env = env
        self._llm = self._build_llm_client()

    def _build_llm_client(self):
        config = get_config()
        if config.llm_provider == "ark":
            return ARKClient(config.ark)
        elif config.llm_provider == "openai":
            if config.openai is None:
                raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not configured")
            return OpenAICompatibleClient(config.openai)
        else:
            raise ValueError(f"unsupported LLM provider: {config.llm_provider}")

    def rule(self) -> dict:
        """Run the trial; returns {"verdict": "guilty"|"not_guilty", "reasoning": "..."}."""
        if not self._prosecution.prosecute:
            logger.info("prosecutor declined to prosecute -> verdict=not_guilty")
            self._llm.close()
            return {"verdict": "not_guilty", "reasoning": "prosecutor declined to prosecute; no substantive issue found"}

        if self._env is None:
            logger.warning("judge has no container access, falling back to paper trial")
            return self._paper_trial()

        logger.info(f"judge begins per-charge verification ({len(self._prosecution.charges)} charges)")
        return self._verify_charges()

    def _verify_charges(self) -> dict:
        """Per-charge verification mode."""
        charges = self._prosecution.charges
        # Phase 2 judge is unbounded by step count.
        max_steps = 9999

        # Environment snapshot helps the judge see the actual container state
        # (e.g., a venv the prosecutor may have missed).
        env_snapshot = self._env.get_env_snapshot()
        logger.info(f"judge env snapshot:\n{env_snapshot[:300]}")

        prosecution_summary = self._format_prosecution()
        setup_summary = self._format_setup_history()
        verify_summary = self._format_verify_messages()

        user_content = (
            f"## Container env snapshot\n\n```\n{env_snapshot}\n```\n\n"
            f"## Setup Agent trajectory (last 20 steps)\n\n{setup_summary}\n\n"
            f"## Verifier conversation\n\n{verify_summary}\n\n"
            f"## Prosecution ({len(charges)} charges)\n\n{prosecution_summary}\n\n"
            f"Verify each charge above. You may run at most {MAX_VERIFY_PER_CHARGE} verification commands per charge. "
            f"After all charges have been verified, output the final verdict."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        effective_step = 0
        api_failures = 0
        for step in range(1, max_steps * 3 + 1):  # leave headroom for retries
            if effective_step >= max_steps:
                break
            logger.info(f"=== Judge Step {effective_step+1}/{max_steps} (raw={step}) ===")

            try:
                raw = self._llm.chat(messages, json_mode=True)
            except Exception as e:
                api_failures += 1
                logger.warning(f"Judge LLM call failed (not counted as a step): {e}, api_failures={api_failures}")
                if api_failures >= 5:
                    logger.error("Judge API failing repeatedly; abort verification")
                    break
                continue
            effective_step += 1

            logger.info(f"LLM output: {raw[:300]}")
            messages.append({"role": "assistant", "content": raw})

            try:
                parsed = self._parse_json(raw)
            except Exception as e:
                messages.append({"role": "user", "content": f"JSON parse failed: {e}; please respond again."})
                continue

            action = parsed.get("action", "")
            args = parsed.get("args", {})

            if action == "verdict":
                verdict = args.get("verdict", "not_guilty")
                reasoning = args.get("reasoning", "")
                charges_review = args.get("charges_review", [])
                upheld = sum(1 for c in charges_review if c.get("upheld"))
                dismissed = len(charges_review) - upheld
                logger.info(
                    f"judge verdict: {verdict} "
                    f"(upheld={upheld}, dismissed={dismissed})"
                )
                self._llm.close()
                return {"verdict": verdict, "reasoning": reasoning}

            elif action == "exec_run":
                cmd = args.get("command", "")
                if not cmd:
                    messages.append({"role": "user", "content": "Error: exec_run is missing 'command'."})
                else:
                    result = self._env.exec_run(cmd)
                    obs = (
                        f"exit_code={result.exit_code}\n"
                        f"stdout:\n{result.stdout}\n"
                        f"stderr:\n{result.stderr}"
                    )
                    messages.append({"role": "user", "content": f"Verification result:\n{obs}"})

            else:
                messages.append({"role": "user", "content": f"Unknown action='{action}'; only exec_run / verdict allowed"})

        # Step-cap reached: force a verdict.
        logger.warning("judge hit step cap; requesting final verdict")
        messages.append({
            "role": "user",
            "content": "Step cap reached. Please immediately output the final verdict (verdict action) based on existing verification results.",
        })
        try:
            raw = self._llm.chat(messages, json_mode=True)
            parsed = self._parse_json(raw)
            if parsed.get("action") == "verdict":
                self._llm.close()
                return {
                    "verdict": parsed["args"].get("verdict", "guilty"),
                    "reasoning": parsed["args"].get("reasoning", "step cap"),
                }
        except Exception as e:
            logger.error(f"forced verdict failed: {e}")

        self._llm.close()
        # Judge could not finish verification -> mark as anomaly, do not make a guilty/not_guilty call.
        return {"verdict": "error", "reasoning": "judge verification did not finish (API failure or step cap)"}

    def _paper_trial(self) -> dict:
        """Paper trial (no container; backwards-compat path)."""
        setup_summary = self._format_setup_history()
        prosecution_summary = self._format_prosecution()

        paper_prompt = (
            "You are the Judge. Issue a verdict based solely on the materials below.\n"
            "Note: you have no container access; you can only judge from the written record.\n"
            "If the prosecutor's evidence is weak or the charges are unreasonable, rule not_guilty.\n\n"
            "Charges should be dismissed when:\n"
            "- the prosecutor misjudged the project type (e.g., checked Python dependencies on a C++ project);\n"
            "- the dependency is only in optional extras, not a core dependency;\n"
            "- the failure is due to an external service / network / test-logic bug;\n"
            "- the prosecutor likely used the wrong environment (system python3 vs the project's venv).\n\n"
            "Output format (a valid JSON object): {\"verdict\": \"guilty\"|\"not_guilty\", \"reasoning\": \"basis of the verdict\"}"
        )

        user_content = (
            f"## Setup Agent trajectory\n\n{setup_summary}\n\n"
            f"## Prosecutor investigation report\n\n{prosecution_summary}\n\n"
            "Please issue a verdict."
        )

        messages = [
            {"role": "system", "content": paper_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = self._llm.chat(messages, json_mode=True)
        logger.info(f"judge paper-trial verdict: {raw[:500]}")
        self._llm.close()

        try:
            result = self._parse_json(raw)
            return {"verdict": result.get("verdict", "error"), "reasoning": result.get("reasoning", "")}
        except Exception as e:
            logger.error(f"failed to parse verdict: {e}")
            return {"verdict": "error", "reasoning": f"failed to parse verdict: {e}"}

    def _format_setup_history(self) -> str:
        recent = self._setup_history[-20:]
        lines = []
        for entry in recent:
            step = entry.get("step", "?")
            action = entry.get("action", {})
            result = entry.get("result", {})
            action_type = action.get("action_type", "?")
            content = action.get("content", {})
            thought = action.get("thought", "")[:100]
            exit_code = result.get("exit_code", "?")
            stdout = (result.get("stdout") or "")[:200]
            lines.append(
                f"[step {step}] {action_type} | thought: {thought}\n"
                f"  content: {json.dumps(content, ensure_ascii=False)[:150]}\n"
                f"  result: exit_code={exit_code}, stdout: {stdout}"
            )
        return "\n\n".join(lines) if lines else "(no history)"

    def _format_verify_messages(self) -> str:
        if not self._verify_messages:
            return "(no verifier conversation)"
        lines = []
        for msg in self._verify_messages:
            role = msg.get("role", "?")
            content = (msg.get("content") or "")[:300]
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    def _format_prosecution(self) -> str:
        if not self._prosecution.prosecute:
            return "Prosecutor declined to prosecute: no substantive issue found."
        lines = ["The prosecutor brings the following charges:\n"]
        for i, charge in enumerate(self._prosecution.charges, 1):
            claim = charge.get("claim", "")
            evidence = charge.get("evidence", "")
            lines.append(f"**Charge {i}**: {claim}\nEvidence:\n{evidence}\n")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"could not extract JSON: {raw[:200]}")
