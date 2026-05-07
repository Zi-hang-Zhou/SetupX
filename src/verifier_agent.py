"""Verification sub-agent (lightweight ReAct).

Pure inspection: decides whether the environment configured by the Setup Agent is acceptable.
- Only runs tests, observes results, makes a verdict; never installs packages or mutates the env.
- Failure caused by setup leftovers     -> success=False
- Failure caused by project-inherent issues (external services / test bugs) -> success=True

The setup agent is a black box from this side; only a VerifyResult is returned.
"""

import base64
import json
import re

from .llm_engine import ARKClient, OpenAICompatibleClient
from .config import get_config
from .environment_manager import EnvironmentManager
from .logger import get_logger
from .models import VerifyResult

logger = get_logger("verifier_agent")

# Step count is unbounded; total Phase 1 wall-time is bounded by main.py via signal.alarm(phase1_timeout).
MAX_STEPS = 9999

SYSTEM_PROMPT = """\
You are a verification Agent working inside a Docker container.
Your task: **examine** whether the environment configured by the Setup Agent is acceptable, and report the result truthfully.

Your role is **a prosecutor, not a fixer.**
You are not in charge of fixing anything; you only run tests, observe results, and pass judgment.

## Verification procedure

**Step 1 is always structure reconnaissance — do not skip:**
```
ls /workspace/repo
cat /workspace/repo/pyproject.toml 2>/dev/null || cat /workspace/repo/setup.cfg 2>/dev/null || cat /workspace/repo/setup.py 2>/dev/null
ls /workspace/repo/tests 2>/dev/null || ls /workspace/repo/test 2>/dev/null
```
Decide how to run the tests only after you have a clear picture of the project structure and the test entry point.

1. Structure reconnaissance (mandatory): `ls` the project root, locate pyproject.toml / setup.cfg / pytest.ini / tox.ini, etc.
2. Locate the test suite: confirm the test directory and framework (pytest / unittest / tox, ...).
3. Run the tests in the project's native way and collect results.
4. Analyze failure causes and make a judgment (see below).
5. If the project has no tests at all, write a smoke test under /tmp/ to verify basic environment usability.

## Judgment standard: success=True or False

After running the tests, analyze each failure / error by root cause:

**success=False (Setup-residual problem):**
- Missing Python package (ImportError, ModuleNotFoundError)
- Wrong path / PYTHONPATH configuration
- Project not installed correctly (e.g., missing editable install)
- Version incompatibility between already-installed packages (e.g., django-reviews 1.x clashing with Django 5.x and producing TypeError/AttributeError) — the Setup Agent should have installed compatible versions
- Anything the Setup Agent was supposed to handle but did not

**success=True (project-inherent limitation, not Setup's responsibility):**
- Test-logic bug (assertion errors, platform-specific issues)
- Dependence on external services (database, API, network) that cannot run inside the container
- Tests of optional dependencies that were skipped (skipif)
- Missing test data (not solvable in the setup stage)

Every judgment must be backed by evidence. The hint must record: what command you ran, what output you observed, and why you reached the conclusion.

## Tools (one per step, must respond with valid JSON)

{"thought": "current observation and next-step reasoning", "action": "exec_run", "args": {"command": "shell command"}}
{"thought": "...", "action": "write_file", "args": {"path": "/tmp/xxx.py", "content": "file content"}}
{"thought": "...", "action": "finish", "args": {"success": true, "hint": "brief note", "test_framework": "pytest", "collect_count": 12}}

## Hard constraints (violation invalidates the verdict)

- **Install no packages**: pip install / apt install / apt-get install / conda install — all forbidden.
- **Modify no environment configuration**: no exporting environment variables, no editing .bashrc / .profile / PATH, etc.
- **Modify no file under /workspace/repo.**
- **write_file may only write into /tmp/** (for smoke tests only; never for monkey-patching or bypassing tests).

If a test fails because of a missing package, the correct response is to report success=False and clearly state which package is missing — not to install it.
"""


class VerifierAgent:
    """Lightweight ReAct verification sub-agent."""

    def __init__(self, env: EnvironmentManager, max_steps: int = MAX_STEPS, setup_summary: str = "", hint: str = ""):
        self._env = env
        self._max_steps = max_steps
        self._setup_summary = setup_summary
        self._hint = hint
        self._llm = self._build_llm_client()

    def _build_llm_client(self):
        """Reuse llm_engine clients; do not re-implement."""
        config = get_config()
        if config.llm_provider == "ark":
            return ARKClient(config.ark)
        elif config.llm_provider == "openai":
            if config.openai is None:
                raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not configured")
            return OpenAICompatibleClient(config.openai)
        else:
            raise ValueError(f"unsupported LLM provider: {config.llm_provider}")

    def verify(self) -> VerifyResult:
        """ReAct main loop; returns a VerifyResult."""
        logger.info("verifier sub-agent started")

        if self._hint:
            logger.info(f"[Verifier] received hint from Setup Agent: {self._hint}")
            first_user_msg = f"[Setup Agent hint] {self._hint}\n\nBegin verification."
        elif self._setup_summary:
            logger.info(f"[Verifier] received setup handoff: {self._setup_summary}")
            first_user_msg = (
                f"Setup Agent handoff (for reference only; verify independently):\n{self._setup_summary}\n\nBegin verification."
            )
        else:
            first_user_msg = "Begin verification."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": first_user_msg},
        ]

        for step in range(1, self._max_steps + 1):
            logger.info(f"=== Verifier Step {step}/{self._max_steps} ===")

            raw = self._llm.chat(messages, json_mode=True)
            logger.info(f"LLM output: {raw}")
            messages.append({"role": "assistant", "content": raw})

            try:
                parsed = self._parse_json(raw)
            except Exception as e:
                obs = f"JSON parse failed: {e}; please respond with valid JSON."
                logger.warning(obs)
                messages.append({"role": "user", "content": obs})
                continue

            action = parsed.get("action", "")
            args = parsed.get("args", {})
            thought = parsed.get("thought", "")
            logger.info(f"action={action}, thought={thought[:80]}")

            # finish
            if action == "finish":
                success = bool(args.get("success", False))
                finish_hint = str(args.get("hint", ""))
                collect_count = int(args.get("collect_count", 0))
                test_framework = str(args.get("test_framework", "unknown"))
                logger.info(f"verification done: success={success}, hint={finish_hint}")
                self._llm.close()
                return VerifyResult(
                    success=success,
                    test_framework=test_framework,
                    collect_count=collect_count,
                    command=args.get("command", ""),
                    exit_code=0 if success else 1,
                    stdout=finish_hint,
                    stderr="",
                    messages=list(messages),
                )

            # exec_run
            elif action == "exec_run":
                cmd = args.get("command", "")
                if not cmd:
                    obs = "Error: exec_run is missing the 'command' argument."
                else:
                    result = self._env.exec_run(cmd)
                    obs = (
                        f"exit_code={result.exit_code}\n"
                        f"stdout:\n{result.stdout}\n"
                        f"stderr:\n{result.stderr}"
                    )
                    logger.debug(f"exec_run [{cmd}] -> exit_code={result.exit_code}")
                messages.append({"role": "user", "content": f"Command result:\n{obs}"})

            # write_file
            elif action == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                if not path.startswith("/tmp/"):
                    obs = "Error: write_file is only allowed under /tmp/."
                else:
                    ok = self._write_file(path, content)
                    obs = f"write_file {'ok' if ok else 'failed'}: {path}"
                    logger.debug(obs)
                messages.append({"role": "user", "content": obs})

            else:
                obs = f"Unknown action='{action}'; only exec_run / write_file / finish are allowed."
                logger.warning(obs)
                messages.append({"role": "user", "content": obs})

        logger.warning("verifier hit max steps without finishing")
        self._llm.close()
        return VerifyResult(
            success=False,
            test_framework="unknown",
            collect_count=0,
            command="",
            exit_code=-1,
            stdout="",
            stderr=f"verifier hit max steps {self._max_steps}",
            messages=list(messages),
        )

    def _write_file(self, path: str, content: str) -> bool:
        """Write a file via base64 to bypass shell quoting issues."""
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        cmd = (
            f"python3 -c \""
            f"import base64; "
            f"open('{path}', 'w').write(base64.b64decode('{b64}').decode('utf-8'))"
            f"\""
        )
        result = self._env.exec_run(cmd)
        return result.success

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Lenient JSON extraction from LLM output."""
        raw = raw.strip()
        # Strip qwen-style <think>...</think> reasoning blocks.
        if "<think>" in raw:
            raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
        # 1. direct parse
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        # 2. ```json ... ``` block
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        # 3. bare {...}
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"could not extract JSON: {raw[:200]}")
