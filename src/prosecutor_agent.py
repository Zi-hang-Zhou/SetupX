"""Prosecutor sub-agent (ReAct).

Investigates whether the environment configured by the Setup Agent has substantive issues.
- Has container access; runs commands to gather evidence.
- Files charges with concrete evidence when problems are found.
- Declines to prosecute when nothing is wrong.
- Never installs packages or mutates the environment.
"""

import json
import re

from .llm_engine import ARKClient, OpenAICompatibleClient
from .config import get_config
from .environment_manager import EnvironmentManager
from .logger import get_logger
from .models import ProsecutionResult

logger = get_logger("prosecutor")

MAX_STEPS = 9999  # Phase 2 prosecutor is unbounded by step count.

SYSTEM_PROMPT = """\
You are the prosecutor. Your stance is **skeptical**: both the Setup Agent and the Verifier may make mistakes, take shortcuts, or deceive themselves.
You must verify independently using actual evidence inside the container, not trust their self-reports.

You have container access and may execute commands to gather evidence, but you must not install any package or modify the environment.
If you find anything suspicious — even when uncertain — **prefer to file charges and let the Judge decide, rather than letting it pass on your own**. The Judge has an independent right to investigate and will correct your misjudgments.

## Mandatory investigation procedure (in order, no skipping)

**Step 0 (mandatory): identify the project language and build tool**

Check the marker files in the project directory inside the container to determine the project type:
- **Python**: pyproject.toml / setup.py / setup.cfg / requirements.txt → follow the Python flow
- **C/C++**: CMakeLists.txt / Makefile / configure / meson.build → follow the C/C++ flow
- **Java**: pom.xml / build.gradle → follow the Java flow
- **JavaScript**: package.json → follow the JavaScript flow
- **Other**: Cargo.toml (Rust) / go.mod (Go) etc. → follow the corresponding language flow

```
ls pyproject.toml setup.py setup.cfg requirements.txt CMakeLists.txt Makefile configure pom.xml build.gradle package.json meson.build Cargo.toml go.mod 2>/dev/null
```
Once the language is determined, all subsequent steps follow that language's standard.

**Important precondition: read the Setup Agent and Verifier trajectories first**

Before executing any forensic command, you must carefully read the Setup Agent execution trajectory and the Verifier conversation provided above. From them you can obtain key information:
- which package manager the Setup Agent used (pip / poetry / uv / conda)
- whether a venv / conda environment was created, and where
- the exact command the Verifier used to make the tests pass
- the project type and build method

**Your subsequent forensics must reuse the same environment and commands as the Setup Agent / Verifier**.
For example: if the Verifier passed tests with `/workspace/repo/.venv/bin/python -m pytest`, your import check must also use `/workspace/repo/.venv/bin/python -c "import X"`, not the system `python3`.

**Step 1 (mandatory): verify that core dependencies are available**

### Python projects
Read core (non-optional) dependencies from pyproject.toml / setup.cfg / requirements.txt and verify each one:
```
cd /workspace/repo && python3 -c "import <package>" 2>&1
```
(If the Setup Agent created a venv/conda, replace `python3` with that environment's interpreter.)
**Note: pip package names and Python import names are often different!** Common mappings:
- beautifulsoup4 → `import bs4`
- GitPython / gitpython → `import git`
- Pillow / pillow → `import PIL`
- PyYAML / pyyaml → `import yaml`
- attrs → `import attr`
- scikit-learn → `import sklearn`
- opencv-python → `import cv2`
- python-dateutil → `import dateutil`
- python-dotenv → `import dotenv`
When unsure, run `pip show <package>` first to see the install location.

### C/C++ projects
Read `find_package()` / `target_link_libraries()` from CMakeLists.txt, or `-l` link libraries from the Makefile:
```
apt list --installed 2>/dev/null | grep -i <keyword>
pkg-config --exists <lib> && echo OK || echo MISSING
```
No `import` check needed; just ensure that headers and libraries are findable at compile time.

### Java projects
```
mvn dependency:tree 2>&1 | tail -30
```
Or `gradle dependencies`. Verify the dependency tree resolves.

### JavaScript projects
```
npm ls --depth=0 2>&1 | tail -30
```
Or `yarn list`. Verify node_modules is complete.

Key checks:
- Python: `ImportError` / `ModuleNotFoundError` → if a core dependency is unavailable, **must prosecute**
- C/C++: `fatal error: xxx.h: No such file` / `undefined reference` → **must prosecute**
- Java: `package does not exist` / `ClassNotFoundException` → **must prosecute**
- JS: `Cannot find module` → **must prosecute**
- An unavailable optional dependency → may be excused

**Step 2 (mandatory): exercise the entry commands declared in README**

A functionality-correct setup is more than just importable core dependencies — the
**example scripts / CLI entry points declared in the README must also start in the
configured environment**; otherwise it is a "documented user-facing commands
unchecked" setup defect.

```
ls /workspace/repo/README* /workspace/repo/docs/README* 2>/dev/null | head
sed -n '1,200p' /workspace/repo/README.md 2>/dev/null
```
Extract at least one documented entry point from README / pyproject.toml `[project.scripts]` / setup.py `entry_points` (a CLI command, an example script, or a `python -m <pkg>` invocation), and run a minimal start-up check (`--help` / `--version` / a single-line example):
```
cd /workspace/repo && <declared venv interpreter> -m <package> --help 2>&1 | head -20
# or (as given in the README)
cd /workspace/repo && <declared venv interpreter> examples/quickstart.py 2>&1 | head -30
```

Adjudication:
- Entry command raises `ModuleNotFoundError` / `ImportError` / `command not found`
  → the Setup Agent left a documented entry point unusable, **must prosecute**.
- Entry command fails only because of an external service (database, API key) but
  the import chain is intact → may be excused.
- Project is a pure library with no CLI entry and no README example → skip this step.

**Step 3 (mandatory): run the test suite yourself**

### Python projects
```
cd /workspace/repo && python3 -m pytest --tb=line -q --timeout=60 2>&1 | tail -60
```
Or follow the project's standard way (poetry run pytest, tox, pytest inside the venv, ...).

### C/C++ projects
If already built, run the tests directly:
```
cd /workspace/repo && ctest --output-on-failure 2>&1 | tail -60
```
Or `make test` / `make check`. If not built, run `cmake . && make -j$(nproc)` first, then test.

### Java projects
```
cd /workspace/repo && mvn test -q 2>&1 | tail -60
```
Or `gradle test`.

### JavaScript projects
```
cd /workspace/repo && npm test 2>&1 | tail -60
```

Record: pass count, failure count, error types, whether killed (exit_code=137/124).

**Step 4: adjudicate each failure category**

| Failure type | Adjudication |
|---|---|
| **Python**: `ImportError`/`ModuleNotFoundError` + the package is in the core dependency declaration | **Must prosecute** |
| **C/C++**: compilation error (missing header/library) or link error | **Must prosecute** |
| **Java**: compilation failure (package not found) or runtime ClassNotFoundException | **Must prosecute** |
| **JS**: `Cannot find module` (core dependency) | **Must prosecute** |
| Package installed but version-incompatible, crashing on run | **Must prosecute** |
| Full suite Killed (exit_code=137/124) **and** a subset run also exhibits dependency-missing errors | **Must prosecute** |
| Full suite Killed, but a small subset shows no dependency-missing — only resource exhaustion | May be excused |
| External service unavailable (database, Redis, network) | May be excused |
| Pure test-logic assertion failures | May be excused |
| Optional dependency not installed; the corresponding tests are skipped | May be excused |

**Step 5: cross-check the credibility of the Verifier's conclusion**
The Verifier claims success=True; does your result agree?
- If the Verifier used test filtering (pytest `--ignore`/`-k`, CTest `-E`, Maven excludes, etc.), check whether any filtered-out test contains a core-dependency miss.
  - Core dependency missing → even if the Verifier dodged it, the Setup Agent is still accountable
  - Failure only because of an external service → the Verifier's avoidance is reasonable, no accountability
- If the full suite was Killed, run a small subset of 10–20 tests to determine whether dependency misses exist.

## Charge format

Each charge must contain:
- **Subject of the charge**: the specific dereliction of the Setup Agent (e.g., "did not install core dependency X")
- **Dependency declaration evidence**: in which file and which field that dependency is declared
- **Forensic command and raw output**: the command you ran personally + the complete output

## Tools (each step must emit a single valid JSON object)

{"thought": "current observation and next-step reasoning", "action": "exec_run", "args": {"command": "shell command"}}
{"thought": "investigation done; all failures fall under excused categories", "action": "finish", "args": {"prosecute": false}}
{"thought": "found accountable problems; filing charges", "action": "finish", "args": {
  "prosecute": true,
  "charges": [
    {"claim": "Setup Agent did not install core dependency X (source: pyproject.toml [project.dependencies])",
     "evidence": "command: python3 -c 'import X'\\noutput: ModuleNotFoundError: No module named 'X'"}
  ]
}}

## Hard constraints

- **Install no packages**: no pip install, apt install, etc.
- **Modify no environment configuration or project file.**
- **The subject of charges is the Setup Agent, not the Verifier**: that the Verifier used --ignore to skip a test is its own judgment; you care whether the Setup Agent made the core dependencies available, not whether the Verifier took shortcuts.
"""


class ProsecutorAgent:
    """Prosecutor ReAct sub-agent with container access."""

    def __init__(
        self,
        env: EnvironmentManager,
        setup_history: list[dict],
        verify_messages: list[dict],
    ):
        self._env = env
        self._setup_history = setup_history
        self._verify_messages = verify_messages
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

    def investigate(self) -> ProsecutionResult:
        """Run the investigation loop and return a ProsecutionResult."""
        logger.info("prosecutor investigation started")

        # Environment snapshot so the prosecutor sees the real container state
        # (avoids blindly probing system python3 when a venv is in play).
        env_snapshot = self._env.get_env_snapshot()
        logger.info(f"env snapshot:\n{env_snapshot[:300]}")

        setup_summary = self._format_setup_history()
        verify_summary = self._format_verify_messages()

        first_user_msg = (
            f"## Container env snapshot\n\n```\n{env_snapshot}\n```\n\n"
            f"## Setup Agent trajectory (last 20 steps)\n\n{setup_summary}\n\n"
            f"## In-loop Verifier conversation\n\n{verify_summary}\n\n"
            "Begin your investigation: decide whether the Setup Agent's environment has substantive issues."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": first_user_msg},
        ]

        successful_steps = 0
        api_failures = 0

        for step in range(1, MAX_STEPS + 1):
            logger.info(f"=== Prosecutor Step {step}/{MAX_STEPS} ===")

            try:
                raw = self._llm.chat(messages, json_mode=True)
            except Exception as e:
                api_failures += 1
                logger.warning(f"Prosecutor LLM call failed (API error/timeout): {e}; skip step (failures={api_failures})")
                continue
            successful_steps += 1
            logger.info(f"LLM output: {raw[:300]}")
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

            if action == "finish":
                prosecute = bool(args.get("prosecute", False))
                charges = args.get("charges", [])
                logger.info(f"investigation done: prosecute={prosecute}, charges={len(charges)}")
                self._llm.close()
                return ProsecutionResult(
                    prosecute=prosecute,
                    charges=charges,
                    messages=list(messages),
                )

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

            else:
                obs = f"Unknown action='{action}'; only exec_run / finish are allowed."
                logger.warning(obs)
                messages.append({"role": "user", "content": obs})

        if successful_steps == 0:
            logger.error(f"Prosecutor: all {MAX_STEPS} LLM calls failed (api_failures={api_failures}); marking as anomaly")
            self._llm.close()
            # Total API failure is infrastructure, not the Setup Agent's fault.
            return ProsecutionResult(
                prosecute=False,
                charges=[{"claim": "[anomaly] prosecutor investigation failed: LLM API entirely unavailable",
                          "evidence": f"all {MAX_STEPS} steps skipped due to API errors; result is unreliable, please rerun"}],
                messages=list(messages),
            )

        logger.warning(f"Prosecutor hit max steps (successful_steps={successful_steps}); defaulting to no prosecution")
        self._llm.close()
        return ProsecutionResult(
            prosecute=False,
            charges=[],
            messages=list(messages),
        )

    def _format_setup_history(self) -> str:
        """Render the last 20 setup steps."""
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
        """Render the verifier conversation."""
        if not self._verify_messages:
            return "(no verifier conversation)"
        lines = []
        for msg in self._verify_messages:
            role = msg.get("role", "?")
            content = (msg.get("content") or "")[:300]
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Lenient JSON extraction from LLM output."""
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
