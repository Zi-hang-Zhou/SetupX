"""LLM inference core (per blueprint section 1.3).

Supports the ARK and OpenAI-compatible APIs. Logs full input / output for traceability.

Responsibilities:
  1. Wrap LLM API calls (ByteDance ARK and OpenAI-compatible backends).
  2. Build the system prompt (action definitions + XPU suggestions + current observation).
  3. Parse the LLM output into a structured AgentAction.
  4. XPU adaptation: turn a generic advice into a concrete command for the current repo.

Module layout:
  LLMClientBase (ABC)
    |-- ARKClient
    `-- OpenAICompatibleClient
  LLMEngine: orchestrates prompt construction + API call + response parsing.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .config import get_config, ARKConfig, OpenAIConfig
from .logger import get_logger
from .models import AgentAction, ActionType, XPUSuggestion

logger = get_logger("llm")


# =============================================================================
# LLM client abstraction
# =============================================================================

class LLMClientBase(ABC):
    """LLM client abstract base.

    All backends implement the unified `chat` interface.
    """

    @abstractmethod
    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        """Send a chat request and return the LLM-generated text.

        Args:
            messages: chat messages (role: system / user / assistant).
            json_mode: ask the LLM to emit JSON.

        Returns:
            The LLM-generated content.
        """
        pass


class ARKClient(LLMClientBase):
    """ByteDance ARK API client.

    ARK is OpenAI-compatible but uses a separate base_url and a `deployment`
    name as the model identifier.
    """

    def __init__(self, config: ARKConfig):
        self._config = config
        # 300s timeout (refinement / audit prompts are long).
        self._client = httpx.Client(timeout=300)

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self._config.deployment,  # ARK uses deployment as the model id
            "messages": messages,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = self._client.post(
            f"{self._config.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._client.close()


class OpenAICompatibleClient(LLMClientBase):
    """OpenAI-compatible API client.

    Supports any backend speaking the OpenAI Chat Completions schema:
    - the official OpenAI API
    - vLLM / Ollama / etc.
    - reasoning models (qwen, glm-4.6, ...) which need extra handling for
      <think> tags and `reasoning_content`.
    """

    def __init__(self, config: OpenAIConfig):
        self._config = config
        # 300s timeout (refinement / audit prompts are long).
        self._client = httpx.Client(timeout=300)

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        """Send a chat request to the OpenAI-compatible API.

        Special handling:
        1. Reasoning-model compat: prefer `content`; fall back to `reasoning_content`.
        2. <think> stripping: qwen-style models embed reasoning in `content`
           (`<think>...</think>actual output`); strip the reasoning block.
        """
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": 4096,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = self._client.post(
            f"{self._config.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        msg = data["choices"][0]["message"]

        # Reasoning-model compat: some return main text in `reasoning_content`.
        content = msg.get("content")
        if not content:
            content = msg.get("reasoning_content", "")

        # Strip <think>...</think> blocks (qwen-style).
        if content and "<think>" in content:
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        return content

    def close(self) -> None:
        self._client.close()


# =============================================================================
# LLM inference engine
# =============================================================================

class LLMEngine:
    """LLM inference engine (per blueprint section 1.3).

    Responsibilities:
    1. Pick the LLM backend by configuration (ARK or OpenAI-compatible).
    2. Build the system prompt (XPU suggestions, current observation, ...).
    3. Call the LLM to produce the next action.
    4. Parse the response into a structured AgentAction.
    5. Adapt generic XPU advice into concrete commands for the current repo.
    """

    # =========================================================================
    # System prompt template (per blueprint section 3.2)
    # =========================================================================
    # Defines all action types and usage rules. Placeholders {cwd}, {os_info},
    # {formatted_xpu_suggestions} are filled at runtime. Use {{ }} to escape
    # literal braces so .format() does not misinterpret them.
    SYSTEM_PROMPT_TEMPLATE = """You are an expert DevOps agent tailored for environment setup.
You have access to a Linux terminal and an external eXPerience Unit (XPU).

Current Status:
- WorkDir: {cwd}
- OS: {os_info}

XPU Suggestions (Proven solutions from history):
{formatted_xpu_suggestions}

## Action Types — Purpose and When to Use

### SHELL_COMMAND
Execute any shell command directly in the container.
- **Use for**: installing packages, setting PYTHONPATH, exploring repo structure, running
  diagnostic commands (e.g. `pytest --co -q` to inspect collection errors), fixing configs.
- **This is your default action.** Use it whenever you are still diagnosing or fixing.
- To check if dependencies are installed, run: `pip list | grep <pkg>` or `python -c "import X"`.
- To understand pytest errors without triggering full verification, run: `pytest --co -q 2>&1 | head -50`.

### TRY_XPU_SUGGESTION
Apply a proven fix from the XPU knowledge base inside a snapshot sandbox.
- **Use for**: applying an "Executable XPU Fix" that is relevant to the current error.
  YOU must adapt the XPU advice to the current environment and write the concrete shell
  commands yourself in the "command" field (semicolon-separated, like SHELL_COMMAND).
  The container is snapshotted before execution and auto-rolled back on failure,
  making it safer than SHELL_COMMAND.
- **Do NOT use** if "Executable XPU Fixes" is absent from the XPU section (means commands are empty).

### SET_ENV
Persist an environment variable across all subsequent commands.
- **Use for**: setting variables like PYTHONPATH, JAVA_HOME that must survive across steps.
- Prefer this over `export VAR=...` in a SHELL_COMMAND, which only lasts for that single command.

### ROLLBACK_ENV
Pop **any number of frames** off the snapshot stack and return the container to
**any earlier known-good checkpoint**, not only the latest one.
- **Use for**: recovering from a broken environment state — e.g. after multiple failed attempts
  left the container in an inconsistent state, or after a bad TRY_XPU_SUGGESTION result.
- Provide `n_frames` in `content` to pop multiple frames at once when several recent
  attempts collectively steered the trajectory into a dead end. Default is 1.
  Example: `"content": {{"n_frames": 3}}` rolls back the three most recent snapshots.
- This is a recovery escape hatch, not a routine action. Use sparingly.

### VERIFY
Trigger the full pytest verification pipeline. This is an **expensive, final-stage action**:
it spins up a sub-agent that probes the project structure and runs `pytest --co -q` then
`pytest -x -q`. It is designed to confirm that setup is complete, NOT to diagnose problems.
- **ONLY call VERIFY when you genuinely believe all dependencies are installed and the
  environment is fully configured.**
- **NEVER call VERIFY to probe the environment, discover missing packages, or get pytest
  output for diagnosis.** Use `SHELL_COMMAND` with `pytest --co -q` for that purpose instead.
- If VERIFY succeeds → call FINISH immediately.
- If VERIFY fails → analyze the output and continue fixing with SHELL_COMMAND.

### FINISH
Signal that the task is complete. **ONLY call after a successful VERIFY.**

---

## Decision Instructions

1. Analyze the Last Error carefully before choosing an action.
2. If "Executable XPU Fixes" lists a fix relevant to the current error, **prefer
   TRY_XPU_SUGGESTION** — read the XPU advice, adapt it to your current environment,
   and write the concrete command yourself. The command is snapshot-protected with
   auto-rollback on failure.
3. Use SHELL_COMMAND when no relevant XPU fix is available, or when you need to run
   diagnostic/exploratory commands (e.g., ls, cat, pip list).
4. Default to SHELL_COMMAND when in doubt.
5. Only call VERIFY when you are confident the environment is ready. Until then, diagnose
   with SHELL_COMMAND.
6. Always explain WHY you chose the action in the "thought" field.

---

You MUST respond in JSON format with this schema:
{{
  "thought": "Analyze the current state and the cause of the error, then explain why you chose this action...",
  "action_type": "SHELL_COMMAND" | "TRY_XPU_SUGGESTION" | "SET_ENV" | "ROLLBACK_ENV" | "VERIFY" | "FINISH",
  "content": {{
    // For SHELL_COMMAND:
    "command": "pip install numpy",

    // For TRY_XPU_SUGGESTION:
    "xpu_suggestion_id": "suggestion_123",
    "command": "pip install numpy==1.23.5",
    "reasoning": "The XPU suggests downgrading numpy; this matches the error closely."

    // For SET_ENV:
    "env_key": "VAR_NAME",
    "env_value": "value"

    // For ROLLBACK_ENV:
    // "n_frames": 1   // default 1; pass >=2 to go back to an earlier checkpoint.
    //
    // For VERIFY / FINISH:
    // (VERIFY needs no extra fields)
    // FINISH requires: "message": "environment setup complete"
  }}
}}
"""

    # =========================================================================
    # Situation-description prompt (used for XPU vector retrieval)
    # =========================================================================
    SITUATION_PROMPT = (
        "You are the situation-sensing module of the environment-setup Agent. "
        "Based on the current work history, describe the current situation in 2-3 sentences "
        "in a form that supports experience retrieval:\n"
        "1. Project profile: language, package manager (pip/poetry/conda), type of dependency file.\n"
        "2. Actions already taken and the current sticking point (if there is an error, describe "
        "the error type; otherwise describe what is currently being done).\n"
        "3. The intent of the next step.\n"
        "Output plain text only — no JSON — and no more than 150 characters.\n"
        "[Hard constraint] Only describe commands actually executed and files actually observed in "
        "the history. Do not infer the name of any tool that has not appeared. "
        "For example, write \"using Poetry\" only if a poetry command was actually run or a "
        "poetry.lock file was observed; otherwise write \"using pip\" or \"package manager unknown\"."
    )

    def __init__(self):
        """Init.

        Picks the LLM backend per LLM_PROVIDER:
        - "ark":    ARK API via ARKClient.
        - "openai": OpenAI-compatible API via OpenAICompatibleClient.
        """
        config = get_config()

        if config.llm_provider == "ark":
            self._client = ARKClient(config.ark)
            logger.info("using ARK LLM client")
        elif config.llm_provider == "openai":
            if config.openai is None:
                raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not configured")
            self._client = OpenAICompatibleClient(config.openai)
            logger.info("using OpenAI-compatible LLM client")
        else:
            raise ValueError(f"unsupported LLM provider: {config.llm_provider}")

    def describe_situation(
        self,
        history: list[dict],
        cwd: str,
        os_info: str,
        last_error: str | None,
    ) -> str:
        """Generate a 2-3 sentence situation description for XPU retrieval.

        Sends the latest 5 history entries + current state to the LLM.
        On failure, falls back to the truncated last_error text.
        """
        recent = history[-5:]
        lines = []
        for entry in recent:
            if "action" in entry:
                a = entry["action"]
                lines.append(f"action: {a.get('action_type', '')} {a.get('command', '')[:80]}")
            if "result" in entry:
                r = entry["result"]
                out = (r.get("stdout") or "")[:100]
                err = (r.get("stderr") or "")[:100]
                lines.append(f"result(exit={r.get('exit_code', '')}): {out or err}")

        history_text = "\n".join(lines) if lines else "(no history; task just started)"
        error_text = f"\nlast error: {last_error[:200]}" if last_error else ""

        messages = [
            {"role": "system", "content": self.SITUATION_PROMPT},
            {"role": "user", "content": (
                f"workdir: {cwd}\nOS: {os_info}{error_text}\n\n"
                f"recent history:\n{history_text}"
            )},
        ]

        try:
            situation = self._client.chat(messages, json_mode=False).strip()
        except Exception as e:
            logger.warning(f"situation description generation failed: {e}")
            situation = last_error[:150] if last_error else "python project environment setup"

        logger.info(f"[situation] {situation[:100]}")
        return situation

    # =========================================================================
    # XPU suggestion formatting
    # =========================================================================

    def _format_xpu_suggestions(
        self,
        suggestions: list[XPUSuggestion],
        tried_ids: set[str],
    ) -> str:
        """Format XPU suggestions as two layers, embedded in the system prompt.

        Two-layer design:
        - Layer 1 (Reference Knowledge): all advice in natural language, so the
          LLM can author its own SHELL_COMMAND informed by the experience.
        - Layer 2 (Executable Fixes): suggestions whose `commands` are non-empty;
          the LLM may invoke these directly via TRY_XPU_SUGGESTION.

        Args:
            suggestions: retrieved XPU suggestions.
            tried_ids: IDs already tried this run; skip them.

        Returns:
            Formatted text inserted into {formatted_xpu_suggestions}.
        """
        if not suggestions:
            return "No XPU knowledge available."

        ref_lines = []   # Layer 1: NL reference (regardless of whether commands exist)
        exec_lines = []  # Layer 2: only when commands are non-empty

        for s in suggestions:
            if s.id in tried_ids:
                continue  # skip already-tried suggestions to prevent loops

            # Layer 1: always show NL advice (description is the joined advice_nl).
            ref_lines.append(f"- [{s.id}] {s.description}")

            # Layer 2: only show as executable when commands exist.
            if s.commands:
                exec_lines.append(
                    f"  [ID: {s.id}] Commands: {s.commands} (confidence: {s.confidence:.2f})"
                )

        parts = []
        if ref_lines:
            parts.append(
                "XPU Reference Knowledge (use to inform your SHELL_COMMAND):\n"
                + "\n".join(ref_lines)
            )
        if exec_lines:
            parts.append(
                "Executable XPU Fixes (use TRY_XPU_SUGGESTION, snapshot-protected):\n"
                + "\n".join(exec_lines)
            )
        return "\n\n".join(parts) if parts else "No applicable XPU knowledge."

    # =========================================================================
    # Main decision interface
    # =========================================================================

    def generate_action(
        self,
        history: list[dict],
        xpu_suggestions: list[XPUSuggestion],
        cwd: str = "/workspace/repo",
        os_info: str = "Ubuntu 22.04",
        last_error: str | None = None,
        tried_suggestion_ids: set[str] | None = None,
        initial_user_prompt: str | None = None,
    ) -> AgentAction:
        """Generate the next action (per blueprint section 1.3).

        Flow:
        1. Build the system prompt (cwd, os_info, XPU suggestions).
        2. Append the latest 10 history entries as assistant/user message pairs.
        3. Append the current observation (last_error) as the final user message.
        4. Call the LLM in JSON mode.
        5. Parse the response into an AgentAction.

        Args:
            history: agent history (action + result pairs).
            xpu_suggestions: currently retrieved XPU suggestions.
            cwd: current working directory.
            os_info: OS info.
            last_error: last error text.
            tried_suggestion_ids: suggestion IDs already tried (success or failure).
            initial_user_prompt: when the history is empty (first iteration), use this
                instead of the generic prompt — used by multi-repo / family runs to
                inject task meta into the first user message.

        Returns:
            A structured AgentAction.
        """
        if tried_suggestion_ids is None:
            tried_suggestion_ids = set()

        # === 1. Build the system prompt ===
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            cwd=cwd,
            os_info=os_info,
            formatted_xpu_suggestions=self._format_xpu_suggestions(
                xpu_suggestions, tried_suggestion_ids
            ),
        )

        messages = [{"role": "system", "content": system_prompt}]

        # === 2. Append history ===
        # Use the latest 10 entries; convert to assistant (action) + user (result) pairs.
        for entry in history[-10:]:
            if "action" in entry:
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(entry["action"], ensure_ascii=False),
                })
            if "result" in entry:
                result = entry["result"]
                content = f"command result:\nexit_code: {result.get('exit_code', 'N/A')}\n"
                if result.get("stdout"):
                    content += f"stdout: {result['stdout']}\n"
                if result.get("stderr"):
                    content += f"stderr: {result['stderr']}\n"
                messages.append({"role": "user", "content": content})

        # === 3. Append the current observation ===
        # First iteration with a custom initial user prompt (multi-repo/family meta) wins.
        if not history and initial_user_prompt:
            user_content = initial_user_prompt
        elif last_error:
            user_content = f"Last Error:\n{last_error}\n\nAnalyze the cause and decide the next action."
        else:
            user_content = "Analyze the current state and decide the next action."

        messages.append({"role": "user", "content": user_content})

        # === Log full LLM input (for debugging) ===
        logger.info("=" * 60)
        logger.info("LLM input (full prompt)")
        logger.info("=" * 60)
        for i, msg in enumerate(messages):
            logger.info(f"[{i}] role={msg['role']}")
            content = msg["content"]
            if len(content) > 2000:
                logger.info(f"    content (truncated): {content[:1000]}...")
                logger.info(f"    ... ({len(content)} chars total)")
            else:
                logger.info(f"    content: {content}")
        logger.info("=" * 60)

        # === 4. Call the LLM ===
        response = self._client.chat(messages, json_mode=True)

        logger.info("=" * 60)
        logger.info("LLM output (raw response)")
        logger.info("=" * 60)
        logger.info(response)
        logger.info("=" * 60)

        # === 5. Parse the response (with one explicit retry: models occasionally emit prose) ===
        try:
            return self._parse_response(response, xpu_suggestions)
        except ValueError as parse_err:
            # On retry, append the previous bad response and explicitly require JSON.
            logger.error(
                f"failed to parse LLM response; triggering 1 explicit retry — reason: {parse_err}"
            )
            retry_messages = list(messages) + [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": (
                        "Your previous response could not be parsed as valid JSON. "
                        "Reply strictly per the system prompt and **output exactly one JSON object**: "
                        '{"thought": "...", "action_type": "...", "content": {...}}. '
                        "No extra prose, no markdown code fence, no leading or trailing characters."
                    ),
                },
            ]
            response_retry = self._client.chat(retry_messages, json_mode=True)
            logger.info("=" * 60)
            logger.info("LLM retry output (raw response, attempt 2)")
            logger.info("=" * 60)
            logger.info(response_retry)
            logger.info("=" * 60)
            # A second failure raises ValueError as-is (per "fail loudly" policy).
            return self._parse_response(response_retry, xpu_suggestions)

    # =========================================================================
    # Response parsing
    # =========================================================================

    def _parse_response(
        self,
        response: str,
        xpu_suggestions: list[XPUSuggestion],
    ) -> AgentAction:
        """Parse the LLM response into an AgentAction.

        Accepts:
        1. Plain JSON text.
        2. JSON wrapped in a ```json ... ``` markdown fence.
        """
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                raise ValueError(f"could not parse LLM response as JSON: {response[:200]}")

        action_type_str = data.get("action_type", "SHELL_COMMAND")
        content = data.get("content", {})
        thought = data.get("thought", "")

        if action_type_str == "SHELL_COMMAND":
            return AgentAction(
                action_type=ActionType.SHELL_COMMAND,
                thought=thought,
                command=content.get("command"),
            )
        elif action_type_str == "TRY_XPU_SUGGESTION":
            return AgentAction(
                action_type=ActionType.TRY_XPU_SUGGESTION,
                thought=thought,
                xpu_suggestion_id=content.get("xpu_suggestion_id"),
                command=content.get("command"),
                reasoning=content.get("reasoning"),
            )
        elif action_type_str == "FINISH":
            return AgentAction(
                action_type=ActionType.FINISH,
                thought=thought,
                message=content.get("message", "task complete"),
            )
        elif action_type_str == "SET_ENV":
            return AgentAction(
                action_type=ActionType.SET_ENV,
                thought=thought,
                env_key=content.get("env_key"),
                env_value=content.get("env_value"),
            )
        elif action_type_str == "ROLLBACK_ENV":
            # Support popping multiple frames at once.
            try:
                n_frames = int(content.get("n_frames", 1))
            except (TypeError, ValueError):
                n_frames = 1
            return AgentAction(
                action_type=ActionType.ROLLBACK_ENV,
                thought=thought,
                rollback_n_frames=max(1, n_frames),
            )
        elif action_type_str == "VERIFY":
            return AgentAction(
                action_type=ActionType.VERIFY,
                thought=thought,
                verify_hint=content.get("hint"),
            )
        else:
            # Unknown action: degrade to SHELL_COMMAND.
            logger.warning(f"unknown action_type: {action_type_str}; defaulting to SHELL_COMMAND")
            return AgentAction(
                action_type=ActionType.SHELL_COMMAND,
                thought=thought,
                command=content.get("command") or data.get("command"),
            )

    # =========================================================================
    # XPU adaptation (Option A: LLM materializes commands from generic advice)
    # =========================================================================

    # System prompt for XPU adaptation: ask the LLM to take a generic advice
    # plus the concrete current error/state and produce executable commands.
    ADAPT_XPU_PROMPT = """You are a senior DevOps engineer. You will be given one environment-fix
suggestion from a historical experience base (advice_nl), together with the current
repository's concrete error message and environment state.

Your task: starting from that suggestion's idea, combine it with the current
repository's specifics (error message, OS, working directory, ...) and generate
**adapted, directly executable shell commands**.

Notes:
1. The suggestion's idea is generic — adjust concrete package names, versions, etc., to the
   actual error.
2. Each emitted command must be a complete, executable shell command.
3. List the commands in execution order.
4. Do not emit commands unrelated to the fix (e.g., echo, comments).

You must reply in JSON format:
{{"commands": ["cmd1", "cmd2", ...]}}
"""

    def adapt_xpu_commands(
        self,
        advice_nl: list[str],
        last_error: str,
        cwd: str,
        os_info: str,
    ) -> list[str]:
        """Materialize executable commands from a generic XPU advice + current context.

        The XPU store keeps generic NL advice (advice_nl); concrete package names /
        versions vary per repo. Here the LLM acts as an adapter: it reads the generic
        idea and emits commands fitting the current error.

        Example:
        - advice_nl: ["downgrade numpy to be compatible with the older API"]
        - last_error: "numpy 1.24 has no attribute 'float'"
        - LLM emits: ["pip install numpy==1.23.5"]

        Args:
            advice_nl: generic NL fix suggestions from the XPU.
            last_error: the current concrete error.
            cwd: current working directory.
            os_info: OS info.

        Returns:
            List of adapted commands; empty list on parse failure.
        """
        user_payload = json.dumps({
            "advice_nl": advice_nl,
            "current_error": last_error[:3000] if last_error else "",
            "cwd": cwd,
            "os_info": os_info,
        }, ensure_ascii=False)

        messages = [
            {"role": "system", "content": self.ADAPT_XPU_PROMPT},
            {"role": "user", "content": user_payload},
        ]

        logger.info("=" * 60)
        logger.info("LLM adapt-XPU input")
        logger.info(f"  advice_nl: {advice_nl}")
        logger.info(f"  error: {(last_error or '')[:200]}...")
        logger.info("=" * 60)

        response = self._client.chat(messages, json_mode=True)

        logger.info("=" * 60)
        logger.info("LLM adapt-XPU output")
        logger.info(response)
        logger.info("=" * 60)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                logger.warning(f"adapt-command parse failed; returning empty list: {response[:200]}")
                return []

        commands = data.get("commands", [])
        if not isinstance(commands, list):
            logger.warning(f"adapt-command shape invalid: {commands}")
            return []

        logger.info(f"LLM emitted {len(commands)} adapted commands: {commands}")
        return commands

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close the LLM client (release HTTP connections)."""
        if hasattr(self._client, "close"):
            self._client.close()
