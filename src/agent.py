"""Main controller (per blueprint sections 1.4 and 2).

Implements the speculative-execution Setup Agent.

Responsibilities:
  1. Perception:  read environment state from the Docker container (pwd, os-release).
  2. Decision:    feed observation + history + XPU suggestions to the LLM, get an action.
  3. Execution:   run commands in the container, try XPU suggestions, set env vars, ...
  4. Observation: collect results and feed them back to the LLM for the next decision.

Main loop:
  observe -> retrieve XPU -> LLM decision -> execute -> log history -> repeat
  until the LLM emits FINISH or max_steps is reached.

Speculative execution:
  When TRY_XPU_SUGGESTION is chosen, we `docker commit` a snapshot first; on
  failure we roll back to the snapshot, leaving the environment clean.
"""

import time

from .logger import get_logger
from .models import (
    AgentState,
    AgentAction,
    ActionType,
    CommandResult,
    AttributionReport,
    XPUSuggestion,
    SetupResult,
)
from .environment_manager import EnvironmentManager
from .xpu_client import create_xpu_client, XPUClientBase, VectorXPUClient
from .llm_engine import LLMEngine
from .retriever_agent import RetrieverAgent
from .verifier_agent import VerifierAgent
from .task_meta import render_first_user_message

logger = get_logger("agent")


class SpeculativeSetupAgent:
    """Speculative-execution environment-setup Agent (per blueprint 1.4).

    Architecture:
      Agent (this class) = orchestrator
        |-- EnvironmentManager: docker container lifecycle.
        |-- XPUClientBase:      query the XPU knowledge base.
        |-- LLMEngine:          LLM inference.
        `-- VerifierAgent:      runs pytest in the verify stage.
    """

    def __init__(
        self,
        repo_url: str,
        max_steps: int = 9999,
        task_meta: dict | None = None,
    ):
        """Init.

        Args:
            repo_url: target Git repo URL.
            max_steps: max iteration steps (default 9999 ~ unlimited; phase 1 is
                actually bounded by main.py's signal.alarm(phase1_timeout)).
            task_meta: optional task metadata for multi-repo / non-atomic family
                runs (rendered into the first user message). Shape:
                  {"repository": "x/y", "primary_repos": [...], "component_repos": [...]}.
                None for ordinary atomic single-repo runs.
        """
        self._state = AgentState(
            repo_url=repo_url,
            max_steps=max_steps,
        )
        self._env: EnvironmentManager = EnvironmentManager()
        self._xpu: XPUClientBase = create_xpu_client()
        self._llm: LLMEngine = LLMEngine()
        self._task_meta = task_meta

        # RetrieverAgent: enabled only with VectorXPUClient.
        # It does the two-layer retrieval + delayed audit in its own context.
        self._retriever: RetrieverAgent | None = None
        if isinstance(self._xpu, VectorXPUClient):
            self._retriever = RetrieverAgent(
                vector_store=self._xpu._store,
                llm_client=self._llm._client,
            )
            logger.info("RetrieverAgent enabled (VectorXPUClient mode)")

        # Cross-step XPU suggestion pool: {id: (suggestion, step_last_seen)}.
        # Keeps suggestions seen in the last 2 steps so the LLM can still
        # invoke a TRY_XPU_SUGGESTION it just saw.
        self._xpu_suggestion_pool: dict[str, tuple[XPUSuggestion, int]] = {}
        # Suggestions retrieved at the current step; used to look up TRY_XPU_SUGGESTION.
        self._current_xpu_suggestions: list[XPUSuggestion] = []
        # Verifier transcript from the last successful verify; used by Phase 2.
        self._last_verify_messages: list[dict] = []

        logger.info(f"agent initialized; target repo: {repo_url}")

    @property
    def env(self) -> EnvironmentManager:
        """Expose the environment manager so the verify stage can reuse the same container.

        After run() returns, the container is intentionally not destroyed; the
        verify stage runs pytest on the same container via this property.
        """
        return self._env

    def run(self) -> SetupResult:
        """Run the main agent loop (per blueprint section 2).

        Flow:
        1. Create the container + clone the repo.
        2. Main loop (up to max_steps):
           a. Observe (pwd, os-release).
           b. If there is an error, retrieve XPU suggestions.
           c. Send observation + history + XPU suggestions to the LLM, get an action.
           d. Execute the action (SHELL_COMMAND / TRY_XPU_SUGGESTION / SET_ENV / ...).
           e. Log the result.
        3. After the loop: close clients and clean up snapshot images.
        4. Return SetupResult (container kept alive for the verify stage).

        Returns:
            SetupResult with completion status, history, etc.
            Note: the container is NOT destroyed here; the caller must destroy it
            after the verify stage via env.destroy().
        """
        logger.info("starting environment-setup task")

        # === 1. Init phase ===
        # Create the container and clone the repo into /workspace/repo.
        container_id = self._env.create_container(self._state.repo_url)
        self._state.container_id = container_id

        # First-iteration user message override: when running with task_meta
        # (multi-repo / non-atomic family), render the family-aware first user message.
        initial_user_prompt: str | None = None
        if self._task_meta:
            family_meta = None
            primary = self._task_meta.get("primary_repos") or []
            component = self._task_meta.get("component_repos") or []
            if primary or component:
                family_meta = {
                    "primary_repos": primary,
                    "component_repos": component,
                }
            repository = (
                self._task_meta.get("repository")
                or self._state.repo_url.rstrip("/").split("github.com/")[-1]
            )
            initial_user_prompt = render_first_user_message(
                repo_url=self._state.repo_url,
                repository=repository,
                family_meta=family_meta,
            )
            logger.info(
                f"task_meta provided; first user message will be rendered "
                f"(family_meta={'yes' if family_meta else 'no'}, repository={repository})"
            )

        # === 2. Main loop ===
        while self._state.step < self._state.max_steps:
            self._state.step += 1
            logger.info(f"=== Step {self._state.step}/{self._state.max_steps} ===")

            # --- 2a. Observation ---
            cwd = self._env.exec_run("pwd").stdout.strip()
            os_info = self._env.exec_run("cat /etc/os-release | head -2").stdout.strip()

            # --- 2b. Diagnose + retrieve ---
            # Only retrieve XPU suggestions when there is an error.
            self._current_xpu_suggestions = []
            if self._state.last_error:
                exclude = list(self._state.tried_suggestions) if self._state.tried_suggestions else None

                # Build a hybrid situation: LLM semantic summary + raw command/error text.
                # The LLM summary improves semantic match against situation_triggers / advice_nl.
                # The raw text preserves keyword hits against keywords / regex.
                llm_summary = self._llm.describe_situation(
                    history=self._state.get_recent_history(),
                    cwd=cwd,
                    os_info=os_info,
                    last_error=self._state.last_error,
                )
                raw_situation = self._build_situation(self._state.last_error)
                situation = f"{llm_summary}\n\n{raw_situation}"

                if self._retriever:
                    self._current_xpu_suggestions = self._retriever.retrieve(
                        situation=situation,
                        exclude_ids=exclude,
                        full_history=self._state.history,
                    )
                else:
                    # Fallback (non-VectorXPUClient): dual retrieval (situation + raw error), dedup.
                    suggestions_by_situation = self._xpu.query(
                        {"query": situation, "os_release": os_info},
                        exclude_ids=exclude,
                    )
                    suggestions_by_error = self._xpu.query(
                        {"error": self._state.last_error, "os_release": os_info},
                        exclude_ids=exclude,
                    )
                    seen_ids = {s.id for s in suggestions_by_situation}
                    for s in suggestions_by_error:
                        if s.id not in seen_ids:
                            suggestions_by_situation.append(s)
                    self._current_xpu_suggestions = suggestions_by_situation

            # Update the cross-step pool: add new ones, evict any unseen for >1 step.
            current_step = self._state.step
            for s in self._current_xpu_suggestions:
                self._xpu_suggestion_pool[s.id] = (s, current_step)
            self._xpu_suggestion_pool = {
                sid: (sg, step)
                for sid, (sg, step) in self._xpu_suggestion_pool.items()
                if current_step - step <= 1  # keep this step + previous step only
            }
            self._current_xpu_suggestions = [sg for sg, _ in self._xpu_suggestion_pool.values()]

            # --- 2c. LLM decision (Thought & Plan) ---
            action = self._llm.generate_action(
                history=self._state.get_recent_history(),
                xpu_suggestions=self._current_xpu_suggestions,
                cwd=cwd,
                os_info=os_info,
                last_error=self._state.last_error,
                tried_suggestion_ids=self._state.tried_suggestions,
                # First iteration only (history empty) — overrides the generic prompt
                # with our task-meta-rendered family-aware message when provided.
                initial_user_prompt=initial_user_prompt if not self._state.history else None,
            )

            logger.info(f"decision: {action}")

            # --- 2d. Execution ---
            if action.action_type == ActionType.SHELL_COMMAND:
                self._handle_shell_command(action)

            elif action.action_type == ActionType.TRY_XPU_SUGGESTION:
                self._handle_try_xpu_suggestion(action)

            elif action.action_type == ActionType.SET_ENV:
                self._handle_set_env(action)

            elif action.action_type == ActionType.ROLLBACK_ENV:
                self._handle_rollback_env(action)

            elif action.action_type == ActionType.VERIFY:
                verified = self._handle_verify(action)
                if verified:
                    break  # verify passed -> exit the main loop

            elif action.action_type == ActionType.FINISH:
                self._handle_finish(action)
                break

        # === 3. Cleanup phase ===
        # Close LLM connection. XPU connection stays open (main.py closes it
        # after experience storage).
        self._llm.close()
        # Final audit before closing the RetrieverAgent.
        if self._retriever:
            self._retriever.close(full_history=self._state.history)
        # Drop snapshot images to free disk; verify stage no longer needs rollbacks.
        self._env.cleanup_snapshots()

        if not self._state.completed:
            logger.warning("max iterations reached; task not completed")

        return SetupResult(
            repo_url=self._state.repo_url,
            container_id=container_id,
            completed=self._state.completed,
            steps_taken=self._state.step,
            final_message=self._state.final_message or "max iterations reached; task not completed",
            history=self._state.history,
            last_verify_messages=self._last_verify_messages,
        )

    # =========================================================================
    # Action handlers
    # =========================================================================

    def _handle_shell_command(self, action: AgentAction) -> None:
        """Execute a shell command directly in the container (default action)."""
        if not action.command:
            logger.warning("SHELL_COMMAND missing 'command'")
            return

        result = self._env.exec_run(action.command)

        if not result.success:
            self._state.last_error = result.stderr or result.stdout
        else:
            self._state.last_error = None  # success clears the previous error

        self._state.add_to_history({
            "action": action.to_dict(),
            "result": result.to_dict(),
        })

    def _handle_try_xpu_suggestion(self, action: AgentAction) -> None:
        """Handle TRY_XPU_SUGGESTION (speculative-execution mode).

        Speculative-execution flow:
        A. Checkpoint: docker commit a snapshot.
        B. Adapt:      LLM materializes commands from advice_nl + current context.
        C. Trial:      execute the adapted commands in order.
        D. Attribute:  evaluate the effect.
        E. Feedback:   post telemetry to the XPU store.
        F. Branch:     keep on success / rollback on failure.
        """
        if not action.xpu_suggestion_id:
            logger.warning("TRY_XPU_SUGGESTION missing 'xpu_suggestion_id'")
            return

        # Look up the suggestion in the current pool.
        suggestion = None
        for s in self._current_xpu_suggestions:
            if s.id == action.xpu_suggestion_id:
                suggestion = s
                break

        # Suggestion not found: filtered out, or the LLM mistyped the ID.
        if not suggestion:
            logger.warning(f"XPU suggestion not found: {action.xpu_suggestion_id}")
            self._state.add_to_history({
                "action": action.to_dict(),
                "result": {
                    "exit_code": 1,
                    "stdout": (
                        f"[XPU BLOCKED] suggestion {action.xpu_suggestion_id} not available "
                        f"or already disabled; use SHELL_COMMAND instead"
                    ),
                    "stderr": "",
                },
            })
            return

        # Capture pre-trial error for attribution comparison.
        error_before = self._state.last_error or ""

        # --- A. Checkpoint ---
        ckpt_tag = f"step_{self._state.step}_pre_xpu"
        self._env.create_checkpoint(ckpt_tag)
        logger.info(f"created checkpoint {ckpt_tag}; starting speculative XPU execution")

        # --- B. Command source: prefer the main agent's adapted command ---
        # The main agent has full context (last 10 steps, pip list output, ...) and is
        # expected to author an adapted command in `action.command`. Fall back to the
        # XPU's atom-rendered commands if the main agent didn't author one.
        if action.command:
            commands = [action.command]
            logger.info(f"main agent supplied adapted command: {commands}")
        elif suggestion.commands:
            commands = suggestion.commands
            logger.info(
                f"main agent did not supply a command; falling back to "
                f"{len(commands)} atom-rendered commands"
            )
        else:
            logger.warning(f"XPU suggestion {suggestion.id} has no executable command; skipping")
            self._state.record_tried_suggestion(suggestion.id)
            self._state.add_to_history({
                "action": action.to_dict(),
                "result": {
                    "exit_code": 1,
                    "stdout": f"[XPU SKIP] {suggestion.id}: no commands; not executed",
                    "stderr": "",
                },
            })
            return

        # --- C. Trial: execute commands in order; abort on first failure. ---
        success = True
        logs: list[CommandResult] = []
        for cmd in commands:
            result = self._env.exec_run(cmd)
            logs.append(result)
            if not result.success:
                success = False
                break

        # --- D. Attribution ---
        error_after = ""
        if not success and logs:
            error_after = logs[-1].stderr or logs[-1].stdout

        if success:
            attribution_score = 1.0
            outcome = "SUCCESS"
        elif error_after and error_after != error_before:
            attribution_score = -1.0  # introduced a new error
            outcome = "FAIL"
        else:
            attribution_score = 0.0   # no effect
            outcome = "FAIL"

        # --- E. Feedback ---
        # In RetrieverAgent mode, telemetry is updated by the delayed audit
        # inside RetrieverAgent.retrieve(); main agent does not push directly.
        # Otherwise (non-vector backends) fall back to immediate feedback.
        if not self._retriever:
            report = AttributionReport(
                suggestion_id=suggestion.id,
                timestamp=time.time(),
                repo_context=self._state.repo_url,
                outcome=outcome,
                error_before=error_before,
                error_after=error_after,
                score=attribution_score,
                logs=logs,
            )
            self._xpu.submit_feedback(report)

        # --- F. Branch ---
        if not success:
            logger.info(f"XPU suggestion {suggestion.id} failed; rolling back...")
            self._env.rollback_to_checkpoint()
            self._state.last_error = error_before  # restore the pre-trial error
        else:
            logger.info(f"XPU suggestion {suggestion.id} verified")
            self._state.last_error = None

        # Mark as tried regardless of outcome to prevent infinite retry on the same suggestion.
        self._state.record_tried_suggestion(suggestion.id)

        cmd_outputs = "\n".join(
            f"$ {log.get('command', '')}\n{(log.get('stdout') or log.get('stderr') or '')[:300]}"
            for log in [l.to_dict() for l in logs[:3]]  # at most 3 commands' output
        )
        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0 if success else 1,
                "stdout": f"[XPU {outcome}] {suggestion.id}\n{cmd_outputs}",
                "stderr": "",
            },
        })

    def _handle_set_env(self, action: AgentAction) -> None:
        """SET_ENV: set a persistent env var on the container.

        Unlike `export VAR=...` inside a SHELL_COMMAND (which dies with the
        single-shell process), SET_ENV is injected into every subsequent
        exec_run via Docker's native `environment` parameter.
        """
        if not action.env_key or action.env_value is None:
            logger.warning("SET_ENV missing env_key or env_value")
            return

        self._env.set_env(action.env_key, action.env_value)
        self._state.env_vars[action.env_key] = action.env_value

        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0,
                "stdout": f"[SET_ENV] {action.env_key}={action.env_value}",
                "stderr": "",
            },
        })
        # SET_ENV is a constructive action; clear the previous error so the LLM
        # does not keep dwelling on stale context.
        self._state.last_error = None

    def _handle_rollback_env(self, action: AgentAction) -> None:
        """ROLLBACK_ENV: pop n_frames snapshots and restore to the resulting checkpoint."""
        n = max(1, int(action.rollback_n_frames or 1))
        success = self._env.rollback_to_checkpoint(n_frames=n)
        status = f"ok (popped {n} frames)" if success else "failed (no snapshots available)"
        logger.info(f"[ROLLBACK_ENV] {status}")
        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0 if success else 1,
                "stdout": f"[ROLLBACK_ENV] {status}",
                "stderr": "",
            },
        })

    def _handle_verify(self, action: AgentAction) -> bool:
        """VERIFY: run pytest verification through VerifierAgent.

        Verify is an expensive action: VerifierAgent spins up a ReAct sub-loop
        to inspect the project structure and run `pytest --co -q` then `pytest -x -q`.
        The LLM should ONLY call VERIFY when it is confident the env is ready.

        On success: mark the task complete; trigger experience storage.
        On failure: feed the pytest output back to the LLM and continue fixing.

        Returns:
            True if verify passed (caller should exit the main loop);
            False otherwise.
        """
        logger.info("[VERIFY] starting pytest verification")
        hint = action.verify_hint or ""
        if hint:
            logger.info(f"[VERIFY] passing hint to verifier: {hint}")
        verifier = VerifierAgent(self._env, hint=hint)
        result = verifier.verify()

        logger.info(
            f"[VERIFY] result: success={result.success}, "
            f"framework={result.test_framework}, "
            f"collected={result.collect_count}, exit_code={result.exit_code}"
        )

        verify_summary = (
            f"framework: {result.test_framework}\n"
            f"collected: {result.collect_count}\n"
            f"exit_code: {result.exit_code}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": result.exit_code,
                "stdout": (
                    f"[VERIFY] success={result.success}, framework={result.test_framework}, "
                    f"collected={result.collect_count}\n{result.stdout or ''}"
                )[:1000],
                "stderr": (result.stderr or "")[:500],
            },
        })

        if result.success:
            logger.info("[VERIFY] passed; marking task complete")
            self._state.completed = True
            self._state.final_message = f"pytest verify passed; {result.collect_count} test cases"
            self._state.last_error = None
            self._last_verify_messages = result.messages
            return True
        else:
            logger.warning("[VERIFY] failed; feeding pytest output back to the LLM")
            self._state.last_error = verify_summary
            return False

    def _handle_finish(self, action: AgentAction) -> None:
        """FINISH: mark the task complete.

        The LLM should only emit this after a successful verify.
        """
        self._state.completed = True
        self._state.final_message = action.message or "task complete"

        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0,
                "stdout": f"[FINISH] {self._state.final_message}",
                "stderr": "",
            },
        })

    # =========================================================================
    # Situation building (for vector retrieval)
    # =========================================================================

    def _build_situation(self, last_error: str) -> str:
        """Build the raw situation text (recent commands + raw error + repo info).

        Used together with `LLMEngine.describe_situation()`:
        - LLM summary: semantic — matches situation_triggers / advice_nl.
        - Raw text:    keyword — matches keywords / regex.
        """
        parts = []

        recent = self._state.get_recent_history(3)
        if recent:
            done_steps = []
            for entry in recent:
                action = entry.get("action", {})
                cmd = action.get("content", {}).get("command", "")
                if cmd:
                    result = entry.get("result", {})
                    exit_code = result.get("exit_code", "?")
                    done_steps.append(f"  {cmd} (exit={exit_code})")
            if done_steps:
                parts.append("recent commands:\n" + "\n".join(done_steps))

        if last_error:
            error_text = last_error[:1500] if len(last_error) > 1500 else last_error
            parts.append(f"current error:\n{error_text}")

        parts.append(f"target repo: {self._state.repo_url}")

        return "\n\n".join(parts)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _close_clients(self) -> None:
        """Close LLM / XPU / Retriever HTTP connections (container is preserved)."""
        logger.info("closing LLM / XPU / retriever connections...")
        if self._retriever:
            self._retriever.close(full_history=self._state.history)
        self._llm.close()
        if hasattr(self._xpu, "close"):
            self._xpu.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close connections and destroy the container."""
        self._close_clients()
        self._env.cleanup()
        return False  # do not swallow exceptions
