"""CLI entry point — three-stage orchestration.

Stage 1: Setup (Agent reasoning, double-bounded by --max-steps and --phase1-timeout)
Stage 2: Phase 2 trial (only when Setup actively FINISHes)
Stage 3: Report (write results)
"""

import json
import os
import signal
import sys
from pathlib import Path

from .logger import get_logger
from .agent import SpeculativeSetupAgent
from .prosecutor_agent import ProsecutorAgent
from .judge_agent import JudgeAgent

logger = get_logger("main")


# Inherit BaseException rather than Exception — prevents downstream `except Exception`
# blocks (in llm_engine.py / retriever_agent.py etc.) from swallowing the SIGALRM
# timeout and letting phase 1 run forever.
class Phase1Timeout(BaseException):
    pass


def _build_traj_from_history(history: list[dict]) -> list[dict]:
    """Convert agent history to JSONL form, preserving full info for LLM extraction."""
    traj = []
    for entry in history:
        action = entry.get("action", {})
        result = entry.get("result", {})

        thought = action.get("thought", "")
        cmd = action.get("content", {}).get("command")
        if cmd:
            parts = []
            if thought:
                parts.append(f"thought: {thought}")
            parts.append(f"```bash\n{cmd}\n```")
            traj.append({
                "role": "assistant",
                "content": "\n".join(parts),
            })

        if result:
            exit_code = result.get("exit_code", "?")
            stdout = result.get("stdout") or ""
            stderr = result.get("stderr") or ""
            parts = [f"exit_code={exit_code}"]
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            output = "\n".join(parts)
            if output.strip() != f"exit_code={exit_code}":
                traj.append({
                    "role": "system",
                    "content": output,
                })
    return traj


def main() -> int:
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Three-stage orchestration: Setup -> Phase 2 -> Report")
    parser.add_argument("repo_url", help="Git repository URL")
    parser.add_argument("--max-steps", type=int, default=9999,
                        help="Max iteration steps (default 9999 ~ unlimited; phase 1 actually bounded by --phase1-timeout)")
    parser.add_argument("--phase1-timeout", type=int, default=1800, help="Phase 1 timeout in seconds")
    parser.add_argument("--no-xpu", action="store_true", help="Disable the XPU knowledge base")
    parser.add_argument("--output-dir", type=str, default="log", help="Output directory for the result JSON")
    parser.add_argument("--meta-json", default=None,
                        help='Task meta JSON: {"repository": "...", "primary_repos": [...], "component_repos": [...]}')
    args = parser.parse_args()

    if args.no_xpu:
        for key in ("dns", "XPU_DB_DNS", "XPU_VECTOR_ENABLED", "XPU_ENABLED"):
            os.environ.pop(key, None)
        import src.config as _cfg
        _cfg._config = None

    repo_url = args.repo_url
    max_iterations = args.max_steps
    phase1_timeout = args.phase1_timeout
    task_meta = json.loads(args.meta_json) if args.meta_json else None

    logger.info("starting three-stage orchestration")
    logger.info(f"target repo: {repo_url}")
    logger.info(f"max iterations: {max_iterations}")
    logger.info(f"phase 1 timeout: {phase1_timeout}s")
    if task_meta:
        logger.info(
            f"task meta: repository={task_meta.get('repository')}, "
            f"primary={len(task_meta.get('primary_repos', []))}, "
            f"component={len(task_meta.get('component_repos', []))}"
        )

    # -- Stage 1: Setup --
    logger.info("=== Stage 1: Setup (Agent reasoning) ===")
    agent = SpeculativeSetupAgent(repo_url, max_iterations, task_meta=task_meta)

    def _phase1_timeout_handler(signum, frame):
        # Raise Phase1Timeout (a BaseException subclass) rather than TimeoutError,
        # so downstream `except Exception` cannot swallow it.
        raise Phase1Timeout(f"phase 1 timed out (>{phase1_timeout}s), forcing termination")

    signal.signal(signal.SIGALRM, _phase1_timeout_handler)
    signal.alarm(phase1_timeout)
    try:
        setup_result = agent.run()
    except Phase1Timeout as e:
        signal.alarm(0)
        logger.error(f"phase 1 timeout: {e}")
        try:
            agent.env.destroy()
        except Exception:
            pass
        return 1
    finally:
        signal.alarm(0)
    logger.info(
        f"setup done: completed={setup_result.completed}, "
        f"steps={setup_result.steps_taken}, container={setup_result.container_id[:12]}"
    )

    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = repo_url.rstrip("/").split("/")[-1]

    # -- Stage 2: Phase 2 trial --
    logger.info("=== Stage 2: Phase 2 trial ===")

    phase2_success: bool | None
    phase2_reason: str
    prosecution_dict = None
    prosecution = None
    judgment = None

    # Phase 2 only runs for atomic single-repo tasks.
    # Multi-repo / non-atomic family runs (task_meta with primary_repos/component_repos)
    # are phase 1 only — Phase 2's prosecutor/judge protocol is single-repo specific.
    is_multi_repo = bool(
        task_meta
        and (task_meta.get("primary_repos") or task_meta.get("component_repos"))
    )

    try:
        if is_multi_repo:
            phase2_success = setup_result.completed
            phase2_reason = (
                "multi-repo / non-atomic family run: phase 2 skipped by design; "
                f"setup completed={setup_result.completed}"
            )
            logger.info(phase2_reason)
        elif setup_result.completed:
            logger.info("setup agent FINISHed; starting Phase 2 trial pipeline")

            # Prosecutor investigation
            logger.info("--- prosecutor investigation ---")
            prosecutor = ProsecutorAgent(
                agent.env,
                setup_result.history,
                setup_result.last_verify_messages,
            )
            prosecution = prosecutor.investigate()
            prosecution_dict = {
                "prosecute": prosecution.prosecute,
                "charges": prosecution.charges,
            }
            logger.info(
                f"prosecutor done: prosecute={prosecution.prosecute}, "
                f"charges={len(prosecution.charges)}"
            )

            if not prosecution.prosecute:
                phase2_success = True
                phase2_reason = "prosecutor found no substantive issue"
                logger.info("prosecutor declined to prosecute -> success=True")
            else:
                # Judge ruling
                logger.info("--- judge ruling ---")
                judgment = JudgeAgent(
                    setup_result.history,
                    setup_result.last_verify_messages,
                    prosecution,
                    env=agent.env,
                ).rule()

                verdict = judgment["verdict"]
                phase2_reason = judgment["reasoning"]
                if verdict == "not_guilty":
                    phase2_success = True
                elif verdict == "guilty":
                    phase2_success = False
                else:
                    phase2_success = None
                    phase2_reason = f"[error] {phase2_reason}"

                logger.info(
                    f"judge verdict: {verdict}, "
                    f"reasoning={phase2_reason[:100]}"
                )
        else:
            phase2_success = False
            phase2_reason = (
                f"setup agent timed out ({setup_result.steps_taken} steps); did not FINISH"
            )
            logger.info(f"setup did not finish, skipping phase 2: {phase2_reason}")

    except Exception as e:
        logger.error(f"phase 2 execution failed: {e}")
        phase2_success = None
        phase2_reason = f"[error] phase 2 execution failed: {e}"

    finally:
        # -- Cleanup: close clients + destroy container --
        if hasattr(agent._xpu, "close"):
            agent._xpu.close()
        # OURSYS_KEEP_CONTAINER=1 keeps the container (for PoC / manual inspection);
        # otherwise destroy it.
        keep_container = os.environ.get("OURSYS_KEEP_CONTAINER", "").strip() in ("1", "true", "yes")
        if keep_container:
            try:
                cid = agent.env.container_id[:12] if agent.env.container_id else "<unknown>"
                logger.info(f"OURSYS_KEEP_CONTAINER=1 -> container retained: {cid}")
            except Exception:
                logger.info("OURSYS_KEEP_CONTAINER=1 -> container retained")
        else:
            try:
                agent.env.destroy()
                logger.info("container destroyed")
            except Exception as e:
                logger.warning(f"container destruction failed: {e}")

    # -- Stage 3: Report --
    logger.info("=== Stage 3: Report ===")
    report = {
        "repo_url": repo_url,
        "setup": setup_result.to_dict(),
        "phase2": {
            "success": phase2_success,
            "reason": phase2_reason,
            "prosecution": prosecution_dict,
            "judgment": judgment,
        },
    }

    output_path = log_dir / f"{safe_name}_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"result written: {output_path}")

    verdict_str = "PASS" if phase2_success is True else ("FAIL" if phase2_success is False else "ERROR")
    print(f"\n{'='*50}")
    print(f"repo: {repo_url}")
    print(f"setup: {'done' if setup_result.completed else 'unfinished'} ({setup_result.steps_taken} steps)")
    print(f"phase2: {verdict_str}")
    print(f"reason: {phase2_reason}")
    print(f"detailed result: {output_path}")
    print(f"{'='*50}")

    if phase2_success is None:
        return 2  # error exit code
    return 0 if phase2_success else 1


if __name__ == "__main__":
    sys.exit(main())
