"""MVP for extracting environment experiences (XPUs) from agent trajectories.

Pipeline (4 stages):
1. Load trajectory files (iter_traj_files / load_traj).
2. Heuristic gating (heuristic_stats_for_traj / heuristic_is_candidate).
3. LLM extraction (build_traj_prompt / openai_compatible_chat_completions).
4. Emit JSONL (one record per line).

Supported trajectory formats:
- EnvBench: structured records with node="commands_history".
- Repo2Run: bash blocks inside Markdown.
- This project's agent format: SHELL_COMMAND actions in JSON.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAJ_DIR = ROOT_DIR / "tmp" / "traj_py_subset_50_kimi"
DEFAULT_OUTPUT = ROOT_DIR / "xpuExtract" / "outputs" / "traj_xpu_mvp.jsonl"

DEFAULT_LLM_MODEL = os.environ.get("XPU_EXTRACT_MODEL", os.environ.get("MOONSHOT_MODEL", "gpt-4o-2024-05-13"))
DEFAULT_API_KEY_ENV = os.environ.get("XPU_EXTRACT_API_KEY_ENV", "OPENAI_API_KEY")
DEFAULT_BASE_URL_ENV = os.environ.get("XPU_EXTRACT_BASE_URL_ENV", "OPENAI_BASE_URL")
DEFAULT_TIMEOUT_SEC = int(os.environ.get("XPU_EXTRACT_TIMEOUT", "60"))

ERROR_KEYWORDS = [
    "ModuleNotFoundError",
    "ImportError",
    "No module named",
    "cannot import name",
    "Could not find a version",
    "command not found",
    "Permission denied",
    "error:",
    "Error:",
    "Traceback",
    "failed with exit code",
]

ENV_CMD_KEYWORDS = [
    "pip install",
    "poetry install",
    "apt-get install",
    "conda install",
    "python setup.py",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_env_or_raise(name: str) -> str:
    """Fetch a required env var; falls back from MOONSHOT_API_KEY to OPENAI_API_KEY."""
    val = os.environ.get(name)
    if not val:
        if name == "MOONSHOT_API_KEY":
            val = os.environ.get("OPENAI_API_KEY")
    if not val:
        raise RuntimeError(f"missing required env var: {name}")
    return val


def openai_compatible_chat_completions(
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    base_url: str,
    timeout_sec: int,
    response_format_json: bool = True,
) -> Dict[str, Any]:
    """OpenAI-compatible chat completions (works with OpenAI / ARK / Kimi)."""
    # ARK uses /v3, OpenAI uses /v1; only auto-append /v1 if neither is present.
    if "v1" not in base_url and "v3" not in base_url and not base_url.endswith("/"):
        base_url += "/v1"
    url = base_url.rstrip("/") + "/chat/completions"

    masked_key = api_key[:8] + "..." if api_key else "None"
    print(f"[DEBUG] LLM Request URL: {url}")
    print(f"[DEBUG] LLM API Key: {masked_key}")
    print(f"[DEBUG] LLM Model: {model}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_sec)
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def parse_llm_json(s: str) -> Dict[str, Any]:
    """Parse JSON out of LLM output: handle BOM, ```json fences, or raw JSON."""
    s = s.strip()
    if s.startswith("﻿"):
        s = s.lstrip("﻿")
    if s.startswith("```"):
        if s.startswith("```json"):
            s = s[len("```json"):].strip()
        else:
            s = s[3:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return json.loads(s)


def truncate(text: Any, max_len: int) -> str:
    """Keep the first and last halves; mark the middle as truncated."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    keep = max_len // 2
    return text[:keep] + "\n... [TRUNCATED] ...\n" + text[-keep:]


def load_llm_config_from_env() -> Dict[str, Any]:
    return {
        "llm_model": DEFAULT_LLM_MODEL,
        "api_key_env_var": DEFAULT_API_KEY_ENV,
        "base_url_env_var": DEFAULT_BASE_URL_ENV,
        "timeout_sec": DEFAULT_TIMEOUT_SEC,
        "llm_language": "en",
    }


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def iter_traj_files(traj_path: Path) -> List[Path]:
    """Single file or directory; in directory mode keep .jsonl files with '@'."""
    if traj_path.is_file():
        return [traj_path]
    if traj_path.is_dir():
        return sorted([p for p in traj_path.glob("*.jsonl") if "@" in p.name])
    raise FileNotFoundError(str(traj_path))


def parse_repo_revision_from_name(path: Path) -> Tuple[str, str]:
    """File name pattern: org__repo@revision.jsonl -> ("org/repo", "revision")."""
    name = path.name
    if not (name.endswith(".jsonl") and "@" in name):
        return "unknown/repo", "unknown"
    base = name[: -len(".jsonl")]
    try:
        repo_part, rev = base.rsplit("@", 1)
    except ValueError:
        return base, "unknown"
    repo = repo_part.replace("__", "/")
    return repo, rev


def load_traj(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# Heuristic gating
# ---------------------------------------------------------------------------

def _iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def extract_commands_history(traj: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pull executed commands from any of the 3 trajectory formats."""
    cmds = []

    bash_pattern = re.compile(r"```bash\s+(.*?)\s+```", re.DOTALL)

    for item in traj:
        # EnvBench
        if item.get("node") == "commands_history":
            raw = item.get("commands") or []
            if isinstance(raw, list):
                return raw

        content = item.get("content", "")
        role = item.get("role", "")

        if role == "assistant" and content:
            # This project's agent JSON format.
            try:
                if isinstance(content, str) and content.strip().startswith("{"):
                    data = json.loads(content)
                    cmd = None
                    if isinstance(data, dict):
                        inner_content = data.get("content")
                        if isinstance(inner_content, dict):
                            # {"action_type": "SHELL_COMMAND", "content": {"command": "..."}}
                            cmd = inner_content.get("command")
                        elif data.get("command"):
                            # {"command": "..."}
                            cmd = data.get("command")

                    if cmd:
                        cmds.append({"command": cmd, "exit_code": 0})
                        continue
            except json.JSONDecodeError:
                pass

            # Repo2Run-style markdown bash blocks.
            matches = bash_pattern.findall(content)
            for match in matches:
                clean_cmd = match.strip()
                cmds.append({"command": clean_cmd, "exit_code": 0})

    return cmds


def heuristic_stats_for_traj(traj: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_agent_steps = 0
    num_error_keywords = 0

    for item in traj:
        if item.get("role") == "assistant" or item.get("node") == "agent":
            num_agent_steps += 1

        for text in _iter_strings(item):
            t_low = text.lower()
            if any(kw.lower() in t_low for kw in ERROR_KEYWORDS):
                num_error_keywords += 1

    cmds = extract_commands_history(traj)
    num_commands = len(cmds)
    num_env_commands = 0

    for c in cmds:
        cmd_str = str(c.get("command", ""))
        cmd_low = cmd_str.lower()
        if any(kw.lower() in cmd_low for kw in ENV_CMD_KEYWORDS):
            num_env_commands += 1

    return {
        "num_agent_steps": num_agent_steps,
        "num_commands": num_commands,
        "num_env_commands": num_env_commands,
        "num_error_keywords": num_error_keywords,
    }


def heuristic_is_candidate(stats: Dict[str, Any]) -> Tuple[bool, float]:
    """Loose gating: env_cmd +5, error_kw +5, any_cmd +1; pass when score > 0."""
    score = 0.0

    if stats.get("num_env_commands", 0) >= 1:
        score += 5.0
    if stats.get("num_error_keywords", 0) >= 1:
        score += 5.0
    if stats.get("num_commands", 0) >= 1:
        score += 1.0

    print(f"[DEBUG] Heuristic Stats: {stats}, Score: {score}")

    return score > 0, score


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def build_traj_prompt(
    repo: str,
    rev: str,
    traj: List[Dict[str, Any]],
    stats: Dict[str, Any],
    cfg: Dict[str, Any],
    phase2_context: Dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
    """Build the LLM prompt for XPU extraction."""
    cmds = extract_commands_history(traj)
    lines_cmds: List[str] = []
    for c in cmds:
        cmd_str = str(c.get("command", ""))
        lines_cmds.append(f"$ {cmd_str}")
    commands_text = truncate("\n".join(lines_cmds), 4000)

    error_lines: List[str] = []
    for item in traj:
        # Our agent records execution results under the user role.
        if item.get("role") in ("system", "user"):
            text = item.get("content", "")
            if any(kw.lower() in text.lower() for kw in ERROR_KEYWORDS):
                error_lines.append(text)

    if len(error_lines) > 30:
        error_lines = error_lines[:15] + ["... [TRUNCATED] ..."] + error_lines[-10:]
    errors_text = truncate("\n".join(error_lines), 4000)

    system_text = (
        "You are a senior expert in Python project environment configuration and dependency issues."
        "\nYou will be given a complete agent trajectory (executed commands and error logs) from"
        "\nan automated environment setup of a repository, and (when available) the Phase 2"
        "\nProsecutor-Judge adjudication signal phase2_context for that trajectory."
        "\n"
        "\n## Mandatory four-step distillation procedure"
        "\n"
        "\n**[Step 1: Verdict-Aware Ingestion] Eat the verdict first, then read the trajectory**"
        "\n   - If phase2_context.prosecution_charges is non-empty: every charge is a candidate"
        "\n     problem with the cleanest causal signal — distill from those **first**."
        "\n     Even when verdict=guilty, distill the generalizable patterns inside the charges."
        "\n   - If verdict=not_guilty but verifier_summary shows test-avoidance (e.g. --ignore to"
        "\n     skip tests), still inspect whether a generalizable environment issue is hiding"
        "\n     behind that avoidance."
        "\n   - When phase2_context is absent, fall back to the agent's self-reported success /"
        "\n     failure signals."
        "\n"
        "\n**[Step 2: Forward Attribution] Trace each problem forward to the action that actually fixed it**"
        "\n   Error recovery is rarely linear — a fix's true effect often only shows up several"
        "\n   steps later. For each independent problem you have identified, run this attribution:"
        "\n     a. Locate the first occurrence of the error in the trajectory (first_error_step)."
        "\n     b. Scan forward through subsequent actions for the command that actually"
        "\n        eliminated the error (fix_step); intermediate try-fail-roll-back actions do not"
        "\n        count as the fix."
        "\n     c. If a single error was resolved by multiple actions together (e.g., install a"
        "\n        system library first, then a Python package), record this whole causal chain"
        "\n        in the atoms array of that XPU, in execution order."
        "\n     d. If an error was never resolved (still present at trajectory end), it can still"
        "\n        be recorded as an [abandoned package / version cliff / pitfall] entry, with"
        "\n        advice_nl explaining the workaround."
        "\n   The distilled command must be the one that actually did the work; do not record"
        "\n   ineffective attempts as the fix."
        "\n"
        "\n**[Step 3: Schema-Level Distillation] Map each problem-fix pair onto the XPU schema**"
        "\n   - Each independent problem-fix causal pair → one XPU."
        "\n   - Do not mix several unrelated problems into a single XPU."
        "\n   - Abstract into a pattern rather than recording the concrete command: lift"
        "\n     repository-specific package names / paths / versions into reusable toolchain"
        "\n     regularities / package-level install patterns / environment-configuration patterns."
        "\n"
        "\n"
        "\n[Distillation principles (mandatory)]"
        "\n- prosecution_charges are the cleanest causal knowledge source — distill from them first."
        "\n- Even when verdict=guilty, distill the generalizable patterns within."
        "\n- Three categories of experience are allowed (in priority order):"
        "\n  1. [Toolchain pattern] Regularities at the build-tool / package-manager level, e.g.:"
        "\n     \"When pyproject.toml contains [tool.poetry], you must use poetry install rather than pip install -r\""
        "\n     \"In a conda virtual environment, packages installed via pip install may be invisible to conda\""
        "\n  2. [Package-level install pattern] Known install pitfalls of specific Python packages —"
        "\n     knowledge that applies whenever a different repository encounters the same package, e.g.:"
        "\n     \"psycopg2 needs the system library libpq-dev or compilation fails; alternatively switch to psycopg2-binary\""
        "\n     \"lxml compilation needs libxml2-dev libxslt1-dev\""
        "\n     \"The latest version of package X is incompatible with Python 3.10; pin X==1.2.3\""
        "\n  3. [Environment-configuration pattern] System-level config / permission / path problems, e.g.:"
        "\n     \"Packages installed with pip install --user are not on PATH; need export PATH=$HOME/.local/bin:$PATH\""
        "\n     \"A Docker container lacks locale settings, causing certain packages to crash on import with UnicodeError\""
        "\n- The only forbidden category: pure repo-specific facts, i.e. \"this repo needs package X\""
        "\n  without explaining WHY (why X is tricky to install)."
        "\n  Test: if you strip out the repo name, is this experience still useful to other repos that"
        "\n  use the same package / tool? If yes, record; otherwise discard."
        "\n- verifier_summary is helpful — the actual cause of test failure is more trustworthy than the"
        "\n  agent's guess."
        "\n- One XPU = one root cause; never mix several unrelated problems."
        "\n- situation_triggers are the key to future queries hitting this experience — fill in"
        "\n  concrete scenarios, never abstract words like \"install failed\"."
        "\n  Example: [\"poetry project\", \"pyproject.toml contains [tool.poetry]\", \"wrongly used pip install instead of poetry install\"]"
        "\n- [Same applies to timed-out / failing trajectories.] The most valuable patterns in"
        "\n  failing trajectories are typically:"
        "\n    1. Operations the agent loops on without converging — e.g., installing dependencies one"
        "\n       by one and verifying after each, when it should have collected all missing packages"
        "\n       and installed them in a single batch."
        "\n    2. Recognition of abandoned packages / version cliffs — the latest version of a package"
        "\n       is incompatible with the current Python / framework; downgrade or abandon."
        "\n    3. A package has no available implementation under the current Python version (e.g.,"
        "\n       PyPI only has a Python 2 release); record the package name and the alternative."
        "\n    These pitfall lessons are extremely important for future agents to avoid the same trap"
        "\n    and must be distilled."
        "\n- Do not produce an id field; the system will assign a unique ID automatically."
        "\n"
        "\nYour answer must be a strict JSON object, with no extra text."
    )

    user_payload: Dict[str, Any] = {
        "repository": repo,
        "revision": rev,
        "stats": stats,
        "commands_history_text": commands_text,
        "error_snippets_text": errors_text,
        "xpu_schema": {
            "signals": {
                "applicability": {
                    "lang": "e.g. python",
                    "os": ["relevant operating systems, e.g. linux"],
                    "python": ["relevant Python version prefixes, e.g. 3.8"],
                    "tools": ["relevant tools, e.g. pytest, pip"],
                },
                "regex": ["regex matching this error"],
                "keywords": ["keywords for coarse retrieval"],
                "situation_triggers": (
                    "2-4 strings describing under which project/tool/state this experience applies, "
                    "e.g. [\"poetry project\", \"pyproject.toml contains [tool.poetry]\", "
                    "\"wrongly used pip install instead of poetry install\"]. "
                    "The more concrete the better — used for vector retrieval recall."
                ),
            },
            "advice_nl": ["1-5 natural-language suggestions explaining the root cause and the fix idea"],
            "atoms": [
                {
                    "name": (
                        "[Must use one of the following names; do not invent new names]\n"
                        "  pip_install   — args: {name: 'package name or . or .[extra]', spec: '>=1.0', flags: []}\n"
                        "  pip_pin       — args: {name: 'package name', spec: '==1.2.3'}\n"
                        "  apt_install   — args: {packages: ['pkg1', 'pkg2']}\n"
                        "  shell         — args: {cmd: 'arbitrary bash command'}  ← generic fallback when the above are insufficient\n"
                        "  set_env       — args: {key: 'VAR', value: 'val'}\n"
                        "  set_umask     — args: {value: '0o022'}\n"
                        "  set_django_setting — args: {key: 'SETTING', value: 'val'}\n"
                        "  or_upgrade_pkg     — args: {name: 'package name', min_version: '1.0'}\n"
                        "  conda_install — args: {packages: ['pkg']}\n"
                        "  npm_install   — args: {packages: ['pkg']}\n"
                        "  set_pytest_flag    — args: {name: '--flag', value: 'val'}\n"
                        "  adjust_command     — args: {cmd: 'corrected full command'}"
                    ),
                    "args": "fill in according to the format for the chosen name above",
                }
            ],
        },
        "output_requirement": (
            "You must output a JSON object of shape {decision, reason, xpus}. "
            "decision is either 'skip' or 'xpu'. "
            "When decision='skip', the trajectory contains nothing worth distilling and xpus is an empty array []. "
            "When decision='xpu', xpus is an array of one or more XPU objects compatible with xpu_schema; "
            "each XPU corresponds to one independent environment problem and its fix in the trajectory. "
            "Do not produce an id field; the system will assign a unique ID automatically. "
            "All explanatory text must be in English."
        ),
        "language": cfg.get("llm_language", "en"),
    }

    if phase2_context:
        user_payload["phase2_context"] = {
            "prosecution_charges": phase2_context.get("prosecution_charges", []),
            "verdict": phase2_context.get("verdict"),
            "judge_reasoning": phase2_context.get("judge_reasoning", ""),
            "verifier_summary": phase2_context.get("verifier_summary", ""),
            "prosecutor_investigation": phase2_context.get("prosecutor_investigation", ""),
        }

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_xpu_from_trajs(
    traj_path: Path,
    output_jsonl: Path,
    phase2_context: Dict[str, Any] | None = None,
) -> None:
    """Batch-extract XPUs from trajectory file(s) and write JSONL output."""
    load_dotenv()
    cfg = load_llm_config_from_env()

    api_key = get_env_or_raise(cfg["api_key_env_var"])
    base_url = os.environ.get(cfg["base_url_env_var"]) or "https://api.openai.com/v1"

    files = iter_traj_files(traj_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for path in tqdm(files, total=len(files), desc="extracting XPU from trajectories"):
            repo, rev = parse_repo_revision_from_name(path)
            traj = load_traj(path)
            stats = heuristic_stats_for_traj(traj)
            is_candidate, score = heuristic_is_candidate(stats)
            stats["heuristic_score"] = score
            stats["heuristic_is_candidate"] = is_candidate

            llm_decision: str = "heuristic_skip"
            llm_reason: str | None = None
            xpu_obj: Dict[str, Any] | None = None
            usage: Dict[str, Any] = {}
            error_info: str | None = None

            if is_candidate:
                try:
                    messages = build_traj_prompt(repo, rev, traj, stats, cfg, phase2_context=phase2_context)
                    raw = openai_compatible_chat_completions(
                        model=cfg["llm_model"],
                        messages=messages,
                        api_key=api_key,
                        base_url=base_url,
                        timeout_sec=cfg["timeout_sec"],
                        response_format_json=True,
                    )
                    content = raw["choices"][0]["message"]["content"]
                    usage = raw.get("usage") or {}
                    parsed = parse_llm_json(content)
                    llm_decision = str(parsed.get("decision") or "error")
                    llm_reason = parsed.get("reason")
                    if llm_decision == "xpu":
                        # Accept both the new "xpus" array and the legacy "xpu" single object.
                        xpu_list = parsed.get("xpus") or []
                        if not xpu_list:
                            single = parsed.get("xpu")
                            if single:
                                xpu_list = [single]
                    elif llm_decision not in {"skip", "xpu"}:
                        llm_decision = "error"
                        error_info = f"unexpected decision value: {parsed!r}"
                except Exception as e:
                    llm_decision = "error"
                    error_info = str(e)

            # One XPU per output line for downstream per-record processing.
            if llm_decision == "xpu" and xpu_list:
                for xpu_obj in xpu_list:
                    out_obj = {
                        "repository": repo,
                        "revision": rev,
                        "traj_path": str(path),
                        "heuristics": stats,
                        "llm_decision": "xpu",
                        "llm_reason": llm_reason,
                        "xpu": xpu_obj,
                        "llm_model": cfg.get("llm_model"),
                        "usage": usage,
                        "error": error_info,
                    }
                    f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            else:
                # Still record skipped/decision-failed trajectories for audit.
                out_obj = {
                    "repository": repo,
                    "revision": rev,
                    "traj_path": str(path),
                    "heuristics": stats,
                    "llm_decision": llm_decision,
                    "llm_reason": llm_reason,
                    "xpu": None,
                    "llm_model": cfg.get("llm_model"),
                    "usage": usage,
                    "error": error_info,
                }
                f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic-gated LLM extraction of XPUs from EnvBench trajectories")
    parser.add_argument("--traj", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    extract_xpu_from_trajs(Path(args.traj), Path(args.output))


if __name__ == "__main__":
    main()
