"""Task-meta rendering: inject family meta into the first user message.

Design principle (strict separation from case-specific hints):
- Inject:    repository, repo_url, primary_repos, component_repos
             (these are objective facts about the task).
- Do NOT inject: per-family `_strategy` / `_runtime_deps` / `_note` /
             `family_name` / role / role_detail descriptions
             (those are case-specific cheats).
- The container-environment block is also an objective fact about how the
  protocol runs (base image / network mode / docker socket mount); shared
  across all families, not case-specific.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_repo_to_family_index(spec_path: Path) -> dict[str, dict]:
    """Build a `repo_full_name -> {primary_repos, component_repos}` index from
    a non-atomic-family JSON spec (e.g. `repo_set_non_atomic_workflows__60.json`).

    Args:
        spec_path: path to the family spec JSON.

    Returns:
        A dict keyed by repo full name (e.g. "org/repo"); the value is
        the family's member list. Repos not in the spec do not appear.
    """
    if not spec_path.exists():
        return {}

    data = json.loads(spec_path.read_text(encoding="utf-8"))
    families = data.get("families", []) or []

    def _extract_names(entries: list) -> list[str]:
        """Pull repo names out of `[{"repository": "x/y", "role": ...}, ...]`
        or `["x/y", ...]`. Drop role / role_detail and any other field that
        could leak case-specific hints into the prompt."""
        names: list[str] = []
        for e in entries or []:
            if isinstance(e, dict):
                name = e.get("repository") or e.get("repo")
                if name:
                    names.append(name)
            elif isinstance(e, str):
                names.append(e)
        return names

    index: dict[str, dict] = {}
    for fam in families:
        primary = _extract_names(fam.get("primary_repos") or [])
        component = _extract_names(fam.get("component_repos") or [])
        meta = {
            "primary_repos": primary,
            "component_repos": component,
        }
        for repo in primary + component:
            index[repo] = meta
    return index


def render_first_user_message(
    repo_url: str,
    repository: str,
    family_meta: dict | None,
) -> str:
    """Render the first user message.

    Args:
        repo_url: repo URL (e.g. https://github.com/org/repo).
        repository: repo full name (e.g. org/repo).
        family_meta: family member list for this repo (from
            `build_repo_to_family_index`); pass None for ordinary atomic repos
            to skip the family block.

    Returns:
        The full first user message text.
    """
    parts: list[str] = []

    parts.append("=== Task ===")
    parts.append(
        "Configure a runnable environment for the target repository under "
        "/workspace/repo, and ultimately pass VERIFY."
    )
    parts.append("")
    parts.append(f"Target repo: {repository}")
    parts.append(f"Repo URL:    {repo_url}")
    parts.append("")

    if family_meta:
        primary = family_meta.get("primary_repos") or []
        component = family_meta.get("component_repos") or []
        parts.append(
            "=== Repository family (members of the family this repo belongs to; "
            "for your own architectural judgment only) ==="
        )
        parts.append(
            "This repository is part of a repository family. The full member list is "
            "below — you do not need to set up each one, but the target repo's core "
            "functionality may depend on some of them:"
        )
        parts.append("")
        if primary:
            parts.append("primary_repos:")
            for r in primary:
                parts.append(f"  - {r}")
        if component:
            shown = component[:5]
            parts.append("component_repos:")
            for r in shown:
                parts.append(f"  - {r}")
            if len(component) > len(shown):
                parts.append(
                    f"  ... ({len(component)} total; only the first {len(shown)} listed)"
                )
        parts.append("")
        parts.append("You decide:")
        parts.append(
            "- whether these repos form a host/plugin, server/client, or framework/data relationship;"
        )
        parts.append(
            "- whether the target repo needs to `git clone` one of the host repos to "
            "/workspace/upstream;"
        )
        parts.append("- whether you need to `docker run` a sibling service.")
        parts.append("")

    parts.append("=== Container environment (read this) ===")
    parts.append(
        "- Base image: python:3.10  (no docker CLI, chromium, mysql-client, or mariadb-client preinstalled)"
    )
    parts.append(
        "- The container starts with --network=host and the host docker socket is "
        "mounted at /var/run/docker.sock"
    )
    parts.append(
        "  - to `docker run` any sibling, install the docker CLI first: "
        "`apt-get update && apt-get install -y docker.io`"
    )
    parts.append(
        "  - sibling containers should also use --network=host; access them via "
        "localhost:<port>"
    )
    parts.append(
        "- Working directory: /workspace/repo (already `git clone --depth=1`-ed)"
    )
    parts.append(
        "- /workspace/upstream is the conventional clone target for any host repo."
    )
    parts.append("")

    parts.append("=== Suggested first steps ===")
    parts.append(
        "1. Inspect README.md and pyproject.toml; combine with the family list "
        "above to judge the architecture."
    )
    parts.append(
        "2. If you need to `docker run` a sibling, install the docker CLI first."
    )
    parts.append("")
    parts.append("Analyze the current state and decide the next action.")

    return "\n".join(parts)
