"""XPU client implementations: Mock / HTTP / Vector / Noop."""

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .config import get_config
from .logger import get_logger
from .models import XPUSuggestion, AttributionReport

logger = get_logger("xpu")


class XPUClientBase(ABC):
    @abstractmethod
    def query(self, context: dict[str, Any], exclude_ids: list[str] | None = None) -> list[XPUSuggestion]:
        pass

    @abstractmethod
    def submit_feedback(self, report: AttributionReport) -> None:
        pass


class MockXPUClient(XPUClientBase):
    """Mock client backed by a static keyword → fix table; for dev/debug only."""

    KNOWLEDGE_BASE: list[dict] = [
        {
            "keywords": ["command not found", "npm"],
            "description": "Install Node.js and npm",
            "commands": ["apt-get update", "apt-get install -y nodejs npm"],
            "confidence": 0.95,
        },
        {
            "keywords": ["command not found", "python", "pip"],
            "description": "Install Python and pip",
            "commands": ["apt-get update", "apt-get install -y python3 python3-pip python3-venv"],
            "confidence": 0.95,
        },
        {
            "keywords": ["ModuleNotFoundError", "No module named"],
            "description": "Install missing Python dependencies",
            "commands": ["pip install -r requirements.txt"],
            "confidence": 0.8,
        },
        {
            "keywords": ["ENOENT", "package.json"],
            "description": "Install Node.js dependencies",
            "commands": ["npm install"],
            "confidence": 0.85,
        },
        {
            "keywords": ["permission denied"],
            "description": "Fix file permissions",
            "commands": ["chmod +x ./script.sh"],
            "confidence": 0.7,
        },
        {
            "keywords": ["EACCES", "npm", "global"],
            "description": "Configure npm global prefix",
            "commands": [
                "npm config set prefix ~/.npm-global",
                "export PATH=~/.npm-global/bin:$PATH",
            ],
            "confidence": 0.8,
        },
        {
            "keywords": ["cargo", "command not found"],
            "description": "Install Rust and Cargo",
            "commands": [
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                "source $HOME/.cargo/env",
            ],
            "confidence": 0.9,
        },
        {
            "keywords": ["go", "command not found"],
            "description": "Install Go",
            "commands": ["apt-get update", "apt-get install -y golang"],
            "confidence": 0.9,
        },
        {
            "keywords": ["java", "command not found", "javac"],
            "description": "Install JDK",
            "commands": ["apt-get update", "apt-get install -y default-jdk"],
            "confidence": 0.9,
        },
        {
            "keywords": ["docker", "command not found"],
            "description": "Install Docker",
            "commands": ["apt-get update", "apt-get install -y docker.io"],
            "confidence": 0.9,
        },
        {
            "keywords": ["make", "command not found"],
            "description": "Install build-essential",
            "commands": ["apt-get update", "apt-get install -y build-essential"],
            "confidence": 0.95,
        },
        {
            "keywords": ["cmake", "command not found"],
            "description": "Install CMake",
            "commands": ["apt-get update", "apt-get install -y cmake"],
            "confidence": 0.95,
        },
        {
            "keywords": ["libmysqlclient", "mysql_config"],
            "description": "Install MySQL client dev headers",
            "commands": ["apt-get update", "apt-get install -y libmysqlclient-dev"],
            "confidence": 0.9,
        },
        {
            "keywords": ["libpq", "pg_config"],
            "description": "Install PostgreSQL client dev headers",
            "commands": ["apt-get update", "apt-get install -y libpq-dev"],
            "confidence": 0.9,
        },
    ]

    def __init__(self):
        self._feedback_history: list[AttributionReport] = []

    def query(self, context: dict[str, Any], exclude_ids: list[str] | None = None) -> list[XPUSuggestion]:
        # score = (matched_keywords / total_keywords) × base_confidence
        error_log = context.get("error", "") or context.get("error_log", "")
        combined_text = f"{error_log}".lower()

        suggestions = []
        for entry in self.KNOWLEDGE_BASE:
            matched_keywords = sum(
                1 for kw in entry["keywords"]
                if kw.lower() in combined_text
            )
            if matched_keywords > 0:
                score = matched_keywords / len(entry["keywords"]) * entry["confidence"]
                if score > 0.3:
                    suggestion = XPUSuggestion(
                        id=f"xpu_{uuid.uuid4().hex[:8]}",
                        description=entry["description"],
                        commands=entry["commands"],
                        confidence=score,
                        source="mock",
                    )
                    suggestions.append((score, suggestion))

        suggestions.sort(key=lambda x: x[0], reverse=True)
        result = [s[1] for s in suggestions[:3]]

        if result:
            logger.info(f"XPU query returned {len(result)} suggestions")
            for s in result:
                logger.info(f"  - {s}")
        else:
            logger.debug("XPU: no matching suggestions")

        return result

    def submit_feedback(self, report: AttributionReport) -> None:
        self._feedback_history.append(report)

        logger.info("=" * 60)
        logger.info("XPU Attribution Report")
        logger.info("=" * 60)
        logger.info(f"  suggestion_id: {report.suggestion_id}")
        logger.info(f"  timestamp: {report.timestamp}")
        logger.info(f"  repo_context: {report.repo_context}")
        logger.info(f"  outcome: {report.outcome}")
        logger.info(f"  score: {report.score}")
        logger.info(f"  error_before: {report.error_before[:200] if report.error_before else 'N/A'}...")
        logger.info(f"  error_after: {report.error_after[:200] if report.error_after else 'N/A'}...")
        logger.info(f"  logs:")
        for i, log in enumerate(report.logs):
            logger.info(f"    [{i+1}] {log.command} -> exit_code={log.exit_code}")
        logger.info("=" * 60)


class HTTPXPUClient(XPUClientBase):
    """XPU client that talks to a remote HTTP service."""

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30)

    def query(self, context: dict[str, Any], exclude_ids: list[str] | None = None) -> list[XPUSuggestion]:
        try:
            response = self._client.post(f"{self._base_url}/api/query", json=context)
            response.raise_for_status()
            data = response.json()

            suggestions = []
            for item in data.get("suggestions", []):
                suggestions.append(XPUSuggestion(
                    id=item["id"],
                    description=item["description"],
                    commands=item.get("commands", []),
                    confidence=item.get("confidence", 0.5),
                    source="http",
                ))

            logger.info(f"XPU HTTP query returned {len(suggestions)} suggestions")
            return suggestions

        except httpx.HTTPError as e:
            logger.warning(f"XPU HTTP query failed: {e}")
            return []

    def submit_feedback(self, report: AttributionReport) -> None:
        try:
            response = self._client.post(
                f"{self._base_url}/api/feedback",
                json=report.to_dict(),
            )
            response.raise_for_status()
            logger.info(f"feedback submitted: {report.suggestion_id}")

            logger.info("=" * 60)
            logger.info("XPU Attribution Report")
            logger.info(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            logger.info("=" * 60)

        except httpx.HTTPError as e:
            logger.warning(f"feedback submission failed: {e}")

    def close(self) -> None:
        self._client.close()


class VectorXPUClient(XPUClientBase):
    """Production XPU client backed by PostgreSQL + pgvector."""

    def __init__(self, dns: str):
        from .xpu.xpu_vector_store import XpuVectorStore, text_to_embedding, build_xpu_text
        from .xpu.xpu_adapter import XpuAtom, render_atom_to_commands
        self._store = XpuVectorStore(connection_string=dns)
        self._text_to_embedding = text_to_embedding
        self._build_xpu_text = build_xpu_text
        self._render_atom_to_commands = render_atom_to_commands
        self._id_to_raw: dict[str, dict] = {}
        host = dns.split('@')[-1] if '@' in dns else '...'
        logger.info(f"VectorXPUClient ready (host: {host})")

    def query(self, context: dict[str, Any], exclude_ids: list[str] | None = None) -> list[XPUSuggestion]:
        from .xpu.xpu_adapter import XpuAtom

        error_text = context.get("error", "") or context.get("error_log", "")
        if not error_text:
            return []

        try:
            embedding = self._text_to_embedding(error_text)
            results = self._store.search(embedding, k=3, exclude_ids=exclude_ids)
        except Exception as e:
            logger.warning(f"VectorXPUClient query failed: {e}")
            return []

        suggestions = []
        result_ids = []

        for res in results:
            xpu_id = res["id"]
            advice_nl = res.get("advice_nl") or []
            atoms = res.get("atoms") or []
            similarity = float(res.get("similarity", 0.5))

            commands = []
            for a in atoms:
                atom = XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                commands.extend(self._render_atom_to_commands(atom))

            composite = float(res.get("composite_score", similarity))
            if composite >= 0.8:
                confidence_level = "high"
            elif composite >= 0.7:
                confidence_level = "medium"
            else:
                confidence_level = "low"

            suggestion = XPUSuggestion(
                id=xpu_id,
                description=f"[{confidence_level}] " + "\n".join(advice_nl),
                commands=commands,
                confidence=composite,
                source="vector_db",
                atoms=atoms,
            )
            suggestions.append(suggestion)
            result_ids.append(xpu_id)
            self._id_to_raw[xpu_id] = res

        if result_ids:
            try:
                self._store.increment_telemetry(result_ids, "hits")
            except Exception as e:
                logger.warning(f"telemetry hits write failed: {e}")

        logger.info(f"VectorXPU query returned {len(suggestions)} suggestions")
        for s in suggestions:
            logger.info(f"  - [{s.confidence:.3f}] {s.id}: {s.description[:60]}...")

        return suggestions

    def submit_feedback(self, report: AttributionReport) -> None:
        try:
            if report.score > 0:
                self._store.increment_telemetry([report.suggestion_id], "successes")
            elif report.score < 0:
                self._store.increment_telemetry([report.suggestion_id], "failures")
        except Exception as e:
            logger.warning(f"telemetry feedback write failed: {e}")

        logger.info("=" * 60)
        logger.info("XPU Attribution Report (VectorXPUClient)")
        logger.info(f"  suggestion_id: {report.suggestion_id}")
        logger.info(f"  outcome: {report.outcome}  score: {report.score}")
        logger.info(f"  error_before: {(report.error_before or '')[:200]}...")
        logger.info(f"  error_after:  {(report.error_after or '')[:200]}...")
        logger.info("=" * 60)

    def close(self) -> None:
        self._store.close()


def create_xpu_client() -> XPUClientBase:
    config = get_config().xpu

    if config.disabled:
        logger.info("XPU disabled; using NoopXPUClient")
        return NoopXPUClient()
    if config.vector_enabled and config.db_dns:
        logger.info("using VectorXPUClient (pgvector)")
        return VectorXPUClient(config.db_dns)
    elif config.enabled:
        logger.info(f"using HTTPXPUClient: {config.base_url}")
        return HTTPXPUClient(config.base_url)
    else:
        logger.info("using MockXPUClient")
        return MockXPUClient()


class NoopXPUClient(XPUClientBase):
    """Fully disabled XPU client: returns nothing, accepts no feedback."""

    def query(self, context: dict[str, Any], exclude_ids: list[str] | None = None) -> list[XPUSuggestion]:
        return []

    def submit_feedback(self, report: AttributionReport) -> None:
        return None
