"""Centralised config injection: all knobs come from env / .env files."""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).parent.parent


def _load_env_files() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    load_dotenv(PROJECT_ROOT / ".env.local", override=True)


_load_env_files()


@dataclass(frozen=True)
class ARKConfig:
    api_key: str
    base_url: str
    deployment: str


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class DockerConfig:
    base_image: str
    work_dir: str
    timeout: int


@dataclass(frozen=True)
class XPUConfig:
    base_url: str
    enabled: bool
    disabled: bool
    db_dns: str | None
    vector_enabled: bool


@dataclass(frozen=True)
class Config:
    ark: ARKConfig | None
    openai: OpenAIConfig | None
    docker: DockerConfig
    xpu: XPUConfig
    llm_provider: str  # "ark" | "openai"
    log_dir: Path


def _get_env(key: str, default: str | None = None) -> str:
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"missing required env var: {key}")
    return value


def _get_env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def _get_env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def load_config() -> Config:
    llm_provider = _get_env("LLM_PROVIDER", "ark")
    if llm_provider not in ("ark", "openai"):
        raise ValueError(f"unsupported LLM provider: {llm_provider}; expected 'ark' or 'openai'")

    ark = None
    if llm_provider == "ark":
        ark = ARKConfig(
            api_key=_get_env("ARK_API_KEY"),
            base_url=_get_env("ARK_BASE_URL"),
            deployment=_get_env("ARK_DEPLOYMENT"),
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    openai = None
    if llm_provider == "openai":
        openai = OpenAIConfig(
            api_key=_get_env("OPENAI_API_KEY"),
            base_url=_get_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=_get_env("OPENAI_MODEL", "gpt-4o"),
        )
    elif openai_key:
        openai = OpenAIConfig(
            api_key=openai_key,
            base_url=_get_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=_get_env("OPENAI_MODEL", "gpt-4o"),
        )

    docker = DockerConfig(
        base_image=_get_env("DOCKER_BASE_IMAGE", "ubuntu:22.04"),
        work_dir=_get_env("DOCKER_WORK_DIR", "/workspace"),
        timeout=_get_env_int("DOCKER_TIMEOUT", 300),
    )

    xpu = XPUConfig(
        base_url=_get_env("XPU_BASE_URL", "http://localhost:8080"),
        enabled=_get_env_bool("XPU_ENABLED", False),
        disabled=_get_env_bool("XPU_DISABLED", False),
        db_dns=os.getenv("dns") or os.getenv("XPU_DB_DNS"),
        vector_enabled=_get_env_bool("XPU_VECTOR_ENABLED", False),
    )

    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(exist_ok=True)

    return Config(
        ark=ark,
        openai=openai,
        docker=docker,
        xpu=xpu,
        llm_provider=llm_provider,
        log_dir=log_dir,
    )


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config
