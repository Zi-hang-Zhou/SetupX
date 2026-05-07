"""Data structures for actions, results, and pipeline state."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(Enum):
    SHELL_COMMAND = "SHELL_COMMAND"
    TRY_XPU_SUGGESTION = "TRY_XPU_SUGGESTION"
    SET_ENV = "SET_ENV"
    ROLLBACK_ENV = "ROLLBACK_ENV"
    VERIFY = "VERIFY"
    FINISH = "FINISH"


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "truncated": self.truncated,
        }

    def __str__(self) -> str:
        status = "ok" if self.success else f"fail(exit={self.exit_code})"
        output = self.stdout or self.stderr
        if self.truncated:
            output += "\n... [truncated]"
        return f"[{status}] {self.command}\n{output}"


@dataclass
class XPUSuggestion:
    id: str
    description: str
    commands: list[str]
    confidence: float
    source: str = "mock"
    atoms: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "commands": self.commands,
            "confidence": self.confidence,
            "source": self.source,
        }

    def __str__(self) -> str:
        return f"[ID: {self.id}] {self.description} (confidence: {self.confidence:.2f})"


@dataclass
class AttributionReport:
    suggestion_id: str
    timestamp: float
    repo_context: str
    outcome: str              # "SUCCESS" | "FAIL" | "PARTIAL"
    error_before: str
    error_after: str
    score: float              # 1.0 (resolved) -> 0.0 (no effect) -> -1.0 (new error)
    logs: list[CommandResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "suggestion_id": self.suggestion_id,
            "timestamp": self.timestamp,
            "repo_context": self.repo_context,
            "outcome": self.outcome,
            "error_before": self.error_before,
            "error_after": self.error_after,
            "score": self.score,
            "logs": [log.to_dict() for log in self.logs],
        }

    def __str__(self) -> str:
        return (
            f"[AttributionReport] suggestion_id={self.suggestion_id}, "
            f"outcome={self.outcome}, score={self.score}"
        )


@dataclass
class AgentAction:
    action_type: ActionType
    thought: str = ""
    command: str | None = None
    xpu_suggestion_id: str | None = None
    reasoning: str | None = None
    env_key: str | None = None
    env_value: str | None = None
    message: str | None = None
    verify_hint: str | None = None
    rollback_n_frames: int = 1

    def to_dict(self) -> dict:
        result = {
            "thought": self.thought,
            "action_type": self.action_type.value,
            "content": {},
        }
        if self.action_type == ActionType.SHELL_COMMAND:
            result["content"]["command"] = self.command
        elif self.action_type == ActionType.TRY_XPU_SUGGESTION:
            result["content"]["xpu_suggestion_id"] = self.xpu_suggestion_id
            result["content"]["command"] = self.command
            result["content"]["reasoning"] = self.reasoning
        elif self.action_type == ActionType.SET_ENV:
            result["content"]["env_key"] = self.env_key
            result["content"]["env_value"] = self.env_value
        elif self.action_type == ActionType.VERIFY:
            if self.verify_hint:
                result["content"]["hint"] = self.verify_hint
        elif self.action_type == ActionType.ROLLBACK_ENV:
            result["content"]["n_frames"] = self.rollback_n_frames
        elif self.action_type == ActionType.FINISH:
            result["content"]["message"] = self.message
        return result

    def __str__(self) -> str:
        if self.action_type == ActionType.SHELL_COMMAND:
            return f"[shell] {self.command}"
        elif self.action_type == ActionType.TRY_XPU_SUGGESTION:
            return f"[try_xpu] id={self.xpu_suggestion_id}, reason: {self.reasoning}"
        elif self.action_type == ActionType.SET_ENV:
            return f"[set_env] {self.env_key}={self.env_value}"
        elif self.action_type == ActionType.ROLLBACK_ENV:
            return f"[rollback x{self.rollback_n_frames}] {self.thought}"
        elif self.action_type == ActionType.VERIFY:
            return f"[verify] {self.thought}"
        elif self.action_type == ActionType.FINISH:
            return f"[finish] {self.message}"
        return f"[{self.action_type.value}]"


@dataclass
class AgentState:
    repo_url: str
    container_id: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    step: int = 0
    max_steps: int = 50
    completed: bool = False
    final_message: str | None = None
    last_error: str | None = None
    tried_suggestions: set[str] = field(default_factory=set)

    def add_to_history(self, entry: dict[str, Any]) -> None:
        self.history.append({
            "step": self.step,
            "timestamp": time.time(),
            **entry,
        })

    def get_recent_history(self, n: int = 10) -> list[dict[str, Any]]:
        return self.history[-n:]

    def record_tried_suggestion(self, suggestion_id: str) -> None:
        self.tried_suggestions.add(suggestion_id)

    def is_suggestion_tried(self, suggestion_id: str) -> bool:
        return suggestion_id in self.tried_suggestions


@dataclass
class SetupResult:
    repo_url: str
    container_id: str
    completed: bool
    steps_taken: int
    final_message: str
    history: list[dict] = field(default_factory=list)
    last_verify_messages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "repo_url": self.repo_url,
            "container_id": self.container_id,
            "completed": self.completed,
            "steps_taken": self.steps_taken,
            "final_message": self.final_message,
            "history": self.history,
            "last_verify_messages": self.last_verify_messages,
        }


@dataclass
class VerifyResult:
    success: bool
    test_framework: str
    collect_count: int
    command: str
    exit_code: int
    stdout: str
    stderr: str
    messages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "test_framework": self.test_framework,
            "collect_count": self.collect_count,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass
class ProsecutionResult:
    prosecute: bool
    charges: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)


@dataclass
class Phase2Result:
    success: bool
    reason: str
    prosecution: "ProsecutionResult | None" = None
    judge_reasoning: str = ""
