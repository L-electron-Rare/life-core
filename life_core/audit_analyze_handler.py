"""Handler for POST /audit/analyze — delegates to AuditAnalyzer."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuditAnalyzerUnavailableError(RuntimeError):
    """Raised when the optional makelife analyzer dependency is unavailable."""


class AuditAnalysisExecutionError(RuntimeError):
    """Raised when the analyzer fails to complete the LLM call."""


def _load_audit_analyzer():
    """Load the optional audit analyzer lazily."""
    try:
        from makelife.audit_analyzer import AnalysisError, AuditAnalyzer
    except ImportError as exc:
        raise AuditAnalyzerUnavailableError(
            "makelife package not found. Install it: cd makelife && uv pip install -e ."
        ) from exc
    return AuditAnalyzer, AnalysisError


class AuditAnalyzeRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the audit file to analyse")
    cross_paths: list[str] = Field(
        default_factory=list,
        description="Optional additional file paths for cross-file analysis",
    )
    model: str = Field(
        default="claude-3-5-haiku-latest",
        description="LLM model to use for analysis",
    )


class AuditAnalyzeResponse(BaseModel):
    issues: list[dict[str, Any]]
    summary: str
    mode: str  # "single" or "cross"


def handle_audit_analyze(request: AuditAnalyzeRequest) -> AuditAnalyzeResponse:
    """Perform LLM analysis on one or more audit files.

    Raises:
        FileNotFoundError: if file_path or any cross_paths file does not exist.
        AnalysisError: if the LLM call fails.
    """
    AuditAnalyzer, AnalysisError = _load_audit_analyzer()
    analyzer = AuditAnalyzer(model=request.model)

    try:
        if request.cross_paths:
            all_paths = [request.file_path] + request.cross_paths
            result = analyzer.analyze_cross(all_paths)
            mode = "cross"
        else:
            result = analyzer.analyze_single(request.file_path)
            mode = "single"
    except FileNotFoundError:
        raise
    except AnalysisError as exc:
        raise AuditAnalysisExecutionError(str(exc)) from exc

    return AuditAnalyzeResponse(
        issues=result.get("issues", []),
        summary=result.get("summary", ""),
        mode=mode,
    )
