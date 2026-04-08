"""Evaluation and grading utilities for Smart Hospital Orchestration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .grader import GraderReport


def grade_environment(*args, **kwargs):
	"""Lazy import wrapper to avoid side effects during module execution."""
	from .grader import grade_environment as _grade_environment

	return _grade_environment(*args, **kwargs)


def write_grader_report(*args, **kwargs):
	"""Lazy import wrapper for report persistence helper."""
	from .grader import write_grader_report as _write_grader_report

	return _write_grader_report(*args, **kwargs)


__all__ = ["grade_environment", "write_grader_report", "GraderReport"]
