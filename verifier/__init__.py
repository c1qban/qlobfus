"""Verification interfaces and placeholder pipeline implementations."""

from .models import CompileResult, TestCase, TestCaseResult, TestResult, VerificationSummary
from .pipeline import VerificationPipeline

__all__ = [
    "CompileResult",
    "TestCase",
    "TestCaseResult",
    "TestResult",
    "VerificationPipeline",
    "VerificationSummary",
]
