"""Brain module -- Decision intelligence organ for Nous agents.

Public API:
    Brain           - Main class with record/query/check/review/calibration
    EmbeddingProvider - OpenAI embedding generation
    Settings        - Configuration (re-exported from nous.config)

Schemas:
    RecordInput, ReviewInput, DecisionDetail, DecisionSummary,
    GuardrailResult, CalibrationReport, GraphEdgeInfo, BridgeInfo,
    ThoughtInfo, ReasonInput
"""

from nous.brain.brain import Brain
from nous.brain.embeddings import EmbeddingProvider
from nous.brain.schemas import (
    BridgeInfo,
    CalibrationReport,
    DecisionDetail,
    DecisionSummary,
    GraphEdgeInfo,
    GuardrailResult,
    ReasonInput,
    RecordInput,
    ReviewInput,
    ThoughtInfo,
)

__all__ = [
    "Brain",
    "EmbeddingProvider",
    "BridgeInfo",
    "CalibrationReport",
    "DecisionDetail",
    "DecisionSummary",
    "GraphEdgeInfo",
    "GuardrailResult",
    "ReasonInput",
    "RecordInput",
    "ReviewInput",
    "ThoughtInfo",
]
