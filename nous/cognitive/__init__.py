"""Cognitive layer — The Nous Loop.

Orchestrates Brain (decisions) and Heart (memory) into a
thinking loop: Sense -> Frame -> Recall -> Deliberate -> Act -> Monitor -> Learn.
"""

from nous.cognitive.context import ContextEngine
from nous.cognitive.dedup import ConversationDeduplicator, DeduplicationResult
from nous.cognitive.deliberation import DeliberationEngine
from nous.cognitive.frames import FrameEngine
from nous.cognitive.intent import (
    IntentClassifier,
    IntentSignals,
    RetrievalPlan,
    RetrievalQuery,
)
from nous.cognitive.layer import CognitiveLayer
from nous.cognitive.monitor import MonitorEngine
from nous.cognitive.schemas import (
    Assessment,
    BuildResult,
    ContextBudget,
    ContextSection,
    FrameSelection,
    FrameType,
    ToolResult,
    TurnContext,
    TurnResult,
)
from nous.cognitive.usage_tracker import MemoryUsageStats, UsageRecord, UsageTracker

__all__ = [
    "CognitiveLayer",
    "ContextEngine",
    "ConversationDeduplicator",
    "DeduplicationResult",
    "DeliberationEngine",
    "FrameEngine",
    "IntentClassifier",
    "IntentSignals",
    "MonitorEngine",
    "Assessment",
    "BuildResult",
    "ContextBudget",
    "ContextSection",
    "FrameSelection",
    "FrameType",
    "MemoryUsageStats",
    "RetrievalPlan",
    "RetrievalQuery",
    "ToolResult",
    "TurnContext",
    "TurnResult",
    "UsageRecord",
    "UsageTracker",
]
