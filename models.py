"""Data models for the Auto QA & Call Grading system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AnswerType(str, Enum):
    BINARY = "binary"
    LIKERT = "likert_0_2"


class SectionCategory(str, Enum):
    OPENING_GREETING = "opening_greeting"
    DISCOVERY_NEEDS_ASSESSMENT = "discovery_needs_assessment"
    PRODUCT_KNOWLEDGE_PITCH = "product_knowledge_pitch"
    TROUBLESHOOTING_RESOLUTION = "troubleshooting_resolution"
    EMPATHY_SOFT_SKILLS = "empathy_soft_skills"
    COMPLIANCE = "compliance"
    CLOSING_NEXT_STEPS = "closing_next_steps"
    CUSTOM = "custom"


class MatchingMethod(str, Enum):
    KEYWORD = "keyword"
    LLM = "llm"
    HYBRID = "hybrid"


@dataclass
class Example:
    transcript_excerpt: str
    correct_answer: str
    reasoning: str


@dataclass
class Question:
    question_id: str
    question_text: str
    answer_type: AnswerType
    answer_definitions: dict[str, str]
    scores: dict[str, int]  # e.g. {"Yes": 10, "No": 0} or {"0": 0, "1": 5, "2": 10}
    matching_method: MatchingMethod = MatchingMethod.LLM
    keywords: list[str] = field(default_factory=list)
    na_eligible: bool = True
    critical_fail: bool = False
    critical_fail_level: Optional[str] = None  # "zero_scorecard", "zero_section", or None
    examples: list[Example] = field(default_factory=list)


@dataclass
class Section:
    name: str
    questions: list[Question]
    description: str = ""
    category: str = ""


@dataclass
class ScorecardConfig:
    scorecard_id: str
    name: str
    sections: list[Section]
    min_duration_seconds: int = 30


@dataclass
class Utterance:
    speaker: str  # "agent" or "customer"
    timestamp: str
    text: str


@dataclass
class CallMetadata:
    call_id: str
    call_direction: str
    duration_seconds: int
    agent_name: str


@dataclass
class Transcript:
    call_metadata: CallMetadata
    utterances: list[Utterance]


@dataclass
class QuestionResult:
    question_id: str
    decision_stage: int
    answer: str
    confidence: int
    reasoning: str
    transcript_evidence: Optional[str]
    method: str  # "keyword_match", "llm_evaluation", "hybrid_keyword", "hybrid_llm"
    points_earned: int = 0
    points_possible: int = 0


@dataclass
class SectionScore:
    section_name: str
    points_earned: int
    points_possible: int
    percentage: float
    critical_fail_triggered: bool = False


@dataclass
class GradingResult:
    call_id: str
    scorecard_id: str
    question_results: list[QuestionResult]
    section_scores: list[SectionScore]
    cumulative_score: float
    critical_fail_triggered: bool
    critical_fail_questions: list[str]
    critical_fail_levels: list[str] = field(default_factory=list)
    final_score: float = 0.0
