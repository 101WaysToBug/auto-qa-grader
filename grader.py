"""Core grading engine: keyword matching, LLM routing, and score calculation."""

import json
import re
from typing import Optional

import anthropic

from models import (
    AnswerType,
    CallMetadata,
    GradingResult,
    MatchingMethod,
    Question,
    QuestionResult,
    ScorecardConfig,
    SectionScore,
    Transcript,
    Utterance,
)
from prompts import BINARY_SYSTEM_PROMPT, LIKERT_SYSTEM_PROMPT


class KeywordMatcher:
    """Handles keyword/phrase matching against agent speech in the transcript."""

    @staticmethod
    def match(question: Question, transcript: Transcript) -> Optional[QuestionResult]:
        """Run keyword match against agent utterances only. Returns result if matched, None if not."""
        agent_text = " ".join(
            u.text for u in transcript.utterances if u.speaker == "agent"
        ).lower()

        for keyword in question.keywords:
            if keyword.lower() in agent_text:
                # Find the timestamp of the match
                evidence = KeywordMatcher._find_evidence_timestamp(
                    keyword, transcript
                )
                return QuestionResult(
                    question_id=question.question_id,
                    decision_stage=2,
                    answer="Yes",
                    confidence=100,
                    reasoning=f'Keyword matched: "{keyword}" found in agent speech.',
                    transcript_evidence=evidence,
                    method="keyword_match",
                )
        return None

    @staticmethod
    def _find_evidence_timestamp(keyword: str, transcript: Transcript) -> str:
        for u in transcript.utterances:
            if u.speaker == "agent" and keyword.lower() in u.text.lower():
                return u.timestamp
        return "unknown"


class LLMGrader:
    """Handles LLM-based grading via Anthropic API."""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def grade_batch(
        self,
        questions: list[Question],
        transcript: Transcript,
        question_type: AnswerType,
    ) -> list[QuestionResult]:
        if not questions:
            return []

        system_prompt = (
            BINARY_SYSTEM_PROMPT
            if question_type == AnswerType.BINARY
            else LIKERT_SYSTEM_PROMPT
        )
        model = (
            "claude-haiku-4-5-20251001"
            if question_type == AnswerType.BINARY
            else "claude-sonnet-4-20250514"
        )

        # Build the input payload per spec
        questions_payload = []
        for q in questions:
            q_obj = {
                "question_id": q.question_id,
                "question_text": q.question_text,
                "answer_type": q.answer_type.value,
                "answer_definitions": q.answer_definitions,
                "na_eligible": q.na_eligible,
                "examples": [
                    {
                        "transcript_excerpt": ex.transcript_excerpt,
                        "correct_answer": ex.correct_answer,
                        "reasoning": ex.reasoning,
                    }
                    for ex in q.examples
                ],
            }
            questions_payload.append(q_obj)

        transcript_payload = {
            "call_metadata": {
                "call_id": transcript.call_metadata.call_id,
                "call_direction": transcript.call_metadata.call_direction,
                "duration_seconds": transcript.call_metadata.duration_seconds,
                "agent_name": transcript.call_metadata.agent_name,
            },
            "transcript": [
                {
                    "speaker": u.speaker,
                    "timestamp": u.timestamp,
                    "text": u.text,
                }
                for u in transcript.utterances
            ],
        }

        user_message = json.dumps(
            {"questions": questions_payload, "transcript": transcript_payload},
            indent=2,
        )

        # Call the API
        print(f"  [>] Calling model={model} for {question_type.value} questions")
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return self._parse_response(response, questions)
        except Exception as e:
            print(f"  [!] LLM call failed (model={model}): {e}")
            # Retry once
            try:
                print("  [!] Retrying...")
                response = self.client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                return self._parse_response(response, questions)
            except Exception as e2:
                print(f"  [!] Retry failed: {e2}")
                return [
                    QuestionResult(
                        question_id=q.question_id,
                        decision_stage=3,
                        answer="N/A",
                        confidence=0,
                        reasoning="LLM call failed after retry.",
                        transcript_evidence=None,
                        method="llm_evaluation",
                    )
                    for q in questions
                ]

    def _parse_response(
        self, response, questions: list[Question]
    ) -> list[QuestionResult]:
        raw = response.content[0].text.strip()

        # Extract JSON from possible markdown fencing
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            print(f"  [!] Could not parse JSON from LLM response")
            return [
                QuestionResult(
                    question_id=q.question_id,
                    decision_stage=3,
                    answer="N/A",
                    confidence=0,
                    reasoning="LLM output parsing failed.",
                    transcript_evidence=None,
                    method="llm_evaluation",
                )
                for q in questions
            ]

        parsed = json.loads(json_match.group())
        results = []
        parsed_by_id = {item["question_id"]: item for item in parsed}

        for q in questions:
            if q.question_id in parsed_by_id:
                item = parsed_by_id[q.question_id]
                results.append(
                    QuestionResult(
                        question_id=item["question_id"],
                        decision_stage=item.get("decision_stage", 2),
                        answer=str(item["answer"]),
                        confidence=item.get("confidence", 50),
                        reasoning=item.get("reasoning", ""),
                        transcript_evidence=item.get("transcript_evidence"),
                        method="llm_evaluation",
                    )
                )
            else:
                results.append(
                    QuestionResult(
                        question_id=q.question_id,
                        decision_stage=3,
                        answer="N/A",
                        confidence=0,
                        reasoning="Question ID not found in LLM response.",
                        transcript_evidence=None,
                        method="llm_evaluation",
                    )
                )

        return results


class GradingEngine:
    """Orchestrates the full grading pipeline per the spec's execution flow."""

    def __init__(self, api_key: str):
        self.keyword_matcher = KeywordMatcher()
        self.llm_grader = LLMGrader(api_key)

    def grade_call(
        self, scorecard: ScorecardConfig, transcript: Transcript
    ) -> GradingResult:
        # Check minimum duration
        if transcript.call_metadata.duration_seconds < scorecard.min_duration_seconds:
            return self._empty_result(
                transcript.call_metadata.call_id,
                scorecard,
                reason="Call too short for evaluation.",
            )

        # Check transcript has enough content
        total_words = sum(len(u.text.split()) for u in transcript.utterances)
        if total_words < 50:
            return self._empty_result(
                transcript.call_metadata.call_id,
                scorecard,
                reason="Transcript insufficient for evaluation (under 50 words).",
            )

        all_results: list[QuestionResult] = []
        binary_llm_questions = []
        likert_questions = []

        for section in scorecard.sections:
            for question in section.questions:
                result = None

                # Phase 1: Route by matching method
                if question.matching_method == MatchingMethod.KEYWORD:
                    result = self.keyword_matcher.match(question, transcript)
                    if result is None:
                        # Keyword didn't match — answer is No
                        result = QuestionResult(
                            question_id=question.question_id,
                            decision_stage=2,
                            answer="No",
                            confidence=100,
                            reasoning="No matching keywords found in agent speech.",
                            transcript_evidence=None,
                            method="keyword_match",
                        )

                elif question.matching_method == MatchingMethod.HYBRID:
                    result = self.keyword_matcher.match(question, transcript)
                    if result:
                        result.method = "hybrid_keyword"
                    else:
                        # Keyword didn't match — fall through to LLM
                        binary_llm_questions.append(question)

                elif question.matching_method == MatchingMethod.LLM:
                    if question.answer_type == AnswerType.BINARY:
                        binary_llm_questions.append(question)
                    else:
                        likert_questions.append(question)

                if result:
                    all_results.append(result)

        # Phase 2: Batch LLM calls (max 10 per call)
        print(f"  Grading scorecard: {scorecard.name}")

        if binary_llm_questions:
            print(
                f"    -> {len(binary_llm_questions)} binary questions -> Haiku"
            )
            for i in range(0, len(binary_llm_questions), 10):
                batch = binary_llm_questions[i : i + 10]
                llm_results = self.llm_grader.grade_batch(
                    batch, transcript, AnswerType.BINARY
                )
                # Tag hybrid fallback results
                for r in llm_results:
                    q = next(
                        (q for q in batch if q.question_id == r.question_id), None
                    )
                    if q and q.matching_method == MatchingMethod.HYBRID:
                        r.method = "hybrid_llm"
                all_results.extend(llm_results)

        if likert_questions:
            print(
                f"    -> {len(likert_questions)} likert questions -> Sonnet"
            )
            for i in range(0, len(likert_questions), 10):
                batch = likert_questions[i : i + 10]
                llm_results = self.llm_grader.grade_batch(
                    batch, transcript, AnswerType.LIKERT
                )
                all_results.extend(llm_results)

        # Phase 5: Scoring
        return self._calculate_scores(
            transcript.call_metadata.call_id, scorecard, all_results
        )

    def _calculate_scores(
        self,
        call_id: str,
        scorecard: ScorecardConfig,
        results: list[QuestionResult],
    ) -> GradingResult:
        # Build lookup: question_id -> Question object and section name
        question_lookup: dict[str, Question] = {}
        question_to_section: dict[str, str] = {}
        for section in scorecard.sections:
            for q in section.questions:
                question_lookup[q.question_id] = q
                question_to_section[q.question_id] = section.name

        # Map points to each result using scores from the Question itself
        for r in results:
            q = question_lookup.get(r.question_id)
            if q and r.answer != "N/A":
                r.points_earned = q.scores.get(r.answer, 0)
                r.points_possible = max(q.scores.values()) if q.scores else 0
            elif r.answer == "N/A":
                r.points_earned = 0
                r.points_possible = 0

        # Critical fail check
        critical_fails = []
        critical_fail_levels = []
        for r in results:
            q = question_lookup.get(r.question_id)
            if q and q.critical_fail and r.answer != "N/A":
                # Binary: "No" is a fail. Likert: "0" is a fail.
                if r.answer in ("No", "0"):
                    critical_fails.append(r.question_id)
                    critical_fail_levels.append(q.critical_fail_level or "zero_scorecard")

        critical_triggered = len(critical_fails) > 0

        # Determine which sections are zeroed by critical fails
        zeroed_sections: set[str] = set()
        zero_scorecard = False
        for qid, level in zip(critical_fails, critical_fail_levels):
            if level == "zero_scorecard":
                zero_scorecard = True
            elif level == "zero_section":
                section_name = question_to_section.get(qid)
                if section_name:
                    zeroed_sections.add(section_name)

        # Per-section scores
        section_scores = []
        for section in scorecard.sections:
            sec_q_ids = {q.question_id for q in section.questions}
            sec_results = [r for r in results if r.question_id in sec_q_ids]
            scored = [r for r in sec_results if r.answer != "N/A"]
            earned = sum(r.points_earned for r in scored)
            possible = sum(r.points_possible for r in scored)
            pct = (earned / possible * 100) if possible > 0 else 0.0

            sec_critical_fail = section.name in zeroed_sections
            if sec_critical_fail or zero_scorecard:
                pct = 0.0
                earned = 0

            section_scores.append(
                SectionScore(
                    section_name=section.name,
                    points_earned=earned,
                    points_possible=possible,
                    percentage=round(pct, 1),
                    critical_fail_triggered=sec_critical_fail,
                )
            )

        # Cumulative score
        all_scored = [r for r in results if r.answer != "N/A"]
        total_earned = sum(r.points_earned for r in all_scored)
        total_possible = sum(r.points_possible for r in all_scored)
        cumulative = (total_earned / total_possible * 100) if total_possible > 0 else 0.0

        # Apply critical fail behavior to final score
        final_score = cumulative
        if zero_scorecard:
            final_score = 0.0
        elif zeroed_sections:
            # Recalculate final score with zeroed sections
            adjusted_earned = total_earned
            for section_name in zeroed_sections:
                for r in all_scored:
                    if question_to_section.get(r.question_id) == section_name:
                        adjusted_earned -= r.points_earned
            final_score = (adjusted_earned / total_possible * 100) if total_possible > 0 else 0.0

        return GradingResult(
            call_id=call_id,
            scorecard_id=scorecard.scorecard_id,
            question_results=results,
            section_scores=section_scores,
            cumulative_score=round(cumulative, 1),
            critical_fail_triggered=critical_triggered,
            critical_fail_questions=critical_fails,
            critical_fail_levels=critical_fail_levels,
            final_score=round(final_score, 1),
        )

    def _empty_result(
        self, call_id: str, scorecard: ScorecardConfig, reason: str
    ) -> GradingResult:
        results = []
        for section in scorecard.sections:
            for q in section.questions:
                results.append(
                    QuestionResult(
                        question_id=q.question_id,
                        decision_stage=3,
                        answer="N/A",
                        confidence=0,
                        reasoning=reason,
                        transcript_evidence=None,
                        method="skipped",
                    )
                )
        return GradingResult(
            call_id=call_id,
            scorecard_id=scorecard.scorecard_id,
            question_results=results,
            section_scores=[],
            cumulative_score=0.0,
            critical_fail_triggered=False,
            critical_fail_questions=[],
            final_score=0.0,
        )
