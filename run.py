#!/usr/bin/env python3
"""CLI runner for the Auto QA & Call Grading prototype."""

import json
import os
import sys

from dotenv import load_dotenv

from grader import GradingEngine
from models import (
    AnswerType,
    CallMetadata,
    Example,
    MatchingMethod,
    Question,
    ScorecardConfig,
    Section,
    Transcript,
    Utterance,
)


def load_transcript(path: str) -> Transcript:
    with open(path) as f:
        data = json.load(f)
    meta = data["call_metadata"]
    return Transcript(
        call_metadata=CallMetadata(
            call_id=meta["call_id"],
            call_direction=meta["call_direction"],
            duration_seconds=meta["duration_seconds"],
            agent_name=meta["agent_name"],
        ),
        utterances=[
            Utterance(speaker=u["speaker"], timestamp=u["timestamp"], text=u["text"])
            for u in data["transcript"]
        ],
    )


def load_scorecard(path: str) -> ScorecardConfig:
    with open(path) as f:
        data = json.load(f)

    sections = []
    for s in data["sections"]:
        questions = []
        for q in s["questions"]:
            examples = [
                Example(
                    transcript_excerpt=ex["transcript_excerpt"],
                    correct_answer=ex["correct_answer"],
                    reasoning=ex["reasoning"],
                )
                for ex in q.get("examples", [])
            ]
            questions.append(
                Question(
                    question_id=q["question_id"],
                    question_text=q["question_text"],
                    answer_type=AnswerType(q["answer_type"]),
                    answer_definitions=q["answer_definitions"],
                    scores=q.get("scores", {}),
                    matching_method=MatchingMethod(q.get("matching_method", "llm")),
                    keywords=q.get("keywords", []),
                    na_eligible=q.get("na_eligible", True),
                    critical_fail=q.get("critical_fail", False),
                    critical_fail_level=q.get("critical_fail_level"),
                    examples=examples,
                )
            )
        sections.append(
            Section(
                name=s["name"],
                questions=questions,
                description=s.get("description", ""),
                category=s.get("category", ""),
            )
        )

    return ScorecardConfig(
        scorecard_id=data["scorecard_id"],
        name=data["name"],
        sections=sections,
        min_duration_seconds=data.get("min_duration_seconds", 30),
    )


def print_results(result, scorecard: ScorecardConfig):
    print("\n" + "=" * 70)
    print(f"  GRADING RESULTS — Call: {result.call_id}")
    print(f"  Scorecard: {scorecard.name}")
    print("=" * 70)

    # Critical fail banner
    if result.critical_fail_triggered:
        levels = result.critical_fail_levels
        has_scorecard_zero = "zero_scorecard" in levels
        if has_scorecard_zero:
            print(
                f"\n  !! CRITICAL FAIL — Entire scorecard zeroed. "
                f"Failed questions: {', '.join(result.critical_fail_questions)}"
            )
        else:
            print(
                f"\n  !! CRITICAL FAIL — Section(s) zeroed. "
                f"Failed questions: {', '.join(result.critical_fail_questions)}"
            )

    # Per-question results grouped by section
    for section in scorecard.sections:
        print(f"\n  [{section.name}]")
        for q in section.questions:
            r = next(
                (r for r in result.question_results if r.question_id == q.question_id),
                None,
            )
            if not r:
                continue

            critical_marker = " [CRITICAL]" if q.critical_fail else ""
            score_str = (
                f"{r.points_earned}/{r.points_possible}"
                if r.answer != "N/A"
                else "N/A"
            )

            print(f"\n    Q: {q.question_text}")
            print(
                f"    Answer: {r.answer}  |  Score: {score_str}  |  "
                f"Confidence: {r.confidence}%  |  Method: {r.method}{critical_marker}"
            )
            print(f"    Reasoning: {r.reasoning}")
            if r.transcript_evidence:
                print(f"    Evidence: {r.transcript_evidence}")

    # Section sub-scores
    print("\n" + "-" * 70)
    print("  SECTION SCORES:")
    for ss in result.section_scores:
        crit = " [ZEROED]" if ss.critical_fail_triggered else ""
        print(f"    {ss.section_name}: {ss.points_earned}/{ss.points_possible} ({ss.percentage}%){crit}")

    # Final score
    print(f"\n  CUMULATIVE SCORE: {result.cumulative_score}%")
    if result.critical_fail_triggered:
        print(f"  FINAL SCORE (after critical fail): {result.final_score}%")
    print("=" * 70)


def main():
    # Load API key
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, ".env.local")
    load_dotenv(env_path)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("Error: Set your ANTHROPIC_API_KEY in .env.local")
        sys.exit(1)

    # Default paths or CLI args
    sample_dir = os.path.join(script_dir, "sample_data")
    transcript_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(sample_dir, "transcript.json")
    scorecard_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(sample_dir, "scorecard.json")

    print(f"Loading transcript: {transcript_path}")
    print(f"Loading scorecard: {scorecard_path}")

    transcript = load_transcript(transcript_path)
    scorecard = load_scorecard(scorecard_path)

    total_q = sum(len(s.questions) for s in scorecard.sections)
    print(
        f"\nCall: {transcript.call_metadata.call_id} | "
        f"Agent: {transcript.call_metadata.agent_name} | "
        f"Duration: {transcript.call_metadata.duration_seconds}s | "
        f"Direction: {transcript.call_metadata.call_direction}"
    )
    print(f"Questions: {total_q}")
    print(f"\nGrading...\n")

    engine = GradingEngine(api_key)
    result = engine.grade_call(scorecard, transcript)

    print_results(result, scorecard)

    # Save raw results to JSON
    output_path = os.path.join(script_dir, "grading_output.json")
    output_data = {
        "call_id": result.call_id,
        "scorecard_id": result.scorecard_id,
        "cumulative_score": result.cumulative_score,
        "final_score": result.final_score,
        "critical_fail_triggered": result.critical_fail_triggered,
        "critical_fail_questions": result.critical_fail_questions,
        "critical_fail_levels": result.critical_fail_levels,
        "section_scores": [
            {
                "section_name": ss.section_name,
                "points_earned": ss.points_earned,
                "points_possible": ss.points_possible,
                "percentage": ss.percentage,
                "critical_fail_triggered": ss.critical_fail_triggered,
            }
            for ss in result.section_scores
        ],
        "question_results": [
            {
                "question_id": qr.question_id,
                "decision_stage": qr.decision_stage,
                "answer": qr.answer,
                "confidence": qr.confidence,
                "reasoning": qr.reasoning,
                "transcript_evidence": qr.transcript_evidence,
                "method": qr.method,
                "points_earned": qr.points_earned,
                "points_possible": qr.points_possible,
            }
            for qr in result.question_results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nRaw results saved to: {output_path}")


if __name__ == "__main__":
    main()
