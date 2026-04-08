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
    EvaluationForm,
    Example,
    MatchingMethod,
    Question,
    ScoreMapping,
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


def load_form(path: str) -> EvaluationForm:
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
                    matching_method=MatchingMethod(q["matching_method"]),
                    keywords=q.get("keywords", []),
                    default_answer=q.get("default_answer"),
                    na_eligible=q.get("na_eligible", True),
                    critical_fail=q.get("critical_fail", False),
                    examples=examples,
                )
            )
        sections.append(
            Section(name=s["name"], questions=questions, description=s.get("description", ""))
        )

    return EvaluationForm(
        form_id=data["form_id"],
        name=data["name"],
        sections=sections,
    )


def load_scorecard(path: str, forms: list[EvaluationForm]) -> ScorecardConfig:
    with open(path) as f:
        data = json.load(f)
    return ScorecardConfig(
        scorecard_id=data["scorecard_id"],
        name=data["name"],
        forms=forms,
        score_mappings=[
            ScoreMapping(question_id=sm["question_id"], scores=sm["scores"])
            for sm in data["score_mappings"]
        ],
        critical_fail_behavior=data.get("critical_fail_behavior", "flag"),
        min_duration_seconds=data.get("min_duration_seconds", 30),
    )


def print_results(result, scorecard: ScorecardConfig):
    print("\n" + "=" * 70)
    print(f"  GRADING RESULTS — Call: {result.call_id}")
    print(f"  Scorecard: {scorecard.name}")
    print("=" * 70)

    # Critical fail banner
    if result.critical_fail_triggered:
        behavior = scorecard.critical_fail_behavior
        if behavior == "zero":
            print(
                f"\n  !! CRITICAL FAIL — Score zeroed. "
                f"Failed questions: {', '.join(result.critical_fail_questions)}"
            )
        else:
            print(
                f"\n  !! CRITICAL FAIL (flagged) — "
                f"Failed questions: {', '.join(result.critical_fail_questions)}"
            )

    # Per-question results grouped by form/section
    for form in scorecard.forms:
        print(f"\n  --- {form.name} ---")
        for section in form.sections:
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

    # Sub-scores
    print("\n" + "-" * 70)
    print("  FORM SUB-SCORES:")
    for fs in result.form_scores:
        print(f"    {fs.form_name}: {fs.points_earned}/{fs.points_possible} ({fs.percentage}%)")

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
    form_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(sample_dir, "evaluation_form.json")
    scorecard_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(sample_dir, "scorecard.json")

    print(f"Loading transcript: {transcript_path}")
    print(f"Loading form: {form_path}")
    print(f"Loading scorecard: {scorecard_path}")

    transcript = load_transcript(transcript_path)
    form = load_form(form_path)
    scorecard = load_scorecard(scorecard_path, [form])

    print(
        f"\nCall: {transcript.call_metadata.call_id} | "
        f"Agent: {transcript.call_metadata.agent_name} | "
        f"Duration: {transcript.call_metadata.duration_seconds}s | "
        f"Direction: {transcript.call_metadata.call_direction}"
    )
    print(f"Questions: {sum(len(s.questions) for s in form.sections)}")
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
        "form_scores": [
            {
                "form_id": fs.form_id,
                "form_name": fs.form_name,
                "points_earned": fs.points_earned,
                "points_possible": fs.points_possible,
                "percentage": fs.percentage,
            }
            for fs in result.form_scores
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
