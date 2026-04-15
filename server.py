#!/usr/bin/env python3
"""FastAPI server that connects the grading engine to the frontend."""

import json
import os
import re
import uuid
from datetime import datetime

from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

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
from prompts import IMPORT_SYSTEM_PROMPT, AUTO_SUGGEST_SYSTEM_PROMPT

# Load env — resolve project root (works when run directly or imported from api/)
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", script_dir)
load_dotenv(os.path.join(PROJECT_ROOT, ".env.local"))

app = FastAPI(title="Auto QA & Call Grading")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample_data")

# --- In-memory storage for prototype ---
stored_scorecards: dict[str, dict] = {}
stored_results: list[dict] = []


# --- Pydantic models for API ---

class GradeRequest(BaseModel):
    transcript: dict
    scorecard: dict


class GradeWithScorecardRequest(BaseModel):
    transcript: dict
    scorecard_id: str


class AutoSuggestRequest(BaseModel):
    question_text: str
    answer_type: str
    section_category: str
    existing_answer_text: Optional[dict] = None


class ImportQuestionsRequest(BaseModel):
    file_content: list[list]
    file_name: str = ""
    sheet_name: str = "Sheet1"


# --- Helpers to convert JSON dicts to internal models ---

def parse_transcript(data: dict) -> Transcript:
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


def parse_scorecard(data: dict) -> ScorecardConfig:
    sections = []
    for s in data.get("sections", []):
        questions = []
        for q in s.get("questions", []):
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


def result_to_dict(result, scorecard) -> dict:
    """Serialize GradingResult to JSON-friendly dict."""
    # Build question metadata lookup from scorecard sections
    q_meta = {}
    for section in scorecard.sections:
        for q in section.questions:
            q_meta[q.question_id] = {
                "question_text": q.question_text,
                "answer_type": q.answer_type.value,
                "critical_fail": q.critical_fail,
                "critical_fail_level": q.critical_fail_level,
                "section": section.name,
                "section_category": section.category,
            }

    return {
        "call_id": result.call_id,
        "scorecard_id": result.scorecard_id,
        "scorecard_name": scorecard.name,
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
                "question_text": q_meta.get(qr.question_id, {}).get("question_text", ""),
                "answer_type": q_meta.get(qr.question_id, {}).get("answer_type", ""),
                "section": q_meta.get(qr.question_id, {}).get("section", ""),
                "section_category": q_meta.get(qr.question_id, {}).get("section_category", ""),
                "critical_fail_flag": q_meta.get(qr.question_id, {}).get("critical_fail", False),
                "critical_fail_level": q_meta.get(qr.question_id, {}).get("critical_fail_level"),
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


# --- Routes ---

@app.get("/")
async def serve_frontend():
    html_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    return FileResponse(html_path)


@app.get("/api/sample-data")
async def get_sample_data():
    """Return sample transcript and scorecard for the frontend to pre-fill."""
    with open(os.path.join(SAMPLE_DIR, "transcript.json")) as f:
        transcript = json.load(f)
    with open(os.path.join(SAMPLE_DIR, "scorecard.json")) as f:
        scorecard = json.load(f)
    return {"transcript": transcript, "scorecard": scorecard}


@app.get("/api/sample-transcripts")
async def get_sample_transcripts():
    """Return all sample transcripts for the frontend dropdown."""
    transcripts = []
    for fname in sorted(os.listdir(SAMPLE_DIR)):
        if fname.startswith("transcript") and fname.endswith(".json"):
            with open(os.path.join(SAMPLE_DIR, fname)) as f:
                data = json.load(f)
            label = fname.replace("transcript_", "").replace("transcript", "original").replace(".json", "").replace("_", " ").title()
            transcripts.append({"label": label, "filename": fname, "data": data})
    return transcripts


@app.get("/api/health")
async def health():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {
        "status": "ok",
        "api_key_configured": bool(api_key) and api_key != "your-api-key-here",
    }


@app.post("/api/grade")
async def grade_call(request: GradeRequest):
    """Run the grading engine on the provided transcript + scorecard."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured in .env.local")

    try:
        transcript = parse_transcript(request.transcript)
        scorecard = parse_scorecard(request.scorecard)

        engine = GradingEngine(api_key)
        result = engine.grade_call(scorecard, transcript)

        return result_to_dict(result, scorecard)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Template Questions Library ---

TEMPLATE_QUESTIONS = [
    {
        "category": "Opening & Greeting",
        "section_category": "opening_greeting",
        "questions": [
            {
                "question_text": "Did the agent greet the customer and introduce themselves by name?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {"Yes": "Agent stated their name during the greeting within the first 15 seconds of the call."},
                "scores": {"Yes": 5, "No": 0},
                "na_eligible": True,
                "critical_fail": False,
                "critical_fail_level": None,
                "examples": []
            },
            {
                "question_text": "Did the agent verify the customer's identity before discussing account details?",
                "answer_type": "binary",
                "matching_method": "hybrid",
                "keywords": ["registered email", "registered phone", "account number", "can you give me your"],
                "answer_definitions": {"Yes": "Agent asked for and received at least one identity verification element before accessing account-specific information."},
                "scores": {"Yes": 10, "No": 0},
                "na_eligible": False,
                "critical_fail": True,
                "critical_fail_level": "zero_scorecard",
                "examples": []
            },
        ]
    },
    {
        "category": "Discovery & Needs Assessment",
        "section_category": "discovery_needs_assessment",
        "questions": [
            {
                "question_text": "Did the agent let the customer fully describe their issue before responding?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {"Yes": "Agent waited for the customer to finish describing their issue before jumping in with a response or solution."},
                "scores": {"Yes": 5, "No": 0},
                "na_eligible": True,
                "critical_fail": False,
                "critical_fail_level": None,
                "examples": []
            },
            {
                "question_text": "Did the agent restate or summarize the customer's issue to confirm understanding?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {"Yes": "Agent explicitly restated, paraphrased, or summarized the customer's issue back to them."},
                "scores": {"Yes": 5, "No": 0},
                "na_eligible": True,
                "critical_fail": False,
                "critical_fail_level": None,
                "examples": []
            },
        ]
    },
    {
        "category": "Empathy & Soft Skills",
        "section_category": "empathy_soft_skills",
        "questions": [
            {
                "question_text": "Rate the agent's acknowledgement of the customer's frustration or urgency.",
                "answer_type": "likert_0_2",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {
                    "0": "Agent did not acknowledge the customer's frustration or urgency at any point.",
                    "1": "Agent acknowledged frustration but used only generic phrases without referencing the specific issue.",
                    "2": "Agent acknowledged the specific issue causing frustration and demonstrated understanding of the impact."
                },
                "scores": {"0": 0, "1": 5, "2": 10},
                "na_eligible": True,
                "critical_fail": False,
                "critical_fail_level": None,
                "examples": []
            },
            {
                "question_text": "Did the agent use the customer's name during the call?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {"Yes": "Agent addressed the customer by name at least once during the call."},
                "scores": {"Yes": 5, "No": 0},
                "na_eligible": True,
                "critical_fail": False,
                "critical_fail_level": None,
                "examples": []
            },
            {
                "question_text": "Rate the agent's overall communication clarity and professionalism.",
                "answer_type": "likert_0_2",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {
                    "0": "Agent was unclear, used excessive jargon, or was unprofessional in tone.",
                    "1": "Agent communicated adequately but could improve on clarity or tone.",
                    "2": "Agent communicated clearly, used appropriate language, and maintained a professional tone throughout."
                },
                "scores": {"0": 0, "1": 5, "2": 10},
                "na_eligible": True,
                "critical_fail": False,
                "critical_fail_level": None,
                "examples": []
            },
        ]
    },
    {
        "category": "Troubleshooting & Resolution",
        "section_category": "troubleshooting_resolution",
        "questions": [
            {
                "question_text": "Did the agent follow a logical troubleshooting sequence?",
                "answer_type": "likert_0_2",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {
                    "0": "Agent did not follow any discernible troubleshooting sequence.",
                    "1": "Agent checked some relevant factors but missed important diagnostic steps.",
                    "2": "Agent followed a clear, logical sequence and identified the root cause."
                },
                "scores": {"0": 0, "1": 5, "2": 10},
                "na_eligible": True,
                "critical_fail": False,
                "critical_fail_level": None,
                "examples": []
            },
            {
                "question_text": "Did the agent provide a clear resolution or escalation path?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {"Yes": "Agent provided either a concrete resolution or a clear escalation path with specific actions or timelines."},
                "scores": {"Yes": 10, "No": 0},
                "na_eligible": True,
                "critical_fail": False,
                "critical_fail_level": None,
                "examples": []
            },
            {
                "question_text": "Did the agent give accurate information?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {"Yes": "All factual statements made by the agent appear accurate based on the transcript context."},
                "scores": {"Yes": 5, "No": 0},
                "na_eligible": True,
                "critical_fail": True,
                "critical_fail_level": "zero_scorecard",
                "examples": []
            },
        ]
    },
    {
        "category": "Compliance",
        "section_category": "compliance",
        "questions": [
            {
                "question_text": "Did the agent read the required compliance disclaimer when applicable?",
                "answer_type": "binary",
                "matching_method": "hybrid",
                "keywords": ["for quality", "this call may be", "recorded", "monitoring purposes"],
                "answer_definitions": {"Yes": "Agent read or referenced the required compliance disclaimer during the call."},
                "scores": {"Yes": 10, "No": 0},
                "na_eligible": True,
                "critical_fail": False,
                "critical_fail_level": None,
                "examples": []
            },
            {
                "question_text": "Did the agent follow the refund/cancellation policy correctly?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {"Yes": "Agent followed the correct policy for refunds or cancellations as described in the transcript."},
                "scores": {"Yes": 10, "No": 0},
                "na_eligible": True,
                "critical_fail": True,
                "critical_fail_level": "zero_section",
                "examples": []
            },
        ]
    },
    {
        "category": "Closing & Next Steps",
        "section_category": "closing_next_steps",
        "questions": [
            {
                "question_text": "Did the agent ask if the customer had any other questions before closing?",
                "answer_type": "binary",
                "matching_method": "hybrid",
                "keywords": ["any other questions", "anything else", "any other issues"],
                "answer_definitions": {"Yes": "Agent explicitly asked whether the customer had additional questions or issues."},
                "scores": {"Yes": 5, "No": 0},
                "na_eligible": True,
                "critical_fail": False,
                "critical_fail_level": None,
                "examples": []
            },
            {
                "question_text": "Did the agent provide a reference number or summary of what was agreed upon?",
                "answer_type": "binary",
                "matching_method": "hybrid",
                "keywords": ["reference number", "ticket", "TKT-", "case number"],
                "answer_definitions": {"Yes": "Agent provided at least one of: a reference/ticket number, or a verbal summary of the agreed-upon actions."},
                "scores": {"Yes": 5, "No": 0},
                "na_eligible": True,
                "critical_fail": False,
                "critical_fail_level": None,
                "examples": []
            },
        ]
    }
]


@app.get("/api/templates")
async def get_templates():
    """Return the template question library."""
    return TEMPLATE_QUESTIONS


# --- Scorecards CRUD ---

@app.get("/api/scorecards")
async def list_scorecards():
    return list(stored_scorecards.values())


@app.get("/api/scorecards/{scorecard_id}")
async def get_scorecard(scorecard_id: str):
    if scorecard_id not in stored_scorecards:
        raise HTTPException(status_code=404, detail="Scorecard not found")
    return stored_scorecards[scorecard_id]


@app.post("/api/scorecards")
async def save_scorecard(scorecard: dict):
    if not scorecard.get("scorecard_id"):
        scorecard["scorecard_id"] = f"sc_{uuid.uuid4().hex[:8]}"
    # Auto-generate question IDs
    q_counter = 1
    for section in scorecard.get("sections", []):
        for question in section.get("questions", []):
            if not question.get("question_id"):
                question["question_id"] = f"q_{q_counter:03d}"
                q_counter += 1
            else:
                if question["question_id"].startswith("q_"):
                    try:
                        q_counter = max(q_counter, int(question["question_id"].split("_")[1]) + 1)
                    except (ValueError, IndexError):
                        pass
    stored_scorecards[scorecard["scorecard_id"]] = scorecard
    return scorecard


@app.delete("/api/scorecards/{scorecard_id}")
async def delete_scorecard(scorecard_id: str):
    if scorecard_id not in stored_scorecards:
        raise HTTPException(status_code=404, detail="Scorecard not found")
    del stored_scorecards[scorecard_id]
    return {"status": "deleted"}


# --- Grade with stored scorecard ---

@app.post("/api/grade-with-scorecard")
async def grade_with_scorecard(request: GradeWithScorecardRequest):
    """Grade a transcript using a stored scorecard."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured in .env.local")

    sc_data = stored_scorecards.get(request.scorecard_id)
    if not sc_data:
        raise HTTPException(status_code=404, detail="Scorecard not found")

    try:
        transcript = parse_transcript(request.transcript)
        scorecard = parse_scorecard(sc_data)
        engine = GradingEngine(api_key)
        result = engine.grade_call(scorecard, transcript)
        result_dict = result_to_dict(result, scorecard)

        # Store the result
        result_dict["graded_at"] = datetime.utcnow().isoformat()
        result_dict["transcript_data"] = request.transcript
        result_id = f"res_{uuid.uuid4().hex[:8]}"
        result_dict["result_id"] = result_id
        stored_results.append(result_dict)

        return result_dict
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Results / Dashboard ---

@app.get("/api/results")
async def list_results():
    """Return all grading results for the dashboard."""
    summary = []
    for r in stored_results:
        summary.append({
            "result_id": r["result_id"],
            "call_id": r["call_id"],
            "scorecard_id": r["scorecard_id"],
            "scorecard_name": r["scorecard_name"],
            "final_score": r["final_score"],
            "cumulative_score": r["cumulative_score"],
            "critical_fail_triggered": r["critical_fail_triggered"],
            "graded_at": r.get("graded_at", ""),
            "agent_name": r.get("transcript_data", {}).get("call_metadata", {}).get("agent_name", "Unknown"),
            "call_direction": r.get("transcript_data", {}).get("call_metadata", {}).get("call_direction", ""),
            "duration_seconds": r.get("transcript_data", {}).get("call_metadata", {}).get("duration_seconds", 0),
            "questions_count": len(r.get("question_results", [])),
            "questions_graded": len([q for q in r.get("question_results", []) if q["answer"] != "N/A"]),
        })
    return summary


@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    for r in stored_results:
        if r["result_id"] == result_id:
            return r
    raise HTTPException(status_code=404, detail="Result not found")


# --- Import Questions (AI-powered) ---

@app.post("/api/import-questions")
async def import_questions(request: ImportQuestionsRequest):
    """Parse uploaded spreadsheet content and extract evaluation questions using LLM."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured")

    if not request.file_content or len(request.file_content) == 0:
        raise HTTPException(status_code=400, detail="File content is empty")

    client = anthropic.Anthropic(api_key=api_key)

    user_message = json.dumps({
        "file_content": request.file_content,
        "file_name": request.file_name,
        "sheet_name": request.sheet_name,
    })

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=IMPORT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text.strip()

        # Extract JSON
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise HTTPException(status_code=500, detail="Could not parse AI response")

        parsed = json.loads(json_match.group())
        questions = parsed.get("questions", [])

        # Enforce 20 question max
        if len(questions) > 20:
            questions = questions[:20]
            parsed["questions"] = questions
            parsed["truncated"] = True
            parsed["truncation_message"] = f"Only the first 20 questions were imported — scorecards have a 20-question maximum."

        return parsed
    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")
    except json.JSONDecodeError:
        # Retry once
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=IMPORT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise HTTPException(status_code=500, detail="Could not process this file. Try simplifying your spreadsheet.")
            return json.loads(json_match.group())
        except Exception:
            raise HTTPException(status_code=500, detail="Could not process this file. Try simplifying your spreadsheet and re-upload.")



# --- Auto-Suggest (Refine with AI) ---

@app.post("/api/auto-suggest")
async def auto_suggest(request: AutoSuggestRequest):
    """Generate refined question text, answer definitions, and examples using LLM."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=api_key)

    user_message = json.dumps({
        "question_text": request.question_text,
        "answer_type": request.answer_type,
        "section_category": request.section_category,
        "existing_answer_text": request.existing_answer_text,
    })

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=AUTO_SUGGEST_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text.strip()

        # Extract JSON
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise HTTPException(status_code=500, detail="Could not generate suggestions. Try again or write definitions manually.")

        return json.loads(json_match.group())
    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")
    except json.JSONDecodeError:
        # Retry once
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=AUTO_SUGGEST_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise HTTPException(status_code=500, detail="Could not generate suggestions. Try again.")
            return json.loads(json_match.group())
        except Exception:
            raise HTTPException(status_code=500, detail="Could not generate suggestions. Try again or write definitions manually.")


# --- Seed sample data on startup ---

@app.on_event("startup")
async def seed_sample_data():
    """Load sample scorecards into storage on startup."""
    try:
        with open(os.path.join(SAMPLE_DIR, "scorecard.json")) as f:
            sample_sc = json.load(f)
        stored_scorecards[sample_sc["scorecard_id"]] = sample_sc
    except Exception as e:
        print(f"Warning: could not seed support sample data: {e}")

    try:
        with open(os.path.join(SAMPLE_DIR, "scorecard_sales.json")) as f:
            sales_sc = json.load(f)
        stored_scorecards[sales_sc["scorecard_id"]] = sales_sc
    except Exception as e:
        print(f"Warning: could not seed sales sample data: {e}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Auto QA Grading server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
