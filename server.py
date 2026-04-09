#!/usr/bin/env python3
"""FastAPI server that connects the grading engine to the frontend."""

import json
import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
stored_forms: dict[str, dict] = {}
stored_scorecards: dict[str, dict] = {}
stored_results: list[dict] = []


# --- Pydantic models for API ---

class GradeRequest(BaseModel):
    transcript: dict
    evaluation_form: dict
    scorecard: dict


class GradeWithScorecardRequest(BaseModel):
    transcript: dict
    scorecard_id: str


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


def parse_form(data: dict) -> EvaluationForm:
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
    return EvaluationForm(form_id=data["form_id"], name=data["name"], sections=sections)


def parse_scorecard(data: dict, forms: list[EvaluationForm]) -> ScorecardConfig:
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


def result_to_dict(result, scorecard) -> dict:
    """Serialize GradingResult to JSON-friendly dict."""
    # Build question metadata lookup from scorecard forms
    q_meta = {}
    for form in scorecard.forms:
        for section in form.sections:
            for q in section.questions:
                q_meta[q.question_id] = {
                    "question_text": q.question_text,
                    "answer_type": q.answer_type.value,
                    "critical_fail": q.critical_fail,
                    "section": section.name,
                    "form_name": form.name,
                }

    return {
        "call_id": result.call_id,
        "scorecard_id": result.scorecard_id,
        "scorecard_name": scorecard.name,
        "cumulative_score": result.cumulative_score,
        "final_score": result.final_score,
        "critical_fail_triggered": result.critical_fail_triggered,
        "critical_fail_behavior": scorecard.critical_fail_behavior,
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
                "question_text": q_meta.get(qr.question_id, {}).get("question_text", ""),
                "answer_type": q_meta.get(qr.question_id, {}).get("answer_type", ""),
                "section": q_meta.get(qr.question_id, {}).get("section", ""),
                "form_name": q_meta.get(qr.question_id, {}).get("form_name", ""),
                "critical_fail_flag": q_meta.get(qr.question_id, {}).get("critical_fail", False),
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
    """Return sample transcript, form, and scorecard for the frontend to pre-fill."""
    with open(os.path.join(SAMPLE_DIR, "transcript.json")) as f:
        transcript = json.load(f)
    with open(os.path.join(SAMPLE_DIR, "evaluation_form.json")) as f:
        form = json.load(f)
    with open(os.path.join(SAMPLE_DIR, "scorecard.json")) as f:
        scorecard = json.load(f)
    return {"transcript": transcript, "evaluation_form": form, "scorecard": scorecard}


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
    """Run the grading engine on the provided transcript + form + scorecard."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured in .env.local")

    try:
        transcript = parse_transcript(request.transcript)
        form = parse_form(request.evaluation_form)
        scorecard = parse_scorecard(request.scorecard, [form])

        engine = GradingEngine(api_key)
        result = engine.grade_call(scorecard, transcript)

        return result_to_dict(result, scorecard)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Template Questions Library ---

TEMPLATE_QUESTIONS = [
    {
        "category": "Opening & Verification",
        "questions": [
            {
                "question_text": "Did the agent greet the customer and introduce themselves by name?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {
                    "Yes": "Agent stated their name during the greeting within the first 15 seconds of the call.",
                    "No": "Agent did not state their name, or greeted without a personal introduction."
                },
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": False,
                "examples": []
            },
            {
                "question_text": "Did the agent verify the customer's identity before discussing account details?",
                "answer_type": "binary",
                "matching_method": "hybrid",
                "keywords": ["registered email", "registered phone", "account number", "can you give me your"],
                "answer_definitions": {
                    "Yes": "Agent asked for and received at least one identity verification element before accessing account-specific information.",
                    "No": "Agent discussed account details without first verifying the customer's identity."
                },
                "default_answer": "No",
                "na_eligible": False,
                "critical_fail": True,
                "examples": []
            },
        ]
    },
    {
        "category": "Issue Understanding",
        "questions": [
            {
                "question_text": "Did the agent let the customer fully describe their issue before responding?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {
                    "Yes": "Agent waited for the customer to finish describing their issue before jumping in with a response or solution.",
                    "No": "Agent interrupted the customer while they were describing the issue."
                },
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": False,
                "examples": []
            },
            {
                "question_text": "Did the agent restate or summarize the customer's issue to confirm understanding?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {
                    "Yes": "Agent explicitly restated, paraphrased, or summarized the customer's issue back to them.",
                    "No": "Agent moved directly to troubleshooting without confirming their understanding."
                },
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": False,
                "examples": []
            },
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
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": False,
                "examples": []
            },
        ]
    },
    {
        "category": "Troubleshooting & Resolution",
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
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": False,
                "examples": []
            },
            {
                "question_text": "Did the agent provide a clear resolution or escalation path?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {
                    "Yes": "Agent provided either a concrete resolution or a clear escalation path with specific actions or timelines.",
                    "No": "Agent ended the call without providing clear next steps."
                },
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": False,
                "examples": []
            },
            {
                "question_text": "Did the agent give accurate information?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {
                    "Yes": "All factual statements made by the agent appear accurate based on the transcript context.",
                    "No": "Agent made at least one factually incorrect statement."
                },
                "default_answer": "Yes",
                "na_eligible": True,
                "critical_fail": True,
                "examples": []
            },
        ]
    },
    {
        "category": "Closing",
        "questions": [
            {
                "question_text": "Did the agent ask if the customer had any other questions before closing?",
                "answer_type": "binary",
                "matching_method": "hybrid",
                "keywords": ["any other questions", "anything else", "any other issues"],
                "answer_definitions": {
                    "Yes": "Agent explicitly asked whether the customer had additional questions or issues.",
                    "No": "Agent closed the call without checking if the customer had additional needs."
                },
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": False,
                "examples": []
            },
            {
                "question_text": "Did the agent provide a reference number or summary of what was agreed upon?",
                "answer_type": "binary",
                "matching_method": "hybrid",
                "keywords": ["reference number", "ticket", "TKT-", "case number"],
                "answer_definitions": {
                    "Yes": "Agent provided at least one of: a reference/ticket number, or a verbal summary of the agreed-upon actions.",
                    "No": "Agent ended the call without providing a reference number or summarizing what was agreed."
                },
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": False,
                "examples": []
            },
        ]
    },
    {
        "category": "Compliance & Policy",
        "questions": [
            {
                "question_text": "Did the agent read the required compliance disclaimer when applicable?",
                "answer_type": "binary",
                "matching_method": "hybrid",
                "keywords": ["for quality", "this call may be", "recorded", "monitoring purposes"],
                "answer_definitions": {
                    "Yes": "Agent read or referenced the required compliance disclaimer during the call.",
                    "No": "Agent did not read or reference any compliance disclaimer."
                },
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": False,
                "examples": []
            },
            {
                "question_text": "Did the agent follow the refund/cancellation policy correctly?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {
                    "Yes": "Agent followed the correct policy for refunds or cancellations as described in the transcript.",
                    "No": "Agent deviated from the expected refund/cancellation process."
                },
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": True,
                "examples": []
            },
        ]
    },
    {
        "category": "Communication Quality",
        "questions": [
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
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": False,
                "examples": []
            },
            {
                "question_text": "Did the agent use the customer's name during the call?",
                "answer_type": "binary",
                "matching_method": "llm",
                "keywords": [],
                "answer_definitions": {
                    "Yes": "Agent addressed the customer by name at least once during the call.",
                    "No": "Agent never used the customer's name."
                },
                "default_answer": None,
                "na_eligible": True,
                "critical_fail": False,
                "examples": []
            },
        ]
    }
]


@app.get("/api/templates")
async def get_templates():
    """Return the template question library."""
    return TEMPLATE_QUESTIONS


# --- Forms CRUD ---

@app.get("/api/forms")
async def list_forms():
    """List all saved evaluation forms."""
    return list(stored_forms.values())


@app.get("/api/forms/{form_id}")
async def get_form(form_id: str):
    if form_id not in stored_forms:
        raise HTTPException(status_code=404, detail="Form not found")
    return stored_forms[form_id]


@app.post("/api/forms")
async def save_form(form: dict):
    """Save an evaluation form. Auto-generates form_id if not provided."""
    if not form.get("form_id"):
        form["form_id"] = f"form_{uuid.uuid4().hex[:8]}"
    # Auto-generate question IDs for questions that don't have them
    q_counter = 1
    for section in form.get("sections", []):
        for question in section.get("questions", []):
            if not question.get("question_id"):
                question["question_id"] = f"q_{q_counter:03d}"
                q_counter += 1
            else:
                q_counter = max(q_counter, int(question["question_id"].split("_")[1]) + 1) if question["question_id"].startswith("q_") else q_counter
    stored_forms[form["form_id"]] = form
    return form


@app.delete("/api/forms/{form_id}")
async def delete_form(form_id: str):
    if form_id not in stored_forms:
        raise HTTPException(status_code=404, detail="Form not found")
    del stored_forms[form_id]
    return {"status": "deleted"}


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
    """Grade a transcript using a stored scorecard (which references stored forms)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured in .env.local")

    sc_data = stored_scorecards.get(request.scorecard_id)
    if not sc_data:
        raise HTTPException(status_code=404, detail="Scorecard not found")

    try:
        transcript = parse_transcript(request.transcript)

        # Resolve forms from stored forms
        forms = []
        for form_id in sc_data.get("form_ids", []):
            form_data = stored_forms.get(form_id)
            if not form_data:
                raise HTTPException(status_code=404, detail=f"Form {form_id} not found")
            forms.append(parse_form(form_data))

        scorecard = parse_scorecard(sc_data, forms)
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
    """Return all grading results for the dashboard (without full transcript data)."""
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


# --- Seed sample data on startup ---

@app.on_event("startup")
async def seed_sample_data():
    """Load sample evaluation forms and scorecards into storage on startup."""
    try:
        with open(os.path.join(SAMPLE_DIR, "evaluation_form.json")) as f:
            sample_form = json.load(f)
        stored_forms[sample_form["form_id"]] = sample_form

        with open(os.path.join(SAMPLE_DIR, "scorecard.json")) as f:
            sample_sc = json.load(f)
        sample_sc["form_ids"] = [sample_form["form_id"]]
        stored_scorecards[sample_sc["scorecard_id"]] = sample_sc
    except Exception as e:
        print(f"Warning: could not seed support sample data: {e}")

    try:
        with open(os.path.join(SAMPLE_DIR, "evaluation_form_sales.json")) as f:
            sales_form = json.load(f)
        stored_forms[sales_form["form_id"]] = sales_form

        with open(os.path.join(SAMPLE_DIR, "scorecard_sales.json")) as f:
            sales_sc = json.load(f)
        sales_sc["form_ids"] = [sales_form["form_id"]]
        stored_scorecards[sales_sc["scorecard_id"]] = sales_sc
    except Exception as e:
        print(f"Warning: could not seed sales sample data: {e}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Auto QA Grading server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
