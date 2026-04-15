# Auto QA & Call Grading

AI-powered quality assurance grading for contact center call transcripts. Built with FastAPI + Claude API, this tool evaluates agent performance against configurable scorecards with sections, questions, and scoring — all in one unified builder.

**Live Demo**: [https://autoqagrader.vercel.app](https://autoqagrader.vercel.app)

## How It Works

1. **Build Scorecards** — Create scorecards with sections and questions directly. Pull from a template library, import from CSV (AI-powered), or write custom questions. Assign point values inline, configure critical fail behavior, and set matching methods per question.
2. **Refine with AI** — Use the "AI Refine" button on any question to auto-generate precise answer definitions and few-shot examples using Claude Sonnet.
3. **Grade Transcripts** — Submit a call transcript and the system automatically grades it using a multi-stage pipeline:
   - **Keyword matching** for fast, deterministic compliance checks
   - **Claude Haiku** for binary (Yes/No) questions
   - **Claude Sonnet** for nuanced Likert-scale (0-2) questions
4. **Review Results** — Dashboard shows all graded calls with scores, section breakdowns, critical fails, and per-question AI reasoning with transcript evidence.

## Key Features

- **Unified Scorecard Builder** — Sections, questions, answer definitions, scoring, and critical fail config in one place. No separate forms to manage.
- **AI-Powered CSV Import** — Upload a spreadsheet-based QA rubric and AI extracts questions, detects answer types (Binary/Likert), and assigns section categories automatically.
- **Auto-Suggest (Refine with AI)** — Per-question AI refinement generates precise answer definitions and two-shot examples optimized for consistent automated grading.
- **Three Matching Methods** — Keyword (deterministic), LLM (contextual), or Hybrid (keyword first, LLM fallback) per question.
- **Critical Fail Behavior** — Per-question granularity: zero the entire scorecard or zero just the section.
- **Template Library** — Pre-built questions across 6 categories (Opening, Discovery, Empathy, Troubleshooting, Compliance, Closing).
- **Duplicate Scorecards** — One-click duplication for iterating on scorecard variants.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│   Frontend   │────>│  FastAPI      │────>│ Grading Engine    │
│  (SPA)       │<────│  Server       │<────│                   │
└─────────────┘     └──────────────┘     │  ┌─────────────┐  │
                                          │  │ Keyword     │  │
                                          │  │ Matcher     │  │
                                          │  ├─────────────┤  │
                                          │  │ LLM Grader  │  │
                                          │  │ (Claude API)│  │
                                          │  └─────────────┘  │
                                          └───────────────────┘
```

### Three-Stage Decision Logic (per question)

1. **Relevance Check** — Is this question applicable to the call? If not, mark N/A.
2. **Evidence Evaluation** — Find transcript evidence and grade against answer definitions.
3. **Insufficient Evidence Fallback** — Mark N/A if na_eligible, otherwise provide low-confidence answer.

### LLM Prompt Architecture

Three distinct LLM use cases, each with dedicated prompts:

| Use Case | Model | Purpose |
|---|---|---|
| **Call Grading** | Haiku (Binary) / Sonnet (Likert) | Evaluate transcript against scorecard questions |
| **CSV Import** | Sonnet | Parse spreadsheet, extract questions, categorize |
| **Auto-Suggest** | Sonnet | Generate answer definitions and few-shot examples |

## Deploy Locally

### Prerequisites

- An [Anthropic API key](https://console.anthropic.com/)
- **Docker** (recommended) or Python 3.10+

### Option 1: Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/101WaysToBug/auto-qa-grader.git
cd auto-qa-grader

# Configure your API key
cp .env.local.example .env.local
# Edit .env.local and add your ANTHROPIC_API_KEY

# Build and run
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Option 2: Local Python

```bash
# Clone the repo
git clone https://github.com/101WaysToBug/auto-qa-grader.git
cd auto-qa-grader

# Create a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.local.example .env.local
# Edit .env.local and add your ANTHROPIC_API_KEY

# Start the server
python server.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Deploy to Vercel

The app is configured for Vercel's Python serverless runtime. The frontend is served as a static file and the FastAPI backend runs as a serverless function.

### Prerequisites

- [Vercel CLI](https://vercel.com/docs/cli) installed (`npm i -g vercel`)
- A Vercel account (`vercel login`)

### Steps

```bash
# 1. Clone and enter the repo
git clone https://github.com/101WaysToBug/auto-qa-grader.git
cd auto-qa-grader

# 2. Link to a Vercel project (creates one if it doesn't exist)
vercel link

# 3. Add your Anthropic API key as a secure environment variable
echo "YOUR_API_KEY" | vercel env add ANTHROPIC_API_KEY production --sensitive

# 4. Deploy to production
vercel --prod
```

Your app will be live at the URL Vercel prints (e.g. `https://your-project.vercel.app`).

### How it works

| File | Purpose |
|---|---|
| `vercel.json` | Routes `/api/*` to the Python serverless function, everything else to the static frontend |
| `api/index.py` | Vercel entry point — imports the FastAPI `app` from `server.py` |
| `frontend/index.html` | Static SPA served directly by Vercel's CDN |

### Updating after code changes

```bash
# Deploy a preview (gets a unique URL for testing)
vercel

# Promote to production when ready
vercel --prod
```

### Environment Variables

| Variable | Description | Where to set |
|---|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (required) | `.env.local` (local) or `vercel env add` (Vercel) |

## Project Structure

```
├── server.py                   # FastAPI server with all API endpoints
├── grader.py                   # Core grading engine (keyword + LLM pipeline)
├── models.py                   # Data models (scorecards, sections, questions, results)
├── prompts.py                  # System prompts (grading, import, auto-suggest)
├── run.py                      # CLI runner (alternative to server)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container image definition
├── docker-compose.yml          # One-command Docker setup
├── vercel.json                 # Vercel deployment config
├── api/
│   └── index.py                # Vercel serverless entry point
├── frontend/
│   └── index.html              # Single-file SPA frontend
└── sample_data/
    ├── scorecard.json                # Support scorecard (sections + questions + scores)
    ├── scorecard_sales.json          # Sales scorecard
    ├── transcript.json               # Support — good call
    ├── transcript_excellent.json     # Support — high-scoring
    ├── transcript_average.json       # Support — mid-scoring
    ├── transcript_poor.json          # Support — low-scoring
    ├── transcript_critical_fail.json # Support — critical fail
    ├── transcript_sales_excellent.json # Sales — high-scoring
    ├── transcript_sales_average.json   # Sales — mid-scoring
    └── transcript_sales_poor.json      # Sales — low-scoring
```

## Sample Data

The repo ships with two scorecard templates and sample transcripts for each.

### Support Scorecard

| File | Agent | Scenario | Expected Score |
|---|---|---|---|
| `transcript.json` | Priya Sharma | Broadcast messages failing (quality rating) | Good |
| `transcript_excellent.json` | Sarah Chen | Template paused, proactive workaround | High |
| `transcript_average.json` | James Okoro | Chatbot trigger misconfigured | Medium |
| `transcript_poor.json` | Kevin Marks | Account locked, dismissive handling | Low |
| `transcript_critical_fail.json` | Derek Powell | Double charge, no verification, inaccurate info | Critical Fail |

### Sales Scorecard

| File | Agent | Scenario | Expected Score |
|---|---|---|---|
| `transcript_sales_excellent.json` | Meera Kapoor | D2C prospect, deep discovery, demo booked | High |
| `transcript_sales_average.json` | Tom Bradley | Trial follow-up, decent but no firm commitment | Medium |
| `transcript_sales_poor.json` | Ryan Mitchell | Cold call, feature dump, pressure tactics | Low |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check + API key status |
| `GET` | `/api/templates` | Template question library |
| `GET` | `/api/scorecards` | List saved scorecards |
| `POST` | `/api/scorecards` | Create/update a scorecard |
| `DELETE` | `/api/scorecards/{id}` | Delete a scorecard |
| `POST` | `/api/grade` | Grade with inline scorecard |
| `POST` | `/api/grade-with-scorecard` | Grade using a stored scorecard |
| `GET` | `/api/results` | Dashboard summary of all results |
| `GET` | `/api/results/{id}` | Full grading detail |
| `GET` | `/api/sample-transcripts` | All sample transcripts |
| `POST` | `/api/import-questions` | AI-powered question extraction from CSV content |
| `POST` | `/api/auto-suggest` | AI-powered question refinement + answer definition generation |

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **AI**: Anthropic Claude API (Haiku for binary grading, Sonnet for Likert grading + import + auto-suggest)
- **Frontend**: Vanilla HTML/CSS/JS (single-file SPA, no build step)
- **Deployment**: Vercel (serverless Python + static CDN)
- **Containerization**: Docker + Docker Compose
- **Storage**: In-memory (prototype)
