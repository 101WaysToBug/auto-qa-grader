# Auto QA & Call Grading

AI-powered quality assurance grading for contact center call transcripts. Built with FastAPI + Claude API, this tool evaluates agent performance against configurable evaluation forms and scorecards.

## How It Works

1. **Create Evaluation Forms** — Define questions to grade agents on (binary Yes/No or Likert 0-2 scale). Pull from a built-in template library or write custom questions.
2. **Build Scorecards** — Select forms, assign point values to each question, and configure critical fail behavior.
3. **Grade Transcripts** — Submit a call transcript and the system automatically grades it using a multi-stage pipeline:
   - **Keyword matching** for fast, deterministic checks
   - **Claude Haiku** for binary (Yes/No) questions
   - **Claude Sonnet** for nuanced Likert-scale questions
4. **Review Results** — Dashboard shows all graded calls with scores, critical fails, and per-question AI reasoning with transcript evidence.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────--┐
│   Frontend   │────▶│  FastAPI      │────▶│ Grading Engine  │
│  (SPA)       │◀────│  Server       │◀────│                 │
└─────────────┘     └──────────────┘      │  ┌─────────────┐ │
                                          │  │ Keyword     │ │
                                          │  │ Matcher     │ │
                                          │  ├─────────────┤ │
                                          │  │ LLM Grader  │ │
                                          │  │ (Claude API)│ │
                                          │  └─────────────┘ │
                                          └─────────────────-|
```

### Three-Stage Decision Logic (per question)

1. **Relevance Check** — Is this question applicable to the call? If not, mark N/A.
2. **Evidence Evaluation** — Find transcript evidence and grade against answer definitions.
3. **Insufficient Evidence Fallback** — Use default answer or mark N/A if evidence is lacking.

## Setup

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

### Option 2: Local Python

```bash
# Clone the repo
git clone https://github.com/101WaysToBug/auto-qa-grader.git
cd auto-qa-grader

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.local.example .env.local
# Edit .env.local and add your ANTHROPIC_API_KEY

# Start the server
python server.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (required) |

## Project Structure

```
├── server.py           # FastAPI server with all API endpoints
├── grader.py           # Core grading engine (keyword + LLM pipeline)
├── models.py           # Data models (forms, scorecards, transcripts, results)
├── prompts.py          # System prompts for Claude (binary + Likert)
├── run.py              # CLI runner (alternative to server)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image definition
├── docker-compose.yml  # One-command Docker setup
├── .dockerignore       # Files excluded from Docker build
├── .env.local          # API key (not committed)
├── frontend/
│   └── index.html      # Single-file SPA frontend
└── sample_data/
    ├── evaluation_form.json
    ├── scorecard.json
    ├── transcript.json           # Original sample (good call)
    ├── transcript_excellent.json # High-scoring example
    ├── transcript_average.json   # Mid-scoring example
    └── transcript_poor.json      # Low-scoring example
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check + API key status |
| `GET` | `/api/templates` | Template question library |
| `GET` | `/api/forms` | List saved evaluation forms |
| `POST` | `/api/forms` | Create/update a form |
| `DELETE` | `/api/forms/{id}` | Delete a form |
| `GET` | `/api/scorecards` | List saved scorecards |
| `POST` | `/api/scorecards` | Create/update a scorecard |
| `DELETE` | `/api/scorecards/{id}` | Delete a scorecard |
| `POST` | `/api/grade` | Grade with inline form + scorecard |
| `POST` | `/api/grade-with-scorecard` | Grade using a stored scorecard |
| `GET` | `/api/results` | Dashboard summary of all results |
| `GET` | `/api/results/{id}` | Full grading detail |
| `GET` | `/api/sample-transcripts` | All sample transcripts |

## Sample Transcripts

The repo includes 4 sample transcripts for testing:

| File | Agent | Scenario | Expected Score |
|---|---|---|---|
| `transcript.json` | Priya Sharma | Broadcast messages failing (quality rating) | Good |
| `transcript_excellent.json` | Sarah Chen | Template paused, proactive workaround | High |
| `transcript_average.json` | James Okoro | Chatbot trigger misconfigured | Medium |
| `transcript_poor.json` | Kevin Marks | Account locked, dismissive handling | Low |

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **AI**: Anthropic Claude API (Haiku for binary, Sonnet for Likert)
- **Frontend**: Vanilla HTML/CSS/JS (single-file SPA, no build step)
- **Containerization**: Docker + Docker Compose
- **Storage**: In-memory (prototype)
