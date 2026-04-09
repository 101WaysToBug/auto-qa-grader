"""Vercel serverless entry point — re-exports the FastAPI app from server.py."""

import os
import sys

# Set PROJECT_ROOT so server.py resolves sample_data/ and frontend/ correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PROJECT_ROOT", project_root)

# Add project root to Python path so imports (grader, models, prompts) work
sys.path.insert(0, project_root)

from server import app  # noqa: E402
