"""
Configuration file for Complex Systems Network Analysis (packaged)
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import json

# Load environment variables from .env file (project root)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # dotenv is optional at runtime
    pass

# --- KEYWORDS CONFIGURATION ---
settings_path = Path(__file__).parent / 'settings.json'
with open(settings_path, 'r', encoding='utf-8') as f:
    keyword_settings = json.load(f)

MAIN_KEYWORDS = keyword_settings.get('MAIN_KEYWORDS', [])
GROUP_KEYWORDS = keyword_settings.get('GROUP_KEYWORDS', [])
EXTRA_KEYWORDS = keyword_settings.get('EXTRA_KEYWORDS', [])

ANALYSIS_CONFIGS = [
    {
        "name": "main_keywords",
        "keywords": MAIN_KEYWORDS,
        "description": "Main keywords only (shared)"
    },
    {
        "name": "main_plus_our",
        "keywords": MAIN_KEYWORDS + GROUP_KEYWORDS,
        "description": "Main keywords + Our words"
    },
    {
        "name": "full_analysis",
        "keywords": MAIN_KEYWORDS + GROUP_KEYWORDS + EXTRA_KEYWORDS,
        "description": "Main keywords + Our words + Extras"
    }
]

# Legacy KEYWORDS variable for backward compatibility (full set)
KEYWORDS = MAIN_KEYWORDS + GROUP_KEYWORDS + EXTRA_KEYWORDS

# --- FILE PATHS ---
INPUT_FILE = "../exports/bluesky_posts_complex.csv"
OUTPUT_DIR = "../exports"

# --- ARCHIVE SETTINGS ---
ARCHIVE_ENABLED = True
ARCHIVE_DIR = "../exports/runs"

# --- GRAPH ANALYSIS SETTINGS ---
MIN_CO_OCCURRENCES = 1

# --- BLUESKY API SETTINGS ---
HANDLE = os.getenv("BLUESKY_HANDLE")
PASSWORD = os.getenv("BLUESKY_PASSWORD")

# --- DATA COLLECTION FILTERS ---
LOCATION_KEYWORDS = keyword_settings.get('LOCATION_KEYWORDS', [])

# --- DATE RANGE ---
DATE_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
DATE_END = datetime(2025, 11, 25, tzinfo=timezone.utc)
