"""
Configuration file for Complex Systems Network Analysis
Shared settings between main.py (data collection) and graph.py (network analysis)
"""

from datetime import datetime, timezone
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in parent directory (project root)
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables must be set manually.")

# --- KEYWORDS CONFIGURATION ---
# Main keywords (shared by entire class)
MAIN_KEYWORDS = [
    "green transition", "greenhouse effect", "loss of biodiversity", "extreme weather events",
    "CO2", "emissions", "global warming", "melting glaciers", "renewable energy", "misinformation"
]

# Group 4 keywords (Our words)
OUR_KEYWORDS = [
    "fossil fuels", "energy consumption", "normatives", "deforestation", "flooding",
    "tesla", "green policies", "rain", "electric vehicles"
]

# Extra keywords
EXTRA_KEYWORDS = [
    "natural disaster", "clean energy", "net zero", "AI", "heatwaves"
]

# Three analysis configurations
ANALYSIS_CONFIGS = [
    {
        "name": "main_keywords",
        "keywords": MAIN_KEYWORDS,
        "description": "Main keywords only (shared)"
    },
    {
        "name": "main_plus_our",
        "keywords": MAIN_KEYWORDS + OUR_KEYWORDS,
        "description": "Main keywords + Our words"
    },
    {
        "name": "full_analysis",
        "keywords": MAIN_KEYWORDS + OUR_KEYWORDS + EXTRA_KEYWORDS,
        "description": "Main keywords + Our words + Extras"
    }
]

# Legacy KEYWORDS variable for backward compatibility (full set)
KEYWORDS = MAIN_KEYWORDS + OUR_KEYWORDS + EXTRA_KEYWORDS

# --- FILE PATHS ---
# Paths relative to the scripts directory, pointing to project root exports folder
INPUT_FILE = "../exports/bluesky_posts_complex.csv"
OUTPUT_DIR = "../exports"

# --- GRAPH ANALYSIS SETTINGS ---
MIN_CO_OCCURRENCES = 1  # Minimum co-occurrences to create an edge in the network

# --- BLUESKY API SETTINGS ---
# Note: Keep your credentials secure. Consider using environment variables in production.
HANDLE = os.getenv("BLUESKY_HANDLE")
PASSWORD = os.getenv("BLUESKY_PASSWORD")

# --- DATA COLLECTION FILTERS ---
LOCATION_KEYWORDS = ["california", "quebec", "norway"]

# --- DATE RANGE ---
DATE_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
DATE_END = datetime(2025, 11, 25, tzinfo=timezone.utc)
