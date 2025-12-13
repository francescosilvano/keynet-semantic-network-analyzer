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
    "Energy Transition", "Greenhouse Effect", "Biodiversity", "Extreme weather events",
    "CO2", "Emissions", "Global Warming", "Glaciers", "Renewable Energy", "Fake News"
]

# Group 4 keywords (Our words)
OUR_KEYWORDS = [
    "Ecosystem", "Fossil Fuels", "Energy Consumption", "Normatives", "Deforestation",
    "Flooding", "Tesla", "Green Policies", "Rain", "Electric Vehicles"
]

# Extra keywords
EXTRA_KEYWORDS = [
    "Natural Disaster", "Clean Energy", "Net Zero", "Tesla", "Heatwaves"
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

# --- ARCHIVE SETTINGS ---
# Enable/disable archiving functionality
ARCHIVE_ENABLED = True

# Base directory for archived runs (UUID-based folders)
ARCHIVE_DIR = "../exports/runs"

# Current run configuration (initialized at runtime by archive module)
CURRENT_RUN_UUID = None
CURRENT_RUN_DIR = None

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
