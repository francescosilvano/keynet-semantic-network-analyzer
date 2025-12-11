"""
Configuration file for Complex Systems Network Analysis
Shared settings between main.py (data collection) and graph.py (network analysis)
"""

from datetime import datetime, timezone
import os

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

# --- GRAPH ANALYSIS SETTINGS ---
MIN_CO_OCCURRENCES = 1  # Minimum co-occurrences to create an edge in the network

# --- BLUESKY API SETTINGS ---
# Note: Keep your credentials secure. Consider using environment variables in production.
HANDLE = os.getenv("BLUESKY_HANDLE")
PASSWORD = os.getenv("BLUESKY_PASSWORD")

# --- DATA COLLECTION FILTERS ---
# --- LOCATION KEYWORDS (User Input) ---
def get_location_keywords():
    """Prompt user for location keywords via terminal input."""
    user_input = input(
        "Enter location keywords (comma-separated) [default: california, quebec, norway]: "
    ).strip()
    
    if not user_input:
        return ["california", "quebec", "norway"]
    
    return [keyword.strip() for keyword in user_input.split(",")]

LOCATION_KEYWORDS = get_location_keywords()

# --- DATE RANGE ---
DATE_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
DATE_END = datetime(2025, 11, 25, tzinfo=timezone.utc)
