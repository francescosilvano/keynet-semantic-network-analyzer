"""
Configuration file for Complex Systems Network Analysis
Shared settings between main.py (data collection) and graph.py (network analysis)
"""

from datetime import datetime, timezone

# --- KEYWORDS CONFIGURATION ---
KEYWORDS = [
    "Energy Transition", "Greenhouse Effect", "Biodiversity", "Extreme weather events",
    "CO2", "Emissions", "Global Warming", "Glaciers", "Renewable Energy", "Fake News",
    "Ecosystem", "Fossil Fuels", "Energy Consumption", "Normatives", "Deforestation",
    "Floodings", "Heatwaves", "Green Policies", "Rain", "Electric Vehicles",
    "Natural Disaster", "Clean Energy", "Net Zero", "Tesla", "AI"
]

# --- FILE PATHS ---
INPUT_FILE = "exports/bluesky_posts_complex.csv"
OUTPUT_DIR = "exports"

# --- GRAPH ANALYSIS SETTINGS ---
MIN_CO_OCCURRENCES = 1  # Minimum co-occurrences to create an edge in the network

# --- BLUESKY API SETTINGS ---
# Note: Keep your credentials secure. Consider using environment variables in production.
HANDLE = "provauni.bsky.social"
PASSWORD = "s.sabatino"

# --- DATA COLLECTION FILTERS ---
LOCATION_KEYWORDS = ["california", "quebec", "norway"]

# --- DATE RANGE ---
DATE_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
DATE_END = datetime(2025, 11, 25, tzinfo=timezone.utc)
