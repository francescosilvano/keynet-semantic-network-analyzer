
# Analyzing Keyword Co-occurrence Networks in Social Media 

## Project Overview

The project focuses on analyzing co-occurrence networks of keywords extracted from posts on the Bluesky social media platform. It collects data using the Bluesky API, constructs a network graph based on keyword co-occurrences, computes various network metrics, and visualizes the results.

The aim is to explore the relationships between keywords and identify communities within the network, providing insights into trending topics and their interconnections.

## Quick Start

### Option 1: Using Docker (Recommended)

1. Clone the repository
2. Navigate to the project directory
3. Create a `.env` file in the root directory with your Bluesky credentials:

   ```env
   BLUESKY_HANDLE=your.handle.bsky.social
   BLUESKY_PASSWORD=your-password
   ```

4. Run with Docker Compose:

   ```bash
   docker compose up --build
   ```

   The analysis will run automatically and outputs will be saved in the `exports/` directory.

### Option 2: Local Development

1. Clone the repository
2. Navigate to the project directory
3. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   pip install -r scripts/requirements.txt
   ```

5. Create a `.env` file in the root directory (see Option 1, step 3)

6. Run the analysis:

   ```powershell
   cd scripts
   python main.py
   ```

## Project Structure

```txt
complex-systems/
├── .github/                           # GitHub workflows and CI/CD configuration
├── .dockerignore                      # Docker ignore file
├── .env                               # Environment variables (not in git)
├── .gitignore                         # Git ignore file
├── .pylintrc                          # Pylint configuration for code quality
├── compose.yaml                       # Docker Compose configuration
├── Dockerfile                         # Docker image definition
├── scripts/                           # Python scripts for analysis
│   ├── config.py                      # Configuration parameters and settings
│   ├── main.py                        # Entry point - data collection from Bluesky API
│   ├── graph.py                       # Network analysis and visualization
│   └── requirements.txt               # Python dependencies
├── docs/                              # Documentation and reports
│   ├── dashboard.qmd                  # Quarto dashboard
│   ├── index.qmd                      # Documentation index
│   └── README.md                      # Documentation README
├── exports/                           # Output directory for all generated files
│   ├── full_analysis/                 # Complete analysis with all keywords
│   ├── main_keywords/                 # Analysis with main keywords only
│   ├── main_plus_our/                 # Analysis with main + group keywords
│   └── archive/                       # Archived previous results
├── README.md                          # Project documentation
└── venv/                              # Virtual environment (not in git)
```

Each analysis subfolder in `exports/` contains:

- `bluesky_posts_complex.csv`: Collected posts from Bluesky
- `community_assignments.csv`: Community detection results
- `global_metrics.csv`: Global network metrics
- `node_metrics.csv`: Per-node metrics (degree, centrality, etc.)
- `keyword_network_edges.txt`: Edge list with weights
- `keyword_network.graphml`: Graph in GraphML format (for Gephi)
- `keyword_network.png`: Network visualization (spring layout)
- `keyword_network_circular.png`: Network visualization (circular layout)
- `network_metrics.png`: Metrics histograms
- `sentiment_distribution.png`: Sentiment analysis chart
- `grafo.xlsx`: Co-occurrence matrix spreadsheet

### Key Files

#### Configuration

**scripts/config.py** contains all configuration parameters for the co-occurrence analysis, including API credentials, keywords, and output settings. Modify these parameters to customize the analysis.

#### Entry Point

**scripts/main.py** is the application entry point. Run this file to execute the complete workflow: data collection, co-occurrence analysis, graph generation, and metrics computation.

#### Graph Operations

**scripts/graph.py** implements the network graph data structure and analysis using NetworkX. This module handles graph construction, metric calculations, and community detection.

#### Docker Configuration

**Dockerfile** defines the containerized environment for running the analysis. It uses Python 3.13-slim as the base image, sets up matplotlib configuration, and installs all dependencies.

**compose.yaml** orchestrates the Docker container, loading environment variables from `.env` and mounting the necessary directories for data persistence.

## Outputs

All generated files are saved in the `exports/` directory, organized by analysis type:

- **`exports/full_analysis/`**: Complete analysis with all 25 keywords
- **`exports/main_keywords/`**: Analysis with main shared keywords only
- **`exports/main_plus_our/`**: Analysis with main + group-specific keywords

## Example Usage: Climate Change Co-occurrence Network Analysis

The output files in the `exports/` directory are generated by running the application with climate-change related keywords.

### Analysis Workflow

**Data Collection** Posts are fetched from Bluesky API filtered by 25 climate keywords including green transition, CO2, global warming, renewable energy, loss of biodiversity, fossil fuels, and more (case-insensitive matching).

The keywords are defined in `scripts/config.py`:

```python
KEYWORDS = [
    "green transition", "greenhouse effect", "loss of biodiversity", "extreme weather events",
    "CO2", "emissions", "global warming", "melting glaciers", "renewable energy", "misinformation",
    "fossil fuels", "energy consumption", "normatives", "deforestation", "flooding",
    "tesla", "green policies", "rain", "electric vehicles",
    "natural disaster", "clean energy", "net zero", "AI", "heatwaves"
]
```

```

Locations are filtered to include only English-speaking countries to refine the analysis. These locations are specified in the `scripts/config.py` file:

```python
LOCATION_KEYWORDS = ["california", "quebec", "norway"]
```

**Co-occurrence Analysis** → For each post, the script counts which keyword pairs appear together. For example, if a post mentions both "Renewable Energy" and "Emissions," that co-occurrence is logged and weighted.

**Network Building** → Keywords become nodes; edges connect pairs that co-occur, weighted by frequency. Strong connections indicate keywords that tend to be discussed together in climate discourse.

### Key Outcomes

The analysis reveals the semantic landscape of climate discussions:

- **`keyword_network_edges.txt`**: Shows co-occurrence frequencies (e.g., "Renewable Energy" ↔ "Emissions" appearing together X times)
- **`node_metrics.csv`**: Reveals central keywords:
  - High degree = frequently co-occurring with many other terms
  - High betweenness = bridge between different climate topics
  - High closeness = central to overall discussion network
- **`community_assignments.csv`**: Groups related keywords into thematic clusters (e.g., "Fossil Fuels," "Deforestation," "Emissions" might form a cluster around environmental impact)
- **Network visualizations**: Visual maps showing how climate concepts interconnect

### Interpretation Example

If "Global Warming" has high betweenness centrality, it bridges discussions between mitigation topics (Renewable Energy, Green Policies) and impact topics (Glaciers, Extreme weather events)—making it a central connecting concept in climate conversations on social media.
