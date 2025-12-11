
# Complex Systems Project

## Project Overview

The project focuses on analyzing co-occurrence networks of keywords extracted from posts on the Bluesky social media platform. It collects data using the Bluesky API, constructs a network graph based on keyword co-occurrences, computes various network metrics, and visualizes the results.

The aim is to explore the relationships between keywords and identify communities within the network, providing insights into trending topics and their interconnections.


## Quick Start

1. Clone the repository
2. Navigate to the project directory
3. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

5. Run the application:

   ```powershell
   python main.py
   ```

6. View the interactive dashboard:

   ```powershell
   cd docs
   quarto render dashboard.qmd
   # Open dashboard.html in your browser
   ```

## Project Structure

```txt
complex-systems/
├── .github/                           # GitHub workflows and CI/CD configuration
├── .pylintrc                          # Pylint configuration for code quality
├── docs/                              # Interactive dashboard documentation
│   ├── dashboard.qmd                   # Quarto dashboard source
│   └── README.md                       # Dashboard documentation
├── scripts/                           # Analysis scripts
│   ├── config.py                       # Configuration parameters and settings
│   ├── main.py                         # Entry point - data collection from Bluesky API
│   ├── graph.py                        # Network analysis and visualization
│   └── requirements.txt                # Python dependencies
├── exports/                           # Output directory for all generated files
│   ├── main_keywords/                  # Analysis with 10 main keywords
│   ├── main_plus_our/                  # Analysis with 20 keywords
│   └── full_analysis/                  # Complete analysis with 25 keywords
│       ├── bluesky_posts_complex.csv   # Collected posts from Bluesky
│       ├── community_assignments.csv   # Community detection results
│       ├── global_metrics.csv          # Global network metrics
│       ├── node_metrics.csv            # Per-node metrics (degree, centrality, etc.)
│       ├── keyword_network_edges.txt   # Edge list with weights
│       ├── keyword_network.graphml     # Graph in GraphML format (for Gephi)
│       ├── keyword_network.png         # Network visualization (spring layout)
│       ├── keyword_network_circular.png # Network visualization (circular layout)
│       ├── network_metrics.png         # Metrics histograms
│       ├── sentiment_distribution.png  # Sentiment analysis chart
│       └── grafo.xlsx                  # Co-occurrence matrix spreadsheet
├── README.md                          # Project documentation
└── venv/                              # Virtual environment (not in git)
```

### Key Files

#### Configuration

**config.py** contains all configuration parameters for the co-occurrence analysis, including API credentials, keywords, and output settings. Modify these parameters to customize the analysis.

#### Entry Point

**main.py** is the application entry point. Run this file to execute the complete workflow: data collection, co-occurrence analysis, graph generation, and metrics computation.

#### Graph Operations

**graph.py** implements the network graph data structure and analysis using NetworkX. This module handles graph construction, metric calculations, and community detection.

#### Interactive Dashboard

**docs/dashboard.qmd** provides an interactive Quarto-based dashboard for visualizing all analysis results. The dashboard features:

- **Tab-based navigation** to switch between three analyses (10, 20, and 25 keywords)
- **Large, clear visualizations** including sentiment charts and network graphs
- **Interactive layout toggles** between spring and circular network layouts
- **Statistical tables** showing key metrics and top keywords
- **Self-contained HTML output** for easy sharing and offline viewing

To render the dashboard:
```bash
cd docs
quarto render dashboard.qmd
```

See `docs/README.md` for detailed dashboard documentation.

## Outputs

All generated files are saved in the `exports/` directory, organized into three analysis subdirectories:

### Analysis Configurations

1. **main_keywords/** - Analysis with 10 main keywords (shared by entire class)
   - Energy Transition, Greenhouse Effect, Biodiversity, Extreme weather events, CO2, Emissions, Global Warming, Glaciers, Renewable Energy, Fake News

2. **main_plus_our/** - Analysis with 20 keywords (main + Group 4 keywords)
   - Adds: Ecosystem, Fossil Fuels, Energy Consumption, Normatives, Deforestation, Flooding, Tesla, Green Policies, Rain, Electric Vehicles

3. **full_analysis/** - Complete analysis with 25 keywords (all keywords)
   - Adds: Natural Disaster, Clean Energy, Net Zero, Tesla, Heatwaves

### Output Files (per analysis directory)

- `bluesky_posts_complex.csv`: Collected posts from Bluesky with sentiment scores.
- `sentiment_distribution.png`: Sentiment analysis visualization.
- `keyword_network.graphml`: Graph in GraphML format for Gephi.
- `community_assignments.csv`: Community detection results.
- `global_metrics.csv`: Global network metrics.
- `node_metrics.csv`: Per-node metrics (degree, centrality, etc.).
- `keyword_network.png`: Network visualization (spring layout).
- `keyword_network_circular.png`: Network visualization (circular layout).
- `network_metrics.png`: Histograms of network metrics.
- `grafo.xlsx`: Co-occurrence matrix spreadsheet.

## Example Usage: Climate Change Co-occurrence Network Analysis

The output files in the `exports/` directory are generated by running the application with climate-change related keywords.

### Analysis Workflow

**Data Collection** → Posts are fetched from Bluesky API filtered by climate keywords (case-insensitive matching). The three analysis configurations allow comparing network structures with different keyword sets.

The keywords are defined in `scripts/config.py`:

```python
# Main keywords (shared by entire class)
MAIN_KEYWORDS = [
    "Energy Transition", "Greenhouse Effect", "Biodiversity", "Extreme weather events",
    "CO2", "Emissions", "Global Warming", "Glaciers", "Renewable Energy", "Fake News"
]

# Group 4 keywords
OUR_KEYWORDS = [
    "Ecosystem", "Fossil Fuels", "Energy Consumption", "Normatives", "Deforestation",
    "Flooding", "Tesla", "Green Policies", "Rain", "Electric Vehicles"
]

# Extra keywords
EXTRA_KEYWORDS = [
    "Natural Disaster", "Clean Energy", "Net Zero", "Tesla", "Heatwaves"
]
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
