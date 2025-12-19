
# Analyzing Keyword Co-occurrence Networks in Social Media

## Project Overview

This project analyzes co-occurrence networks of keywords extracted from posts on the Bluesky social media platform. It collects data using the Bluesky API, constructs a network graph based on keyword co-occurrences, computes various network metrics, and visualizes the results.

The aim is to explore the relationships between keywords and identify communities within the network, providing insights into trending topics and their interconnections.

## Features

- **Automated Data Collection**: Fetch posts from Bluesky API with customizable keyword filters
- **Network Analysis**: Compute comprehensive local and global network metrics
- **Community Detection**: Identify thematic clusters using Louvain algorithm
- **Visualization**: Generate network graphs and metric distributions
- **Run Archiving**: Automatic versioning of analysis runs with UUID-based directories
- **Multiple Analysis Modes**: Run different keyword configurations in parallel
- **Docker Support**: Containerized environment for reproducible analysis

## Quick Start

### Prerequisites

**Python Version**: This project requires **Python 3.11, 3.12, 3.13, or 3.14** (as specified by [NetworkX installation requirements](https://networkx.org/documentation/stable/install.html)).

### Option 1: Using Docker (Recommended)

1. Clone the repository
2. Navigate to the project directory
3. Populate the `.env` file in the root directory with your Bluesky credentials:

   ```env
   BLUESKY_HANDLE=your.handle.bsky.social
   BLUESKY_PASSWORD=your-password
   ```

4. Run with Docker Compose:

   ```bash
   docker compose up --build
   ```

   The analysis will run automatically and outputs will be saved in the `exports/` directory.

#### Docker Configuration

- **Dockerfile**: Uses Python 3.13-slim base image with matplotlib configuration
- **compose.yaml**: Orchestrates container with environment variables and volume mounts


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

5. Populate the `.env` file in the root directory (see Option 1, step 3)

6. Run the analysis:

   ```powershell
   cd scripts
   python main.py
   ```

## Project Structure

```txt
keyword-network-analysis/
├── .github/                           # GitHub Copilot instructions
│   └── copilot-instructions.md
├── compose.yaml                       # Docker Compose configuration
├── Dockerfile                         # Docker image definition
├── README.md                          # Project documentation
├── scripts/                           # Python scripts for analysis
│   ├── config.py                      # Configuration parameters and settings
│   ├── archive.py                     # Archive management utilities
│   ├── main.py                        # Entry point - data collection from Bluesky API
│   ├── graph.py                       # Network analysis and visualization
│   └── requirements.txt               # Python dependencies
└── exports/                           # Output directory for all generated files
    ├── runs/                          # Versioned analysis runs
    │   ├── index.json                 # Index of all archived runs
    │   └── YYYY-MM-DD_HH-MM-SS_UUID/  # Timestamped run directories
    │       ├── manifest.json          # Run metadata and configuration
    │       ├── full_analysis/         # Complete analysis with all keywords
    │       ├── main_keywords/         # Analysis with main keywords only
    │       └── main_plus_our/         # Analysis with main + group keywords
    └── archive/                       # Legacy archived results
```

Each analysis subfolder contains:

- **`bluesky_posts_complex.csv`**: Collected posts from Bluesky with metadata
- **`keyword_network_edges.txt`**: Edge list with co-occurrence weights
- **`node_metrics.csv`**: Per-node metrics (degree, strength, betweenness, closeness, community)
- **`community_assignments.csv`**: Community detection results
- **`global_metrics.csv`**: Global network metrics summary
- **`keyword_network.graphml`**: Graph in GraphML format (for Gephi, Cytoscape)
- **`keyword_network.png`**: Network visualization (spring layout)
- **`keyword_network_circular.png`**: Network visualization (circular layout)
- **`network_metrics.png`**: Metrics histograms
- **`sentiment_distribution.png`**: Sentiment analysis chart (if enabled)
- **`grafo.xlsx`**: Co-occurrence matrix spreadsheet

## Configuration
Configuration parameters are defined in `scripts/config.py`. Learn more in the [Configuration Documentation](guide/CONFIG.md).

## Network Analysis

The script analyses the keyword co-occurrence network using NetworkX, computing local and global metrics, and performing community detection. Learn more in the [Network Analysis Documentation](guide/ANALYSIS.md).

## Outputs

Analysis results are saved in the `exports/runs/` directory, organized by timestamped UUID folders. Learn more in the [Outputs Documentation](guide/OUTPUTS.md).

## Archiving

Each analysis run is archived with a unique UUID and timestamp. The `exports/runs/index.json` file maintains an index of all archived runs for easy reference. Learn more in the [Archiving Documentation](guide/ARCHIVE.md).

## Use cases

Refer to [Use Cases](guide/USE_CASES.md) for examples of how to utilize KNA for different keyword analysis scenarios. 


## Future Features

Planned enhancements:

- [ ] Parallel Runs: Handle multiple simultaneous executions
- [ ] Export Formats: ZIP/TAR export for sharing and backup
- [ ] Index Search: Search by date, keywords, configuration parameters

# License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.