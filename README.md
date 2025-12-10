
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

## Project Structure

```txt
complex-systems/
├── .github/                           # GitHub workflows and CI/CD configuration
├── .pylintrc                          # Pylint configuration for code quality
├── config.py                          # Configuration parameters and settings
├── main.py                            # Entry point - data collection from Bluesky API
├── graph.py                           # Network analysis and visualization
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── exports/                           # Output directory for all generated files
│   ├── bluesky_posts_complex.csv      # Collected posts from Bluesky
│   ├── community_assignments.csv      # Community detection results
│   ├── global_metrics.csv             # Global network metrics
│   ├── node_metrics.csv               # Per-node metrics (degree, centrality, etc.)
│   ├── keyword_network_edges.txt      # Edge list with weights
│   ├── keyword_network.graphml        # Graph in GraphML format (for Gephi)
│   ├── keyword_network.png            # Network visualization (spring layout)
│   ├── keyword_network_circular.png   # Network visualization (circular layout)
│   ├── network_metrics.png            # Metrics histograms
│   └── grafo.xlsx                     # Co-occurrence matrix spreadsheet
└── venv/                              # Virtual environment (not in git)
```

### Key Files

#### Configuration

**config.py** contains all configuration parameters for the co-occurrence analysis, including API credentials, keywords, and output settings. Modify these parameters to customize the analysis.

#### Entry Point

**main.py** is the application entry point. Run this file to execute the complete workflow: data collection, co-occurrence analysis, graph generation, and metrics computation.

#### Graph Operations

**graph.py** implements the network graph data structure and analysis using NetworkX. This module handles graph construction, metric calculations, and community detection.
