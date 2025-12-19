
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

### Key Configuration Files

#### **scripts/config.py**
Contains all configuration parameters:
- **API Credentials**: Bluesky authentication settings
- **Keywords**: Define which keywords to track
- **Analysis Settings**: Co-occurrence thresholds, date ranges, filters
- **Archive Settings**: Enable/disable run archiving
- **Output Paths**: Configure where results are saved

#### **scripts/main.py**
Application entry point that orchestrates:
1. Data collection from Bluesky API
2. Co-occurrence analysis
3. Network graph generation
4. Metric computation
5. Visualization and export

#### **scripts/graph.py**
Network analysis implementation using NetworkX:
- Graph construction from co-occurrence data
- Local metrics: degree, strength, betweenness, closeness
- Global metrics: density, clustering, diameter
- Community detection (Louvain algorithm)
- Visualization generation

#### **scripts/archive.py**
Run management utilities:
- UUID-based directory creation
- Manifest generation
- Run indexing
- CLI tools for browsing and cleanup

### Customizing Keywords

Edit `scripts/settings.json` to modify the keyword sets:

```json

{
    "MAIN_KEYWORDS": [
        "main_keyword_1",
        "main_keyword_2",
        "main_keyword_3",
        "main_keyword_4",
        "main_keyword_5",
        "main_keyword_6",
        "main_keyword_7",
        "main_keyword_8",
        "main_keyword_9",
        "main_keyword_10"
    ],
    "GROUP_KEYWORDS": [
    "group_keyword_1",
    "group_keyword_2",
    "group_keyword_3",
    "group_keyword_4",
    "group_keyword_5",
    "group_keyword_6",
    "group_keyword_7",
    "group_keyword_8",
    "group_keyword_9",
    "group_keyword_10"
    ],
    "EXTRA_KEYWORDS": [
        "extra_keyword_1",
        "extra_keyword_2",
        "extra_keyword_3",
        "extra_keyword_4",
        "extra_keyword_5"
    ],
    "LOCATION_KEYWORDS": [
        "location-1",
        "location-2",
        "location-3"
    ]
}

KEYWORDS = [
    "Climate Change", "Global Warming", "Sustainability", "Renewable Energy",
    # Add your keywords here...
]
```

All keyword searches are **case-insensitive** for comprehensive matching.

### Environment Variables

Create a `.env` file in the project root:

```env
BLUESKY_HANDLE=your.handle.bsky.social
BLUESKY_PASSWORD=your-password
```

## Analysis Outputs

All generated files are saved in the `exports/runs/` directory, organized by timestamped run:

- **Timestamped Directories**: Each run creates a unique folder with format `YYYY-MM-DD_HH-MM-SS_UUID`
- **Multiple Analyses**: Each run contains subdirectories for different keyword configurations:
  - **`full_analysis/`**: Complete analysis with all 25 keywords
  - **`main_keywords/`**: Analysis with main shared keywords only
  - **`main_plus_our/`**: Analysis with main + group-specific keywords

### Run Manifest

Each run includes a `manifest.json` file containing:
- Run UUID and timestamp
- Configuration parameters (keywords, date range, filters)
- Results summary (posts collected, network metrics)
- List of generated files
- Execution duration and any errors/warnings

## Archive Management

### Managing Archives

The archive module provides CLI utilities:

```powershell
# List all archived runs
python scripts/archive.py list

# Show details of a specific run
python scripts/archive.py show <run_id_or_uuid>

# Show the latest run
python scripts/archive.py latest

# Cleanup old runs (keep last 10)
python scripts/archive.py cleanup --keep 10
```

### Archiving Configuration

Archive settings can be configured in `scripts/config.py`:
- `ARCHIVE_ENABLED`: Enable/disable archiving (default: `True`)
- `ARCHIVE_DIR`: Base directory for archived runs (default: `"../exports/runs"`)

To disable archiving and use legacy output structure, set `ARCHIVE_ENABLED = False` in `config.py`.

## Example Usage: Climate Change Co-occurrence Network Analysis

This example demonstrates how the application analyzes climate-related discussions on Bluesky.

### Analysis Workflow

1. **Data Collection**: Posts are fetched from Bluesky API, filtered by 25 climate keywords (case-insensitive):
   - Energy Transition, CO₂, Global Warming, Renewable Energy, Biodiversity
   - Fossil Fuels, Emissions, Extreme Weather, Clean Energy, and more

2. **Co-occurrence Analysis**: For each post, the script counts keyword pairs that appear together. If "Renewable Energy" and "Emissions" co-occur, that relationship is logged and weighted.

3. **Network Building**: Keywords become nodes; edges connect pairs that co-occur, weighted by frequency. Strong connections indicate frequently co-discussed topics.

### Understanding the Results

### Understanding the Results

The analysis reveals the semantic landscape of climate discussions:

#### Node Metrics (`node_metrics.csv`)
- **High degree**: Keywords co-occurring with many other terms
- **High betweenness**: Keywords bridging different climate topics
- **High closeness**: Keywords central to overall discussion network

#### Community Detection (`community_assignments.csv`)
Groups related keywords into thematic clusters (e.g., "Fossil Fuels," "Deforestation," "Emissions" forming an environmental impact cluster)

#### Network Visualizations
Visual maps showing how climate concepts interconnect

### Interpretation Example

If "Global Warming" has high betweenness centrality, it bridges discussions between:
- **Mitigation topics**: Renewable Energy, Green Policies
- **Impact topics**: Glaciers, Extreme Weather Events

This makes it a central connecting concept in climate conversations on social media.

## Network Metrics

### Local Metrics (Per Node)
- **Degree distribution**: Number of connections per keyword
- **Strength distribution**: Sum of edge weights per keyword
- **Betweenness centrality**: Importance as bridge between keywords
- **Closeness centrality**: Average distance to all other keywords

### Global Metrics (Entire Network)
- **Average degree**: Mean number of connections
- **Average strength**: Mean sum of edge weights
- **Graph density**: Ratio of actual edges to possible edges
- **Global clustering coefficient**: Overall clustering tendency
- **Graph diameter**: Longest shortest path (uses largest component if disconnected)
- **Community detection**: Louvain algorithm for modularity optimization
- **Modularity value**: Quality measure of community structure

## Future Features

Planned enhancements:

- [ ] **Symlink for Latest Run**: Quick access to most recent analysis
- [ ] **Automated Cleanup Policy**: Auto-cleanup after N runs or X days
- [ ] **Export Formats**: ZIP/TAR export for sharing and backup
- [ ] **Index Search**: Search by date, keywords, configuration parameters
- [ ] **Parallel Runs**: Handle multiple simultaneous executions

# License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.