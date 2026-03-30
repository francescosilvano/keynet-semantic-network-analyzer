# NetworkLens CLI

Social media platforms are among the richest and most current sources of unstructured text data. NetworkLens is a Python-based CLI tool that collects posts from the Bluesky API and analyses keyword co-occurrence to build semantic networks. Using graph theory, it maps relationships between keywords, detects thematic communities, and computes network metrics — making it useful for researchers, analysts, and investigators studying trends, discourse patterns, or misinformation in online text.

## Use Cases

- **Semantic Mapping**: Reveals associations between keywords through co-occurrence analysis, surfacing emerging topics or narrative clusters.
- **Network Metrics**: Computes degree centrality, betweenness, and community structures for quantitative analysis of keyword dynamics.
- **Visualization**: Generates graphs and metric charts suitable for reports or exploratory analysis.
- **Reproducibility**: Docker support ensures consistent environments; automated archiving maintains versioned, indexed run histories.

## Features

- Bluesky post collection via API with keyword filtering
- Co-occurrence graph construction using NetworkX
- Local (node-level) and global network metric computation
- Community detection via the Louvain algorithm
- Visualizations: spring and circular layouts, metric histograms, optional sentiment distributions
- Multi-mode analysis: full keyword sets, main-only, or extended groups
- UUID-timestamped run archiving with metadata indexing
- Export formats: CSV, GraphML (Gephi/Cytoscape), PNG, XLSX matrices


## Installation

### Prerequisites
- Requires Python 3.11 or higher (due to dependencies compatibility).
- Bluesky account credentials for API access.

### Option 1: Docker (Recommended)
1. Clone the repo: `git clone https://github.com/francescosilvano/networklens.git`
2. Create `.env` file with:
   ```
   BLUESKY_HANDLE=your.handle.bsky.social
   BLUESKY_PASSWORD=your-password
   ```
3. Build and run: `docker compose up --build`
   - Outputs saved in `exports/`.

### Option 2: Local Setup
1. Clone the repo and navigate to it.
2. Create virtual environment: `python -m venv venv && source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows).
3. Install the package (recommended): `pip install -e .` or `pip install .`
4. Add `.env` (BLUESKY_HANDLE, BLUESKY_PASSWORD) to the project root.
5. Run the CLI: `networklens`


## Dependencies

Core dependencies are installed automatically via pip:

- **atproto** - Bluesky AT Protocol client for post collection
- **textblob** - Sentiment analysis
- **pandas** - Data manipulation and CSV export
- **matplotlib** - Graph and histogram visualisation
- **networkx** - Co-occurrence graph construction and metric computation
- **python-dotenv** - Environment variable management
- **openpyxl** - XLSX matrix export

For development (linting, testing, and building):

```bash
pip install networklens-semantic-network-analyzer[dev]
```

This installs `pylint`, `pytest`, and `build` in addition to the core dependencies.

**Dependencies** are declared in `pyproject.toml` and will be installed by pip.

## Usage
Configure parameters in `networklens/config.py` (e.g., keywords, fetch limits). Run the CLI to collect data, build graphs, and generate outputs in `exports/runs/<timestamp_uuid>/`.

The tool supports customization of fetch periods, keyword lists, and optional sentiment analysis for deeper insights, while also allowing parallel configurations to run multiple analyses simultaneously for comparative purposes. Outputs can be explored interactively by loading GraphML files into external tools, or reviewed quantitatively through the exported CSV metrics.

## Limitations
- Currently Bluesky-focused; extendable to other APIs (e.g., X/Twitter) with modifications;
- No built-in multilingual support or advanced NLP (e.g., embeddings);
- No parallel executions, compressed exports, searchable run indexes.

## Contributors

This project began as part of the Complex Systems course at the University of Siena in the 2025/26 academic year. [Francesco Silvano](https://github.com/francescosilvano)is the primary author and maintainer, and [Raphael](https://github.com/RaphaelNoah)  provided substantial contributions, particularly in refining the codebase to improve the accuracy and reliability of the data processing and results.

This work is the result of a joint project to explore and implement concepts from complex systems theory within an academic context.

## License

This work is distributed under the MIT License. 
