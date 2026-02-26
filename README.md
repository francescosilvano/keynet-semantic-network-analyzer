# KeyNet CLI

KeyNet is a Python-based command-line tool designed for analyzing keyword co-occurrence networks in text corpora, with a primary focus on social media data from the Bluesky platform. By leveraging graph theory it enables users to map semantic relationships, detect thematic communities, and extract actionable insights from unstructured text. This tool is particularly useful for researchers, data analysts, and social media investigators exploring trends, interconnections, and sentiment in online discussions.

## Benefits and Use Cases

- Semantic Mapping: Uncovers hidden associations between keywords through co-occurrence analysis, helping identify emerging topics or misinformation patterns.
- Network Insights: Computes metrics like degree centrality, betweenness, and community structures, providing quantitative views of keyword dynamics.
- Visualization: Generates graphs and charts for intuitive understanding, aiding in reports or presentations.
- Reproducibility: Supports Docker for consistent environments and automated archiving for versioned runs.

## Features

- Automated Bluesky post collection via API with keyword filtering.
- Graph construction using NetworkX for co-occurrence networks.
- Computation of local (node-specific) and global network metrics.
- Community detection via Louvain algorithm.
- Visualizations: Spring and circular layouts, metric histograms, sentiment distributions (optional).
- Multi-mode analysis: Full keywords, main-only, or extended groups.
- Archiving: UUID-timestamped runs with metadata indexing.
- Export formats: CSV, GraphML (for tools like Gephi/Cytoscape), PNG images, XLSX matrices.

## Installation

### Prerequisites
- Python 3.11–3.14 (due to NetworkX compatibility).
- Bluesky account credentials for API access.

### Option 1: Docker (Recommended for Reproducibility)
1. Clone the repo: `git clone https://github.com/francescosilvano/keynet-semantic-network-analyzer.git`
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
5. Run the CLI: `keynet`

**Dependencies** are declared in `pyproject.toml` and will be installed by pip (NetworkX, Matplotlib, pandas, Bluesky client, etc.).

## Usage
Configure parameters in `keynet/config.py` (e.g., keywords, fetch limits). Run the CLI to collect data, build graphs, and generate outputs in `exports/runs/<timestamp_uuid>/`.

Example command: `python scripts/main.py`

- **Customization**: Adjust fetch periods, keyword lists, or enable sentiment analysis for deeper insights.
- **Multiple Runs**: Supports parallel configurations for comparative analyses.
- **Viewing Outputs**: Use GraphML in external tools for interactive exploration; review CSVs for metrics.

## Project Structure
- `keynet/`: Core Python package (main.py, graph.py, config.py).
- `exports/`: Results directory with archived runs.
- `guide/`: Detailed docs on config, analysis, outputs, etc.
- Docker files for containerization.

## Limitations and Future Work
- Currently Bluesky-focused; extendable to other APIs (e.g., X/Twitter) with modifications.
- No built-in multilingual support or advanced NLP (e.g., embeddings); considering integrating [spaCy](https://spacy.io/) for enhancements.
- Future: Parallel executions, compressed exports, searchable run indexes.

## Contributors

This project was developed as part of the Complex Systems course at the University of Siena during the academic year 2025/2026.

The primary author and maintainer is [Francesco Silvano](https://github.com/francescosilvano); [Raphael](https://github.com/RaphaelNoah) provided substantial contributions, particularly in refining the codebase to achieve greater accuracy and reliability in the data processing and results.

This work reflects a collaborative effort to explore and implement concepts from complex systems theory within an academic setting.

## License

This work is distributed under the MIT License.
