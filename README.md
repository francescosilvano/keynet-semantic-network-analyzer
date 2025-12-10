
# Complex Systems Project

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

### Configuration

**config.py** contains all configuration parameters for the co-occurrence analysis, including API credentials, keywords, and output settings. Modify these parameters to customize the analysis.

### Entry Point

**main.py** is the application entry point. Run this file to execute the complete workflow: data collection, co-occurrence analysis, graph generation, and metrics computation.

### Graph Operations

**graph.py** implements the network graph data structure and analysis using NetworkX. This module handles graph construction, metric calculations, and community detection.
