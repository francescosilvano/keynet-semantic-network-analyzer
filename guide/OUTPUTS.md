# Analysis Outputs

All generated files are saved in the `exports/runs/` directory, organized by timestamped run:

- **Timestamped Directories**: Each run creates a unique folder with format `YYYY-MM-DD_HH-MM-SS_UUID`
- **Multiple Analyses**: Each run contains subdirectories for different keyword configurations:
  - **`full_analysis/`**: Complete analysis with all 25 keywords
  - **`main_keywords/`**: Analysis with main shared keywords only
  - **`main_plus_our/`**: Analysis with main + group-specific keywords

## Run Manifest

Each run includes a `manifest.json` file containing:
- Run UUID and timestamp
- Configuration parameters (keywords, date range, filters)
- Results summary (posts collected, network metrics)
- List of generated files
- Execution duration and any errors/warnings