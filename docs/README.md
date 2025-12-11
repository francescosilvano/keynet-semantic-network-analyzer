# Dashboard Documentation

This directory contains the interactive dashboard for visualizing the Complex Systems network analysis results.

## Files

- `dashboard.qmd` - Quarto Markdown source file for the dashboard
- `dashboard.html` - Generated HTML dashboard (not tracked in git)

## How to Use

### Viewing the Dashboard

1. **Render the dashboard** (requires Quarto):
   ```bash
   cd docs
   quarto render dashboard.qmd
   ```

2. **Open the generated HTML file**:
   ```bash
   open dashboard.html
   # or
   python -m http.server 8000
   # then navigate to http://localhost:8000/dashboard.html
   ```

### Prerequisites

To render the dashboard, you need:

1. **Quarto CLI**: Download from https://quarto.org/docs/get-started/
2. **Python packages**:
   ```bash
   pip install pandas matplotlib ipython jupyter tabulate
   ```

### Dashboard Features

The dashboard presents three network analyses:

1. **Main Keywords Only** - 10 shared keywords
2. **Main + Our Keywords** - 20 keywords total
3. **Full Analysis** - 25 keywords total

Each analysis section includes:
- Sentiment distribution chart
- Network visualizations (spring and circular layouts)
- Network metrics histograms
- Key statistics tables

### Navigation

- Use the **top-level tabs** to switch between analyses
- Within each analysis, use **nested tabs** to toggle between network layouts
- The **Table of Contents** (right side) allows quick jumps to different sections

### Customization

To customize the dashboard:

1. Edit `dashboard.qmd`
2. Modify the YAML header for styling options
3. Update Python code blocks to change data processing
4. Re-render with `quarto render dashboard.qmd`

## Troubleshooting

**Issue**: Charts not displaying
- **Solution**: Ensure the exports directory structure exists with all required image files

**Issue**: Python errors during rendering
- **Solution**: Install all required packages listed in `scripts/requirements.txt`

**Issue**: Quarto not found
- **Solution**: Install Quarto CLI from https://quarto.org/

## Advanced Usage

### Publishing Online

To publish the dashboard as a website:

```bash
quarto publish dashboard.qmd
```

Options include GitHub Pages, Netlify, or Quarto Pub.

### Embedding in Other Documents

The dashboard uses `embed-resources: true`, making it a self-contained HTML file that can be:
- Shared via email or cloud storage
- Embedded in other websites
- Opened offline without dependencies

## Support

For issues or questions, refer to:
- Quarto documentation: https://quarto.org/docs/
- Project README: ../README.md
