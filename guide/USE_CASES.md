# Use Cases

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