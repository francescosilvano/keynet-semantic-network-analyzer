# Co-occurrence Network Analysis

## Task Overview
You are an expert in Python, data collection, and Complex Network Analysis. Modify an existing script (Bluesky API + sentiment + co-occurrence analysis) to use the specified keyword sets, collect posts from Bluesky, generate the co-occurrence graph, and compute all required network measures.

---

## Keywords Configuration

### Main Keywords (Shared by Entire Class)
green transition, greenhouse effect, loss of biodiversity, extreme weather events, CO2, emissions, global warming, melting glaciers, renewable energy, misinformation

### Group 4 Keywords
fossil fuels, energy consumption, normatives, deforestation, flooding, tesla, green policies, rain, electric vehicles

### Extra Keywords (Optional)
natural disaster, clean energy, net zero, AI, heatwaves

### Final Python Keywords List
```python
KEYWORDS = [
    "green transition", "greenhouse effect", "loss of biodiversity", "extreme weather events",
    "CO2", "emissions", "global warming", "melting glaciers", "renewable energy", "misinformation",
    "fossil fuels", "energy consumption", "normatives", "deforestation", "flooding",
    "tesla", "green policies", "rain", "electric vehicles",
    "natural disaster", "clean energy", "net zero", "AI", "heatwaves"
]
```
**Important**: All keyword searches must be case-insensitive.

---

## Dependencies
Ensure the following Python libraries are installed:
```bash
pip install networkx matplotlib pandas openpyxl
```

---

## Required Network Metrics

### Local Metrics (Per Node)
- **Degree distribution**: Number of connections per keyword
- **Strength distribution**: Sum of edge weights per keyword
- **Betweenness centrality**: Importance as bridge between other keywords
- **Closeness centrality**: Average distance to all other keywords

### Global Metrics (Entire Network)
- **Average degree**: Mean number of connections
- **Average strength**: Mean sum of edge weights
- **Graph density**: Ratio of actual edges to possible edges
- **Global clustering coefficient**: Overall clustering tendency
- **Graph diameter**: Longest shortest path (for largest component if disconnected)
- **Community detection**: Louvain algorithm for modularity optimization
- **Modularity value**: Quality measure of community structure
- **Cluster assignments**: Community membership for each node

---

## Workflow

1. **Data Collection**:
   - Connect to Bluesky API
   - Collect posts containing specified keywords
   - Store data in CSV format with `testo` column

2. **Co-occurrence Analysis**:
   - For each pair of keywords, count co-occurrences in posts (case-insensitive)
   - Build co-occurrence matrix
   - Filter pairs with co-occurrences >= `MIN_CO_OCCURRENCES` (default: 1)

3. **Graph Construction**:
   - Create NetworkX graph with keywords as nodes
   - Add edges weighted by co-occurrence counts

4. **Network Analysis**:
   - Calculate all local metrics for each node
   - Calculate all global metrics for the network
   - Perform community detection using Louvain algorithm
   - Handle disconnected graphs by using largest component for diameter

5. **Visualization**:
   - Generate network graph visualizations (spring and circular layouts)
   - Create metrics histograms (degree, strength, centrality, clustering)

6. **Output Generation**:
   - Export all results to CSV/Excel files
   - Generate summary text of global measures

---

## Expected Outputs

| File Name                     | Description                                      |
|-------------------------------|--------------------------------------------------|
| `keyword_network_edges.csv`   | Edge list: w1, w2, weight (co-occurrences)      |
| `node_metrics.csv`            | Per-node: degree, strength, betweenness, closeness, community |
| `global_metrics.csv`          | Global network metrics summary                   |
| `community_assignments.csv`   | Community assignments for each keyword           |
| `keyword_network.png`         | Network graph (spring layout)                    |
| `keyword_network_circular.png`| Network graph (circular layout)                  |
| `network_metrics.png`         | Histograms of degree, strength, centrality, etc. |
| `keyword_network.graphml`     | GraphML file for external tools (e.g., Gephi)   |

Save all the outcomes in /exports folder

---

## Code Requirements

1. **Integrate Keywords**: Use the full `KEYWORDS` list with all 24 keywords
2. **Case-Insensitive Search**: Ensure all keyword matching is case-insensitive
3. **Co-occurrence Matrix**: Rebuild using new keywords
4. **NetworkX Metrics**: Compute all required local and global metrics
5. **Export Formats**: Generate CSV/XLSX files for all outputs
6. **Documentation**: Include clear comments and instructions
7. **Clean Code**: Keep code organized, commented, and ready to run

---

## How to Read Outputs

- **`keyword_network_edges.csv`**: Shows which keyword pairs co-occur and how frequently (weight column)
- **`node_metrics.csv`**: 
  - High degree/strength = frequently co-occurring keywords
  - High betweenness = keywords that bridge different topics
  - High closeness = keywords central to the discussion
  - Community = group assignment from Louvain algorithm
- **`global_metrics.csv`**: Overall network structure characteristics
  - Density close to 1 = highly interconnected topics
  - High clustering = topics form tight groups
  - Modularity > 0.3 = well-defined communities
- **`community_assignments.csv`**: Which keywords belong to which thematic cluster

---

## Notes
- **Disconnected Graphs**: If the graph is disconnected, diameter is computed for the largest connected component
- **Performance**: For large datasets, consider increasing `MIN_CO_OCCURRENCES` to reduce noise
- **Community Detection**: Louvain algorithm provides automatic clustering; higher modularity indicates better community structure
- **Customization**: All parameters (keywords, thresholds, visualizations) can be adjusted as needed

---
