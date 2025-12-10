import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import os

# --- IMPORT SHARED CONFIGURATION ---
from config import (
    KEYWORDS,
    INPUT_FILE,
    MIN_CO_OCCURRENCES,
    OUTPUT_DIR
)

# Create exports directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📊 Loading data...")
# Load the CSV data
df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df)} posts")

# --- COMPUTE KEYWORD CO-OCCURRENCES ---
print("\n🔍 Analyzing keyword co-occurrences...")
co_occurrence_data = []

for kw1, kw2 in combinations(KEYWORDS, 2):
    # Find posts containing both keywords
    mask = df.text.str.contains(kw1, case=False, na=False) & \
           df.text.str.contains(kw2, case=False, na=False)
    count = mask.sum()
    
    if count >= MIN_CO_OCCURRENCES:
        co_occurrence_data.append({
            'w1': kw1,
            'w2': kw2,
            'count': count
        })
        print(f"   {kw1} + {kw2}: {count}")

# Create DataFrame for co-occurrences
co_df = pd.DataFrame(co_occurrence_data)
print(f"\n✅ Found {len(co_df)} keyword pairs with co-occurrences")

# --- CREATE NETWORKX GRAPH ---
print("\n🌐 Building graph...")
G = nx.Graph()

# Add nodes (keywords)
G.add_nodes_from(KEYWORDS)

# Add edges (co-occurrences) with weights
edges = []
for _, row in co_df.iterrows():
    edges.append((row['w1'], row['w2'], {'weight': row['count']}))

G.add_edges_from(edges)

print(f"   Nodes: {G.number_of_nodes()}")
print(f"   Edges: {G.number_of_edges()}")

# --- GRAPH ANALYSIS ---
print("\n📈 Graph metrics:")

# Node degrees
degrees = dict(G.degree())
degree_values = list(degrees.values())

# Degree distribution
print(f"\n🔗 DEGREE DISTRIBUTION:")
print(f"   Average degree: {sum(degree_values) / len(degree_values):.3f}")
print(f"   Min degree: {min(degree_values)}")
print(f"   Max degree: {max(degree_values)}")
print(f"   Top keywords by connections:")
for keyword, degree in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"      {keyword}: {degree}")

# Strength distribution (weighted degree)
strength = dict(G.degree(weight='weight'))
strength_values = list(strength.values())
print(f"\n💪 STRENGTH DISTRIBUTION (weighted degree):")
print(f"   Average strength: {sum(strength_values) / len(strength_values):.3f}")
print(f"   Min strength: {min(strength_values)}")
print(f"   Max strength: {max(strength_values)}")
print(f"   Top keywords by strength:")
for keyword, strn in sorted(strength.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"      {keyword}: {strn}")

# Betweenness centrality
print(f"\n🎯 BETWEENNESS CENTRALITY:")
betweenness = nx.betweenness_centrality(G, weight='weight')
print(f"   Average betweenness: {sum(betweenness.values()) / len(betweenness):.4f}")
print(f"   Top keywords by betweenness:")
for keyword, bc in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"      {keyword}: {bc:.4f}")

# Closeness centrality
print(f"\n📍 CLOSENESS CENTRALITY:")
if nx.is_connected(G):
    closeness = nx.closeness_centrality(G, distance='weight')
    print(f"   Average closeness: {sum(closeness.values()) / len(closeness):.4f}")
    print(f"   Top keywords by closeness:")
    for keyword, cc in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {cc:.4f}")
else:
    print(f"   Network is disconnected - computing for largest component")
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc).copy()
    closeness = nx.closeness_centrality(G_largest, distance='weight')
    print(f"   Average closeness (largest component): {sum(closeness.values()) / len(closeness):.4f}")
    print(f"   Top keywords by closeness:")
    for keyword, cc in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {cc:.4f}")

# Global measures
print(f"\n🌍 GLOBAL MEASURES:")
print(f"   Graph density: {nx.density(G):.4f}")
print(f"   Global clustering coefficient: {nx.transitivity(G):.4f}")
print(f"   Connected components: {nx.number_connected_components(G)}")

if nx.is_connected(G):
    print(f"   Graph diameter: {nx.diameter(G)}")
    print(f"   Average shortest path length: {nx.average_shortest_path_length(G):.3f}")
else:
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc).copy()
    print(f"   Graph diameter (largest component): {nx.diameter(G_largest)}")
    print(f"   Average shortest path length (largest component): {nx.average_shortest_path_length(G_largest):.3f}")

# Local clustering
print(f"\n🔄 LOCAL CLUSTERING:")
clustering = nx.clustering(G)
avg_clustering = sum(clustering.values()) / len(clustering)
print(f"   Average local clustering: {avg_clustering:.4f}")
print(f"   Top keywords by local clustering:")
for keyword, clust in sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"      {keyword}: {clust:.4f}")

# Community detection (modularity)
print(f"\n🏘️ COMMUNITY DETECTION:")
if G.number_of_edges() > 0:
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G, weight='weight')
    modularity = community.modularity(G, communities, weight='weight')
    print(f"   Number of communities: {len(communities)}")
    print(f"   Modularity: {modularity:.4f}")
    print(f"   Community sizes:")
    for i, comm in enumerate(sorted(communities, key=len, reverse=True), 1):
        print(f"      Community {i}: {len(comm)} keywords - {', '.join(sorted(comm)[:5])}{'...' if len(comm) > 5 else ''}")
else:
    print(f"   No edges in graph - cannot detect communities")
    communities = []
    modularity = 0.0

# --- VISUALIZATION ---
print("\n🎨 Creating visualizations...")

# Figure 1: Network graph with node sizes based on degree
plt.figure(figsize=(16, 12))

# Calculate node sizes based on degree
node_sizes = [degrees[node] * 300 + 500 for node in G.nodes()]

# Calculate edge widths based on weight
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
max_weight = max(edge_weights) if edge_weights else 1
edge_widths = [2 + (w / max_weight) * 8 for w in edge_weights]

# Use spring layout for better visualization
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Draw the graph
nx.draw_networkx_nodes(G, pos, 
                       node_size=node_sizes,
                       node_color='lightblue',
                       edgecolors='darkblue',
                       linewidths=2,
                       alpha=0.9)

nx.draw_networkx_edges(G, pos,
                       width=edge_widths,
                       alpha=0.6,
                       edge_color='gray')

nx.draw_networkx_labels(G, pos,
                        font_size=10,
                        font_weight='bold',
                        font_family='sans-serif')

# Add edge labels showing co-occurrence counts
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

plt.title("Keyword Co-occurrence Network\n(Node size = connections, Edge width = co-occurrences)",
          fontsize=16, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/keyword_network.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_DIR}/keyword_network.png")

# Figure 2: Circular layout
plt.figure(figsize=(14, 14))

pos_circular = nx.circular_layout(G)

nx.draw_networkx_nodes(G, pos_circular,
                       node_size=node_sizes,
                       node_color='lightcoral',
                       edgecolors='darkred',
                       linewidths=2,
                       alpha=0.9)

nx.draw_networkx_edges(G, pos_circular,
                       width=edge_widths,
                       alpha=0.5,
                       edge_color='gray')

nx.draw_networkx_labels(G, pos_circular,
                        font_size=11,
                        font_weight='bold')

plt.title("Keyword Co-occurrence Network (Circular Layout)",
          fontsize=16, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/keyword_network_circular.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_DIR}/keyword_network_circular.png")

# Figure 3: Degree distribution
plt.figure(figsize=(12, 8))

# Subplot 1: Degree distribution
plt.subplot(2, 2, 1)
degree_values = list(degrees.values())
plt.hist(degree_values, bins=range(0, max(degree_values) + 2), 
         edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Degree', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Degree Distribution', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Subplot 2: Strength distribution
plt.subplot(2, 2, 2)
strength_values = list(strength.values())
plt.hist(strength_values, bins=15, edgecolor='black', alpha=0.7, color='coral')
plt.xlabel('Strength (Weighted Degree)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Strength Distribution', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Subplot 3: Betweenness centrality
plt.subplot(2, 2, 3)
betweenness_values = list(betweenness.values())
plt.hist(betweenness_values, bins=15, edgecolor='black', alpha=0.7, color='lightgreen')
plt.xlabel('Betweenness Centrality', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Betweenness Centrality Distribution', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Subplot 4: Clustering coefficient
plt.subplot(2, 2, 4)
clustering_values = list(clustering.values())
plt.hist(clustering_values, bins=15, edgecolor='black', alpha=0.7, color='orchid')
plt.xlabel('Local Clustering Coefficient', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Clustering Coefficient Distribution', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/network_metrics.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {OUTPUT_DIR}/network_metrics.png")

# --- SAVE GRAPH DATA ---
# Save edge list with weights
nx.write_weighted_edgelist(G, f"{OUTPUT_DIR}/keyword_network_edges.txt")
print(f"   ✅ Saved: {OUTPUT_DIR}/keyword_network_edges.txt")

# Save graph in GraphML format (for other tools like Gephi)
nx.write_graphml(G, f"{OUTPUT_DIR}/keyword_network.graphml")
print(f"   ✅ Saved: {OUTPUT_DIR}/keyword_network.graphml")

# Save all metrics to CSV
print("\n💾 Saving metrics to CSV files...")

# Degree and strength metrics
metrics_df = pd.DataFrame({
    'keyword': list(degrees.keys()),
    'degree': list(degrees.values()),
    'strength': [strength[k] for k in degrees.keys()],
    'betweenness': [betweenness[k] for k in degrees.keys()],
    'clustering': [clustering[k] for k in degrees.keys()]
})
metrics_df = metrics_df.sort_values('degree', ascending=False)
metrics_df.to_csv(f'{OUTPUT_DIR}/node_metrics.csv', index=False)
print(f"   ✅ Saved: {OUTPUT_DIR}/node_metrics.csv")

# Community assignments
if communities:
    community_assignments = []
    for i, comm in enumerate(communities, 1):
        for keyword in comm:
            community_assignments.append({'keyword': keyword, 'community': i})
    community_df = pd.DataFrame(community_assignments).sort_values(['community', 'keyword'])
    community_df.to_csv(f'{OUTPUT_DIR}/community_assignments.csv', index=False)
    print(f"   ✅ Saved: {OUTPUT_DIR}/community_assignments.csv")
else:
    print("   ⚠️ No communities to save")

# Global metrics summary
if nx.is_connected(G):
    diameter_val = nx.diameter(G)
    avg_path_val = nx.average_shortest_path_length(G)
    diameter_label = 'Diameter'
    path_label = 'Average path length'
elif G.number_of_edges() > 0:
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc).copy()
    diameter_val = nx.diameter(G_largest)
    avg_path_val = nx.average_shortest_path_length(G_largest)
    diameter_label = 'Diameter (largest component)'
    path_label = 'Average path length (largest component)'
else:
    diameter_val = 0
    avg_path_val = 0
    diameter_label = 'Diameter'
    path_label = 'Average path length'

global_metrics = {
    'Metric': [
        'Number of nodes',
        'Number of edges',
        'Average degree',
        'Average strength',
        'Graph density',
        'Global clustering coefficient',
        'Number of communities',
        'Modularity',
        'Connected components',
        diameter_label,
        path_label
    ],
    'Value': [
        G.number_of_nodes(),
        G.number_of_edges(),
        round(sum(degree_values) / len(degree_values), 3) if degree_values else 0,
        round(sum(strength_values) / len(strength_values), 3) if strength_values else 0,
        round(nx.density(G), 4),
        round(nx.transitivity(G), 4),
        len(communities) if communities else 0,
        round(modularity, 4),
        nx.number_connected_components(G),
        diameter_val,
        round(avg_path_val, 3)
    ]
}
global_df = pd.DataFrame(global_metrics)
global_df.to_csv(f'{OUTPUT_DIR}/global_metrics.csv', index=False)
print(f"   ✅ Saved: {OUTPUT_DIR}/global_metrics.csv")

# --- SUMMARY STATISTICS ---
print("\n" + "="*60)
print("📊 NETWORK ANALYSIS SUMMARY")
print("="*60)
print(f"\n📐 BASIC STRUCTURE:")
print(f"   Nodes: {G.number_of_nodes()}")
print(f"   Edges: {G.number_of_edges()}")
print(f"   Density: {nx.density(G):.4f}")

print(f"\n🔗 DEGREE METRICS:")
print(f"   Average degree: {sum(degree_values) / len(degree_values):.3f}")
print(f"   Average strength: {sum(strength_values) / len(strength_values):.3f}")

print(f"\n🎯 CENTRALITY METRICS:")
print(f"   Average betweenness: {sum(betweenness.values()) / len(betweenness):.4f}")
if nx.is_connected(G):
    print(f"   Average closeness: {sum(nx.closeness_centrality(G).values()) / len(G.nodes()):.4f}")

print(f"\n🌍 GLOBAL METRICS:")
print(f"   Clustering coefficient: {nx.transitivity(G):.4f}")
print(f"   Average local clustering: {avg_clustering:.4f}")
if nx.is_connected(G):
    print(f"   Diameter: {nx.diameter(G)}")
    print(f"   Average path length: {nx.average_shortest_path_length(G):.3f}")
else:
    print(f"   Diameter (largest): {nx.diameter(G_largest)}")
    print(f"   Average path length (largest): {nx.average_shortest_path_length(G_largest):.3f}")

print(f"\n🏘️ COMMUNITY STRUCTURE:")
print(f"   Communities: {len(communities) if communities else 0}")
print(f"   Modularity: {modularity:.4f}")

print("\n" + "="*60)
print("✨ Analysis complete!")
print("="*60)