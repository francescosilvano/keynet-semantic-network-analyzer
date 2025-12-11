"""Network analysis module for keyword co-occurrence graphs"""

import os
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms import community

# --- IMPORT SHARED CONFIGURATION ---
from config import (
    ANALYSIS_CONFIGS,
    MIN_CO_OCCURRENCES,
    OUTPUT_DIR
)


def analyze_network(keywords, input_file, output_dir, description):
    """
    Run complete network analysis for a given keyword set
    
    Parameters:
    -----------
    keywords : list
        List of keywords to analyze
    input_file : str
        Path to input CSV file with posts
    output_dir : str
        Directory to save outputs
    description : str
        Description of this analysis (for titles/labels)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"NETWORK ANALYSIS: {description}")
    print(f"Keywords: {len(keywords)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")

    print("\nLoading data...")
    # Load the CSV data
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} posts")

    # --- COMPUTE KEYWORD CO-OCCURRENCES ---
    print("\nAnalyzing keyword co-occurrences...")
    co_occurrence_data = []

    for kw1, kw2 in combinations(keywords, 2):
        # Find posts containing both keywords (case-insensitive)
        mask = (df.text.str.contains(kw1, case=False, na=False) &
                df.text.str.contains(kw2, case=False, na=False))
        count = mask.sum()

        if count >= MIN_CO_OCCURRENCES:
            co_occurrence_data.append({
                'w1': kw1,
                'w2': kw2,
                'count': count
            })

    # Create DataFrame for co-occurrences
    co_df = pd.DataFrame(co_occurrence_data)
    print(f"\nFound {len(co_df)} keyword pairs with co-occurrences")

    if len(co_df) == 0:
        print("WARNING: No co-occurrences found. Skipping network analysis.")
        return

    # --- CREATE NETWORKX GRAPH ---
    print("\nBuilding graph...")
    G = nx.Graph()

    # Add nodes (keywords)
    G.add_nodes_from(keywords)

    # Add edges (co-occurrences) with weights
    edges = []
    for _, row in co_df.iterrows():
        edges.append((row['w1'], row['w2'], {'weight': row['count']}))

    G.add_edges_from(edges)

    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")

    # --- GRAPH ANALYSIS ---
    print("\nGraph metrics:")

    # Node degrees
    degrees = dict(G.degree())
    degree_values = list(degrees.values())

    # Degree distribution
    print("\nDEGREE DISTRIBUTION:")
    print(f"   Average degree: {sum(degree_values) / len(degree_values):.3f}")
    print(f"   Min degree: {min(degree_values)}")
    print(f"   Max degree: {max(degree_values)}")
    print("   Top keywords by connections:")
    for keyword, degree in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {degree}")

    # Strength distribution (weighted degree)
    strength = dict(G.degree(weight='weight'))
    strength_values = list(strength.values())
    print("\nSTRENGTH DISTRIBUTION (weighted degree):")
    print(f"   Average strength: {sum(strength_values) / len(strength_values):.3f}")
    print(f"   Min strength: {min(strength_values)}")
    print(f"   Max strength: {max(strength_values)}")
    print("   Top keywords by strength:")
    for keyword, strn in sorted(strength.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {strn}")

    # Betweenness centrality
    print("\nBETWEENNESS CENTRALITY:")
    betweenness = nx.betweenness_centrality(G, weight='weight')
    print(f"   Average betweenness: {sum(betweenness.values()) / len(betweenness):.4f}")
    print("   Top keywords by betweenness:")
    for keyword, bc in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {bc:.4f}")

    # Closeness centrality
    print("\nCLOSENESS CENTRALITY:")
    if nx.is_connected(G):
        closeness = nx.closeness_centrality(G, distance='weight')
        print(f"   Average closeness: {sum(closeness.values()) / len(closeness):.4f}")
        print("   Top keywords by closeness:")
        for keyword, cc in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {keyword}: {cc:.4f}")
    else:
        print("   Graph is disconnected - calculating closeness for largest component")
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc).copy()
        closeness = nx.closeness_centrality(G_largest, distance='weight')
        # Fill in zeros for nodes not in largest component
        closeness_full = {node: closeness.get(node, 0) for node in G.nodes()}
        closeness = closeness_full
        print(f"   Average closeness (largest component): {sum(closeness.values()) / len(closeness):.4f}")
        print("   Top keywords by closeness:")
        for keyword, cc in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {keyword}: {cc:.4f}")

    # Global measures
    print("\nGLOBAL MEASURES:")
    print(f"   Graph density: {nx.density(G):.4f}")
    print(f"   Global clustering coefficient: {nx.transitivity(G):.4f}")
    print(f"   Connected components: {nx.number_connected_components(G)}")

    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"   Diameter: {diameter}")
        print(f"   Average path length: {avg_path_length:.3f}")
        DIAMETER_LABEL = "Diameter"
        DIAMETER_VAL = diameter
        PATH_LABEL = "Average path length"
        AVG_PATH_VAL = avg_path_length
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc).copy()
        diameter = nx.diameter(G_largest)
        avg_path_length = nx.average_shortest_path_length(G_largest)
        print(f"   Diameter (largest component): {diameter}")
        print(f"   Average path length (largest component): {avg_path_length:.3f}")
        DIAMETER_LABEL = "Diameter (largest component)"
        DIAMETER_VAL = diameter
        PATH_LABEL = "Average path length (largest component)"
        AVG_PATH_VAL = avg_path_length

    # Local clustering
    print("\nLOCAL CLUSTERING:")
    clustering = nx.clustering(G)
    avg_clustering = sum(clustering.values()) / len(clustering)
    print(f"   Average local clustering: {avg_clustering:.4f}")
    print("   Top keywords by local clustering:")
    for keyword, clust in sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {clust:.4f}")

    # Community detection (modularity)
    print("\nCOMMUNITY DETECTION:")
    if G.number_of_edges() > 0:
        communities = list(community.greedy_modularity_communities(G, weight='weight'))
        MODULARITY = community.modularity(G, communities, weight='weight')
        print(f"   Number of communities: {len(communities)}")
        print(f"   Modularity: {MODULARITY:.4f}")
        print("\n   Community assignments:")
        for i, comm in enumerate(communities, 1):
            print(f"      Community {i} ({len(comm)} keywords): {', '.join(sorted(comm))}")
    else:
        communities = []
        MODULARITY = 0
        print("   No edges - no communities detected")

    # --- VISUALIZATION ---
    print("\nCreating visualizations...")

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

    TITLE_TEXT = (f"Keyword Co-occurrence Network - {description}\n"
                  "(Node size = connections, Edge width = co-occurrences)")
    plt.title(TITLE_TEXT,
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/keyword_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/keyword_network.png")

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

    plt.title(f"Keyword Co-occurrence Network (Circular Layout) - {description}",
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/keyword_network_circular.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/keyword_network_circular.png")

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

    plt.suptitle(f'Network Metrics - {description}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/network_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/network_metrics.png")

    # --- SAVE GRAPH DATA ---
    # Save edge list with weights
    nx.write_weighted_edgelist(G, f"{output_dir}/keyword_network_edges.txt")
    print(f"   Saved: {output_dir}/keyword_network_edges.txt")

    # Save graph in GraphML format (for other tools like Gephi)
    nx.write_graphml(G, f"{output_dir}/keyword_network.graphml")
    print(f"   Saved: {output_dir}/keyword_network.graphml")

    # Save all metrics to CSV
    print("\nSaving metrics to CSV files...")

    # Degree and strength metrics
    metrics_df = pd.DataFrame({
        'keyword': list(degrees.keys()),
        'degree': list(degrees.values()),
        'strength': [strength[k] for k in degrees.keys()],
        'betweenness': [betweenness[k] for k in degrees.keys()],
        'closeness': [closeness[k] for k in degrees.keys()],
        'clustering': [clustering[k] for k in degrees.keys()]
    })
    
    # Add community assignments if available
    if communities:
        node_to_community = {}
        for i, comm in enumerate(communities, 1):
            for node in comm:
                node_to_community[node] = i
        metrics_df['community'] = [node_to_community.get(k, 0) for k in degrees.keys()]
    
    metrics_df = metrics_df.sort_values('degree', ascending=False)
    metrics_df.to_csv(f'{output_dir}/node_metrics.csv', index=False)
    print(f"   Saved: {output_dir}/node_metrics.csv")

    # Community assignments
    if communities:
        community_data = []
        for i, comm in enumerate(communities, 1):
            for node in comm:
                community_data.append({
                    'keyword': node,
                    'community': i
                })
        community_df = pd.DataFrame(community_data).sort_values(['community', 'keyword'])
        community_df.to_csv(f'{output_dir}/community_assignments.csv', index=False)
        print(f"   Saved: {output_dir}/community_assignments.csv")
    else:
        print("   No communities to save")

    # Global metrics summary
    global_metrics = {
        'Metric': [
            'Number of nodes',
            'Number of edges',
            'Average degree',
            'Average strength',
            'Graph density',
            'Global clustering coefficient',
            'Average local clustering',
            'Number of communities',
            'Modularity',
            'Connected components',
            DIAMETER_LABEL,
            PATH_LABEL
        ],
        'Value': [
            G.number_of_nodes(),
            G.number_of_edges(),
            round(sum(degree_values) / len(degree_values), 3) if degree_values else 0,
            round(sum(strength_values) / len(strength_values), 3) if strength_values else 0,
            round(nx.density(G), 4),
            round(nx.transitivity(G), 4),
            round(avg_clustering, 4),
            len(communities) if communities else 0,
            round(MODULARITY, 4),
            nx.number_connected_components(G),
            DIAMETER_VAL,
            round(AVG_PATH_VAL, 3)
        ]
    }
    global_df = pd.DataFrame(global_metrics)
    global_df.to_csv(f'{output_dir}/global_metrics.csv', index=False)
    print(f"   Saved: {output_dir}/global_metrics.csv")

    # --- SUMMARY STATISTICS ---
    print("\n" + "="*80)
    print("NETWORK ANALYSIS SUMMARY")
    print("="*80)
    print("\nBASIC STRUCTURE:")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Density: {nx.density(G):.4f}")

    print("\nDEGREE METRICS:")
    print(f"   Average degree: {sum(degree_values) / len(degree_values):.3f}")
    print(f"   Average strength: {sum(strength_values) / len(strength_values):.3f}")

    print("\nCENTRALITY METRICS:")
    print(f"   Average betweenness: {sum(betweenness.values()) / len(betweenness):.4f}")
    if nx.is_connected(G):
        print(f"   Average closeness: {sum(closeness.values()) / len(closeness):.4f}")

    print("\nGLOBAL METRICS:")
    print(f"   Clustering coefficient: {nx.transitivity(G):.4f}")
    print(f"   Average local clustering: {avg_clustering:.4f}")
    if nx.is_connected(G):
        print(f"   Diameter: {DIAMETER_VAL}")
        print(f"   Average path length: {AVG_PATH_VAL:.3f}")
    else:
        print(f"   Diameter (largest component): {DIAMETER_VAL}")
        print(f"   Average path length (largest component): {AVG_PATH_VAL:.3f}")

    print("\nCOMMUNITY STRUCTURE:")
    print(f"   Communities: {len(communities) if communities else 0}")
    print(f"   Modularity: {MODULARITY:.4f}")

    print("\n" + "="*80)
    print(f"Analysis complete for: {description}")
    print("="*80)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING NETWORK ANALYSIS FOR ALL CONFIGURATIONS")
    print("="*80)
    
    for config_idx, analysis_config in enumerate(ANALYSIS_CONFIGS, 1):
        config_name = analysis_config["name"]
        keywords = analysis_config["keywords"]
        description = analysis_config["description"]
        
        # Define paths
        input_file = f"{OUTPUT_DIR}/{config_name}/bluesky_posts_complex.csv"
        output_dir = f"{OUTPUT_DIR}/{config_name}"
        
        # Run analysis
        analyze_network(keywords, input_file, output_dir, description)
    
    print("\n" + "="*80)
    print("ALL NETWORK ANALYSES COMPLETE!")
    print("="*80)
