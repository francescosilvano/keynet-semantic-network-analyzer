"""Network analysis module for keyword co-occurrence graphs (packaged)
"""

import os
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community

# --- IMPORT SHARED CONFIGURATION ---
from .config import (
    ANALYSIS_CONFIGS,
    MIN_CO_OCCURRENCES,
    OUTPUT_DIR,
    ARCHIVE_ENABLED,
    ARCHIVE_DIR
)
from .archive import RunArchive, get_latest_run


def analyze_network(keywords, input_file, output_dir, description, archive=None):
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
    archive : RunArchive, optional
        Archive object for tracking files
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

    # --- KEYWORD SENTIMENT AGGREGATION ---
    sentiment_rows = []
    for kw in keywords:
        mask = df["text"].str.contains(kw, case=False, na=False)
        subset = df[mask]

        n_posts = len(subset)
        counts = subset["sentiment"].value_counts()
        n_positive = counts.get("positive", 0)
        n_neutral = counts.get("neutral", 0)
        n_negative = counts.get("negative", 0)

        if n_posts > 0:
            share_positive = n_positive / n_posts
            share_neutral = n_neutral / n_posts
            share_negative = n_negative / n_posts
            mean_score = subset["score"].mean()
            median_score = subset["score"].median()
        else:
            share_positive = 0
            share_neutral = 0
            share_negative = 0
            mean_score = float("nan")
            median_score = float("nan")

        sentiment_rows.append({
            "keyword": kw,
            "n_posts": n_posts,
            "n_positive": n_positive,
            "n_neutral": n_neutral,
            "n_negative": n_negative,
            "share_positive": share_positive,
            "share_neutral": share_neutral,
            "share_negative": share_negative,
            "mean_score": mean_score,
            "median_score": median_score
        })

    sentiment_df = pd.DataFrame(sentiment_rows)
    sentiment_df.to_csv(f"{output_dir}/keyword_sentiment.csv", index=False)
    print(f"   Saved: {output_dir}/keyword_sentiment.csv")

    plot_df = sentiment_df.sort_values("n_posts", ascending=False)
    fig = plt.figure(figsize=(12, 8))
    plt.bar(plot_df["keyword"], plot_df["share_positive"],
            label="positive")
    plt.bar(plot_df["keyword"], plot_df["share_neutral"],
            bottom=plot_df["share_positive"], label="neutral")
    plt.bar(plot_df["keyword"], plot_df["share_negative"],
            bottom=plot_df["share_positive"] + plot_df["share_neutral"],
            label="negative")
    plt.ylabel("Share")
    plt.title(f"Keyword sentiment distribution - {description}")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/keyword_sentiment_stacked.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {output_dir}/keyword_sentiment_stacked.png")

    # --- COMPUTE KEYWORD CO-OCCURRENCES ---
    print("\nAnalyzing keyword co-occurrences...")
    co_occurrence_data = []

    for kw1, kw2 in combinations(keywords, 2):
        mask = (df.text.str.contains(kw1, case=False, na=False) &
                df.text.str.contains(kw2, case=False, na=False))
        count = mask.sum()

        if count >= MIN_CO_OCCURRENCES:
            co_occurrence_data.append({
                'w1': kw1,
                'w2': kw2,
                'count': count
            })

    co_df = pd.DataFrame(co_occurrence_data)
    print(f"\nFound {len(co_df)} keyword pairs with co-occurrences")

    if len(co_df) == 0:
        print("WARNING: No co-occurrences found. Skipping network analysis.")
        return

    # --- CO-OCCURRENCE HEATMAP ---
    co_matrix = pd.DataFrame(0, index=keywords, columns=keywords, dtype=float)
    for _, row in co_df.iterrows():
        co_matrix.at[row['w1'], row['w2']] = row['count']
        co_matrix.at[row['w2'], row['w1']] = row['count']
    np.fill_diagonal(co_matrix.values, 0)

    co_matrix_file = f"{output_dir}/keyword_cooccurrence_matrix.csv"
    co_matrix.to_csv(co_matrix_file)
    print(f"   Saved: {co_matrix_file}")
    if archive:
        rel_path = os.path.basename(output_dir) + "/keyword_cooccurrence_matrix.csv"
        archive.add_file(rel_path)

    heatmap_values = np.log1p(co_matrix.values)
    plt.figure(figsize=(12, 10))
    plt.imshow(heatmap_values, cmap="viridis", aspect="auto")
    plt.colorbar(label="Co-occurrence count")
    plt.title(f"Keyword Co-occurrence Heatmap (log1p) - {description}")
    plt.xticks(ticks=range(len(keywords)), labels=keywords, rotation=45, ha='right')
    plt.yticks(ticks=range(len(keywords)), labels=keywords)
    plt.tight_layout()
    heatmap_file = f"{output_dir}/keyword_cooccurrence_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {heatmap_file}")
    if archive:
        rel_path = os.path.basename(output_dir) + "/keyword_cooccurrence_heatmap.png"
        archive.add_file(rel_path)

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
    for _, _, data in G.edges(data=True):
        weight = data.get('weight', 1)
        data['distance'] = 1 / weight if weight else 0

    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")

    # --- GRAPH ANALYSIS ---
    print("\nGraph metrics:")

    degrees = dict(G.degree())
    degree_values = list(degrees.values())

    print("\nDEGREE DISTRIBUTION:")
    print(f"   Average degree: {sum(degree_values) / len(degree_values):.3f}")
    print(f"   Min degree: {min(degree_values)}")
    print(f"   Max degree: {max(degree_values)}")
    print("   Top keywords by connections:")
    for keyword, degree in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {degree}")

    strength = dict(G.degree(weight='weight'))
    strength_values = list(strength.values())
    print("\nSTRENGTH DISTRIBUTION (weighted degree):")
    print(f"   Average strength: {sum(strength_values) / len(strength_values):.3f}")
    print(f"   Min strength: {min(strength_values)}")
    print(f"   Max strength: {max(strength_values)}")
    print("   Top keywords by strength:")
    for keyword, strn in sorted(strength.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {strn}")

    print("\nBETWEENNESS CENTRALITY:")
    betweenness = nx.betweenness_centrality(G, weight='distance')
    print(f"   Average betweenness: {sum(betweenness.values()) / len(betweenness):.4f}")
    print("   Top keywords by betweenness:")
    for keyword, bc in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {bc:.4f}")

    print("\nCLOSENESS CENTRALITY:")
    if nx.is_connected(G):
        closeness = nx.closeness_centrality(G, distance='distance')
        print(f"   Average closeness: {sum(closeness.values()) / len(closeness):.4f}")
        print("   Top keywords by closeness:")
        for keyword, cc in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {keyword}: {cc:.4f}")
    else:
        print("   Graph is disconnected - calculating closeness for largest component")
        largest_cc = max(nx.connected_components(G), key=len)
        g_largest = G.subgraph(largest_cc).copy()
        closeness = nx.closeness_centrality(g_largest, distance='distance')
        closeness_full = {node: closeness.get(node, 0) for node in G.nodes()}
        closeness = closeness_full
        avg_closeness = sum(closeness.values()) / len(closeness)
        print(f"   Average closeness (largest component): {avg_closeness:.4f}")
        print("   Top keywords by closeness:")
        for keyword, cc in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {keyword}: {cc:.4f}")

    print("\nGLOBAL MEASURES:")
    print(f"   Graph density: {nx.density(G):.4f}")
    print(f"   Global clustering coefficient: {nx.transitivity(G):.4f}")
    print(f"   Connected components: {nx.number_connected_components(G)}")

    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"   Diameter: {diameter}")
        print(f"   Average path length: {avg_path_length:.3f}")
        diameter_label = "Diameter"
        diameter_val = diameter
        path_label = "Average path length"
        avg_path_val = avg_path_length
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        g_largest = G.subgraph(largest_cc).copy()
        diameter = nx.diameter(g_largest)
        avg_path_length = nx.average_shortest_path_length(g_largest)
        print(f"   Diameter (largest component): {diameter}")
        print(f"   Average path length (largest component): {avg_path_length:.3f}")
        diameter_label = "Diameter (largest component)"
        diameter_val = diameter
        path_label = "Average path length (largest component)"
        avg_path_val = avg_path_length

    print("\nLOCAL CLUSTERING:")
    clustering = nx.clustering(G)
    avg_clustering = sum(clustering.values()) / len(clustering)
    print(f"   Average local clustering: {avg_clustering:.4f}")
    print("   Top keywords by local clustering:")
    for keyword, clust in sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {clust:.4f}")

    print("\nCOMMUNITY DETECTION:")
    if G.number_of_edges() > 0:
        communities = list(community.greedy_modularity_communities(G, weight='weight'))
        modularity_score = community.modularity(G, communities, weight='weight')
        print(f"   Number of communities: {len(communities)}")
        print(f"   Modularity: {modularity_score:.4f}")
        print("\n   Community assignments:")
        for i, comm in enumerate(communities, 1):
            print(f"      Community {i} ({len(comm)} keywords): {', '.join(sorted(comm))}")
    else:
        communities = []
        modularity_score = 0
        print("   No edges - no communities detected")

    node_to_community = {}
    if communities:
        for i, comm in enumerate(communities, 1):
            for node in comm:
                node_to_community[node] = i

    print("\nCreating visualizations...")

    TOP_N_EDGE_LABELS = 30

    plt.figure(figsize=(16, 12))

    node_sizes = [degrees[node] * 300 + 500 for node in G.nodes()]

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 + (w / max_weight) * 8 for w in edge_weights]

    cmap = plt.cm.tab20
    if communities:
        node_colors = [cmap((node_to_community.get(node, 1) - 1) % cmap.N) for node in G.nodes()]
        circular_node_colors = node_colors
    else:
        node_colors = ['lightblue' for _ in G.nodes()]
        circular_node_colors = ['lightcoral' for _ in G.nodes()]

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors=node_colors,
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

    edge_labels = nx.get_edge_attributes(G, 'weight')
    sorted_weights = sorted(edge_labels.values(), reverse=True)
    if sorted_weights:
        threshold_idx = min(TOP_N_EDGE_LABELS, len(sorted_weights)) - 1
        min_label_weight = sorted_weights[threshold_idx]
        filtered_edge_labels = {edge: w for edge, w in edge_labels.items()
                                if w >= min_label_weight}
    else:
        min_label_weight = 0
        filtered_edge_labels = {}

    nx.draw_networkx_edge_labels(G, pos, filtered_edge_labels, font_size=8)

    title_text = (f"Keyword Co-occurrence Network - {description}\n"
                  "(Node size = connections, Edge width = co-occurrences)\n"
                  f"(Edge labels shown for top {TOP_N_EDGE_LABELS} edges; weight ≥ {min_label_weight})")
    plt.title(title_text,
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    network_file = f'{output_dir}/keyword_network.png'
    plt.savefig(network_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {network_file}")
    if archive:
        rel_path = os.path.basename(output_dir) + "/keyword_network.png"
        archive.add_file(rel_path)

    # End of function (clean)

    # Add nodes (keywords)
    G.add_nodes_from(keywords)

    # Add edges (co-occurrences) with weights
    edges = []
    for _, row in co_df.iterrows():
        edges.append((row['w1'], row['w2'], {'weight': row['count']}))

    G.add_edges_from(edges)
    for _, _, data in G.edges(data=True):
        weight = data.get('weight', 1)
        data['distance'] = 1 / weight if weight else 0

    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")

    # --- GRAPH ANALYSIS ---
    print("\nGraph metrics:")

    degrees = dict(G.degree())
    degree_values = list(degrees.values())

    print("\nDEGREE DISTRIBUTION:")
    print(f"   Average degree: {sum(degree_values) / len(degree_values):.3f}")
    print(f"   Min degree: {min(degree_values)}")
    print(f"   Max degree: {max(degree_values)}")
    print("   Top keywords by connections:")
    for keyword, degree in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {degree}")

    strength = dict(G.degree(weight='weight'))
    strength_values = list(strength.values())
    print("\nSTRENGTH DISTRIBUTION (weighted degree):")
    print(f"   Average strength: {sum(strength_values) / len(strength_values):.3f}")
    print(f"   Min strength: {min(strength_values)}")
    print(f"   Max strength: {max(strength_values)}")
    print("   Top keywords by strength:")
    for keyword, strn in sorted(strength.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {strn}")

    print("\nBETWEENNESS CENTRALITY:")
    betweenness = nx.betweenness_centrality(G, weight='distance')
    print(f"   Average betweenness: {sum(betweenness.values()) / len(betweenness):.4f}")
    print("   Top keywords by betweenness:")
    for keyword, bc in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {bc:.4f}")

    print("\nCLOSENESS CENTRALITY:")
    if nx.is_connected(G):
        closeness = nx.closeness_centrality(G, distance='distance')
        print(f"   Average closeness: {sum(closeness.values()) / len(closeness):.4f}")
        print("   Top keywords by closeness:")
        for keyword, cc in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {keyword}: {cc:.4f}")
    else:
        print("   Graph is disconnected - calculating closeness for largest component")
        largest_cc = max(nx.connected_components(G), key=len)
        g_largest = G.subgraph(largest_cc).copy()
        closeness = nx.closeness_centrality(g_largest, distance='distance')
        closeness_full = {node: closeness.get(node, 0) for node in G.nodes()}
        closeness = closeness_full
        avg_closeness = sum(closeness.values()) / len(closeness)
        print(f"   Average closeness (largest component): {avg_closeness:.4f}")
        print("   Top keywords by closeness:")
        for keyword, cc in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {keyword}: {cc:.4f}")

    print("\nGLOBAL MEASURES:")
    print(f"   Graph density: {nx.density(G):.4f}")
    print(f"   Global clustering coefficient: {nx.transitivity(G):.4f}")
    print(f"   Connected components: {nx.number_connected_components(G)}")

    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"   Diameter: {diameter}")
        print(f"   Average path length: {avg_path_length:.3f}")
        diameter_label = "Diameter"
        diameter_val = diameter
        path_label = "Average path length"
        avg_path_val = avg_path_length
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        g_largest = G.subgraph(largest_cc).copy()
        diameter = nx.diameter(g_largest)
        avg_path_length = nx.average_shortest_path_length(g_largest)
        print(f"   Diameter (largest component): {diameter}")
        print(f"   Average path length (largest component): {avg_path_length:.3f}")
        diameter_label = "Diameter (largest component)"
        diameter_val = diameter
        path_label = "Average path length (largest component)"
        avg_path_val = avg_path_length

    print("\nLOCAL CLUSTERING:")
    clustering = nx.clustering(G)
    avg_clustering = sum(clustering.values()) / len(clustering)
    print(f"   Average local clustering: {avg_clustering:.4f}")
    print("   Top keywords by local clustering:")
    for keyword, clust in sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {clust:.4f}")

    print("\nCOMMUNITY DETECTION:")
    if G.number_of_edges() > 0:
        communities = list(community.greedy_modularity_communities(G, weight='weight'))
        modularity_score = community.modularity(G, communities, weight='weight')
        print(f"   Number of communities: {len(communities)}")
        print(f"   Modularity: {modularity_score:.4f}")
        print("\n   Community assignments:")
        for i, comm in enumerate(communities, 1):
            print(f"      Community {i} ({len(comm)} keywords): {', '.join(sorted(comm))}")
    else:
        communities = []
        modularity_score = 0
        print("   No edges - no communities detected")

    node_to_community = {}
    if communities:
        for i, comm in enumerate(communities, 1):
            for node in comm:
                node_to_community[node] = i

    print("\nCreating visualizations...")

    TOP_N_EDGE_LABELS = 30

    plt.figure(figsize=(16, 12))

    node_sizes = [degrees[node] * 300 + 500 for node in G.nodes()]

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 + (w / max_weight) * 8 for w in edge_weights]

    cmap = plt.cm.tab20
    if communities:
        node_colors = [cmap((node_to_community.get(node, 1) - 1) % cmap.N) for node in G.nodes()]
        circular_node_colors = node_colors
    else:
        node_colors = ['lightblue' for _ in G.nodes()]
        circular_node_colors = ['lightcoral' for _ in G.nodes()]

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors=node_colors,
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

    edge_labels = nx.get_edge_attributes(G, 'weight')
    sorted_weights = sorted(edge_labels.values(), reverse=True)
    if sorted_weights:
        threshold_idx = min(TOP_N_EDGE_LABELS, len(sorted_weights)) - 1
        min_label_weight = sorted_weights[threshold_idx]
        filtered_edge_labels = {edge: w for edge, w in edge_labels.items()
                                if w >= min_label_weight}
    else:
        min_label_weight = 0
        filtered_edge_labels = {}

    nx.draw_networkx_edge_labels(G, pos, filtered_edge_labels, font_size=8)

    title_text = (f"Keyword Co-occurrence Network - {description}\n"
                  "(Node size = connections, Edge width = co-occurrences)\n"
                  f"(Edge labels shown for top {TOP_N_EDGE_LABELS} edges; weight ≥ {min_label_weight})")
    plt.title(title_text,
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    network_file = f'{output_dir}/keyword_network.png'
    plt.savefig(network_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {network_file}")
    if archive:
        rel_path = os.path.basename(output_dir) + "/keyword_network.png"
        archive.add_file(rel_path)

    # Figure 2: Circular layout
    plt.figure(figsize=(14, 14))

    pos_circular = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos_circular,
                           node_size=node_sizes,
                           node_color=circular_node_colors,
                           edgecolors=circular_node_colors,
                           linewidths=2,
                           alpha=0.9)

    nx.draw_networkx_edges(G, pos_circular,
                           width=edge_widths,
                           alpha=0.5,
                           edge_color='gray')

    nx.draw_networkx_labels(G, pos_circular,
                            font_size=11,
                            font_weight='bold')

    nx.draw_networkx_edge_labels(G, pos_circular, filtered_edge_labels,
                                font_size=8, label_pos=0.6)

    plt.title(
        f"Keyword Co-occurrence Network (Circular Layout) - {description}\n"
        f"(Edge labels shown for top {TOP_N_EDGE_LABELS} edges; weight ≥ {min_label_weight})",
        fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    circular_file = f'{output_dir}/keyword_network_circular.png'
    plt.savefig(circular_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {circular_file}")
    if archive:
        rel_path = os.path.basename(output_dir) + "/keyword_network_circular.png"
        archive.add_file(rel_path)

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
    metrics_file = f'{output_dir}/network_metrics.png'
    plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {metrics_file}")
    if archive:
        rel_path = os.path.basename(output_dir) + "/network_metrics.png"
        archive.add_file(rel_path)

    # --- SAVE GRAPH DATA ---
    # Save edge list with weights
    edges_file = f"{output_dir}/keyword_network_edges.txt"
    nx.write_weighted_edgelist(G, edges_file)
    print(f"   Saved: {edges_file}")
    if archive:
        rel_path = os.path.basename(output_dir) + "/keyword_network_edges.txt"
        archive.add_file(rel_path)

    # Save graph in GraphML format (for other tools like Gephi)
    graphml_file = f"{output_dir}/keyword_network.graphml"
    nx.write_graphml(G, graphml_file)
    print(f"   Saved: {graphml_file}")
    if archive:
        rel_path = os.path.basename(output_dir) + "/keyword_network.graphml"
        archive.add_file(rel_path)

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

    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"   Diameter: {diameter}")
        print(f"   Average path length: {avg_path_length:.3f}")
        diameter_label = "Diameter"
        diameter_val = diameter
        path_label = "Average path length"
        avg_path_val = avg_path_length
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        g_largest = G.subgraph(largest_cc).copy()
        diameter = nx.diameter(g_largest)
        avg_path_length = nx.average_shortest_path_length(g_largest)
        print(f"   Diameter (largest component): {diameter}")
        print(f"   Average path length (largest component): {avg_path_length:.3f}")
        diameter_label = "Diameter (largest component)"
        diameter_val = diameter
        path_label = "Average path length (largest component)"
        avg_path_val = avg_path_length

    print("\nLOCAL CLUSTERING:")
    clustering = nx.clustering(G)
    avg_clustering = sum(clustering.values()) / len(clustering)
    print(f"   Average local clustering: {avg_clustering:.4f}")
    print("   Top keywords by local clustering:")
    for keyword, clust in sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {keyword}: {clust:.4f}")

    print("\nCOMMUNITY DETECTION:")
    if G.number_of_edges() > 0:
        communities = list(community.greedy_modularity_communities(G, weight='weight'))
        modularity_score = community.modularity(G, communities, weight='weight')
        print(f"   Number of communities: {len(communities)}")
        print(f"   Modularity: {modularity_score:.4f}")
        print("\n   Community assignments:")
        for i, comm in enumerate(communities, 1):
            print(f"      Community {i} ({len(comm)} keywords): {', '.join(sorted(comm))}")
    else:
        communities = []
        modularity_score = 0
        print("   No edges - no communities detected")

    node_to_community = {}
    if communities:
        for i, comm in enumerate(communities, 1):
            for node in comm:
                node_to_community[node] = i

    print("\nCreating visualizations...")

    TOP_N_EDGE_LABELS = 30

    plt.figure(figsize=(16, 12))

    node_sizes = [degrees[node] * 300 + 500 for node in G.nodes()]

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 + (w / max_weight) * 8 for w in edge_weights]

    cmap = plt.cm.tab20
    if communities:
        node_colors = [cmap((node_to_community.get(node, 1) - 1) % cmap.N) for node in G.nodes()]
        circular_node_colors = node_colors
    else:
        node_colors = ['lightblue' for _ in G.nodes()]
        circular_node_colors = ['lightcoral' for _ in G.nodes()]

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors=node_colors,
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

    edge_labels = nx.get_edge_attributes(G, 'weight')
    sorted_weights = sorted(edge_labels.values(), reverse=True)
    if sorted_weights:
        threshold_idx = min(TOP_N_EDGE_LABELS, len(sorted_weights)) - 1
        min_label_weight = sorted_weights[threshold_idx]
        filtered_edge_labels = {edge: w for edge, w in edge_labels.items()
                                if w >= min_label_weight}
    else:
        min_label_weight = 0
        filtered_edge_labels = {}

    nx.draw_networkx_edge_labels(G, pos, filtered_edge_labels, font_size=8)

    title_text = (f"Keyword Co-occurrence Network - {description}\n"
                  "(Node size = connections, Edge width = co-occurrences)\n"
                  f"(Edge labels shown for top {TOP_N_EDGE_LABELS} edges; weight ≥ {min_label_weight})")
    plt.title(title_text,
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    network_file = f'{output_dir}/keyword_network.png'
    plt.savefig(network_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {network_file}")
    if archive:
        rel_path = os.path.basename(output_dir) + "/keyword_network.png"
        archive.add_file(rel_path)

