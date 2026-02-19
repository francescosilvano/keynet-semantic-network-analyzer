import matplotlib
matplotlib.use("Agg")  # ensure headless backend for CI

import os
import pandas as pd
from keynet.graph import analyze_network


def make_input_csv(path):
    rows = [
        {"text": "CO2 emissions and renewable energy are linked", "sentiment": "neutral", "score": 0.0},
        {"text": "Emissions cause global warming", "sentiment": "negative", "score": -0.2},
        {"text": "Renewable energy reduces CO2 and emissions", "sentiment": "positive", "score": 0.6},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_analyze_network_creates_outputs(tmp_path):
    inp = tmp_path / "input.csv"
    outdir = tmp_path / "out"
    make_input_csv(inp)

    keywords = ["CO2", "emissions", "renewable energy"]
    analyze_network(keywords, str(inp), str(outdir), "test")

    # expected output files
    expected = [
        "keyword_cooccurrence_matrix.csv",
        "keyword_cooccurrence_heatmap.png",
        "keyword_network.png",
        "keyword_network_circular.png",
        "node_metrics.csv",
        "global_metrics.csv",
    ]

    for fn in expected:
        assert (outdir / fn).exists(), f"Missing output file: {fn}"

    # check node_metrics contains our keywords
    nm = pd.read_csv(outdir / "node_metrics.csv")
    names = {s.lower() for s in nm['keyword'].astype(str)}
    for kw in keywords:
        assert kw.lower() in names

    # simple global metric check
    gm = pd.read_csv(outdir / "global_metrics.csv")
    num_nodes = int(gm.loc[gm['Metric'] == 'Number of nodes', 'Value'].iloc[0])
    assert num_nodes == 3
