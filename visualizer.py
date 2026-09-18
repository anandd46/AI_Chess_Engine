"""
visualizer.py
==========================================================================
Search-tree and evaluation visualization.

Two independent visualizations are produced:

1. Search Tree (search_tree.png)
   Renders the explored Minimax / Alpha-Beta tree using Graphviz,
   showing:
     - the current (root) node
     - children and leaf nodes
     - evaluation values at each node
     - the chosen principal-variation path (highlighted green)
     - pruned branches (shown greyed-out, dashed)
     - alpha / beta values (Alpha-Beta only)
     - depth

   Because real chess search trees have thousands to millions of nodes,
   we cap the number of RENDERED nodes (config.MAX_VISUALIZED_NODES) —
   the underlying search itself is never capped, only the drawing. We
   render breadth-first so the most structurally important (shallow)
   nodes are always included first.

2. Evaluation Graph (evaluation_graph.png)
   A matplotlib line chart of the evaluation score and nodes-expanded
   across the whole game, so trends are visible move-by-move.

networkx is used as the intermediate graph representation (nodes/edges
are added to an nx.DiGraph), which is then hand-translated into a
graphviz.Digraph for the actual PNG rendering — this satisfies the
project's requirement to use BOTH matplotlib+networkx AND graphviz.
==========================================================================
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # Headless-safe backend; GUI runs pygame, not matplotlib.
import matplotlib.pyplot as plt
import networkx as nx

try:
    import graphviz
    _GRAPHVIZ_AVAILABLE = True
except ImportError:
    _GRAPHVIZ_AVAILABLE = False

from config import MAX_VISUALIZED_NODES
from performance import PerformanceStats
from search import SearchNode, SearchTree
from utils import format_score


# ==========================================================================
# 1. SEARCH TREE VISUALIZATION
# ==========================================================================

def build_networkx_graph(tree: SearchTree, max_nodes: int = MAX_VISUALIZED_NODES) -> nx.DiGraph:
    """Convert a SearchTree into an nx.DiGraph, breadth-first, capped at
    `max_nodes` nodes for legibility. Node attributes carry everything
    the graphviz renderer needs (label text, color, style)."""
    graph = nx.DiGraph()
    if tree.root is None:
        return graph

    queue = deque([tree.root])
    visited = 0

    while queue and visited < max_nodes:
        node = queue.popleft()
        visited += 1

        label = _node_label(node)
        graph.add_node(
            node.node_id,
            label=label,
            pruned=node.is_pruned,
            on_pv=node.on_principal_variation,
            is_leaf=node.is_leaf,
            is_root=(node.move is None),
        )

        for child in node.children:
            graph.add_edge(node.node_id, child.node_id)
            queue.append(child)

    return graph


def _node_label(node: SearchNode) -> str:
    """Multi-line label text for one node in the rendered tree."""
    if node.move is None:
        move_text = "ROOT"
    else:
        move_text = node.move.uci()

    if node.is_pruned:
        return f"{move_text}\n[PRUNED]"

    value_text = format_score(node.value) if node.value is not None else "?"
    lines = [move_text, f"eval: {value_text}", f"depth: {node.depth}"]

    if node.alpha != float("-inf") or node.beta != float("inf"):
        alpha_text = "-inf" if node.alpha == float("-inf") else f"{node.alpha:.0f}"
        beta_text = "inf" if node.beta == float("inf") else f"{node.beta:.0f}"
        lines.append(f"a={alpha_text} b={beta_text}")

    return "\n".join(lines)


def render_search_tree_png(tree: SearchTree, output_path: Path) -> bool:
    """Render the search tree to a PNG file using Graphviz.

    Returns True on success, False if the graphviz system binary is not
    installed (the Python package can be installed via pip, but the
    actual `dot` renderer is a separate system dependency — see
    README.md for installation instructions per OS).
    """
    if not _GRAPHVIZ_AVAILABLE:
        return False

    nx_graph = build_networkx_graph(tree)
    if nx_graph.number_of_nodes() == 0:
        return False

    dot = graphviz.Digraph(
        comment="AI Chess Engine Search Tree",
        format="png",
        graph_attr={
            "bgcolor": "#121216",
            "rankdir": "TB",
            "splines": "line",
            "nodesep": "0.25",
            "ranksep": "0.6",
            "size": "16,16",
            "ratio": "compress",
        },
        node_attr={
            "shape": "box",
            "style": "filled,rounded",
            "fontname": "Helvetica",
            "fontsize": "10",
            "fontcolor": "white",
        },
        edge_attr={"color": "#666677"},
    )

    for node_id, attrs in nx_graph.nodes(data=True):
        if attrs.get("pruned"):
            fill = "#3a3a3a"
            pen_color = "#888888"
            style = "filled,dashed,rounded"
        elif attrs.get("on_pv"):
            fill = "#2e7d4f"
            pen_color = "#7fffa0"
            style = "filled,bold,rounded"
        elif attrs.get("is_root"):
            fill = "#2b5b8c"
            pen_color = "#7fc4ff"
            style = "filled,bold,rounded"
        elif attrs.get("is_leaf"):
            fill = "#3d3d4d"
            pen_color = "#9999aa"
            style = "filled,rounded"
        else:
            fill = "#2a2a35"
            pen_color = "#666677"
            style = "filled,rounded"

        dot.node(
            str(node_id),
            label=attrs.get("label", ""),
            fillcolor=fill,
            color=pen_color,
            style=style,
        )

    for source, target in nx_graph.edges():
        dot.edge(str(source), str(target))

    rendered_path = dot.render(filename=output_path.stem, directory=str(output_path.parent),
                                cleanup=True)
    # graphviz appends its own extension; normalize to the requested path.
    rendered = Path(rendered_path)
    if rendered != output_path and rendered.exists():
        rendered.replace(output_path)

    return True


# ==========================================================================
# 2. EVALUATION / PERFORMANCE GRAPH (matplotlib)
# ==========================================================================

def plot_evaluation_graph(history: List[PerformanceStats], output_path: Path) -> bool:
    """Render a two-panel matplotlib chart:
        (top)    evaluation score over the course of the game.
        (bottom) nodes expanded per move over the course of the game.

    Returns True if a chart was written, False if there is no history
    yet to plot.
    """
    if not history:
        return False

    plies = [s.ply_number for s in history]
    evals = [s.evaluation_score / 100.0 for s in history]  # convert to pawns
    nodes = [s.nodes_expanded for s in history]
    colors = ["#5aa8dc" if s.algorithm == "Minimax" else "#5adc8f" for s in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.patch.set_facecolor("#121216")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1c1c22")
        ax.tick_params(colors="#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#444455")

    ax1.plot(plies, evals, color="#e0b060", linewidth=1.5, marker="o", markersize=3)
    ax1.axhline(0, color="#777788", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Evaluation (pawns)", color="#dddddd")
    ax1.set_title("Board Evaluation & Search Cost Over the Game",
                   color="#ffffff", fontsize=13)

    ax2.bar(plies, nodes, color=colors)
    ax2.set_ylabel("Nodes Expanded", color="#dddddd")
    ax2.set_xlabel("Ply", color="#dddddd")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True
