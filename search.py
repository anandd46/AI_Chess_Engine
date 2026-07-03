"""
search.py
==========================================================================
Shared search-tree data structures used by BOTH minimax.py and
alphabeta.py.

Keeping the tree/node representation in one common module means:
- minimax.py and alphabeta.py stay focused purely on *algorithm logic*.
- visualizer.py has a single, consistent data structure to render,
  regardless of which algorithm produced it.
- performance.py and the GUI can compare Minimax vs Alpha-Beta trees
  built from an identical schema.

Classes
-------
SearchNode
    One node in the explored game-tree: a board position reached by a
    specific move, along with the search metadata attached to it
    (depth, alpha/beta at time of visit, evaluation value, whether it
    was pruned, whether it lies on the chosen principal variation).

SearchTree
    Wraps the root SearchNode plus tree-wide bookkeeping (node counters,
    principal variation extraction).
==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import chess


@dataclass
class SearchNode:
    """A single node explored during Minimax / Alpha-Beta search.

    Attributes
    ----------
    move : Optional[chess.Move]
        The move that led to this position from its parent (None for
        the root node, which represents the current actual board).
    depth : int
        Remaining search depth AT this node (0 = leaf / terminal).
    value : Optional[float]
        The minimax value backed up to this node once fully evaluated.
    alpha, beta : float
        Alpha/beta bounds active when this node was visited (only
        meaningful for Alpha-Beta search; Minimax nodes just keep the
        initial -inf/+inf bounds for a consistent schema).
    is_maximizing : bool
        True if this node's value is computed by taking the MAX of its
        children (White to move), False if MIN (Black to move).
    is_leaf : bool
        True if this node was evaluated directly (depth 0 or terminal
        position) rather than expanded further.
    is_pruned : bool
        True if this node's *siblings* caused this branch to be cut off
        before it was ever explored (Alpha-Beta only). A pruned node has
        no children and no value.
    on_principal_variation : bool
        True if this node lies on the best line of play found by the
        search (used to highlight the chosen path in the visualization).
    children : List[SearchNode]
        Child nodes (one per move considered from this position).
    node_id : int
        Unique incrementing id, assigned by SearchTree, used as the
        node identifier in the exported Graphviz graph.
    """

    move: Optional[chess.Move] = None
    depth: int = 0
    value: Optional[float] = None
    alpha: float = float("-inf")
    beta: float = float("inf")
    is_maximizing: bool = True
    is_leaf: bool = False
    is_pruned: bool = False
    on_principal_variation: bool = False
    children: List["SearchNode"] = field(default_factory=list)
    node_id: int = -1

    def add_child(self, child: "SearchNode") -> None:
        self.children.append(child)

    def move_label(self) -> str:
        """Short label for the move that produced this node (used in the
        exported search tree image)."""
        return self.move.uci() if self.move is not None else "root"


class SearchTree:
    """Container for a full search-tree, built during one AI move
    decision. Provides node-id assignment and principal-variation
    extraction."""

    def __init__(self) -> None:
        self.root: Optional[SearchNode] = None
        self._next_id: int = 0
        self.nodes_expanded: int = 0
        self.nodes_pruned: int = 0

    def new_node(self, **kwargs) -> SearchNode:
        """Factory for a new SearchNode with a unique id, tracked by
        this tree. Both minimax.py and alphabeta.py should create nodes
        exclusively through this method so counters stay accurate."""
        node = SearchNode(node_id=self._next_id, **kwargs)
        self._next_id += 1
        return node

    def set_root(self, node: SearchNode) -> None:
        self.root = node

    def principal_variation(self) -> List[chess.Move]:
        """Walk from the root, always following the child marked
        `on_principal_variation`, to reconstruct the best line of play
        the search found (used for the GUI's 'Show principal variation'
        display)."""
        pv: List[chess.Move] = []
        node = self.root
        while node is not None:
            next_node = next((c for c in node.children if c.on_principal_variation), None)
            if next_node is None or next_node.move is None:
                break
            pv.append(next_node.move)
            node = next_node
        return pv

    def total_node_count(self) -> int:
        """Count of all nodes actually created (expanded or leaf),
        EXCLUDING pruned placeholder nodes, by walking the tree."""
        if self.root is None:
            return 0

        count = 0

        def _walk(n: SearchNode) -> None:
            nonlocal count
            count += 1
            for c in n.children:
                _walk(c)

        _walk(self.root)
        return count
