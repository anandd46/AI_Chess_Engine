"""
minimax.py
==========================================================================
Manual implementation of the Minimax algorithm for two-player,
zero-sum, perfect-information games (chess).

Minimax explores the game tree via Depth-First Search: at each node the
side to move either MAXIMIZES the evaluation (White) or MINIMIZES it
(Black), recursively, until a fixed depth or a terminal position
(checkmate/stalemate) is reached. The value of a leaf is given by
`evaluation.evaluate()`. Values are then backed up ("backtracked") from
leaves to the root, one level at a time.

This file intentionally implements NO pruning — that is alphabeta.py's
job — so the two algorithms can be compared head-to-head on identical
positions (same evaluation function, same move ordering disabled) to
demonstrate the raw benefit Alpha-Beta Pruning provides.

Complexity
----------
Time:  O(b^d)   where b = branching factor (~35 for chess), d = depth.
Space: O(d)     for the recursion stack (O(b*d) if the full tree is
                 retained in memory for visualization, as we do here).
==========================================================================
"""

from __future__ import annotations

from typing import Optional, Tuple

import chess

from evaluation import evaluate
from search import SearchNode, SearchTree


class MinimaxAI:
    """Minimax search engine (no pruning)."""

    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        self.tree: SearchTree = SearchTree()

    def search(self, board: chess.Board) -> Tuple[Optional[chess.Move], float, SearchTree]:
        """Run Minimax from the given board position (current position,
        NOT yet the AI's move) down to `self.max_depth` plies.

        Returns
        -------
        (best_move, best_value, search_tree)
            best_move  : the move Minimax selected (None if no legal
                          moves exist, i.e. the game is already over).
            best_value : the backed-up evaluation of that move.
            search_tree: the full explored tree, for visualization and
                          statistics.
        """
        self.tree = SearchTree()
        maximizing = board.turn == chess.WHITE

        root = self.tree.new_node(
            move=None, depth=self.max_depth, is_maximizing=maximizing,
            alpha=float("-inf"), beta=float("inf"),
        )
        self.tree.set_root(root)

        value = self._minimax(board, self.max_depth, maximizing, root)

        best_child = next((c for c in root.children if c.on_principal_variation), None)
        best_move = best_child.move if best_child is not None else None

        return best_move, value, self.tree

    # ------------------------------------------------------------------
    # Core recursive Depth-First Search
    # ------------------------------------------------------------------

    def _minimax(
        self,
        board: chess.Board,
        depth: int,
        maximizing: bool,
        node: SearchNode,
    ) -> float:
        """Recursive Minimax DFS.

        Parameters
        ----------
        board : chess.Board
            Current position (mutated in place via push/pop for speed,
            then restored — this is standard practice for game-tree
            search and is far faster than copying the board at every
            node).
        depth : int
            Remaining plies to search.
        maximizing : bool
            True if the side to move at THIS node is the maximizer
            (White).
        node : SearchNode
            The tree node corresponding to this call, to be filled in
            with the backed-up value and, at the end, one child flagged
            as the principal variation.
        """
        self.tree.nodes_expanded += 1

        # --- Terminal detection -----------------------------------------
        if depth == 0 or board.is_game_over():
            node.is_leaf = True
            node.value = evaluate(board)
            return node.value

        legal_moves = list(board.legal_moves)

        if maximizing:
            best_value = float("-inf")
            best_child: Optional[SearchNode] = None

            for move in legal_moves:
                board.push(move)
                child = self.tree.new_node(
                    move=move, depth=depth - 1, is_maximizing=not maximizing,
                    alpha=float("-inf"), beta=float("inf"),
                )
                node.add_child(child)

                child_value = self._minimax(board, depth - 1, False, child)
                board.pop()

                if child_value > best_value:
                    best_value = child_value
                    best_child = child

            node.value = best_value
        else:
            best_value = float("inf")
            best_child = None

            for move in legal_moves:
                board.push(move)
                child = self.tree.new_node(
                    move=move, depth=depth - 1, is_maximizing=not maximizing,
                    alpha=float("-inf"), beta=float("inf"),
                )
                node.add_child(child)

                child_value = self._minimax(board, depth - 1, True, child)
                board.pop()

                if child_value < best_value:
                    best_value = child_value
                    best_child = child

            node.value = best_value

        # --- Backtracking: mark the winning child as part of the PV -----
        if best_child is not None:
            best_child.on_principal_variation = True

        return node.value
