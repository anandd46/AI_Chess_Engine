"""
alphabeta.py
==========================================================================
Manual implementation of Minimax search enhanced with Alpha-Beta
Pruning.

Alpha-Beta Pruning explores the exact same game tree as plain Minimax
and is GUARANTEED to produce the identical best move and value — but it
skips ("prunes") branches that cannot possibly influence the final
decision, because a better alternative has already been found elsewhere
in the tree. In the best case (with good move ordering) this reduces
the effective branching factor from b to roughly sqrt(b), letting the
engine search roughly twice as deep in the same time budget.

Two bounds are tracked while descending the tree:
    alpha : the best value the MAXIMIZING player can guarantee so far.
    beta  : the best value the MINIMIZING player can guarantee so far.

If at any point alpha >= beta, the current node's remaining siblings
cannot change the outcome the parent will choose, so we stop searching
them ("beta cutoff" / "alpha cutoff") — this is the prune.

Move Ordering
-------------
Pruning effectiveness depends heavily on search order: if the best move
is examined FIRST, far more subsequent branches get pruned. We apply a
cheap, standard move-ordering heuristic before recursing:
    1. Captures, ordered by MVV-LVA (Most Valuable Victim, Least
       Valuable Attacker) — try queen-takes-pawn last, pawn-takes-queen
       first.
    2. Checks.
    3. All other (quiet) moves.

Complexity
----------
Time:  O(b^d) worst case (no pruning happens), O(b^(d/2)) best case
       with optimal move ordering.
Space: O(d) for the recursion stack (O(b*d) if retaining the full tree
       for visualization, as this project does).
==========================================================================
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import chess

from evaluation import PIECE_VALUES, evaluate
from search import SearchNode, SearchTree


class AlphaBetaAI:
    """Minimax search engine WITH Alpha-Beta Pruning and move ordering."""

    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        self.tree: SearchTree = SearchTree()

    def search(self, board: chess.Board) -> Tuple[Optional[chess.Move], float, SearchTree]:
        """Run Alpha-Beta search from the given board position.

        Returns
        -------
        (best_move, best_value, search_tree)
        """
        self.tree = SearchTree()
        maximizing = board.turn == chess.WHITE

        root = self.tree.new_node(
            move=None, depth=self.max_depth, is_maximizing=maximizing,
            alpha=float("-inf"), beta=float("inf"),
        )
        self.tree.set_root(root)

        value = self._alphabeta(
            board, self.max_depth, float("-inf"), float("inf"), maximizing, root
        )

        best_child = next((c for c in root.children if c.on_principal_variation), None)
        best_move = best_child.move if best_child is not None else None

        return best_move, value, self.tree

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _order_moves(board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Sort moves to maximize early pruning: captures (best victim /
        weakest attacker first) > checks > quiet moves."""

        def score_move(move: chess.Move) -> int:
            score = 0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                victim_value = PIECE_VALUES[victim.piece_type] if victim else 100  # en passant
                attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
                # MVV-LVA: prioritize high victim value, low attacker value.
                score += 10_000 + (victim_value * 10 - attacker_value)
            if move.promotion:
                score += 5_000
            if board.gives_check(move):
                score += 1_000
            return score

        return sorted(moves, key=score_move, reverse=True)

    # ------------------------------------------------------------------
    # Core recursive Depth-First Search with pruning
    # ------------------------------------------------------------------

    def _alphabeta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        node: SearchNode,
    ) -> float:
        """Recursive Alpha-Beta DFS. See module docstring for the
        alpha/beta bound semantics."""
        self.tree.nodes_expanded += 1
        node.alpha = alpha
        node.beta = beta

        # --- Terminal detection -----------------------------------------
        if depth == 0 or board.is_game_over():
            node.is_leaf = True
            node.value = evaluate(board)
            return node.value

        legal_moves = self._order_moves(board, list(board.legal_moves))

        best_child: Optional[SearchNode] = None

        if maximizing:
            best_value = float("-inf")

            for i, move in enumerate(legal_moves):
                board.push(move)
                child = self.tree.new_node(
                    move=move, depth=depth - 1, is_maximizing=not maximizing,
                    alpha=alpha, beta=beta,
                )
                node.add_child(child)

                child_value = self._alphabeta(board, depth - 1, alpha, beta, False, child)
                board.pop()

                if child_value > best_value:
                    best_value = child_value
                    best_child = child

                alpha = max(alpha, best_value)

                if alpha >= beta:
                    # Beta cutoff: remaining siblings can't matter — the
                    # minimizing parent above already has a better
                    # (lower) option than anything more we could find
                    # here. Mark the rest of the *unexplored* moves as
                    # pruned nodes (for visualization) and stop.
                    self._record_pruned_siblings(node, legal_moves, i + 1, depth, not maximizing)
                    break

            node.value = best_value
        else:
            best_value = float("inf")

            for i, move in enumerate(legal_moves):
                board.push(move)
                child = self.tree.new_node(
                    move=move, depth=depth - 1, is_maximizing=not maximizing,
                    alpha=alpha, beta=beta,
                )
                node.add_child(child)

                child_value = self._alphabeta(board, depth - 1, alpha, beta, True, child)
                board.pop()

                if child_value < best_value:
                    best_value = child_value
                    best_child = child

                beta = min(beta, best_value)

                if alpha >= beta:
                    # Alpha cutoff: symmetric case for the minimizer.
                    self._record_pruned_siblings(node, legal_moves, i + 1, depth, not maximizing)
                    break

            node.value = best_value

        # --- Backtracking: mark the winning child as part of the PV -----
        if best_child is not None:
            best_child.on_principal_variation = True

        return node.value

    def _record_pruned_siblings(
        self,
        node: SearchNode,
        ordered_moves: List[chess.Move],
        start_index: int,
        depth: int,
        child_is_maximizing: bool,
    ) -> None:
        """Add lightweight placeholder nodes (marked `is_pruned=True`,
        no further expansion, no evaluation) for every sibling move that
        the cutoff prevented us from exploring. These exist purely so
        the visualizer can show *where* pruning happened — they cost a
        node-creation but no recursive search."""
        for move in ordered_moves[start_index:]:
            pruned_node = self.tree.new_node(
                move=move,
                depth=depth - 1,
                is_maximizing=child_is_maximizing,
                is_pruned=True,
                is_leaf=True,
            )
            node.add_child(pruned_node)
            self.tree.nodes_pruned += 1
