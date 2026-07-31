"""
ai.py
==========================================================================
AI player orchestration layer.

This module ties together:
    - config.Algorithm / config.Difficulty  (what to run)
    - minimax.MinimaxAI / alphabeta.AlphaBetaAI  (how to run it)
    - performance.PerformanceStats  (what happened)
    - search.SearchTree  (the explored tree, for visualization)

The GUI (main.py) only ever talks to `AIEngine` — it doesn't need to
know whether Minimax or Alpha-Beta is running underneath, which keeps
main.py focused on rendering and input handling rather than search
internals.
==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess

from alphabeta import AlphaBetaAI
from config import Algorithm, DIFFICULTY_DEPTH, Difficulty
from minimax import MinimaxAI
from performance import PerformanceStats
from search import SearchTree
from utils import Timer


@dataclass
class AIDecision:
    """Everything the GUI needs after asking the AI to move."""

    move: Optional[chess.Move]
    evaluation_score: float
    search_tree: SearchTree
    stats: PerformanceStats


class AIEngine:
    """High-level AI player. Holds the currently selected algorithm and
    difficulty, and exposes a single `choose_move()` entry point."""

    def __init__(
        self,
        algorithm: Algorithm = Algorithm.ALPHA_BETA,
        difficulty: Difficulty = Difficulty.MEDIUM,
    ) -> None:
        self.algorithm = algorithm
        self.difficulty = difficulty

    def set_algorithm(self, algorithm: Algorithm) -> None:
        self.algorithm = algorithm

    def set_difficulty(self, difficulty: Difficulty) -> None:
        self.difficulty = difficulty

    @property
    def depth(self) -> int:
        return DIFFICULTY_DEPTH[self.difficulty]

    def choose_move(self, board: chess.Board, ply_number: int) -> AIDecision:
        """Run the currently selected search algorithm at the currently
        selected difficulty's depth, and package the result for the GUI.

        Parameters
        ----------
        board : chess.Board
            The live game board (read-only from this method's
            perspective — the search algorithms push/pop moves
            internally but always restore the board before returning).
        ply_number : int
            The ply index this move corresponds to (for stats/exports).
        """
        player = "White" if board.turn == chess.WHITE else "Black"
        depth = self.depth

        with Timer() as timer:
            if self.algorithm == Algorithm.MINIMAX:
                engine = MinimaxAI(max_depth=depth)
            else:
                engine = AlphaBetaAI(max_depth=depth)

            best_move, value, tree = engine.search(board)

        stats = PerformanceStats.build(
            ply_number=ply_number,
            player=player,
            algorithm=self.algorithm,
            difficulty=self.difficulty,
            depth=depth,
            nodes_expanded=tree.nodes_expanded,
            nodes_pruned=tree.nodes_pruned,
            execution_time_seconds=timer.elapsed_seconds,
            evaluation_score=value,
            best_move_uci=best_move.uci() if best_move else "none",
        )

        return AIDecision(move=best_move, evaluation_score=value, search_tree=tree, stats=stats)
