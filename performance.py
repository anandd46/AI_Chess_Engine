"""
performance.py
==========================================================================
Performance statistics tracking, storage, and export.

After every AI move, the GUI records a PerformanceStats snapshot
(nodes expanded, nodes pruned, depth, execution time, evaluation score,
estimated memory usage). PerformanceTracker accumulates these across the
whole game and can export them to CSV / JSON, plus generate a simple
text comparison table (Minimax vs Alpha-Beta) so you can literally see
the pruning benefit in numbers.
==========================================================================
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

from config import Algorithm, Difficulty
from utils import estimate_memory_usage_kb


@dataclass
class PerformanceStats:
    """One AI-move's worth of search statistics."""

    ply_number: int
    player: str                 # "White" or "Black"
    algorithm: str              # Algorithm.value
    difficulty: str             # Difficulty.value
    depth: int
    nodes_expanded: int
    nodes_pruned: int
    execution_time_seconds: float
    evaluation_score: float
    memory_estimate_kb: float
    best_move_uci: str

    @classmethod
    def build(
        cls,
        ply_number: int,
        player: str,
        algorithm: Algorithm,
        difficulty: Difficulty,
        depth: int,
        nodes_expanded: int,
        nodes_pruned: int,
        execution_time_seconds: float,
        evaluation_score: float,
        best_move_uci: str,
    ) -> "PerformanceStats":
        return cls(
            ply_number=ply_number,
            player=player,
            algorithm=algorithm.value,
            difficulty=difficulty.value,
            depth=depth,
            nodes_expanded=nodes_expanded,
            nodes_pruned=nodes_pruned,
            execution_time_seconds=round(execution_time_seconds, 6),
            evaluation_score=round(evaluation_score, 2),
            memory_estimate_kb=estimate_memory_usage_kb(nodes_expanded),
            best_move_uci=best_move_uci,
        )


class PerformanceTracker:
    """Accumulates PerformanceStats across a game and exports them."""

    def __init__(self) -> None:
        self.history: List[PerformanceStats] = []

    def record(self, stats: PerformanceStats) -> None:
        self.history.append(stats)

    def clear(self) -> None:
        self.history.clear()

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def export_csv(self, path: Path) -> None:
        """Write all recorded stats to a CSV file."""
        if not self.history:
            fieldnames = [f.name for f in PerformanceStats.__dataclass_fields__.values()]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return

        fieldnames = list(asdict(self.history[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for stat in self.history:
                writer.writerow(asdict(stat))

    def export_json(self, path: Path) -> None:
        """Write all recorded stats to a JSON file."""
        data = [asdict(stat) for stat in self.history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def generate_comparison_table(self) -> str:
        """Build a human-readable Minimax vs Alpha-Beta comparison table
        (averaged across all recorded moves for each algorithm),
        demonstrating the pruning benefit quantitatively."""
        minimax_stats = [s for s in self.history if s.algorithm == Algorithm.MINIMAX.value]
        ab_stats = [s for s in self.history if s.algorithm == Algorithm.ALPHA_BETA.value]

        def avg(values: List[float]) -> float:
            return round(sum(values) / len(values), 2) if values else 0.0

        rows = []
        header = f"{'Metric':<28}{'Minimax':>16}{'Alpha-Beta':>16}"
        rows.append(header)
        rows.append("-" * len(header))

        mm_nodes = avg([s.nodes_expanded for s in minimax_stats])
        ab_nodes = avg([s.nodes_expanded for s in ab_stats])
        rows.append(f"{'Avg. Nodes Expanded':<28}{mm_nodes:>16}{ab_nodes:>16}")

        mm_time = avg([s.execution_time_seconds for s in minimax_stats])
        ab_time = avg([s.execution_time_seconds for s in ab_stats])
        rows.append(f"{'Avg. Execution Time (s)':<28}{mm_time:>16}{ab_time:>16}")

        ab_pruned = avg([s.nodes_pruned for s in ab_stats])
        rows.append(f"{'Avg. Nodes Pruned':<28}{'N/A':>16}{ab_pruned:>16}")

        mm_mem = avg([s.memory_estimate_kb for s in minimax_stats])
        ab_mem = avg([s.memory_estimate_kb for s in ab_stats])
        rows.append(f"{'Avg. Memory Estimate (KB)':<28}{mm_mem:>16}{ab_mem:>16}")

        if mm_nodes > 0 and ab_nodes > 0:
            reduction_pct = round(100 * (1 - (ab_nodes / mm_nodes)), 1)
            rows.append("")
            rows.append(f"Node reduction from pruning: {reduction_pct}%")

        rows.append(f"\nTotal moves recorded: Minimax={len(minimax_stats)}, "
                     f"Alpha-Beta={len(ab_stats)}")

        return "\n".join(rows)
