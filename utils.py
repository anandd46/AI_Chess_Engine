"""
utils.py
==========================================================================
Small, dependency-light helper functions shared across the project.

Nothing in this file knows about chess rules or AI search strategy — it
only provides generic utilities:

- A timing context manager / decorator for measuring execution time.
- A rough memory-usage estimator for search statistics.
- Coordinate conversion helpers between pygame pixel space and chess
  board squares.
- A couple of formatting helpers used by the GUI and exports.
==========================================================================
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Iterator, Optional

import chess

from config import SQUARE_SIZE


# ==========================================================================
# TIMING
# ==========================================================================

@dataclass
class Timer:
    """Simple stopwatch. Use as a context manager:

        with Timer() as t:
            do_work()
        print(t.elapsed_seconds)
    """

    start_time: float = 0.0
    end_time: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time else time.perf_counter()
        return end - self.start_time

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_seconds * 1000.0


@contextmanager
def measure_time() -> Generator[Timer, None, None]:
    """Context-manager convenience wrapper around Timer."""
    t = Timer()
    t.start_time = time.perf_counter()
    try:
        yield t
    finally:
        t.end_time = time.perf_counter()


# ==========================================================================
# MEMORY ESTIMATION
# ==========================================================================

# A rough, empirically-reasonable estimate of how many bytes a single
# SearchNode (see search.py) occupies once you account for the Python
# object header, its dataclass fields, and the FEN string it stores.
# This is intentionally an *estimate* (the project spec asks for a
# "memory estimate", not exact profiling via tracemalloc, which would
# slow the search down significantly).
_ESTIMATED_BYTES_PER_NODE: int = 420


def estimate_memory_usage_kb(node_count: int) -> float:
    """Estimate memory footprint (in kilobytes) of a search that expanded
    `node_count` SearchNode objects.

    Parameters
    ----------
    node_count : int
        Number of nodes expanded during a Minimax / Alpha-Beta search.

    Returns
    -------
    float
        Estimated memory usage in kilobytes.
    """
    total_bytes = node_count * _ESTIMATED_BYTES_PER_NODE
    return round(total_bytes / 1024.0, 2)


def object_size_bytes(obj: object) -> int:
    """Thin wrapper around sys.getsizeof for shallow size inspection."""
    return sys.getsizeof(obj)


# ==========================================================================
# COORDINATE CONVERSION (pygame pixels <-> chess squares)
# ==========================================================================

def pixel_to_square(x: int, y: int, flipped: bool = False) -> Optional[chess.Square]:
    """Convert a pygame pixel coordinate (relative to the board's
    top-left corner) into a python-chess Square index (0-63).

    Parameters
    ----------
    x, y : int
        Pixel coordinates relative to the board's top-left corner.
    flipped : bool
        Whether the board is rendered flipped (Black at the bottom).

    Returns
    -------
    Optional[chess.Square]
        The corresponding square, or None if (x, y) is outside the board.
    """
    if x < 0 or y < 0 or x >= SQUARE_SIZE * 8 or y >= SQUARE_SIZE * 8:
        return None

    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE

    if not flipped:
        file_index = col
        rank_index = 7 - row
    else:
        file_index = 7 - col
        rank_index = row

    return chess.square(file_index, rank_index)


def square_to_pixel(square: chess.Square, flipped: bool = False) -> tuple[int, int]:
    """Convert a python-chess Square index into the top-left pixel
    coordinate of that square (relative to the board's top-left corner).
    """
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)

    if not flipped:
        col = file_index
        row = 7 - rank_index
    else:
        col = 7 - file_index
        row = rank_index

    return col * SQUARE_SIZE, row * SQUARE_SIZE


# ==========================================================================
# FORMATTING HELPERS
# ==========================================================================

def format_score(score: float) -> str:
    """Format an evaluation score for display, e.g. '+1.35' or '-0.40'
    or 'M3' (mate in 3)."""
    from config import CHECKMATE_SCORE, MATE_DISTANCE_PENALTY

    if abs(score) >= CHECKMATE_SCORE - (MATE_DISTANCE_PENALTY * 50):
        # Close enough to a mate score to report as "mate in N".
        plies_to_mate = round((CHECKMATE_SCORE - abs(score)) / MATE_DISTANCE_PENALTY)
        moves_to_mate = max(1, (plies_to_mate + 1) // 2)
        sign = "M" if score > 0 else "-M"
        return f"{sign}{moves_to_mate}"

    pawns = score / 100.0
    sign = "+" if pawns >= 0 else ""
    return f"{sign}{pawns:.2f}"


def format_time(seconds: float) -> str:
    """Format a duration in a human-friendly way."""
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def piece_unicode_symbol(piece: chess.Piece) -> str:
    """Return the unicode glyph for a python-chess Piece object."""
    from config import UNICODE_PIECES
    return UNICODE_PIECES[piece.symbol()]


def square_name_safe(square: Optional[chess.Square]) -> str:
    """Return algebraic square name, or '-' if square is None."""
    return chess.square_name(square) if square is not None else "-"


def iter_board_squares() -> Iterator[chess.Square]:
    """Iterate over all 64 squares (0-63) in order."""
    for sq in chess.SQUARES:
        yield sq
