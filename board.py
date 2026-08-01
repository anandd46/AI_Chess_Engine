"""
board.py
==========================================================================
Board representation and game-state management.

This module wraps the `python-chess` library's `chess.Board` class. We
deliberately use python-chess for *rules enforcement* (legal move
generation, check/checkmate/stalemate detection, castling, en passant,
promotion, threefold repetition, FEN/PGN parsing) — reimplementing all
of standard chess rules from scratch would add hundreds of lines of code
that have nothing to do with the AI concepts this project is meant to
demonstrate (Minimax, Alpha-Beta, evaluation functions, search trees).

Everything AI-related (search, evaluation, pruning) is implemented
manually in minimax.py / alphabeta.py / evaluation.py.

Classes
-------
ChessBoard
    A thin, higher-level wrapper around chess.Board that adds:
    - move history tracking (SAN + UCI) for the GUI's move list
    - simple undo/redo-friendly push/pop
    - convenience queries used by the GUI (legal destinations for a
      square, capture detection, check detection, game-over reporting)
==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import chess


@dataclass
class MoveRecord:
    """A single played move, stored for the GUI's move-history panel and
    for game_history.txt export."""

    ply_number: int
    san: str
    uci: str
    player: str          # "White" or "Black"
    is_capture: bool
    is_check: bool
    is_checkmate: bool


class ChessBoard:
    """High-level chess board wrapper used throughout the project.

    All AI modules (minimax.py, alphabeta.py, evaluation.py) operate
    directly on the underlying `chess.Board` (via `self.board`) for
    speed — copying/pushing/popping a raw `chess.Board` is fast and is
    exactly what high-performance search algorithms need. This wrapper
    exists for the *GUI and game-flow* layer, which cares about move
    history, undo, and display concerns that the AI search does not.
    """

    def __init__(self) -> None:
        self.board: chess.Board = chess.Board()
        self.history: List[MoveRecord] = []

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to the standard starting position and clear history."""
        self.board.reset()
        self.history.clear()

    # ------------------------------------------------------------------
    # Move application
    # ------------------------------------------------------------------

    def push_move(self, move: chess.Move) -> MoveRecord:
        """Play `move` on the board and record it in history.

        Raises
        ------
        ValueError
            If the move is not legal in the current position.
        """
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move attempted: {move}")

        player = "White" if self.board.turn == chess.WHITE else "Black"
        is_capture = self.board.is_capture(move)
        san = self.board.san(move)

        self.board.push(move)

        record = MoveRecord(
            ply_number=len(self.history) + 1,
            san=san,
            uci=move.uci(),
            player=player,
            is_capture=is_capture,
            is_check=self.board.is_check(),
            is_checkmate=self.board.is_checkmate(),
        )
        self.history.append(record)
        return record

    def undo_move(self) -> Optional[MoveRecord]:
        """Undo the last move, if any. Returns the undone MoveRecord, or
        None if there was nothing to undo."""
        if not self.board.move_stack:
            return None
        self.board.pop()
        return self.history.pop() if self.history else None

    # ------------------------------------------------------------------
    # Queries used by the GUI
    # ------------------------------------------------------------------

    def legal_destinations(self, from_square: chess.Square) -> List[chess.Move]:
        """All legal moves originating from `from_square` (used to
        highlight valid moves when the human clicks a piece)."""
        return [m for m in self.board.legal_moves if m.from_square == from_square]

    def piece_at(self, square: chess.Square) -> Optional[chess.Piece]:
        return self.board.piece_at(square)

    def turn_name(self) -> str:
        return "White" if self.board.turn == chess.WHITE else "Black"

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def result_description(self) -> str:
        """Human-readable game-over description for the GUI."""
        if not self.board.is_game_over():
            return "In progress"
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"Checkmate — {winner} wins"
        if self.board.is_stalemate():
            return "Draw — Stalemate"
        if self.board.is_insufficient_material():
            return "Draw — Insufficient material"
        if self.board.is_seventyfive_moves():
            return "Draw — 75-move rule"
        if self.board.is_fivefold_repetition():
            return "Draw — Fivefold repetition"
        if self.board.can_claim_draw():
            return "Draw — Claimable draw"
        return "Game over"

    def last_move(self) -> Optional[chess.Move]:
        return self.board.move_stack[-1] if self.board.move_stack else None

    def king_square_in_check(self) -> Optional[chess.Square]:
        """Return the square of the king currently in check, or None."""
        if self.board.is_check():
            return self.board.king(self.board.turn)
        return None

    def fen(self) -> str:
        return self.board.fen()

    def move_count(self) -> int:
        return len(self.history)

    def history_as_text(self) -> str:
        """Format move history as a PGN-like move list, used both by the
        side panel and by the game_history.txt export."""
        lines = []
        for i in range(0, len(self.history), 2):
            white_move = self.history[i]
            move_no = (i // 2) + 1
            if i + 1 < len(self.history):
                black_move = self.history[i + 1]
                lines.append(f"{move_no}. {white_move.san}  {black_move.san}")
            else:
                lines.append(f"{move_no}. {white_move.san}")
        return "\n".join(lines)
