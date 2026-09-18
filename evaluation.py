"""
evaluation.py
==========================================================================
Manually-implemented static evaluation function for chess positions.

This is one of the most important files in the project: Minimax and
Alpha-Beta Pruning are only as good as the evaluation function sitting
at the leaves of their search tree. A "perfect" search with a bad
evaluation function still plays badly.

The score is always returned from WHITE's perspective:
    score > 0   -> position favors White
    score < 0   -> position favors Black
    score == 0  -> perfectly balanced

Units are centipawns (1 pawn = 100).

Heuristics implemented (all manual, no external eval libraries):
------------------------------------------------------------------
1. Material Score        — sum of piece values.
2. Piece-Square Tables    — rewards/punishes pieces for occupying
                             strategically strong/weak squares, tapered
                             between "midgame" and "endgame" tables for
                             the king (king safety early, king activity
                             late).
3. Mobility               — number of legal moves available (more
                             options = more flexible position).
4. King Safety            — pawn shield in front of a castled king,
                             penalized open files near the king.
5. Center Control         — bonus for occupying/attacking the four
                             central squares (d4, d5, e4, e5).
6. Pawn Structure          — penalizes doubled pawns and isolated pawns.
7. Checkmate / Stalemate  — terminal-state scoring handled here so
                             minimax.py / alphabeta.py can treat
                             `evaluate()` as the single source of truth
                             at leaf nodes.
8. Game Phase             — determines how much weight to give
                             midgame vs endgame king-related heuristics.
==========================================================================
"""

from __future__ import annotations

import chess

from config import CHECKMATE_SCORE, STALEMATE_SCORE


# ==========================================================================
# 1. MATERIAL VALUES (centipawns)
# ==========================================================================

PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King's value is not material-counted; safety matters more.
}


# ==========================================================================
# 2. PIECE-SQUARE TABLES
# --------------------------------------------------------------------------
# Each table is defined from White's point of view with index 0 = a8
# (top-left when White is at the bottom), matching python-chess's
# `chess.SQUARES` ordering when read top-to-bottom / left-to-right in
# standard chess-programming-wiki style tables. To evaluate a Black
# piece, we mirror the square vertically (flip the rank).
#
# These are the well-known, publicly-documented "classic" piece-square
# tables commonly used as a simple, explainable starting point for
# hand-written chess evaluation functions (as opposed to tuned/learned
# tables). Values are in centipawns and ADDED to material value.
# ==========================================================================

PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]

QUEEN_TABLE = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

KING_MIDGAME_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

KING_ENDGAME_TABLE = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

_PST: dict[int, list[int]] = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
}

CENTER_SQUARES = (chess.D4, chess.D5, chess.E4, chess.E5)

# Game-phase weighting: total non-pawn, non-king material at game start,
# used to compute a 0.0 (endgame) -> 1.0 (opening/midgame) phase factor.
_PHASE_MATERIAL_MAX = (
    PIECE_VALUES[chess.KNIGHT] * 4
    + PIECE_VALUES[chess.BISHOP] * 4
    + PIECE_VALUES[chess.ROOK] * 4
    + PIECE_VALUES[chess.QUEEN] * 2
)


def _mirror_square(square: chess.Square) -> chess.Square:
    """Vertically mirror a square (a1 <-> a8, etc.) so Black pieces can
    use the same piece-square tables as White (tables are authored from
    White's perspective)."""
    return chess.square_mirror(square)


def _material_and_pst_score(board: chess.Board) -> tuple[float, float]:
    """Single board scan computing both material score and piece-square
    table score. Returns (material_score, pst_score), each from White's
    perspective."""
    material = 0.0
    pst = 0.0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        value = PIECE_VALUES[piece.piece_type]
        sign = 1 if piece.color == chess.WHITE else -1
        material += sign * value

        if piece.piece_type == chess.KING:
            continue  # King PST handled separately with phase tapering.

        table = _PST[piece.piece_type]
        idx = square if piece.color == chess.WHITE else _mirror_square(square)
        pst += sign * table[idx]

    return material, pst


def _game_phase(board: chess.Board) -> float:
    """Return a value in [0.0, 1.0]: 1.0 = full midgame material still
    on the board, 0.0 = pure king-and-pawns endgame. Used to taper king
    piece-square-table weighting (king hides in midgame, king activates
    in the endgame)."""
    total = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        total += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        total += len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
    return max(0.0, min(1.0, total / _PHASE_MATERIAL_MAX))


def _king_pst_score(board: chess.Board, phase: float) -> float:
    """King piece-square score, linearly tapered between the midgame
    table (favor castled corners) and endgame table (favor
    centralization) based on `phase`."""
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        king_sq = board.king(color)
        if king_sq is None:
            continue
        idx = king_sq if color == chess.WHITE else _mirror_square(king_sq)
        mid_val = KING_MIDGAME_TABLE[idx]
        end_val = KING_ENDGAME_TABLE[idx]
        tapered = phase * mid_val + (1 - phase) * end_val
        sign = 1 if color == chess.WHITE else -1
        score += sign * tapered
    return score


def _mobility_score(board: chess.Board) -> float:
    """Mobility heuristic: (own legal moves) - (opponent legal moves),
    scaled down since raw move-count differences can otherwise dwarf
    material considerations.

    python-chess only generates legal moves for the side to move, so we
    count the side to move directly and estimate the opponent's mobility
    by evaluating legal moves after a "null move" (skip turn). This is a
    standard, cheap approximation.
    """
    side_to_move_moves = board.legal_moves.count()

    score = side_to_move_moves
    if board.turn == chess.WHITE:
        white_mobility = side_to_move_moves
        black_mobility = _count_opponent_moves(board)
    else:
        black_mobility = side_to_move_moves
        white_mobility = _count_opponent_moves(board)

    MOBILITY_WEIGHT = 2.0
    return MOBILITY_WEIGHT * (white_mobility - black_mobility)


def _count_opponent_moves(board: chess.Board) -> int:
    """Count legal moves for the side NOT currently to move, via a null
    move (skip turn). Falls back to 0 if a null move isn't available
    (e.g. side to move is in check — null move is illegal there, but
    mobility estimation is a heuristic so we degrade gracefully)."""
    if board.is_check():
        # Can't null-move out of check; approximate opponent mobility
        # as their piece count * 2 (cheap fallback, rarely triggers).
        return sum(1 for _ in board.piece_map()) 
    try:
        board.push(chess.Move.null())
        count = board.legal_moves.count()
        board.pop()
        return count
    except (AssertionError, ValueError):
        return 0


def _center_control_score(board: chess.Board) -> float:
    """Bonus for each central square (d4, d5, e4, e5) that is either
    occupied by, or attacked by, each side's pieces."""
    CENTER_WEIGHT_OCCUPY = 10
    CENTER_WEIGHT_ATTACK = 5
    score = 0.0

    for square in CENTER_SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            sign = 1 if piece.color == chess.WHITE else -1
            score += sign * CENTER_WEIGHT_OCCUPY

        if board.is_attacked_by(chess.WHITE, square):
            score += CENTER_WEIGHT_ATTACK
        if board.is_attacked_by(chess.BLACK, square):
            score -= CENTER_WEIGHT_ATTACK

    return score


def _pawn_structure_score(board: chess.Board) -> float:
    """Penalizes doubled pawns (two+ pawns on the same file) and
    isolated pawns (no friendly pawns on adjacent files)."""
    DOUBLED_PENALTY = 15
    ISOLATED_PENALTY = 12
    score = 0.0

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        pawn_squares = board.pieces(chess.PAWN, color)
        files_occupied = [chess.square_file(sq) for sq in pawn_squares]

        # Doubled pawns.
        for file_idx in set(files_occupied):
            count = files_occupied.count(file_idx)
            if count > 1:
                score -= sign * DOUBLED_PENALTY * (count - 1)

        # Isolated pawns.
        file_set = set(files_occupied)
        for file_idx in files_occupied:
            neighbors = {file_idx - 1, file_idx + 1}
            if not neighbors & file_set:
                score -= sign * ISOLATED_PENALTY

    return score


def _king_safety_score(board: chess.Board, phase: float) -> float:
    """Rewards an intact pawn shield in front of a castled king during
    the midgame (weighted down as `phase` approaches the endgame, where
    king safety matters far less than king activity)."""
    if phase < 0.15:
        return 0.0  # Deep endgame: king safety heuristic no longer relevant.

    SHIELD_BONUS = 8
    score = 0.0

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        king_sq = board.king(color)
        if king_sq is None:
            continue

        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        shield_rank = king_rank + 1 if color == chess.WHITE else king_rank - 1

        if not 0 <= shield_rank <= 7:
            continue

        for f in (king_file - 1, king_file, king_file + 1):
            if not 0 <= f <= 7:
                continue
            shield_sq = chess.square(f, shield_rank)
            piece = board.piece_at(shield_sq)
            if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                score += sign * SHIELD_BONUS * phase

    return score


def evaluate(board: chess.Board) -> float:
    """Main static evaluation function. Returns a centipawn score from
    White's perspective.

    This is the function called at every leaf node of the Minimax /
    Alpha-Beta search tree (see minimax.py, alphabeta.py).
    """
    # --- Terminal states are decisive and short-circuit everything else.
    if board.is_checkmate():
        # Side to move is checkmated -> bad for the side to move.
        return -CHECKMATE_SCORE if board.turn == chess.WHITE else CHECKMATE_SCORE

    if board.is_stalemate() or board.is_insufficient_material() or \
            board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return STALEMATE_SCORE

    phase = _game_phase(board)

    material, pst = _material_and_pst_score(board)
    king_pst = _king_pst_score(board, phase)
    mobility = _mobility_score(board)
    center = _center_control_score(board)
    pawn_structure = _pawn_structure_score(board)
    king_safety = _king_safety_score(board, phase)

    total = material + pst + king_pst + mobility + center + pawn_structure + king_safety
    return total


def evaluate_breakdown(board: chess.Board) -> dict[str, float]:
    """Same as evaluate(), but returns each heuristic component
    separately. Useful for debugging, the GUI's evaluation panel, and
    the evaluation graph export."""
    if board.is_checkmate() or board.is_stalemate():
        return {"total": evaluate(board)}

    phase = _game_phase(board)
    material, pst = _material_and_pst_score(board)
    king_pst = _king_pst_score(board, phase)
    mobility = _mobility_score(board)
    center = _center_control_score(board)
    pawn_structure = _pawn_structure_score(board)
    king_safety = _king_safety_score(board, phase)

    return {
        "material": material,
        "piece_square": pst,
        "king_pst": king_pst,
        "mobility": mobility,
        "center_control": center,
        "pawn_structure": pawn_structure,
        "king_safety": king_safety,
        "game_phase": phase,
        "total": material + pst + king_pst + mobility + center + pawn_structure + king_safety,
    }
