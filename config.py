"""
config.py
==========================================================================
Central configuration module for the AI Chess Engine project.

This file has NO game logic. It only defines constants, enums and
configuration values that every other module imports from. Keeping all
"magic numbers" here (colors, sizes, depths, file paths, etc.) means the
rest of the codebase stays readable and easy to tune.

Sections
--------
1. Window / board geometry
2. Color palette (modern dark theme)
3. Fonts
4. Algorithm & difficulty enums
5. AI search configuration
6. File / export paths
7. Misc gameplay constants
==========================================================================
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


# ==========================================================================
# 1. WINDOW / BOARD GEOMETRY
# ==========================================================================

# The board itself is always 8x8 squares. SQUARE_SIZE controls how large
# (in pixels) each square is rendered, which in turn determines the total
# board size in pixels.
SQUARE_SIZE: int = 80
BOARD_SIZE: int = SQUARE_SIZE * 8          # 640 px

# The side panel shows game state, statistics, move history and buttons.
SIDE_PANEL_WIDTH: int = 380

# Total application window size.
WINDOW_WIDTH: int = BOARD_SIZE + SIDE_PANEL_WIDTH
WINDOW_HEIGHT: int = max(BOARD_SIZE, 760)

FPS: int = 60
WINDOW_TITLE: str = "AI Chess Engine — Minimax & Alpha-Beta Pruning"


# ==========================================================================
# 2. COLOR PALETTE (modern dark theme)
# ==========================================================================

class Colors:
    """RGB color constants for the dark-theme GUI."""

    # Board squares
    LIGHT_SQUARE = (235, 236, 208)
    DARK_SQUARE = (119, 149, 86)

    # Highlights
    SELECTED_SQUARE = (246, 246, 105)
    LEGAL_MOVE_DOT = (30, 30, 30)
    LEGAL_CAPTURE_RING = (220, 80, 80)
    LAST_MOVE_HIGHLIGHT = (170, 162, 58)
    CHECK_HIGHLIGHT = (220, 60, 60)
    CANDIDATE_MOVE = (90, 160, 220)
    CHOSEN_MOVE = (80, 220, 130)

    # App background / panel (dark theme)
    APP_BACKGROUND = (18, 18, 22)
    PANEL_BACKGROUND = (28, 28, 34)
    PANEL_CARD = (38, 38, 46)
    PANEL_BORDER = (58, 58, 68)

    # Text
    TEXT_PRIMARY = (235, 235, 240)
    TEXT_SECONDARY = (160, 160, 172)
    TEXT_ACCENT = (110, 200, 255)
    TEXT_SUCCESS = (110, 220, 150)
    TEXT_WARNING = (240, 190, 90)
    TEXT_DANGER = (235, 100, 100)

    # Buttons
    BUTTON_IDLE = (46, 46, 56)
    BUTTON_HOVER = (62, 62, 76)
    BUTTON_ACTIVE = (80, 130, 220)
    BUTTON_TEXT = (240, 240, 245)

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


# ==========================================================================
# 3. FONTS
# ==========================================================================

class FontConfig:
    """Font family / size configuration. pygame falls back to a default
    system font automatically if the named font is unavailable."""

    FONT_NAME = "Segoe UI"          # falls back gracefully on non-Windows
    FALLBACK_FONT = None            # None -> pygame default font

    SIZE_TITLE = 22
    SIZE_HEADER = 17
    SIZE_BODY = 15
    SIZE_SMALL = 12
    SIZE_PIECE_UNICODE = 56


# ==========================================================================
# 4. ALGORITHM & DIFFICULTY ENUMS
# ==========================================================================

class Algorithm(Enum):
    """The two search algorithms the AI can use. Switchable at runtime."""

    MINIMAX = "Minimax"
    ALPHA_BETA = "Alpha-Beta Pruning"

    def next(self) -> "Algorithm":
        """Cycle to the other algorithm (used by the 'Switch Algorithm'
        button)."""
        members = list(Algorithm)
        idx = members.index(self)
        return members[(idx + 1) % len(members)]


class Difficulty(Enum):
    """Difficulty levels. Each maps to a fixed Minimax/Alpha-Beta search
    depth, as specified in the project requirements."""

    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    EXPERT = "Expert"

    def next(self) -> "Difficulty":
        """Cycle to the next difficulty (used by 'Change Difficulty')."""
        members = list(Difficulty)
        idx = members.index(self)
        return members[(idx + 1) % len(members)]


# Difficulty -> search depth mapping (plies), exactly as specified.
DIFFICULTY_DEPTH: dict[Difficulty, int] = {
    Difficulty.EASY: 2,
    Difficulty.MEDIUM: 3,
    Difficulty.HARD: 4,
    Difficulty.EXPERT: 5,
}


# ==========================================================================
# 5. AI SEARCH CONFIGURATION
# ==========================================================================

# Large positive/negative numbers used to represent "AI wins" / "AI loses"
# in the evaluation function. Kept well below/above float overflow, and
# far outside any realistic material evaluation swing, so checkmate always
# dominates the search.
CHECKMATE_SCORE: float = 100_000.0
STALEMATE_SCORE: float = 0.0

# Small score shaved off per additional ply to checkmate, so the engine
# prefers the *fastest* checkmate line rather than any checkmate line.
MATE_DISTANCE_PENALTY: float = 10.0

# Safety cap: even at "Expert", never let a single search run away
# indefinitely (protects the GUI from freezing on pathological positions).
MAX_SEARCH_DEPTH_HARD_CAP: int = 6

# When exporting the search tree to PNG, full chess trees have thousands
# of nodes and are unreadable. We cap how many nodes get rendered so the
# exported image stays legible; the underlying search itself is NOT
# capped, only the visualization.
MAX_VISUALIZED_NODES: int = 300


# ==========================================================================
# 6. FILE / EXPORT PATHS
# ==========================================================================

BASE_DIR: Path = Path(__file__).resolve().parent

EXPORT_SEARCH_TREE_PNG: Path = BASE_DIR / "search_tree.png"
EXPORT_PERFORMANCE_CSV: Path = BASE_DIR / "performance.csv"
EXPORT_PERFORMANCE_JSON: Path = BASE_DIR / "performance.json"
EXPORT_GAME_HISTORY_TXT: Path = BASE_DIR / "game_history.txt"
EXPORT_EVALUATION_GRAPH_PNG: Path = BASE_DIR / "evaluation_graph.png"


# ==========================================================================
# 7. MISC GAMEPLAY CONSTANTS
# ==========================================================================

# Unicode glyphs used to render pieces without needing external image
# assets (keeps the project to a single folder, no "assets" directory).
UNICODE_PIECES: dict[str, str] = {
    "P": "\u2659", "N": "\u2658", "B": "\u2657",
    "R": "\u2656", "Q": "\u2655", "K": "\u2654",
    "p": "\u265F", "n": "\u265E", "b": "\u265D",
    "r": "\u265C", "q": "\u265B", "k": "\u265A",
}

AI_THINKING_MIN_DELAY_MS: int = 150  # small delay so "thinking" is visible
