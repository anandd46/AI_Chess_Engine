"""
main.py
==========================================================================
Application entry point and GUI layer, built with pygame.

This file is intentionally the "dumbest" module in the project on
purpose: it contains NO chess rules (board.py / python-chess), NO
search algorithms (minimax.py / alphabeta.py), and NO evaluation logic
(evaluation.py). Its only job is:

    1. Draw the board, pieces, side panel, and buttons every frame.
    2. Translate mouse/keyboard input into game actions.
    3. Ask ai.AIEngine for a move when it's the AI's turn.
    4. Ask performance.PerformanceTracker / visualizer.py to export
       data when the user clicks the relevant buttons.

Run with:
    python main.py
==========================================================================
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import chess
import pygame

from ai import AIDecision, AIEngine
from board import ChessBoard
from config import (
    Algorithm,
    Colors,
    Difficulty,
    DIFFICULTY_DEPTH,
    EXPORT_EVALUATION_GRAPH_PNG,
    EXPORT_GAME_HISTORY_TXT,
    EXPORT_PERFORMANCE_CSV,
    EXPORT_PERFORMANCE_JSON,
    EXPORT_SEARCH_TREE_PNG,
    FPS,
    FontConfig,
    SIDE_PANEL_WIDTH,
    SQUARE_SIZE,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    WINDOW_WIDTH,
)
from evaluation import evaluate
from performance import PerformanceTracker
from search import SearchTree
from utils import format_score, format_time, piece_unicode_symbol, pixel_to_square, square_to_pixel
from visualizer import plot_evaluation_graph, render_search_tree_png


# ==========================================================================
# BUTTON WIDGET
# ==========================================================================

@dataclass
class Button:
    """A simple clickable rectangle button with a label and callback."""

    rect: pygame.Rect
    label: str
    callback: Callable[[], None]
    hovered: bool = False
    enabled: bool = True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if not self.enabled:
            color = Colors.PANEL_CARD
            text_color = Colors.TEXT_SECONDARY
        elif self.hovered:
            color = Colors.BUTTON_HOVER
            text_color = Colors.BUTTON_TEXT
        else:
            color = Colors.BUTTON_IDLE
            text_color = Colors.BUTTON_TEXT

        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, Colors.PANEL_BORDER, self.rect, width=1, border_radius=8)

        text_surf = font.render(self.label, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_mouse_move(self, pos: tuple[int, int]) -> None:
        self.hovered = self.enabled and self.rect.collidepoint(pos)

    def handle_click(self, pos: tuple[int, int]) -> bool:
        if self.enabled and self.rect.collidepoint(pos):
            self.callback()
            return True
        return False


# ==========================================================================
# MAIN GAME APPLICATION
# ==========================================================================

class ChessGameApp:
    """Top-level pygame application: owns the window, game state, AI
    engine, performance tracker, and all GUI widgets."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_title = self._load_font(FontConfig.SIZE_TITLE, bold=True)
        self.font_header = self._load_font(FontConfig.SIZE_HEADER, bold=True)
        self.font_body = self._load_font(FontConfig.SIZE_BODY)
        self.font_small = self._load_font(FontConfig.SIZE_SMALL)
        self.font_piece = self._load_font(FontConfig.SIZE_PIECE_UNICODE)

        # --- Core game state ---
        self.chess_board = ChessBoard()
        self.human_color = chess.WHITE
        self.ai_engine = AIEngine(algorithm=Algorithm.ALPHA_BETA, difficulty=Difficulty.MEDIUM)
        self.tracker = PerformanceTracker()

        self.selected_square: Optional[chess.Square] = None
        self.legal_targets: List[chess.Move] = []
        self.last_decision: Optional[AIDecision] = None
        self.last_search_tree: Optional[SearchTree] = None
        self.status_message: str = "Your move — click a piece to begin."
        self.status_color = Colors.TEXT_SECONDARY

        self.ai_is_thinking: bool = False

        self.buttons: List[Button] = []
        self._build_buttons()

        self.running = True

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        try:
            font = pygame.font.SysFont(FontConfig.FONT_NAME, size, bold=bold)
        except Exception:
            font = pygame.font.Font(None, size)
        return font

    def _build_buttons(self) -> None:
        panel_x = SQUARE_SIZE * 8 + 20
        width = SIDE_PANEL_WIDTH - 40
        btn_h = 38
        gap = 10
        y = WINDOW_HEIGHT - (btn_h * 6 + gap * 5) - 20

        specs = [
            ("New Game", self.action_new_game),
            ("Undo Move", self.action_undo),
            (f"Switch Algorithm", self.action_switch_algorithm),
            (f"Change Difficulty", self.action_change_difficulty),
            ("Export Search Tree", self.action_export_tree),
            ("Export Statistics", self.action_export_stats),
        ]

        for label, callback in specs:
            rect = pygame.Rect(panel_x, y, width, btn_h)
            self.buttons.append(Button(rect=rect, label=label, callback=callback))
            y += btn_h + gap

    # ------------------------------------------------------------------
    # Actions (button callbacks)
    # ------------------------------------------------------------------

    def action_new_game(self) -> None:
        self.chess_board.reset()
        self.tracker.clear()
        self.selected_square = None
        self.legal_targets = []
        self.last_decision = None
        self.last_search_tree = None
        self.status_message = "New game started. Your move."
        self.status_color = Colors.TEXT_SECONDARY

    def action_undo(self) -> None:
        # Undo both the AI's move and the human's move so it's the
        # human's turn again (skip if there's nothing to undo).
        undone_any = False
        for _ in range(2):
            if self.chess_board.board.move_stack:
                self.chess_board.undo_move()
                undone_any = True
            if self.chess_board.turn_name() == ("White" if self.human_color == chess.WHITE else "Black"):
                break
        self.selected_square = None
        self.legal_targets = []
        if undone_any:
            self.status_message = "Move undone."
            self.status_color = Colors.TEXT_WARNING
        else:
            self.status_message = "Nothing to undo."
            self.status_color = Colors.TEXT_SECONDARY

    def action_switch_algorithm(self) -> None:
        self.ai_engine.set_algorithm(self.ai_engine.algorithm.next())
        self.status_message = f"Algorithm switched to {self.ai_engine.algorithm.value}."
        self.status_color = Colors.TEXT_ACCENT

    def action_change_difficulty(self) -> None:
        self.ai_engine.set_difficulty(self.ai_engine.difficulty.next())
        self.status_message = f"Difficulty changed to {self.ai_engine.difficulty.value} " \
                               f"(depth {self.ai_engine.depth})."
        self.status_color = Colors.TEXT_ACCENT

    def action_export_tree(self) -> None:
        if self.last_search_tree is None:
            self.status_message = "No search tree yet — let the AI make a move first."
            self.status_color = Colors.TEXT_WARNING
            return
        ok = render_search_tree_png(self.last_search_tree, EXPORT_SEARCH_TREE_PNG)
        if ok:
            self.status_message = f"Search tree exported to {EXPORT_SEARCH_TREE_PNG.name}"
            self.status_color = Colors.TEXT_SUCCESS
        else:
            self.status_message = "Graphviz not available — install the Graphviz system binary."
            self.status_color = Colors.TEXT_DANGER

    def action_export_stats(self) -> None:
        self.tracker.export_csv(EXPORT_PERFORMANCE_CSV)
        self.tracker.export_json(EXPORT_PERFORMANCE_JSON)
        plot_evaluation_graph(self.tracker.history, EXPORT_EVALUATION_GRAPH_PNG)
        with open(EXPORT_GAME_HISTORY_TXT, "w", encoding="utf-8") as f:
            f.write(self.chess_board.history_as_text())
            f.write("\n\n--- Algorithm Comparison ---\n")
            f.write(self.tracker.generate_comparison_table())
        self.status_message = "Statistics, evaluation graph, and game history exported."
        self.status_color = Colors.TEXT_SUCCESS

    # ------------------------------------------------------------------
    # Game flow
    # ------------------------------------------------------------------

    def is_human_turn(self) -> bool:
        return self.chess_board.board.turn == self.human_color and not self.chess_board.is_game_over()

    def handle_square_click(self, square: chess.Square) -> None:
        if not self.is_human_turn():
            return

        piece = self.chess_board.piece_at(square)

        # Case 1: nothing selected yet -> select if it's the human's piece.
        if self.selected_square is None:
            if piece is not None and piece.color == self.human_color:
                self.selected_square = square
                self.legal_targets = self.chess_board.legal_destinations(square)
            return

        # Case 2: clicking the already-selected square -> deselect.
        if square == self.selected_square:
            self.selected_square = None
            self.legal_targets = []
            return

        # Case 3: clicking another one of the human's own pieces -> reselect.
        if piece is not None and piece.color == self.human_color:
            self.selected_square = square
            self.legal_targets = self.chess_board.legal_destinations(square)
            return

        # Case 4: attempt to move to `square`.
        move = self._find_matching_move(self.selected_square, square)
        if move is not None:
            self.chess_board.push_move(move)
            self.status_message = f"You played {move.uci()}."
            self.status_color = Colors.TEXT_SECONDARY
        self.selected_square = None
        self.legal_targets = []

    def _find_matching_move(self, frm: chess.Square, to: chess.Square) -> Optional[chess.Move]:
        """Find a legal move from `frm` to `to`, auto-promoting to Queen
        if this is a pawn-promotion move (kept simple for GUI clarity —
        a full promotion-choice dialog is a natural future improvement,
        noted in README.md)."""
        for m in self.legal_targets:
            if m.to_square == to:
                if m.promotion and m.promotion != chess.QUEEN:
                    continue
                return m
        return None

    def maybe_trigger_ai_move(self) -> None:
        if self.is_human_turn() or self.chess_board.is_game_over():
            return

        ply = self.chess_board.move_count() + 1
        decision = self.ai_engine.choose_move(self.chess_board.board, ply_number=ply)
        self.last_decision = decision
        self.last_search_tree = decision.search_tree
        self.tracker.record(decision.stats)

        if decision.move is not None:
            self.chess_board.push_move(decision.move)
            self.status_message = (
                f"AI ({decision.stats.algorithm}, {decision.stats.difficulty}) played "
                f"{decision.stats.best_move_uci} | eval {format_score(decision.evaluation_score)} "
                f"| {decision.stats.nodes_expanded} nodes in {format_time(decision.stats.execution_time_seconds)}"
            )
            self.status_color = Colors.TEXT_ACCENT
        else:
            self.status_message = "AI has no legal moves."
            self.status_color = Colors.TEXT_DANGER

        if self.chess_board.is_game_over():
            self.status_message = self.chess_board.result_description()
            self.status_color = Colors.TEXT_WARNING

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while self.running:
            self._handle_events()
            self.maybe_trigger_ai_move()
            self._draw()
            self.clock.tick(FPS)

        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEMOTION:
                for btn in self.buttons:
                    btn.handle_mouse_move(event.pos)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                handled = False
                for btn in self.buttons:
                    if btn.handle_click(event.pos):
                        handled = True
                        break
                if not handled and event.pos[0] < SQUARE_SIZE * 8:
                    square = pixel_to_square(event.pos[0], event.pos[1])
                    if square is not None:
                        self.handle_square_click(square)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self.screen.fill(Colors.APP_BACKGROUND)
        self._draw_board()
        self._draw_pieces()
        self._draw_side_panel()
        pygame.display.flip()

    def _draw_board(self) -> None:
        last_move = self.chess_board.last_move()
        check_square = self.chess_board.king_square_in_check()

        for square in chess.SQUARES:
            x, y = square_to_pixel(square)
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            is_light = (file_idx + rank_idx) % 2 == 1
            color = Colors.LIGHT_SQUARE if is_light else Colors.DARK_SQUARE

            rect = pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(self.screen, color, rect)

            if last_move is not None and square in (last_move.from_square, last_move.to_square):
                overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                overlay.fill((*Colors.LAST_MOVE_HIGHLIGHT, 110))
                self.screen.blit(overlay, (x, y))

            if square == self.selected_square:
                overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                overlay.fill((*Colors.SELECTED_SQUARE, 140))
                self.screen.blit(overlay, (x, y))

            if check_square is not None and square == check_square:
                overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                overlay.fill((*Colors.CHECK_HIGHLIGHT, 130))
                self.screen.blit(overlay, (x, y))

        # Legal move indicators for the selected piece.
        for move in self.legal_targets:
            x, y = square_to_pixel(move.to_square)
            center = (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2)
            is_capture = self.chess_board.board.is_capture(move)
            if is_capture:
                pygame.draw.circle(self.screen, Colors.LEGAL_CAPTURE_RING, center,
                                    SQUARE_SIZE // 2 - 4, width=4)
            else:
                pygame.draw.circle(self.screen, Colors.LEGAL_MOVE_DOT, center, 10)

        # Board border.
        pygame.draw.rect(self.screen, Colors.PANEL_BORDER,
                          pygame.Rect(0, 0, SQUARE_SIZE * 8, SQUARE_SIZE * 8), width=2)

    def _draw_pieces(self) -> None:
        for square in chess.SQUARES:
            piece = self.chess_board.piece_at(square)
            if piece is None:
                continue
            x, y = square_to_pixel(square)
            glyph = piece_unicode_symbol(piece)
            color = Colors.WHITE if piece.color == chess.WHITE else Colors.BLACK
            outline = Colors.BLACK if piece.color == chess.WHITE else (90, 90, 90)

            text_surf = self.font_piece.render(glyph, True, color)
            # Faux outline for legibility against either square color.
            rect = text_surf.get_rect(center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2))
            self.screen.blit(text_surf, rect)

    def _draw_side_panel(self) -> None:
        panel_rect = pygame.Rect(SQUARE_SIZE * 8, 0, SIDE_PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, Colors.PANEL_BACKGROUND, panel_rect)

        x = SQUARE_SIZE * 8 + 20
        y = 18

        title = self.font_title.render("AI Chess Engine", True, Colors.TEXT_PRIMARY)
        self.screen.blit(title, (x, y))
        y += 34

        subtitle = self.font_small.render("Minimax & Alpha-Beta Pruning", True, Colors.TEXT_SECONDARY)
        self.screen.blit(subtitle, (x, y))
        y += 30

        y = self._draw_info_card(x, y, "Game State", [
            ("Turn", self.chess_board.turn_name()),
            ("Result", self.chess_board.result_description()),
            ("Ply", str(self.chess_board.move_count())),
        ])

        y += 14
        y = self._draw_info_card(x, y, "AI Configuration", [
            ("Algorithm", self.ai_engine.algorithm.value),
            ("Difficulty", f"{self.ai_engine.difficulty.value} (depth {self.ai_engine.depth})"),
        ])

        y += 14
        if self.last_decision is not None:
            s = self.last_decision.stats
            y = self._draw_info_card(x, y, "Last AI Move Statistics", [
                ("Move", s.best_move_uci),
                ("Nodes Expanded", str(s.nodes_expanded)),
                ("Nodes Pruned", str(s.nodes_pruned)),
                ("Search Depth", str(s.depth)),
                ("Exec. Time", format_time(s.execution_time_seconds)),
                ("Evaluation", format_score(s.evaluation_score)),
                ("Est. Memory", f"{s.memory_estimate_kb} KB"),
            ])
        else:
            y = self._draw_info_card(x, y, "Last AI Move Statistics", [("Status", "No AI move yet")])

        y += 14
        y = self._draw_move_history(x, y)

        # Status bar (above the buttons).
        status_y = WINDOW_HEIGHT - (38 * 6 + 10 * 5) - 20 - 30
        status_surf = self._wrap_text(self.status_message, self.font_small,
                                       SIDE_PANEL_WIDTH - 40)
        sy = status_y
        for line in status_surf:
            rendered = self.font_small.render(line, True, self.status_color)
            self.screen.blit(rendered, (x, sy))
            sy += 16

        for btn in self.buttons:
            btn.draw(self.screen, self.font_body)

    def _draw_info_card(self, x: int, y: int, title: str, rows: list[tuple[str, str]]) -> int:
        padding = 10
        row_h = 20
        card_h = padding * 2 + 22 + row_h * len(rows)
        card_rect = pygame.Rect(x - 4, y, SIDE_PANEL_WIDTH - 32, card_h)
        pygame.draw.rect(self.screen, Colors.PANEL_CARD, card_rect, border_radius=8)
        pygame.draw.rect(self.screen, Colors.PANEL_BORDER, card_rect, width=1, border_radius=8)

        ty = y + padding
        header = self.font_header.render(title, True, Colors.TEXT_PRIMARY)
        self.screen.blit(header, (x + 4, ty))
        ty += 24

        for label, value in rows:
            label_surf = self.font_small.render(f"{label}:", True, Colors.TEXT_SECONDARY)
            value_surf = self.font_small.render(value, True, Colors.TEXT_PRIMARY)
            self.screen.blit(label_surf, (x + 4, ty))
            self.screen.blit(value_surf, (x + 150, ty))
            ty += row_h

        return y + card_h

    def _draw_move_history(self, x: int, y: int) -> int:
        max_height = 130
        card_rect = pygame.Rect(x - 4, y, SIDE_PANEL_WIDTH - 32, max_height)
        pygame.draw.rect(self.screen, Colors.PANEL_CARD, card_rect, border_radius=8)
        pygame.draw.rect(self.screen, Colors.PANEL_BORDER, card_rect, width=1, border_radius=8)

        header = self.font_header.render("Move History", True, Colors.TEXT_PRIMARY)
        self.screen.blit(header, (x + 4, y + 10))

        lines = self.chess_board.history_as_text().split("\n")
        visible_lines = lines[-6:] if len(lines) > 6 else lines
        ty = y + 36
        for line in visible_lines:
            rendered = self.font_small.render(line, True, Colors.TEXT_SECONDARY)
            self.screen.blit(rendered, (x + 4, ty))
            ty += 16

        return y + max_height

    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
        words = text.split(" ")
        lines: List[str] = []
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if font.size(trial)[0] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines[:2]  # cap to 2 lines to avoid overlapping buttons


def main() -> None:
    app = ChessGameApp()
    app.run()


if __name__ == "__main__":
    main()
