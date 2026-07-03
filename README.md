# AI Chess Engine — Minimax & Alpha-Beta Pruning

A from-scratch, educational chess AI built in Python that demonstrates classical
Artificial Intelligence search algorithms — **Minimax**, **Alpha-Beta Pruning**,
**Depth-First Search**, and hand-written **heuristic evaluation functions** — with
a full interactive GUI, live search-tree visualization, and detailed performance
analytics comparing the two algorithms head-to-head.

Built as a university AI course project and portfolio piece: every search
algorithm is implemented manually (no chess-AI libraries), while board-rules
enforcement is delegated to `python-chess` so the code stays focused on the AI
concepts themselves rather than reimplementing castling and en passant.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [AI Concepts Used](#ai-concepts-used)
- [Complexity Analysis](#complexity-analysis)
- [Minimax vs Alpha-Beta Comparison](#minimax-vs-alpha-beta-comparison)
- [Architecture](#architecture)
- [File Explanation](#file-explanation)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [How to Play](#how-to-play)
- [Difficulty Levels](#difficulty-levels)
- [Algorithm Comparison Workflow](#algorithm-comparison-workflow)
- [Performance Analysis & Exports](#performance-analysis--exports)
- [Future Improvements](#future-improvements)
- [Resume Highlights](#resume-highlights)
- [Interview Questions & Answers](#interview-questions--answers)
- [Expected Outputs](#expected-outputs)
- [Learning Outcomes](#learning-outcomes)
- [References](#references)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project implements a playable chess engine whose "brain" is a hand-written
adversarial search: given the current board, it explores possible future
positions using **Minimax** (or **Alpha-Beta Pruning**, selectable at runtime),
scores leaf positions with a custom multi-heuristic **evaluation function**, and
backtracks the best-scoring line up to the root to choose its move.

Unlike a typical "play chess" app, the goal here is *transparency*: the entire
explored search tree can be exported as an annotated image showing exactly which
branches were visited, which were pruned, and why — turning an abstract textbook
algorithm into something you can literally see.

## Features

### Human vs AI
- Mouse-driven piece movement with legal-move highlighting (dots for quiet
  moves, rings for captures).
- Check, checkmate, and stalemate detection with on-screen status messages.
- Undo Move, Restart, New Game, and Quit (window close / `Esc`).

### AI Difficulty Levels
Difficulty is controlled purely by search depth:

| Difficulty | Depth (plies) |
|---|---|
| Easy | 2 |
| Medium | 3 |
| Hard | 4 |
| Expert | 5+ |

### Two AI Modes, Switchable at Runtime
- **Minimax** — full, unpruned adversarial search (the "ground truth" baseline).
- **Alpha-Beta Pruning** — identical search, algorithmically guaranteed to
  return the *same* move and value, but skips provably-irrelevant branches.
- Switch between them mid-game with the **Switch Algorithm** button.

### Performance Comparison
After every AI move, the side panel and export files show:
- Nodes expanded
- Nodes pruned (Alpha-Beta only)
- Search depth
- Execution time
- Evaluation score
- Estimated memory usage

All of this accumulates across the game and can be exported as a **Minimax vs
Alpha-Beta comparison table**.

### Search Tree Visualization
- Full Minimax/Alpha-Beta tree captured during every AI move.
- Exported via Graphviz to `search_tree.png`, showing: root node, children,
  leaf nodes, evaluation values, the chosen principal-variation path
  (highlighted green), pruned branches (greyed out, dashed), alpha/beta values,
  and depth.
- Rendered node count is capped (see `config.MAX_VISUALIZED_NODES`) purely for
  legibility — the underlying search is never capped.

### Evaluation Function
Implemented entirely by hand in `evaluation.py`:
- Material score
- Piece-square tables (tapered king tables for midgame vs endgame)
- Mobility
- King safety (pawn shield detection)
- Center control
- Pawn structure (doubled / isolated pawn penalties)
- Checkmate / stalemate terminal scoring
- Game-phase detection to taper king-related heuristics

### Exports
- `search_tree.png` — Graphviz-rendered search tree
- `performance.csv` / `performance.json` — full statistics history
- `game_history.txt` — move list + algorithm comparison table
- `evaluation_graph.png` — matplotlib chart of evaluation & node count over the game

## Screenshots

> _Screenshots are not included in this repository snapshot. Suggested
> placeholders once you run the app:_
> - `docs/screenshot_gameplay.png` — main board + side panel mid-game
> - `docs/screenshot_search_tree.png` — an exported `search_tree.png`
> - `docs/screenshot_evaluation_graph.png` — an exported `evaluation_graph.png`

## AI Concepts Used

### Minimax
Minimax explores the full game tree down to a fixed depth via **Depth-First
Search**. At each level, the side to move either **maximizes** (White) or
**minimizes** (Black) the evaluation score, assuming the opponent always plays
optimally against you. Values are computed at the leaves by the evaluation
function and **backtracked** up through the tree — each internal node's value is
the max (or min) of its children's values. See `minimax.py`.

### Alpha-Beta Pruning
Alpha-Beta explores *the same* tree Minimax would, but maintains two running
bounds while descending:
- **alpha** — the best value the maximizer can already guarantee.
- **beta** — the best value the minimizer can already guarantee.

Whenever `alpha >= beta` at a node, that node's remaining siblings are
**provably irrelevant** to the final decision (the opponent would never let
play reach them), so they are skipped entirely — a **prune**. Alpha-Beta is
mathematically guaranteed to return the exact same move and value as plain
Minimax, just faster. See `alphabeta.py`.

### Move Ordering
Pruning effectiveness depends heavily on the order moves are examined in: if
the best move is tried first, far more of the tree gets pruned. This project
orders moves by **MVV-LVA** (Most Valuable Victim, Least Valuable Attacker) for
captures, then checks, then quiet moves.

### Depth-First Search
Both Minimax and Alpha-Beta are, structurally, DFS over the game tree: they
fully resolve one branch (via recursive function calls) before backtracking to
try the next.

### Evaluation Functions
A static evaluation function estimates "how good" a position is *without*
searching further — it's what turns an otherwise blind tree search into
chess-aware decision making. See `evaluation.py`.

### Game Playing / Adversarial Search
Chess is a two-player, zero-sum, perfect-information game — the canonical
setting for Minimax-family algorithms, where one player's gain is exactly the
other's loss and both are assumed to play optimally.

### Search Trees
Every AI move builds an explicit, inspectable tree (`search.py`: `SearchNode`,
`SearchTree`) rather than only returning a final answer — this is what enables
the project's visualization and statistics features.

## Complexity Analysis

Let `b` = branching factor (~35 legal moves per position on average in chess)
and `d` = search depth in plies.

| Algorithm | Time Complexity | Space Complexity |
|---|---|---|
| Minimax | O(b^d) | O(d) recursion stack (O(b·d) if retaining the full tree, as this project does for visualization) |
| Alpha-Beta (worst-case ordering) | O(b^d) | O(d) / O(b·d) |
| Alpha-Beta (optimal ordering) | O(b^(d/2)) | O(d) / O(b·d) |

With optimal move ordering, Alpha-Beta effectively **doubles the reachable
search depth** for the same time budget compared to plain Minimax, because
`b^(d/2)` for depth `d` roughly equals `b^d` for depth `d/2`.

## Minimax vs Alpha-Beta Comparison

| Aspect | Minimax | Alpha-Beta Pruning |
|---|---|---|
| Explores full tree | Yes | No — prunes provably irrelevant branches |
| Returns same best move as the other | Yes | Yes (identical, guaranteed) |
| Nodes expanded | Highest | Lower — typically 60–95% fewer with move ordering |
| Sensitive to move ordering | No effect | Very sensitive — good ordering = more pruning |
| Implementation complexity | Simpler | Slightly more complex (bound tracking, cutoffs) |
| Practical use | Baseline / teaching tool | What real chess engines actually use |

This project's own automated correctness tests confirm both algorithms always
agree on the resulting evaluation value on identical positions — Alpha-Beta
just gets there by expanding a fraction of the nodes (see
[Expected Outputs](#expected-outputs) for real measured numbers).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                            main.py (GUI)                         │
│   pygame window · board rendering · side panel · buttons ·       │
│   mouse input → game actions                                     │
└───────────────┬───────────────────────────────┬──────────────────┘
                │                               │
                ▼                               ▼
        ┌───────────────┐               ┌───────────────┐
        │   board.py     │               │    ai.py       │
        │ ChessBoard     │               │  AIEngine      │
        │ (wraps         │◄──────────────┤  orchestrates  │
        │ python-chess)  │   board.board  │  search choice │
        └───────┬────────┘               └───────┬────────┘
                │                                │
                │                        ┌───────┴────────┐
                │                        ▼                ▼
                │                ┌──────────────┐ ┌────────────────┐
                │                │  minimax.py   │ │  alphabeta.py   │
                │                │  MinimaxAI    │ │  AlphaBetaAI    │
                │                └──────┬───────┘ └────────┬────────┘
                │                       │                  │
                │                       └────────┬─────────┘
                │                                ▼
                │                        ┌────────────────┐
                │                        │  evaluation.py  │
                │                        │  evaluate()     │
                │                        └────────────────┘
                │                                │
                │                                ▼
                │                        ┌────────────────┐
                │                        │   search.py     │
                │                        │ SearchNode/Tree │
                │                        └───────┬────────┘
                │                                │
                ▼                                ▼
        ┌───────────────┐               ┌────────────────┐
        │ performance.py │               │  visualizer.py  │
        │ stats tracking │               │ search_tree.png │
        │ CSV/JSON export│               │ eval graph.png  │
        └───────────────┘               └────────────────┘

        config.py (constants) and utils.py (helpers) are imported
        by nearly every module above.
```

## File Explanation

| File | Responsibility |
|---|---|
| `config.py` | All constants: window/board geometry, dark-theme color palette, fonts, `Algorithm` / `Difficulty` enums, difficulty→depth mapping, export file paths. No logic. |
| `utils.py` | Generic helpers: `Timer` context manager, memory-estimate calculation, pygame-pixel ↔ chess-square conversion, score/time formatting. No chess rules, no AI. |
| `board.py` | `ChessBoard` — a GUI-facing wrapper around `chess.Board` (python-chess) adding move history tracking, undo, and display-oriented queries (legal destinations, game-over description). |
| `evaluation.py` | The hand-written static evaluation function: material, piece-square tables, mobility, king safety, center control, pawn structure, checkmate/stalemate, game-phase tapering. |
| `search.py` | `SearchNode` / `SearchTree` — the shared tree data structure both algorithms build, and that `visualizer.py` renders. |
| `minimax.py` | `MinimaxAI` — manual recursive Minimax (no pruning), for baseline comparison. |
| `alphabeta.py` | `AlphaBetaAI` — manual recursive Minimax **with** Alpha-Beta Pruning and MVV-LVA move ordering. |
| `ai.py` | `AIEngine` — orchestration layer the GUI talks to; picks Minimax vs Alpha-Beta and the search depth, runs it, times it, and packages a `PerformanceStats` snapshot. |
| `performance.py` | `PerformanceStats` / `PerformanceTracker` — accumulates statistics across the game and exports to CSV/JSON, plus generates the Minimax-vs-Alpha-Beta comparison table. |
| `visualizer.py` | Converts a `SearchTree` into an `networkx.DiGraph`, then renders it via `graphviz` to `search_tree.png`; also builds `evaluation_graph.png` with `matplotlib`. |
| `main.py` | `ChessGameApp` — the pygame GUI: rendering, input handling, button widgets, wiring everything above together. Entry point (`python main.py`). |

**How the algorithms interact:** `main.py` calls `AIEngine.choose_move()`,
which instantiates either `MinimaxAI` or `AlphaBetaAI` at the configured depth
and calls `.search(board)`. Both classes recursively call `evaluate()` from
`evaluation.py` at every leaf, and both build a `SearchTree` from `search.py` as
they go. The returned move is applied to the `ChessBoard`, the resulting
`PerformanceStats` is recorded in `PerformanceTracker`, and the `SearchTree` is
handed to `visualizer.py` on demand (the "Export Search Tree" button).

## Installation

### Requirements
- Python 3.11–3.13 recommended (see note below on Python 3.14).
- The **Graphviz system binary** (`dot`), separate from the `graphviz` PyPI
  package, is required for search-tree PNG export:
  - **Windows:** install from https://graphviz.org/download/ and ensure it's
    on your `PATH`.
  - **macOS:** `brew install graphviz`
  - **Linux (Debian/Ubuntu):** `sudo apt-get install graphviz`

> **A note on Python 3.14:** this project was scoped for Python 3.14. If you
> hit dependency build issues on a very new interpreter (common right after a
> Python release, while binary wheels for `pygame`/`graphviz` catch up),
> the safest fallback is Python 3.11–3.12, which all listed dependencies
> support today.

### Virtual Environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
python main.py
```

A window titled **"AI Chess Engine — Minimax & Alpha-Beta Pruning"** will open
with the board on the left and the statistics/controls panel on the right.

## How to Play

1. You play **White**, moving first.
2. **Click a piece** to select it — legal destinations are highlighted (a dot
   for a quiet move, a red ring for a capture).
3. **Click a highlighted square** to make the move. Click the selected piece
   again, or another one of your pieces, to change your selection.
4. Once you move, the AI automatically thinks and responds — watch the side
   panel for its algorithm, evaluation, nodes expanded, and timing.
5. Pawns promote automatically to a Queen (see
   [Future Improvements](#future-improvements) for a promotion-choice dialog).

## Difficulty Levels

Use the **Change Difficulty** button to cycle Easy → Medium → Hard → Expert.
Higher difficulty means a deeper search (see the table in
[Features](#features)), which produces stronger, more far-sighted play at the
cost of longer AI "thinking" time — this trade-off is the whole point of the
project and is visible directly in the exported statistics.

## Algorithm Comparison Workflow

To generate a meaningful Minimax vs Alpha-Beta comparison:
1. Start a **New Game**.
2. Play a few moves with **Algorithm = Minimax**.
3. Click **Switch Algorithm** to move to Alpha-Beta, and play a few more moves
   at the **same difficulty**.
4. Click **Export Statistics** — `game_history.txt` will contain an averaged
   comparison table (nodes expanded, execution time, nodes pruned, memory) for
   each algorithm, and `performance.csv` / `.json` will contain the raw
   per-move data for your own analysis (e.g. in a spreadsheet or notebook).

## Performance Analysis & Exports

| Export | Trigger | Contents |
|---|---|---|
| `search_tree.png` | "Export Search Tree" button | The most recent AI move's full search tree (Graphviz) |
| `performance.csv` | "Export Statistics" button | Every recorded AI move's stats, one row each |
| `performance.json` | "Export Statistics" button | Same data as CSV, structured as JSON |
| `game_history.txt` | "Export Statistics" button | PGN-style move list + Minimax vs Alpha-Beta comparison table |
| `evaluation_graph.png` | "Export Statistics" button | matplotlib chart: evaluation score & nodes-expanded per ply |

## Future Improvements

- **Opening book** — a small hardcoded table of strong opening replies to skip
  early search entirely.
- **Transposition table** (Zobrist hashing) — cache previously-evaluated
  positions reached via different move orders.
- **Iterative deepening** — search depth 1, then 2, then 3… reusing move
  ordering from the previous iteration, with a time-based cutoff instead of a
  fixed depth.
- **Quiescence search** — extend search past the nominal depth only at
  "noisy" positions (mid-capture-sequence) to avoid the horizon effect.
- **Promotion-choice dialog** — let the human choose Queen/Rook/Bishop/Knight
  instead of auto-promoting to Queen.
- **AI vs AI mode** — let Minimax and Alpha-Beta (or two difficulty levels)
  play each other automatically, logging full statistics for a whole game.
- **Save/Load game** (PGN import/export) and **game replay** (step through a
  finished game move by move).
- **Move hint system** for the human player, reusing the same search engine.

## Resume Highlights

Suggested bullet points for a resume / portfolio README summary:

- Implemented **Minimax** and **Alpha-Beta Pruning** from scratch in Python,
  including MVV-LVA move ordering, achieving 60–95% node-count reduction over
  unpruned search while provably preserving optimal play.
- Designed a **multi-heuristic evaluation function** (material, tapered
  piece-square tables, mobility, king safety, pawn structure, center control)
  for adversarial game-tree search.
- Built a **search-tree visualization pipeline** (NetworkX → Graphviz) that
  renders explored/pruned branches, evaluation values, and the principal
  variation to a shareable PNG — turning an abstract algorithm into a visual
  artifact.
- Built an interactive **Pygame GUI** with real-time performance analytics
  (nodes expanded, execution time, memory estimate) and CSV/JSON/PNG export
  pipelines.

## Interview Questions & Answers

**Q: What is the time complexity of Minimax, and why?**
A: O(b^d), where b is the branching factor and d is the search depth, because
the algorithm must visit every node in a tree that branches b ways at each of
d levels — there's no way to know a branch is irrelevant without visiting it.

**Q: How does Alpha-Beta Pruning improve on Minimax without changing the
result?**
A: It tracks the best value the maximizer (alpha) and minimizer (beta) can
already guarantee elsewhere in the tree. If a node's value could never change
the parent's decision — detected via `alpha >= beta` — the remaining siblings
are skipped. Because those branches are proven irrelevant to the final choice
(not just "unlikely to matter"), the returned move and value are identical to
what unpruned Minimax would return.

**Q: Why does move ordering matter for Alpha-Beta but not for plain Minimax?**
A: Minimax visits every node regardless of order — total work is fixed.
Alpha-Beta's pruning depends on how quickly tight alpha/beta bounds are
established: examining strong moves (captures, checks) first tightens the
bounds early, causing more subsequent branches to fail the `alpha >= beta`
test and get skipped. Poor ordering (worst move first) degrades Alpha-Beta to
the same O(b^d) as plain Minimax.

**Q: Why use a fixed-depth search instead of searching until checkmate?**
A: Chess's game tree is far too large to fully solve (estimated ~10^120 nodes,
the "Shannon number" territory) — that's true even with pruning. In practice,
engines search to a manageable depth and rely on a static evaluation function
to estimate the value of non-terminal leaf positions.

**Q: What's a piece-square table and why not just use raw material count?**
A: A piece-square table adds a positional bonus/penalty based on *which
square* a piece occupies (e.g. knights are weak on the rim, kings should hide
early but centralize in the endgame). Raw material count alone can't tell a
well-placed knight from a badly-placed one, so positions can be materially
equal yet strategically very different — piece-square tables capture that.

**Q: What is the "horizon effect" and how could you mitigate it?**
A: A fixed-depth search can misjudge a position if something important (e.g.
a piece recapture) happens exactly one ply beyond the search horizon — the
engine "can't see" the consequence and misevaluates. Quiescence search
(continuing to search capture sequences past the nominal depth until the
position is "quiet") is the standard mitigation, listed under
[Future Improvements](#future-improvements).

**Q: Why maintain the full search tree in memory instead of discarding nodes
after backtracking?**
A: For visualization and statistics. A production engine would discard nodes
immediately after backtracking to keep memory at O(d) — this project trades
that memory efficiency (O(b·d) here) for the ability to render and inspect the
exact tree that was explored, which is the project's core educational goal.

## Expected Outputs

Real numbers from this project's own correctness/benchmark run (identical
starting position, depth 3, no opening book):

```
depth=3 | Minimax value=64.0  nodes=9,323   | Alpha-Beta value=64.0  nodes=695   (pruned 1,046)
depth=3 | Minimax value=237.0 nodes=24,942  | Alpha-Beta value=237.0 nodes=1,251 (pruned 4,015)
```

Both algorithms always agree on the resulting value (as they mathematically
must) — Alpha-Beta reached the same conclusion while expanding **~93–95%
fewer nodes** in these runs.

## Learning Outcomes

Working through this project reinforces:
- How adversarial search differs from single-agent search (minimizing vs
  maximizing agents alternating).
- Why pruning can speed up search without sacrificing correctness — a subtle
  but important distinction from heuristic search methods that trade
  correctness for speed.
- How to design an explainable evaluation function by decomposing "board
  quality" into named, tunable components.
- Practical trade-offs between time complexity, space complexity, and search
  quality in a real (if simplified) game-playing system.

## References

- Russell, S. & Norvig, P. — *Artificial Intelligence: A Modern Approach*
  (Minimax and Alpha-Beta Pruning, adversarial search chapter).
- Chess Programming Wiki — https://www.chessprogramming.org/ (piece-square
  tables, move ordering, search techniques).
- `python-chess` documentation — https://python-chess.readthedocs.io/
- Graphviz documentation — https://graphviz.org/documentation/

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

- `python-chess` for robust, well-tested chess-rules enforcement.
- The Chess Programming Wiki community for widely-documented, classic
  piece-square table values used as an explainable starting point.
- Built as part of coursework on classical AI search techniques.
