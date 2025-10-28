"""Solara implementation of a classic 10x20 Tetris game."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import reacton.ipyvuetify as v
import solara
from solara.lab import use_task

ROWS = 20
COLS = 10
DROP_BASE_INTERVAL = 0.8
DROP_ACCELERATION = 0.05
MIN_DROP_INTERVAL = 0.1
SOFT_DROP_REWARD = 1
HARD_DROP_REWARD = 2

SCORE_BY_LINES: Dict[int, int] = {
    1: 100,
    2: 300,
    3: 500,
    4: 800,
}

T_COLOR: Dict[str, str] = {
    "I": "#00F0F0",
    "O": "#F0F000",
    "T": "#A000F0",
    "S": "#00F000",
    "Z": "#F00000",
    "J": "#0000F0",
    "L": "#F0A000",
}

TETROMINO_PATTERNS: Dict[str, Sequence[Sequence[str]]] = {
    "I": (
        (
            "....",
            "####",
            "....",
            "....",
        ),
        (
            "..#.",
            "..#.",
            "..#.",
            "..#.",
        ),
    ),
    "O": (
        (
            "....",
            ".##.",
            ".##.",
            "....",
        ),
    ),
    "T": (
        (
            "....",
            ".###",
            "..#.",
            "....",
        ),
        (
            "..#.",
            ".##.",
            "..#.",
            "....",
        ),
        (
            "....",
            "..#.",
            ".###",
            "....",
        ),
        (
            "..#.",
            "..##",
            "..#.",
            "....",
        ),
    ),
    "S": (
        (
            "....",
            "..##",
            ".##.",
            "....",
        ),
        (
            "..#.",
            "..##",
            "...#",
            "....",
        ),
    ),
    "Z": (
        (
            "....",
            ".##.",
            "..##",
            "....",
        ),
        (
            "...#",
            "..##",
            "..#.",
            "....",
        ),
    ),
    "J": (
        (
            "....",
            ".###",
            "...#",
            "....",
        ),
        (
            "..##",
            "..#.",
            "..#.",
            "....",
        ),
        (
            "....",
            "#...",
            "###.",
            "....",
        ),
        (
            "..#.",
            "..#.",
            ".##.",
            "....",
        ),
    ),
    "L": (
        (
            "....",
            ".###",
            "#...",
            "....",
        ),
        (
            ".#..",
            ".#..",
            ".##.",
            "....",
        ),
        (
            "....",
            "..#.",
            "###.",
            "....",
        ),
        (
            "..##",
            "..#.",
            "..#.",
            "....",
        ),
    ),
}


def _parse_pattern(pattern: Sequence[str]) -> Tuple[Tuple[int, int], ...]:
    """Convert a 4x4 ascii representation into coordinate tuples."""

    result: List[Tuple[int, int]] = []
    for row_index, row in enumerate(pattern):
        for col_index, cell in enumerate(row):
            if cell == "#":
                result.append((row_index, col_index))
    return tuple(result)


TETROMINOES: Dict[str, Tuple[Tuple[Tuple[int, int], ...], ...]] = {
    name: tuple(_parse_pattern(orientation) for orientation in orientations)
    for name, orientations in TETROMINO_PATTERNS.items()
}

Board = List[List[Optional[str]]]


@dataclass
class GameState:
    """Immutable-style container for the full game state."""

    board: Board
    current_piece: str
    rotation: int
    position: Tuple[int, int]
    next_piece: str
    bag: List[str]
    score: int = 0
    level: int = 1
    lines_cleared: int = 0
    running: bool = False
    started: bool = False
    game_over: bool = False


def create_empty_board() -> Board:
    """Return a fresh empty game board."""

    return [[None for _ in range(COLS)] for _ in range(ROWS)]


def create_bag() -> List[str]:
    """Generate a shuffled 7-bag of tetromino identifiers."""

    bag = list(TETROMINOES.keys())
    random.shuffle(bag)
    return bag


def draw_piece(bag: List[str]) -> Tuple[str, List[str]]:
    """Pop the next tetromino from the bag, replenishing as needed."""

    working_bag = bag
    if not working_bag:
        working_bag = create_bag()
    piece = working_bag[0]
    return piece, working_bag[1:]


def spawn_position(piece: str) -> Tuple[int, int]:
    """Return the default spawn position for a piece."""

    return 0, COLS // 2 - 2


def compute_level(total_lines: int) -> int:
    """Level increases every 10 cleared lines."""

    return 1 + total_lines // 10


def compute_drop_interval(level: int) -> float:
    """Drop speed accelerates with the current level."""

    interval = DROP_BASE_INTERVAL - (level - 1) * DROP_ACCELERATION
    return max(MIN_DROP_INTERVAL, interval)


def check_collision(
    board: Board,
    piece: str,
    rotation: int,
    position: Tuple[int, int],
) -> bool:
    """Determine whether the piece collides at the target position."""

    row_offset, col_offset = position
    for block_row, block_col in TETROMINOES[piece][rotation % len(TETROMINOES[piece])]:
        board_row = row_offset + block_row
        board_col = col_offset + block_col
        if board_col < 0 or board_col >= COLS or board_row >= ROWS:
            return True
        if board_row >= 0 and board[board_row][board_col] is not None:
            return True
    return False


def merge_piece(board: Board, piece: str, rotation: int, position: Tuple[int, int]) -> Board:
    """Embed the current piece into the board producing a new matrix."""

    new_board = [row.copy() for row in board]
    row_offset, col_offset = position
    for block_row, block_col in TETROMINOES[piece][rotation % len(TETROMINOES[piece])]:
        board_row = row_offset + block_row
        board_col = col_offset + block_col
        if 0 <= board_row < ROWS and 0 <= board_col < COLS:
            new_board[board_row][board_col] = piece
    return new_board


def clear_complete_lines(board: Board) -> Tuple[Board, int]:
    """Remove completely filled rows and return the new board and count."""

    remaining_rows: List[List[Optional[str]]] = [row for row in board if None in row]
    cleared = ROWS - len(remaining_rows)
    new_rows = [[None for _ in range(COLS)] for _ in range(cleared)]
    return new_rows + [row.copy() for row in remaining_rows], cleared


def try_move(game_state: GameState, row_delta: int, col_delta: int) -> Tuple[GameState, bool]:
    """Attempt to move the active piece by the supplied delta."""

    target = (game_state.position[0] + row_delta, game_state.position[1] + col_delta)
    if check_collision(game_state.board, game_state.current_piece, game_state.rotation, target):
        return game_state, False
    return replace(game_state, position=target), True


def rotate_piece(game_state: GameState) -> Tuple[GameState, bool]:
    """Rotate the active piece clockwise with simple wall kicks."""

    new_rotation = (game_state.rotation + 1) % len(TETROMINOES[game_state.current_piece])
    kicks = [
        (0, 0),
        (0, -1),
        (0, 1),
        (0, -2),
        (0, 2),
        (-1, 0),
        (-1, -1),
        (-1, 1),
    ]
    for row_adjust, col_adjust in kicks:
        candidate_position = (
            game_state.position[0] + row_adjust,
            game_state.position[1] + col_adjust,
        )
        if not check_collision(
            game_state.board,
            game_state.current_piece,
            new_rotation,
            candidate_position,
        ):
            return replace(game_state, rotation=new_rotation, position=candidate_position), True
    return game_state, False


def spawn_next_piece(game_state: GameState) -> GameState:
    """Bring the next piece into play and queue up a new preview piece."""

    next_piece, new_bag = draw_piece(game_state.bag)
    updated = replace(
        game_state,
        current_piece=game_state.next_piece,
        rotation=0,
        position=spawn_position(game_state.next_piece),
        next_piece=next_piece,
        bag=new_bag,
    )
    if check_collision(updated.board, updated.current_piece, updated.rotation, updated.position):
        return replace(updated, running=False, game_over=True)
    return updated


def lock_piece(game_state: GameState) -> GameState:
    """Embed the active piece into the board, clear lines, and spawn anew."""

    board_with_piece = merge_piece(game_state.board, game_state.current_piece, game_state.rotation, game_state.position)
    board_after_clear, cleared = clear_complete_lines(board_with_piece)
    total_lines = game_state.lines_cleared + cleared
    level = compute_level(total_lines)
    score = game_state.score + SCORE_BY_LINES.get(cleared, 0) * level
    updated = replace(
        game_state,
        board=board_after_clear,
        score=score,
        lines_cleared=total_lines,
        level=level,
    )
    if game_state.next_piece:
        return spawn_next_piece(updated)
    # This should not happen, but keep the state consistent.
    return replace(updated, running=False, game_over=True)


def step_down(game_state: GameState) -> GameState:
    """Advance the active piece by one row, locking it if necessary."""

    candidate_state, moved = try_move(game_state, 1, 0)
    if moved:
        return candidate_state
    return lock_piece(game_state)


def soft_drop(game_state: GameState) -> GameState:
    """Move the piece down a single row and grant a small reward."""

    candidate_state, moved = try_move(game_state, 1, 0)
    if moved:
        return replace(candidate_state, score=candidate_state.score + SOFT_DROP_REWARD)
    return lock_piece(game_state)


def hard_drop(game_state: GameState) -> GameState:
    """Drop the piece to the lowest legal position in a single action."""

    current = game_state
    steps = 0
    while True:
        candidate, moved = try_move(current, 1, 0)
        if not moved:
            break
        current = candidate
        steps += 1
    if steps:
        current = replace(current, score=current.score + steps * HARD_DROP_REWARD)
    return lock_piece(current)


def build_render_grid(game_state: GameState) -> List[List[Dict[str, Optional[str]]]]:
    """Produce a matrix describing how the board should be rendered."""

    grid: List[List[Dict[str, Optional[str]]]] = [
        [
            {"value": cell, "active": False}
            for cell in row
        ]
        for row in game_state.board
    ]
    for block_row, block_col in TETROMINOES[game_state.current_piece][game_state.rotation % len(TETROMINOES[game_state.current_piece])]:
        board_row = game_state.position[0] + block_row
        board_col = game_state.position[1] + block_col
        if 0 <= board_row < ROWS and 0 <= board_col < COLS:
            grid[board_row][board_col] = {"value": game_state.current_piece, "active": True}
    return grid


def build_preview_grid(piece: str) -> List[List[Optional[str]]]:
    """Generate a compact 4x4 preview matrix for the supplied piece."""

    base_orientation = TETROMINOES[piece][0]
    min_row = min(r for r, _ in base_orientation)
    min_col = min(c for _, c in base_orientation)
    grid = [[None for _ in range(4)] for _ in range(4)]
    for row, col in base_orientation:
        adjusted_row = row - min_row
        adjusted_col = col - min_col
        if 0 <= adjusted_row < 4 and 0 <= adjusted_col < 4:
            grid[adjusted_row][adjusted_col] = piece
    return grid


def extract_key(data: Optional[Dict]) -> Optional[str]:
    """Attempt to retrieve the keyboard key from the event payload."""

    if not data:
        return None
    if "key" in data:
        return data["key"]
    event = data.get("event") if isinstance(data, dict) else None
    if isinstance(event, dict) and "key" in event:
        return event["key"]
    return None


def create_initial_state(start_running: bool = False) -> GameState:
    """Construct a brand-new game state, optionally already running."""

    bag = create_bag()
    current, bag = draw_piece(bag)
    upcoming, bag = draw_piece(bag)
    initial = GameState(
        board=create_empty_board(),
        current_piece=current,
        rotation=0,
        position=spawn_position(current),
        next_piece=upcoming,
        bag=bag,
        running=start_running,
        started=start_running,
    )
    if check_collision(initial.board, initial.current_piece, initial.rotation, initial.position):
        return replace(initial, running=False, game_over=True)
    return initial


STYLE = """
.tetris-page {
    gap: 24px;
    padding: 16px;
}

.tetris-layout {
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: flex-start;
}

.tetris-board-wrapper {
    position: relative;
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
    border-radius: 12px;
    overflow: hidden;
}

.tetris-grid {
    display: grid;
    grid-template-columns: repeat(10, minmax(0, 1fr));
    gap: 2px;
    padding: 12px;
    background: linear-gradient(160deg, #0f172a, #1f2937);
    outline: none;
    width: min(72vw, 360px);
    max-width: 420px;
}

.tetris-grid.focused {
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.6);
}

.tetris-cell {
    position: relative;
    width: 100%;
    padding-top: 100%;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.04);
    overflow: hidden;
}

.tetris-cell::after {
    content: "";
    position: absolute;
    inset: 12%;
    border-radius: 6px;
    background: transparent;
    transition: transform 120ms ease, background 120ms ease;
}

.tetris-cell.filled::after {
    background: var(--cell-color, #e5e7eb);
    box-shadow: inset 0 2px 6px rgba(255, 255, 255, 0.25), inset 0 -4px 10px rgba(0, 0, 0, 0.35);
}

.tetris-cell.active::after {
    transform: scale(0.94);
    filter: brightness(1.1);
}

.tetris-preview {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4px;
    padding: 8px;
    background: rgba(15, 23, 42, 0.85);
    border-radius: 12px;
}

.tetris-preview-cell {
    position: relative;
    width: 100%;
    padding-top: 100%;
    border-radius: 4px;
    background: rgba(148, 163, 184, 0.15);
}

.tetris-preview-cell.filled::after {
    content: "";
    position: absolute;
    inset: 18%;
    border-radius: 6px;
    background: var(--cell-color, #e5e7eb);
    box-shadow: inset 0 1px 4px rgba(255, 255, 255, 0.2), inset 0 -3px 6px rgba(0, 0, 0, 0.25);
}

.tetris-sidebar {
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-width: 220px;
    max-width: 280px;
}

.tetris-status {
    min-height: 48px;
}

.tetris-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}

@media (max-width: 768px) {
    .tetris-grid {
        width: 92vw;
        max-width: none;
    }
    .tetris-sidebar {
        flex-direction: row;
        flex-wrap: wrap;
        justify-content: space-between;
        width: 100%;
    }
}
"""


def render_board(game_state: GameState, focused: bool) -> v.Html:
    """Render the main game board as a Vuetify HTML widget."""

    cells = []
    for row in build_render_grid(game_state):
        for cell in row:
            classes = ["tetris-cell"]
            style = ""
            if cell["value"]:
                classes.append("filled")
                if cell.get("active"):
                    classes.append("active")
                style = f"--cell-color: {T_COLOR[cell['value']]};"
            cells.append(
                v.Html(
                    tag="div",
                    class_=" ".join(classes),
                    style_=style,
                )
            )
    classes = ["tetris-grid"]
    if focused:
        classes.append("focused")
    board_widget = v.Html(
        tag="div",
        children=cells,
        class_=" ".join(classes),
        attributes={"tabindex": "0", "role": "application", "aria-label": "Tetris board"},
    )
    return board_widget


def render_preview(piece: str) -> v.Html:
    """Render the next piece preview as a compact grid."""

    cells = []
    for row in build_preview_grid(piece):
        for value in row:
            classes = ["tetris-preview-cell"]
            style = ""
            if value:
                classes.append("filled")
                style = f"--cell-color: {T_COLOR[value]};"
            cells.append(v.Html(tag="div", class_=" ".join(classes), style_=style))
    return v.Html(tag="div", class_="tetris-preview", children=cells)


@solara.component
def Game():
    """Composable Tetris game surface."""

    game_state = solara.use_reactive(create_initial_state())
    focused, set_focused = solara.use_state(False)

    async def gravity_loop():
        if not game_state.value.running or game_state.value.game_over:
            return
        while True:
            current = game_state.value
            if not current.running or current.game_over:
                break
            await asyncio.sleep(compute_drop_interval(current.level))
            current = game_state.value
            if not current.running or current.game_over:
                break
            game_state.set(step_down(current))

    use_task(gravity_loop, dependencies=[game_state.value.running, game_state.value.game_over])

    def start_or_resume():
        state = game_state.value
        if state.game_over:
            game_state.set(create_initial_state(start_running=True))
            return
        if not state.started:
            game_state.set(replace(state, running=True, started=True))
            return
        if not state.running:
            game_state.set(replace(state, running=True))

    def pause():
        state = game_state.value
        if state.running:
            game_state.set(replace(state, running=False))

    def restart():
        game_state.set(create_initial_state(start_running=True))

    def handle_key(widget, event, data):  # noqa: ARG001 - signature defined by ipyvue
        if event != "keydown":
            return
        state = game_state.value
        if state.game_over:
            return
        key = extract_key(data)
        if key is None:
            return
        if key in {"p", "P"}:
            if state.running:
                pause()
            elif not state.game_over:
                start_or_resume()
            return
        if not state.running:
            if key in {" ", "Space", "Spacebar"} and not state.game_over:
                start_or_resume()
            return
        if key in {"ArrowLeft", "a", "A"}:
            new_state, moved = try_move(state, 0, -1)
            if moved:
                game_state.set(new_state)
        elif key in {"ArrowRight", "d", "D"}:
            new_state, moved = try_move(state, 0, 1)
            if moved:
                game_state.set(new_state)
        elif key in {"ArrowDown", "s", "S"}:
            game_state.set(soft_drop(state))
        elif key in {"ArrowUp", "w", "W"}:
            new_state, rotated = rotate_piece(state)
            if rotated:
                game_state.set(new_state)
        elif key in {" ", "Space", "Spacebar"}:
            game_state.set(hard_drop(state))

    board_widget = render_board(game_state.value, focused)
    v.use_event(board_widget, "keydown", handle_key)
    v.use_event(board_widget, "focus", lambda *_: set_focused(True))
    v.use_event(board_widget, "blur", lambda *_: set_focused(False))

    status_message = None
    state = game_state.value
    if state.game_over:
        status_message = solara.Error("游戏结束，按“重新开始”再试一次！")
    elif state.started and not state.running:
        status_message = solara.Warning("游戏已暂停，按开始或 P 继续。")
    elif not state.started:
        status_message = solara.Info("点击开始或按空格/向上键开始游戏。")

    lines_label = f"消除行数：{state.lines_cleared}"
    level_label = f"当前等级：{state.level}"
    score_label = f"当前得分：{state.score}"

    with solara.Column(classes=["tetris-page"]):
        solara.Title("Solara Tetris")
        solara.Style(STYLE)
        solara.Markdown(
            "## 俄罗斯方块\n"
            "\n"
            "- ⬅️ / ➡️ 或 A / D：移动方块\n"
            "- ⬆️ 或 W：旋转\n"
            "- ⬇️ 或 S：加速下落\n"
            "- 空格：一键落地\n"
            "- P：暂停 / 继续\n"
            "点击棋盘使其获得焦点以启用键盘控制。"
        )
        with solara.Div(class_="tetris-layout"):
            with solara.Div(class_="tetris-board-wrapper"):
                solara.display(board_widget)
            with solara.Div(class_="tetris-sidebar"):
                solara.Text(score_label)
                solara.Text(level_label)
                solara.Text(lines_label)
                if state.next_piece:
                    solara.Markdown("**下一个方块**")
                    solara.display(render_preview(state.next_piece))
                with solara.Div(class_="tetris-controls"):
                    solara.Button("开始 / 继续", on_click=start_or_resume, disabled=state.running and not state.game_over)
                    solara.Button("暂停", on_click=pause, disabled=not state.running)
                    solara.Button("重新开始", on_click=restart)
                solara.Div()
                if status_message is not None:
                    with solara.Div(class_="tetris-status"):
                        solara.display(status_message)
        if not focused:
            solara.Info("提示：点击棋盘以激活键盘操作。")


@solara.component
def Page():
    """Default Solara entry point."""

    Game()


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    solara.run(Game)
