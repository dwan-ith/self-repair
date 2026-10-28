"""Small named patterns used to test structure-level repair.

Patterns are stored as 2-D ``uint8`` matrices.  They are intended as controlled
functional targets, not as a claim that a pattern is meaningful on every
tessellation.
"""

from __future__ import annotations

import numpy as np


def _pattern(rows: list[str]) -> np.ndarray:
    return np.array(
        [[1 if ch in {"#", "1", "O"} else 0 for ch in row] for row in rows],
        dtype=np.uint8,
    )


PATTERNS: dict[str, np.ndarray] = {
    "block": _pattern(["##", "##"]),
    "beehive": _pattern([".##.", "#..#", ".##."]),
    "loaf": _pattern([".##.", "#..#", ".#.#", "..#."]),
    "boat": _pattern(["##.", "#.#", ".#."]),
    "tub": _pattern([".#.", "#.#", ".#."]),
    "blinker": _pattern(["###"]),
    "toad": _pattern([".###", "###."]),
    "beacon": _pattern(["##..", "##..", "..##", "..##"]),
    "glider": _pattern([".#.", "..#", "###"]),
    "lwss": _pattern([
        ".####",
        "#...#",
        "....#",
        "#..#.",
    ]),
    "pulsar": _pattern([
        "..###...###..",
        ".............",
        "#....#.#....#",
        "#....#.#....#",
        "#....#.#....#",
        "..###...###..",
        ".............",
        "..###...###..",
        "#....#.#....#",
        "#....#.#....#",
        "#....#.#....#",
        ".............",
        "..###...###..",
    ]),
    "single": _pattern(["#"]),
}


def get_pattern(name: str) -> np.ndarray:
    key = name.strip().lower()
    if key not in PATTERNS:
        available = ", ".join(sorted(PATTERNS))
        raise KeyError(f"unknown pattern {name!r}; available: {available}")
    return PATTERNS[key].copy()


def make_pattern_state(
    shape: tuple[int, int],
    pattern: str | np.ndarray,
    *,
    position: str = "center",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Place a pattern into a zero state of the requested shape."""

    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError("shape must be positive")

    matrix = get_pattern(pattern) if isinstance(pattern, str) else np.asarray(
        pattern, dtype=np.uint8
    ).copy()
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("pattern must be a non-empty 2-D matrix")
    if matrix.shape[0] > rows or matrix.shape[1] > cols:
        raise ValueError(
            f"pattern {matrix.shape} does not fit in grid {(rows, cols)}"
        )

    height, width = matrix.shape
    if position == "center":
        top, left = (rows - height) // 2, (cols - width) // 2
    elif position == "top_left":
        top, left = 0, 0
    elif position == "random":
        rng = np.random.default_rng() if rng is None else rng
        top = int(rng.integers(0, rows - height + 1))
        left = int(rng.integers(0, cols - width + 1))
    else:
        raise ValueError("position must be 'center', 'top_left' or 'random'")

    state = np.zeros(shape, dtype=np.uint8)
    state[top:top + height, left:left + width] = matrix
    return state
