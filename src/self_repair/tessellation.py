"""Tessellation geometry and neighbor counting.

States are always 2-D ``uint8`` arrays.  For non-square lattices the array is a
storage representation, not a picture of physical space:

* ``HEXAGONAL`` uses axial coordinates.  A cell at ``(row, col)`` has six
  edge-neighbors at the offsets below.
* ``TRIANGULAR`` is the edge-adjacency graph of a triangular tiling (a
  honeycomb graph, three neighbors per cell).  Its adjacency alternates with
  the parity of ``row + col``.
* ``TRIANGULAR_VERTEX`` is the twelve-neighbor variant in which cells that
  share a vertex are also neighbors.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class Tessellation(str, Enum):
    SQUARE_MOORE = "square_moore"
    SQUARE_VON_NEUMANN = "square_von_neumann"
    HEXAGONAL = "hexagonal"
    TRIANGULAR = "triangular"
    TRIANGULAR_VERTEX = "triangular_vertex"


class Boundary(str, Enum):
    WRAP = "wrap"
    DEAD = "dead"
    ALIVE = "alive"


_SQUARE_MOORE = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
_SQUARE_VON_NEUMANN = ((-1, 0), (0, -1), (0, 1), (1, 0))
_HEXAGONAL = ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))
_TRIANGULAR_EVEN = ((0, 1), (1, 0), (0, -1))
_TRIANGULAR_ODD = ((0, -1), (-1, 0), (0, 1))
_TRIANGULAR_VERTEX = _HEXAGONAL + (
    (0, 2), (2, 0), (2, -2), (0, -2), (-2, 0), (-2, 2),
)

OFFSETS = {
    Tessellation.SQUARE_MOORE: _SQUARE_MOORE,
    Tessellation.SQUARE_VON_NEUMANN: _SQUARE_VON_NEUMANN,
    Tessellation.HEXAGONAL: _HEXAGONAL,
    Tessellation.TRIANGULAR_VERTEX: _TRIANGULAR_VERTEX,
}

MAX_NEIGHBORS = {
    Tessellation.SQUARE_MOORE: 8,
    Tessellation.SQUARE_VON_NEUMANN: 4,
    Tessellation.HEXAGONAL: 6,
    Tessellation.TRIANGULAR: 3,
    Tessellation.TRIANGULAR_VERTEX: 12,
}


def coerce_tessellation(value: Tessellation | str) -> Tessellation:
    if isinstance(value, Tessellation):
        return value
    return Tessellation(value)


def coerce_boundary(value: Boundary | str) -> Boundary:
    if isinstance(value, Boundary):
        return value
    return Boundary(value)


def _fill(boundary: Boundary) -> float:
    if boundary is Boundary.ALIVE:
        return 1.0
    return 0.0


def _shift(state: np.ndarray, dr: int, dc: int, boundary: Boundary) -> np.ndarray:
    """Return ``out[i, j] = state[i + dr, j + dc]`` under the boundary mode."""

    if boundary is Boundary.WRAP:
        return np.roll(state, (-dr, -dc), axis=(0, 1))

    rows, cols = state.shape
    pad_r, pad_c = abs(dr), abs(dc)
    padded = np.pad(
        state,
        ((pad_r, pad_r), (pad_c, pad_c)),
        mode="constant",
        constant_values=_fill(boundary),
    )
    return padded[pad_r + dr: pad_r + dr + rows, pad_c + dc: pad_c + dc + cols]


def neighbor_count(
    state: np.ndarray,
    tessellation: Tessellation | str = Tessellation.SQUARE_MOORE,
    boundary: Boundary | str = Boundary.WRAP,
) -> np.ndarray:
    """Count live neighbors for every cell.

    The returned array has the same shape as ``state`` and integer dtype.
    """

    state = np.asarray(state)
    if state.ndim != 2:
        raise ValueError(f"state must be 2-D, got shape {state.shape}")

    tess = coerce_tessellation(tessellation)
    bound = coerce_boundary(boundary)

    if tess is Tessellation.TRIANGULAR:
        rows, cols = state.shape
        parity = (np.arange(rows)[:, None] + np.arange(cols)[None, :]) % 2 == 0
        count = np.zeros(state.shape, dtype=np.int16)
        for dr, dc in _TRIANGULAR_EVEN:
            count += _shift(state, dr, dc, bound) * parity
        for dr, dc in _TRIANGULAR_ODD:
            count += _shift(state, dr, dc, bound) * (~parity)
        return count

    count = np.zeros(state.shape, dtype=np.int16)
    for dr, dc in OFFSETS[tess]:
        count += _shift(state, dr, dc, bound)
    return count
