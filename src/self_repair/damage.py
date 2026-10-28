"""Controlled damage operators.

Every operator is deterministic given an ``numpy.random.Generator`` and returns
the damaged state together with the number of cells that actually changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DamageResult:
    state: np.ndarray
    changed: int
    kind: str
    description: str
    fraction: float


@dataclass(frozen=True)
class Damage:
    """A controlled perturbation.

    ``amount`` is interpreted as a fraction of the grid for ``flip``, ``noise``,
    ``kill``, ``lesion`` and ``scramble``, and as a radius/width in cells for
    ``disk``, ``stripe`` and ``border``.
    """

    kind: str
    amount: float = 0.05
    orientation: str = "row"
    cells: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        valid = {
            "flip",
            "noise",
            "kill",
            "lesion",
            "disk",
            "stripe",
            "border",
            "scramble",
            "point",
        }
        kind = self.kind.strip().lower()
        if kind not in valid:
            raise ValueError(f"unknown damage kind {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        if self.amount < 0:
            raise ValueError("damage amount must be non-negative")
        if self.orientation not in {"row", "col"}:
            raise ValueError("stripe orientation must be 'row' or 'col'")
        object.__setattr__(
            self,
            "cells",
            tuple((int(r), int(c)) for r, c in self.cells),
        )
        if kind == "point" and not self.cells:
            raise ValueError("point damage requires at least one (row, column) cell")

    def describe(self) -> str:
        if self.kind == "stripe":
            return f"stripe:{self.amount:g}:{self.orientation}"
        if self.kind == "point":
            cells = ";".join(f"{r}:{c}" for r, c in self.cells)
            return f"point:{cells}"
        return f"{self.kind}:{self.amount:g}"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "amount": self.amount,
            "orientation": self.orientation,
            "cells": [list(cell) for cell in self.cells],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Damage":
        return cls(
            kind=data["kind"],
            amount=float(data.get("amount", 0.05)),
            orientation=data.get("orientation", "row"),
            cells=tuple(tuple(cell) for cell in data.get("cells", ())),
        )

    @classmethod
    def parse(cls, text: str) -> "Damage":
        parts = [p.strip() for p in text.split(":")]
        kind = parts[0].lower()
        defaults = {
            "flip": 0.05,
            "noise": 0.02,
            "kill": 0.05,
            "lesion": 0.10,
            "disk": 3.0,
            "stripe": 2.0,
            "border": 1.0,
            "scramble": 0.10,
        }
        if kind == "point":
            payload = text.split(":", 1)[1].strip() if ":" in text else ""
            if not payload:
                raise ValueError("point damage needs cells, e.g. point:4:4;5:5")
            cells = []
            for item in payload.split(";"):
                row, col = item.split(":", 1)
                cells.append((int(row), int(col)))
            return cls(kind="point", cells=tuple(cells))
        if kind not in defaults:
            raise ValueError(f"unknown damage specification {text!r}")
        amount = float(parts[1]) if len(parts) > 1 and parts[1] else defaults[kind]
        orientation = parts[2] if len(parts) > 2 else "row"
        return cls(kind=kind, amount=amount, orientation=orientation)

    def apply(
        self,
        state: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> DamageResult:
        rng = np.random.default_rng() if rng is None else rng
        before = np.asarray(state, dtype=np.uint8).copy()
        out = before.copy()
        rows, cols = out.shape
        total = rows * cols

        if self.kind == "flip":
            k = _count(self.amount, total)
            if k:
                flat = rng.choice(total, size=k, replace=False)
                out.reshape(-1)[flat] ^= 1

        elif self.kind == "noise":
            mask = rng.random(out.shape) < min(self.amount, 1.0)
            out[mask] ^= 1

        elif self.kind == "kill":
            k = _count(self.amount, total)
            if k:
                flat = rng.choice(total, size=k, replace=False)
                out.reshape(-1)[flat] = 0

        elif self.kind == "lesion":
            rs, cs = _rectangle(out.shape, self.amount, rng)
            out[rs, cs] = 0

        elif self.kind == "scramble":
            rs, cs = _rectangle(out.shape, self.amount, rng)
            region = out[rs, cs].reshape(-1)
            out[rs, cs] = rng.permutation(region).reshape(out[rs, cs].shape)

        elif self.kind == "disk":
            radius = max(1, int(round(self.amount)))
            center_r = int(rng.integers(0, rows))
            center_c = int(rng.integers(0, cols))
            rr = np.arange(rows)[:, None] - center_r
            cc = np.arange(cols)[None, :] - center_c
            out[rr * rr + cc * cc <= radius * radius] = 0

        elif self.kind == "stripe":
            width = max(1, int(round(self.amount)))
            if self.orientation == "row":
                width = min(width, rows)
                start = int(rng.integers(0, rows - width + 1))
                out[start:start + width, :] = 0
            else:
                width = min(width, cols)
                start = int(rng.integers(0, cols - width + 1))
                out[:, start:start + width] = 0

        elif self.kind == "border":
            depth = max(1, int(round(self.amount)))
            depth = min(depth, max(rows, cols))
            out[:depth, :] = 0
            out[-depth:, :] = 0
            out[:, :depth] = 0
            out[:, -depth:] = 0

        elif self.kind == "point":
            for row, col in self.cells:
                if not (0 <= row < rows and 0 <= col < cols):
                    raise IndexError(
                        f"point damage cell {(row, col)} is outside {out.shape}"
                    )
                out[row, col] = 0

        else:  # pragma: no cover - guarded by __post_init__
            raise AssertionError(self.kind)

        changed = int(np.count_nonzero(out != before))
        return DamageResult(
            state=out,
            changed=changed,
            kind=self.kind,
            description=self.describe(),
            fraction=changed / total if total else 0.0,
        )


def _count(fraction: float, total: int) -> int:
    if fraction <= 0 or total <= 0:
        return 0
    return min(total, max(1, int(round(fraction * total))))


def _rectangle(
    shape: tuple[int, int],
    fraction: float,
    rng: np.random.Generator,
) -> tuple[slice, slice]:
    rows, cols = shape
    total = rows * cols
    area = min(total, max(1, int(round(fraction * total)))) if fraction > 0 else 0
    if area >= total:
        return slice(0, rows), slice(0, cols)

    side = max(1, int(np.ceil(np.sqrt(area))))
    height = min(rows, side)
    width = min(cols, max(1, int(np.ceil(area / height))))
    top = int(rng.integers(0, rows - height + 1))
    left = int(rng.integers(0, cols - width + 1))
    return slice(top, top + height), slice(left, left + width)
