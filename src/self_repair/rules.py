"""Totalistic binary cellular-automaton rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .tessellation import (
    Boundary,
    MAX_NEIGHBORS,
    Tessellation,
    coerce_boundary,
    coerce_tessellation,
    neighbor_count,
)


def _digits(text: str) -> frozenset[int]:
    return frozenset(int(ch) for ch in text if ch.isdigit())


def parse_bs(spec: str) -> tuple[frozenset[int], frozenset[int]]:
    """Parse ``B3/S23``, ``3/23``, ``b3s23`` or ``3/2-3``-style shorthand."""

    text = spec.strip().upper().replace(" ", "").replace("-", "")
    if not text:
        raise ValueError("empty rule specification")
    if not re.fullmatch(r"[BS/0-9]+", text):
        raise ValueError(
            f"cannot parse rule {spec!r}; expected B<counts>/S<counts>"
        )

    if "/" in text:
        left, right = text.split("/", 1)
        birth_text = left[1:] if left.startswith("B") else left
        survival_text = right[1:] if right.startswith("S") else right
    elif "S" in text and "B" in text and text.index("S") > text.index("B"):
        left, right = text.split("S", 1)
        birth_text = left[1:] if left.startswith("B") else left
        survival_text = right
    else:
        raise ValueError(
            f"cannot parse rule {spec!r}; expected B<counts>/S<counts>"
        )

    birth = _digits(birth_text)
    survival = _digits(survival_text)
    if any(c > 12 for c in birth | survival):
        raise ValueError(f"neighbor counts must be between 0 and 12: {spec!r}")
    return birth, survival


def _lut(counts: frozenset[int], limit: int = 12) -> np.ndarray:
    table = np.zeros(limit + 1, dtype=bool)
    for count in counts:
        if count <= limit:
            table[count] = True
    return table


@dataclass(frozen=True)
class TotalisticRule:
    """A binary birth/survival rule over neighbor counts."""

    birth: frozenset[int]
    survival: frozenset[int]
    name: str = ""

    def __post_init__(self) -> None:
        if any(not 0 <= c <= 12 for c in self.birth | self.survival):
            raise ValueError("neighbor counts must be between 0 and 12")

    @classmethod
    def from_bs(cls, spec: str, name: str = "") -> "TotalisticRule":
        birth, survival = parse_bs(spec)
        return cls(birth=birth, survival=survival, name=name)

    def to_bs(self) -> str:
        birth = "".join(str(c) for c in sorted(self.birth))
        survival = "".join(str(c) for c in sorted(self.survival))
        return f"B{birth}/S{survival}"

    def step(
        self,
        state: np.ndarray,
        tessellation: Tessellation | str = Tessellation.SQUARE_MOORE,
        boundary: Boundary | str = Boundary.WRAP,
    ) -> np.ndarray:
        tess = coerce_tessellation(tessellation)
        bound = coerce_boundary(boundary)
        counts = neighbor_count(state, tess, bound)

        limit = max(MAX_NEIGHBORS[tess], 12)
        birth = _lut(self.birth, limit)[counts]
        survival = _lut(self.survival, limit)[counts]

        alive = np.asarray(state) != 0
        return ((birth & ~alive) | (survival & alive)).astype(np.uint8)


RULES: dict[str, TotalisticRule] = {
    "life": TotalisticRule.from_bs("B3/S23", name="life"),
    "highlife": TotalisticRule.from_bs("B36/S23", name="highlife"),
    "daynight": TotalisticRule.from_bs("B3678/S34678", name="daynight"),
    "seeds": TotalisticRule.from_bs("B2/S", name="seeds"),
    "maze": TotalisticRule.from_bs("B3/S12345", name="maze"),
    "mazectric": TotalisticRule.from_bs("B3/S1234", name="mazectric"),
    "coral": TotalisticRule.from_bs("B3/S45678", name="coral"),
    "diamoeba": TotalisticRule.from_bs("B35678/S5678", name="diamoeba"),
    "anneal": TotalisticRule.from_bs("B4678/S35678", name="anneal"),
    "replicator": TotalisticRule.from_bs("B1357/S1357", name="replicator"),
    "twox2": TotalisticRule.from_bs("B36/S125", name="twox2"),
    "assimilation": TotalisticRule.from_bs("B345/S4567", name="assimilation"),
    "coagulations": TotalisticRule.from_bs("B378/S235678", name="coagulations"),
    "stains": TotalisticRule.from_bs("B3678/S235678", name="stains"),
    "serviettes": TotalisticRule.from_bs("B234/S", name="serviettes"),
    "vote": TotalisticRule.from_bs("B5678/S45678", name="vote"),
}


def get_rule(spec: str) -> TotalisticRule:
    """Look up a named rule or parse a B/S specification."""

    key = spec.strip().lower()
    if key in RULES:
        return RULES[key]
    compact = spec.replace(" ", "").upper()
    if "/" in spec or re.fullmatch(r"B[0-9]*S[0-9]*", compact):
        return TotalisticRule.from_bs(spec, name=spec.strip())
    available = ", ".join(sorted(RULES))
    raise KeyError(f"unknown rule {spec!r}; available names: {available}")
