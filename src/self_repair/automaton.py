"""Exact automata and initial-condition helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rules import TotalisticRule
from .tessellation import (
    Boundary,
    Tessellation,
    coerce_boundary,
    coerce_tessellation,
)


@dataclass(frozen=True)
class Automaton:
    """A deterministic, synchronous, binary cellular automaton."""

    rule: TotalisticRule
    tessellation: Tessellation = Tessellation.SQUARE_MOORE
    boundary: Boundary = Boundary.WRAP

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tessellation", coerce_tessellation(self.tessellation)
        )
        object.__setattr__(self, "boundary", coerce_boundary(self.boundary))

    def step(self, state: np.ndarray) -> np.ndarray:
        return self.rule.step(state, self.tessellation, self.boundary)

    def run(
        self,
        state: np.ndarray,
        steps: int,
        *,
        record: bool = False,
    ) -> np.ndarray | list[np.ndarray]:
        """Advance ``state`` by ``steps`` synchronous updates.

        When ``record`` is true a list is returned containing the state after
        each update.  The initial state is not included.
        """

        if steps < 0:
            raise ValueError("steps must be non-negative")
        current = np.asarray(state, dtype=np.uint8)
        if not record:
            for _ in range(steps):
                current = self.step(current)
            return current

        history = []
        for _ in range(steps):
            current = self.step(current)
            history.append(current.copy())
        return history


def empty_state(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def full_state(shape: tuple[int, int]) -> np.ndarray:
    return np.ones(shape, dtype=np.uint8)


def random_state(
    shape: tuple[int, int],
    density: float = 0.35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Independent Bernoulli live cells."""

    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"invalid shape {shape}")
    rng = np.random.default_rng() if rng is None else rng
    return (rng.random(shape) < density).astype(np.uint8)


def run(
    state: np.ndarray,
    rule: TotalisticRule | str,
    steps: int,
    *,
    tessellation: Tessellation | str = Tessellation.SQUARE_MOORE,
    boundary: Boundary | str = Boundary.WRAP,
    record: bool = False,
) -> np.ndarray | list[np.ndarray]:
    """Functional helper around :class:`Automaton`."""

    from .rules import get_rule

    resolved = get_rule(rule) if isinstance(rule, str) else rule
    automaton = Automaton(
        resolved,
        coerce_tessellation(tessellation),
        coerce_boundary(boundary),
    )
    return automaton.run(state, steps, record=record)


@dataclass
class BatchAutomaton:
    """Step several equal-shaped states at once.

    This is useful for benchmarks that evaluate many seeds in parallel.
    """

    rule: TotalisticRule
    tessellation: Tessellation = Tessellation.SQUARE_MOORE
    boundary: Boundary = Boundary.WRAP
    _automaton: Automaton = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_automaton", Automaton(self.rule, self.tessellation, self.boundary)
        )

    def step(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        if states.ndim != 3:
            raise ValueError("states must have shape (batch, rows, cols)")
        return np.stack([self._automaton.step(s) for s in states], axis=0)
