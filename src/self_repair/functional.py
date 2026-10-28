"""Measuring return to a functional state after controlled damage.

The reference run and the damaged run start from the same configuration.  Both
evolve for ``warmup`` steps, then the damaged copy is perturbed.  Recovery is
measured against the reference trajectory, so a chaotic automaton is expected
to fail exact recovery even though its coarse behaviour may remain similar.

Two guards matter for interpretation:

* a reference that dies or saturates makes repair meaningless, so such trials
  are marked degenerate and excluded from recovery rates;
* recovery must persist for ``sustain`` steps, which rejects a lucky one-step
  coincidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .automaton import Automaton, random_state
from .damage import Damage
from .rules import TotalisticRule
from .tessellation import (
    Boundary,
    Tessellation,
    coerce_boundary,
    coerce_tessellation,
)


@dataclass(frozen=True)
class RecoveryConfig:
    tolerance: float = 0.05
    absolute_fraction: float = 0.0
    sustain: int = 5
    density_tolerance: float = 0.05
    min_reference_density: float = 1e-3
    max_reference_density: float = 0.999


@dataclass
class TrialResult:
    rule: str
    rule_name: str
    tessellation: str
    boundary: str
    shape: tuple[int, int]
    seed: int
    density: float
    damage: str
    warmup: int
    horizon: int
    initial_distance: int
    damage_fraction: float
    distances: list[int] = field(default_factory=list)
    reference_density_start: float = 0.0
    reference_density_end: float = 0.0
    damaged_density_end: float = 0.0
    density_distances: list[float] = field(default_factory=list)
    recovery_time: int | None = None
    recovered: bool = False
    density_recovered: bool = False
    valid: bool = False
    degenerate_reason: str | None = None
    residual_distance: float = 0.0
    quality: float = 0.0
    survived: bool = False
    exact_steps: int = 0

    def to_dict(self, *, include_series: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "rule": self.rule,
            "rule_name": self.rule_name,
            "tessellation": self.tessellation,
            "boundary": self.boundary,
            "shape": list(self.shape),
            "seed": self.seed,
            "density": self.density,
            "damage": self.damage,
            "warmup": self.warmup,
            "horizon": self.horizon,
            "initial_distance": self.initial_distance,
            "damage_fraction": self.damage_fraction,
            "reference_density_start": self.reference_density_start,
            "reference_density_end": self.reference_density_end,
            "damaged_density_end": self.damaged_density_end,
            "recovery_time": self.recovery_time,
            "recovered": self.recovered,
            "density_recovered": self.density_recovered,
            "valid": self.valid,
            "degenerate_reason": self.degenerate_reason,
            "residual_distance": self.residual_distance,
            "quality": self.quality,
            "survived": self.survived,
            "exact_steps": self.exact_steps,
        }
        if include_series:
            data["distances"] = list(self.distances)
            data["density_distances"] = list(self.density_distances)
        return data


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))


def _first_sustained(mask: np.ndarray, sustain: int) -> int | None:
    run = 0
    for index, ok in enumerate(mask):
        run = run + 1 if bool(ok) else 0
        if run >= sustain:
            return index - sustain + 1
    return None


def evaluate_repair(
    rule: TotalisticRule,
    shape: tuple[int, int],
    seed: int,
    damage: Damage,
    *,
    tessellation: Tessellation | str = Tessellation.SQUARE_MOORE,
    boundary: Boundary | str = Boundary.WRAP,
    warmup: int = 40,
    horizon: int = 80,
    density: float = 0.35,
    recovery: RecoveryConfig | None = None,
    initial_state: np.ndarray | None = None,
) -> TrialResult:
    """Run one reference/damage pair and measure recovery."""

    recovery = RecoveryConfig() if recovery is None else recovery
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if recovery.sustain < 1 or recovery.sustain > horizon:
        raise ValueError("sustain must be in [1, horizon]")

    rng = np.random.default_rng(seed)
    if initial_state is None:
        state0 = random_state(shape, density, rng)
    else:
        state0 = np.asarray(initial_state, dtype=np.uint8).copy()
        if state0.ndim != 2:
            raise ValueError("initial_state must be 2-D")
        shape = (int(state0.shape[0]), int(state0.shape[1]))

    tess = coerce_tessellation(tessellation)
    bound = coerce_boundary(boundary)
    automaton = Automaton(rule, tess, bound)
    reference = state0.copy()
    damaged = state0.copy()

    for _ in range(warmup):
        reference = automaton.step(reference)
        damaged = automaton.step(damaged)

    reference_density_start = float(reference.mean())
    perturbation = damage.apply(damaged, rng)
    damaged = perturbation.state
    initial_distance = hamming(reference, damaged)
    total_cells = int(np.prod(shape))

    distances: list[int] = []
    density_distances: list[float] = []
    reference_density_end = reference_density_start
    damaged_density_end = float(damaged.mean())

    for _ in range(horizon):
        reference = automaton.step(reference)
        damaged = automaton.step(damaged)
        distances.append(hamming(reference, damaged))
        reference_density_end = float(reference.mean())
        damaged_density_end = float(damaged.mean())
        density_distances.append(
            abs(reference_density_end - damaged_density_end)
        )

    threshold = max(
        recovery.tolerance * initial_distance,
        recovery.absolute_fraction * total_cells,
    )
    distance_array = np.asarray(distances, dtype=float)
    recovered_mask = distance_array <= threshold
    recovery_time = _first_sustained(recovered_mask, recovery.sustain)

    density_array = np.asarray(density_distances, dtype=float)
    density_recovered = (
        _first_sustained(
            density_array <= recovery.density_tolerance,
            recovery.sustain,
        )
        is not None
    )

    degenerate_reason: str | None = None
    if initial_distance == 0:
        degenerate_reason = "no_damage"
    elif reference_density_start < recovery.min_reference_density:
        degenerate_reason = "reference_empty"
    elif reference_density_start > recovery.max_reference_density:
        degenerate_reason = "reference_saturated"
    elif reference_density_end < recovery.min_reference_density:
        degenerate_reason = "reference_died"
    elif reference_density_end > recovery.max_reference_density:
        degenerate_reason = "reference_saturated_end"

    valid = degenerate_reason is None
    window = min(recovery.sustain, len(distances))
    residual = float(np.mean(distance_array[-window:])) if window else 0.0
    quality = 0.0
    if initial_distance > 0:
        quality = float(np.clip(1.0 - residual / initial_distance, 0.0, 1.0))

    survived = damaged_density_end >= recovery.min_reference_density
    return TrialResult(
        rule=rule.to_bs(),
        rule_name=rule.name or rule.to_bs(),
        tessellation=tess.value,
        boundary=bound.value,
        shape=shape,
        seed=seed,
        density=density,
        damage=damage.describe(),
        warmup=warmup,
        horizon=horizon,
        initial_distance=initial_distance,
        damage_fraction=initial_distance / total_cells if total_cells else 0.0,
        distances=distances,
        reference_density_start=reference_density_start,
        reference_density_end=reference_density_end,
        damaged_density_end=damaged_density_end,
        density_distances=density_distances,
        recovery_time=recovery_time,
        recovered=bool(recovery_time is not None and valid),
        density_recovered=bool(density_recovered and valid),
        valid=valid,
        degenerate_reason=degenerate_reason,
        residual_distance=residual,
        quality=quality,
        survived=survived,
        exact_steps=2 * (warmup + horizon),
    )
