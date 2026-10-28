"""Stress testing and failure mining for a single rule.

A rule that repairs damage on one seed, size or boundary may fail on another.
This module enumerates a condition grid and reports the worst conditions and
the concrete failing trials, so a claimed repair property can be challenged
rather than assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .damage import Damage
from .functional import RecoveryConfig, TrialResult, evaluate_repair
from .patterns import make_pattern_state
from .rules import TotalisticRule, get_rule
from .search import RuleGenome
from .tessellation import (
    Boundary,
    Tessellation,
    coerce_boundary,
    coerce_tessellation,
)


@dataclass(frozen=True)
class StressConfig:
    sizes: tuple[tuple[int, int], ...] = ((32, 32), (64, 64))
    seeds: tuple[int, ...] = tuple(range(8))
    damages: tuple[Damage, ...] = (
        Damage.parse("flip:0.05"),
        Damage.parse("flip:0.2"),
        Damage.parse("disk:3"),
        Damage.parse("lesion:0.1"),
        Damage.parse("stripe:2"),
    )
    boundaries: tuple[Boundary, ...] = (Boundary.WRAP, Boundary.DEAD)
    tessellation: Tessellation = Tessellation.SQUARE_MOORE
    warmup: int = 40
    horizon: int = 80
    density: float = 0.35
    pattern: str | None = None
    pattern_position: str = "center"
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    def __post_init__(self) -> None:
        if not self.sizes or not self.seeds or not self.damages or not self.boundaries:
            raise ValueError("stress conditions must be non-empty")
        object.__setattr__(self, "sizes", tuple(self.sizes))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(
            self,
            "damages",
            tuple(
                Damage.parse(damage) if isinstance(damage, str) else damage
                for damage in self.damages
            ),
        )
        object.__setattr__(
            self,
            "boundaries",
            tuple(coerce_boundary(boundary) for boundary in self.boundaries),
        )
        object.__setattr__(
            self, "tessellation", coerce_tessellation(self.tessellation)
        )

    @property
    def trial_count(self) -> int:
        return (
            len(self.sizes)
            * len(self.seeds)
            * len(self.damages)
            * len(self.boundaries)
        )


@dataclass
class StressCase:
    shape: tuple[int, int]
    seed: int
    damage: str
    boundary: str
    trial: TrialResult

    @property
    def condition(self) -> tuple[str, str, tuple[int, int]]:
        return (self.boundary, self.damage, self.shape)

    def to_dict(self) -> dict[str, Any]:
        data = self.trial.to_dict(include_series=False)
        data["condition"] = {
            "shape": list(self.shape),
            "seed": self.seed,
            "damage": self.damage,
            "boundary": self.boundary,
        }
        return data


@dataclass
class StressResult:
    rule: str
    rule_name: str
    cases: list[StressCase]
    config: StressConfig

    @property
    def valid_cases(self) -> list[StressCase]:
        return [case for case in self.cases if case.trial.valid]

    @property
    def exact_steps(self) -> int:
        return sum(case.trial.exact_steps for case in self.cases)

    def overall(self) -> dict[str, Any]:
        total = len(self.cases)
        valid = self.valid_cases
        recovered = [case for case in valid if case.trial.recovered]
        return {
            "rule": self.rule,
            "trials": total,
            "valid": len(valid),
            "coverage": len(valid) / total if total else 0.0,
            "recovered": len(recovered),
            "recovery_rate": len(recovered) / len(valid) if valid else 0.0,
            "density_recovery_rate": (
                sum(case.trial.density_recovered for case in valid) / len(valid)
                if valid
                else 0.0
            ),
            "mean_quality": (
                float(np.mean([case.trial.quality for case in valid])) if valid else 0.0
            ),
            "failures": len(self.failures()),
            "exact_steps": self.exact_steps,
        }

    def failures(self) -> list[StressCase]:
        cases = [
            case
            for case in self.cases
            if case.trial.valid and not case.trial.recovered
        ]
        return sorted(
            cases,
            key=lambda case: (case.trial.quality, -case.trial.initial_distance),
        )

    def worst_conditions(self, n: int | None = None) -> list[dict[str, Any]]:
        rows = self.summary_by_condition()
        return rows if n is None else rows[:n]

    def summary_by_condition(self) -> list[dict[str, Any]]:
        groups: dict[tuple, list[StressCase]] = {}
        for case in self.cases:
            groups.setdefault(case.condition, []).append(case)

        rows = []
        for (boundary, damage, shape), group in groups.items():
            valid = [case for case in group if case.trial.valid]
            recovered = [case for case in valid if case.trial.recovered]
            rows.append(
                {
                    "boundary": boundary,
                    "damage": damage,
                    "shape": list(shape),
                    "trials": len(group),
                    "valid": len(valid),
                    "coverage": len(valid) / len(group) if group else 0.0,
                    "recovery_rate": (
                        len(recovered) / len(valid) if valid else 0.0
                    ),
                    "mean_quality": (
                        float(np.mean([case.trial.quality for case in valid]))
                        if valid
                        else 0.0
                    ),
                    "mean_damage_fraction": float(
                        np.mean([case.trial.damage_fraction for case in group])
                    ),
                    "failing_seeds": sorted(
                        case.seed for case in valid if not case.trial.recovered
                    ),
                }
            )

        rows.sort(
            key=lambda row: (row["recovery_rate"], row["coverage"], row["mean_quality"])
        )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall(),
            "worst_conditions": self.summary_by_condition(),
            "failures": [case.to_dict() for case in self.failures()],
            "cases": [case.to_dict() for case in self.cases],
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def stress_test(
    rule: TotalisticRule | RuleGenome | str,
    config: StressConfig | None = None,
) -> StressResult:
    """Enumerate the condition grid for one rule and report failures."""

    if isinstance(rule, str):
        resolved = get_rule(rule)
    elif isinstance(rule, RuleGenome):
        resolved = rule.to_rule()
    else:
        resolved = rule

    config = StressConfig() if config is None else config
    cases: list[StressCase] = []

    for shape, seed, damage, boundary in product(
        config.sizes,
        config.seeds,
        config.damages,
        config.boundaries,
    ):
        initial_state = None
        if config.pattern is not None:
            initial_state = make_pattern_state(
                shape,
                config.pattern,
                position=config.pattern_position,
                rng=np.random.default_rng(seed + 1_000_003),
            )
        trial = evaluate_repair(
            resolved,
            shape,
            seed,
            damage,
            tessellation=config.tessellation,
            boundary=boundary,
            warmup=config.warmup,
            horizon=config.horizon,
            density=config.density,
            recovery=config.recovery,
            initial_state=initial_state,
        )
        cases.append(
            StressCase(
                shape=shape,
                seed=seed,
                damage=damage.describe(),
                boundary=boundary.value,
                trial=trial,
            )
        )

    return StressResult(
        rule=resolved.to_bs(),
        rule_name=resolved.name or resolved.to_bs(),
        cases=cases,
        config=config,
    )


def mine_failures(
    result: StressResult,
    n: int = 10,
) -> list[StressCase]:
    """Return the worst valid, non-recovered cases."""

    return result.failures()[:n]
