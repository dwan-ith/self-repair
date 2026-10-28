"""Benchmark suites over rules, seeds, sizes, boundaries and damage types."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .damage import Damage
from .functional import RecoveryConfig, TrialResult, evaluate_repair
from .patterns import make_pattern_state
from .rules import RULES, TotalisticRule, get_rule
from .tessellation import (
    Boundary,
    Tessellation,
    coerce_boundary,
    coerce_tessellation,
)

DEFAULT_RULES = (
    "life",
    "highlife",
    "daynight",
    "maze",
    "coral",
    "diamoeba",
)
DEFAULT_DAMAGES = ("flip:0.05", "disk:3")
DEFAULT_SIZES = ((32, 32), (48, 48))


@dataclass(frozen=True)
class BenchmarkSuite:
    rules: tuple[TotalisticRule, ...]
    tessellations: tuple[Tessellation, ...] = (Tessellation.SQUARE_MOORE,)
    boundaries: tuple[Boundary, ...] = (Boundary.WRAP, Boundary.DEAD)
    sizes: tuple[tuple[int, int], ...] = DEFAULT_SIZES
    seeds: tuple[int, ...] = (0, 1, 2)
    damages: tuple[Damage, ...] = (
        Damage.parse("flip:0.05"),
        Damage.parse("disk:3"),
    )
    warmup: int = 40
    horizon: int = 80
    density: float = 0.35
    pattern: str | None = None
    pattern_position: str = "center"
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError("at least one rule is required")
        if not self.damages:
            raise ValueError("at least one damage operator is required")
        object.__setattr__(
            self,
            "rules",
            tuple(get_rule(r) if isinstance(r, str) else r for r in self.rules),
        )
        object.__setattr__(
            self,
            "damages",
            tuple(Damage.parse(d) if isinstance(d, str) else d for d in self.damages),
        )
        object.__setattr__(
            self,
            "tessellations",
            tuple(coerce_tessellation(t) for t in self.tessellations),
        )
        object.__setattr__(
            self,
            "boundaries",
            tuple(coerce_boundary(b) for b in self.boundaries),
        )
        object.__setattr__(self, "sizes", tuple(self.sizes))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "damages", tuple(self.damages))

    @property
    def trial_count(self) -> int:
        return (
            len(self.rules)
            * len(self.tessellations)
            * len(self.boundaries)
            * len(self.sizes)
            * len(self.seeds)
            * len(self.damages)
        )

    def iter_trials(self) -> Iterable[TrialResult]:
        for rule, tess, boundary, shape, seed, damage in product(
            self.rules,
            self.tessellations,
            self.boundaries,
            self.sizes,
            self.seeds,
            self.damages,
        ):
            initial_state = None
            if self.pattern is not None:
                initial_state = make_pattern_state(
                    shape,
                    self.pattern,
                    position=self.pattern_position,
                    rng=np.random.default_rng(seed + 1_000_003),
                )
            yield evaluate_repair(
                rule,
                shape,
                seed,
                damage,
                tessellation=tess,
                boundary=boundary,
                warmup=self.warmup,
                horizon=self.horizon,
                density=self.density,
                recovery=self.recovery,
                initial_state=initial_state,
            )


def run_suite(
    suite: BenchmarkSuite,
    *,
    progress: Callable[[int, int, TrialResult], None] | None = None,
) -> list[TrialResult]:
    total = suite.trial_count
    results = []
    for index, trial in enumerate(suite.iter_trials(), start=1):
        results.append(trial)
        if progress is not None:
            progress(index, total, trial)
    return results


def default_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        rules=tuple(RULES[name] for name in DEFAULT_RULES),
        damages=tuple(Damage.parse(text) for text in DEFAULT_DAMAGES),
    )


def summarize(trials: Iterable[TrialResult]) -> list[dict]:
    """Aggregate trials by rule and tessellation, sorted by robust score."""

    groups: dict[tuple[str, str], list[TrialResult]] = {}
    for trial in trials:
        groups.setdefault((trial.rule, trial.tessellation), []).append(trial)

    rows = []
    for (rule, tessellation), items in groups.items():
        valid = [t for t in items if t.valid]
        recovered = [t for t in valid if t.recovered]
        coverage = len(valid) / len(items) if items else 0.0
        recovery_rate = len(recovered) / len(valid) if valid else 0.0
        recovery_rate_all = len(recovered) / len(items) if items else 0.0
        qualities = [t.quality for t in valid]
        quality = float(np.mean(qualities)) if qualities else 0.0

        times = [t.recovery_time for t in recovered if t.recovery_time is not None]
        median_time = float(np.median(times)) if times else None

        condition_rates = []
        conditions: dict[tuple, list[TrialResult]] = {}
        for trial in items:
            key = (trial.boundary, trial.damage, trial.shape)
            conditions.setdefault(key, []).append(trial)
        for condition_trials in conditions.values():
            condition_valid = [t for t in condition_trials if t.valid]
            if condition_valid:
                condition_rates.append(
                    sum(t.recovered for t in condition_valid) / len(condition_valid)
                )

        worst_condition = min(condition_rates) if condition_rates else 0.0
        condition_std = float(np.std(condition_rates)) if condition_rates else 0.0
        density_recovery = (
            sum(t.density_recovered for t in valid) / len(valid) if valid else 0.0
        )

        score = coverage * recovery_rate * (0.5 + 0.5 * quality)

        rows.append(
            {
                "rule": rule,
                "rule_name": items[0].rule_name,
                "tessellation": tessellation,
                "trials": len(items),
                "valid": len(valid),
                "coverage": coverage,
                "recovery_rate": recovery_rate,
                "recovery_rate_all": recovery_rate_all,
                "density_recovery_rate": density_recovery,
                "quality": quality,
                "score": score,
                "worst_condition_recovery": worst_condition,
                "condition_recovery_std": condition_std,
                "median_recovery_time": median_time,
                "mean_damage_fraction": float(
                    np.mean([t.damage_fraction for t in items])
                ),
                "mean_reference_density_end": float(
                    np.mean([t.reference_density_end for t in items])
                ),
                "mean_damaged_density_end": float(
                    np.mean([t.damaged_density_end for t in items])
                ),
                "degenerate_counts": _degenerate_counts(items),
            }
        )

    rows.sort(key=lambda row: (row["score"], row["worst_condition_recovery"]), reverse=True)
    return rows


def _degenerate_counts(trials: Iterable[TrialResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in trials:
        if trial.degenerate_reason is not None:
            counts[trial.degenerate_reason] = counts.get(trial.degenerate_reason, 0) + 1
    return counts


@dataclass
class BenchmarkReport:
    suite: BenchmarkSuite
    trials: list[TrialResult]

    def summary(self) -> list[dict]:
        return summarize(self.trials)

    def to_dict(self, *, include_series: bool = False) -> dict:
        return {
            "config": {
                "rules": [rule.to_bs() for rule in self.suite.rules],
                "tessellations": [t.value for t in self.suite.tessellations],
                "boundaries": [b.value for b in self.suite.boundaries],
                "sizes": [list(size) for size in self.suite.sizes],
                "seeds": list(self.suite.seeds),
                "damages": [damage.describe() for damage in self.suite.damages],
                "warmup": self.suite.warmup,
                "horizon": self.suite.horizon,
                "density": self.suite.density,
                "pattern": self.suite.pattern,
                "pattern_position": self.suite.pattern_position,
                "recovery": {
                    "tolerance": self.suite.recovery.tolerance,
                    "absolute_fraction": self.suite.recovery.absolute_fraction,
                    "sustain": self.suite.recovery.sustain,
                    "density_tolerance": self.suite.recovery.density_tolerance,
                    "min_reference_density": self.suite.recovery.min_reference_density,
                    "max_reference_density": self.suite.recovery.max_reference_density,
                },
            },
            "summary": self.summary(),
            "trials": [
                trial.to_dict(include_series=include_series) for trial in self.trials
            ],
        }

    def save_json(self, path: str | Path, *, include_series: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(include_series=include_series), indent=2),
            encoding="utf-8",
        )

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [trial.to_dict(include_series=False) for trial in self.trials]
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["shape"] = "x".join(str(v) for v in row["shape"])
                writer.writerow(row)


def format_summary(rows: list[dict]) -> str:
    header = (
        f"{'rule':<14} {'tess':<20} {'valid':>7} {'recov':>7} "
        f"{'worst':>7} {'score':>7} {'q':>6} {'t_med':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        median = "-" if row["median_recovery_time"] is None else f"{row['median_recovery_time']:.1f}"
        lines.append(
            f"{row['rule']:<14} {row['tessellation']:<20} "
            f"{row['coverage']:>7.2f} {row['recovery_rate']:>7.2f} "
            f"{row['worst_condition_recovery']:>7.2f} {row['score']:>7.3f} "
            f"{row['quality']:>6.2f} {median:>7}"
        )
    return "\n".join(lines)
