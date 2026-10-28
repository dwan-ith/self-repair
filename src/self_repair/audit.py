"""Audit a rule-fitness surrogate by attacking its own predictions.

A surrogate is only useful if its confident predictions survive contact with
the exact automaton.  This module trains the surrogate on a set of exact
evaluations, then deliberately selects candidates that the model likes, that it
is uncertain about, or that are simply random, and records predicted versus
exact fitness for each.

The output is a calibration report: mean absolute error, rank correlation,
whether the predictive band covers the exact value, and the specific rules
where the surrogate was most over-confident.  The exact automaton remains the
judge.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .search import (
    GenomeEvaluation,
    RuleGenome,
    SearchConfig,
    evaluate_genome,
    initial_population,
)
from .surrogate import RidgeSurrogate, _candidate_pool, _config_dict, genome_features

AUDIT_MODES: tuple[str, ...] = ("exploit", "uncertain", "random")


@dataclass
class AuditRecord:
    """One surrogate prediction checked against an exact evaluation."""

    genome: RuleGenome
    predicted: float
    predicted_std: float
    actual: float
    selected_by: str

    @property
    def rule(self) -> str:
        return self.genome.to_bs()

    @property
    def error(self) -> float:
        """Exact minus predicted.  Negative means the surrogate was optimistic."""

        return self.actual - self.predicted

    @property
    def overprediction(self) -> float:
        return self.predicted - self.actual

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "predicted": self.predicted,
            "predicted_std": self.predicted_std,
            "actual": self.actual,
            "error": self.error,
            "overprediction": self.overprediction,
            "selected_by": self.selected_by,
        }


def _ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranked = np.empty(values.shape[0], dtype=float)
    start = 0
    while start < values.shape[0]:
        stop = start
        while stop + 1 < values.shape[0] and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranked[start : stop + 1] = (start + stop) / 2.0
        start = stop + 1
    out = np.empty(values.shape[0], dtype=float)
    out[order] = ranked
    return out


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[0] < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(left * right) / denom)


def records_metrics(
    records: list[AuditRecord], mode: str = ""
) -> dict[str, Any]:
    """Calibration metrics for a set of predictions checked against the truth."""

    empty = {
        "mode": mode,
        "audit_size": 0,
        "audit_r2": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "pearson": 0.0,
        "spearman": 0.0,
        "optimism_gap": 0.0,
        "coverage_1sigma": 0.0,
        "coverage_2sigma": 0.0,
        "mean_actual": 0.0,
        "best_actual": 0.0,
    }
    if not records:
        return empty

    predicted = np.asarray([record.predicted for record in records], dtype=float)
    actual = np.asarray([record.actual for record in records], dtype=float)
    std = np.asarray([record.predicted_std for record in records], dtype=float)
    error = actual - predicted
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    residual = float(np.sum(error**2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 0.0 if total <= 1e-12 else 1.0 - residual / total
    band = np.maximum(std, 1e-9)
    return {
        "mode": mode,
        "audit_size": len(records),
        "audit_r2": r2,
        "mae": mae,
        "rmse": rmse,
        "pearson": _pearson(predicted, actual),
        "spearman": _pearson(_ranks(predicted), _ranks(actual)),
        "optimism_gap": float(np.mean(predicted - actual)),
        "coverage_1sigma": float(np.mean(np.abs(error) <= band)),
        "coverage_2sigma": float(np.mean(np.abs(error) <= 2.0 * band)),
        "mean_actual": float(actual.mean()),
        "best_actual": float(actual.max()),
    }


@dataclass
class SurrogateAudit:
    mode: str
    config: SearchConfig
    train_evaluations: list[GenomeEvaluation]
    records: list[AuditRecord]
    train_r2: float
    alpha: float
    candidates: int
    elapsed_seconds: float = 0.0

    @property
    def predicted(self) -> np.ndarray:
        return np.asarray([record.predicted for record in self.records], dtype=float)

    @property
    def actual(self) -> np.ndarray:
        return np.asarray([record.actual for record in self.records], dtype=float)

    @property
    def std(self) -> np.ndarray:
        return np.asarray(
            [record.predicted_std for record in self.records], dtype=float
        )

    def metrics(self) -> dict[str, Any]:
        metrics = records_metrics(self.records, self.mode)
        metrics.update(
            {
                "train_size": len(self.train_evaluations),
                "candidates": self.candidates,
                "train_r2": self.train_r2,
            }
        )
        return metrics

    def optimism_gap(self) -> float:
        if not self.records:
            return 0.0
        return float(np.mean([record.overprediction for record in self.records]))

    def worst_overpredictions(self, n: int = 5) -> list[AuditRecord]:
        return sorted(
            self.records, key=lambda record: record.overprediction, reverse=True
        )[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "alpha": self.alpha,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": self.metrics(),
            "worst_overpredictions": [
                record.to_dict() for record in self.worst_overpredictions(10)
            ],
            "records": [record.to_dict() for record in self.records],
            "config": _config_dict(self.config),
        }


def audit_surrogate(
    config: SearchConfig | None = None,
    *,
    train_size: int = 24,
    audit_size: int = 12,
    seed: int = 0,
    mode: str = "exploit",
    alpha: float = 1.0,
    candidates_per_step: int = 256,
    exploration: float = 1.0,
) -> SurrogateAudit:
    """Train on exact evaluations, then test the surrogate on chosen rules.

    ``mode`` selects how the audit set is chosen:

    * ``exploit``: highest predicted fitness (where the surrogate is confident),
    * ``uncertain``: highest predictive standard deviation,
    * ``random``: a control sample from the same candidate pool.
    """

    config = SearchConfig() if config is None else config
    if mode not in AUDIT_MODES:
        raise ValueError(f"mode must be one of {AUDIT_MODES}")
    if train_size < 2:
        raise ValueError("train_size must be at least 2")
    if audit_size < 1:
        raise ValueError("audit_size must be at least 1")
    if candidates_per_step < audit_size:
        raise ValueError("candidates_per_step must be at least audit_size")

    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    train_genomes = initial_population(config, train_size, rng)
    train_evaluations = [evaluate_genome(genome, config) for genome in train_genomes]

    features = np.vstack(
        [genome_features(evaluation.genome) for evaluation in train_evaluations]
    )
    target = np.asarray(
        [evaluation.fitness for evaluation in train_evaluations], dtype=float
    )
    model = RidgeSurrogate(alpha=alpha).fit(features, target)
    train_r2 = model.score(features, target)

    ranked = sorted(
        train_evaluations, key=lambda item: item.fitness, reverse=True
    )
    pool = _candidate_pool(
        ranked,
        {genome.key() for genome in train_genomes},
        config,
        rng,
        candidates_per_step,
    )
    if not pool:
        raise RuntimeError("could not build an audit candidate pool")

    candidate_features = np.vstack([genome_features(genome) for genome in pool])
    mean, std = model.predict(candidate_features)

    if mode == "exploit":
        selection = mean
    elif mode == "uncertain":
        selection = std
    else:
        selection = rng.random(len(pool))

    order = np.argsort(-selection, kind="stable")
    chosen = order[:audit_size]

    records: list[AuditRecord] = []
    for index in chosen:
        genome = pool[int(index)]
        evaluation = evaluate_genome(genome, config)
        records.append(
            AuditRecord(
                genome=genome,
                predicted=float(mean[int(index)]),
                predicted_std=float(std[int(index)]),
                actual=float(evaluation.fitness),
                selected_by=mode,
            )
        )

    return SurrogateAudit(
        mode=mode,
        config=config,
        train_evaluations=train_evaluations,
        records=records,
        train_r2=train_r2,
        alpha=alpha,
        candidates=len(pool),
        elapsed_seconds=time.perf_counter() - started,
    )


@dataclass
class AuditComparison:
    audits: list[SurrogateAudit]

    def summary(self) -> list[dict[str, Any]]:
        return [audit.metrics() for audit in self.audits]

    def by_mode(self, mode: str) -> SurrogateAudit:
        for audit in self.audits:
            if audit.mode == mode:
                return audit
        raise KeyError(mode)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "audits": {audit.mode: audit.to_dict() for audit in self.audits},
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def compare_audit_modes(
    config: SearchConfig | None = None,
    *,
    train_size: int = 24,
    audit_size: int = 12,
    seed: int = 0,
    modes: Iterable[str] = AUDIT_MODES,
    alpha: float = 1.0,
    candidates_per_step: int = 256,
    exploration: float = 1.0,
) -> AuditComparison:
    """Audit the same surrogate under several candidate-selection policies."""

    config = SearchConfig() if config is None else config
    audits = [
        audit_surrogate(
            config,
            train_size=train_size,
            audit_size=audit_size,
            seed=seed,
            mode=mode,
            alpha=alpha,
            candidates_per_step=candidates_per_step,
            exploration=exploration,
        )
        for mode in modes
    ]
    return AuditComparison(audits=audits)
