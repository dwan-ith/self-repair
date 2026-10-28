"""Cross-geometry transfer for the rule-fitness surrogate.

A surrogate fitted on one tessellation is only meaningful elsewhere if it can
predict repair behaviour on a lattice it never saw.  This module tests exactly
that: train on a source tessellation, then audit the model on a target
tessellation, and compare it against a model trained natively on the target.

Features are padded to a shared neighbourhood size so that rules from different
lattices live in the same space.  The neighbourhood size is also supplied as an
explicit feature, so the model can learn geometry-dependent effects instead of
silently confusing, say, three-of-eight with three-of-four.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .audit import AUDIT_MODES, AuditRecord, records_metrics
from .search import (
    RuleGenome,
    SearchConfig,
    evaluate_genome,
    initial_population,
    random_genome,
)
from .surrogate import RidgeSurrogate, _config_dict, genome_features
from .tessellation import MAX_NEIGHBORS

GLOBAL_MAX_NEIGHBORS = max(MAX_NEIGHBORS.values())


def transfer_features(
    genome: RuleGenome, global_max: int = GLOBAL_MAX_NEIGHBORS
) -> np.ndarray:
    """Rule features padded to a common neighbourhood size.

    The raw birth/survival indicators are shared across geometries; the
    explicit geometry terms let the model separate lattices with different
    neighbourhood sizes.
    """

    if global_max < genome.max_neighbors:
        raise ValueError(
            "global_max must be at least the genome's neighbourhood size"
        )
    canonical = RuleGenome(
        birth=genome.birth,
        survival=genome.survival,
        max_neighbors=global_max,
    )
    base = genome_features(canonical)
    geometry = np.asarray(
        [
            genome.max_neighbors / global_max,
            1.0 if genome.max_neighbors == global_max else 0.0,
        ],
        dtype=float,
    )
    return np.concatenate([base, geometry])


def _fit_model(
    config: SearchConfig,
    train_size: int,
    rng: np.random.Generator,
    alpha: float,
) -> tuple[RidgeSurrogate, list[RuleGenome], float]:
    genomes = initial_population(config, train_size, rng)
    evaluations = [evaluate_genome(genome, config) for genome in genomes]
    features = np.vstack([transfer_features(genome) for genome in genomes])
    target = np.asarray([evaluation.fitness for evaluation in evaluations], dtype=float)
    model = RidgeSurrogate(alpha=alpha).fit(features, target)
    return model, genomes, model.score(features, target)


def _target_pool(
    config: SearchConfig,
    rng: np.random.Generator,
    size: int,
    excluded: set[tuple],
) -> list[RuleGenome]:
    pool: list[RuleGenome] = []
    seen = set(excluded)
    attempts = 0
    while len(pool) < size and attempts < size * 200:
        attempts += 1
        genome = random_genome(rng, config.max_neighbors)
        key = genome.key()
        if key in seen:
            continue
        seen.add(key)
        pool.append(genome)
    if not pool:
        raise RuntimeError("could not build a target audit pool")
    return pool


@dataclass
class TransferAudit:
    source: SearchConfig
    target: SearchConfig
    mode: str
    alpha: float
    records: list[AuditRecord]
    native_records: list[AuditRecord]
    source_train_r2: float
    native_train_r2: float
    elapsed_seconds: float = 0.0

    def transfer_metrics(self) -> dict[str, Any]:
        return records_metrics(self.records, self.mode)

    def native_metrics(self) -> dict[str, Any]:
        return records_metrics(self.native_records, self.mode)

    def penalty(self) -> dict[str, float]:
        transfer = self.transfer_metrics()
        native = self.native_metrics()
        return {
            "mae_penalty": transfer["mae"] - native["mae"],
            "r2_penalty": native["audit_r2"] - transfer["audit_r2"],
            "optimism_penalty": transfer["optimism_gap"] - native["optimism_gap"],
        }

    def summary(self) -> dict[str, Any]:
        transfer = self.transfer_metrics()
        native = self.native_metrics()
        penalty = self.penalty()
        return {
            "mode": self.mode,
            "source": self.source.tessellation.value,
            "target": self.target.tessellation.value,
            "source_train_r2": self.source_train_r2,
            "native_train_r2": self.native_train_r2,
            "transfer_r2": transfer["audit_r2"],
            "native_r2": native["audit_r2"],
            "transfer_mae": transfer["mae"],
            "native_mae": native["mae"],
            "transfer_optimism": transfer["optimism_gap"],
            "native_optimism": native["optimism_gap"],
            **penalty,
        }

    def worst_transfer_errors(self, n: int = 5) -> list[AuditRecord]:
        return sorted(
            self.records, key=lambda record: record.overprediction, reverse=True
        )[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "alpha": self.alpha,
            "elapsed_seconds": self.elapsed_seconds,
            "summary": self.summary(),
            "worst_transfer_errors": [
                record.to_dict() for record in self.worst_transfer_errors(10)
            ],
            "transfer_records": [record.to_dict() for record in self.records],
            "native_records": [record.to_dict() for record in self.native_records],
            "source_config": _config_dict(self.source),
            "target_config": _config_dict(self.target),
        }


def cross_geometry_audit(
    source: SearchConfig,
    target: SearchConfig,
    *,
    train_size: int = 24,
    audit_size: int = 12,
    seed: int = 0,
    mode: str = "exploit",
    alpha: float = 1.0,
    candidates_per_step: int = 256,
) -> TransferAudit:
    """Train on ``source``, then audit against exact evaluations on ``target``.

    A native model trained on ``target`` is fitted on the same budget so the
    report says how much accuracy was lost purely to the geometry gap.
    """

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

    source_model, _, source_r2 = _fit_model(source, train_size, rng, alpha)
    native_model, native_genomes, native_r2 = _fit_model(
        target, train_size, rng, alpha
    )

    excluded = {genome.key() for genome in native_genomes}
    pool = _target_pool(target, rng, candidates_per_step, excluded)
    pool_features = np.vstack([transfer_features(genome) for genome in pool])
    transfer_mean, transfer_std = source_model.predict(pool_features)
    native_mean, native_std = native_model.predict(pool_features)

    if mode == "exploit":
        selection = transfer_mean
    elif mode == "uncertain":
        selection = transfer_std
    else:
        selection = rng.random(len(pool))

    order = np.argsort(-selection, kind="stable")[:audit_size]

    records: list[AuditRecord] = []
    native_records: list[AuditRecord] = []
    for index in order:
        genome = pool[int(index)]
        evaluation = evaluate_genome(genome, target)
        actual = float(evaluation.fitness)
        records.append(
            AuditRecord(
                genome=genome,
                predicted=float(transfer_mean[int(index)]),
                predicted_std=float(transfer_std[int(index)]),
                actual=actual,
                selected_by=f"transfer:{mode}",
            )
        )
        native_records.append(
            AuditRecord(
                genome=genome,
                predicted=float(native_mean[int(index)]),
                predicted_std=float(native_std[int(index)]),
                actual=actual,
                selected_by=f"native:{mode}",
            )
        )

    return TransferAudit(
        source=source,
        target=target,
        mode=mode,
        alpha=alpha,
        records=records,
        native_records=native_records,
        source_train_r2=source_r2,
        native_train_r2=native_r2,
        elapsed_seconds=time.perf_counter() - started,
    )


@dataclass
class TransferComparison:
    audits: list[TransferAudit]

    def summary(self) -> list[dict[str, Any]]:
        return [audit.summary() for audit in self.audits]

    def by_mode(self, mode: str) -> TransferAudit:
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


def compare_transfer_modes(
    source: SearchConfig,
    target: SearchConfig,
    *,
    train_size: int = 24,
    audit_size: int = 12,
    seed: int = 0,
    modes: Iterable[str] = AUDIT_MODES,
    alpha: float = 1.0,
    candidates_per_step: int = 256,
) -> TransferComparison:
    audits = [
        cross_geometry_audit(
            source,
            target,
            train_size=train_size,
            audit_size=audit_size,
            seed=seed,
            mode=mode,
            alpha=alpha,
            candidates_per_step=candidates_per_step,
        )
        for mode in modes
    ]
    return TransferComparison(audits=audits)
