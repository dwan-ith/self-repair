"""Surrogate-assisted search and budgeted search comparisons.

The surrogate is intentionally simple: a ridge regressor over structural
features of a B/S rule.  It is not claimed to be a good model of arbitrary
automata.  Its purpose is to make the experiment honest: compare random search,
evolutionary search and a surrogate-guided search under the same number of
exact evaluations, and report whether the surrogate actually helps.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .search import (
    GenomeEvaluation,
    RuleGenome,
    SearchConfig,
    evaluate_genome,
    initial_population,
    random_genome,
)


def genome_features(genome: RuleGenome) -> np.ndarray:
    """Structural features for a totalistic rule.

    The feature vector is deliberately small and cheap.  It contains birth and
    survival indicators, summary statistics for each set, and a few
    interaction terms.  It does not run the automaton.
    """

    m = genome.max_neighbors
    denom = float(m) if m > 0 else 1.0
    birth = np.zeros(m + 1, dtype=float)
    survival = np.zeros(m + 1, dtype=float)
    for count in genome.birth:
        birth[count] = 1.0
    for count in genome.survival:
        survival[count] = 1.0

    def summary(mask: np.ndarray) -> list[float]:
        counts = np.flatnonzero(mask).astype(float)
        if counts.size == 0:
            return [0.0] * 6
        return [
            counts.size / (m + 1),
            counts.sum() / (m * (m + 1) / 2.0) if m > 0 else 0.0,
            counts.mean() / denom,
            counts.min() / denom,
            counts.max() / denom,
            counts.std() / denom,
        ]

    intersection = float(np.dot(birth, survival))
    union = float(np.count_nonzero(birth + survival))
    symmetric = float(np.count_nonzero(np.abs(birth - survival)))
    extras = [
        intersection / (m + 1),
        union / (m + 1),
        symmetric / (m + 1),
        1.0 if 0 in genome.birth else 0.0,
        1.0 if 0 in genome.survival else 0.0,
        1.0 if m in genome.birth else 0.0,
        1.0 if m in genome.survival else 0.0,
    ]
    return np.asarray(
        list(birth) + list(survival) + summary(birth) + summary(survival) + extras,
        dtype=float,
    )


@dataclass
class RidgeSurrogate:
    """Ridge regression with a predictive standard deviation."""

    alpha: float = 1.0
    x_mean: np.ndarray | None = field(default=None, repr=False)
    x_scale: np.ndarray | None = field(default=None, repr=False)
    coef: np.ndarray | None = field(default=None, repr=False)
    xtx_inv: np.ndarray | None = field(default=None, repr=False)
    sigma2: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        if X.shape[0] == 0:
            raise ValueError("cannot fit a surrogate on zero examples")

        self.x_mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-9] = 1.0
        self.x_scale = scale
        Z = (X - self.x_mean) / self.x_scale
        Z = np.column_stack([np.ones(Z.shape[0]), Z])

        reg = self.alpha * np.eye(Z.shape[1])
        reg[0, 0] = 0.0
        self.xtx_inv = np.linalg.pinv(Z.T @ Z + reg)
        self.coef = self.xtx_inv @ Z.T @ y

        residual = y - Z @ self.coef
        dof = max(1, X.shape[0] - Z.shape[1])
        self.sigma2 = max(float(residual @ residual / dof), 1e-8)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.coef is None or self.xtx_inv is None or self.x_scale is None:
            raise RuntimeError("surrogate has not been fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        Z = (X - self.x_mean) / self.x_scale
        Z = np.column_stack([np.ones(Z.shape[0]), Z])
        mean = Z @ self.coef
        variance = self.sigma2 * (
            1.0 + np.einsum("ij,jk,ik->i", Z, self.xtx_inv, Z)
        )
        return mean, np.sqrt(np.maximum(variance, 0.0))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        mean, _ = self.predict(X)
        y = np.asarray(y, dtype=float)
        residual = float(np.sum((y - mean) ** 2))
        total = float(np.sum((y - y.mean()) ** 2))
        if total <= 1e-12:
            return 0.0
        return 1.0 - residual / total


@dataclass
class BudgetedSearchResult:
    strategy: str
    best: GenomeEvaluation
    evaluations: list[GenomeEvaluation]
    history: list[dict[str, Any]]
    config: SearchConfig
    budget: int
    elapsed_seconds: float = 0.0

    @property
    def exact_evaluations(self) -> int:
        return len(self.evaluations)

    @property
    def exact_steps(self) -> int:
        return sum(
            trial.exact_steps
            for evaluation in self.evaluations
            for trial in evaluation.trials
        )

    def leaderboard(self, n: int = 10) -> list[GenomeEvaluation]:
        unique: dict[tuple, GenomeEvaluation] = {}
        for evaluation in self.evaluations:
            unique.setdefault(evaluation.genome.key(), evaluation)
        return sorted(
            unique.values(), key=lambda item: item.fitness, reverse=True
        )[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "budget": self.budget,
            "exact_evaluations": self.exact_evaluations,
            "exact_steps": self.exact_steps,
            "elapsed_seconds": self.elapsed_seconds,
            "best": self.best.to_dict(),
            "leaderboard": [item.to_dict() for item in self.leaderboard()],
            "evaluations": [item.to_dict() for item in self.evaluations],
            "history": self.history,
            "config": _config_dict(self.config),
        }


def _config_dict(config: SearchConfig) -> dict[str, Any]:
    return {
        "sizes": [list(size) for size in config.sizes],
        "seeds": list(config.seeds),
        "damages": [damage.describe() for damage in config.damages],
        "boundaries": [boundary.value for boundary in config.boundaries],
        "tessellation": config.tessellation.value,
        "warmup": config.warmup,
        "horizon": config.horizon,
        "density": config.density,
        "pattern": config.pattern,
        "pattern_position": config.pattern_position,
    }


def _leader(evaluated: dict[tuple, GenomeEvaluation]) -> GenomeEvaluation:
    return max(evaluated.values(), key=lambda item: item.fitness)


def _finish(
    strategy: str,
    evaluated: dict[tuple, GenomeEvaluation],
    history: list[dict[str, Any]],
    config: SearchConfig,
    budget: int,
    started: float,
) -> BudgetedSearchResult:
    if not evaluated:
        raise RuntimeError("no genomes were evaluated")
    return BudgetedSearchResult(
        strategy=strategy,
        best=_leader(evaluated),
        evaluations=list(evaluated.values()),
        history=history,
        config=config,
        budget=budget,
        elapsed_seconds=time.perf_counter() - started,
    )


def random_search(
    config: SearchConfig | None = None,
    *,
    budget: int = 24,
    seed: int = 0,
) -> BudgetedSearchResult:
    """Uniform random genomes under an exact-evaluation budget."""

    config = SearchConfig() if config is None else config
    _validate_budget(budget)
    rng = np.random.default_rng(seed)
    evaluated: dict[tuple, GenomeEvaluation] = {}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    attempts = 0

    while len(evaluated) < budget and attempts < budget * 2000:
        attempts += 1
        genome = random_genome(rng, config.max_neighbors)
        key = genome.key()
        if key in evaluated:
            continue
        evaluation = evaluate_genome(genome, config)
        evaluated[key] = evaluation
        history.append(
            {
                "step": len(evaluated),
                "best_fitness": _leader(evaluated).fitness,
                "best_rule": _leader(evaluated).genome.to_bs(),
                "fitness": evaluation.fitness,
                "rule": genome.to_bs(),
            }
        )

    return _finish("random", evaluated, history, config, budget, started)


def evolutionary_search(
    config: SearchConfig | None = None,
    *,
    budget: int = 24,
    seed: int = 0,
    population_size: int = 12,
    mutation_rate: float = 0.5,
    crossover_rate: float = 0.2,
) -> BudgetedSearchResult:
    """A (mu + lambda) evolutionary baseline under the same budget."""

    config = SearchConfig() if config is None else config
    _validate_budget(budget)
    if population_size < 2:
        raise ValueError("population_size must be at least 2")

    rng = np.random.default_rng(seed)
    evaluated: dict[tuple, GenomeEvaluation] = {}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    initial = initial_population(config, min(population_size, budget), rng)
    for genome in initial:
        if len(evaluated) >= budget:
            break
        evaluated[genome.key()] = evaluate_genome(genome, config)

    while len(evaluated) < budget:
        ranked = sorted(
            evaluated.values(), key=lambda item: item.fitness, reverse=True
        )
        parents = ranked[:population_size]
        children: list[RuleGenome] = []
        child_keys: set[tuple] = set()
        attempts = 0

        while (
            len(evaluated) + len(children) < budget
            and attempts < budget * 200
        ):
            attempts += 1
            parent = parents[int(rng.integers(0, len(parents)))]
            if len(parents) > 1 and rng.random() < crossover_rate:
                other = parents[int(rng.integers(0, len(parents)))]
                child = parent.genome.crossover(other.genome, rng)
            else:
                child = parent.genome.mutate(rng, mutation_rate)
            key = child.key()
            if key in evaluated or key in child_keys:
                continue
            child_keys.add(key)
            children.append(child)

        if not children:
            genome = random_genome(rng, config.max_neighbors)
            if genome.key() in evaluated:
                break
            children = [genome]

        for genome in children:
            if len(evaluated) >= budget:
                break
            evaluated[genome.key()] = evaluate_genome(genome, config)

        history.append(
            {
                "step": len(evaluated),
                "best_fitness": _leader(evaluated).fitness,
                "best_rule": _leader(evaluated).genome.to_bs(),
                "generation": len(history),
                "unique_rules": len(evaluated),
            }
        )

    return _finish("evolutionary", evaluated, history, config, budget, started)


def _candidate_pool(
    ranked: list[GenomeEvaluation],
    evaluated_keys: set[tuple],
    config: SearchConfig,
    rng: np.random.Generator,
    size: int,
) -> list[RuleGenome]:
    if not ranked:
        return []

    top = [item.genome for item in ranked[: min(8, len(ranked))]]
    pool: list[RuleGenome] = []
    seen = set(evaluated_keys)
    attempts = 0

    def add(genome: RuleGenome) -> None:
        key = genome.key()
        if key not in seen:
            seen.add(key)
            pool.append(genome)

    while len(pool) < size and attempts < size * 50:
        attempts += 1
        draw = rng.random()
        if draw < 0.50:
            parent = top[int(rng.integers(0, len(top)))]
            candidate = parent.mutate(rng, rate=0.5)
        elif draw < 0.75 and len(top) > 1:
            first = top[int(rng.integers(0, len(top)))]
            second = top[int(rng.integers(0, len(top)))]
            candidate = first.crossover(second, rng)
        else:
            candidate = random_genome(rng, config.max_neighbors)
        add(candidate)

    return pool


def surrogate_search(
    config: SearchConfig | None = None,
    *,
    budget: int = 24,
    seed: int = 0,
    initial_random: int = 6,
    batch_size: int = 3,
    candidates_per_step: int = 256,
    exploration: float = 1.0,
    alpha: float = 1.0,
) -> BudgetedSearchResult:
    """Use a learned rule-fitness surrogate to choose exact evaluations.

    The model is refit after every exact batch.  Candidates are ranked by an
    upper-confidence bound, so the policy can spend some evaluations on
    uncertain regions rather than only on its current best guess.
    """

    config = SearchConfig() if config is None else config
    _validate_budget(budget)
    if initial_random < 1:
        raise ValueError("initial_random must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if candidates_per_step < 1:
        raise ValueError("candidates_per_step must be at least 1")

    rng = np.random.default_rng(seed)
    evaluated: dict[tuple, GenomeEvaluation] = {}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    initial_size = min(initial_random, budget)
    initial: list[RuleGenome] = []
    initial_keys: set[tuple] = set()
    attempts = 0
    while len(initial) < initial_size and attempts < initial_size * 2000:
        attempts += 1
        genome = random_genome(rng, config.max_neighbors)
        if genome.key() in initial_keys:
            continue
        initial_keys.add(genome.key())
        initial.append(genome)

    for genome in initial:
        evaluated[genome.key()] = evaluate_genome(genome, config)

    while len(evaluated) < budget:
        ranked = sorted(
            evaluated.values(), key=lambda item: item.fitness, reverse=True
        )
        X = np.vstack([genome_features(item.genome) for item in ranked])
        y = np.asarray([item.fitness for item in ranked], dtype=float)
        model = RidgeSurrogate(alpha=alpha).fit(X, y)
        candidates = _candidate_pool(
            ranked, set(evaluated), config, rng, candidates_per_step
        )
        if not candidates:
            break

        CX = np.vstack([genome_features(genome) for genome in candidates])
        mean, std = model.predict(CX)
        acquisition = mean + exploration * std
        take = min(batch_size, budget - len(evaluated))
        chosen_indices = np.argsort(-acquisition)[:take]
        chosen = [candidates[int(index)] for index in chosen_indices]
        predicted = [float(mean[int(index)]) for index in chosen_indices]
        actual: list[float] = []

        for genome in chosen:
            evaluation = evaluate_genome(genome, config)
            evaluated[genome.key()] = evaluation
            actual.append(float(evaluation.fitness))

        history.append(
            {
                "step": len(evaluated),
                "best_fitness": _leader(evaluated).fitness,
                "best_rule": _leader(evaluated).genome.to_bs(),
                "surrogate_r2": model.score(X, y),
                "predicted": predicted,
                "actual": actual,
                "candidates": len(candidates),
                "evaluated": len(evaluated),
            }
        )

    return _finish("surrogate", evaluated, history, config, budget, started)


@dataclass
class ComparisonResult:
    runs: dict[str, list[BudgetedSearchResult]]

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for strategy in ("random", "evolutionary", "surrogate"):
            runs = self.runs.get(strategy, [])
            if not runs:
                continue
            fitness = np.asarray([run.best.fitness for run in runs], dtype=float)
            steps = np.asarray([run.exact_steps for run in runs], dtype=float)
            seconds = np.asarray([run.elapsed_seconds for run in runs], dtype=float)
            best_run = max(runs, key=lambda run: run.best.fitness)
            rows.append(
                {
                    "strategy": strategy,
                    "runs": len(runs),
                    "mean_best_fitness": float(fitness.mean()),
                    "std_best_fitness": float(fitness.std()),
                    "best_fitness": float(fitness.max()),
                    "best_rule": best_run.best.genome.to_bs(),
                    "mean_exact_steps": float(steps.mean()),
                    "mean_seconds": float(seconds.mean()),
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "runs": {
                strategy: [run.to_dict() for run in runs]
                for strategy, runs in self.runs.items()
            },
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def compare_search(
    config: SearchConfig | None = None,
    *,
    budget: int = 24,
    seeds: Iterable[int] = (0,),
    evolutionary_population: int = 12,
    surrogate_options: dict[str, Any] | None = None,
) -> ComparisonResult:
    """Run all three strategies with identical exact-evaluation budgets."""

    config = SearchConfig() if config is None else config
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("at least one search seed is required")

    options = dict(surrogate_options or {})
    runs: dict[str, list[BudgetedSearchResult]] = {
        "random": [],
        "evolutionary": [],
        "surrogate": [],
    }
    for seed in seed_list:
        runs["random"].append(random_search(config, budget=budget, seed=seed))
        runs["evolutionary"].append(
            evolutionary_search(
                config,
                budget=budget,
                seed=seed,
                population_size=evolutionary_population,
            )
        )
        runs["surrogate"].append(
            surrogate_search(config, budget=budget, seed=seed, **options)
        )

    return ComparisonResult(runs=runs)


def _validate_budget(budget: int) -> None:
    if budget < 1:
        raise ValueError("budget must be at least 1")
