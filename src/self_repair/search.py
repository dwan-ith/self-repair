"""Evolutionary search for rules with robust recovery behaviour.

The search is deliberately small and deterministic.  It is a baseline for
comparing against surrogate-assisted or policy-guided search later; it is not
claimed to be a state-of-the-art rule-discovery algorithm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np

from .damage import Damage
from .functional import RecoveryConfig, TrialResult, evaluate_repair
from .patterns import make_pattern_state
from .rules import RULES, TotalisticRule
from .tessellation import Boundary, MAX_NEIGHBORS, Tessellation, coerce_tessellation


@dataclass(frozen=True)
class RuleGenome:
    birth: frozenset[int]
    survival: frozenset[int]
    max_neighbors: int = 8

    def __post_init__(self) -> None:
        if self.max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")
        counts = self.birth | self.survival
        if any(c < 0 or c > self.max_neighbors for c in counts):
            raise ValueError(
                f"rule counts must be in [0, {self.max_neighbors}]: {counts}"
            )

    @classmethod
    def from_rule(cls, rule: TotalisticRule, max_neighbors: int = 8) -> "RuleGenome":
        return cls(
            birth=frozenset(rule.birth),
            survival=frozenset(rule.survival),
            max_neighbors=max_neighbors,
        )

    def to_rule(self, name: str = "") -> TotalisticRule:
        return TotalisticRule(
            birth=self.birth,
            survival=self.survival,
            name=name or self.to_bs(),
        )

    def to_bs(self) -> str:
        birth = "".join(str(c) for c in sorted(self.birth))
        survival = "".join(str(c) for c in sorted(self.survival))
        return f"B{birth}/S{survival}"

    def key(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (tuple(sorted(self.birth)), tuple(sorted(self.survival)))

    def mutate(
        self,
        rng: np.random.Generator,
        rate: float = 0.5,
    ) -> "RuleGenome":
        birth = set(self.birth)
        survival = set(self.survival)
        for target in (birth, survival):
            if rng.random() < rate:
                toggles = 1 if rng.random() < 0.75 else 2
                for _ in range(toggles):
                    count = int(rng.integers(0, self.max_neighbors + 1))
                    if count in target:
                        target.remove(count)
                    else:
                        target.add(count)
        return RuleGenome(
            birth=frozenset(birth),
            survival=frozenset(survival),
            max_neighbors=self.max_neighbors,
        )

    def crossover(self, other: "RuleGenome", rng: np.random.Generator) -> "RuleGenome":
        if other.max_neighbors != self.max_neighbors:
            raise ValueError("cannot cross genomes with different neighborhood sizes")
        birth = {
            c
            for c in range(self.max_neighbors + 1)
            if (c in self.birth if rng.random() < 0.5 else c in other.birth)
        }
        survival = {
            c
            for c in range(self.max_neighbors + 1)
            if (c in self.survival if rng.random() < 0.5 else c in other.survival)
        }
        return RuleGenome(
            birth=frozenset(birth),
            survival=frozenset(survival),
            max_neighbors=self.max_neighbors,
        )


@dataclass(frozen=True)
class SearchConfig:
    sizes: tuple[tuple[int, int], ...] = ((24, 24), (32, 32))
    seeds: tuple[int, ...] = (0,)
    damages: tuple[Damage, ...] = (
        Damage.parse("flip:0.05"),
        Damage.parse("disk:2"),
    )
    boundaries: tuple[Boundary, ...] = (Boundary.WRAP, Boundary.DEAD)
    tessellation: Tessellation = Tessellation.SQUARE_MOORE
    warmup: int = 25
    horizon: int = 50
    density: float = 0.35
    pattern: str | None = None
    pattern_position: str = "center"
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sizes", tuple(self.sizes))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "damages", tuple(self.damages))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(
            self, "tessellation", coerce_tessellation(self.tessellation)
        )

    @property
    def max_neighbors(self) -> int:
        return MAX_NEIGHBORS[self.tessellation]


@dataclass
class GenomeEvaluation:
    genome: RuleGenome
    fitness: float
    coverage: float
    mean_recovery: float
    worst_recovery: float
    recovery_std: float
    quality: float
    density_recovery: float
    trials: list[TrialResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rule": self.genome.to_bs(),
            "fitness": self.fitness,
            "coverage": self.coverage,
            "mean_recovery": self.mean_recovery,
            "worst_recovery": self.worst_recovery,
            "recovery_std": self.recovery_std,
            "quality": self.quality,
            "density_recovery": self.density_recovery,
            "trials": len(self.trials),
            "valid_trials": sum(t.valid for t in self.trials),
        }


@dataclass
class SearchResult:
    best: GenomeEvaluation
    leaderboard: list[GenomeEvaluation]
    history: list[dict]
    config: SearchConfig

    def to_dict(self) -> dict:
        return {
            "config": {
                "sizes": [list(size) for size in self.config.sizes],
                "seeds": list(self.config.seeds),
                "damages": [d.describe() for d in self.config.damages],
                "boundaries": [b.value for b in self.config.boundaries],
                "tessellation": self.config.tessellation.value,
                "warmup": self.config.warmup,
                "horizon": self.config.horizon,
                "density": self.config.density,
                "pattern": self.config.pattern,
                "pattern_position": self.config.pattern_position,
            },
            "best": self.best.to_dict(),
            "leaderboard": [item.to_dict() for item in self.leaderboard],
            "history": self.history,
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def evaluate_genome(
    genome: RuleGenome,
    config: SearchConfig,
) -> GenomeEvaluation:
    rule = genome.to_rule()
    trials: list[TrialResult] = []
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
        trials.append(
            evaluate_repair(
                rule,
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
        )

    valid = [trial for trial in trials if trial.valid]
    coverage = len(valid) / len(trials) if trials else 0.0

    if not valid:
        return GenomeEvaluation(
            genome=genome,
            fitness=-2.0,
            coverage=0.0,
            mean_recovery=0.0,
            worst_recovery=0.0,
            recovery_std=0.0,
            quality=0.0,
            density_recovery=0.0,
            trials=trials,
        )

    groups: dict[tuple, list[TrialResult]] = {}
    for trial in valid:
        key = (trial.boundary, trial.damage, trial.shape)
        groups.setdefault(key, []).append(trial)
    group_rates = [
        sum(t.recovered for t in group) / len(group)
        for group in groups.values()
    ]
    mean_recovery = float(np.mean(group_rates))
    worst_recovery = float(np.min(group_rates))
    recovery_std = float(np.std(group_rates))
    quality = float(np.mean([t.quality for t in valid]))
    density_recovery = float(np.mean([t.density_recovered for t in valid]))

    fitness = (
        0.55 * mean_recovery
        + 0.25 * worst_recovery
        + 0.20 * quality
        - 0.50 * (1.0 - coverage)
    )

    return GenomeEvaluation(
        genome=genome,
        fitness=fitness,
        coverage=coverage,
        mean_recovery=mean_recovery,
        worst_recovery=worst_recovery,
        recovery_std=recovery_std,
        quality=quality,
        density_recovery=density_recovery,
        trials=trials,
    )


def initial_population(
    config: SearchConfig,
    size: int,
    rng: np.random.Generator,
) -> list[RuleGenome]:
    max_neighbors = config.max_neighbors
    genomes: list[RuleGenome] = []
    seen: set[tuple] = set()

    for rule in RULES.values():
        if max(rule.birth | rule.survival) > max_neighbors:
            continue
        genome = RuleGenome.from_rule(rule, max_neighbors=max_neighbors)
        if genome.key() not in seen:
            seen.add(genome.key())
            genomes.append(genome)

    while len(genomes) < size:
        genome = random_genome(rng, max_neighbors)
        if genome.key() in seen:
            continue
        seen.add(genome.key())
        genomes.append(genome)

    rng.shuffle(genomes)
    return genomes[:size]


def random_genome(rng: np.random.Generator, max_neighbors: int = 8) -> RuleGenome:
    birth_count = int(rng.integers(1, min(4, max_neighbors) + 1))
    survival_count = int(rng.integers(1, min(6, max_neighbors) + 1))
    birth = rng.choice(
        np.arange(max_neighbors + 1), size=birth_count, replace=False
    )
    survival = rng.choice(
        np.arange(max_neighbors + 1), size=survival_count, replace=False
    )
    return RuleGenome(
        birth=frozenset(int(v) for v in birth),
        survival=frozenset(int(v) for v in survival),
        max_neighbors=max_neighbors,
    )


def _tournament(
    evaluations: list[GenomeEvaluation],
    rng: np.random.Generator,
    k: int = 3,
) -> GenomeEvaluation:
    indices = rng.choice(len(evaluations), size=min(k, len(evaluations)), replace=False)
    return max((evaluations[int(i)] for i in indices), key=lambda item: item.fitness)


def evolve(
    config: SearchConfig | None = None,
    *,
    population_size: int = 20,
    generations: int = 10,
    elite: int = 3,
    mutation_rate: float = 0.5,
    crossover_rate: float = 0.2,
    seed: int = 0,
    verbose: bool = False,
    on_generation: Callable[[int, list[GenomeEvaluation]], None] | None = None,
) -> SearchResult:
    config = SearchConfig() if config is None else config
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if not 0 <= elite < population_size:
        raise ValueError("elite must be in [0, population_size)")

    rng = np.random.default_rng(seed)
    population = initial_population(config, population_size, rng)
    cache: dict[tuple, GenomeEvaluation] = {}
    history: list[dict] = []

    def evaluate(genome: RuleGenome) -> GenomeEvaluation:
        key = genome.key()
        if key not in cache:
            cache[key] = evaluate_genome(genome, config)
        return cache[key]

    best: GenomeEvaluation | None = None
    for generation in range(generations):
        evaluations = [evaluate(genome) for genome in population]
        evaluations.sort(key=lambda item: item.fitness, reverse=True)
        if best is None or evaluations[0].fitness > best.fitness:
            best = evaluations[0]

        record = {
            "generation": generation,
            "best_fitness": evaluations[0].fitness,
            "best_rule": evaluations[0].genome.to_bs(),
            "mean_fitness": float(np.mean([e.fitness for e in evaluations])),
            "unique_rules": len({e.genome.key() for e in evaluations}),
            "evaluated_total": len(cache),
        }
        history.append(record)
        if verbose:
            print(
                f"gen {generation:3d}  best {record['best_fitness']:.3f}  "
                f"{record['best_rule']:<16} mean {record['mean_fitness']:.3f}  "
                f"cache {record['evaluated_total']}",
                flush=True,
            )
        if on_generation is not None:
            on_generation(generation, evaluations)

        next_population = [item.genome for item in evaluations[:elite]]
        while len(next_population) < population_size:
            parent = _tournament(evaluations, rng)
            if rng.random() < crossover_rate:
                other = _tournament(evaluations, rng)
                child = parent.genome.crossover(other.genome, rng)
            else:
                child = parent.genome
            child = child.mutate(rng, mutation_rate)
            next_population.append(child)
        population = next_population

    final_evaluations = sorted(
        (evaluate(genome) for genome in population),
        key=lambda item: item.fitness,
        reverse=True,
    )
    if best is None or final_evaluations[0].fitness > best.fitness:
        best = final_evaluations[0]

    unique: dict[tuple, GenomeEvaluation] = {}
    for evaluation in final_evaluations + list(cache.values()):
        unique.setdefault(evaluation.genome.key(), evaluation)
    leaderboard = sorted(unique.values(), key=lambda item: item.fitness, reverse=True)[:10]

    return SearchResult(
        best=best,
        leaderboard=leaderboard,
        history=history,
        config=config,
    )
