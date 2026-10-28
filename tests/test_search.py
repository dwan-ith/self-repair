import numpy as np

from self_repair.damage import Damage
from self_repair.functional import RecoveryConfig
from self_repair.search import (
    RuleGenome,
    SearchConfig,
    evaluate_genome,
    evolve,
    random_genome,
)
from self_repair.tessellation import Boundary, Tessellation


def _config():
    return SearchConfig(
        sizes=((12, 12),),
        seeds=(0,),
        damages=(Damage("flip", 0.1),),
        boundaries=(Boundary.WRAP,),
        tessellation=Tessellation.SQUARE_MOORE,
        warmup=2,
        horizon=8,
        recovery=RecoveryConfig(
            sustain=2, min_reference_density=0.0, max_reference_density=1.0
        ),
    )


def test_fill_genome_scores_perfectly():
    genome = RuleGenome(
        birth=frozenset(range(9)),
        survival=frozenset(range(9)),
    )
    score = evaluate_genome(genome, _config())
    assert score.coverage == 1.0
    assert score.mean_recovery == 1.0
    assert score.quality == 1.0
    assert score.fitness > 0.99


def test_genome_mutation_stays_in_range():
    rng = np.random.default_rng(0)
    genome = random_genome(rng, 8)
    for _ in range(50):
        genome = genome.mutate(rng, rate=1.0)
    assert all(0 <= c <= 8 for c in genome.birth | genome.survival)


def test_crossover_keeps_neighborhood_size():
    rng = np.random.default_rng(1)
    a = random_genome(rng, 8)
    b = random_genome(rng, 8)
    child = a.crossover(b, rng)
    assert child.max_neighbors == 8
    assert all(0 <= c <= 8 for c in child.birth | child.survival)


def test_search_returns_a_leaderboard():
    result = evolve(
        _config(),
        population_size=6,
        generations=2,
        elite=1,
        seed=0,
        verbose=False,
    )
    assert result.best is not None
    assert result.leaderboard
    assert all(item.fitness >= -2.0 for item in result.leaderboard)
    assert len(result.history) == 2
    assert result.best.genome.max_neighbors == 8


def test_search_is_reproducible():
    kwargs = dict(population_size=5, generations=1, elite=1, seed=7, verbose=False)
    a = evolve(_config(), **kwargs)
    b = evolve(_config(), **kwargs)
    assert a.best.genome.to_bs() == b.best.genome.to_bs()
    assert a.best.fitness == b.best.fitness
