import numpy as np
import pytest

from self_repair.damage import Damage
from self_repair.functional import RecoveryConfig
from self_repair.search import RuleGenome, SearchConfig
from self_repair.surrogate import (
    RidgeSurrogate,
    compare_search,
    evolutionary_search,
    genome_features,
    random_search,
    surrogate_search,
)
from self_repair.tessellation import Boundary, Tessellation


def _config():
    return SearchConfig(
        sizes=((8, 8),),
        seeds=(0,),
        damages=(Damage("flip", 0.1),),
        boundaries=(Boundary.WRAP,),
        tessellation=Tessellation.SQUARE_MOORE,
        warmup=1,
        horizon=4,
        recovery=RecoveryConfig(
            sustain=1, min_reference_density=0.0, max_reference_density=1.0
        ),
    )


def test_genome_features_are_stable_and_finite():
    life = RuleGenome(frozenset({3}), frozenset({2, 3}))
    highlife = RuleGenome(frozenset({3, 6}), frozenset({2, 3}))
    first = genome_features(life)
    second = genome_features(life)
    other = genome_features(highlife)
    assert first.shape == other.shape
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert not np.array_equal(first, other)


def test_ridge_regression_fits_linear_data():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = 3.0 * X[:, 0] + 2.0
    model = RidgeSurrogate(alpha=1e-6).fit(X, y)
    mean, std = model.predict(X)
    assert np.allclose(mean, y, atol=1e-3)
    assert np.all(std >= 0.0)
    assert model.score(X, y) > 0.999


def test_ridge_uncertainty_grows_away_from_data():
    X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    y = np.sin(X[:, 0])
    model = RidgeSurrogate(alpha=0.1).fit(X, y)
    _, std = model.predict(np.array([[0.5], [5.0]]))
    assert std[1] > std[0]


def test_random_search_respects_budget():
    result = random_search(_config(), budget=5, seed=0)
    assert result.exact_evaluations == 5
    assert result.exact_steps > 0
    assert result.best.fitness >= -2.0


def test_evolutionary_search_respects_budget():
    result = evolutionary_search(_config(), budget=6, seed=0, population_size=4)
    assert result.exact_evaluations == 6
    assert result.best.fitness >= -2.0


def test_surrogate_search_respects_budget_and_is_reproducible():
    kwargs = dict(
        budget=8,
        seed=0,
        initial_random=4,
        batch_size=2,
        candidates_per_step=32,
    )
    first = surrogate_search(_config(), **kwargs)
    second = surrogate_search(_config(), **kwargs)
    assert first.exact_evaluations == 8
    assert first.best.genome.to_bs() == second.best.genome.to_bs()
    assert first.best.fitness == second.best.fitness
    assert len(first.history) == 2
    for record in first.history:
        assert "surrogate_r2" in record
        assert len(record["predicted"]) == len(record["actual"])


def test_compare_search_covers_all_strategies():
    result = compare_search(
        _config(),
        budget=4,
        seeds=(0,),
        evolutionary_population=3,
        surrogate_options={
            "initial_random": 2,
            "batch_size": 2,
            "candidates_per_step": 16,
        },
    )
    rows = result.summary()
    assert {row["strategy"] for row in rows} == {
        "random",
        "evolutionary",
        "surrogate",
    }
    assert all(row["runs"] == 1 for row in rows)
    assert all("best_rule" in row for row in rows)


def test_invalid_budget_is_rejected():
    with pytest.raises(ValueError):
        random_search(_config(), budget=0, seed=0)
