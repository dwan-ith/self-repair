import numpy as np
import pytest

from self_repair.damage import Damage, DamageResult
from self_repair.functional import RecoveryConfig, evaluate_repair
from self_repair.patterns import get_pattern, make_pattern_state
from self_repair.rules import get_rule


def test_block_pattern_placement():
    state = make_pattern_state((10, 10), "block")
    assert state.sum() == 4
    assert state[4:6, 4:6].tolist() == [[1, 1], [1, 1]]
    assert make_pattern_state((10, 10), "block", position="top_left")[0:2, 0:2].sum() == 4


def test_pattern_does_not_fit():
    with pytest.raises(ValueError):
        make_pattern_state((2, 2), "pulsar")


def test_unknown_pattern():
    with pytest.raises(KeyError):
        get_pattern("not-a-pattern")


def test_life_repairs_one_cell_removed_from_block():
    target = make_pattern_state((12, 12), "block")
    damaged_cell = (5, 5)
    assert target[damaged_cell] == 1

    config = RecoveryConfig(sustain=2, tolerance=0.05)
    trial = evaluate_repair(
        get_rule("life"),
        target.shape,
        seed=0,
        damage=Damage("point", cells=(damaged_cell,)),
        warmup=0,
        horizon=5,
        recovery=config,
        initial_state=target,
    )

    assert trial.valid
    assert trial.initial_distance == 1
    assert trial.recovered
    assert trial.recovery_time == 0
    assert trial.quality == 1.0


def test_point_damage_roundtrip():
    damage = Damage("point", cells=((1, 2), (3, 4)))
    assert damage.describe() == "point:1:2;3:4"
    parsed = Damage.parse(damage.describe())
    assert parsed.cells == ((1, 2), (3, 4))
    assert Damage.from_dict(damage.to_dict()).cells == ((1, 2), (3, 4))

    state = np.ones((6, 6), dtype=np.uint8)
    result = damage.apply(state, np.random.default_rng(0))
    assert isinstance(result, DamageResult)
    assert result.changed == 2
    assert result.state[1, 2] == 0 and result.state[3, 4] == 0


def test_point_damage_out_of_bounds():
    with pytest.raises(IndexError):
        Damage("point", cells=((99, 99),)).apply(
            np.ones((4, 4), dtype=np.uint8), np.random.default_rng(0)
        )
