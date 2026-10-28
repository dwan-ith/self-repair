import json

import pytest

from self_repair.damage import Damage
from self_repair.functional import RecoveryConfig
from self_repair.stress import StressConfig, StressResult, mine_failures, stress_test
from self_repair.tessellation import Boundary


def test_life_block_repairs_across_the_condition_grid():
    config = StressConfig(
        sizes=((16, 16),),
        seeds=(0,),
        damages=(Damage("point", cells=((7, 7),)),),
        boundaries=(Boundary.DEAD,),
        warmup=0,
        horizon=5,
        pattern="block",
        recovery=RecoveryConfig(sustain=2),
    )
    result = stress_test("life", config)
    assert isinstance(result, StressResult)
    overall = result.overall()
    assert overall["valid"] == 1
    assert overall["recovery_rate"] == 1.0
    assert overall["failures"] == 0
    assert not result.failures()


def test_stress_finds_a_genuine_failure():
    config = StressConfig(
        sizes=((16, 16),),
        seeds=(0,),
        damages=(Damage("point", cells=((7, 7),)),),
        boundaries=(Boundary.DEAD,),
        warmup=0,
        horizon=6,
        pattern="blinker",
        recovery=RecoveryConfig(sustain=2),
    )
    result = stress_test("life", config)
    overall = result.overall()
    assert overall["valid"] == 1
    assert overall["failures"] == 1

    failures = mine_failures(result)
    assert len(failures) == 1
    assert failures[0].seed == 0
    assert not failures[0].trial.recovered

    conditions = result.summary_by_condition()
    assert conditions[0]["recovery_rate"] == 0.0
    assert conditions[0]["failing_seeds"] == [0]


def test_stress_result_serializes(tmp_path):
    config = StressConfig(
        sizes=((16, 16),),
        seeds=(0,),
        damages=(Damage("point", cells=((7, 7),)),),
        boundaries=(Boundary.DEAD,),
        warmup=0,
        horizon=5,
        pattern="block",
        recovery=RecoveryConfig(sustain=2),
    )
    result = stress_test("life", config)
    path = tmp_path / "stress.json"
    result.save_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["overall"]["valid"] == 1
    assert data["worst_conditions"]
    assert len(data["cases"]) == 1


def test_empty_condition_grid_is_rejected():
    with pytest.raises(ValueError):
        StressConfig(seeds=())
