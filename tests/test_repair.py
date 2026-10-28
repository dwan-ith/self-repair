import numpy as np

from self_repair.damage import Damage
from self_repair.functional import RecoveryConfig, evaluate_repair
from self_repair.rules import TotalisticRule, get_rule
from self_repair.tessellation import Boundary, Tessellation


def _fill_rule():
    return TotalisticRule.from_bs("B012345678/S012345678", name="fill")


def _extinction_rule():
    return TotalisticRule.from_bs("B/S", name="extinction")


def test_repairing_rule_recovers_after_flip_damage():
    config = RecoveryConfig(
        tolerance=0.05,
        sustain=3,
        min_reference_density=0.0,
        max_reference_density=1.0,
    )
    trial = evaluate_repair(
        _fill_rule(),
        (16, 16),
        seed=0,
        damage=Damage("flip", 0.1),
        warmup=2,
        horizon=6,
        recovery=config,
    )
    assert trial.valid
    assert trial.initial_distance > 0
    assert trial.recovered
    assert trial.recovery_time == 0
    assert trial.quality == 1.0
    assert trial.density_recovered


def test_extinction_after_warmup_is_degenerate():
    config = RecoveryConfig(sustain=1)
    trial = evaluate_repair(
        _extinction_rule(),
        (12, 12),
        seed=0,
        damage=Damage("flip", 0.1),
        warmup=0,
        horizon=5,
        recovery=config,
    )
    assert not trial.valid
    assert trial.degenerate_reason == "reference_died"


def test_zero_damage_is_degenerate():
    config = RecoveryConfig(sustain=1)
    trial = evaluate_repair(
        get_rule("life"),
        (12, 12),
        seed=0,
        damage=Damage("flip", 0.0),
        warmup=2,
        horizon=5,
        recovery=config,
    )
    assert not trial.valid
    assert trial.degenerate_reason == "no_damage"


def test_saturated_reference_is_degenerate_by_default():
    trial = evaluate_repair(
        _fill_rule(),
        (12, 12),
        seed=0,
        damage=Damage("flip", 0.1),
        warmup=2,
        horizon=6,
    )
    assert not trial.valid
    assert trial.degenerate_reason in {"reference_saturated", "reference_saturated_end"}


def test_trial_is_deterministic():
    kwargs = dict(
        shape=(16, 16),
        seed=3,
        damage=Damage("noise", 0.03),
        warmup=5,
        horizon=10,
    )
    a = evaluate_repair(get_rule("life"), **kwargs)
    b = evaluate_repair(get_rule("life"), **kwargs)
    assert a.distances == b.distances
    assert a.recovered == b.recovered
    assert a.initial_distance == b.initial_distance


def test_initial_state_target_is_supported():
    target = np.zeros((10, 10), dtype=np.uint8)
    target[3:7, 3:7] = 1
    config = RecoveryConfig(
        sustain=2,
        min_reference_density=0.0,
        max_reference_density=1.0,
    )
    trial = evaluate_repair(
        _fill_rule(),
        target.shape,
        seed=0,
        damage=Damage("noise", 0.1),
        warmup=0,
        horizon=4,
        recovery=config,
        initial_state=target,
    )
    assert trial.valid
    assert trial.recovered


def test_life_soup_trial_has_expected_fields():
    trial = evaluate_repair(
        get_rule("life"),
        (24, 24),
        seed=1,
        damage=Damage("disk", 2),
        tessellation=Tessellation.SQUARE_MOORE,
        boundary=Boundary.WRAP,
        warmup=10,
        horizon=20,
    )
    assert trial.tessellation == "square_moore"
    assert trial.boundary == "wrap"
    assert trial.exact_steps == 2 * (10 + 20)
    assert len(trial.distances) == 20
    assert trial.to_dict()["shape"] == [24, 24]
