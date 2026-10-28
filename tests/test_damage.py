import numpy as np

from self_repair.damage import Damage


def test_parse_damage_specs():
    assert Damage.parse("flip:0.1").kind == "flip"
    assert Damage.parse("flip:0.1").amount == 0.1
    assert Damage.parse("disk:3").amount == 3
    stripe = Damage.parse("stripe:2:col")
    assert stripe.orientation == "col"
    assert stripe.describe() == "stripe:2:col"


def test_flip_changes_exact_count():
    state = np.zeros((10, 10), dtype=np.uint8)
    result = Damage("flip", 0.2).apply(state, np.random.default_rng(0))
    assert result.changed == 20
    assert result.state.sum() == 20


def test_lesion_and_border_areas():
    state = np.ones((20, 20), dtype=np.uint8)
    lesion = Damage("lesion", 0.25).apply(state, np.random.default_rng(1))
    assert lesion.changed == 100

    border = Damage("border", 2).apply(state, np.random.default_rng(1))
    assert border.changed == 20 * 20 - 16 * 16


def test_stripe_and_disk():
    state = np.ones((10, 10), dtype=np.uint8)
    stripe = Damage("stripe", 3).apply(state, np.random.default_rng(2))
    assert stripe.changed == 30

    disk = Damage("disk", 2).apply(state, np.random.default_rng(2))
    assert 5 <= disk.changed <= 13


def test_scramble_preserves_live_count():
    state = np.zeros((12, 12), dtype=np.uint8)
    state[2:6, 2:6] = 1
    result = Damage("scramble", 0.25).apply(state, np.random.default_rng(3))
    assert result.state.sum() == state.sum()


def test_damage_is_deterministic_given_rng():
    state = np.zeros((16, 16), dtype=np.uint8)
    a = Damage("noise", 0.05).apply(state, np.random.default_rng(4))
    b = Damage("noise", 0.05).apply(state, np.random.default_rng(4))
    assert np.array_equal(a.state, b.state)
    assert a.changed == b.changed


def test_input_is_not_mutated():
    state = np.zeros((8, 8), dtype=np.uint8)
    original = state.copy()
    Damage("flip", 0.5).apply(state, np.random.default_rng(5))
    assert np.array_equal(state, original)
