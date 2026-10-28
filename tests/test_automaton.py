import numpy as np
import pytest

from self_repair.automaton import Automaton, random_state
from self_repair.rules import get_rule, parse_bs
from self_repair.tessellation import Boundary, Tessellation, neighbor_count


def test_parse_bs_forms():
    assert parse_bs("B3/S23") == (frozenset({3}), frozenset({2, 3}))
    assert parse_bs("3/23") == (frozenset({3}), frozenset({2, 3}))
    assert parse_bs("b36s23") == (frozenset({3, 6}), frozenset({2, 3}))
    assert parse_bs("B/S") == (frozenset(), frozenset())
    assert get_rule("life").to_bs() == "B3/S23"
    with pytest.raises(ValueError):
        parse_bs("BEEHIVE")
    with pytest.raises(KeyError):
        get_rule("beacon")


def test_life_blinker():
    life = get_rule("life")
    state = np.zeros((5, 5), dtype=np.uint8)
    state[2, 1:4] = 1
    nxt = life.step(state, Tessellation.SQUARE_MOORE, Boundary.DEAD)
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:4, 2] = 1
    assert np.array_equal(nxt, expected)
    assert np.array_equal(
        life.step(nxt, Tessellation.SQUARE_MOORE, Boundary.DEAD), state
    )


def test_life_glider_translates():
    life = get_rule("life")
    state = np.zeros((12, 12), dtype=np.uint8)
    glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)
    state[2:5, 2:5] = glider
    moved = Automaton(life, boundary=Boundary.WRAP).run(state, 4)

    expected = np.zeros_like(state)
    expected[3:6, 3:6] = glider
    assert np.array_equal(moved, expected)


def test_square_neighbor_counts_and_boundaries():
    state = np.ones((3, 3), dtype=np.uint8)
    moore = neighbor_count(state, Tessellation.SQUARE_MOORE, Boundary.WRAP)
    assert np.all(moore == 8)

    dead = neighbor_count(state, Tessellation.SQUARE_MOORE, Boundary.DEAD)
    assert dead[1, 1] == 8
    assert dead[0, 0] == 3

    alive = neighbor_count(state, Tessellation.SQUARE_MOORE, Boundary.ALIVE)
    assert alive[0, 0] == 8

    von_neumann = neighbor_count(
        state, Tessellation.SQUARE_VON_NEUMANN, Boundary.WRAP
    )
    assert np.all(von_neumann == 4)


def test_hex_and_triangular_neighbor_counts():
    state = np.ones((7, 7), dtype=np.uint8)
    hex_counts = neighbor_count(state, Tessellation.HEXAGONAL, Boundary.WRAP)
    assert np.all(hex_counts == 6)

    tri_counts = neighbor_count(state, Tessellation.TRIANGULAR, Boundary.WRAP)
    assert np.all(tri_counts == 3)

    vertex_counts = neighbor_count(
        state, Tessellation.TRIANGULAR_VERTEX, Boundary.WRAP
    )
    assert np.all(vertex_counts == 12)


def test_random_state_density_and_seed():
    a = random_state((50, 50), 0.3, np.random.default_rng(7))
    b = random_state((50, 50), 0.3, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert 0.2 < a.mean() < 0.4
