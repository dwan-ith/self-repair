import json

import numpy as np
import pytest

from self_repair.audit import AUDIT_MODES
from self_repair.damage import Damage
from self_repair.functional import RecoveryConfig
from self_repair.search import RuleGenome, SearchConfig
from self_repair.tessellation import Boundary, Tessellation
from self_repair.transfer import (
    GLOBAL_MAX_NEIGHBORS,
    compare_transfer_modes,
    cross_geometry_audit,
    transfer_features,
)


def _config(tessellation):
    return SearchConfig(
        sizes=((8, 8),),
        seeds=(0,),
        damages=(Damage("flip", 0.1),),
        boundaries=(Boundary.WRAP,),
        tessellation=tessellation,
        warmup=1,
        horizon=4,
        recovery=RecoveryConfig(
            sustain=1, min_reference_density=0.0, max_reference_density=1.0
        ),
    )


def _options():
    return dict(
        train_size=6,
        audit_size=4,
        seed=0,
        alpha=1.0,
        candidates_per_step=32,
    )


def test_transfer_features_share_a_common_space():
    moore = RuleGenome(frozenset({3}), frozenset({2, 3}), max_neighbors=8)
    von_neumann = RuleGenome(frozenset({3}), frozenset({2, 3}), max_neighbors=4)
    wide = transfer_features(moore)
    narrow = transfer_features(von_neumann)
    assert wide.shape == narrow.shape
    expected = 2 * (GLOBAL_MAX_NEIGHBORS + 1) + 12 + 7 + 2
    assert wide.shape[0] == expected
    assert np.all(np.isfinite(wide))
    assert narrow[-2] < wide[-2]


def test_transfer_features_reject_impossible_padding():
    genome = RuleGenome(frozenset({3}), frozenset({2, 3}), max_neighbors=8)
    with pytest.raises(ValueError):
        transfer_features(genome, global_max=4)


def test_cross_geometry_audit_reports_penalty():
    result = cross_geometry_audit(
        _config(Tessellation.SQUARE_MOORE),
        _config(Tessellation.SQUARE_VON_NEUMANN),
        mode="exploit",
        **_options(),
    )
    assert len(result.records) == 4
    assert len(result.native_records) == 4
    assert [record.rule for record in result.records] == [
        record.rule for record in result.native_records
    ]

    summary = result.summary()
    for key in (
        "transfer_r2",
        "native_r2",
        "transfer_mae",
        "native_mae",
        "mae_penalty",
        "r2_penalty",
    ):
        assert key in summary
    penalty = result.penalty()
    assert penalty["mae_penalty"] == pytest.approx(
        summary["transfer_mae"] - summary["native_mae"]
    )


def test_transfer_audit_is_reproducible():
    source = _config(Tessellation.SQUARE_MOORE)
    target = _config(Tessellation.SQUARE_VON_NEUMANN)
    first = cross_geometry_audit(source, target, mode="random", **_options())
    second = cross_geometry_audit(source, target, mode="random", **_options())
    assert [record.rule for record in first.records] == [
        record.rule for record in second.records
    ]
    assert first.summary()["transfer_mae"] == second.summary()["transfer_mae"]


def test_compare_transfer_modes_covers_all_modes(tmp_path):
    comparison = compare_transfer_modes(
        _config(Tessellation.SQUARE_MOORE),
        _config(Tessellation.HEXAGONAL),
        **_options(),
    )
    assert [audit.mode for audit in comparison.audits] == list(AUDIT_MODES)
    path = tmp_path / "transfer.json"
    comparison.save_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["audits"]) == set(AUDIT_MODES)
    assert data["summary"][0]["source"] == "square_moore"
    assert data["audits"]["exploit"]["worst_transfer_errors"]


def test_invalid_transfer_inputs_are_rejected():
    source = _config(Tessellation.SQUARE_MOORE)
    target = _config(Tessellation.HEXAGONAL)
    with pytest.raises(ValueError):
        cross_geometry_audit(source, target, mode="bogus")
    with pytest.raises(ValueError):
        cross_geometry_audit(source, target, train_size=1)
    with pytest.raises(ValueError):
        cross_geometry_audit(source, target, audit_size=8, candidates_per_step=4)
