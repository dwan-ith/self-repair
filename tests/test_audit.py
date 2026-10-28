import json

import pytest

from self_repair.audit import (
    AUDIT_MODES,
    audit_surrogate,
    compare_audit_modes,
)
from self_repair.damage import Damage
from self_repair.functional import RecoveryConfig
from self_repair.search import SearchConfig
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


def _options():
    return dict(
        train_size=6,
        audit_size=4,
        seed=0,
        alpha=1.0,
        candidates_per_step=64,
    )


def test_compare_audit_modes_covers_every_mode():
    comparison = compare_audit_modes(_config(), **_options())
    assert [audit.mode for audit in comparison.audits] == list(AUDIT_MODES)
    for audit in comparison.audits:
        assert len(audit.records) == 4
        assert audit.candidates >= 4
        assert audit.train_r2 <= 1.0 + 1e-9


def test_audit_is_reproducible():
    first = audit_surrogate(_config(), mode="exploit", **_options())
    second = audit_surrogate(_config(), mode="exploit", **_options())
    assert [record.rule for record in first.records] == [
        record.rule for record in second.records
    ]
    assert first.metrics()["mae"] == second.metrics()["mae"]


def test_exploit_predicts_higher_than_random_control():
    exploit = audit_surrogate(_config(), mode="exploit", **_options())
    control = audit_surrogate(_config(), mode="random", **_options())
    assert exploit.predicted.mean() >= control.predicted.mean()
    assert all(record.selected_by == "exploit" for record in exploit.records)


def test_metrics_and_overpredictions_are_consistent():
    audit = audit_surrogate(_config(), mode="exploit", **_options())
    metrics = audit.metrics()
    for key in (
        "audit_r2",
        "mae",
        "rmse",
        "pearson",
        "spearman",
        "optimism_gap",
        "coverage_1sigma",
        "coverage_2sigma",
    ):
        assert key in metrics
    assert metrics["mae"] >= 0.0

    worst = audit.worst_overpredictions(4)
    overlaps = [record.overprediction for record in worst]
    assert overlaps == sorted(overlaps, reverse=True)
    for record in audit.records:
        assert record.error == pytest.approx(-record.overprediction)


def test_audit_comparison_serializes(tmp_path):
    comparison = compare_audit_modes(_config(), **_options())
    path = tmp_path / "audit.json"
    comparison.save_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["summary"]) == len(AUDIT_MODES)
    assert set(data["audits"]) == set(AUDIT_MODES)
    assert data["audits"]["exploit"]["worst_overpredictions"]


def test_invalid_audit_inputs_are_rejected():
    with pytest.raises(ValueError):
        audit_surrogate(_config(), mode="bogus")
    with pytest.raises(ValueError):
        audit_surrogate(_config(), train_size=1)
    with pytest.raises(ValueError):
        audit_surrogate(_config(), audit_size=0)
    with pytest.raises(ValueError):
        audit_surrogate(_config(), audit_size=8, candidates_per_step=4)
