import csv
import json

from self_repair.benchmark import BenchmarkReport, BenchmarkSuite, format_summary, run_suite
from self_repair.damage import Damage
from self_repair.functional import RecoveryConfig
from self_repair.rules import TotalisticRule


def _suite():
    return BenchmarkSuite(
        rules=(
            TotalisticRule.from_bs("B012345678/S012345678", name="fill"),
            TotalisticRule.from_bs("B/S", name="extinction"),
        ),
        sizes=((12, 12),),
        seeds=(0,),
        damages=(Damage("flip", 0.1),),
        boundaries=("wrap",),
        warmup=2,
        horizon=6,
        recovery=RecoveryConfig(
            sustain=2, min_reference_density=1e-3, max_reference_density=1.0
        ),
    )


def test_suite_runs_and_summarizes():
    suite = _suite()
    assert suite.trial_count == 2
    report = BenchmarkReport(suite, run_suite(suite))
    rows = report.summary()
    assert len(rows) == 2
    assert rows[0]["rule"] == "B012345678/S012345678"
    assert rows[0]["coverage"] == 1.0
    assert rows[0]["recovery_rate"] == 1.0
    assert rows[1]["coverage"] == 0.0
    assert "B012345678/S012345678" in format_summary(rows)


def test_pattern_suite_repairs_a_block():
    suite = BenchmarkSuite(
        rules=("life",),
        sizes=((12, 12),),
        seeds=(0,),
        damages=(Damage("point", cells=((5, 5),)),),
        boundaries=("dead",),
        warmup=0,
        horizon=5,
        pattern="block",
        recovery=RecoveryConfig(sustain=2),
    )
    trials = run_suite(suite)
    assert len(trials) == 1
    assert trials[0].valid
    assert trials[0].recovered
    assert trials[0].recovery_time == 0


def test_report_writes_json_and_csv(tmp_path):
    report = BenchmarkReport(_suite(), run_suite(_suite()))
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "trials.csv"
    report.save_json(json_path, include_series=True)
    report.save_csv(csv_path)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data["trials"]) == 2
    assert "distances" in data["trials"][0]
    assert data["summary"][0]["rule_name"] == "fill"

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["shape"] == "12x12"
