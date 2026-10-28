"""Run a small self-repair benchmark without the command-line interface."""

from self_repair import Damage, RecoveryConfig, get_rule
from self_repair.benchmark import BenchmarkReport, BenchmarkSuite, format_summary, run_suite


suite = BenchmarkSuite(
    rules=(get_rule("life"), get_rule("maze"), get_rule("coral"), get_rule("daynight")),
    sizes=((32, 32), (48, 48)),
    seeds=(0, 1, 2),
    damages=(Damage.parse("flip:0.05"), Damage.parse("disk:3")),
    warmup=40,
    horizon=80,
    recovery=RecoveryConfig(tolerance=0.05, sustain=5),
)

report = BenchmarkReport(suite, run_suite(suite))
print(format_summary(report.summary()))
