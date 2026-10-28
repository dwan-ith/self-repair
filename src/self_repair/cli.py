"""Command-line interface for self-repair benchmarks and rule search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audit import AUDIT_MODES, AuditComparison, compare_audit_modes
from .benchmark import (
    DEFAULT_DAMAGES,
    DEFAULT_RULES,
    DEFAULT_SIZES,
    BenchmarkReport,
    BenchmarkSuite,
    format_summary,
    run_suite,
)
from .damage import Damage
from .functional import RecoveryConfig, evaluate_repair
from .patterns import make_pattern_state
from .rules import get_rule
from .search import SearchConfig, evolve
from .stress import StressConfig, StressResult, stress_test
from .surrogate import compare_search
from .tessellation import Tessellation, coerce_boundary, coerce_tessellation
from .transfer import TransferComparison, compare_transfer_modes


def _flatten(values: list[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def _parse_ints(texts: list[str] | None) -> tuple[int, ...]:
    return tuple(int(item) for item in _flatten(texts))


def _parse_sizes(texts: list[str]) -> tuple[tuple[int, int], ...]:
    sizes = []
    for text in _flatten(texts):
        if "x" in text.lower():
            rows, cols = text.lower().split("x", 1)
            sizes.append((int(rows), int(cols)))
        else:
            size = int(text)
            sizes.append((size, size))
    if any(rows <= 0 or cols <= 0 for rows, cols in sizes):
        raise ValueError("grid dimensions must be positive")
    return tuple(sizes)


def _recovery_from_args(args: argparse.Namespace) -> RecoveryConfig:
    return RecoveryConfig(
        tolerance=args.tolerance,
        absolute_fraction=args.absolute_fraction,
        sustain=args.sustain,
        density_tolerance=args.density_tolerance,
        min_reference_density=args.min_density,
        max_reference_density=args.max_density,
    )


def _format_stress(result: StressResult) -> str:
    overall = result.overall()
    lines = [
        f"rule {result.rule_name} ({result.rule})",
        (
            f"trials {overall['trials']}  valid {overall['valid']}  "
            f"coverage {overall['coverage']:.2f}  "
            f"recovery {overall['recovery_rate']:.2f}  "
            f"quality {overall['mean_quality']:.2f}  "
            f"failures {overall['failures']}"
        ),
        "",
        "worst conditions",
        (
            f"{'boundary':<7} {'damage':<16} {'shape':<9} "
            f"{'valid':>5} {'recov':>6} {'quality':>7}  failing seeds"
        ),
        "-" * 72,
    ]
    for row in result.worst_conditions(10):
        shape = "x".join(str(value) for value in row["shape"])
        seeds = ",".join(str(seed) for seed in row["failing_seeds"][:8])
        lines.append(
            f"{row['boundary']:<7} {row['damage']:<16} {shape:<9} "
            f"{row['valid']:>5} {row['recovery_rate']:>6.2f} "
            f"{row['mean_quality']:>7.2f}  {seeds}"
        )
    failures = result.failures()
    if failures:
        lines.extend(["", "failure examples"])
        for case in failures[:5]:
            trial = case.trial
            shape = "x".join(str(value) for value in case.shape)
            lines.append(
                f"  {case.boundary} {case.damage} {shape} seed={case.seed} "
                f"damage={trial.initial_distance} "
                f"residual={trial.residual_distance:.1f} "
                f"quality={trial.quality:.2f}"
            )
    return "\n".join(lines)


def _format_comparison(rows: list[dict]) -> str:
    header = (
        f"{'strategy':<14} {'runs':>4} {'best_mean':>9} {'best_std':>8} "
        f"{'best':>6} {'exact_steps':>11} {'rule':<18}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['strategy']:<14} {row['runs']:>4} "
            f"{row['mean_best_fitness']:>9.3f} {row['std_best_fitness']:>8.3f} "
            f"{row['best_fitness']:>6.3f} {row['mean_exact_steps']:>11.0f} "
            f"{row['best_rule']:<18}"
        )
    return "\n".join(lines)


def _format_audit(comparison: AuditComparison) -> str:
    header = (
        f"{'mode':<10} {'train':>5} {'audit':>5} {'trainR2':>8} "
        f"{'auditR2':>8} {'mae':>6} {'spearman':>9} {'gap':>7} {'cov2s':>6}"
    )
    lines = [header, "-" * len(header)]
    for row in comparison.summary():
        lines.append(
            f"{row['mode']:<10} {row['train_size']:>5} {row['audit_size']:>5} "
            f"{row['train_r2']:>8.3f} {row['audit_r2']:>8.3f} {row['mae']:>6.3f} "
            f"{row['spearman']:>9.3f} {row['optimism_gap']:>7.3f} "
            f"{row['coverage_2sigma']:>6.2f}"
        )
    for audit in comparison.audits:
        worst = audit.worst_overpredictions(5)
        if not worst:
            continue
        lines.extend(["", f"worst overpredictions ({audit.mode})"])
        for record in worst:
            lines.append(
                f"  {record.rule:<18} predicted {record.predicted:>7.3f} "
                f"(+/-{record.predicted_std:.3f})  actual {record.actual:>7.3f}  "
                f"gap {record.overprediction:>7.3f}"
            )
    return "\n".join(lines)


def _format_transfer(comparison: TransferComparison) -> str:
    header = (
        f"{'mode':<10} {'transferR2':>11} {'nativeR2':>9} {'transferMAE':>12} "
        f"{'nativeMAE':>10} {'maePen':>7} {'r2Pen':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in comparison.summary():
        lines.append(
            f"{row['mode']:<10} {row['transfer_r2']:>11.3f} {row['native_r2']:>9.3f} "
            f"{row['transfer_mae']:>12.3f} {row['native_mae']:>10.3f} "
            f"{row['mae_penalty']:>7.3f} {row['r2_penalty']:>7.3f}"
        )
    for audit in comparison.audits:
        worst = audit.worst_transfer_errors(5)
        if not worst:
            continue
        lines.extend(
            [
                "",
                (
                    f"worst transfer errors ({audit.mode})  "
                    f"{audit.source.tessellation.value} -> "
                    f"{audit.target.tessellation.value}"
                ),
            ]
        )
        for record in worst:
            lines.append(
                f"  {record.rule:<18} predicted {record.predicted:>7.3f}  "
                f"actual {record.actual:>7.3f}  gap {record.overprediction:>7.3f}"
            )
    return "\n".join(lines)


def _add_common_benchmark_args(
    parser: argparse.ArgumentParser,
    *,
    seed_count: int = 3,
    warmup: int = 40,
    horizon: int = 80,
) -> None:
    parser.add_argument(
        "--sizes",
        action="append",
        help="grid size, e.g. 32x32 or 32; repeat or comma-separate",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=seed_count,
        help="number of seeds starting at 0",
    )
    parser.add_argument(
        "--seed-start", type=int, default=0, help="first seed value"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help=(
            "steps before damage "
            f"(default: 0 for patterns, {warmup} otherwise)"
        ),
    )
    parser.add_argument(
        "--horizon", type=int, default=horizon, help="steps after damage"
    )
    parser.add_argument("--density", type=float, default=0.35, help="soup density")
    parser.add_argument(
        "--pattern",
        default=None,
        help="start from a named pattern instead of a random soup",
    )
    parser.add_argument(
        "--pattern-position",
        default="center",
        choices=("center", "top_left", "random"),
        help="where to place the initial pattern",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.05, help="recovery distance tolerance"
    )
    parser.add_argument(
        "--absolute-fraction",
        type=float,
        default=0.0,
        help="absolute distance floor as a fraction of grid cells",
    )
    parser.add_argument("--sustain", type=int, default=5, help="steps at tolerance")
    parser.add_argument(
        "--density-tolerance", type=float, default=0.05, help="density match tolerance"
    )
    parser.add_argument(
        "--min-density",
        type=float,
        default=1e-3,
        help="minimum reference density for a valid trial",
    )
    parser.add_argument(
        "--max-density",
        type=float,
        default=0.999,
        help="maximum reference density for a valid trial",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self-repair",
        description=(
            "Controlled-damage experiments for self-repairing tessellation "
            "automata."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="benchmark rules and damage operators")
    run_parser.add_argument(
        "--rule",
        action="append",
        help=f"rule name or B/S spec; default: {', '.join(DEFAULT_RULES)}",
    )
    run_parser.add_argument(
        "--tessellation",
        action="append",
        help="square_moore, square_von_neumann, hexagonal, triangular, triangular_vertex",
    )
    run_parser.add_argument(
        "--boundary",
        action="append",
        help="wrap, dead or alive; repeat or comma-separate",
    )
    run_parser.add_argument(
        "--damage",
        action="append",
        help=(
            "damage spec, e.g. flip:0.05, disk:3, lesion:0.1, "
            "stripe:2:col, point:7:7"
        ),
    )
    _add_common_benchmark_args(run_parser)
    run_parser.add_argument("--output", type=Path, help="write JSON report")
    run_parser.add_argument("--csv", type=Path, help="write trial-level CSV")
    run_parser.add_argument(
        "--include-series", action="store_true", help="include distances in JSON"
    )
    run_parser.add_argument(
        "--progress", action="store_true", help="print progress to stderr"
    )

    search_parser = sub.add_parser(
        "search", help="evolve rules that repair damage robustly"
    )
    search_parser.add_argument("--generations", type=int, default=10)
    search_parser.add_argument("--population", type=int, default=20)
    search_parser.add_argument("--elite", type=int, default=3)
    search_parser.add_argument("--mutation-rate", type=float, default=0.5)
    search_parser.add_argument("--crossover-rate", type=float, default=0.2)
    search_parser.add_argument(
        "--tessellation", default="square_moore", help="tessellation to search over"
    )
    search_parser.add_argument(
        "--boundary", action="append", help="wrap, dead or alive"
    )
    search_parser.add_argument("--damage", action="append", help="damage spec")
    search_parser.add_argument(
        "--validate",
        action="store_true",
        help="evaluate the best rule on a larger held-out suite",
    )
    _add_common_benchmark_args(
        search_parser, seed_count=1, warmup=25, horizon=50
    )
    search_parser.add_argument("--seed", type=int, default=0, help="search RNG seed")
    search_parser.add_argument("--output", type=Path, help="write JSON result")
    search_parser.add_argument("--quiet", action="store_true")

    stress_parser = sub.add_parser(
        "stress", help="stress test one rule and mine failing conditions"
    )
    stress_parser.add_argument(
        "--rule", required=True, help="rule name or B/S specification"
    )
    stress_parser.add_argument(
        "--tessellation", default="square_moore", help="tessellation to test"
    )
    stress_parser.add_argument(
        "--boundary", action="append", help="wrap, dead or alive"
    )
    stress_parser.add_argument("--damage", action="append", help="damage spec")
    _add_common_benchmark_args(
        stress_parser, seed_count=8, warmup=40, horizon=80
    )
    stress_parser.add_argument("--output", type=Path, help="write JSON result")

    compare_parser = sub.add_parser(
        "compare",
        help="compare random, evolutionary and surrogate search under a budget",
    )
    compare_parser.add_argument("--budget", type=int, default=24)
    compare_parser.add_argument(
        "--search-seeds",
        default="0,1,2",
        help="comma-separated search RNG seeds",
    )
    compare_parser.add_argument(
        "--evolutionary-population", type=int, default=12
    )
    compare_parser.add_argument("--surrogate-initial", type=int, default=6)
    compare_parser.add_argument("--surrogate-batch", type=int, default=3)
    compare_parser.add_argument(
        "--surrogate-candidates", type=int, default=256
    )
    compare_parser.add_argument("--surrogate-alpha", type=float, default=1.0)
    compare_parser.add_argument("--exploration", type=float, default=1.0)
    compare_parser.add_argument(
        "--tessellation", default="square_moore", help="tessellation to search over"
    )
    compare_parser.add_argument(
        "--boundary", action="append", help="wrap, dead or alive"
    )
    compare_parser.add_argument("--damage", action="append", help="damage spec")
    _add_common_benchmark_args(
        compare_parser, seed_count=1, warmup=25, horizon=50
    )
    compare_parser.add_argument("--output", type=Path, help="write JSON result")

    audit_parser = sub.add_parser(
        "audit", help="audit surrogate predictions against exact evaluations"
    )
    audit_parser.add_argument("--train-size", type=int, default=24)
    audit_parser.add_argument("--audit-size", type=int, default=12)
    audit_parser.add_argument(
        "--audit-mode",
        action="append",
        choices=AUDIT_MODES,
        help="selection policy; repeat to compare (default: all)",
    )
    audit_parser.add_argument("--surrogate-alpha", type=float, default=1.0)
    audit_parser.add_argument("--surrogate-candidates", type=int, default=256)
    audit_parser.add_argument(
        "--search-seed", type=int, default=0, help="seed for the audit sample"
    )
    audit_parser.add_argument(
        "--tessellation", default="square_moore", help="tessellation to audit over"
    )
    audit_parser.add_argument(
        "--boundary", action="append", help="wrap, dead or alive"
    )
    audit_parser.add_argument("--damage", action="append", help="damage spec")
    _add_common_benchmark_args(audit_parser, seed_count=1, warmup=25, horizon=50)
    audit_parser.add_argument("--output", type=Path, help="write JSON result")

    transfer_parser = sub.add_parser(
        "transfer", help="audit surrogate transfer across tessellations"
    )
    transfer_parser.add_argument(
        "--source-tessellation", default="square_moore", help="lattice to train on"
    )
    transfer_parser.add_argument(
        "--target-tessellation", default="hexagonal", help="lattice to audit on"
    )
    transfer_parser.add_argument("--train-size", type=int, default=24)
    transfer_parser.add_argument("--audit-size", type=int, default=12)
    transfer_parser.add_argument(
        "--audit-mode",
        action="append",
        choices=AUDIT_MODES,
        help="selection policy; repeat to compare (default: all)",
    )
    transfer_parser.add_argument("--surrogate-alpha", type=float, default=1.0)
    transfer_parser.add_argument("--surrogate-candidates", type=int, default=256)
    transfer_parser.add_argument(
        "--search-seed", type=int, default=0, help="seed for the audit sample"
    )
    transfer_parser.add_argument(
        "--boundary", action="append", help="wrap, dead or alive"
    )
    transfer_parser.add_argument("--damage", action="append", help="damage spec")
    _add_common_benchmark_args(
        transfer_parser, seed_count=1, warmup=25, horizon=50
    )
    transfer_parser.add_argument("--output", type=Path, help="write JSON result")

    demo_parser = sub.add_parser("demo", help="small end-to-end demonstration")
    demo_parser.add_argument("--output", type=Path, help="write JSON report")

    return parser


def _command_run(args: argparse.Namespace) -> int:
    rules = _flatten(args.rule) or list(DEFAULT_RULES)
    damages = _flatten(args.damage) or list(DEFAULT_DAMAGES)
    tessellations = _flatten(args.tessellation) or [Tessellation.SQUARE_MOORE.value]
    boundaries = _flatten(args.boundary) or ["wrap", "dead"]
    sizes = _parse_sizes(args.sizes) if args.sizes else DEFAULT_SIZES
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    warmup = args.warmup if args.warmup is not None else (0 if args.pattern else 40)

    suite = BenchmarkSuite(
        rules=tuple(get_rule(rule) for rule in rules),
        tessellations=tuple(coerce_tessellation(t) for t in tessellations),
        boundaries=tuple(coerce_boundary(b) for b in boundaries),
        sizes=sizes,
        seeds=seeds,
        damages=tuple(Damage.parse(text) for text in damages),
        warmup=warmup,
        horizon=args.horizon,
        density=args.density,
        pattern=args.pattern,
        pattern_position=args.pattern_position,
        recovery=_recovery_from_args(args),
    )

    print(
        f"running {suite.trial_count} trials "
        f"({len(suite.rules)} rules, {len(suite.sizes)} sizes, "
        f"{len(suite.seeds)} seeds, {len(suite.damages)} damage types)",
        file=sys.stderr,
    )

    def progress(index: int, total: int, _trial) -> None:
        if args.progress and (index == total or index % max(1, total // 10) == 0):
            print(f"  {index}/{total}", file=sys.stderr)

    trials = run_suite(suite, progress=progress)
    report = BenchmarkReport(suite, trials)
    print(format_summary(report.summary()))

    valid = sum(trial.valid for trial in trials)
    recovered = sum(trial.recovered for trial in trials)
    print()
    print(
        f"trials {len(trials)}  valid {valid}  recovered {recovered}  "
        f"degenerate {len(trials) - valid}"
    )

    if args.output:
        report.save_json(args.output, include_series=args.include_series)
        print(f"wrote {args.output}")
    if args.csv:
        report.save_csv(args.csv)
        print(f"wrote {args.csv}")
    return 0


def _command_search(args: argparse.Namespace) -> int:
    damages = _flatten(args.damage)
    boundaries = _flatten(args.boundary)
    sizes = _parse_sizes(args.sizes) if args.sizes else SearchConfig().sizes
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    warmup = args.warmup if args.warmup is not None else (0 if args.pattern else 25)

    config = SearchConfig(
        sizes=sizes,
        seeds=seeds,
        damages=tuple(Damage.parse(d) for d in damages) if damages else SearchConfig().damages,
        boundaries=tuple(coerce_boundary(b) for b in boundaries) if boundaries else SearchConfig().boundaries,
        tessellation=coerce_tessellation(args.tessellation),
        warmup=warmup,
        horizon=args.horizon,
        density=args.density,
        pattern=args.pattern,
        pattern_position=args.pattern_position,
        recovery=_recovery_from_args(args),
    )

    result = evolve(
        config,
        population_size=args.population,
        generations=args.generations,
        elite=args.elite,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print()
    print("leaderboard")
    print("-" * 78)
    print(
        f"{'rule':<18} {'fitness':>8} {'coverage':>9} {'mean':>7} "
        f"{'worst':>7} {'quality':>8}"
    )
    for item in result.leaderboard:
        print(
            f"{item.genome.to_bs():<18} {item.fitness:>8.3f} "
            f"{item.coverage:>9.2f} {item.mean_recovery:>7.2f} "
            f"{item.worst_recovery:>7.2f} {item.quality:>8.2f}"
        )

    if args.validate:
        best_rule = result.best.genome.to_rule(
            name=f"evolved-{result.best.genome.to_bs()}"
        )
        validation_suite = BenchmarkSuite(
            rules=(best_rule,),
            tessellations=(config.tessellation,),
            boundaries=config.boundaries,
            sizes=((32, 32), (48, 48)),
            seeds=(100, 101, 102),
            damages=config.damages,
            warmup=config.warmup,
            horizon=max(80, config.horizon),
            density=config.density,
            pattern=config.pattern,
            pattern_position=config.pattern_position,
            recovery=config.recovery,
        )
        print()
        print("held-out validation")
        print(format_summary(BenchmarkReport(
            validation_suite, run_suite(validation_suite)
        ).summary()))

    if args.output:
        result.save_json(args.output)
        print(f"wrote {args.output}")
    return 0


def _command_stress(args: argparse.Namespace) -> int:
    defaults = StressConfig()
    sizes = _parse_sizes(args.sizes) if args.sizes else defaults.sizes
    damages = _flatten(args.damage) or [
        damage.describe() for damage in defaults.damages
    ]
    boundaries = _flatten(args.boundary) or [
        boundary.value for boundary in defaults.boundaries
    ]
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    warmup = (
        args.warmup if args.warmup is not None else (0 if args.pattern else defaults.warmup)
    )

    config = StressConfig(
        sizes=sizes,
        seeds=seeds,
        damages=tuple(Damage.parse(text) for text in damages),
        boundaries=tuple(coerce_boundary(b) for b in boundaries),
        tessellation=coerce_tessellation(args.tessellation),
        warmup=warmup,
        horizon=args.horizon,
        density=args.density,
        pattern=args.pattern,
        pattern_position=args.pattern_position,
        recovery=_recovery_from_args(args),
    )

    print(
        f"stress testing {args.rule} over {config.trial_count} conditions",
        file=sys.stderr,
    )
    result = stress_test(args.rule, config)
    print(_format_stress(result))

    if args.output:
        result.save_json(args.output)
        print(f"wrote {args.output}")
    return 0


def _command_compare(args: argparse.Namespace) -> int:
    defaults = SearchConfig()
    sizes = _parse_sizes(args.sizes) if args.sizes else defaults.sizes
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    search_seeds = _parse_ints(args.search_seeds)
    if not search_seeds:
        raise ValueError("--search-seeds must contain at least one seed")
    damages = _flatten(args.damage) or [
        damage.describe() for damage in defaults.damages
    ]
    boundaries = _flatten(args.boundary) or [
        boundary.value for boundary in defaults.boundaries
    ]
    warmup = args.warmup if args.warmup is not None else (0 if args.pattern else 25)

    config = SearchConfig(
        sizes=sizes,
        seeds=seeds,
        damages=tuple(Damage.parse(text) for text in damages),
        boundaries=tuple(coerce_boundary(b) for b in boundaries),
        tessellation=coerce_tessellation(args.tessellation),
        warmup=warmup,
        horizon=args.horizon,
        density=args.density,
        pattern=args.pattern,
        pattern_position=args.pattern_position,
        recovery=_recovery_from_args(args),
    )

    options = {
        "initial_random": args.surrogate_initial,
        "batch_size": args.surrogate_batch,
        "candidates_per_step": args.surrogate_candidates,
        "alpha": args.surrogate_alpha,
        "exploration": args.exploration,
    }
    print(
        f"comparing strategies with exact budget {args.budget} "
        f"over search seeds {search_seeds}",
        file=sys.stderr,
    )
    result = compare_search(
        config,
        budget=args.budget,
        seeds=search_seeds,
        evolutionary_population=args.evolutionary_population,
        surrogate_options=options,
    )
    print(_format_comparison(result.summary()))

    if args.output:
        result.save_json(args.output)
        print(f"wrote {args.output}")
    return 0


def _command_audit(args: argparse.Namespace) -> int:
    defaults = SearchConfig()
    sizes = _parse_sizes(args.sizes) if args.sizes else defaults.sizes
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    damages = _flatten(args.damage) or [
        damage.describe() for damage in defaults.damages
    ]
    boundaries = _flatten(args.boundary) or [
        boundary.value for boundary in defaults.boundaries
    ]
    warmup = args.warmup if args.warmup is not None else (0 if args.pattern else 25)
    modes = tuple(args.audit_mode) if args.audit_mode else AUDIT_MODES

    config = SearchConfig(
        sizes=sizes,
        seeds=seeds,
        damages=tuple(Damage.parse(text) for text in damages),
        boundaries=tuple(coerce_boundary(b) for b in boundaries),
        tessellation=coerce_tessellation(args.tessellation),
        warmup=warmup,
        horizon=args.horizon,
        density=args.density,
        pattern=args.pattern,
        pattern_position=args.pattern_position,
        recovery=_recovery_from_args(args),
    )

    print(
        f"auditing surrogate with {args.train_size} training rules "
        f"and {args.audit_size} audit rules per mode {modes}",
        file=sys.stderr,
    )
    comparison = compare_audit_modes(
        config,
        train_size=args.train_size,
        audit_size=args.audit_size,
        seed=args.search_seed,
        modes=modes,
        alpha=args.surrogate_alpha,
        candidates_per_step=args.surrogate_candidates,
    )
    print(_format_audit(comparison))

    if args.output:
        comparison.save_json(args.output)
        print(f"wrote {args.output}")
    return 0


def _command_transfer(args: argparse.Namespace) -> int:
    defaults = SearchConfig()
    sizes = _parse_sizes(args.sizes) if args.sizes else ((16, 16),)
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    damages = _flatten(args.damage) or [
        damage.describe() for damage in defaults.damages
    ]
    boundaries = _flatten(args.boundary) or [
        boundary.value for boundary in defaults.boundaries
    ]
    warmup = args.warmup if args.warmup is not None else (0 if args.pattern else 25)

    common = dict(
        sizes=sizes,
        seeds=seeds,
        damages=tuple(Damage.parse(text) for text in damages),
        boundaries=tuple(coerce_boundary(b) for b in boundaries),
        warmup=warmup,
        horizon=args.horizon,
        density=args.density,
        pattern=args.pattern,
        pattern_position=args.pattern_position,
        recovery=_recovery_from_args(args),
    )
    source = SearchConfig(
        tessellation=coerce_tessellation(args.source_tessellation), **common
    )
    target = SearchConfig(
        tessellation=coerce_tessellation(args.target_tessellation), **common
    )
    modes = tuple(args.audit_mode) if args.audit_mode else AUDIT_MODES

    print(
        f"transfer audit {source.tessellation.value} -> "
        f"{target.tessellation.value}, {args.train_size} training rules, "
        f"{args.audit_size} audit rules per mode {modes}",
        file=sys.stderr,
    )
    comparison = compare_transfer_modes(
        source,
        target,
        train_size=args.train_size,
        audit_size=args.audit_size,
        seed=args.search_seed,
        modes=modes,
        alpha=args.surrogate_alpha,
        candidates_per_step=args.surrogate_candidates,
    )
    print(_format_transfer(comparison))

    if args.output:
        comparison.save_json(args.output)
        print(f"wrote {args.output}")
    return 0


def _command_demo(args: argparse.Namespace) -> int:
    suite = BenchmarkSuite(
        rules=tuple(get_rule(name) for name in ("life", "maze", "coral")),
        damages=(Damage.parse("flip:0.05"), Damage.parse("disk:3")),
        sizes=((32, 32),),
        seeds=(0, 1),
        warmup=30,
        horizon=60,
    )
    report = BenchmarkReport(suite, run_suite(suite))
    print(format_summary(report.summary()))
    print()
    print(
        "Recovery means the damaged run returned to the undamaged reference "
        "trajectory for several consecutive steps. Degenerate trials (empty or "
        "saturated references) are excluded from recovery rates."
    )

    target = make_pattern_state((16, 16), "block")
    structure = evaluate_repair(
        get_rule("life"),
        target.shape,
        seed=0,
        damage=Damage("point", cells=((7, 7),)),
        warmup=0,
        horizon=8,
        recovery=RecoveryConfig(sustain=3),
        initial_state=target,
    )
    print()
    print(
        "structure example: Life block, one cell removed -> "
        f"recovered={structure.recovered} "
        f"time={structure.recovery_time} "
        f"quality={structure.quality:.2f}"
    )
    if args.output:
        report.save_json(args.output)
        print(f"wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return _command_run(args)
        if args.command == "search":
            return _command_search(args)
        if args.command == "stress":
            return _command_stress(args)
        if args.command == "compare":
            return _command_compare(args)
        if args.command == "audit":
            return _command_audit(args)
        if args.command == "transfer":
            return _command_transfer(args)
        if args.command == "demo":
            return _command_demo(args)
    except (ValueError, KeyError, IndexError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
