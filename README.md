# Self-Repair

Controlled-damage experiments for tessellation automata.

This project asks a narrow, testable question:

> If a system is perturbed, does it return to the same functional state, and
> does it keep doing so across seeds, sizes, boundaries and damage types?

The repository provides an exact automaton engine, a library of damage
operators, a recovery metric with explicit degeneracy guards, a benchmark
matrix, and an evolutionary search for rules that repair damage robustly.

It is deliberately an exact baseline. A learned surrogate or RL policy should
be compared against this harness rather than trusted on its own.

## Why the measurement needs care

A naive "distance to the undamaged run" metric produces false positives. If
both the reference and the damaged system die, their distance goes to zero.
If the reference saturates to an all-live grid, damage is trivially overwritten.
A chaotic automaton can look permanently broken at the cell level while
remaining statistically similar at the level of density or structure.

The harness therefore reports both exact recovery and density recovery, and
marks a trial degenerate when:

- no cells actually changed (`no_damage`),
- the reference was already empty or saturated after warm-up,
- the reference died or saturated during the measurement horizon.

Degenerate trials are excluded from recovery rates but retained in the raw
results. Coverage (the fraction of non-degenerate trials) is reported
separately, so a rule cannot score well by making every trial meaningless.

## Functional state and recovery

For each trial:

1. Start a reference run and a damaged run from the same initial configuration.
2. Advance both for `warmup` steps.
3. Apply a controlled perturbation to the damaged copy only.
4. Advance both for `horizon` steps.
5. Record the Hamming distance between the two states after every step.

A trial counts as recovered when the distance stays at or below

```text
max(tolerance * initial_damage, absolute_fraction * number_of_cells)
```

for `sustain` consecutive steps. `quality` is the fraction of the initial
damage that is absent at the end of the run, clipped to `[0, 1]`. Density
recovery is measured separately on the absolute difference between live-cell
fractions.

This definition measures return to the *undamaged trajectory*, not merely
survival. For a chaotic rule, exact recovery is expected to fail; that is a
result about the system, not a bug in the metric.

## Experiment matrix

A benchmark suite crosses:

- rules (`B3/S23`, `B36/S23`, arbitrary B/S specifications),
- tessellations (square Moore, square von Neumann, hexagonal, triangular
  edge-adjacency, triangular vertex-sharing),
- boundary conditions (wrap, dead, alive),
- grid sizes,
- random seeds,
- damage operators and intensities,
- warm-up and recovery horizons.

Damage operators:

| Operator | Meaning |
| --- | --- |
| `flip:f` | flip a random fraction `f` of cells |
| `noise:f` | flip each cell independently with probability `f` |
| `kill:f` | set a random fraction of cells to dead |
| `lesion:f` | zero a random rectangle covering fraction `f` |
| `disk:r` | zero a disk of radius `r` |
| `stripe:w[:row|col]` | zero a band of width `w` |
| `border:d` | zero a border of depth `d` |
| `scramble:f` | permute cell values inside a random rectangle |
| `point:r:c[;r:c...]` | zero specific cells (deterministic) |

## Starting from a structure

A random soup is useful for statistical repair, but "repair" often refers to a
particular structure. The `patterns` module places named structures into a
grid, and the same recovery harness measures whether damage to that structure
is undone.

```powershell
self-repair run --rule life --pattern block --sizes 16x16 `
  --seed-count 1 --warmup 0 --horizon 8 `
  --damage point:7:7 --boundary dead
```

A Life block with one corner removed is a three-cell L triomino, which becomes
a block again on the next update. The trial reports `recovered=True` with
`recovery_time=0`.

Available patterns include `block`, `beehive`, `loaf`, `boat`, `tub`,
`blinker`, `toad`, `beacon`, `glider`, `lwss`, `pulsar` and `single`.

For pattern experiments use `--warmup 0` unless you intentionally want the
pattern to evolve before damage. Large patterns must fit inside the chosen
grid.

## Metrics

Per rule and tessellation the summary reports:

- `coverage`: fraction of trials that were non-degenerate,
- `recovery_rate`: recovered / valid trials,
- `recovery_rate_all`: recovered / all trials,
- `density_recovery_rate`: density-level recovery / valid trials,
- `quality`: mean residual-damage quality on valid trials,
- `worst_condition_recovery`: worst recovery rate over
  boundary/damage/size conditions,
- `condition_recovery_std`: variation across those conditions,
- `median_recovery_time`,
- `score = coverage * recovery_rate * (0.5 + 0.5 * quality)`.

`worst_condition_recovery` is the robustness signal. A rule that repairs only
one convenient condition should not outrank a rule that repairs several.

## Install

```powershell
python -m pip install -e .
```

The only runtime dependency is NumPy. Tests use pytest.

## Quick start

```powershell
self-repair demo
```

Benchmark named rules:

```powershell
self-repair run --rule life --rule maze --rule coral `
  --sizes 32x32,48x48 --seed-count 3 --warmup 40 --horizon 80 `
  --damage flip:0.05 --damage disk:3 `
  --boundary wrap --boundary dead `
  --output results/benchmark.json --csv results/trials.csv
```

Search for robustly repairing rules:

```powershell
self-repair search --generations 12 --population 24 `
  --sizes 24x24,32x32 --seed-count 2 `
  --damage flip:0.05 --damage disk:2 `
  --validate --output results/search.json
```

Search for rules that repair a specific structure:

```powershell
self-repair search --pattern block --warmup 0 --horizon 8 `
  --sizes 16x16 --damage point:7:7 --boundary dead `
  --generations 8 --population 16
```

Compare search strategies under an exact-evaluation budget:

```powershell
self-repair compare --budget 24 --search-seeds 0,1,2 --output results/compare.json
```

Stress test a rule and mine failing conditions:

```powershell
self-repair stress --rule life --sizes 32x32 --seed-count 4 `
  --damage flip:0.1 --boundary wrap --output results/stress.json
```

Audit whether the surrogate's confident predictions hold up exactly:

```powershell
self-repair audit --train-size 16 --audit-size 8 --output results/audit.json
```

Search over another tessellation:

```powershell
self-repair search --tessellation hexagonal --generations 10
```

Run without installing:

```powershell
python -m self_repair demo
```

## Search

`self_repair.search` is a small evolutionary baseline over birth/survival
counts. It uses:

- a cache keyed by rule,
- tournament selection,
- mutation and optional crossover,
- elitism,
- a fitness that rewards mean recovery, worst-condition recovery and quality,
  while penalising low coverage.

The search is not presented as state of the art. Its purpose is to establish
the exact-evaluation baseline that a surrogate-assisted or policy-guided
search must beat. A useful comparison would hold the exact-step budget constant
and measure confirmed discoveries per exact evaluation, not raw inference
speed.

## Surrogate-assisted search

`self_repair.surrogate` adds a learned model over rule structure and uses it to
choose which rules deserve an expensive exact evaluation.

- `genome_features` turns a B/S rule into a small structural feature vector.
- `RidgeSurrogate` fits rule fitness with a predictive standard deviation.
- `surrogate_search` refits the model after every exact batch and ranks new
  candidates by an upper-confidence bound, so it explores uncertain regions
  instead of only exploiting its current prediction.
- `random_search` and `evolutionary_search` are budget-matched baselines.

Every search counts exact evaluations, not inference calls:

```powershell
self-repair compare --budget 24 --search-seeds 0,1,2 `
  --sizes 16x16 --seed-count 1 --warmup 10 --horizon 30 `
  --damage flip:0.1 --boundary wrap --output results/compare.json
```

`self-repair compare` runs all three strategies under identical exact budgets
and reports best fitness, robustness across search seeds and exact steps spent.
The honest expectation is that a surrogate often does not help on small, cheap
rule families; the point is to measure that rather than assume it.

## Auditing the surrogate

A surrogate is only useful if its confident predictions survive contact with the
exact automaton. `self_repair.audit` trains the model on exact evaluations, then
picks candidates by three policies and records predicted versus exact fitness:

- `exploit`: highest predicted fitness (where the model is confident),
- `uncertain`: highest predictive standard deviation,
- `random`: a control sample from the same candidate pool.

```powershell
self-repair audit --pattern block --warmup 0 --horizon 10 --sizes 16x16 `
  --damage point:7:7 --boundary dead `
  --train-size 16 --audit-size 8 --output results/audit.json
```

Each mode reports in-sample and audit-set R², MAE, Pearson and Spearman
correlation, the optimism gap (mean predicted minus exact) and whether the
predictive band covers the exact value. It then lists the rules where the
surrogate was most over-confident. This is the point of the whole exercise: a
model can fit its training rules beautifully (high `train_r2`) and still be
useless or actively misleading on the rules it chose to trust. The exact
automaton is the only judge.

## Cross-geometry transfer

A surrogate trained on one lattice should not be trusted on another without a
test. `self_repair.transfer` pads rule features to a shared neighbourhood size
and adds the neighbourhood size as an explicit feature, then trains on a source
tessellation and audits on a target one. A native model trained on the target
under the same budget is the reference point.

```powershell
self-repair transfer --source-tessellation square_moore `
  --target-tessellation hexagonal `
  --sizes 16x16 --warmup 10 --horizon 20 `
  --train-size 12 --audit-size 6 --output results/transfer.json
```

The report shows transfer versus native R² and MAE, the MAE/R² penalties, and
the rules whose transferred predictions were most wrong. Typically the native
model wins by a wide margin: rule behaviour is tied to neighbourhood size, so a
model fitted on eight-neighbour cells does not carry over to six-neighbour
cells for free.

## Stress testing and failure mining

A rule that repairs on one condition may fail on another. `self_repair.stress`
enumerates a grid of sizes, seeds, damages and boundaries for a single rule,
then reports the worst conditions and the concrete failing trials.

```powershell
self-repair stress --rule life --pattern blinker --sizes 16x16 `
  --seed-count 1 --warmup 0 --horizon 6 `
  --damage point:7:7 --boundary dead
```

```powershell
self-repair stress --rule B3678/S34678 --sizes 32x32,64x64 `
  --seed-count 8 --warmup 40 --horizon 80 `
  --damage flip:0.05 --damage flip:0.2 --damage disk:3 `
  --boundary wrap --boundary dead --output results/stress.json
```

`summary_by_condition()` ranks conditions by recovery rate and coverage, while
`failures()` returns valid trials that did not recover, ordered by residual
quality. This is the adversarial half of the loop: the exact automaton remains
the judge, and every claimed repair is challenged with unfamiliar conditions.

## Tessellations

The storage array is always 2-D. For non-square lattices it is a coordinate
representation, not a picture of physical space.

- `square_moore`: eight neighbors.
- `square_von_neumann`: four neighbors.
- `hexagonal`: six axial neighbors.
- `triangular`: three edge-neighbors; adjacency alternates with the parity of
  `row + col` (the honeycomb adjacency graph of edge-sharing triangles).
- `triangular_vertex`: the six axial neighbors plus six at double distance
  (twelve vertex-sharing neighbors).

All rules are totalistic binary rules. This is enough to compare repair
behaviour, but it excludes non-totalistic rules such as the Outlier rule. A
generic local lookup-table rule is a natural extension.

## Project layout

```text
src/self_repair/
  tessellation.py   neighbor geometry and boundary handling
  rules.py          B/S parsing and totalistic stepping
  automaton.py      exact synchronous automaton and initial conditions
  damage.py         controlled perturbations
  patterns.py       named structures and placement
  functional.py     reference/damage recovery measurement
  benchmark.py      experiment matrix, summaries, CSV/JSON output
  search.py         evolutionary rule search
  surrogate.py      rule features, ridge surrogate, budgeted comparisons
  audit.py          calibration and over-confidence audit of the surrogate
  transfer.py       cross-geometry surrogate transfer audit
  stress.py         condition grids, failure mining for one rule
  cli.py            command-line interface
tests/              unit and integration tests
examples/           library-level example
```

## Interpretation and limitations

- Exact recovery is stricter than functional recovery. A system can fail the
  Hamming test while preserving its density, structure or qualitative regime.
  Both are reported.
- A finite random soup is not a controlled target pattern. Results depend on
  the initial-condition distribution and warm-up length; both are explicit
  parameters. Use `--pattern` and `--warmup 0` for structure-level experiments.
- The default density bounds classify empty and saturated references as
  degenerate. This excludes trivial "repair by extinction" and "repair by
  filling everything".
- Totalistic rules cannot express all interesting automata. The search results
  should be read as evidence about this rule family, not about all cellular
  automata.
- This repository measures behaviour and learns a small, deliberately auditable
  surrogate. It does not yet train a control policy.

## Roadmap

Surrogate-assisted search, surrogate auditing, cross-geometry transfer and
condition-level failure mining are implemented. The remaining extensions are:

1. **Policy-guided design.** Treat rule/topology/initial-condition edits as
   actions and learn a search policy that transfers across rule families and
   tessellations.
2. **Learned effective dynamics.** Search for compressed state representations
   and transition rules that predict coarse behaviour when microscopic
   forecasting fails.
3. **Transfer-aware search.** Use the transfer penalty as a search objective, so
   that the search prefers rules that generalise across tessellations.

Each extension should preserve the same rule: the exact automaton is the
oracle, and every claimed repair is a verified property of the exact system.

## Tests

```powershell
python -m pytest
```
