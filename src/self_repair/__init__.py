"""Self-repair benchmarks for tessellation automata.

The package separates four concerns:

* exact automata (tessellation, rule, stepping),
* controlled damage operators,
* functional-recovery measurement,
* benchmark, search, surrogate, audit, transfer and stress-test harnesses.
"""

from .audit import (
    AUDIT_MODES,
    AuditComparison,
    AuditRecord,
    SurrogateAudit,
    audit_surrogate,
    compare_audit_modes,
)
from .automaton import Automaton, random_state, run
from .damage import Damage, DamageResult
from .functional import RecoveryConfig, TrialResult, evaluate_repair
from .patterns import PATTERNS, get_pattern, make_pattern_state
from .rules import RULES, TotalisticRule, get_rule, parse_bs
from .stress import StressConfig, StressResult, mine_failures, stress_test
from .surrogate import (
    BudgetedSearchResult,
    ComparisonResult,
    RidgeSurrogate,
    compare_search,
    evolutionary_search,
    genome_features,
    random_search,
    surrogate_search,
)
from .tessellation import Boundary, Tessellation, neighbor_count
from .transfer import (
    TransferAudit,
    TransferComparison,
    compare_transfer_modes,
    cross_geometry_audit,
    transfer_features,
)

__all__ = [
    "AUDIT_MODES",
    "AuditComparison",
    "AuditRecord",
    "Automaton",
    "Boundary",
    "BudgetedSearchResult",
    "ComparisonResult",
    "Damage",
    "DamageResult",
    "PATTERNS",
    "RULES",
    "RecoveryConfig",
    "RidgeSurrogate",
    "StressConfig",
    "StressResult",
    "SurrogateAudit",
    "Tessellation",
    "TotalisticRule",
    "TransferAudit",
    "TransferComparison",
    "TrialResult",
    "audit_surrogate",
    "compare_audit_modes",
    "compare_search",
    "compare_transfer_modes",
    "cross_geometry_audit",
    "transfer_features",
    "evaluate_repair",
    "evolutionary_search",
    "genome_features",
    "get_pattern",
    "get_rule",
    "make_pattern_state",
    "mine_failures",
    "neighbor_count",
    "parse_bs",
    "random_search",
    "random_state",
    "run",
    "stress_test",
    "surrogate_search",
]
