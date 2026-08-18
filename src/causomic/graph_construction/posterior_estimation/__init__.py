"""Learn causal structure from data, constrained by an INDRA prior.

Two learning algorithms, both restricted to edges the prior network allows:

:mod:`~causomic.graph_construction.posterior_estimation.hill_climb`
    Discrete greedy search (``SparseHillClimb``). Run many times over
    resampled or randomly-initialized data via
    :mod:`~causomic.graph_construction.posterior_estimation.bootstrap`, then
    reduced to one graph by
    :mod:`~causomic.graph_construction.posterior_estimation.selection`.

:mod:`~causomic.graph_construction.posterior_estimation.dagma`
    Continuous acyclicity-constrained optimization (``run_dagma``). A single
    fit over the full data, so no resampling or reduction step.

Supporting modules:
:mod:`~causomic.graph_construction.posterior_estimation.edge_priors` converts
INDRA evidence counts into the edge probabilities both algorithms consume;
:mod:`~causomic.graph_construction.posterior_estimation.scores` holds the
AIC/BIC scoring functions the hill climb maximizes;
:mod:`~causomic.graph_construction.posterior_estimation.causal_paths` trims a
finished graph to the nodes that matter; and
:mod:`~causomic.graph_construction.posterior_estimation.diagnostics` checks
whether learned edges reflect data or just search-path luck.

:func:`~causomic.network.estimate_posterior_dag` wires these together; import
from here when you need a single stage on its own.
"""

from causomic.graph_construction.posterior_estimation.bootstrap import (
    process_bootstrap,
    run_bootstrap,
)
from causomic.graph_construction.posterior_estimation.causal_paths import (
    filter_to_causal_subgraph,
    nodes_on_causal_paths,
)
from causomic.graph_construction.posterior_estimation.dagma import (
    evidence_penalty,
    run_dagma,
)
from causomic.graph_construction.posterior_estimation.diagnostics import (
    compare_dag_sets,
    run_single_random_init,
    search_path_diagnostic,
)
from causomic.graph_construction.posterior_estimation.edge_priors import (
    calculate_edge_probabilities,
    prepare_indra_priors,
    remove_high_corr_edges_from_blacklist,
)
from causomic.graph_construction.posterior_estimation.hill_climb import (
    SparseHillClimb,
    random_acyclic_subgraph,
)
from causomic.graph_construction.posterior_estimation.scores import (
    AICGaussIndraPriors,
    AICGaussNoPriors,
    BICGaussIndraPriors,
    BICGaussNoPriors,
)
from causomic.graph_construction.posterior_estimation.selection import (
    best_scoring_dag,
    consensus_dag,
)

__all__ = [
    "AICGaussIndraPriors",
    "AICGaussNoPriors",
    "BICGaussIndraPriors",
    "BICGaussNoPriors",
    "SparseHillClimb",
    "best_scoring_dag",
    "calculate_edge_probabilities",
    "compare_dag_sets",
    "consensus_dag",
    "evidence_penalty",
    "filter_to_causal_subgraph",
    "nodes_on_causal_paths",
    "prepare_indra_priors",
    "process_bootstrap",
    "random_acyclic_subgraph",
    "remove_high_corr_edges_from_blacklist",
    "run_bootstrap",
    "run_dagma",
    "run_single_random_init",
    "search_path_diagnostic",
]
