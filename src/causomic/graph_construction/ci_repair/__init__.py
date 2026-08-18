"""Test an estimated DAG's implied independences and repair what fails.

Structure learning returns a DAG, not a guarantee. Every DAG implies a set of
conditional independences, and testing those against the data locates the places
the graph is wrong. In biological networks the usual reason is an unmeasured
common cause.

The two stages:

:mod:`~causomic.graph_construction.ci_repair.falsification`
    Convert the estimated edge list to a y0 graph and run its implied
    conditional-independence tests, returning the violations.

:mod:`~causomic.graph_construction.ci_repair.confounders`
    For each violation, search the INDRA prior for a measured variable that
    explains it. Success adds directed edges from that confounder; failure adds
    a bidirected edge recording confounding that could not be resolved.

:func:`~causomic.network.repair_confounding` runs both stages end to end.
"""

from causomic.graph_construction.ci_repair.confounders import (
    lookup_confounder_candidates,
    process_failed_test,
)
from causomic.graph_construction.ci_repair.falsification import (
    convert_to_y0_graph,
    find_failed_tests,
)

__all__ = [
    "convert_to_y0_graph",
    "find_failed_tests",
    "lookup_confounder_candidates",
    "process_failed_test",
]
