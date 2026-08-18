"""Build a prior interaction network from INDRA.

Two backends produce the same thing -- a ``source``/``target``/
``evidence_count``/``source_count`` DataFrame of candidate causal edges -- from
different sources:

``nx`` (:mod:`~causomic.graph_construction.prior_extraction.nx_backend`)
    Queries a locally loaded INDRA ``networkx`` graph (typically unpickled from
    an INDRA dump). No credentials or network access needed, so this is the
    default and the path most projects should use.

``neo4j`` (:mod:`~causomic.graph_construction.prior_extraction.neo4j_backend`,
:mod:`~causomic.graph_construction.prior_extraction.neo4j_pulls`)
    Queries a live INDRA-CoGEx Neo4j instance. Needs the optional ``indra-cogex``
    dependency and an authenticated client, but sees the current database rather
    than a snapshot.

:func:`~causomic.network.extract_indra_prior` dispatches between the two; the
functions here are the pieces it composes, exposed for callers who need finer
control than the presets give.

Supporting modules:
:mod:`~causomic.graph_construction.prior_extraction.identifiers` resolves gene
symbols / UniProt accessions / chemical names to the CURIEs the Neo4j backend
requires, and
:mod:`~causomic.graph_construction.prior_extraction.formatting` turns raw INDRA
``Relation`` objects into DataFrames.
"""

from causomic.graph_construction.prior_extraction.formatting import format_query_results
from causomic.graph_construction.prior_extraction.identifiers import (
    SUPPORTED_ID_TYPES,
    resolve_curies,
)
from causomic.graph_construction.prior_extraction.neo4j_backend import (
    compound_query,
    get_four_step_root,
    get_neighbor_network,
    get_one_step_root_down,
    get_one_step_root_up,
    get_three_step_root,
    get_two_step_root,
    get_two_step_root_known_med,
    mesh_query,
    query_between_relationships,
    query_confounder_relationships,
    query_mediator_relationships,
)
from causomic.graph_construction.prior_extraction.neo4j_pulls import (
    pull_compound_data,
    pull_downstream_network,
    pull_go_data,
    pull_mesh_data,
    pull_upstream_network,
)
from causomic.graph_construction.prior_extraction.nx_backend import (
    add_evidence_info,
    filter_graph_by_evidence_count,
    prepare_graph,
    query_confounders,
    query_drug_targets,
    query_effect_nodes,
    query_forward_paths,
    query_neighborhood_paths,
)

__all__ = [
    "SUPPORTED_ID_TYPES",
    "add_evidence_info",
    "compound_query",
    "filter_graph_by_evidence_count",
    "format_query_results",
    "get_four_step_root",
    "get_neighbor_network",
    "get_one_step_root_down",
    "get_one_step_root_up",
    "get_three_step_root",
    "get_two_step_root",
    "get_two_step_root_known_med",
    "mesh_query",
    "prepare_graph",
    "pull_compound_data",
    "pull_downstream_network",
    "pull_go_data",
    "pull_mesh_data",
    "pull_upstream_network",
    "query_between_relationships",
    "query_confounder_relationships",
    "query_confounders",
    "query_drug_targets",
    "query_effect_nodes",
    "query_forward_paths",
    "query_mediator_relationships",
    "query_neighborhood_paths",
    "resolve_curies",
]
