"""Top-level entry points for building a causal network with causomic.

One function per stage of the pipeline, in the order you call them:

1. :func:`extract_indra_prior` -- pull a candidate edge set out of INDRA. Reads
   a local ``networkx`` graph by default; pass ``backend="neo4j"`` to query a
   live INDRA-CoGEx instance instead.
2. :func:`estimate_posterior_dag` -- learn which of those candidate edges the
   data supports, by constrained hill climbing or DAGMA.
3. :func:`repair_confounding` -- test the learned graph's implied conditional
   independences and repair failures with confounders found in the prior.

Each is a thin composition over
:mod:`causomic.graph_construction`; reach into
:mod:`~causomic.graph_construction.prior_extraction`,
:mod:`~causomic.graph_construction.posterior_estimation`, or
:mod:`~causomic.graph_construction.ci_repair` directly when you need a single
step or a component these entry points don't expose.

:func:`filter_to_causal_subgraph` and :func:`nodes_on_causal_paths` are
re-exported here as well, for trimming an estimated graph to the nodes that lie
between your exposures and outcomes.
"""

import copy
from collections import Counter
from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pgmpy.estimators import ExpertKnowledge
from tqdm import tqdm
from y0.dsl import Variable
from y0.graph import NxMixedGraph

try:
    from indra_cogex.client import Neo4jClient
except ImportError:  # optional dependency, see causomic._optional
    from causomic._optional import missing_cogex as Neo4jClient

from causomic.graph_construction.ci_repair import (
    convert_to_y0_graph,
    find_failed_tests,
    lookup_confounder_candidates,
    process_failed_test,
)
from causomic.graph_construction.posterior_estimation import (
    BICGaussIndraPriors,
    SparseHillClimb,
    best_scoring_dag,
    consensus_dag,
    filter_to_causal_subgraph,
    nodes_on_causal_paths,
    prepare_indra_priors,
    run_bootstrap,
    run_dagma,
)
from causomic.graph_construction.prior_extraction import (
    format_query_results,
    get_one_step_root_down,
    get_three_step_root,
    get_two_step_root_known_med,
    prepare_graph,
    query_forward_paths,
    resolve_curies,
)

__all__ = [
    "estimate_posterior_dag",
    "extract_indra_prior",
    "filter_to_causal_subgraph",
    "nodes_on_causal_paths",
    "repair_confounding",
]

# Statement types treated as causal for prior extraction. Amount changes are the
# right readout for total-protein abundance data; add phospho-specific types via
# `stmt_types` when the measurements are themselves phosphorylation levels.
CAUSAL_STMT_TYPES = ["IncreaseAmount", "DecreaseAmount"]

PRIOR_COLUMNS = ["source", "target", "evidence_count", "source_count"]


def _normalize_prior_network(
    relations: pd.DataFrame, verbose: bool = True, label: str = "edges"
) -> pd.DataFrame:
    """Collapse a raw relation table into the canonical prior-network contract.

    Both backends funnel through here so their output is interchangeable:
    hyphens stripped from names (matching how ``estimate_posterior_dag`` cleans
    ``data.columns``), one row per ordered (source, target) pair, evidence
    summed and source counts maxed across duplicate relations, and exactly the
    columns in :data:`PRIOR_COLUMNS`.
    """
    if relations.empty:
        return pd.DataFrame(columns=PRIOR_COLUMNS)

    prior_network = relations.loc[:, PRIOR_COLUMNS].copy()
    prior_network["source"] = prior_network["source"].astype(str).str.replace("-", "")
    prior_network["target"] = prior_network["target"].astype(str).str.replace("-", "")

    # Sum evidence across duplicate edges, but take the max source count: source
    # counts are the number of *distinct* databases/readers behind an edge, so
    # summing them across duplicate rows would double-count the same sources.
    prior_network = prior_network.groupby(["source", "target"], as_index=False).agg(
        evidence_count=("evidence_count", "sum"),
        source_count=("source_count", "max"),
    )

    if verbose:
        node_count = len(pd.unique(prior_network[["source", "target"]].values.ravel()))
        print(f"Number of proteins pulled: {node_count}")
        print(f"Number of reconciled {label} pulled: {len(prior_network)}")

    return prior_network


def _extract_prior_nx(
    source: list,
    target: list,
    measured_proteins: list,
    graph: nx.DiGraph,
    n_mediators: int,
    node_types: Optional[List[str]],
    stmt_types: Optional[List[str]],
    med_ev_filter: Optional[List[int]],
    med_src_filter: Optional[List[int]],
    verbose: bool,
) -> pd.DataFrame:
    """Extract a prior network from a locally loaded INDRA networkx graph."""
    prepared = prepare_graph(
        graph,
        measured_nodes=measured_proteins,
        node_types=node_types,
        stmt_types=stmt_types,
    )
    relations = query_forward_paths(
        prepared,
        start_nodes=source,
        end_nodes=target,
        n_mediators=n_mediators,
        med_ev_filter=med_ev_filter,
        med_src_filter=med_src_filter,
        verbose=verbose,
    )
    return _normalize_prior_network(relations, verbose=verbose, label="edges")


def _extract_prior_neo4j(
    source: list,
    target: list,
    measured_proteins: list,
    client: Neo4jClient,
    one_step_evidence: int,
    two_step_evidence: int,
    three_step_evidence: int,
    verbose: bool,
) -> pd.DataFrame:
    """Extract a prior network from a live INDRA-CoGEx Neo4j instance."""
    source_curies = resolve_curies(source, "gene")
    target_curies = resolve_curies(target, "gene")
    mediator_curies = resolve_curies(measured_proteins, "gene")

    # Path length is capped at three steps, with a rising evidence threshold per
    # step: longer chains have more ways to be spurious, so they must clear a
    # higher bar to earn a place in the prior.
    one_step_relations = format_query_results(
        get_one_step_root_down(
            root_nodes=source_curies,
            downstream_nodes=target_curies,
            client=client,
            relation=CAUSAL_STMT_TYPES,
            minimum_evidence_count=one_step_evidence,
        )
    )
    two_step_relations = format_query_results(
        get_two_step_root_known_med(
            root_nodes=source_curies,
            downstream_nodes=target_curies,
            client=client,
            relation=CAUSAL_STMT_TYPES,
            minimum_evidence_count=two_step_evidence,
            mediators=mediator_curies,
        )
    )
    three_step_relations = format_query_results(
        get_three_step_root(
            root_nodes=source_curies,
            downstream_nodes=target_curies,
            client=client,
            relation=CAUSAL_STMT_TYPES,
            minimum_evidence_count=three_step_evidence,
            mediators=mediator_curies,
        )
    )

    all_relations = pd.concat(
        [one_step_relations, two_step_relations, three_step_relations], ignore_index=True
    )
    if all_relations.empty:
        return pd.DataFrame(columns=PRIOR_COLUMNS)

    # format_query_results returns `source_counts` as a per-statement dict of
    # {database: count}; the prior contract wants the number of distinct sources,
    # matching what the nx backend's add_evidence_info computes.
    all_relations["source_count"] = [
        len(sc) if isinstance(sc, dict) else 0 for sc in all_relations["source_counts"]
    ]

    return _normalize_prior_network(all_relations, verbose=verbose, label="edges")


def extract_indra_prior(
    source: list,
    target: list,
    measured_proteins: list,
    *,
    backend: str = "nx",
    graph: Optional[nx.DiGraph] = None,
    client: Optional[Neo4jClient] = None,
    n_mediators: int = 2,
    node_types: Optional[List[str]] = None,
    stmt_types: Optional[List[str]] = None,
    med_ev_filter: Optional[List[int]] = None,
    med_src_filter: Optional[List[int]] = None,
    one_step_evidence: int = 1,
    two_step_evidence: int = 1,
    three_step_evidence: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract a prior causal network from INDRA.

    Searches INDRA for causal paths running from ``source`` to ``target``
    through ``measured_proteins``, and returns the edges on those paths as a
    candidate edge set for :func:`estimate_posterior_dag`.

    Two backends produce the same output contract, so switching between them
    changes only where the knowledge comes from:

    - ``backend="nx"`` (default) reads a local INDRA ``networkx`` graph, usually
      unpickled from an INDRA dump. No credentials or network access, and the
      whole search runs in-process.
    - ``backend="neo4j"`` queries a live INDRA-CoGEx instance. Sees the current
      database rather than a snapshot, but needs the optional ``indra-cogex``
      dependency and an authenticated client.

    Parameters
    ----------
    source : list of str
        Upstream gene symbols -- the regulators or treatment conditions
        (e.g. ``['EGFR', 'IGF1']``).
    target : list of str
        Downstream gene symbols -- the outcomes of interest
        (e.g. ``['MEK', 'ERK']``).
    measured_proteins : list of str
        Every protein measured in the dataset. Only these are eligible to serve
        as mediators, keeping the prior restricted to variables you can actually
        condition on.
    backend : {"nx", "neo4j"}, default="nx"
        Which INDRA source to query.
    graph : nx.DiGraph, optional
        INDRA graph to search. Required when ``backend="nx"``, ignored otherwise.
    client : Neo4jClient, optional
        Authenticated INDRA-CoGEx client. Required when ``backend="neo4j"``,
        ignored otherwise.

    Other Parameters
    ----------------
    n_mediators : int, default=2
        (``nx`` only) Maximum intermediate nodes on a source -> target path;
        path length is ``n_mediators + 1`` edges.
    node_types : list of str, optional
        (``nx`` only) Allowed node namespaces, e.g. ``["HGNC"]``. Default keeps all.
    stmt_types : list of str, optional
        (``nx`` only) Allowed INDRA statement types. Defaults to
        :data:`CAUSAL_STMT_TYPES`; include the phospho-specific types when the
        readout is phosphorylation data rather than total abundance.
    med_ev_filter, med_src_filter : list of int, optional
        (``nx`` only) Per-depth evidence and source-count thresholds, each of
        length ``n_mediators + 1``. Default is all ones.
    one_step_evidence, two_step_evidence, three_step_evidence : int
        (``neo4j`` only) Minimum evidence count for direct, 2-step, and 3-step
        relationships. Defaults 1, 1, and 3 -- the threshold rises with path
        length because longer chains have more ways to be spurious.
    verbose : bool, default=True
        Print a summary of what was extracted.

    Returns
    -------
    pd.DataFrame
        Columns ``source``, ``target``, ``evidence_count``, ``source_count``.
        One row per ordered edge, evidence summed across duplicate relations,
        hyphens stripped from names to match ``estimate_posterior_dag``'s
        column cleaning. Empty (but correctly shaped) when nothing is found.

    Raises
    ------
    ValueError
        If ``backend`` is unrecognized, or the argument that backend requires
        (``graph`` or ``client``) is missing.

    Examples
    --------
    Local graph, the default:

    >>> import pickle
    >>> with open("indra_network.pkl", "rb") as fh:
    ...     indra_graph = pickle.load(fh)
    >>> priors = extract_indra_prior(
    ...     source=["EGFR"],
    ...     target=["ERK"],
    ...     measured_proteins=data.columns.tolist(),
    ...     graph=indra_graph,
    ... )

    Live CoGEx instance:

    >>> from indra_cogex.client import Neo4jClient
    >>> client = Neo4jClient(url=api_url, auth=("neo4j", password))
    >>> priors = extract_indra_prior(
    ...     source=["EGFR"],
    ...     target=["ERK"],
    ...     measured_proteins=data.columns.tolist(),
    ...     backend="neo4j",
    ...     client=client,
    ... )
    """
    if backend == "nx":
        if graph is None:
            raise ValueError("backend='nx' requires a `graph` argument (an INDRA nx.DiGraph).")
        return _extract_prior_nx(
            source=source,
            target=target,
            measured_proteins=measured_proteins,
            graph=graph,
            n_mediators=n_mediators,
            node_types=node_types,
            stmt_types=CAUSAL_STMT_TYPES if stmt_types is None else stmt_types,
            med_ev_filter=med_ev_filter,
            med_src_filter=med_src_filter,
            verbose=verbose,
        )

    if backend == "neo4j":
        if client is None:
            raise ValueError("backend='neo4j' requires a `client` argument (a Neo4jClient).")
        return _extract_prior_neo4j(
            source=source,
            target=target,
            measured_proteins=measured_proteins,
            client=client,
            one_step_evidence=one_step_evidence,
            two_step_evidence=two_step_evidence,
            three_step_evidence=three_step_evidence,
            verbose=verbose,
        )

    raise ValueError(f"Unknown backend={backend!r}; use 'nx' or 'neo4j'.")


def estimate_posterior_dag(
    data: pd.DataFrame,
    indra_priors: pd.DataFrame,
    prior_strength: float = 5.0,
    scoring_function: type = BICGaussIndraPriors,
    search_algorithm: type = SparseHillClimb,
    n_bootstrap: int = 100,
    add_high_corr_edges_to_priors: bool = False,
    corr_threshold: float = 0.95,
    edge_probability: float = 0.5,
    convert_to_probability: bool = True,
    use_source_counts: bool = False,
    return_bootstrap_dags: bool = False,
    random_init: bool = False,
    selection: str = "best_of",
    dagma_lambda1: float = 0.02,
    dagma_w_threshold: float = 0.2,
    dagma_loss_type: str = "l2",
    dagma_evidence_clip: float = 3.0,
    dagma_evidence_center: bool = False,
    dagma_fit_kwargs: Optional[dict] = None,
    return_runs: bool = False,
    verbose: bool = True,
    interventional: bool = False,
    arm_labels: Optional[pd.Series] = None,
    clamped_nodes: Optional[dict] = None,
    arm_resample_floor: int = 0,
    consensus_subsample_frac: Optional[float] = None,
) -> NxMixedGraph:
    """
    Estimate a posterior directed acyclic graph (DAG) using bootstrap sampling.

    This function combines observational data with prior biological knowledge to learn
    a causal network structure. It uses bootstrap resampling to quantify uncertainty
    in the learned edges and returns only those edges that appear with sufficient
    frequency across bootstrap samples. The function automatically creates expert
    knowledge constraints by forbidding edges not present in the prior network.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data matrix where rows are samples and columns are variables.
        Should contain numeric values for all variables in the network.
        Column names should match protein names in indra_priors.

    indra_priors : pd.DataFrame
        Prior knowledge about causal relationships extracted from INDRA databases.
        Should contain columns: 'source', 'target', 'evidence_count'.
        Typically generated using the extract_indra_prior function.

    prior_strength : float, optional
        Weight given to prior knowledge relative to data. Higher values give more
        importance to the priors, while lower values rely more heavily on the data.
        Default is 5.0. Typical range is 0.1 to 10.0.

    scoring_function : type, optional
        Class implementing the scoring function for evaluating DAG quality.
        Default is BICGaussIndraPriors which incorporates INDRA prior information.
        Other options include standard BIC or BDeu scores.

    search_algorithm : type, optional
        Class implementing the structure learning algorithm for DAG search.
        Default is SparseHillClimb which is optimized for sparse biological networks.
        Other options include standard hill climbing or genetic algorithms.

    n_bootstrap : int, optional
        Number of bootstrap samples to generate. Higher values provide more
        stable estimates but increase computational cost. Default is 100.
        Typical range: 50-1000.

    edge_probability : float, optional
        Minimum probability threshold for including edges in the final network.
        Edges appearing in fewer than this fraction of bootstrap samples are
        excluded. Default is 0.5 (50% threshold).

    convert_to_probability : bool, optional
        Whether to convert edge counts to probabilities before thresholding. Default is True.

    use_source_counts : bool, optional
        If True, use 'source_count' column instead of 'evidence_count' when weighting
        prior edges. Default is False (uses evidence counts).

    return_bootstrap_dags : bool, optional
        If True, return a tuple of (y0_graph, bootstrap_dags) instead of just the
        y0 graph. Default is False.

    random_init : bool, optional
        If True, initialize each bootstrap hill climb from a random acyclic subgraph
        rather than an empty DAG. This can help escape local optima at the cost of
        increased run-to-run variability. Default is False.

    selection : str, optional
        How to reduce candidate DAGs to a single posterior DAG. One of:
        - "best_of": n_bootstrap random-restart hill climbs on the full data;
          keep the highest-scoring acyclic DAG (uses scoring_function/search_algorithm).
        - "consensus": bootstrap resamples + >=edge_probability edge vote
          (uses scoring_function/search_algorithm).
        - "dagma": a single DAGMA continuous-optimization fit on the full data,
          hard-restricted to edges present in indra_priors (see dagma_lambda1,
          dagma_w_threshold, dagma_loss_type). scoring_function, search_algorithm,
          n_bootstrap, edge_probability, and random_init are ignored in this mode;
          edge_prob is set to 1.0 for every returned edge.
        - "dagma_weighted": same as "dagma", but additionally scales the L1
          penalty per allowed edge by its INDRA evidence log-odds (see
          dagma_evidence_clip, dagma_evidence_center): edges with strong
          evidence face a smaller effective penalty, weak-evidence edges a
          larger one. This is the DAGMA analogue of how BICGaussIndraPriors
          combines a hard allowed-edge restriction with a soft log-odds bonus
          for SparseHillClimb.
        Default is "best_of".

    dagma_lambda1 : float, optional
        L1 sparsity penalty for DAGMA's structural loss. Only used when
        selection is "dagma" or "dagma_weighted". Default is 0.02.

    dagma_w_threshold : float, optional
        Post-hoc weight threshold for DAGMA's estimated adjacency matrix; entries
        with magnitude below this are zeroed out. Only used when selection is
        "dagma" or "dagma_weighted". Default is 0.2.

    dagma_loss_type : str, optional
        Loss type passed to DAGMA's DagmaLinear estimator. Only used when
        selection is "dagma" or "dagma_weighted". Default is "l2".

    dagma_evidence_clip : float, optional
        Bounds the per-edge evidence log-odds before exponentiating into a
        penalty multiplier, so a single very strong/weak prior can't dominate.
        Only used when selection="dagma_weighted". Default is 3.0.

    dagma_evidence_center : bool, optional
        If True, center the per-edge log-odds by their mean (over prior-covered
        edges) before computing the penalty multiplier, so the penalty is
        relative to the average prior strength in this graph rather than to
        p=0.5. Only used when selection="dagma_weighted". Default is False.

    dagma_fit_kwargs : dict, optional
        Extra keyword arguments forwarded to DAGMA's DagmaLinear.fit, most
        usefully its convergence schedule (T, warm_iter, max_iter, mu_init,
        mu_factor, s, lr, ...). DAGMA's defaults (T=5, warm_iter=3e4,
        max_iter=6e4) assume a cheap per-iteration cost, but each iteration
        does a dense (d, d) matrix inversion, so wall-clock time scales with
        the number of data columns regardless of how few edges the INDRA
        prior allows. For graphs of a few hundred nodes or more, a lighter
        schedule (e.g. {"T": 3, "warm_iter": 3000, "max_iter": 6000}) is
        often necessary. Only used when selection is "dagma" or
        "dagma_weighted". Default is None (DagmaLinear.fit's own defaults).

    interventional : bool, optional
        If True (and `arm_labels` is given), every scorer constructed inside this
        call - both the per-restart/per-resample scorers in `run_bootstrap` and,
        for `selection="best_of"`, the final re-scoring in `best_scoring_dag` -
        uses `scoring_function`'s pooled GIES-style interventional local score
        instead of a flat GLM fit over all of `data`: each variable is scored by
        one GLM over the rows whose arm does not clamp it. Default False, and the
        fallback whenever `arm_labels` is None, reproduces this function's prior
        behavior exactly - `scoring_function` is never even passed these kwargs
        in that case. Only meaningful with a `scoring_function` that supports
        `interventional`/`arm_labels`/`clamped_nodes` (currently
        `BICGaussIndraPriors` and `BICGaussNoPriors`).

    arm_labels : Optional[pd.Series], optional
        Per-sample experimental-arm label, one entry per row of `data`, sharing
        `data`'s index. Required for `interventional` to take effect.

    clamped_nodes : Optional[dict], optional
        Maps an arm label (as found in `arm_labels`) to the list of node names
        pharmacologically clamped in that arm. Forwarded unchanged wherever
        `interventional` is active - see `BICGaussIndraPriors`.

    arm_resample_floor : int, optional
        Only meaningful when `arm_labels` is not None and `selection="consensus"`
        (i.e. `subsample_frac<1`) - `selection="best_of"` never resamples at all, so
        this has no effect there. Arms with fewer than this many rows are kept in full
        on every bootstrap draw rather than being bootstrap-resampled like the rest of
        the data, since a small arm resampled at a typical frac<1 collapses to too few
        unique rows to reliably fit a multi-parent model - see
        `posterior_estimation.bootstrap._resample_with_arm_floor`. Default is 0, which
        disables this and reproduces the original pooled-resample behavior exactly.

    consensus_subsample_frac : Optional[float], optional
        Only meaningful for `selection="consensus"` - overrides its hardcoded
        `subsample_frac=0.65`. Added for contexts where the full dataset is already so
        small (e.g. n=5-6) that a further 65% subsample collapses to too few rows to
        fit almost any candidate parent set (near-universal degenerate/-inf scores,
        see BT20's HPN-DREAM contexts). `consensus_subsample_frac=1.0` recovers the
        standard bootstrap (resample with replacement at the original size, not a
        smaller subsample) while still producing genuine resampling variability for the
        edge vote. Default `None` leaves the original 0.65 behavior exactly unchanged -
        this parameter has zero effect unless explicitly set.

    Returns
    -------
    NxMixedGraph or tuple[NxMixedGraph, list]
        y0 graph object representing the posterior DAG edges. If return_bootstrap_dags
        is True, returns a tuple of (y0_graph, bootstrap_dags) where bootstrap_dags
        is the list of nx.DiGraph objects from each bootstrap run.

    Examples
    --------
    >>> import pandas as pd
    >>> from indra_cogex.client import Neo4jClient
    >>>
    >>> # Load your data
    >>> data = pd.read_csv('expression_data.csv')
    >>>
    >>> # Extract priors from INDRA
    >>> client = Neo4jClient(url=api_url, auth=("neo4j", password))
    >>> priors = extract_indra_prior(
    ...     source=['EGFR'], target=['ERK'],
    ...     measured_proteins=data.columns.tolist(), client=client
    ... )
    >>>
    >>> # Estimate network
    >>> posterior_dag = estimate_posterior_dag(
    ...     data=data,
    ...     indra_priors=priors,
    ...     prior_strength=5.0,
    ...     n_bootstrap=100,
    ...     edge_probability=0.8
    ... )

    Notes
    -----
    - The function automatically creates expert knowledge constraints by forbidding
      all edges not present in the INDRA prior network
    - Protein names are cleaned by removing hyphens for consistency
    - Higher edge_probability thresholds result in sparser but more confident networks
    - Computational complexity scales with n_bootstrap and the size of the search space
    - Failed bootstrap runs (returning None) are excluded from probability calculations
    """

    indra_priors = indra_priors.reset_index(drop=True)

    # Extract unique nodes from prior network and clean names
    nodes = pd.unique(indra_priors[["source", "target"]].values.ravel())
    nodes = np.array([node.replace("-", "") for node in nodes])

    # Generate all possible edges between nodes
    all_possible_edges = [
        (u.replace("-", ""), v.replace("-", "")) for u in nodes for v in nodes if u != v
    ]

    # Extract observed edges from prior network
    obs_edges = {
        (
            indra_priors.loc[i, "source"].replace("-", ""),
            indra_priors.loc[i, "target"].replace("-", ""),
        )
        for i in range(len(indra_priors))
    }

    # Define forbidden edges as all edges not in the prior network
    forbidden_edges = [edge for edge in all_possible_edges if edge not in obs_edges]

    # Create expert knowledge object with forbidden edges constraint
    expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)

    # Remove hyphens from data column names
    data.columns = [str(col).replace("-", "") for col in data.columns]

    # Verify that every node from the priors appears in the data columns
    missing_nodes = [str(n) for n in nodes if str(n) not in data.columns]
    if missing_nodes:
        raise ValueError(
            "The following nodes from indra_priors are not present in data.columns: "
            + ", ".join(sorted(missing_nodes))
        )

    # ------------------------------------------------------------------
    # Run the search many times, then reduce to a single posterior DAG.
    #   selection="best_of"   -> n_bootstrap random-restart hill climbs on the
    #                            FULL data; keep the highest-scoring acyclic DAG
    #                            (use a BIC scoring_function to control false edges).
    #   selection="consensus" -> bootstrap resamples + >=edge_probability edge vote
    #                            (the original behaviour).
    #   selection="dagma"     -> a single DAGMA continuous-optimization fit on the
    #                            full data, hard-restricted to indra_priors edges.
    #   selection="dagma_weighted" -> same as "dagma", plus a per-edge L1 penalty
    #                            scaled by INDRA evidence log-odds.
    # ------------------------------------------------------------------
    run_scores = None
    if selection in ("dagma", "dagma_weighted"):
        dagma_dag = run_dagma(
            data,
            indra_priors,
            lambda1=dagma_lambda1,
            w_threshold=dagma_w_threshold,
            loss_type=dagma_loss_type,
            use_evidence_weights=(selection == "dagma_weighted"),
            convert_to_probability=convert_to_probability,
            use_source_counts=use_source_counts,
            evidence_clip=dagma_evidence_clip,
            evidence_center=dagma_evidence_center,
            dagma_fit_kwargs=dagma_fit_kwargs,
            verbose=verbose,
        )
        bootstrap_dags = [dagma_dag]
        posterior_dag = pd.DataFrame(list(dagma_dag.edges()), columns=["source", "target"])
    else:
        if selection == "best_of":
            run_frac, run_replace, run_random_init = 1.0, False, True
        elif selection == "consensus":
            run_frac, run_replace, run_random_init = 0.65, True, random_init
            if consensus_subsample_frac is not None:
                run_frac = consensus_subsample_frac
        else:
            raise ValueError(
                f"Unknown selection={selection!r}; use 'best_of', 'consensus', 'dagma', "
                "or 'dagma_weighted'."
            )

        # Run the search to generate multiple DAG hypotheses
        bootstrap_dags = run_bootstrap(
            data,
            indra_priors,
            prior_strength,
            scoring_function,
            search_algorithm,
            expert_knowledge,
            add_high_corr_edges_to_priors,
            corr_threshold,
            n_bootstrap,
            convert_to_probability,
            use_source_counts,
            run_random_init,
            subsample_frac=run_frac,
            replace=run_replace,
            verbose=verbose,
            interventional=interventional,
            arm_labels=arm_labels,
            clamped_nodes=clamped_nodes,
            arm_resample_floor=arm_resample_floor,
        )

        # Reduce the runs to one posterior DAG
        if selection == "best_of":
            edge_priors = prepare_indra_priors(
                indra_priors, convert_to_probability, use_source_counts
            )
            best_dag, run_scores = best_scoring_dag(
                bootstrap_dags,
                data,
                edge_priors,
                scoring_function,
                prior_strength,
                interventional=interventional,
                arm_labels=arm_labels,
                clamped_nodes=clamped_nodes,
            )
            posterior_dag = pd.DataFrame(list(best_dag.edges()), columns=["source", "target"])
        else:
            cons = consensus_dag(bootstrap_dags, indra_priors, lam=0.25, min_freq=edge_probability)
            posterior_dag = pd.DataFrame(list(cons.edges()), columns=["source", "target"])

    # Convert posterior DAG to y0 graph format
    y0_graph = convert_to_y0_graph(posterior_dag)

    # Per-edge frequency across runs, stored as edge_prob (bootstrap frequency for
    # consensus; restart-stability for best_of).
    valid_dags = [d for d in bootstrap_dags if d is not None]
    total = len(valid_dags)
    edge_counts: Counter = Counter()
    for dag in valid_dags:
        edge_counts.update(list(dag.edges()))

    for u, v in y0_graph.directed.edges():
        y0_graph.directed[u][v]["edge_prob"] = (
            edge_counts[(str(u), str(v))] / total if total > 0 else 0.5
        )

    if return_runs:
        return y0_graph, bootstrap_dags, run_scores
    if return_bootstrap_dags:
        return y0_graph, bootstrap_dags
    return y0_graph


def repair_confounding(
    data: pd.DataFrame,
    posterior_dag: NxMixedGraph,
    indra_graph: nx.DiGraph,
    max_conditional: int = 2,
    n_jobs: int = -2,
    confounder_evidence: int = 1,
    verbose: bool = True,
) -> NxMixedGraph:
    """Detect and repair confounding in an estimated posterior DAG.

    Every DAG implies conditional independences; the ones the data rejects mark
    places the structure is wrong, most often an unmeasured common cause. For
    each rejected test this looks in ``indra_graph`` for measured variables that
    are shared upstream regulators of the two nodes involved, and checks whether
    conditioning on any of them restores independence.

    Outcomes are recorded differently depending on what is found:

    - **Resolved** -- a candidate set restores independence, so its variables are
      added to the graph with directed edges to both nodes. Edges that would
      create a cycle are skipped.
    - **Unresolved** -- nothing restores independence, so a bidirected edge is
      added, recording latent confounding explicitly rather than leaving a DAG
      the data has already contradicted.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data over the graph's nodes. Missing values are
        KNN-imputed before testing.
    posterior_dag : NxMixedGraph
        Estimated graph to check. Not modified; a repaired copy is returned.
    indra_graph : nx.DiGraph
        INDRA prior network, annotated with edge evidence, searched for
        confounder candidates.
    max_conditional : int, default=2
        Maximum size of both the tested conditioning sets and the candidate
        confounder combinations.
    n_jobs : int, default=-2
        Parallel workers for testing failures; -2 means all cores but one.
    confounder_evidence : int, default=1
        Minimum evidence count for a candidate confounder relationship.
    verbose : bool, default=True
        Print a summary of what was repaired.

    Returns
    -------
    NxMixedGraph
        A repaired copy of ``posterior_dag``, with directed edges added for
        resolved confounders and bidirected edges for unresolved ones.
    """
    repaired_dag = copy.deepcopy(posterior_dag)

    # Identify relations whose implied independence the data rejects.
    failed_tests, imputed_data = find_failed_tests(
        repaired_dag, data, max_conditional=max_conditional
    )

    # Ask INDRA which measured variables could explain each failure.
    confounder_relations = lookup_confounder_candidates(indra_graph, failed_tests, verbose=verbose)

    n = len(failed_tests)
    if verbose:
        print(f"Processing {n} failed tests for confounding repair...")

    # Pre-convert rows to dicts to avoid serializing the full DataFrame per worker
    failed_test_rows = [failed_tests.loc[i].to_dict() for i in range(n)]

    results = list(
        tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(process_failed_test)(
                    row, confounder_relations, imputed_data, max_conditional
                )
                for row in failed_test_rows
            ),
            total=n,
            desc="Processing failed tests",
        )
    )

    # Process results and collect statistics
    total_results = len(results)
    none_results = sum(1 for res in results if not res)
    valid_results = total_results - none_results
    repaired_count = 0
    unrepaired_count = 0
    new_nodes_added = set()
    new_edges_added = 0

    for res in results:
        if not res:
            continue
        src = Variable(res.get("source"))
        tgt = Variable(res.get("target"))
        Z = res.get("Z")
        if res.get("add_latent") or Z is None:
            repaired_dag.add_undirected_edge(src, tgt)
            unrepaired_count += 1
        else:
            repaired_count += 1
            # add nodes and directed edges from Z -> source and Z -> target
            for node in Z:
                node = Variable(node)
                if node not in repaired_dag.directed.nodes:
                    repaired_dag.add_node(node)
                    new_nodes_added.add(str(node))
                if ((node, src) not in repaired_dag.directed.edges) and (
                    not nx.has_path(repaired_dag.directed, src, node)
                ):
                    repaired_dag.add_directed_edge(node, src, directed=True)
                    new_edges_added += 1
                if ((node, tgt) not in repaired_dag.directed.edges) and (
                    not nx.has_path(repaired_dag.directed, tgt, node)
                ):
                    repaired_dag.add_directed_edge(node, tgt, directed=True)
                    new_edges_added += 1

    # Print summary of confounding repair results
    if verbose:
        repair_rate = (repaired_count / valid_results * 100) if valid_results > 0 else 0
        print("\n" + "=" * 60)
        print("CONFOUNDING REPAIR SUMMARY")
        print("=" * 60)
        print(f"Total failed tests processed: {total_results}")
        print(f"Valid results obtained: {valid_results}")
        print(f"Failed/invalid results: {none_results}")
        print(f"Successfully repaired confounders: {repaired_count}")
        print(f"Unrepaired confounders: {unrepaired_count}")
        if new_nodes_added:
            print(f"New confounder nodes added: {len(new_nodes_added)}")
            print(f"Added nodes: {', '.join(sorted(new_nodes_added))}")
        else:
            print("No new confounder nodes were added")
        if new_edges_added > 0:
            print(f"New edges added to repair confounding: {new_edges_added}")
        else:
            print("No new edges were added during confounding repair")
        print(f"Repair success rate: {repair_rate:.1f}%")
        print("=" * 60 + "\n")

    return repaired_dag
