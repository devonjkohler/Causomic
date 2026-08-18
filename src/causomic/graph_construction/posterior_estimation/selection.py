"""Reduce a population of candidate DAGs to a single posterior DAG.

:mod:`~causomic.graph_construction.posterior_estimation.bootstrap` produces many
DAGs; exactly one has to come out the other end. The two strategies here answer
that differently:

:func:`consensus_dag`
    Vote per edge. Keep edges appearing in at least ``min_freq`` of runs, then
    insert them best-first, skipping any that would close a cycle. Suited to
    bootstrap resamples, where run-to-run variation is sampling noise and
    frequency is a confidence estimate.

:func:`best_scoring_dag`
    Keep the single best run. Rescore every candidate on the *full* data and
    take the winner. Suited to random restarts, where variation is search-path
    dependence and averaging would blend incompatible local optima into a graph
    no run actually proposed.
"""

from collections import Counter

import networkx as nx
import numpy as np


def consensus_dag(bootstrap_dags, indra_priors, lam=0.25, min_freq=0.5):
    """Build a consensus DAG by frequency-voting edges across bootstrap runs.

    Edges are ranked by bootstrap frequency plus a prior-belief bonus,

    .. code-block:: text

        weight(e) = freq(e) + lam * log(p(e) / (1 - p(e)))

    and then added highest-weight-first, skipping any edge that would create a
    cycle. Ranking matters because of that skip: when two edges conflict, the
    higher-weighted one is inserted and the other is dropped, so the prior acts
    as a tie-breaker among edges the data supports comparably.

    Parameters
    ----------
    bootstrap_dags : list of DAG or None
        Candidate DAGs; ``None`` entries (failed runs) are ignored, and the
        frequency denominator counts only the runs that succeeded.
    indra_priors : pd.DataFrame
        Prior network with 'source'/'target' columns and an 'edge_p' column of
        edge probabilities. Hyphens are stripped from names to match the rest
        of the pipeline. Edges absent here are treated as p=0.5 (no opinion,
        zero bonus).
    lam : float, default=0.25
        Weight on the prior log-odds term. ``lam=0`` ranks by bootstrap
        frequency alone.
    min_freq : float, default=0.5
        Minimum fraction of runs an edge must appear in to be a candidate.

    Returns
    -------
    nx.DiGraph
        Acyclic consensus graph over all nodes seen in ``bootstrap_dags``.
    """
    # build edge priors dict from indra_priors DataFrame
    df = indra_priors.copy()
    df["source"] = df["source"].astype(str).str.replace("-", "")
    df["target"] = df["target"].astype(str).str.replace("-", "")

    edge_priors = {(row["source"], row["target"]): row["edge_p"] for _, row in df.iterrows()}

    counts = Counter()
    total = 0

    for dag in bootstrap_dags:
        if dag is None:
            continue
        counts.update(list(dag.edges()))
        total += 1

    G = nx.DiGraph()
    for dag in bootstrap_dags:
        if dag:
            G.add_nodes_from(dag.nodes())

    def weight(edge):
        f = counts[edge] / max(total, 1)
        p = np.clip(edge_priors.get(edge, 0.5), 1e-6, 1 - 1e-6)
        return f + lam * np.log(p / (1 - p))

    candidates = [e for e, c in counts.items() if c / max(total, 1) >= min_freq]

    candidates.sort(key=weight, reverse=True)
    for u, v in candidates:
        G.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(u, v)
    return G


def best_scoring_dag(
    dags,
    data,
    edge_priors,
    scoring_function,
    prior_strength,
    interventional: bool = False,
    arm_labels=None,
    clamped_nodes=None,
):
    """Select the single highest-scoring acyclic DAG from candidate runs.

    Each candidate is scored by its total local score
    (``sum_v scoring_function.local_score(v, parents_v)``) on the full ``data``.
    This implements "best-of-restarts" selection: run the hill climb from many
    random initializations and keep the run that reached the best-scoring local
    optimum, rather than voting on individual edges across bootstrap resamples.

    Parameters
    ----------
    dags : list of DAG or None
        Candidate DAGs (e.g. the per-restart outputs of ``run_bootstrap``).
    data : pd.DataFrame
        Full observational data used to score every candidate on equal footing.
    edge_priors : dict
        {(parent, child): prior_probability} for the allowed edges.
    scoring_function : type
        Scoring class (e.g. ``BICGaussIndraPriors``); BIC penalizes complexity
        more heavily than AIC and is the recommended choice here.
    prior_strength : float
        Passed through to the scoring function.
    interventional : bool, optional
        Forwarded to ``scoring_function`` only when True (default False never
        adds these kwargs to the ``scoring_function(...)`` call at all, so
        scoring classes without this parameter are unaffected).
    arm_labels : Optional[pd.Series], optional
        Per-sample experimental-arm label aligned to ``data``'s index. Unlike
        ``run_bootstrap``'s bootstrap resamples, ``data`` here is used as-is
        (never resampled), so ``arm_labels`` is passed through unmodified.
    clamped_nodes : Optional[dict], optional
        Forwarded to ``scoring_function`` unchanged when ``interventional`` is True.

    Returns
    -------
    (best_dag, scores) : tuple[nx.DiGraph, list[Optional[float]]]
        ``best_dag`` is the top-scoring candidate (an empty DiGraph over the data
        columns if none are valid); ``scores`` is the per-candidate total score
        (``None`` for missing or cyclic runs), aligned with ``dags``.
    """
    interventional_kwargs = {}
    if interventional:
        interventional_kwargs = dict(
            interventional=True, arm_labels=arm_labels, clamped_nodes=clamped_nodes
        )
    scorer = scoring_function(
        data, edge_priors=edge_priors, prior_strength=prior_strength, **interventional_kwargs
    )
    best, best_score, scores = None, -np.inf, []
    for dag in dags:
        if dag is None:
            scores.append(None)
            continue
        G = nx.DiGraph()
        G.add_nodes_from(data.columns)
        G.add_edges_from(dag.edges())
        if not nx.is_directed_acyclic_graph(G):
            scores.append(None)
            continue
        s = float(sum(scorer.local_score(v, list(G.predecessors(v))) for v in data.columns))
        scores.append(s)
        if s > best_score:
            best_score, best = s, dag
    if best is None:
        best = nx.DiGraph()
        best.add_nodes_from(data.columns)
    return best, scores
