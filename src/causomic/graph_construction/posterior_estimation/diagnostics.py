"""Diagnostics for how much of a learned structure is real signal.

Bootstrap edge frequency is easy to over-read: an edge appearing in 90% of runs
looks confident, but if random initialization alone produces it 90% of the time
the frequency is telling you about the search, not the data.

:func:`search_path_diagnostic` isolates that confound by running K hill climbs
from random starts on the *same, unresampled* data, so the only source of
variation is the search path. :func:`compare_dag_sets` then puts those
frequencies side by side with bootstrap frequencies. Edges with a large
``abs_diff`` are the ones whose apparent stability comes from one source and not
the other, and are worth inspecting before being believed.
"""

from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from causomic.graph_construction.posterior_estimation.hill_climb import (
    random_acyclic_subgraph,
)


def run_single_random_init(
    data,
    edge_priors,
    score_fn,
    estimator,
    expert_knowledge,
    allowed_edges,
    nodes,
    inclusion_prob,
    max_indegree,
    seed,
):
    """Single Hill Climb run from a random initial DAG on the full dataset."""
    import logging

    logging.getLogger("pgmpy").setLevel(logging.WARNING)

    rng = np.random.default_rng(seed)
    # Left at random_acyclic_subgraph's default max_indegree on purpose: this must
    # match how process_bootstrap builds its random_init start DAGs, or
    # compare_dag_sets would be contrasting two differently-seeded searches rather
    # than isolating search-path dependence.
    start_dag = random_acyclic_subgraph(nodes, allowed_edges, inclusion_prob, rng)

    scorer = score_fn(data, edge_priors=edge_priors)
    est = estimator(data=data, allowed_additions=set(allowed_edges))

    estimated_dag = est.estimate(
        scoring_method=scorer,
        start_dag=start_dag,
        expert_knowledge=expert_knowledge,
        max_indegree=max_indegree,
        epsilon=0.01,
        show_progress=False,
    )
    return estimated_dag


def search_path_diagnostic(
    data,
    edge_priors,
    score_fn,
    estimator,
    expert_knowledge,
    K=50,
    inclusion_prob=0.15,
    max_indegree=5,
):
    """
    Run K Hill Climb searches from random initializations on the SAME full dataset.
    Compare edge sets to diagnose search path dependence vs genuine signal.
    """
    nodes = list(data.columns)
    allowed_edges = list(edge_priors.keys())

    dags = Parallel(n_jobs=-2)(
        delayed(run_single_random_init)(
            data,
            edge_priors,
            score_fn,
            estimator,
            expert_knowledge,
            allowed_edges,
            nodes,
            inclusion_prob,
            max_indegree,
            seed=i,
        )
        for i in tqdm(range(K), desc="Random init runs")
    )

    dags = [d for d in dags if d is not None]
    return dags


def compare_dag_sets(dags_random_init, dags_bootstrap, label_a="random_init", label_b="bootstrap"):
    """
    Compare edge stability between two sets of DAGs.
    Returns a DataFrame with per-edge frequencies from each approach.
    """

    def edge_frequencies(dags):
        counts = Counter()
        for dag in dags:
            counts.update(list(dag.edges()))
        n = len(dags)
        return {edge: count / n for edge, count in counts.items()}

    freq_a = edge_frequencies(dags_random_init)
    freq_b = edge_frequencies(dags_bootstrap)

    all_edges = set(freq_a.keys()) | set(freq_b.keys())
    rows = []
    for edge in sorted(all_edges):
        rows.append(
            {
                "source": edge[0],
                "target": edge[1],
                f"freq_{label_a}": freq_a.get(edge, 0.0),
                f"freq_{label_b}": freq_b.get(edge, 0.0),
            }
        )

    df = pd.DataFrame(rows)
    df["abs_diff"] = abs(df[f"freq_{label_a}"] - df[f"freq_{label_b}"])
    df = df.sort_values("abs_diff", ascending=False)
    return df
