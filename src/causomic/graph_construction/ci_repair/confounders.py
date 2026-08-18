"""Explain failed independence tests with confounders drawn from INDRA.

When a conditional-independence test fails, an unmeasured common cause is the
usual suspect. Rather than immediately declaring a latent variable, this module
first asks whether a *measured* variable can account for the dependence:

1. :func:`lookup_confounder_candidates` asks the INDRA prior graph which nodes
   are shared upstream regulators of the two variables in question.
2. :func:`process_failed_test` conditions on combinations of those candidates
   and re-tests, keeping the first set that restores independence.

A candidate set that works becomes real directed edges in the repaired graph. If
nothing works, the dependence is attributed to a genuine latent confounder and a
bidirected edge is added instead -- an honest record of unresolved confounding
rather than a silently mis-specified DAG.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from pgmpy.estimators.CITests import pearsonr
from tqdm import tqdm

from causomic.graph_construction.prior_extraction.nx_backend import query_confounders


def lookup_confounder_candidates(indra_graph, failed_tests: pd.DataFrame, verbose: bool = True):
    """Collect candidate confounders from INDRA for each failed test pair.

    Parameters
    ----------
    indra_graph : nx.DiGraph
        INDRA prior network annotated with edge evidence.
    failed_tests : pd.DataFrame
        Output of
        :func:`~causomic.graph_construction.ci_repair.falsification.find_failed_tests`;
        only its 'left'/'right' columns are read.
    verbose : bool, default=True
        Show a progress bar over the unique node pairs.

    Returns
    -------
    dict
        ``{(left, right): [candidate, ...]}`` with candidates ordered by total
        supporting evidence, strongest first. Pairs with no shared upstream
        regulator in ``indra_graph`` map to an empty array.
    """
    query_relations = failed_tests[["left", "right"]].drop_duplicates().reset_index(drop=True)

    confounder_relations = {}
    iterator = range(len(query_relations))
    if verbose:
        iterator = tqdm(iterator, desc="Pulling confounder relations")

    for i in iterator:
        nodes = [query_relations.loc[i, "left"], query_relations.loc[i, "right"]]
        indra_relations = query_confounders(indra_graph, nodes)
        indra_relations = (
            indra_relations.groupby(["source"], as_index=False)["evidence_count"]
            .sum()
            .sort_values(by="evidence_count", ascending=False)["source"]
            .values
        )
        confounder_relations[tuple(nodes)] = indra_relations

    return confounder_relations


def process_failed_test(
    row: pd.Series, confounder_relations: dict, data: pd.DataFrame, max_conditional: int = 2
):
    """
    Process a single failed conditional independence test to identify potential confounders.

    This function attempts to repair confounding relationships by testing whether
    adding observed confounding variables can restore conditional independence
    between two variables. If successful, it returns the confounding variables
    that should be added to the causal graph. If unsuccessful, it indicates
    that a latent (unobserved) confounder should be considered.

    The function systematically tests combinations of potential confounders up to
    a maximum size, looking for a set that renders the source and target variables
    conditionally independent given the existing conditioning set plus the new
    confounders.

    Parameters
    ----------
    row : pd.Series or dict
        Row from failed conditional independence tests containing:
        - 'left': Source variable name (str)
        - 'right': Target variable name (str)
        - 'given': Existing conditioning variables (str, list, or empty)

    confounder_relations : dict
        Dictionary mapping (source, target) tuples to lists of potential
        confounder variable names extracted from biological databases.
        Format: {(source, target): [confounder1, confounder2, ...]}

    data : pd.DataFrame
        Observational data matrix where rows are samples and columns are variables.
        Must contain all variables referenced in row and confounder_relations.

    max_conditional : int
        Maximum number of confounding variables to test in combination.
        Higher values allow more complex confounding patterns but increase
        computational cost. Typical range: 1-3.

    Returns
    -------
    dict
        Dictionary containing repair results with keys:
        - 'source': Source variable name (str)
        - 'target': Target variable name (str)
        - 'add_latent': Whether to add latent confounder (bool)
        - 'Z': Confounding variables that restore independence (tuple or None)
        - 'error': Error message if exception occurred (str, optional)

        If add_latent=False and Z is not None, the variables in Z should be
        added as confounders in the causal graph with edges to both source
        and target. If add_latent=True, a bidirectional edge or latent
        confounder should be considered.

    Examples
    --------
    >>> # Example failed test row
    >>> failed_test = {'left': 'EGFR', 'right': 'ERK', 'given': 'MEK'}
    >>>
    >>> # Potential confounders from INDRA
    >>> confounders = {('EGFR', 'ERK'): ['AKT', 'PI3K', 'RAS']}
    >>>
    >>> # Process the failed test
    >>> result = process_failed_test(
    ...     failed_test, confounders, data, max_conditional=2
    ... )
    >>>
    >>> if not result['add_latent']:
    ...     print(f"Add confounders: {result['Z']}")
    ... else:
    ...     print("Add latent confounder")

    """

    try:
        # Create or reuse a client in this process
        source = row["left"]
        target = row["right"]
        given = row["given"]

        add_latent = False
        found_adjustment = False
        found_Z = None

        # build all non-empty confounder combos (kept same range as original: r in [1])
        confounders = confounder_relations[(source, target)]
        confounders = [i for i in confounders if i != given and i in data.columns]

        # sort by combined absolute correlation with source and target so the
        # most promising candidates are tested first, improving early termination
        if confounders:
            corr_scores = (
                data[confounders].corrwith(data[source]).abs()
                + data[confounders].corrwith(data[target]).abs()
            )
            confounders = corr_scores.sort_values(ascending=False).index.tolist()

        conf_list = list(confounders)
        all_combos = [
            combo for r in range(1, max_conditional + 1) for combo in combinations(conf_list, r)
        ]

        # normalize 'given' once
        if isinstance(given, (list, tuple, np.ndarray)):
            given_list = list(given)
        elif given is None or (isinstance(given, str) and given == "") or pd.isna(given):
            given_list = []
        else:
            given_list = [given]

        # no confounders → plan to add latent
        if not all_combos:
            return {"source": source, "target": target, "add_latent": True, "Z": None}

        # test combos; stop at first success
        for combo in all_combos:
            Z = given_list + list(combo)
            try:
                independent = pearsonr(source, target, Z, data, significance_level=0.05)
            except Exception:
                independent = False
            if independent:
                found_adjustment = True
                found_Z = combo
                break

        if found_adjustment:
            return {"source": source, "target": target, "add_latent": False, "Z": found_Z}
        else:
            return {"source": source, "target": target, "add_latent": True, "Z": None}

    except Exception as e:
        # On error be conservative: mark as latent confounding
        return {
            "source": row.get("left"),
            "target": row.get("right"),
            "add_latent": True,
            "Z": None,
            "error": str(e),
        }
