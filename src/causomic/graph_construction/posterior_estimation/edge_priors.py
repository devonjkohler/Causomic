"""Turn INDRA evidence counts into per-edge prior probabilities.

This is the bridge between prior extraction and structure learning. Extraction
yields raw evidence counts; the scoring functions in
:mod:`~causomic.graph_construction.posterior_estimation.scores` and the DAGMA
penalty in :mod:`~causomic.graph_construction.posterior_estimation.dagma` both
want a probability in ``(0, 1)`` per candidate edge.

The conversion fits a discrete power law to the observed counts, which reflects
how INDRA evidence actually distributes: a handful of relationships carry
hundreds of citations while the long tail carries one or two. A linear rescaling
would let those few dominate entirely.
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt


def calculate_edge_probabilities(
    indra_priors: pd.DataFrame, count_col: str = "evidence_count"
) -> dict:
    """
    Calculate edge probabilities from INDRA evidence counts using power law modeling.

    Converts raw INDRA evidence counts to edge probabilities by fitting a discrete
    power law distribution to the evidence count data. This approach recognizes that
    biological evidence follows heavy-tailed distributions where few relationships
    have extensive evidence while most have modest support.

    The power law model P(X = k) ∝ k^(-α) provides a principled way to transform
    evidence counts into probabilities that appropriately weight strong evidence
    while not completely dismissing weaker relationships.

    Parameters
    ----------
    indra_priors : pd.DataFrame
        DataFrame containing INDRA prior information with 'evidence_count' column

    Returns
    -------
    dict
        Mapping from evidence count values to cumulative probabilities [0,1].
        Higher evidence counts map to higher probabilities.

    Examples
    --------
    >>> # Process INDRA evidence counts
    >>> indra_df = pd.DataFrame({
    ...     'source': ['AKT1', 'TP53', 'MDM2'],
    ...     'target': ['MDM2', 'MDM2', 'TP53'],
    ...     'evidence_count': [15, 25, 8]
    ... })
    >>> prob_mapping = calculate_edge_probabilities(indra_df)
    >>> # Returns: {8: 0.2, 15: 0.6, 25: 0.9} (example values)

    Notes
    -----
    Algorithm steps:
    1. Extract evidence counts and find minimum value (xmin)
    2. Fit power law exponent α using maximum likelihood estimation
    3. Compute discrete power law PMF: P(k) = k^(-α) / ζ(α, xmin)
    4. Calculate cumulative distribution function (CDF) values
    5. Return mapping from counts to CDF probabilities

    The power law model is particularly appropriate for biological networks where:
    - Few relationships have extensive experimental validation
    - Many relationships have limited but meaningful evidence
    - Evidence accumulation follows preferential attachment dynamics

    CDF transformation ensures that higher evidence counts receive higher
    probabilities while maintaining proper probability interpretation.
    """

    edge_evidence = indra_priors[count_col].values.astype(int)

    xmin = edge_evidence.min()

    # Discrete Power Law Log-Likelihood
    def powerlaw_log_likelihood(alpha, data, xmin):
        n = len(data)
        log_sum = -alpha * np.sum(np.log(data))
        zeta = np.sum([k ** (-alpha) for k in range(xmin, max(data) + 1)])
        return -(log_sum - n * np.log(zeta))

    # Fit alpha using MLE
    res = opt.minimize_scalar(
        powerlaw_log_likelihood, bounds=(1.01, 10), args=(edge_evidence, xmin), method="bounded"
    )
    alpha_hat = res.x

    # Compute CDF values (discrete power law)
    support = np.arange(xmin, max(edge_evidence) + 1)
    pmf = support ** (-alpha_hat)
    pmf /= pmf.sum()
    cdf_vals = np.cumsum(pmf)

    value_to_cdf = dict(zip(support, cdf_vals))

    return value_to_cdf


def prepare_indra_priors(
    indra_priors: pd.DataFrame, convert_to_probability: bool, use_source_counts: bool = False
) -> dict:
    """
    Prepare INDRA prior data for causal discovery by converting to edge probabilities.

    Transforms INDRA evidence counts into edge probability dictionary suitable for
    constrained causal discovery algorithms. This function combines power law
    modeling of evidence counts with proper edge formatting for downstream analysis.

    The preparation process ensures that biological prior knowledge is properly
    encoded as soft constraints that can guide but not override strong data evidence
    during causal discovery.

    Parameters
    ----------
    indra_priors : pd.DataFrame
        DataFrame with INDRA prior information containing columns:
        - 'source': Source protein/gene symbol
        - 'target': Target protein/gene symbol
        - 'evidence_count': Number of supporting evidence instances
        - 'source_count': Number of distinct sources (used when use_source_counts=True)

    convert_to_probability : bool
        Whether to convert counts to probabilities via power law modeling.

    use_source_counts : bool, optional
        If True, use the 'source_count' column instead of 'evidence_count'.
        Default is False (uses evidence counts).

    Returns
    -------
    dict
        Dictionary mapping (source, target) tuples to edge probabilities [0,1].
        Format: {(source, target): probability}

    Examples
    --------
    >>> # Prepare INDRA priors for causal discovery
    >>> indra_df = pd.DataFrame({
    ...     'source': ['AKT1', 'TP53', 'MDM2'],
    ...     'target': ['MDM2', 'MDM2', 'TP53'],
    ...     'evidence_count': [15, 25, 8]
    ... })
    >>> edge_priors = prepare_indra_priors(indra_df)
    >>> # Returns: {('AKT1', 'MDM2'): 0.6, ('TP53', 'MDM2'): 0.9, ('MDM2', 'TP53'): 0.2}
    >>>
    >>> # Use in constrained causal discovery
    >>> search = SparseHillClimb(data, allowed_additions=list(edge_priors.keys()))
    >>> scorer = BICGaussIndraPriors(data, edge_priors=edge_priors)
    >>> dag = search.estimate(scoring_method=scorer)

    Notes
    -----
    This function serves as the bridge between INDRA biological knowledge and
    causal discovery algorithms by:

    1. Converting evidence counts to probabilities using power law modeling
    2. Formatting edges as (source, target) tuples for algorithm compatibility
    3. Handling missing evidence with default high probability (1.0)
    4. Ensuring consistent edge representation across the pipeline

    The resulting edge probabilities can be used in:
    - Constrained search algorithms (allowed_additions parameter)
    - Scoring functions with biological priors
    - Expert knowledge specification for hard constraints

    Missing evidence counts are filled with probability 1.0 to ensure all
    edges in the prior network are considered, even if evidence is sparse.
    """
    count_col = "source_count" if use_source_counts else "evidence_count"
    if convert_to_probability:
        # edge_probability_mapper = calculate_edge_probabilities(indra_priors, count_col)
        # indra_priors["edge_p"] = indra_priors[count_col].map(edge_probability_mapper).fillna(1.0)
        log_ev = np.log1p(indra_priors[count_col])
        # median_log_ev = np.median(log_ev)
        # Values extracted from all INDRA HGNC edges
        indra_priors["edge_p"] = 1 / (1 + np.exp(-(log_ev - 1.1) / 0.552))

    else:
        indra_priors["edge_p"] = indra_priors[count_col]

    edge_probabilities = {
        (
            indra_priors.loc[i, "source"],
            indra_priors.loc[i, "target"],
        ): indra_priors.loc[i, "edge_p"]
        for i in range(len(indra_priors))
    }

    return edge_probabilities


def remove_high_corr_edges_from_blacklist(
    data: pd.DataFrame,
    indra_priors: pd.DataFrame,
    black_list: set,
    corr_threshold: float = 0.8,
    verbose: bool = True,
) -> set:
    """
    Remove edges between highly correlated variables from the blacklist.

    This function identifies pairs of variables in the dataset that exhibit
    high correlation (above a specified threshold) and removes any edges
    between these variables from the provided blacklist. It then adds the edges
    to the indra_priors DataFrame with a low prior probability (floor of
    observed probabilities). This is useful in causal discovery to avoid
    excluding potentially valid edges that may represent true causal
    relationships rather than mere correlations.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the variables of interest.
    indra_priors : pd.DataFrame
        DataFrame containing INDRA prior information with columns:
        - 'source': Source protein/gene symbols
        - 'target': Target protein/gene symbols
        - 'evidence_count': Evidence count for each relationship
    black_list : set
        A set of (parent, child) tuples representing edges to be blacklisted.
    corr_threshold : float, default=0.9
        The correlation threshold above which edges will be removed from the blacklist.

    Returns
    -------
    set
        Updated blacklist with edges between highly correlated variables removed.

    Examples
    --------
    >>> # Example dataset
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [2, 4, 6, 8, 10],
    ...     'C': [5, 4, 3, 2, 1]
    ... })
    >>>
    >>> # Initial blacklist with edges to be removed if highly correlated
    >>> blacklist = {('A', 'B'), ('B', 'C')}
    >>>
    >>> # Remove edges between highly correlated variables (threshold=0.9)
    >>> updated_blacklist = remove_high_corr_edges_from_blacklist(df, blacklist, corr_threshold=0.9)
    >>> print(updated_blacklist)
    {('B', 'C')}  # Edge ('A', 'B') removed due to high correlation

    Notes
    -----
    - The function computes the absolute correlation matrix of the dataset.
    - It identifies variable pairs with correlation above the specified threshold.
    - Edges between these highly correlated pairs are removed from the blacklist.
    - This helps retain potentially valid causal edges that might otherwise be excluded.
    """

    # Compute absolute correlation matrix
    corr_matrix = data.corr().abs()

    # Find pairs with correlation above threshold (excluding self-pairs)
    high_corr_pairs = set()
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and corr_matrix.loc[i, j] >= corr_threshold:
                high_corr_pairs.add((i, j))
                high_corr_pairs.add((j, i))  # Both directions

    if verbose:
        print(f"High correlation pairs (threshold={corr_threshold}): {len(high_corr_pairs)}")

    # Remove highly correlated edges from blacklist
    updated_blacklist = set(edge for edge in black_list if edge not in high_corr_pairs)

    # Add missing high-corr edges to indra_priors DataFrame
    for src, tgt in high_corr_pairs:
        if not (((indra_priors["source"] == src) & (indra_priors["target"] == tgt)).any()):
            new_row = {"source": src, "target": tgt, "evidence_count": 1}
            indra_priors.loc[len(indra_priors)] = new_row

    return indra_priors, updated_blacklist
