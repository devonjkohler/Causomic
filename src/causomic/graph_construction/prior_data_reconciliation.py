"""
Prior-Data Reconciliation for Causal Network Discovery

This module implements sophisticated algorithms for reconciling INDRA-derived
biological prior knowledge with experimental proteomics data to construct
refined causal graphs. The implementation leverages constrained Hill Climb
search algorithms with custom scoring functions that balance data fit with
prior biological knowledge.

The core innovation is the SparseHillClimb algorithm, which restricts the
search space to biologically plausible edges defined by INDRA prior knowledge,
significantly improving computational efficiency while maintaining causal
discovery performance.

Key Components
--------------
- SparseHillClimb: Constrained Hill Climb search with predefined edge sets
- Custom Scoring Functions: AIC/BIC variants with soft INDRA priors
- Bootstrap Framework: Robust causal discovery with uncertainty quantification
- Prior Integration: Seamless combination of biological knowledge and data

Typical Workflow
----------------
1. Define allowed edges from INDRA prior knowledge
2. Compute edge probabilities from biological evidence
3. Run constrained Hill Climb search with custom scoring
4. Perform bootstrap analysis for uncertainty quantification
5. Extract consensus causal network structure

Examples
--------
>>> # Basic prior-data reconciliation
>>> edge_priors = {("AKT1", "MDM2"): 0.8, ("TP53", "MDM2"): 0.9}
>>> allowed_edges = list(edge_priors.keys())
>>>
>>> # Initialize constrained search
>>> search = SparseHillClimb(data, allowed_additions=allowed_edges)
>>> scoring = AICGaussIndraPriors(data, edge_priors=edge_priors)
>>>
>>> # Discover causal network
>>> causal_dag = search.estimate(scoring_method=scoring)

Author: Devon Kohler
Date: 2024
"""

# Standard library imports
import logging
from collections import deque
from typing import (
    Any,
    Callable,
    Deque,
    Generator,
    Hashable,
    Iterable,
    Optional,
    Set,
    Tuple,
)
from xml.parsers.expat import model

# Scientific computing imports
import networkx as nx
import numpy as np
import pandas as pd
import scipy.optimize as opt

# Parallel processing and progress tracking
from joblib import Parallel, delayed

# Causal discovery and probabilistic graphical models
from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge, HillClimbSearch, LogLikelihoodGauss
from pgmpy.estimators.StructureScore import get_scoring_method
from tqdm import tqdm
from tqdm.auto import trange

try:
    import scipy.linalg as sla
    from dagma.linear import DagmaLinear
    from scipy.special import expit as sigmoid
except ImportError:  # optional dependency, see causomic._optional
    from causomic._optional import MissingDagmaLinear as DagmaLinear


class SparseHillClimb(HillClimbSearch):
    """
    Constrained Hill Climb search for causal discovery with prior knowledge.

    This class extends pgmpy's HillClimbSearch to support restricting edge
    additions to a predefined set of biologically plausible relationships.
    Unlike the standard implementation that considers all possible edges,
    this sparse variant dramatically reduces search space complexity while
    incorporating prior biological knowledge.

    The key innovation is constraining the edge addition operations to only
    those relationships supported by prior evidence (e.g., from INDRA database),
    which both speeds up discovery and improves biological plausibility of
    the resulting causal networks.

    Parameters
    ----------
    data : pd.DataFrame
        Observational dataset with samples as rows and variables as columns
    allowed_additions : Optional[Iterable[Tuple[str, str]]], default=None
        Set of (parent, child) pairs representing biologically plausible edges.
        If None, falls back to standard HillClimbSearch behavior
    use_cache : bool, default=True
        Whether to cache scoring computations for efficiency
    **kwargs
        Additional arguments passed to parent HillClimbSearch class

    Attributes
    ----------
    allowed_additions : Optional[Set[Tuple[str, str]]]
        Set of allowed edge additions for constrained search

    Examples
    --------
    >>> # Define biologically plausible edges from prior knowledge
    >>> allowed_edges = [("AKT1", "MDM2"), ("TP53", "MDM2"), ("MDM2", "TP53")]
    >>>
    >>> # Initialize constrained search
    >>> search = SparseHillClimb(data, allowed_additions=allowed_edges)
    >>>
    >>> # Run causal discovery with biological constraints
    >>> causal_dag = search.estimate(scoring_method="bic")

    Notes
    -----
    This implementation is particularly valuable for biological applications where:
    - Prior knowledge about regulatory relationships exists
    - Computational efficiency is important for large networks
    - Biological plausibility of discovered edges is crucial

    The sparse constraint can reduce search space from O(n²) to O(k) where
    k is the number of allowed edges, providing substantial speedup for
    large biological networks.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        allowed_additions: Optional[Iterable[Tuple[str, str]]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(data, use_cache=use_cache, **kwargs)
        self.allowed_additions = set(allowed_additions) if allowed_additions else None

    def estimate(
        self,
        scoring_method=None,
        start_dag: Optional[DAG] = None,
        tabu_length: int = 100,
        max_indegree: Optional[int] = None,
        expert_knowledge: Optional[ExpertKnowledge] = None,
        epsilon: float = 1e-4,
        max_iter: int = int(1e6),
        show_progress: bool = True,
        on_step: Optional[Callable[[int, Tuple[str, Tuple[str, str]], float], None]] = None,
    ) -> DAG:
        """
        Estimate causal DAG using constrained Hill Climb search.

        Performs iterative local search through DAG space, constrained by
        allowed edge additions from prior knowledge. Each iteration evaluates
        add, remove, and flip operations, selecting the change that most
        improves the scoring function while respecting biological constraints.

        Parameters
        ----------
        scoring_method : str or scoring class, default=None
            Scoring function to optimize. Can be string ("bic", "aic") or
            custom scoring class instance
        start_dag : Optional[DAG], default=None
            Initial DAG structure. If None, starts with empty graph
        tabu_length : int, default=100
            Length of tabu list to prevent cycling in search
        max_indegree : Optional[int], default=None
            Maximum number of parents allowed per node
        expert_knowledge : Optional[ExpertKnowledge], default=None
            Hard constraints on required/forbidden edges
        epsilon : float, default=1e-4
            Minimum score improvement to continue search
        max_iter : int, default=1000000
            Maximum number of search iterations
        show_progress : bool, default=True
            Whether to display progress bar during search
        on_step : Optional[Callable], default=None
            Callback function called after each search step

        Returns
        -------
        DAG
            Estimated causal directed acyclic graph

        Examples
        --------
        >>> # Basic constrained search
        >>> dag = search.estimate(scoring_method="bic")
        >>>
        >>> # With custom scoring and constraints
        >>> expert = ExpertKnowledge()
        >>> expert.add_required_edge(("AKT1", "MDM2"))
        >>> dag = search.estimate(
        ...     scoring_method=custom_scorer,
        ...     expert_knowledge=expert,
        ...     max_indegree=3
        ... )

        Notes
        -----
        The algorithm terminates when either:
        - No operation improves score by more than epsilon
        - Maximum iterations reached
        - No legal operations remain

        Constraint enforcement significantly reduces computational complexity
        compared to unconstrained search, especially for large biological networks.
        """
        score, score_c = get_scoring_method(scoring_method, self.data, self.use_cache)
        score_fn = score_c.local_score

        if start_dag is None:
            start_dag = DAG()
            start_dag.add_nodes_from(self.variables)

        expert_knowledge = expert_knowledge or ExpertKnowledge()

        if not nx.is_directed_acyclic_graph(start_dag):
            raise ValueError("required_edges create a cycle in start_dag.")

        max_indegree = float("inf") if max_indegree is None else max_indegree
        tabu_list = deque(maxlen=tabu_length)
        current_model = start_dag

        it = trange(int(max_iter)) if show_progress else range(int(max_iter))
        for t in it:
            best_op, best_delta = max(
                self._legal_operations(
                    current_model,
                    score_fn,
                    score.structure_prior_ratio,
                    tabu_list,
                    max_indegree,
                    expert_knowledge.forbidden_edges,
                    expert_knowledge.required_edges,
                ),
                key=lambda x: x[1],
                default=(None, None),
            )

            if show_progress:
                try:
                    it.set_postfix({"Δscore": f"{best_delta:.4f}"})
                except Exception:
                    pass

            if on_step is not None:
                on_step(t, best_op, best_delta)

            if best_op is None or best_delta < epsilon:
                break
            if best_op[0] == "+":
                current_model.add_edge(*best_op[1])
                tabu_list.append(("-", best_op[1]))
            elif best_op[0] == "-":
                current_model.remove_edge(*best_op[1])
                tabu_list.append(("+", best_op[1]))
            else:  # flip
                X, Y = best_op[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list.append(best_op)

        return current_model

    def _legal_operations(
        self,
        model: DAG,
        score: Callable,
        structure_score: Callable,
        tabu_list: Deque[Tuple[str, Tuple[Hashable, Hashable]]],
        max_indegree: int,
        forbidden_edges: Iterable[Tuple[str, str]],
        required_edges: Iterable[Tuple[str, str]],
    ) -> Generator[Tuple[Tuple[str, Tuple[Hashable, Hashable]], float], None, None]:
        """
        Generate all legal operations with their score improvements.

        Evaluates three types of operations: edge addition, edge removal, and
        edge reversal. For addition operations, restricts candidates to the
        allowed_additions set if provided, dramatically reducing search space
        for biological applications.

        Parameters
        ----------
        model : DAG
            Current DAG structure being evaluated
        score : Callable
            Local scoring function for individual variables
        structure_score : Callable
            Prior probability function for structure changes
        tabu_list : Deque
            Recent operations to avoid cycling
        max_indegree : int
            Maximum allowed parents per node
        forbidden_edges : Iterable[Tuple[str, str]]
            Hard-forbidden edge constraints
        required_edges : Iterable[Tuple[str, str]]
            Hard-required edge constraints

        Yields
        ------
        Tuple[Tuple[str, Tuple[str, str]], float]
            Operation and its score improvement: ((op_type, (parent, child)), delta)
            where op_type is "+", "-", or "flip"

        Notes
        -----
        The key innovation is constraining ADD operations to allowed_additions,
        which reduces complexity from O(n²) to O(k) where k is the number of
        biologically plausible edges. This maintains discovery quality while
        dramatically improving computational efficiency.

        Operations are filtered by:
        - Tabu list (avoid recent operations)
        - Expert knowledge constraints
        - Acyclicity requirements
        - Maximum indegree limits
        - Biological plausibility (for additions)
        """
        tabu = set(tabu_list)
        existing = set(model.edges())

        # --- ADD: iterate only allowed candidates (if provided)
        if self.allowed_additions is not None:
            potential = self.allowed_additions - existing - {(y, x) for (x, y) in existing}
        else:
            # fall back to full scan
            from itertools import permutations

            potential = (
                set(permutations(self.variables, 2)) - existing - {(y, x) for (x, y) in existing}
            )

        forbidden = set(forbidden_edges)
        required = set(required_edges)

        for X, Y in potential:
            op = ("+", (X, Y))
            # cheap checks first; avoid expensive path query early
            if (op in tabu) or ((X, Y) in forbidden):
                continue
            # cycle check
            if nx.has_path(model, Y, X):
                continue
            parents_old = model.get_parents(Y)
            if len(parents_old) + 1 <= max_indegree:
                delta = score(Y, parents_old + [X]) - score(Y, parents_old)
                delta += structure_score("+")
                yield (op, delta)

        # --- REMOVE: only current edges (unchanged)
        for X, Y in list(existing):
            op = ("-", (X, Y))
            if (op in tabu) or ((X, Y) in required):
                continue
            p_old = model.get_parents(Y)
            p_new = [v for v in p_old if v != X]
            delta = score(Y, p_new) - score(Y, p_old)
            delta += structure_score("-")
            yield (op, delta)

        # --- FLIP: only if reverse is allowed (if using allowed_additions)
        for X, Y in list(existing):
            op = ("flip", (X, Y))
            if (op in tabu) or (("flip", (Y, X)) in tabu) or ((X, Y) in required):
                continue
            if self.allowed_additions is not None and (Y, X) not in self.allowed_additions:
                continue
            if (Y, X) in forbidden:
                continue
            # Cycle check for the flip X->Y => Y->X. After removing X->Y, adding
            # Y->X creates a cycle iff a directed path X~>Y still exists (X~>Y plus
            # Y->X closes a loop), so we must test that direction. The previous
            # check used has_path(Y, X), which let cycle-creating flips through and
            # produced non-DAG search outputs.
            model.remove_edge(X, Y)
            if nx.has_path(model, X, Y):
                model.add_edge(X, Y)
                continue
            model.add_edge(X, Y)

            Xp = model.get_parents(X)
            Yp = model.get_parents(Y)
            if len(Xp) + 1 <= max_indegree:
                delta = (score(X, Xp + [Y]) - score(X, Xp)) + (
                    score(Y, [v for v in Yp if v != X]) - score(Y, Yp)
                )
                delta += structure_score("flip")
                yield (op, delta)


class AICGaussIndraPriors(LogLikelihoodGauss):
    """
    AIC scoring with soft INDRA biological priors.

    Extends standard AIC (Akaike Information Criterion) scoring to incorporate
    soft biological priors from INDRA knowledge base. The scoring function
    balances data fit with biological plausibility, encouraging edges with
    strong prior evidence while penalizing model complexity.

    The score combines:
    - Standard AIC: log-likelihood - (df + 2)
    - Prior bonus: Σ log(p_ij / (1 - p_ij)) for edges with prior probability p_ij

    This approach provides a principled way to incorporate biological knowledge
    without hard constraints, allowing data to override weak priors when
    evidence is strong.

    Parameters
    ----------
    data : pd.DataFrame
        Observational dataset for scoring
    edge_priors : Optional[Dict[Tuple[str, str], float]], default=None
        Dictionary mapping (parent, child) tuples to prior probabilities [0,1].
        Higher values indicate stronger biological evidence for the edge
    prior_strength : float, default=1.0
        Scaling factor for prior influence (λ parameter)
    **kwargs
        Additional arguments passed to LogLikelihoodGauss

    Attributes
    ----------
    edge_priors : Dict[Tuple[str, str], float]
        Edge prior probabilities
    prior_strength : float
        Prior influence scaling parameter

    Examples
    --------
    >>> # Define biological priors from INDRA evidence
    >>> priors = {
    ...     ("AKT1", "MDM2"): 0.8,    # Strong evidence
    ...     ("TP53", "MDM2"): 0.9,    # Very strong evidence
    ...     ("MDM2", "TP53"): 0.7     # Moderate evidence
    ... }
    >>>
    >>> # Initialize AIC scorer with priors
    >>> scorer = AICGaussIndraPriors(data, edge_priors=priors, prior_strength=2.0)
    >>>
    >>> # Use in causal discovery
    >>> search = SparseHillClimb(data)
    >>> dag = search.estimate(scoring_method=scorer)

    Notes
    -----
    The log-odds transformation log(p/(1-p)) provides symmetric treatment
    of prior evidence: strong positive evidence (p=0.9) gives +log(9),
    while strong negative evidence (p=0.1) gives -log(9).

    Prior strength parameter allows tuning the balance between data fit
    and biological plausibility based on confidence in prior knowledge.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        edge_priors: Optional[dict] = None,
        prior_strength: float = 1.0,
        scale_with_n: bool = False,
        **kwargs,
    ):
        super(AICGaussIndraPriors, self).__init__(data, **kwargs)
        self.edge_priors = edge_priors or {}
        self.prior_strength = prior_strength  # This is lambda
        self.scale_with_n = scale_with_n

    def local_score(self, variable: str, parents: list) -> float:
        """
        Compute AIC score with biological priors for a variable given its parents.

        Calculates the local score for a variable conditioned on its parent set,
        combining standard AIC penalty with biological prior information.
        Higher scores indicate better model fit and/or stronger prior support.

        Parameters
        ----------
        variable : str
            Target variable to score
        parents : list
            List of parent variable names

        Returns
        -------
        float
            Local AIC score with prior bonus. Higher values are better.

        Notes
        -----
        Score decomposition:
        - Base AIC: log-likelihood - (degrees_of_freedom + 2)
        - Prior bonus: Σ log(p/(1-p)) × prior_strength for each parent edge
        - Final score: AIC + prior_bonus

        The log-odds formulation ensures that:
        - p > 0.5 gives positive bonus (encourages edge)
        - p < 0.5 gives negative bonus (discourages edge)
        - p = 0.5 gives zero bonus (neutral)

        When scale_with_n=True, prior_bonus is additionally multiplied by log(n).
        This anchors the prior to the BIC/AIC complexity-penalty scale: both the
        penalty and the bonus grow as O(log n), while the data log-likelihood grows
        as O(n).  The net effect is that the prior's influence relative to the
        data signal shrinks as n increases, while remaining proportional to the
        complexity penalty for any fixed n.
        """
        try:
            ll, df_model = self._log_likelihood(variable=variable, parents=parents)
        except:
            # statsmodels will raise ValueError if X is singular
            return -np.inf

        # Standard AIC score
        aic_score = ll - (df_model + 2)

        # Soft prior component
        prior_bonus = 0
        for parent in parents:
            p = self.edge_priors[(parent, variable)]
            p = np.clip(p, 1e-6, 1 - 1e-6)  # Avoid log(0)

            log_odds = np.log(p / (1 - p))
            prior_bonus += log_odds

        # prior_bonus *= self.prior_strength
        # if self.scale_with_n:
        #     prior_bonus *= np.log(self.data.shape[0])
        return aic_score + prior_bonus


class AICGaussNoPriors(LogLikelihoodGauss):

    def __init__(
        self,
        data: pd.DataFrame,
        edge_priors: Optional[dict] = None,
        prior_strength: float = 1.0,
        scale_with_n: bool = False,
        **kwargs,
    ):
        super(AICGaussNoPriors, self).__init__(data, **kwargs)
        self.edge_priors = edge_priors or {}
        self.prior_strength = prior_strength  # This is lambda
        self.scale_with_n = scale_with_n

    def local_score(self, variable: str, parents: list) -> float:

        try:
            ll, df_model = self._log_likelihood(variable=variable, parents=parents)
        except:
            # statsmodels will raise ValueError if X is singular
            return -np.inf

        # Standard AIC score
        aic_score = ll - (df_model + 2)

        return aic_score


class BICGaussIndraPriors(LogLikelihoodGauss):
    """
    BIC scoring with soft INDRA biological priors.

    Extends standard BIC (Bayesian Information Criterion) scoring to incorporate
    soft biological priors from INDRA knowledge base. BIC applies stronger
    penalty for model complexity than AIC, making it more conservative in
    edge selection while still benefiting from biological prior knowledge.

    The score combines:
    - Standard BIC: log-likelihood - ((df + 2)/2) × log(n)
    - Prior bonus: Σ log(p_ij / (1 - p_ij)) for edges with prior probability p_ij

    BIC's stronger complexity penalty makes it particularly suitable when
    aiming for sparse, interpretable causal networks with high confidence.

    Parameters
    ----------
    data : pd.DataFrame
        Observational dataset for scoring
    edge_priors : Optional[Dict[Tuple[str, str], float]], default=None
        Dictionary mapping (parent, child) tuples to prior probabilities [0,1]
    prior_strength : float, default=1.0
        Scaling factor for prior influence (λ parameter)
    **kwargs
        Additional arguments passed to LogLikelihoodGauss

    Attributes
    ----------
    edge_priors : Dict[Tuple[str, str], float]
        Edge prior probabilities
    prior_strength : float
        Prior influence scaling parameter

    Examples
    --------
    >>> # Initialize BIC scorer with biological priors
    >>> priors = {("AKT1", "MDM2"): 0.85, ("TP53", "MDM2"): 0.92}
    >>> scorer = BICGaussIndraPriors(data, edge_priors=priors)
    >>>
    >>> # Use in constrained causal discovery
    >>> search = SparseHillClimb(data, allowed_additions=list(priors.keys()))
    >>> dag = search.estimate(scoring_method=scorer)

    Notes
    -----
    BIC vs AIC trade-offs:
    - BIC: More conservative, stronger complexity penalty, better for sparse networks
    - AIC: More liberal, weaker complexity penalty, better for predictive models

    Choose BIC when interpretability and network sparsity are priorities.
    Choose AIC when predictive performance is the primary concern.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        edge_priors: Optional[dict] = None,
        prior_strength: float = 1.0,
        scale_with_n: bool = False,
        **kwargs,
    ):
        super().__init__(data, **kwargs)
        self.edge_priors = edge_priors or {}
        self.prior_strength = prior_strength  # This is lambda
        self.scale_with_n = scale_with_n

    def local_score(self, variable: str, parents: list) -> float:
        """
        Compute BIC score with biological priors for a variable given its parents.

        Calculates the local score using BIC criterion enhanced with biological
        prior information. BIC applies stronger complexity penalty than AIC,
        promoting sparser models while incorporating prior biological knowledge.

        Parameters
        ----------
        variable : str
            Target variable to score
        parents : list
            List of parent variable names

        Returns
        -------
        float
            Local BIC score with prior bonus. Higher values are better.
            Returns -inf if computation fails (singular matrix, etc.)

        Notes
        -----
        Score decomposition:
        - Base BIC: log-likelihood - ((df + 2)/2) × log(n)
        - Prior bonus: Σ log(p/(1-p)) × prior_strength for each parent edge
        - Final score: BIC + prior_bonus

        BIC's log(n) factor creates stronger penalty for complexity than AIC,
        making it more conservative in edge selection. This is beneficial when
        seeking interpretable, sparse causal networks.

        When scale_with_n=True, prior_bonus is additionally multiplied by log(n).
        This anchors the prior to the BIC complexity-penalty scale: both the
        penalty and the bonus grow as O(log n), while the data log-likelihood grows
        as O(n).  The net effect is that the prior's influence relative to the
        data signal shrinks as n increases, while remaining proportional to the
        complexity penalty for any fixed n.

        Error handling returns -inf for degenerate cases (singular covariance
        matrices, etc.) to exclude them from consideration.
        """
        try:
            ll, df_model = self._log_likelihood(variable=variable, parents=parents)
        except:
            # statsmodels will raise ValueError if X is singular
            return -np.inf

        # Standard BIC score
        bic_score = ll - (((df_model + 2) / 2) * np.log(self.data.shape[0]))

        # Soft prior component
        prior_bonus = 0
        for parent in parents:
            p = self.edge_priors[(parent, variable)]
            p = np.clip(p, 1e-6, 1 - 1e-6)  # Avoid log(0)
            log_odds = np.log(p / (1 - p))
            prior_bonus += log_odds

        # prior_bonus *= self.prior_strength
        # if self.scale_with_n:
        #     prior_bonus *= np.log(self.data.shape[0])
        return bic_score + prior_bonus


class BICGaussNoPriors(LogLikelihoodGauss):
    def __init__(
        self,
        data: pd.DataFrame,
        edge_priors: Optional[dict] = None,
        prior_strength: float = 1.0,
        **kwargs,
    ):
        super().__init__(data, **kwargs)
        self.edge_priors = edge_priors or {}
        self.prior_strength = prior_strength  # This is lambda

    def local_score(self, variable: str, parents: list) -> float:

        try:
            ll, df_model = self._log_likelihood(variable=variable, parents=parents)
        except:
            # statsmodels will raise ValueError if X is singular
            return -np.inf

        # Standard BIC score
        bic_score = ll - (((df_model + 2) / 2) * np.log(self.data.shape[0]))

        return bic_score


def random_acyclic_subgraph(nodes, allowed_edges, inclusion_prob=0.15, rng=None, max_indegree=2):
    """Generate a random DAG by greedily adding allowed edges without creating cycles.

    Parameters
    ----------
    nodes : list
        Node labels for the DAG
    allowed_edges : iterable of (str, str)
        Candidate edges to sample from
    inclusion_prob : float, default=0.15
        Probability of attempting to include each edge
    rng : numpy Generator, optional
        Random number generator for reproducibility
    max_indegree : int, default=2
        Maximum number of parents allowed per node
    """
    if rng is None:
        rng = np.random.default_rng()

    dag = DAG()
    dag.add_nodes_from(nodes)

    edges = list(allowed_edges)
    rng.shuffle(edges)

    for u, v in edges:
        if rng.random() > inclusion_prob:
            continue
        if len(dag.get_parents(v)) >= max_indegree:
            continue
        dag.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(dag):
            dag.remove_edge(u, v)

    return dag


def process_bootstrap(
    data: pd.DataFrame,
    edge_priors: dict,
    prior_strength: float,
    score_fn: type,
    estimator: type,
    expert_knowledge: ExpertKnowledge,
    seed: int = 0,
    random_init: bool = False,
    subsample_frac: float = 0.65,
    replace: bool = True,
) -> Optional[DAG]:
    """
    Process single bootstrap sample for causal discovery with uncertainty quantification.

    Performs causal discovery on a bootstrap resample of the data using constrained
    Hill Climb search with biological priors. This function is designed for parallel
    execution to enable robust uncertainty estimation through bootstrap aggregation.

    The bootstrap procedure helps quantify uncertainty in causal edge discovery by:
    1. Resampling data with replacement
    2. Running constrained causal discovery
    3. Aggregating results across multiple bootstrap samples

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset to resample
    edge_priors : dict
        Dictionary of edge prior probabilities for biological constraints
    prior_strength : float
        Scaling factor for prior influence in scoring
    score_fn : type
        Scoring function class (AICGaussIndraPriors or BICGaussIndraPriors)
    estimator : type
        Causal discovery algorithm class (typically SparseHillClimb)
    expert_knowledge : ExpertKnowledge
        Hard constraints on required/forbidden edges
    seed : int, optional
        Random seed for reproducible bootstrap resampling. Default is 0.
    random_init : bool, optional
        If True, initialize the hill climb search from a random acyclic subgraph
        rather than an empty DAG. This can help escape local optima but increases
        run-to-run variability. Default is False.

    Returns
    -------
    Optional[DAG]
        Estimated causal DAG from bootstrap sample, or None if discovery fails

    Examples
    --------
    >>> # Single bootstrap iteration
    >>> dag = process_bootstrap(
    ...     data=proteomics_data,
    ...     edge_priors=indra_priors,
    ...     prior_strength=2.0,
    ...     score_fn=BICGaussIndraPriors,
    ...     estimator=SparseHillClimb,
    ...     expert_knowledge=constraints
    ... )

    Notes
    -----
    This function includes error handling to gracefully manage numerical
    issues or convergence failures that may occur during bootstrap resampling.
    Failed bootstrap samples return None and are excluded from aggregation.

    The logging suppression prevents verbose output during parallel execution
    while maintaining error reporting for debugging.
    """
    import logging

    # try:
    # Suppress INFO logs from pgmpy in this subprocess
    logging.getLogger("pgmpy").setLevel(logging.WARNING)

    rng = np.random.RandomState(seed)
    # subsample_frac<1 with replace=True -> a bootstrap resample (consensus mode);
    # subsample_frac=1 with replace=False -> the full data (best-of-restarts mode).
    resampled_data = data.sample(frac=subsample_frac, replace=replace, random_state=rng)

    # Initialize the custom scoring function
    custom_score = score_fn(resampled_data, edge_priors=edge_priors, prior_strength=prior_strength)

    allowed = set(edge_priors.keys())
    est = estimator(data=resampled_data, allowed_additions=allowed)

    start_dag = None
    if random_init:
        nodes = list(resampled_data.columns)
        start_dag = random_acyclic_subgraph(nodes, allowed, 0.15, np.random.default_rng(seed))

    # Estimate the DAG using the custom scoring function
    estimated_dag = est.estimate(
        scoring_method=custom_score,
        start_dag=start_dag,
        expert_knowledge=expert_knowledge,
        max_indegree=3,
        epsilon=0.0001,
        show_progress=False,
    )
    return estimated_dag


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


def run_bootstrap(
    data: pd.DataFrame,
    indra_priors: pd.DataFrame,
    prior_strength: float,
    scoring_function: type,
    search_algorithm: type,
    expert_knowledge: ExpertKnowledge,
    add_high_corr_edges_to_priors: bool = False,
    corr_threshold: float = 0.8,
    n_bootstrap: int = 100,
    convert_to_probability: bool = True,
    use_source_counts: bool = False,
    random_init: bool = False,
    subsample_frac: float = 0.65,
    replace: bool = True,
    verbose: bool = True,
) -> list:
    """
    Run parallel bootstrap analysis for robust causal discovery with INDRA priors.

    Performs bootstrap resampling and causal discovery to quantify uncertainty
    in learned causal relationships using INDRA biological prior knowledge.
    This approach provides confidence estimates for individual edges by examining
    their frequency across bootstrap samples while leveraging biological constraints.

    The function automatically processes INDRA priors to extract edge probabilities
    using power law modeling, then runs parallel bootstrap analysis for efficient
    uncertainty quantification in biologically-informed causal discovery.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset for bootstrap resampling with samples as rows and variables as columns
    indra_priors : pd.DataFrame
        DataFrame containing INDRA prior information with columns:
        - 'source': Source protein/gene symbols
        - 'target': Target protein/gene symbols
        - 'evidence_count': Evidence count for each relationship
    prior_strength : float
        Scaling factor for biological prior influence in scoring functions
    scoring_function : type
        Scoring function class with prior integration (AICGaussIndraPriors or BICGaussIndraPriors)
    search_algorithm : type
        Causal discovery algorithm class (typically SparseHillClimb)
    expert_knowledge : ExpertKnowledge
        Hard constraints on graph structure (required/forbidden edges)
    add_high_corr_edges_to_priors: bool
        If True, identify highly correlated variable pairs in the data and
        remove edges between them from the blacklist. This helps retain
        potentially valid causal edges that might otherwise be excluded.
    n_bootstrap : int
        Number of bootstrap samples to generate for uncertainty quantification
    convert_to_probability : bool
        If True, convert INDRA evidence counts to edge probabilities using power law modeling
    random_init : bool, optional
        If True, initialize each bootstrap hill climb from a random acyclic subgraph
        rather than an empty DAG. Default is False.

    Returns
    -------
    list
        List of estimated DAGs from bootstrap samples.
        Failed samples are excluded (None values filtered out).

    Examples
    --------
    >>> # Prepare INDRA prior data
    >>> indra_df = pd.DataFrame({
    ...     'source': ['AKT1', 'TP53', 'MDM2'],
    ...     'target': ['MDM2', 'MDM2', 'TP53'],
    ...     'evidence_count': [15, 25, 8]
    ... })
    >>>
    >>> # Run bootstrap causal discovery with biological priors
    >>> bootstrap_dags = run_bootstrap(
    ...     data=proteomics_data,
    ...     indra_priors=indra_df,
    ...     prior_strength=2.0,
    ...     scoring_function=BICGaussIndraPriors,
    ...     search_algorithm=SparseHillClimb,
    ...     expert_knowledge=ExpertKnowledge(),
    ...     n_bootstrap=100
    ... )
    >>>
    >>> # Analyze edge confidence from bootstrap results
    >>> edge_counts = Counter()
    >>> for dag in bootstrap_dags:
    ...     if dag is not None:  # Filter out failed bootstraps
    ...         edge_counts.update(dag.edges())
    >>> edge_frequencies = {
    ...     edge: count/len([d for d in bootstrap_dags if d is not None])
    ...     for edge, count in edge_counts.items()
    ... }

    Notes
    -----
    Workflow:
    1. Convert INDRA evidence counts to edge probabilities using power law modeling
    2. Run parallel bootstrap resampling with constrained causal discovery
    3. Aggregate results for uncertainty quantification

    The parallel execution uses n_jobs=-2 to reserve one CPU core for system
    processes while maximizing computational throughput. This prevents system
    overload during intensive bootstrap computations.

    Bootstrap aggregation provides several benefits for biological applications:
    - Confidence intervals for individual causal relationships
    - Robust consensus network structure from noisy biological data
    - Uncertainty quantification for causal claims in publications
    - Model stability assessment across data perturbations

    Typical bootstrap sample sizes for biological networks:
    - Small networks (< 20 nodes): 50-100 samples
    - Medium networks (20-100 nodes): 100-500 samples
    - Large networks (> 100 nodes): 200-1000 samples

    The choice depends on computational resources and required precision
    for downstream biological interpretation and hypothesis generation.
    """
    if verbose:
        print("INFO: Starting bootstrap causal discovery:")
    if add_high_corr_edges_to_priors:
        if verbose:
            print("INFO: Adding high-corr edges to priors:")
        updated_indra_priors, updated_blacklist = remove_high_corr_edges_from_blacklist(
            data, indra_priors, expert_knowledge.forbidden_edges, corr_threshold, verbose=verbose
        )
        expert_knowledge.forbidden_edges = updated_blacklist
    else:
        updated_indra_priors = indra_priors

    if verbose:
        print("INFO: Calculating edge probabilities.")

    edge_probabilities = prepare_indra_priors(
        updated_indra_priors, convert_to_probability, use_source_counts
    )

    if verbose:
        print("INFO: Running bootstrap.")
    bootstrap_dags = Parallel(n_jobs=-2)(
        delayed(process_bootstrap)(
            data,
            edge_probabilities,
            prior_strength,
            scoring_function,
            search_algorithm,
            expert_knowledge,
            seed=i,
            random_init=random_init,
            subsample_frac=subsample_frac,
            replace=replace,
        )
        for i in tqdm(range(n_bootstrap), desc="Hill Climb runs")
    )
    # for _ in range(n_bootstrap):
    #     process_bootstrap(
    #         data,
    #         edge_probabilities,
    #         prior_strength,
    #         scoring_function,
    #         search_algorithm,
    #         expert_knowledge,
    #     )

    return bootstrap_dags


def _build_dagma_exclude_edges(
    node_order: list, allowed_edges: Set[Tuple[str, str]]
) -> Tuple[Tuple[int, int], ...]:
    """Build a hard edge blacklist for ``DagmaLinear.fit``'s ``exclude_edges``.

    Every ordered pair of distinct nodes in ``node_order`` that is not present
    in ``allowed_edges`` is blacklisted, mirroring how
    ``SparseHillClimb.allowed_additions`` restricts edge additions to the
    INDRA prior network during hill climbing.

    Parameters
    ----------
    node_order : list
        Node names in the same order as the columns of the data matrix
        passed to ``DagmaLinear.fit``. Blacklist indices are positional.
    allowed_edges : set of (str, str)
        (parent, child) pairs that are allowed to appear in the learned DAG.

    Returns
    -------
    tuple of (int, int)
        Row/column index pairs to exclude, in the same (parent, child)
        convention as the estimated weight matrix ``W`` (``W[i, j]`` is the
        edge from node ``i`` to node ``j``). ``DagmaLinear.fit`` requires an
        actual ``tuple`` here -- a ``list`` fails its internal type check and
        is silently ignored, so the return type is load-bearing.
    """
    index = {node: i for i, node in enumerate(node_order)}
    return tuple(
        (index[u], index[v])
        for u in node_order
        for v in node_order
        if u != v and (u, v) not in allowed_edges
    )


def evidence_penalty(
    belief: np.ndarray, mask: np.ndarray, clip: float = 3.0, center: bool = False
) -> np.ndarray:
    """Per-edge L1 penalty multiplier from INDRA evidence log-odds.

    Edges with belief > 0.5 (net positive evidence) get a multiplier < 1,
    lowering their effective L1 penalty in ``DagmaLinear``'s loss (more
    likely to survive thresholding); edges with belief < 0.5 get a
    multiplier > 1 (more discouraged). Only positions where ``mask`` is
    True are reweighted; everywhere else the multiplier is 1.0 (DAGMA's
    unweighted default).

    Parameters
    ----------
    belief : np.ndarray
        (d, d) matrix of prior edge probabilities in [0, 1].
    mask : np.ndarray
        (d, d) boolean matrix marking which positions have real prior
        evidence (as opposed to a filler default value in ``belief``).
    clip : float, default=3.0
        Bounds the log-odds before exponentiating, so a single very
        strong/weak prior can't drive the multiplier to 0 or blow it up.
    center : bool, default=False
        If True, subtract the mean log-odds (over masked entries) before
        exponentiating, so the penalty is relative to the average prior
        strength in this graph rather than to p=0.5.

    Returns
    -------
    np.ndarray
        (d, d) multiplier matrix; DAGMA's per-edge L1 penalty becomes
        ``lambda1 * multiplier`` (elementwise).

    Raises
    ------
    ValueError
        If any ``belief[mask]`` value falls outside [0, 1]. Silently clipping
        out-of-range values (e.g. raw evidence counts left un-converted by
        ``prepare_indra_priors(..., convert_to_probability=False)``) would
        collapse every evidenced edge to the same maximal log-odds -- a
        uniform, not per-edge, discount -- without ever raising an error.
    """
    m = mask.astype(bool)
    raw = belief[m]
    if raw.size and ((raw < 0) | (raw > 1)).any():
        bad = np.unique(raw[(raw < 0) | (raw > 1)])[:5]
        raise ValueError(
            "evidence_penalty expected belief[mask] to be probabilities in "
            f"[0, 1], but found values outside that range (e.g. {bad.tolist()}). "
            "This usually means raw evidence counts (or some other unconverted "
            "quantity) were passed as `belief` -- convert them to probabilities "
            "first, e.g. via prepare_indra_priors(..., convert_to_probability=True)."
        )
    p = np.clip(raw, 1e-6, 1 - 1e-6)
    ell = np.log(p / (1 - p))
    if center:
        ell = ell - ell.mean()
    C = np.ones_like(belief)
    C[m] = np.exp(np.clip(-ell, -clip, clip))
    return C


def _build_dagma_belief_matrix(
    node_order: list, edge_priors: dict, default_belief: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (belief, mask) matrices ``evidence_penalty`` expects.

    Parameters
    ----------
    node_order : list
        Node names in the same order as the DAGMA data columns.
    edge_priors : dict
        {(parent, child): probability} as returned by ``prepare_indra_priors``.
    default_belief : float, default=0.5
        Filler value for positions with no prior evidence (masked out, so
        this value never actually affects the penalty).

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``belief`` -- (d, d) prior probabilities, ``mask`` -- (d, d) boolean
        array marking which positions came from ``edge_priors``.
    """
    d = len(node_order)
    index = {node: i for i, node in enumerate(node_order)}
    belief = np.full((d, d), default_belief, dtype=float)
    mask = np.zeros((d, d), dtype=bool)
    for (u, v), p in edge_priors.items():
        i, j = index.get(u), index.get(v)
        if i is None or j is None or i == j:
            continue
        belief[i, j] = p
        mask[i, j] = True
    return belief, mask


class _EvidenceWeightedDagmaLinear(DagmaLinear):
    """``DagmaLinear`` with a per-edge L1 penalty multiplier from INDRA evidence.

    ``DagmaLinear`` (dagma==1.1.1) applies its L1 penalty as a scalar
    subgradient term (``lambda1 * sign(W)``) inline inside ``minimize``, with
    no public hook to vary it per edge. This subclass duplicates
    ``minimize``/``_func`` from that version, replacing the scalar
    ``lambda1`` term with an elementwise product against ``penalty_weights``
    (see ``evidence_penalty``) wherever it appears, so edges with strong
    INDRA evidence face a smaller effective penalty and edges with weak/no
    evidence face a larger one. Everything else (Adam updates, the
    include/exclude masks, the M-matrix domain checks) is unchanged from the
    parent class.

    Set ``penalty_weights`` (a (d, d) array, or ``None``) before calling
    ``fit()``; ``None`` (the default set in ``__init__``) reproduces vanilla
    ``DagmaLinear`` behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty_weights = None

    def _func(
        self, W: np.ndarray, mu: float, s: float = 1.0
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        weights = 1.0 if self.penalty_weights is None else self.penalty_weights
        score, _ = self._score(W)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * (weights * np.abs(W)).sum()) + h
        return obj, score, h

    def minimize(
        self,
        W: np.ndarray,
        mu: float,
        max_iter: int,
        s: float,
        lr: float,
        tol: float = 1e-6,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        pbar: Optional[Any] = None,
    ) -> Tuple[np.ndarray, bool]:
        weights = 1.0 if self.penalty_weights is None else self.penalty_weights

        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        self.vprint(
            f"\n\nMinimize with -- mu:{mu} -- lr: {lr} -- s: {s} -- "
            f"l1: {self.lambda1} for {max_iter} max iterations"
        )
        mask_inc = np.zeros((self.d, self.d))
        if self.inc_c is not None:
            mask_inc[self.inc_r, self.inc_c] = -2 * mu * self.lambda1
        mask_exc = np.ones((self.d, self.d), dtype=self.dtype)
        if self.exc_c is not None:
            mask_exc[self.exc_r, self.exc_c] = 0.0

        for iter in range(1, max_iter + 1):
            # Compute the (sub)gradient of the objective
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0):  # sI - W o W is not an M-matrix
                if iter == 1 or s <= 0.9:
                    self.vprint(f"W went out of domain for s={s} at iteration {iter}")
                    return W, False
                else:
                    W += lr * grad
                    lr *= 0.5
                    if lr <= 1e-16:
                        return W, True
                    W -= lr * grad
                    M = sla.inv(s * self.Id - W * W) + 1e-16
                    self.vprint(f"Learning rate decreased to lr: {lr}")

            if self.loss_type == "l2":
                G_score = -mu * self.cov @ (self.Id - W)
            elif self.loss_type == "logistic":
                G_score = mu / self.n * self.X.T @ sigmoid(self.X @ W) - mu * self.cov

            Gobj = (
                G_score
                + mu * self.lambda1 * weights * np.sign(W)
                + 2 * W * M.T
                + mask_inc * np.sign(W)
            )

            # Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad
            W *= mask_exc

            # Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, score, h = self._func(W, mu, s)
                self.vprint(f"\nInner iteration {iter}")
                self.vprint(f"\th(W_est): {h:.4e}")
                self.vprint(f"\tscore(W_est): {score:.4e}")
                self.vprint(f"\tobj(W_est): {obj_new:.4e}")
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter - iter + 1)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return W, True


def _refit_unpenalized_ols(X: np.ndarray, dag: nx.DiGraph, node_order: list) -> None:
    """Replace DAGMA's (penalized, shrunk) edge weights with unbiased OLS coefficients.

    DAGMA's L1 penalty -- and, when ``use_evidence_weights`` is set, the
    per-edge evidence multiplier on top of it -- shrinks every surviving
    coefficient by a different amount depending on its prior strength. That
    heterogeneous shrinkage means the raw penalized weight magnitude isn't
    comparable across edges, which is fine for *selecting* the support (that's
    ``w_threshold``'s job, already applied by the time this runs) but wrong
    for *reporting* edge strength. This refits each node's parents with plain
    OLS on the already-selected support -- unbiased, comparable magnitudes,
    with no further change to which edges are in ``dag``.

    Mutates ``dag`` in place, adding a ``weight`` attribute (the signed OLS
    coefficient) to every edge.

    Parameters
    ----------
    X : np.ndarray
        The same (already centered) data matrix passed to ``DagmaLinear.fit``.
    dag : nx.DiGraph
        The thresholded DAGMA graph; nodes must be a subset of ``node_order``.
    node_order : list
        Node names in the same order as ``X``'s columns.
    """
    index = {node: i for i, node in enumerate(node_order)}
    for node in dag.nodes():
        parents = list(dag.predecessors(node))
        if not parents:
            continue
        child_col = X[:, index[node]]
        parent_cols = X[:, [index[p] for p in parents]]
        coefs, *_ = np.linalg.lstsq(parent_cols, child_col, rcond=None)
        for parent, coef in zip(parents, coefs):
            dag[parent][node]["weight"] = float(coef)


def run_dagma(
    data: pd.DataFrame,
    indra_priors: pd.DataFrame,
    lambda1: float = 0.02,
    w_threshold: float = 0.2,
    loss_type: str = "l2",
    use_evidence_weights: bool = False,
    convert_to_probability: bool = True,
    use_source_counts: bool = False,
    evidence_clip: float = 3.0,
    evidence_center: bool = False,
    dagma_fit_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> nx.DiGraph:
    """Learn a DAG with DAGMA, restricted to edges present in the INDRA prior.

    This is a continuous-optimization baseline alternative to
    ``SparseHillClimb``: instead of a discrete hill-climb search, DAGMA
    (Bello et al., 2022) solves a differentiable acyclicity-constrained
    optimization over the full data in one shot. Edges outside the INDRA
    prior network are hard-blacklisted via ``exclude_edges``, the same
    "only allow prior edges" restriction that ``SparseHillClimb`` enforces
    with ``allowed_additions``.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data; columns are variables.
    indra_priors : pd.DataFrame
        Prior network with 'source'/'target' columns; only these edges
        (after stripping hyphens, matching the rest of this module) are
        allowed in the learned DAG.
    lambda1 : float, default=0.02
        L1 sparsity penalty passed to ``DagmaLinear.fit``.
    w_threshold : float, default=0.2
        Post-hoc weight threshold; entries of the estimated weighted
        adjacency matrix with magnitude below this are zeroed out.
    loss_type : str, default="l2"
        Loss type passed to ``DagmaLinear`` (see the dagma package docs).
    use_evidence_weights : bool, default=False
        If True, additionally scale the L1 penalty per allowed edge by its
        INDRA evidence strength (see ``evidence_penalty``): edges with
        strong evidence (probability > 0.5) get a smaller effective
        penalty, edges with weak evidence get a larger one. If False, all
        allowed edges share the same ``lambda1`` (the hard prior blacklist
        is unaffected either way).
    convert_to_probability : bool, default=True
        Passed to ``prepare_indra_priors`` when building evidence weights.
    use_source_counts : bool, default=False
        Passed to ``prepare_indra_priors`` when building evidence weights.
    evidence_clip : float, default=3.0
        Passed to ``evidence_penalty`` as ``clip``.
    evidence_center : bool, default=False
        Passed to ``evidence_penalty`` as ``center``.
    dagma_fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``DagmaLinear.fit`` -- most
        usefully its convergence schedule (``T``, ``warm_iter``, ``max_iter``,
        ``mu_init``, ``mu_factor``, ``s``, ``lr``, ...). DAGMA's defaults
        (T=5, warm_iter=3e4, max_iter=6e4) assume a per-iteration cost that
        stays cheap; each iteration does a dense (d, d) matrix inversion
        (``d`` = number of columns in ``data``), so wall-clock time scales
        with ``d`` regardless of how few edges the INDRA prior allows. For
        graphs of a few hundred nodes or more, a lighter schedule (e.g.
        ``{"T": 3, "warm_iter": 3000, "max_iter": 6000}``) is often necessary
        to keep runtime reasonable. Must not set ``lambda1``, ``w_threshold``,
        or ``exclude_edges`` (already supplied by this function's own
        arguments). Default is None (use ``DagmaLinear.fit``'s own defaults).
    verbose : bool, default=False
        Passed through to ``DagmaLinear``.

    Returns
    -------
    nx.DiGraph
        Learned DAG over ``data.columns``, containing only edges allowed by
        ``indra_priors``. DAGMA's acyclicity constraint only holds exactly in
        the limit of the augmented-Lagrangian's central path (as ``T`` -> inf);
        at a finite ``T`` the raw weighted matrix can retain a residual cycle,
        which thresholding cannot introduce but also cannot guarantee away.
        This function raises ``ValueError`` if the thresholded result is not
        a DAG, rather than silently returning a cyclic graph. Each edge's
        ``weight`` attribute is a refit, unpenalized OLS coefficient (see
        ``_refit_unpenalized_ols``), not DAGMA's raw penalized weight -- the
        penalized weight only decided *which* edges survive ``w_threshold``;
        it's a biased (shrunk) estimate of their strength, especially with
        ``use_evidence_weights=True`` where the shrinkage varies per edge.

    Notes
    -----
    ``DagmaLinear`` represents the learned graph as a weighted adjacency
    matrix ``W`` where ``W[i, j] != 0`` means an edge from node ``i``
    (parent) to node ``j`` (child) -- verified empirically against a
    synthetic linear SCM with a known effect direction.
    """
    node_order = list(data.columns)

    allowed_edges = {
        (
            str(indra_priors.loc[i, "source"]).replace("-", ""),
            str(indra_priors.loc[i, "target"]).replace("-", ""),
        )
        for i in range(len(indra_priors))
    }

    exclude_edges = _build_dagma_exclude_edges(node_order, allowed_edges) or None

    if verbose:
        print(
            f"INFO: Running DAGMA on {len(node_order)} nodes with "
            f"{len(allowed_edges)} allowed edges (lambda1={lambda1}, "
            f"w_threshold={w_threshold}, use_evidence_weights={use_evidence_weights})."
        )

    if use_evidence_weights:
        cleaned_priors = indra_priors.copy()
        cleaned_priors["source"] = cleaned_priors["source"].astype(str).str.replace("-", "")
        cleaned_priors["target"] = cleaned_priors["target"].astype(str).str.replace("-", "")
        edge_priors = prepare_indra_priors(
            cleaned_priors, convert_to_probability, use_source_counts
        )
        belief, mask = _build_dagma_belief_matrix(node_order, edge_priors)
        penalty_weights = evidence_penalty(belief, mask, clip=evidence_clip, center=evidence_center)

        dagma_model = _EvidenceWeightedDagmaLinear(loss_type=loss_type, verbose=False)
        dagma_model.penalty_weights = penalty_weights
    else:
        dagma_model = DagmaLinear(loss_type=loss_type, verbose=False)

    # DagmaLinear.fit centers internally for loss_type="l2", but it does so by
    # mutating whatever array it's given in place -- which, via pandas'
    # .values, leaks back into (and silently zero-means) the caller's own
    # DataFrame. Centering explicitly here, into a fresh array, avoids both
    # that side effect and any dependence on dagma's internal behavior.
    X = data.values
    X = X - X.mean(axis=0, keepdims=True)

    W_est = dagma_model.fit(
        X,
        lambda1=lambda1,
        w_threshold=w_threshold,
        exclude_edges=exclude_edges,
        **(dagma_fit_kwargs or {}),
    )

    dag = nx.DiGraph()
    dag.add_nodes_from(node_order)
    d = len(node_order)
    for i in range(d):
        for j in range(d):
            if W_est[i, j] != 0:
                dag.add_edge(node_order[i], node_order[j])

    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError(
            "DAGMA returned a graph with a cycle after thresholding at w_threshold="
            f"{w_threshold}. This means the fit hadn't converged onto the acyclic "
            "manifold (h(W) not ~0) at the configured schedule -- try a larger T "
            "(and/or warm_iter/max_iter) via dagma_fit_kwargs."
        )

    _refit_unpenalized_ols(X, dag, node_order)

    return dag
