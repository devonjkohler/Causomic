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

    Interventional (GIES-style) scoring
    ------------------------------------
    Passing `interventional=True` together with `arm_labels` (and, usually,
    `clamped_nodes`) switches `local_score` to a second code path that sums
    log-likelihood contributions across experimental arms instead of fitting one
    flat GLM over all of `data` - see `local_score` / `_local_score_interventional`.
    This is strictly opt-in: with the default `interventional=False`, or with
    `interventional=True` but no `arm_labels`, `local_score` runs the exact same
    single-GLM-over-self.data code path as before this feature existed. Existing
    callers (all of which construct this class without either argument) are
    unaffected byte-for-byte.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        edge_priors: Optional[dict] = None,
        prior_strength: float = 1.0,
        scale_with_n: bool = False,
        interventional: bool = False,
        arm_labels: Optional[pd.Series] = None,
        clamped_nodes: Optional[dict] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Observational (or, with `interventional=True`, pooled multi-arm)
            dataset for scoring.
        edge_priors : Optional[dict], default=None
            Dictionary mapping (parent, child) tuples to prior probabilities [0,1].
        prior_strength : float, default=1.0
            Scaling factor for prior influence (λ parameter).
        scale_with_n : bool, default=False
            See `local_score`'s docstring for the (currently inert - the
            multiplication is commented out) intended effect.
        interventional : bool, default=False
            If True *and* `arm_labels` is given, `local_score` sums per-arm
            log-likelihoods (skipping arms where `variable` was clamped) instead
            of fitting one GLM over all of `data`. Default False, and the
            fallback when `arm_labels` is missing, reproduces prior behavior
            exactly - see class docstring.
        arm_labels : Optional[pd.Series], default=None
            Per-sample experimental-arm label, one entry per row of `data`.
            Must share `data`'s index. Required for the interventional branch to
            activate at all; if None, `local_score` always uses the flat
            observational path regardless of `interventional`.
        clamped_nodes : Optional[dict], default=None
            Maps an arm label (as found in `arm_labels`) to the list of node
            names pharmacologically clamped in that arm. An arm absent from this
            dict (or the dict being None/empty) is treated as having no clamped
            nodes. A clamped node is only excluded from *its own* local-score
            arm-contribution in the arm(s) where it's clamped; it remains a valid
            regressor for every other variable's score in that same arm - that's
            the whole point of interventional data (see
            `_local_score_interventional`).
        """
        super().__init__(data, **kwargs)
        self.edge_priors = edge_priors or {}
        self.prior_strength = prior_strength  # This is lambda
        self.scale_with_n = scale_with_n
        self.interventional = interventional
        if arm_labels is not None:
            assert arm_labels.index.equals(
                data.index
            ), "arm_labels must share data's index (one label per row of data)."
        self.arm_labels = arm_labels
        self.clamped_nodes = clamped_nodes or {}

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

        If `self.interventional` and `self.arm_labels` are both set, this
        delegates to `_local_score_interventional` instead - see that method.
        Otherwise (the default), everything below is unchanged from before that
        branch existed.
        """
        if self.interventional and self.arm_labels is not None:
            return self._local_score_interventional(variable, parents)

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

    def _local_score_interventional(self, variable: str, parents: list) -> float:
        """
        GIES-style interventional local score: sum log-likelihood of
        `variable | parents` across experimental arms, then apply the BIC
        complexity penalty once over the pooled effective sample size.

        For each arm in `self.arm_labels`, `variable`'s contribution is dropped
        entirely if `variable` is listed in `self.clamped_nodes[arm]` - a clamped
        node's value there is experimenter-set, not generated by its parents, so
        that arm carries no information about *this* edge. Every other arm still
        contributes, including arms where one of `parents` (not `variable`) is
        clamped: clamping never removes a column from `self.data`, only changes
        which arms' *rows* count toward `variable`'s own score, so a clamped
        parent's (fixed) value remains a perfectly ordinary regressor for scoring
        `variable` in that arm. This is asserted explicitly below rather than
        left implicit.

        `_log_likelihood` (inherited from pgmpy's LogLikelihoodGauss) always reads
        `self.data`, with no data-argument override, so each arm's contribution is
        computed by temporarily pointing `self.data` at that arm's row-subset,
        calling `_log_likelihood`, and restoring `self.data` in a `finally` block -
        reusing the exact same GLM-fitting code the observational branch uses,
        rather than duplicating it.

        Returns -inf if `variable` is clamped in every arm (nothing to score), if
        any contributing arm's fit is singular (matches the observational
        branch's all-or-nothing -inf handling), or if arms disagree on
        `df_model` (a small, post-resampling arm can be rank-deficient for the
        full parent set - treated as a degenerate fit, not raised on).
        """
        contributing_arms = [
            arm
            for arm in pd.unique(self.arm_labels)
            if variable not in self.clamped_nodes.get(arm, [])
        ]
        if not contributing_arms:
            return -np.inf

        original_data = self.data
        arm_lls = []
        arm_df_models = []
        arm_n_rows = []
        try:
            for arm in contributing_arms:
                arm_data = original_data.loc[self.arm_labels == arm]

                # Clamping a *parent* must never remove it as a regressor for
                # `variable`'s score in an arm that still contributes - assert
                # that explicitly rather than assuming the column survived.
                missing_parent_cols = [p for p in parents if p not in arm_data.columns]
                assert not missing_parent_cols, (
                    f"Parent column(s) {missing_parent_cols} missing from arm "
                    f"{arm!r}'s data - a clamped parent must remain usable as a "
                    "regressor in every arm where it isn't the variable being scored."
                )
                if arm_data.empty:
                    continue

                self.data = arm_data
                try:
                    ll, df_model = self._log_likelihood(variable=variable, parents=parents)
                except Exception:
                    return -np.inf
                finally:
                    self.data = original_data

                arm_lls.append(ll)
                arm_df_models.append(df_model)
                arm_n_rows.append(len(arm_data))
        finally:
            self.data = original_data

        if len(set(arm_df_models)) != 1:
            # A contributing arm can legitimately have too few (post-resampling)
            # rows to estimate the full parameter set - e.g. a bootstrap
            # resample of a 6-row arm can leave 1-2 rows, so statsmodels' GLM
            # fit there is rank-deficient and reports a SMALLER df_model than an
            # arm with enough rows for every parent. That's not a bug to raise
            # on (unlike the missing-parent-column case above, which can never
            # legitimately happen) - it's a degenerate fit, treated the same as
            # any other singular fit: exclude this candidate.
            return -np.inf
        df_model = arm_df_models[0]
        n_eff = sum(arm_n_rows)

        # Complexity penalty applied ONCE over n_eff (pooled rows across
        # contributing arms) - not per-arm, per the class's interventional-scoring
        # contract.
        bic_score = sum(arm_lls) - (((df_model + 2) / 2) * np.log(n_eff))

        # Soft prior component - identical to the observational branch.
        prior_bonus = 0
        for parent in parents:
            p = self.edge_priors[(parent, variable)]
            p = np.clip(p, 1e-6, 1 - 1e-6)  # Avoid log(0)
            log_odds = np.log(p / (1 - p))
            prior_bonus += log_odds

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


def _resample_with_arm_floor(
    combined: pd.DataFrame,
    arm_col: str,
    frac: float,
    replace: bool,
    floor: int,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Bootstrap-resample ``combined``, holding any arm smaller than ``floor`` rows fixed.

    An arm with fewer rows than ``floor`` carries too little information to usefully
    bootstrap: with-replacement resampling at a typical ``frac<1`` setting collapses it
    to just 1-2 unique rows a meaningful fraction of the time, which isn't enough to
    identify a multi-parent GLM fit and mostly produces candidates that get discarded as
    singular (see the HPN-DREAM interventional consensus run that motivated this - a
    6-row arm resampled at frac=0.65 from a 30-row pool got <=2 rows ~20% of the time).
    Those arms are kept in full, unresampled, on every draw; only arms at or above
    ``floor`` rows are bootstrap-resampled at (``frac``, ``replace``). ``floor=0``
    (the default everywhere this is called) reproduces the original single pooled
    ``.sample(...)`` call exactly, so this is opt-in only.
    """
    if not floor:
        return combined.sample(frac=frac, replace=replace, random_state=rng)

    parts = []
    for _, arm_group in combined.groupby(arm_col, sort=False):
        if len(arm_group) < floor:
            parts.append(arm_group)
        else:
            parts.append(arm_group.sample(frac=frac, replace=replace, random_state=rng))
    return pd.concat(parts)


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
    interventional: bool = False,
    arm_labels: Optional[pd.Series] = None,
    clamped_nodes: Optional[dict] = None,
    arm_resample_floor: int = 0,
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
    interventional : bool, optional
        Passed through to `score_fn` (only meaningful for a scoring class that
        supports it, e.g. `BICGaussIndraPriors`). Default is False, which never
        changes this function's resampling or scoring-construction code path -
        see Notes.
    arm_labels : Optional[pd.Series], optional
        Per-sample experimental-arm label aligned to `data`'s index. Required
        for `interventional` to take effect - default is None.
    clamped_nodes : Optional[dict], optional
        Passed through to `score_fn` unchanged when `interventional` is True.
    arm_resample_floor : int, optional
        Only meaningful when `arm_labels` is not None. Arms with fewer than this many
        rows are kept in full (unresampled) on every bootstrap draw rather than being
        bootstrap-resampled at (`subsample_frac`, `replace`) like the rest of the data
        - see `_resample_with_arm_floor`. Default is 0, which disables this and
        reproduces the original single pooled `.sample(...)` call exactly.

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

    With `arm_labels=None` (the default), `resampled_data`/`custom_score`
    construction are exactly what they were before `interventional` existed -
    the two branches below are never merged into one code path so that case
    stays byte-for-byte unchanged.
    """
    import logging

    # try:
    # Suppress INFO logs from pgmpy in this subprocess
    logging.getLogger("pgmpy").setLevel(logging.WARNING)

    rng = np.random.RandomState(seed)
    # subsample_frac<1 with replace=True -> a bootstrap resample (consensus mode);
    # subsample_frac=1 with replace=False -> the full data (best-of-restarts mode).
    if arm_labels is not None:
        # Resample data and arm_labels TOGETHER, in one .sample() call on a
        # combined frame, so a bootstrap resample can never desynchronize which
        # arm label goes with which resampled row. Two separate .sample() calls
        # sharing one RandomState would each advance its internal state and draw
        # DIFFERENT rows on the second call - not the "same" resample.
        combined = data.copy()
        combined["__arm_label__"] = arm_labels
        resampled_combined = _resample_with_arm_floor(
            combined, "__arm_label__", subsample_frac, replace, arm_resample_floor, rng
        )
        resampled_arm_labels = resampled_combined.pop("__arm_label__")
        resampled_data = resampled_combined
    else:
        resampled_data = data.sample(frac=subsample_frac, replace=replace, random_state=rng)
        resampled_arm_labels = None

    # Initialize the custom scoring function. interventional_kwargs stays empty
    # unless interventional=True was explicitly requested, so score_fn classes
    # that don't accept these kwargs at all (anything but BICGaussIndraPriors,
    # today) are unaffected by this parameter existing.
    interventional_kwargs = {}
    if interventional:
        interventional_kwargs = dict(
            interventional=True, arm_labels=resampled_arm_labels, clamped_nodes=clamped_nodes
        )
    custom_score = score_fn(
        resampled_data,
        edge_priors=edge_priors,
        prior_strength=prior_strength,
        **interventional_kwargs,
    )

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
    interventional: bool = False,
    arm_labels: Optional[pd.Series] = None,
    clamped_nodes: Optional[dict] = None,
    arm_resample_floor: int = 0,
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

    interventional, arm_labels, clamped_nodes, arm_resample_floor are forwarded to
    `process_bootstrap` (and from there to `score_fn`) unchanged - default is
    `interventional=False`, `arm_labels=None`, `arm_resample_floor=0`, which never
    alters this function's own behavior; only the values ultimately reaching
    `process_bootstrap` change.
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
            interventional=interventional,
            arm_labels=arm_labels,
            clamped_nodes=clamped_nodes,
            arm_resample_floor=arm_resample_floor,
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
