"""Structure scores that blend data fit with INDRA prior belief.

Four scores, along two axes -- AIC vs BIC complexity penalty, and whether INDRA
edge priors contribute:

===========================  =========  ===================================
Class                        Penalty    Prior term
===========================  =========  ===================================
``AICGaussNoPriors``         AIC        none
``AICGaussIndraPriors``      AIC        log-odds of the edge prior
``BICGaussNoPriors``         BIC        none
``BICGaussIndraPriors``      BIC        log-odds of the edge prior
===========================  =========  ===================================

The ``*NoPriors`` pair is the like-for-like baseline for measuring what the
prior actually buys you: same search, same penalty, prior term removed.

Both BIC variants also support interventional data through
:class:`_PooledInterventionalScoreMixin`, which pools observational and
interventional arms into one likelihood while dropping rows where the child
variable was itself clamped (an intervened-on variable carries no information
about its parents).

All four are pgmpy ``StructureScore`` subclasses, so they plug directly into
:class:`~causomic.graph_construction.posterior_estimation.hill_climb.SparseHillClimb`
or any pgmpy search.
"""

from typing import Optional

import numpy as np
import pandas as pd
from pgmpy.estimators import LogLikelihoodGauss


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


class _PooledInterventionalScoreMixin:
    """
    Pooled (Hauser-Buhlmann style) GIES interventional scoring, shared by the
    BIC scorers that support it (`BICGaussIndraPriors`, `BICGaussNoPriors`).

    Provides the arm/clamp bookkeeping (`_init_interventional`), the
    branch-active predicate (`_interventional_active`) and the interventional BIC
    itself (`_pooled_interventional_bic`). That BIC carries NO prior term, so each
    scorer's own `_local_score_interventional` adds whatever prior component it
    has (none, for `BICGaussNoPriors`) on top - keeping the row-selection and
    penalty logic in exactly one place.

    Mix in ahead of `LogLikelihoodGauss`, and call `_init_interventional` from the
    subclass's `__init__` after `super().__init__(data, ...)`.
    """

    def _init_interventional(
        self,
        data: pd.DataFrame,
        interventional: bool,
        arm_labels: Optional[pd.Series],
        clamped_nodes: Optional[dict],
    ) -> None:
        """Store the interventional configuration, validating `arm_labels`."""
        self.interventional = interventional
        if arm_labels is not None:
            assert arm_labels.index.equals(
                data.index
            ), "arm_labels must share data's index (one label per row of data)."
        self.arm_labels = arm_labels
        self.clamped_nodes = clamped_nodes or {}

    @property
    def _interventional_active(self) -> bool:
        """True when `local_score` should take the interventional branch.

        `arm_labels` is what actually gates it: `interventional=True` alone, with
        no labels, stays on the flat observational path.
        """
        return self.interventional and self.arm_labels is not None

    def _pooled_interventional_bic(self, variable: str, parents: list) -> float:
        """
        Pooled (Hauser-Buhlmann style) GIES interventional local score: fit ONE GLM
        of `variable | parents` over every row whose arm does not clamp `variable`,
        then apply the BIC complexity penalty once over that retained row count.

        Returns the BIC only, with no prior component; a scorer with priors adds
        that in its own `_local_score_interventional`.

        The defining property is that a single intercept is estimated across all
        retained rows, rather than one per experimental arm. `_log_likelihood`
        regresses `variable ~ parents` *with a constant*, so fitting once per arm
        would hand each arm its own free intercept, and that intercept absorbs the
        arm's mean shift in `variable`. Those between-arm mean shifts are precisely
        the interventional signal used to orient edges, so a per-arm fit discards
        the orientation information and scores only within-arm covariance - which
        is symmetric in the two nodes of an edge and therefore says nothing about
        direction. Pooling to one intercept means an unexplained between-arm shift
        in `variable` must be accounted for by `variable`'s parents or it lands in
        the residual, which is what makes a wrong orientation score worse than the
        right one.

        Pooling matters twice more for designs with many small arms (e.g.
        Perturb-seq pseudobulk, ~10^3 arms of 2-5 rows): the complexity penalty is
        charged once over the parameters the single fit actually consumes, instead
        of undercounting by a factor of K arms and making an extra parent nearly
        free; and there is no cross-arm `df_model` agreement to satisfy, so a small
        rank-deficient arm can't discard an otherwise-scorable candidate.

        With no clamped nodes this reduces exactly to the observational branch's
        single flat GLM over all of `self.data` - the arm partitioning alone never
        changes a score. Rows are dropped only where `variable` itself is clamped:
        a clamped node's value there is experimenter-set, not generated by its
        parents, so those rows carry no information about *this* node's local
        mechanism. Clamping a *parent* drops no rows and removes no regressor - a
        clamped parent's fixed value is an ordinary regressor for another node's
        score - which is asserted explicitly below rather than left implicit.

        `_log_likelihood` (inherited from pgmpy's LogLikelihoodGauss) always reads
        `self.data`, with no data-argument override, so the pooled fit is computed
        by temporarily pointing `self.data` at the retained row-subset and
        restoring it in a `finally` block - reusing the same GLM-fitting code the
        observational branch uses rather than duplicating it.

        Returns -inf if too few rows are retained to identify the model
        (`n_used <= len(parents) + 2`, which also covers `variable` being clamped
        in every arm) or if the pooled fit is singular - matching the
        observational branch's all-or-nothing -inf handling.
        """
        clamped_arms = [
            arm for arm in pd.unique(self.arm_labels) if variable in self.clamped_nodes.get(arm, [])
        ]
        mask = (~self.arm_labels.isin(clamped_arms)).values

        n_used = int(mask.sum())
        if n_used <= len(parents) + 2:
            return -np.inf

        original_data = self.data
        pooled_data = original_data.loc[mask]

        # Clamping a *parent* must never remove it as a regressor for `variable`'s
        # score - assert that explicitly rather than assuming the column survived.
        missing_parent_cols = [p for p in parents if p not in pooled_data.columns]
        assert not missing_parent_cols, (
            f"Parent column(s) {missing_parent_cols} missing from the pooled "
            "row-subset - a clamped parent must remain usable as a regressor when "
            "it isn't the variable being scored."
        )

        try:
            self.data = pooled_data
            ll, df_model = self._log_likelihood(variable=variable, parents=parents)
        except Exception:
            # statsmodels raises if X is singular - same treatment as elsewhere.
            return -np.inf
        finally:
            self.data = original_data

        # Complexity penalty applied ONCE over the retained row count, for the one
        # pooled fit's parameters - so it does not scale with the number of arms.
        return ll - (((df_model + 2) / 2) * np.log(n_used))


class BICGaussIndraPriors(_PooledInterventionalScoreMixin, LogLikelihoodGauss):
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
    `clamped_nodes`) switches `local_score` to a second code path: a pooled
    (Hauser-Buhlmann style) GIES score that fits one GLM - and therefore estimates
    one intercept - over every row where the scored variable is not clamped,
    dropping only the rows whose arm clamps that variable. See `local_score` /
    `_local_score_interventional`.

    This is strictly opt-in: with the default `interventional=False`, or with
    `interventional=True` but no `arm_labels`, `local_score` runs the exact same
    single-GLM-over-self.data code path as before this feature existed. Existing
    callers (all of which construct this class without either argument) are
    unaffected byte-for-byte. Because the interventional score pools rather than
    fitting per arm, it also reduces exactly to that observational path whenever
    no node is clamped - the arm partitioning by itself never changes a score.
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
            If True *and* `arm_labels` is given, `local_score` uses the pooled
            GIES interventional score - one GLM over every row where `variable`
            isn't clamped, dropping the rows whose arm clamps it - instead of
            fitting one GLM over all of `data`. Default False, and the fallback
            when `arm_labels` is missing, reproduces prior behavior exactly - see
            class docstring.
        arm_labels : Optional[pd.Series], default=None
            Per-sample experimental-arm label, one entry per row of `data`.
            Must share `data`'s index. Required for the interventional branch to
            activate at all; if None, `local_score` always uses the flat
            observational path regardless of `interventional`.
        clamped_nodes : Optional[dict], default=None
            Maps an arm label (as found in `arm_labels`) to the list of node
            names pharmacologically clamped in that arm. An arm absent from this
            dict (or the dict being None/empty) is treated as having no clamped
            nodes. A clamped node's rows are only excluded from *its own* local
            score, in the arm(s) where it's clamped; it remains a valid regressor
            for every other variable's score in those same rows - that's the whole
            point of interventional data (see `_local_score_interventional`).
        """
        super().__init__(data, **kwargs)
        self.edge_priors = edge_priors or {}
        self.prior_strength = prior_strength  # This is lambda
        self.scale_with_n = scale_with_n
        self._init_interventional(data, interventional, arm_labels, clamped_nodes)

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
        if self._interventional_active:
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
        The shared pooled interventional BIC (see
        `_PooledInterventionalScoreMixin._pooled_interventional_bic` for what it
        computes and why it pools), plus this class's soft prior component. The
        shared method deliberately returns no prior term so `BICGaussNoPriors` can
        reuse it; the edge log-odds are added here, so every score this class
        returns includes them on the interventional branch exactly as on the
        observational one.

        The prior term is the same sum of edge log-odds the observational branch
        adds, so switching a scorer between the two branches changes only the
        likelihood/penalty part.
        """
        bic_score = self._pooled_interventional_bic(variable, parents)
        if not np.isfinite(bic_score):
            # Degenerate fit - the prior can't rescue a candidate with no
            # identifiable likelihood, and -inf + finite is -inf anyway.
            return bic_score

        # Soft prior component - identical to the observational branch.
        prior_bonus = 0.0
        for parent in parents:
            p = self.edge_priors[(parent, variable)]
            p = np.clip(p, 1e-6, 1 - 1e-6)  # Avoid log(0)
            log_odds = np.log(p / (1 - p))
            prior_bonus += log_odds

        return bic_score + prior_bonus


class BICGaussNoPriors(_PooledInterventionalScoreMixin, LogLikelihoodGauss):
    """
    Plain BIC scoring - the same criterion as `BICGaussIndraPriors` without the
    edge log-odds prior bonus. `edge_priors`/`prior_strength` are accepted (so the
    bootstrap drivers can construct any scorer uniformly) but never used, which
    makes this the natural no-prior baseline for isolating how much of a learned
    structure comes from the data rather than from INDRA.

    Interventional (GIES-style) scoring
    ------------------------------------
    Supports the same opt-in pooled interventional score as
    `BICGaussIndraPriors`: pass `interventional=True` together with `arm_labels`
    (and usually `clamped_nodes`) and `local_score` fits one GLM - one intercept -
    over every row whose arm does not clamp the scored variable, instead of a flat
    GLM over all of `data`. Since there is no prior term, that pooled BIC is the
    entire score. See `_PooledInterventionalScoreMixin` for the mechanics and the
    reasons pooling (rather than fitting per arm) is what makes the score
    orientation-aware. With the default `interventional=False`, or with
    `interventional=True` but no `arm_labels`, this class behaves exactly as it did
    before the branch existed.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        edge_priors: Optional[dict] = None,
        prior_strength: float = 1.0,
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
            Accepted for interface compatibility with the prior-aware scorers;
            ignored, since this class adds no prior bonus.
        prior_strength : float, default=1.0
            Accepted for interface compatibility; ignored.
        interventional : bool, default=False
            If True *and* `arm_labels` is given, `local_score` uses the pooled GIES
            interventional score instead of a flat GLM over all of `data`. Default
            False, and the fallback when `arm_labels` is missing, reproduces prior
            behavior exactly.
        arm_labels : Optional[pd.Series], default=None
            Per-sample experimental-arm label, one entry per row of `data`. Must
            share `data`'s index. Required for the interventional branch to
            activate at all.
        clamped_nodes : Optional[dict], default=None
            Maps an arm label (as found in `arm_labels`) to the list of node names
            pharmacologically clamped in that arm. An arm absent from this dict is
            treated as having no clamped nodes. A clamped node's rows are excluded
            only from *its own* local score; it remains a valid regressor for every
            other variable's score in those same rows.
        """
        super().__init__(data, **kwargs)
        self.edge_priors = edge_priors or {}
        self.prior_strength = prior_strength  # This is lambda
        self._init_interventional(data, interventional, arm_labels, clamped_nodes)

    def local_score(self, variable: str, parents: list) -> float:
        """
        Plain BIC: log-likelihood - ((df + 2)/2) * log(n), no prior term.

        Delegates to `_local_score_interventional` when `self.interventional` and
        `self.arm_labels` are both set; otherwise everything below is unchanged
        from before that branch existed. Returns -inf for degenerate fits
        (singular design matrix).
        """
        if self._interventional_active:
            return self._local_score_interventional(variable, parents)

        try:
            ll, df_model = self._log_likelihood(variable=variable, parents=parents)
        except:
            # statsmodels will raise ValueError if X is singular
            return -np.inf

        # Standard BIC score
        bic_score = ll - (((df_model + 2) / 2) * np.log(self.data.shape[0]))

        return bic_score

    def _local_score_interventional(self, variable: str, parents: list) -> float:
        """
        The shared pooled interventional BIC, which with no prior component to add
        is the whole score - see
        `_PooledInterventionalScoreMixin._pooled_interventional_bic`.
        """
        return self._pooled_interventional_bic(variable, parents)
