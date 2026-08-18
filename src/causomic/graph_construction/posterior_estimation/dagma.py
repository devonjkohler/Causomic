"""DAGMA continuous-optimization structure learning with INDRA priors.

An alternative to the discrete hill climb in
:mod:`~causomic.graph_construction.posterior_estimation.hill_climb`: DAGMA
(Bello et al., 2022) recasts structure learning as a differentiable
acyclicity-constrained optimization solved in one shot over the full data,
rather than a sequence of greedy local moves.

INDRA priors enter at two strengths:

- **Hard restriction** (always): every node pair outside the prior network is
  blacklisted via ``exclude_edges``, mirroring ``SparseHillClimb``'s
  ``allowed_additions``.
- **Soft weighting** (``use_evidence_weights=True``): each allowed edge's L1
  penalty is scaled by its evidence log-odds, so well-supported edges are
  cheaper to keep. See :func:`evidence_penalty`.

Because that weighting shrinks every coefficient by a different amount,
reported edge weights are refit by unpenalized OLS on the selected support --
see :func:`_refit_unpenalized_ols`.

``dagma`` is an optional dependency; see :mod:`causomic._optional`.
"""

from typing import Any, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

try:
    import scipy.linalg as sla
    from dagma.linear import DagmaLinear
    from scipy.special import expit as sigmoid
except ImportError:  # optional dependency, see causomic._optional
    from causomic._optional import MissingDagmaLinear as DagmaLinear

from causomic.graph_construction.posterior_estimation.edge_priors import (
    prepare_indra_priors,
)


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
