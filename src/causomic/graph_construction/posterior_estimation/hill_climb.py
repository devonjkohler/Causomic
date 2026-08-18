"""Hill-climb structure search restricted to a prior edge set.

:class:`SparseHillClimb` is pgmpy's ``HillClimbSearch`` with one change that
matters at biological scale: candidate edge additions are drawn from an explicit
``allowed_additions`` set (the INDRA prior network) rather than from all
``n * (n - 1)`` ordered node pairs. On a few hundred proteins that turns each
iteration from quadratic in node count into linear in prior-edge count.

It also fixes a cycle check in the FLIP operation -- see ``_legal_operations``.
"""

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

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge, HillClimbSearch
from pgmpy.estimators.StructureScore import get_scoring_method
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
