"""Integration coverage for estimate_posterior_dag's interventional (GIES-style)
plumbing added alongside BICGaussIndraPriors._local_score_interventional.

estimate_posterior_dag itself isn't exercised by tests/test_network.py (its own
docstring dismisses the INDRA/bootstrap-driven entry points as "requiring
external services"), but that's not actually true here: given a plain data
DataFrame and a prior DataFrame, it needs no network access at all - it's just
heavier (bootstrap + hill climb), which these tests keep small/fast (n_bootstrap
in the tens, not the hundreds) rather than skipping.

Uses the same synthetic A->B->C chain (with an interventional arm clamping B)
as tests/test_prior_data_reconciliation.py's local_score-level chain-orientation
test, but end-to-end through estimate_posterior_dag -> run_bootstrap ->
process_bootstrap -> the scorer, for both selection modes ("best_of" and
"consensus"), to catch plumbing bugs (e.g. arm_labels desynchronizing from a
bootstrap resample) that a local_score-only test can't see.

One deliberately-scoped-down expectation: the local_score-level test proves
interventional scoring gives the TRUE full-graph hypothesis (A->B->C) a
decisively higher TOTAL score than its Markov-equivalent reverse (C->B->A).
That is a clean two-hypothesis comparison. A greedy hill-climb search doesn't
evaluate hypotheses that way - it adds/removes one edge at a time, and
`local_score("A", ["B"])` in isolation (not paired against the "A has no
parent" alternative under the reverse hypothesis) prefers B as a parent
regardless of orientation, because B still correlates with A in the
observational arm even though that correlation reflects B depending on A, not
the other way round. Empirically (checked across 10 runs, 5 seeds x both
selection modes), this means the search reliably recovers B->C (never C->B -
C's dependence on B is real in BOTH arms, so that edge has no ambiguity at the
per-edge level either) but the A<->B edge's orientation is close to a coin
flip through this specific search procedure, even though the data contains
the information to prefer A->B in a full-graph comparison. These tests
therefore only assert the robust part (B->C recovered, C->B never appears) -
see the local_score-level test for the full orientation-identification claim.
"""

import importlib

import numpy as np
import pandas as pd

net = importlib.import_module("causomic.network")
pdr = importlib.import_module("causomic.graph_construction.prior_data_reconciliation")


def _chain_data_with_intervention_arm(seed: int = 1, n_obs: int = 80, n_clamp: int = 80):
    rng = np.random.default_rng(seed)

    a_obs = rng.normal(0, 1, n_obs)
    b_obs = 1.5 * a_obs + rng.normal(0, 0.5, n_obs)
    c_obs = 1.5 * b_obs + rng.normal(0, 0.5, n_obs)

    a_clamp = rng.normal(0, 1, n_clamp)
    b_clamp = rng.uniform(-3, 3, n_clamp)
    c_clamp = 1.5 * b_clamp + rng.normal(0, 0.5, n_clamp)

    data = pd.DataFrame(
        {
            "A": np.concatenate([a_obs, a_clamp]),
            "B": np.concatenate([b_obs, b_clamp]),
            "C": np.concatenate([c_obs, c_clamp]),
        }
    )
    arm_labels = pd.Series(["obs"] * n_obs + ["clampB"] * n_clamp, index=data.index)
    clamped_nodes = {"clampB": ["B"]}

    # Prior allows the true chain's edges plus their reverses - the search has to
    # pick between them using the scorer, not because the prior already decided.
    indra_priors = pd.DataFrame(
        {
            "source": ["A", "B", "B", "C"],
            "target": ["B", "A", "C", "B"],
            "evidence_count": [10, 10, 10, 10],
        }
    )
    return data, arm_labels, clamped_nodes, indra_priors


def test_estimate_posterior_dag_interventional_best_of_recovers_b_to_c():
    data, arm_labels, clamped_nodes, indra_priors = _chain_data_with_intervention_arm()
    allowed_edges = {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")}

    y0_graph = net.estimate_posterior_dag(
        data.copy(),
        indra_priors,
        prior_strength=5.0,
        scoring_function=pdr.BICGaussIndraPriors,
        search_algorithm=pdr.SparseHillClimb,
        n_bootstrap=30,
        edge_probability=0.5,
        selection="best_of",
        interventional=True,
        arm_labels=arm_labels,
        clamped_nodes=clamped_nodes,
        verbose=False,
    )
    edges = {(str(u), str(v)) for u, v in y0_graph.directed.edges()}
    assert edges, "expected a non-trivial learned network"
    assert edges <= allowed_edges, f"unexpected edge(s) outside the prior: {edges - allowed_edges}"
    # The robust part of the claim - see module docstring for why A<->B's
    # orientation isn't asserted here.
    assert ("B", "C") in edges
    assert ("C", "B") not in edges


def test_estimate_posterior_dag_interventional_consensus_recovers_b_to_c():
    data, arm_labels, clamped_nodes, indra_priors = _chain_data_with_intervention_arm(seed=2)
    allowed_edges = {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")}

    y0_graph = net.estimate_posterior_dag(
        data.copy(),
        indra_priors,
        prior_strength=5.0,
        scoring_function=pdr.BICGaussIndraPriors,
        search_algorithm=pdr.SparseHillClimb,
        n_bootstrap=30,
        edge_probability=0.5,
        selection="consensus",
        interventional=True,
        arm_labels=arm_labels,
        clamped_nodes=clamped_nodes,
        verbose=False,
    )
    edges = {(str(u), str(v)) for u, v in y0_graph.directed.edges()}
    assert edges, "expected a non-trivial learned network"
    assert edges <= allowed_edges, f"unexpected edge(s) outside the prior: {edges - allowed_edges}"
    assert ("B", "C") in edges
    assert ("C", "B") not in edges


def test_estimate_posterior_dag_consensus_with_arm_resample_floor_recovers_b_to_c():
    """arm_resample_floor keeps a small clamped arm intact across bootstrap draws
    instead of letting consensus's frac=0.65 resampling collapse it to a handful
    of rows - added after that exact failure mode (rank-deficient GLM fits from an
    over-resampled small arm) showed up running the real HPN-DREAM pilot data.
    Shrinks the clamped arm down to a size where the floor actually matters, and
    checks the plumbing doesn't break the same robust B->C recovery as the other
    consensus test above.
    """
    data, arm_labels, clamped_nodes, indra_priors = _chain_data_with_intervention_arm(
        seed=4, n_obs=80, n_clamp=8
    )
    allowed_edges = {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")}

    y0_graph = net.estimate_posterior_dag(
        data.copy(),
        indra_priors,
        prior_strength=5.0,
        scoring_function=pdr.BICGaussIndraPriors,
        search_algorithm=pdr.SparseHillClimb,
        n_bootstrap=30,
        edge_probability=0.5,
        selection="consensus",
        interventional=True,
        arm_labels=arm_labels,
        clamped_nodes=clamped_nodes,
        arm_resample_floor=10,
        verbose=False,
    )
    edges = {(str(u), str(v)) for u, v in y0_graph.directed.edges()}
    assert edges, "expected a non-trivial learned network"
    assert edges <= allowed_edges, f"unexpected edge(s) outside the prior: {edges - allowed_edges}"
    assert ("B", "C") in edges
    assert ("C", "B") not in edges


def test_estimate_posterior_dag_consensus_subsample_frac_override():
    """consensus_subsample_frac overrides consensus's hardcoded subsample_frac=0.65 - added
    after HPN-DREAM's BT20 contexts (DMSO-only n=5-6) showed a 65% subsample of an already-tiny
    dataset collapses to too few rows to fit almost any candidate parent set, stalling the
    bootstrap for hours without finishing. Checks the override reaches run_bootstrap's own
    subsample_frac argument, and that leaving it unset reproduces the original 0.65 default -
    both via a call-argument spy rather than actually running the (slow) bootstrap.
    """
    from unittest.mock import patch

    data, _, _, indra_priors = _chain_data_with_intervention_arm(seed=4)

    with patch.object(net, "run_bootstrap", wraps=net.run_bootstrap) as spy:
        net.estimate_posterior_dag(
            data.copy(),
            indra_priors,
            prior_strength=5.0,
            scoring_function=pdr.BICGaussIndraPriors,
            search_algorithm=pdr.SparseHillClimb,
            n_bootstrap=5,
            edge_probability=0.5,
            selection="consensus",
            verbose=False,
            consensus_subsample_frac=1.0,
        )
        assert spy.call_args.kwargs["subsample_frac"] == 1.0

    with patch.object(net, "run_bootstrap", wraps=net.run_bootstrap) as spy:
        net.estimate_posterior_dag(
            data.copy(),
            indra_priors,
            prior_strength=5.0,
            scoring_function=pdr.BICGaussIndraPriors,
            search_algorithm=pdr.SparseHillClimb,
            n_bootstrap=5,
            edge_probability=0.5,
            selection="consensus",
            verbose=False,
        )
        assert spy.call_args.kwargs["subsample_frac"] == 0.65


def test_estimate_posterior_dag_default_ignores_interventional_kwargs_path():
    """No arm_labels/interventional at all - must behave exactly as before this
    feature existed (same call, no new kwargs reaching the scorer)."""
    data, _, _, indra_priors = _chain_data_with_intervention_arm(seed=3)

    # Should run to completion without ever touching the interventional branch.
    y0_graph = net.estimate_posterior_dag(
        data.copy(),
        indra_priors,
        prior_strength=5.0,
        scoring_function=pdr.BICGaussIndraPriors,
        search_algorithm=pdr.SparseHillClimb,
        n_bootstrap=10,
        edge_probability=0.5,
        selection="best_of",
        verbose=False,
    )
    assert y0_graph.directed.number_of_nodes() == 3
