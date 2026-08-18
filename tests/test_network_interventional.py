"""Integration coverage for estimate_posterior_dag's interventional (GIES-style)
plumbing alongside BICGaussIndraPriors._local_score_interventional.

estimate_posterior_dag itself isn't exercised by tests/test_network.py (its own
docstring dismisses the INDRA/bootstrap-driven entry points as "requiring
external services"), but that's not actually true here: given a plain data
DataFrame and a prior DataFrame, it needs no network access at all - it's just
heavier (bootstrap + hill climb), which these tests keep small/fast (n_bootstrap
in the tens, not the hundreds) rather than skipping.

Uses the same synthetic A->B->C chain (with an interventional arm clamping B)
as tests/test_posterior_estimation.py's local_score-level chain-orientation
test, but end-to-end through estimate_posterior_dag -> run_bootstrap ->
process_bootstrap -> the scorer, for both selection modes ("best_of" and
"consensus"), to catch plumbing bugs (e.g. arm_labels desynchronizing from a
bootstrap resample) that a local_score-only test can't see.

These tests assert recovery of the whole true chain (A->B and B->C present, and
neither reverse edge), which the pooled interventional score achieves through the
greedy search: measured across 13 runs (5 seeds x both selection modes, plus the
small-clamped-arm/arm_resample_floor config at 3 seeds), every run returned
exactly {A->B, B->C}.

Worth recording why that is a stronger claim than it used to be. The interventional
score originally summed a separate per-arm GLM fit, which gave each arm its own
free intercept and so discarded the between-arm mean shifts that carry the
orientation signal; under that scorer these tests could only assert the B->C half.
B->C was never ambiguous per-edge (C's dependence on B is real in BOTH arms), but
A<->B's orientation was close to a coin flip, because a greedy search evaluates
`local_score("A", ["B"])` in isolation rather than pairing it against the "A has no
parent" alternative under the reverse hypothesis, and B still correlates with A in
the observational arm regardless of direction. Pooling to a single intercept made
the per-edge scores themselves orientation-aware, so the search now settles on the
true orientation without needing a full-graph comparison to break the tie. If a
future scoring change makes these A->B assertions flaky, that is a real regression
in orientation power, not a too-strict test.
"""

import importlib

import numpy as np
import pandas as pd

net = importlib.import_module("causomic.network")
pe = importlib.import_module("causomic.graph_construction.posterior_estimation")


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


def test_estimate_posterior_dag_interventional_best_of_recovers_the_chain():
    data, arm_labels, clamped_nodes, indra_priors = _chain_data_with_intervention_arm()
    allowed_edges = {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")}

    y0_graph = net.estimate_posterior_dag(
        data.copy(),
        indra_priors,
        prior_strength=5.0,
        scoring_function=pe.BICGaussIndraPriors,
        search_algorithm=pe.SparseHillClimb,
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
    # The whole true chain, both orientations - see module docstring.
    assert edges == {("A", "B"), ("B", "C")}


def test_estimate_posterior_dag_interventional_consensus_recovers_the_chain():
    data, arm_labels, clamped_nodes, indra_priors = _chain_data_with_intervention_arm(seed=2)
    allowed_edges = {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")}

    y0_graph = net.estimate_posterior_dag(
        data.copy(),
        indra_priors,
        prior_strength=5.0,
        scoring_function=pe.BICGaussIndraPriors,
        search_algorithm=pe.SparseHillClimb,
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
    assert edges == {("A", "B"), ("B", "C")}


def test_estimate_posterior_dag_consensus_with_arm_resample_floor_recovers_the_chain():
    """arm_resample_floor keeps a small clamped arm intact across bootstrap draws
    instead of letting consensus's frac=0.65 resampling collapse it to a handful
    of rows - added after that exact failure mode (rank-deficient GLM fits from an
    over-resampled small arm) showed up running the real HPN-DREAM pilot data.
    Shrinks the clamped arm down to a size where the floor actually matters, and
    checks the plumbing doesn't break the same chain recovery as the other
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
        scoring_function=pe.BICGaussIndraPriors,
        search_algorithm=pe.SparseHillClimb,
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
    assert edges == {("A", "B"), ("B", "C")}


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
            scoring_function=pe.BICGaussIndraPriors,
            search_algorithm=pe.SparseHillClimb,
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
            scoring_function=pe.BICGaussIndraPriors,
            search_algorithm=pe.SparseHillClimb,
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
        scoring_function=pe.BICGaussIndraPriors,
        search_algorithm=pe.SparseHillClimb,
        n_bootstrap=10,
        edge_probability=0.5,
        selection="best_of",
        verbose=False,
    )
    assert y0_graph.directed.number_of_nodes() == 3
