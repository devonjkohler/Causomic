"""Tests for the pure helper functions in prior_data_reconciliation.

Covers the deterministic / data-transform helpers only:

- random_acyclic_subgraph: acyclicity, node/edge invariants, inclusion_prob
  extremes, seeded determinism, and max_indegree enforcement.
- calculate_edge_probabilities: power-law CDF mapping over evidence counts.
- prepare_indra_priors: (source, target) -> probability dictionary building
  with and without the sigmoid probability conversion.
- remove_high_corr_edges_from_blacklist: blacklist pruning and prior augmentation
  driven by a small correlation setup.
- BICGaussIndraPriors.local_score: the observational (default) path is covered by
  a regression/snapshot test; the opt-in interventional (GIES-style) path is
  covered separately, including the textbook chain-orientation identifiability
  case observational scoring cannot resolve.
- _resample_with_arm_floor: the disabled (floor=0) path matches a plain pooled
  .sample() call exactly; the enabled path keeps any arm below the floor fully
  intact across many seeds while still resampling arms at/above it.

The Parallel/HillClimb bootstrap drivers (process_bootstrap / run_bootstrap) are
heavier and are not exercised here.
"""

import importlib

import networkx as nx
import numpy as np
import pandas as pd
import pytest

pdr = importlib.import_module("causomic.graph_construction.prior_data_reconciliation")


# ---------------------------------------------------------------------------
# random_acyclic_subgraph
# ---------------------------------------------------------------------------
def test_random_acyclic_subgraph_is_acyclic_and_subset():
    nodes = ["A", "B", "C", "D"]
    # Contains a cycle A->B->C->D->A; result must still be acyclic.
    allowed = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    rng = np.random.default_rng(0)
    dag = pdr.random_acyclic_subgraph(nodes, allowed, inclusion_prob=1.0, rng=rng)
    assert nx.is_directed_acyclic_graph(dag)
    assert set(dag.nodes()) == set(nodes)
    assert set(dag.edges()).issubset(set(allowed))


def test_random_acyclic_subgraph_zero_prob_adds_no_edges():
    nodes = ["A", "B", "C"]
    allowed = [("A", "B"), ("B", "C")]
    rng = np.random.default_rng(1)
    dag = pdr.random_acyclic_subgraph(nodes, allowed, inclusion_prob=0.0, rng=rng)
    assert dag.number_of_edges() == 0
    assert set(dag.nodes()) == set(nodes)


def test_random_acyclic_subgraph_deterministic_with_seed():
    nodes = ["A", "B", "C", "D"]
    allowed = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "C")]
    d1 = pdr.random_acyclic_subgraph(nodes, allowed, 0.6, np.random.default_rng(42))
    d2 = pdr.random_acyclic_subgraph(nodes, allowed, 0.6, np.random.default_rng(42))
    assert set(d1.edges()) == set(d2.edges())


def test_random_acyclic_subgraph_respects_max_indegree():
    # Every allowed edge points into node "T"; with inclusion_prob=1.0 all edges
    # would be attempted, but max_indegree caps the accepted parents.
    nodes = ["A", "B", "C", "T"]
    allowed = [("A", "T"), ("B", "T"), ("C", "T")]
    rng = np.random.default_rng(7)
    dag = pdr.random_acyclic_subgraph(nodes, allowed, inclusion_prob=1.0, rng=rng, max_indegree=1)
    assert len(dag.get_parents("T")) <= 1
    assert nx.is_directed_acyclic_graph(dag)


# ---------------------------------------------------------------------------
# calculate_edge_probabilities
# ---------------------------------------------------------------------------
def test_calculate_edge_probabilities_cdf_properties():
    df = pd.DataFrame(
        {
            "source": ["A", "B", "C", "D"],
            "target": ["W", "X", "Y", "Z"],
            "evidence_count": [1, 2, 5, 10],
        }
    )
    mapping = pdr.calculate_edge_probabilities(df)

    xmin = 1
    xmax = 10
    # Keys span the full integer support from xmin..xmax.
    assert set(mapping.keys()) == set(range(xmin, xmax + 1))

    values = [mapping[k] for k in range(xmin, xmax + 1)]
    # CDF: all in [0, 1], non-decreasing, terminating at ~1.0.
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))
    assert np.isclose(values[-1], 1.0)
    # Larger evidence counts receive larger cumulative probability.
    assert mapping[10] > mapping[1]


def test_calculate_edge_probabilities_custom_count_col():
    df = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["X", "Y"],
            "source_count": [3, 7],
        }
    )
    mapping = pdr.calculate_edge_probabilities(df, count_col="source_count")
    assert set(mapping.keys()) == set(range(3, 8))
    assert np.isclose(mapping[7], 1.0)


# ---------------------------------------------------------------------------
# prepare_indra_priors
# ---------------------------------------------------------------------------
def test_prepare_indra_priors_no_conversion_returns_raw_counts():
    df = pd.DataFrame(
        {
            "source": ["AKT1", "TP53", "MDM2"],
            "target": ["MDM2", "MDM2", "TP53"],
            "evidence_count": [15, 25, 8],
        }
    )
    priors = pdr.prepare_indra_priors(df, convert_to_probability=False)
    assert priors == {
        ("AKT1", "MDM2"): 15,
        ("TP53", "MDM2"): 25,
        ("MDM2", "TP53"): 8,
    }


def test_prepare_indra_priors_sigmoid_conversion():
    df = pd.DataFrame(
        {
            "source": ["AKT1", "TP53"],
            "target": ["MDM2", "MDM2"],
            "evidence_count": [15, 25],
        }
    )
    priors = pdr.prepare_indra_priors(df, convert_to_probability=True)

    # Keys are the (source, target) tuples.
    assert set(priors.keys()) == {("AKT1", "MDM2"), ("TP53", "MDM2")}

    # Values match the closed-form sigmoid of log1p(count).
    def expected(count):
        log_ev = np.log1p(count)
        return 1 / (1 + np.exp(-(log_ev - 1.1) / 0.552))

    assert np.isclose(priors[("AKT1", "MDM2")], expected(15))
    assert np.isclose(priors[("TP53", "MDM2")], expected(25))
    # All resulting probabilities lie in (0, 1) and grow with evidence.
    assert 0.0 < priors[("AKT1", "MDM2")] < priors[("TP53", "MDM2")] < 1.0


def test_prepare_indra_priors_use_source_counts():
    df = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["X", "Y"],
            "evidence_count": [1, 2],
            "source_count": [50, 60],
        }
    )
    priors = pdr.prepare_indra_priors(df, convert_to_probability=False, use_source_counts=True)
    assert priors == {("A", "X"): 50, ("B", "Y"): 60}


# ---------------------------------------------------------------------------
# remove_high_corr_edges_from_blacklist
# ---------------------------------------------------------------------------
def test_remove_high_corr_edges_from_blacklist():
    # A and B are perfectly correlated; C is anti-correlated with both.
    data = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],
            "C": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    indra_priors = pd.DataFrame({"source": ["A"], "target": ["C"], "evidence_count": [10]})
    black_list = {("A", "B"), ("B", "A"), ("A", "C")}

    updated_priors, updated_blacklist = pdr.remove_high_corr_edges_from_blacklist(
        data, indra_priors, black_list, corr_threshold=0.99, verbose=False
    )

    # A<->B are highly correlated so both directions are pulled from the blacklist.
    assert ("A", "B") not in updated_blacklist
    assert ("B", "A") not in updated_blacklist
    # A->C correlation is 1.0 in magnitude (perfect anti-correlation), so it is
    # also removed given the abs-correlation threshold.
    assert ("A", "C") not in updated_blacklist
    assert updated_blacklist == set()

    # The high-correlation edges not already present are appended to the priors.
    prior_edges = set(zip(updated_priors["source"], updated_priors["target"]))
    assert ("A", "B") in prior_edges
    assert ("B", "A") in prior_edges
    # Pre-existing (A, C) prior row is retained, not duplicated.
    assert sum((updated_priors["source"] == "A") & (updated_priors["target"] == "C")) == 1


def test_remove_high_corr_edges_from_blacklist_keeps_low_corr():
    # Two independent-ish columns whose |corr| stays below the threshold.
    data = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    indra_priors = pd.DataFrame({"source": ["A"], "target": ["B"], "evidence_count": [3]})
    black_list = {("A", "B")}

    _, updated_blacklist = pdr.remove_high_corr_edges_from_blacklist(
        data, indra_priors, black_list, corr_threshold=0.9, verbose=False
    )
    # Correlation below threshold: blacklist edge is preserved.
    assert ("A", "B") in updated_blacklist


# ---------------------------------------------------------------------------
# BICGaussIndraPriors.local_score - interventional (GIES-style) scoring
# ---------------------------------------------------------------------------
def test_bic_gauss_indra_priors_observational_regression():
    """interventional=False (the default) must reproduce local_score's behavior
    from before the interventional branch existed, exactly. Also covers the
    documented fallback: interventional=True with no arm_labels must give the
    identical value, since arm_labels is what actually gates the new branch.
    """
    data = pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 1.5],
            "Y": [2.1, 3.9, 6.2, 7.8, 10.1, 5.0, 7.1, 3.0],
            "Z": [0.5, 1.5, 1.0, 2.5, 3.0, 1.2, 2.0, 0.8],
        }
    )
    edge_priors = {("X", "Y"): 0.8, ("Z", "Y"): 0.3}

    scorer_default = pdr.BICGaussIndraPriors(data, edge_priors=edge_priors, prior_strength=2.0)
    score_default = scorer_default.local_score("Y", ["X", "Z"])

    # Snapshot value - computed once from this exact implementation. A future
    # change that silently alters the observational code path should break this.
    assert score_default == pytest.approx(4.747972022632462)

    scorer_flag_only = pdr.BICGaussIndraPriors(
        data, edge_priors=edge_priors, prior_strength=2.0, interventional=True
    )
    score_flag_only = scorer_flag_only.local_score("Y", ["X", "Z"])
    assert score_flag_only == score_default


def test_bic_gauss_indra_priors_clamped_parent_still_usable_as_regressor():
    """A node clamped in an arm is only dropped from ITS OWN local-score
    contribution in that arm - it must remain a real, informative regressor for
    other nodes' scores there. Isolate this with a single all-clamped arm: if
    B's clamped values were (incorrectly) ignored as a C regressor, C|[B] would
    score no better than C|[] despite the strong true B->C relationship built
    into the data.
    """
    rng = np.random.default_rng(3)
    n = 100
    b_vals = rng.uniform(-3, 3, n)
    c_vals = 2.0 * b_vals + rng.normal(0, 0.1, n)
    data = pd.DataFrame({"B": b_vals, "C": c_vals})
    arm_labels = pd.Series(["clampB"] * n, index=data.index)
    clamped_nodes = {"clampB": ["B"]}

    scorer = pdr.BICGaussIndraPriors(
        data,
        edge_priors={("B", "C"): 0.5},
        interventional=True,
        arm_labels=arm_labels,
        clamped_nodes=clamped_nodes,
    )
    score_with_b = scorer.local_score("C", ["B"])
    score_without_b = scorer.local_score("C", [])
    assert score_with_b > score_without_b + 10  # clear margin, not a coin-flip tie


def test_bic_gauss_indra_priors_variable_clamped_in_every_arm_returns_neg_inf():
    data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 4.0, 6.0]})
    arm_labels = pd.Series(["only_arm"] * 3, index=data.index)
    clamped_nodes = {"only_arm": ["A"]}

    scorer = pdr.BICGaussIndraPriors(
        data,
        edge_priors={("B", "A"): 0.5},
        interventional=True,
        arm_labels=arm_labels,
        clamped_nodes=clamped_nodes,
    )
    # A is clamped in the only arm that exists, so there is no data anywhere in
    # which A's value reflects a response to its hypothesized parent B.
    assert scorer.local_score("A", ["B"]) == -np.inf


def test_bic_gauss_indra_priors_mismatched_df_model_across_arms_returns_neg_inf():
    """Regression test for a real failure hit running this against the HPN-DREAM
    pilot's actual arms (6 rows before any resampling): a bootstrap resample of
    a small arm can leave too few rows to estimate the full parent set, so
    statsmodels' GLM fit there is rank-deficient and reports a SMALLER df_model
    than a well-powered arm - previously a hard `assert` on this raised
    AssertionError, which propagates through joblib and aborts an entire
    Parallel(...) bootstrap run for every worker, not just the offending one.
    Must return -inf (a degenerate-fit signal, like any other singular fit)
    instead of raising.
    """
    rng = np.random.default_rng(0)
    n_big, n_tiny = 20, 2
    data = pd.DataFrame(
        {
            "P1": np.concatenate([rng.normal(0, 1, n_big), rng.normal(0, 1, n_tiny)]),
            "P2": np.concatenate([rng.normal(0, 1, n_big), rng.normal(0, 1, n_tiny)]),
            "Y": np.concatenate([rng.normal(0, 1, n_big), rng.normal(0, 1, n_tiny)]),
        }
    )
    arm_labels = pd.Series(["big"] * n_big + ["tiny"] * n_tiny, index=data.index)
    edge_priors = {("P1", "Y"): 0.5, ("P2", "Y"): 0.5}

    scorer = pdr.BICGaussIndraPriors(
        data, edge_priors=edge_priors, interventional=True, arm_labels=arm_labels, clamped_nodes={}
    )
    # Must not raise - the 2-row "tiny" arm can't identify both P1 and P2's
    # coefficients (2 predictors + intercept from 2 points), so its df_model
    # differs from the well-powered "big" arm's.
    assert scorer.local_score("Y", ["P1", "P2"]) == -np.inf


def test_bic_gauss_indra_priors_interventional_identifies_chain_orientation():
    """The textbook Markov-equivalence case: a chain A->B->C and its reverse
    C->B->A share the same skeleton and no v-structures, so they encode the same
    joint distribution and get IDENTICAL total observational BIC scores - no
    amount of purely observational data can prefer one over the other.

    Clamping the middle node B breaks the tie: in the arm where B is
    experimenter-set, the true model's B->C edge still shows up (C responds to
    the clamped B value) but the reverse model's B->A edge does not (A was
    generated independently of the clamped B value) - because A is never
    clamped, its own local score still incorporates that arm's rows, so the
    "A depends on B" hypothesis is penalized by data that doesn't support it,
    which the true "C depends on B" hypothesis is not.
    """
    rng = np.random.default_rng(0)
    n_obs, n_clamp = 150, 150

    # Observational arm: true generative chain A -> B -> C.
    a_obs = rng.normal(0, 1, n_obs)
    b_obs = 1.5 * a_obs + rng.normal(0, 0.5, n_obs)
    c_obs = 1.5 * b_obs + rng.normal(0, 0.5, n_obs)

    # Interventional arm: B clamped to experimenter-chosen values, independent
    # of A; the real B -> C mechanism still operates on whatever value B takes.
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
    edge_priors = {(p, v): 0.5 for p in ["A", "B", "C"] for v in ["A", "B", "C"] if p != v}

    def total_score(scorer, edges_by_node):
        return sum(scorer.local_score(v, parents) for v, parents in edges_by_node.items())

    true_chain = {"A": [], "B": ["A"], "C": ["B"]}
    reverse_chain = {"C": [], "B": ["C"], "A": ["B"]}

    # Observational-only: the two Markov-equivalent hypotheses tie exactly.
    scorer_obs = pdr.BICGaussIndraPriors(data.loc[arm_labels == "obs"], edge_priors=edge_priors)
    score_true_obs = total_score(scorer_obs, true_chain)
    score_reverse_obs = total_score(scorer_obs, reverse_chain)
    assert score_true_obs == pytest.approx(score_reverse_obs, abs=1e-6)

    # Interventional: clamping B breaks the tie decisively in favor of the truth.
    scorer_int = pdr.BICGaussIndraPriors(
        data,
        edge_priors=edge_priors,
        interventional=True,
        arm_labels=arm_labels,
        clamped_nodes=clamped_nodes,
    )
    score_true_int = total_score(scorer_int, true_chain)
    score_reverse_int = total_score(scorer_int, reverse_chain)
    assert score_true_int > score_reverse_int + 100  # decisive, not marginal


# ---------------------------------------------------------------------------
# _resample_with_arm_floor
# ---------------------------------------------------------------------------


def _combined_big_small(n_big=20, n_small=3):
    combined = pd.DataFrame({"x": np.arange(n_big + n_small, dtype=float)})
    combined["__arm_label__"] = ["big"] * n_big + ["small"] * n_small
    return combined


def test_resample_with_arm_floor_disabled_matches_plain_sample():
    combined = _combined_big_small()
    got = pdr._resample_with_arm_floor(
        combined, "__arm_label__", frac=0.65, replace=True, floor=0, rng=np.random.RandomState(0)
    )
    want = combined.sample(frac=0.65, replace=True, random_state=np.random.RandomState(0))
    pd.testing.assert_frame_equal(got, want)


def test_resample_with_arm_floor_keeps_small_arm_intact_across_seeds():
    combined = _combined_big_small(n_big=20, n_small=3)
    small_rows = combined[combined["__arm_label__"] == "small"]

    big_arm_varied = False
    for seed in range(20):
        resampled = pdr._resample_with_arm_floor(
            combined,
            "__arm_label__",
            frac=0.65,
            replace=True,
            floor=10,
            rng=np.random.RandomState(seed),
        )
        resampled_small = resampled[resampled["__arm_label__"] == "small"]
        # The small (3-row) arm is below the floor=10 threshold: every one of its
        # rows must appear, unresampled, exactly once, every single time.
        pd.testing.assert_frame_equal(resampled_small.sort_index(), small_rows.sort_index())
        resampled_big = resampled[resampled["__arm_label__"] == "big"]
        # The big (20-row) arm is at/above the floor: it IS bootstrap-resampled,
        # so it should vary across seeds (and not always be all 20 original rows).
        if len(resampled_big) != 20 or not resampled_big["x"].is_monotonic_increasing:
            big_arm_varied = True
    assert big_arm_varied, "expected the big arm's resample to vary across seeds"


def test_resample_with_arm_floor_all_arms_above_floor_resamples_everything():
    combined = _combined_big_small(n_big=20, n_small=12)
    resampled = pdr._resample_with_arm_floor(
        combined,
        "__arm_label__",
        frac=0.5,
        replace=True,
        floor=10,
        rng=np.random.RandomState(0),
    )
    # Both arms are >= floor=10, so both get bootstrap-resampled to round(frac*len).
    assert len(resampled[resampled["__arm_label__"] == "big"]) == round(0.5 * 20)
    assert len(resampled[resampled["__arm_label__"] == "small"]) == round(0.5 * 12)
