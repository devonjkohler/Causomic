"""Coverage for BICGaussIndraPriors._local_score_interventional_pooled, the opt-in
(`pooled_interventional=True`) pooled interventional local score.

The point of the pooled path is that it fits ONE GLM - and therefore estimates ONE
intercept - over every row where the scored variable isn't clamped, instead of one
GLM per experimental arm. Per-arm fits hand each arm its own free intercept, which
absorbs that arm's mean shift in the scored variable; those between-arm mean shifts
are exactly the interventional signal that orients an edge, so the per-arm path
scores only within-arm covariance, which is symmetric in the two nodes of an edge
and says nothing about direction.

These tests use a synthetic two-node SCM with a known direction (A -> B) and three
arms, and check that the pooled path recovers the direction, that the per-arm path
does not, that the pooled complexity penalty doesn't scale with the arm count, and
that clamping is applied to the scored variable only (never to a parent).
"""

import importlib
import logging

import numpy as np
import pandas as pd
import pytest

pdr = importlib.import_module("causomic.graph_construction.prior_data_reconciliation")

logging.getLogger("pgmpy").setLevel(logging.ERROR)

# 0.5 both ways: log(p/(1-p)) == 0, so the prior contributes exactly nothing and
# orientation is decided purely by the likelihood/penalty terms.
EDGE_PRIORS = {("A", "B"): 0.5, ("B", "A"): 0.5}

CLAMPED_NODES = {"do_A": ["A"], "do_B": ["B"]}


def _two_node_scm(seed: int = 0, n_per_arm: int = 30):
    """Linear SCM with ground truth A -> B: A = eps_A, B = 1.5*A + eps_B.

    Three arms of `n_per_arm` rows each:
      "obs"  - no clamp, no shift.
      "do_A" - A shifted by +4; B follows through the mechanism (mean ~ +6).
      "do_B" - B shifted by +4; A unaffected.
    """
    rng = np.random.default_rng(seed)

    frames, labels = [], []
    for arm in ("obs", "do_A", "do_B"):
        a = rng.normal(0, 1, n_per_arm) + (4.0 if arm == "do_A" else 0.0)
        b = 1.5 * a + rng.normal(0, 0.5, n_per_arm)
        if arm == "do_B":
            b = b + 4.0
        frames.append(pd.DataFrame({"A": a, "B": b}))
        labels += [arm] * n_per_arm

    data = pd.concat(frames, ignore_index=True)
    # arm_labels must share data's index - the scorer asserts this.
    arm_labels = pd.Series(labels, index=data.index)
    return data, arm_labels, CLAMPED_NODES


def _scorer(data, arm_labels, clamped_nodes, pooled: bool):
    return pdr.BICGaussIndraPriors(
        data,
        edge_priors=EDGE_PRIORS,
        interventional=True,
        arm_labels=arm_labels,
        clamped_nodes=clamped_nodes,
        pooled_interventional=pooled,
    )


def _orientation_margin(scorer) -> float:
    """Total score of the true graph (A->B) minus that of its reverse (B->A).

    Both hypotheses are scored as full graphs, i.e. every node's local score under
    that hypothesis, so the comparison is between two complete factorizations
    rather than between one edge's score in isolation.
    """
    true_dir = scorer.local_score("B", ["A"]) + scorer.local_score("A", [])
    reverse_dir = scorer.local_score("A", ["B"]) + scorer.local_score("B", [])
    return true_dir - reverse_dir


def test_pooled_score_orients_edge_correctly():
    """The pooled path prefers the true A->B over B->A by a decisive margin."""
    data, arm_labels, clamped_nodes = _two_node_scm()

    margin = _orientation_margin(_scorer(data, arm_labels, clamped_nodes, pooled=True))

    # Observed ~ +108 at this seed; ~ +98 to +111 across seeds 0-4.
    assert margin > 50.0, f"pooled score failed to orient A->B (margin={margin})"


def test_armwise_score_does_not_orient_edge():
    """Regression test DOCUMENTING A KNOWN LIMITATION of the per-arm path.

    This asserts that the per-arm interventional score (`pooled_interventional=False`,
    the default) does NOT meaningfully prefer the true orientation, because each arm's
    own intercept absorbs that arm's mean shift in the scored variable and leaves only
    direction-free within-arm covariance. That is the defect the pooled path exists to
    fix - it is NOT a property we want, and it is NOT a test to "fix" by flipping the
    comparison. If a future change makes the per-arm path orientation-aware, delete
    this test deliberately with that reasoning; do not invert it to make it pass.

    The per-arm margin at these seeds sits near zero and is sometimes negative (i.e.
    it favors the WRONG orientation), which is why this only asserts the absence of a
    meaningful preference rather than a particular sign.
    """
    data, arm_labels, clamped_nodes = _two_node_scm()

    armwise_margin = _orientation_margin(_scorer(data, arm_labels, clamped_nodes, pooled=False))
    pooled_margin = _orientation_margin(_scorer(data, arm_labels, clamped_nodes, pooled=True))

    # Observed ~ +6 at this seed; ~ -5 to +9 across seeds 0-4, versus ~ +100 pooled.
    assert armwise_margin < 25.0, (
        "per-arm interventional score unexpectedly oriented the edge decisively "
        f"(margin={armwise_margin}); see this test's docstring before changing it"
    )
    assert pooled_margin > armwise_margin + 50.0, (
        f"pooled margin ({pooled_margin}) is not substantially larger than the "
        f"per-arm margin ({armwise_margin})"
    )


def test_pooled_penalty_uses_retained_row_count():
    """The pooled penalty is exactly ((df_model + 2)/2) * log(n_used).

    `n_used` is the number of retained rows (arms not clamping the scored variable),
    and the prior bonus is exactly 0 here, so the local score must equal
    `ll - penalty` for a plain fit over precisely those rows.
    """
    data, arm_labels, clamped_nodes = _two_node_scm()

    # Scoring "B": only "do_B" clamps B, so "obs" + "do_A" rows are retained.
    retained = data.loc[(arm_labels != "do_B").values]
    reference = pdr.BICGaussIndraPriors(retained, edge_priors=EDGE_PRIORS)
    ll, df_model = reference._log_likelihood(variable="B", parents=["A"])
    expected = ll - (((df_model + 2) / 2) * np.log(len(retained)))

    pooled = _scorer(data, arm_labels, clamped_nodes, pooled=True)
    assert pooled.local_score("B", ["A"]) == expected


def test_pooled_penalty_does_not_scale_with_arm_count():
    """Splitting every arm into two identically-clamped halves leaves the pooled score
    unchanged (same rows, same single fit, same penalty over the same `n_used`), while
    the per-arm score shifts because it pays a fit - and an intercept - per arm."""
    data, arm_labels, clamped_nodes = _two_node_scm()

    # Split each 30-row arm into two 15-row sub-arms carrying the same clamping.
    halves = pd.Series(
        np.where(np.arange(len(arm_labels)) % 30 < 15, "__a", "__b"), index=arm_labels.index
    )
    split_labels = arm_labels + halves
    split_clamped = {
        sub: clamped_nodes[sub.split("__")[0]]
        for sub in pd.unique(split_labels)
        if sub.split("__")[0] in clamped_nodes
    }
    assert len(pd.unique(split_labels)) == 2 * len(pd.unique(arm_labels))

    pooled_before = _scorer(data, arm_labels, clamped_nodes, pooled=True).local_score("B", ["A"])
    pooled_after = _scorer(data, split_labels, split_clamped, pooled=True).local_score("B", ["A"])
    assert pooled_before == pooled_after, (
        f"pooled score changed with arm count ({pooled_before} -> {pooled_after}); "
        "the penalty must depend only on the retained row count"
    )

    armwise_before = _scorer(data, arm_labels, clamped_nodes, pooled=False).local_score("B", ["A"])
    armwise_after = _scorer(data, split_labels, split_clamped, pooled=False).local_score("B", ["A"])
    # Not a desired property - just pinning the contrast that motivates the pooled path.
    assert armwise_before != armwise_after


def test_pooled_score_retains_rows_where_a_parent_is_clamped():
    """Clamping a *parent* must drop no rows and remove no regressor.

    A clamped parent's experimenter-set value is an ordinary regressor for another
    node's local mechanism, so scoring "B" with parent "A" keeps the "do_A" rows and
    scoring "A" with parent "B" keeps the "do_B" rows. Each is checked by matching
    the score against an explicit fit over exactly the rows that should be retained.
    """
    data, arm_labels, clamped_nodes = _two_node_scm()
    pooled = _scorer(data, arm_labels, clamped_nodes, pooled=True)

    for variable, parent, clamping_arm in (("B", "A", "do_B"), ("A", "B", "do_A")):
        # Everything except the arm that clamps `variable` itself - which includes
        # the arm that clamps `parent`.
        retained = data.loc[(arm_labels != clamping_arm).values]
        retained_arms = set(pd.unique(arm_labels[(arm_labels != clamping_arm).values]))
        assert f"do_{parent}" in retained_arms
        assert len(retained) == 60

        reference = pdr.BICGaussIndraPriors(retained, edge_priors=EDGE_PRIORS)
        ll, df_model = reference._log_likelihood(variable=variable, parents=[parent])
        expected = ll - (((df_model + 2) / 2) * np.log(len(retained)))

        assert pooled.local_score(variable, [parent]) == expected


def test_pooled_score_is_neg_inf_when_variable_clamped_in_every_arm():
    """No arm carries information about a variable clamped everywhere -> -inf."""
    data, arm_labels, _ = _two_node_scm()
    clamped_everywhere = {arm: ["B"] for arm in pd.unique(arm_labels)}

    pooled = _scorer(data, arm_labels, clamped_everywhere, pooled=True)

    assert pooled.local_score("B", ["A"]) == -np.inf


def test_default_construction_is_unchanged_by_the_new_flag():
    """A scorer built with no interventional kwargs returns the pre-change value.

    The literals below were captured from the observational code path at commit
    e58b54c (before `pooled_interventional` existed) on this exact fixed data.
    They guard the requirement that nothing changes unless the new flag is
    explicitly set to True.

    Compared with a relative tolerance rather than `==` because the underlying GLM
    fit goes through whatever BLAS/LAPACK the platform provides, which reorders
    floating-point reductions and shifts the last digit or two of the result (the
    observed cross-platform spread is ~1e-16 relative, i.e. 1-2 ULP). `rel=1e-9`
    is still many orders of magnitude tighter than any genuine behavior change -
    reaching the wrong branch, mis-scaling the penalty, or retaining the wrong rows
    all move these values by whole units or more - so it keeps the regression
    guard's teeth while surviving a different BLAS build. Don't tighten this back
    to exact equality.
    """
    data, _, _ = _two_node_scm()
    scorer = pdr.BICGaussIndraPriors(data, edge_priors=EDGE_PRIORS)

    assert scorer.local_score("B", ["A"]) == pytest.approx(-181.9218239637686, rel=1e-9)
    assert scorer.local_score("A", ["B"]) == pytest.approx(-154.11763570887948, rel=1e-9)
    # Default flag value must be off, so local_score never reaches the pooled branch.
    assert scorer.pooled_interventional is False


def test_armwise_path_unchanged_when_flag_left_off():
    """With `pooled_interventional` unset, the interventional branch still produces
    the per-arm value captured at commit e58b54c on this fixed data.

    See `test_default_construction_is_unchanged_by_the_new_flag` for why these are
    compared with a relative tolerance instead of exact equality.
    """
    data, arm_labels, clamped_nodes = _two_node_scm()

    scorer = pdr.BICGaussIndraPriors(
        data,
        edge_priors=EDGE_PRIORS,
        interventional=True,
        arm_labels=arm_labels,
        clamped_nodes=clamped_nodes,
    )

    assert scorer.local_score("B", ["A"]) == pytest.approx(-42.70524914189197, rel=1e-9)
    assert _orientation_margin(scorer) == pytest.approx(6.02689564594472, rel=1e-9)
