"""Coverage for the pooled (Hauser-Buhlmann style) GIES interventional local score,
shared by BICGaussIndraPriors and BICGaussNoPriors via
_PooledInterventionalScoreMixin.

Its defining property is that it fits ONE GLM - and therefore estimates ONE
intercept - over every row where the scored variable isn't clamped. Fitting per
arm instead would hand each arm its own free intercept, and that intercept
absorbs the arm's mean shift in the scored variable; those between-arm mean shifts
are exactly the interventional signal that orients an edge, so a per-arm fit is
left scoring only within-arm covariance, which is symmetric in the two nodes of an
edge and says nothing about direction. An earlier per-arm implementation of this
method was removed for that reason.

These tests use a synthetic two-node SCM with a known direction (A -> B) and three
arms, and check that the score recovers the direction, that the complexity penalty
depends only on the retained row count (not the arm count), that clamping applies
to the scored variable only and never to a parent, and that with nothing clamped
the score collapses exactly onto the flat observational score.

The last section covers BICGaussNoPriors specifically: it must get the identical
pooled BIC (same retained rows, same penalty) with the prior log-odds term simply
absent, since both classes route through the same mixin method.
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

# Deliberately lopsided, for the tests that need a NONZERO prior contribution in
# order to show it is the only thing BICGaussNoPriors leaves out.
SKEWED_EDGE_PRIORS = {("A", "B"): 0.9, ("B", "A"): 0.2}

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


def _scorer(data, arm_labels, clamped_nodes):
    return pdr.BICGaussIndraPriors(
        data,
        edge_priors=EDGE_PRIORS,
        interventional=True,
        arm_labels=arm_labels,
        clamped_nodes=clamped_nodes,
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


def test_interventional_score_orients_edge_correctly():
    """The pooled score prefers the true A->B over B->A by a decisive margin."""
    data, arm_labels, clamped_nodes = _two_node_scm()

    margin = _orientation_margin(_scorer(data, arm_labels, clamped_nodes))

    # Observed ~ +108 at this seed; ~ +98 to +111 across seeds 0-4. The removed
    # per-arm implementation scored ~ +6 here, and went negative on some seeds.
    assert margin > 50.0, f"failed to orient A->B (margin={margin})"


def test_penalty_uses_retained_row_count():
    """The penalty is exactly ((df_model + 2)/2) * log(n_used).

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

    assert _scorer(data, arm_labels, clamped_nodes).local_score("B", ["A"]) == expected


def test_penalty_does_not_scale_with_arm_count():
    """Splitting every arm into two identically-clamped halves leaves the score
    unchanged: same rows retained, one fit either way, same penalty over the same
    `n_used`. The removed per-arm implementation shifted by ~1.2 here, because it
    paid an extra fit - and an extra intercept - per arm."""
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

    before = _scorer(data, arm_labels, clamped_nodes).local_score("B", ["A"])
    after = _scorer(data, split_labels, split_clamped).local_score("B", ["A"])
    assert before == after, (
        f"score changed with arm count ({before} -> {after}); the penalty must "
        "depend only on the retained row count"
    )


def test_score_reduces_to_observational_when_nothing_is_clamped():
    """With no clamped nodes, the interventional score IS the flat observational
    score - pooling means the arm partitioning alone never moves a score. This is
    the invariant that replaced the removed per-arm path, which changed the score
    (here by ~117) purely from how rows were grouped into arms, even with zero
    interventions."""
    data, arm_labels, _ = _two_node_scm()

    flat = pdr.BICGaussIndraPriors(data, edge_priors=EDGE_PRIORS).local_score("B", ["A"])

    # Multi-arm labels, but no clamps anywhere.
    assert _scorer(data, arm_labels, {}).local_score("B", ["A"]) == flat
    assert _scorer(data, arm_labels, None).local_score("B", ["A"]) == flat
    # And a single all-observational arm.
    one_arm = pd.Series(["obs"] * len(data), index=data.index)
    assert _scorer(data, one_arm, {}).local_score("B", ["A"]) == flat


def test_score_retains_rows_where_a_parent_is_clamped():
    """Clamping a *parent* must drop no rows and remove no regressor.

    A clamped parent's experimenter-set value is an ordinary regressor for another
    node's local mechanism, so scoring "B" with parent "A" keeps the "do_A" rows and
    scoring "A" with parent "B" keeps the "do_B" rows. Each is checked by matching
    the score against an explicit fit over exactly the rows that should be retained.
    """
    data, arm_labels, clamped_nodes = _two_node_scm()
    scorer = _scorer(data, arm_labels, clamped_nodes)

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

        assert scorer.local_score(variable, [parent]) == expected


def test_score_is_neg_inf_when_variable_clamped_in_every_arm():
    """No arm carries information about a variable clamped everywhere -> -inf."""
    data, arm_labels, _ = _two_node_scm()
    clamped_everywhere = {arm: ["B"] for arm in pd.unique(arm_labels)}

    assert _scorer(data, arm_labels, clamped_everywhere).local_score("B", ["A"]) == -np.inf


def test_observational_path_unaffected_by_the_interventional_branch():
    """A scorer built with no interventional kwargs returns the pre-change value.

    The literals below were captured from the observational code path at commit
    e58b54c (before any of this interventional work) on this exact fixed data.
    They guard the requirement that the flat path is untouched.

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

    # interventional=True without arm_labels must stay on the flat path.
    flag_only = pdr.BICGaussIndraPriors(data, edge_priors=EDGE_PRIORS, interventional=True)
    assert flag_only.local_score("B", ["A"]) == scorer.local_score("B", ["A"])


# ---------------------------------------------------------------------------
# BICGaussNoPriors - same pooled interventional score, minus the prior term
# ---------------------------------------------------------------------------
def _no_priors_scorer(data, arm_labels, clamped_nodes):
    return pdr.BICGaussNoPriors(
        data, interventional=True, arm_labels=arm_labels, clamped_nodes=clamped_nodes
    )


def test_no_priors_interventional_differs_from_prior_aware_only_by_the_prior():
    """BICGaussNoPriors must produce the identical pooled BIC, with the edge
    log-odds term simply absent.

    Uses lopsided priors so the bonus is nonzero and this is a real constraint: the
    difference between the two scorers must equal sum(log(p/(1-p))) over the parents
    exactly. That pins both halves of the contract at once - same retained rows and
    same complexity penalty (otherwise the difference wouldn't be exactly the
    bonus), and no prior contribution in the no-priors class.
    """
    data, arm_labels, clamped_nodes = _two_node_scm()

    no_priors = _no_priors_scorer(data, arm_labels, clamped_nodes)
    with_priors = pdr.BICGaussIndraPriors(
        data,
        edge_priors=SKEWED_EDGE_PRIORS,
        interventional=True,
        arm_labels=arm_labels,
        clamped_nodes=clamped_nodes,
    )

    for variable, parents in (("B", ["A"]), ("A", ["B"]), ("B", []), ("A", [])):
        bonus = sum(
            np.log(SKEWED_EDGE_PRIORS[(p, variable)] / (1 - SKEWED_EDGE_PRIORS[(p, variable)]))
            for p in parents
        )
        assert with_priors.local_score(variable, parents) - no_priors.local_score(
            variable, parents
        ) == pytest.approx(bonus, abs=1e-12)


def test_no_priors_interventional_score_orients_edge_correctly():
    """With no prior term at all, the pooled likelihood alone must orient A -> B."""
    data, arm_labels, clamped_nodes = _two_node_scm()

    margin = _orientation_margin(_no_priors_scorer(data, arm_labels, clamped_nodes))

    # Matches the prior-aware scorer's ~+108 at p=0.5, where the prior contributes 0.
    assert margin > 50.0, f"failed to orient A->B without priors (margin={margin})"


def test_no_priors_penalty_uses_retained_row_count():
    """The no-priors interventional score is exactly the pooled BIC over the
    retained rows - no prior term, so nothing else is added."""
    data, arm_labels, clamped_nodes = _two_node_scm()

    retained = data.loc[(arm_labels != "do_B").values]
    reference = pdr.BICGaussNoPriors(retained)
    ll, df_model = reference._log_likelihood(variable="B", parents=["A"])
    expected = ll - (((df_model + 2) / 2) * np.log(len(retained)))

    assert _no_priors_scorer(data, arm_labels, clamped_nodes).local_score("B", ["A"]) == expected


def test_no_priors_score_reduces_to_observational_when_nothing_is_clamped():
    """Same pooling invariant as the prior-aware class: arm labels alone never
    move a score."""
    data, arm_labels, _ = _two_node_scm()

    flat = pdr.BICGaussNoPriors(data).local_score("B", ["A"])

    assert _no_priors_scorer(data, arm_labels, {}).local_score("B", ["A"]) == flat
    assert _no_priors_scorer(data, arm_labels, None).local_score("B", ["A"]) == flat


def test_no_priors_score_is_neg_inf_when_variable_clamped_in_every_arm():
    data, arm_labels, _ = _two_node_scm()
    clamped_everywhere = {arm: ["B"] for arm in pd.unique(arm_labels)}

    assert (
        _no_priors_scorer(data, arm_labels, clamped_everywhere).local_score("B", ["A"]) == -np.inf
    )


def test_no_priors_observational_path_unaffected():
    """The default (no interventional kwargs) path still returns plain BIC, and
    `interventional=True` without `arm_labels` stays on it."""
    data, _, _ = _two_node_scm()
    scorer = pdr.BICGaussNoPriors(data)

    ll, df_model = scorer._log_likelihood(variable="B", parents=["A"])
    expected = ll - (((df_model + 2) / 2) * np.log(len(data)))
    assert scorer.local_score("B", ["A"]) == expected

    # No prior bonus: identical to the prior-aware scorer at p=0.5, where the
    # bonus is exactly log(1) == 0.
    with_neutral_priors = pdr.BICGaussIndraPriors(data, edge_priors=EDGE_PRIORS)
    assert scorer.local_score("B", ["A"]) == with_neutral_priors.local_score("B", ["A"])

    flag_only = pdr.BICGaussNoPriors(data, interventional=True)
    assert flag_only.local_score("B", ["A"]) == expected


def test_no_priors_rejects_misaligned_arm_labels():
    """The shared `_init_interventional` validation applies here too."""
    data, _, _ = _two_node_scm()
    bad_labels = pd.Series(["obs"] * len(data), index=range(1000, 1000 + len(data)))

    with pytest.raises(AssertionError, match="arm_labels must share data's index"):
        pdr.BICGaussNoPriors(data, interventional=True, arm_labels=bad_labels)
