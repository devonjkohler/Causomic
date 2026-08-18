"""Tests for the pure helpers in posterior_estimation.diagnostics.

compare_dag_sets is deterministic given fixed DAG sets; the Parallel/HillClimb-driven
search_path_diagnostic itself needs an estimator and is not exercised here.

random_acyclic_subgraph used to be duplicated in this module and is now imported
from posterior_estimation.hill_climb, where tests/test_posterior_estimation.py
covers it (including the max_indegree cap the copy here was missing).
"""

import importlib

import numpy as np
from pgmpy.base import DAG

spd = importlib.import_module("causomic.graph_construction.posterior_estimation.diagnostics")


def _dag(edges):
    d = DAG()
    d.add_edges_from(edges)
    return d


def test_compare_dag_sets_frequencies_and_diff():
    # Edge A->B in all of set-a, only half of set-b.
    dags_a = [_dag([("A", "B")]), _dag([("A", "B")])]
    dags_b = [_dag([("A", "B")]), _dag([("C", "D")])]
    df = spd.compare_dag_sets(dags_a, dags_b)
    row_ab = df[(df["source"] == "A") & (df["target"] == "B")].iloc[0]
    assert row_ab["freq_random_init"] == 1.0
    assert row_ab["freq_bootstrap"] == 0.5
    assert np.isclose(row_ab["abs_diff"], 0.5)
    # Both distinct edges represented.
    assert set(zip(df["source"], df["target"])) == {("A", "B"), ("C", "D")}


def test_compare_dag_sets_sorted_by_abs_diff_desc():
    dags_a = [_dag([("A", "B"), ("C", "D")])]
    dags_b = [_dag([("C", "D")])]
    df = spd.compare_dag_sets(dags_a, dags_b)
    # A->B differs by 1.0, C->D by 0.0; the larger diff sorts first.
    assert df.iloc[0]["source"] == "A"
    assert df["abs_diff"].is_monotonic_decreasing


def test_compare_dag_sets_custom_labels():
    dags_a = [_dag([("A", "B")])]
    dags_b = [_dag([("A", "B")])]
    df = spd.compare_dag_sets(dags_a, dags_b, label_a="rand", label_b="boot")
    assert "freq_rand" in df.columns
    assert "freq_boot" in df.columns
