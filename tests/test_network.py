"""Tests for the pure graph-processing helpers in causomic.network.

Covers the causal-path filtering used to prune a posterior graph down to the
nodes relevant to a treatment/outcome query, and the consensus-DAG aggregation
that turns a set of bootstrap DAGs plus edge priors into a single acyclic graph.
extract_indra_prior and repair_confounding require external services and are
not exercised here. estimate_posterior_dag's hill-climb-driven selections
("best_of"/"consensus") are too heavy for unit tests, but its "dagma" selection
is a single, cheap continuous-optimization fit and is exercised below (gated
behind pytest.importorskip("dagma") since it's an optional dependency).
"""

import importlib

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from y0.dsl import Variable
from y0.graph import NxMixedGraph

net = importlib.import_module("causomic.network")


def _mixed_graph(directed_edges):
    return NxMixedGraph.from_edges(directed=directed_edges)


def test_nodes_on_causal_paths_basic_chain():
    # X -> M -> Y, with A -> B a disconnected side chain.
    G = _mixed_graph([("X", "M"), ("M", "Y"), ("A", "B")])
    on_path = net.nodes_on_causal_paths(G, [Variable("X")], [Variable("Y")])
    assert {str(n) for n in on_path} == {"X", "M", "Y"}


def test_nodes_on_causal_paths_excludes_off_path_nodes():
    # M lies on X->Y; D branches off X but never reaches Y.
    G = _mixed_graph([("X", "M"), ("M", "Y"), ("X", "D")])
    on_path = net.nodes_on_causal_paths(G, [Variable("X")], [Variable("Y")])
    assert "D" not in {str(n) for n in on_path}
    assert {str(n) for n in on_path} == {"X", "M", "Y"}


def test_nodes_on_causal_paths_no_connection_is_empty():
    # No directed path from X to Y.
    G = _mixed_graph([("X", "M"), ("Y", "Z")])
    on_path = net.nodes_on_causal_paths(G, [Variable("X")], [Variable("Y")])
    assert on_path == set()


def test_filter_to_causal_subgraph_keeps_only_path_nodes():
    G = _mixed_graph([("X", "M"), ("M", "Y"), ("A", "B")])
    H = net.filter_to_causal_subgraph(G, [Variable("X")], [Variable("Y")])
    assert {str(n) for n in H.directed.nodes} == {"X", "M", "Y"}
    assert {(str(u), str(v)) for u, v in H.directed.edges} == {("X", "M"), ("M", "Y")}


def _priors(rows):
    return pd.DataFrame(rows, columns=["source", "target", "edge_p"])


def test_consensus_dag_includes_frequent_edges():
    dags = []
    for _ in range(4):
        g = nx.DiGraph()
        g.add_edge("X", "Y")
        dags.append(g)
    priors = _priors([("X", "Y", 0.9)])
    out = net.consensus_dag(dags, priors, min_freq=0.5)
    assert ("X", "Y") in out.edges()


def test_consensus_dag_drops_infrequent_edges():
    # X->Y in all 4 (freq 1.0); M->Y in only 1 (freq 0.25) -> dropped at min_freq=0.5.
    dags = []
    for i in range(4):
        g = nx.DiGraph()
        g.add_edge("X", "Y")
        if i == 0:
            g.add_edge("M", "Y")
        dags.append(g)
    priors = _priors([("X", "Y", 0.9), ("M", "Y", 0.9)])
    out = net.consensus_dag(dags, priors, min_freq=0.5)
    assert ("X", "Y") in out.edges()
    assert ("M", "Y") not in out.edges()


def test_consensus_dag_result_is_acyclic():
    # Conflicting directions both appear frequently; consensus must stay acyclic.
    dags = []
    for _ in range(3):
        g = nx.DiGraph()
        g.add_edge("X", "Y")
        g.add_edge("Y", "X")
        dags.append(g)
    priors = _priors([("X", "Y", 0.9), ("Y", "X", 0.9)])
    out = net.consensus_dag(dags, priors, min_freq=0.5)
    assert nx.is_directed_acyclic_graph(out)


def test_extract_indra_prior_runs_and_respects_verbose(monkeypatch, capsys):
    # Regression: extract_indra_prior referenced an undefined `verbose` at the
    # summary-print step, raising NameError whenever called. Mock the INDRA
    # query helpers so the function runs end-to-end without a Neo4j client.
    one_row = pd.DataFrame(
        {
            "source": ["A"],
            "target": ["B"],
            "relation": ["IncreaseAmount"],
            "evidence_count": [2],
        }
    )
    monkeypatch.setattr(net, "get_ids", lambda names, kind: list(names))
    monkeypatch.setattr(net, "get_one_step_root_down", lambda **kw: "q1")
    monkeypatch.setattr(net, "get_two_step_root_known_med", lambda **kw: "q2")
    monkeypatch.setattr(net, "get_three_step_root", lambda **kw: "q3")
    monkeypatch.setattr(net, "format_query_results", lambda _q: one_row.copy())

    result = net.extract_indra_prior(
        source=["A"], target=["B"], measured_proteins=["A", "B"], client=object(), verbose=True
    )
    # Three identical mocked queries -> evidence counts summed for the A->B edge.
    assert list(result.columns) == ["source", "target", "evidence_count"]
    assert result.loc[0, "source"] == "A"
    assert result.loc[0, "target"] == "B"
    assert result.loc[0, "evidence_count"] == 6
    assert capsys.readouterr().out != ""

    net.extract_indra_prior(
        source=["A"], target=["B"], measured_proteins=["A", "B"], client=object(), verbose=False
    )
    assert capsys.readouterr().out == ""


def test_consensus_dag_ignores_none_entries():
    dags = [None, None]
    g = nx.DiGraph()
    g.add_edge("X", "Y")
    dags.append(g)
    priors = _priors([("X", "Y", 0.9)])
    out = net.consensus_dag(dags, priors, min_freq=0.5)
    # Only one valid DAG, edge freq 1.0.
    assert ("X", "Y") in out.edges()


def test_estimate_posterior_dag_dagma_selection_restricts_to_prior_edges():
    pytest.importorskip("dagma")

    rng = np.random.default_rng(0)
    n = 500
    A = rng.normal(size=n)
    B = 2.0 * A + rng.normal(scale=0.1, size=n)
    C = rng.normal(size=n)  # independent noise, absent from the prior
    data = pd.DataFrame({"A": A, "B": B, "C": C})

    indra_priors = pd.DataFrame({"source": ["A"], "target": ["B"], "evidence_count": [10]})

    y0_graph, bootstrap_dags = net.estimate_posterior_dag(
        data,
        indra_priors,
        selection="dagma",
        return_bootstrap_dags=True,
        verbose=False,
    )

    edges = {(str(u), str(v)) for u, v in y0_graph.directed.edges()}
    assert edges == {("A", "B")}
    # Single deterministic full-data fit -> exactly one candidate DAG.
    assert len(bootstrap_dags) == 1
    for u, v in y0_graph.directed.edges():
        assert y0_graph.directed[u][v]["edge_prob"] == 1.0


def test_estimate_posterior_dag_dagma_weighted_selection_uses_evidence_strength():
    pytest.importorskip("dagma")

    # Two structurally identical edges (A->B, C->D); only the INDRA evidence
    # strength differs. At this lambda1, plain "dagma" keeps both, but
    # "dagma_weighted" should favor the well-evidenced edge and drop the
    # poorly-evidenced one despite the identical true effect size.
    rng = np.random.default_rng(0)
    n = 400
    A = rng.normal(size=n)
    C = rng.normal(size=n)
    coef = 0.5
    B = coef * A + rng.normal(scale=1.0, size=n)
    D = coef * C + rng.normal(scale=1.0, size=n)
    data = pd.DataFrame({"A": A, "B": B, "C": C, "D": D})

    indra_priors = pd.DataFrame(
        {"source": ["A", "C"], "target": ["B", "D"], "evidence_count": [50, 1]}
    )

    y0_plain = net.estimate_posterior_dag(
        data.copy(),
        indra_priors.copy(),
        selection="dagma",
        dagma_lambda1=0.3,
        dagma_w_threshold=0.1,
        verbose=False,
    )
    y0_weighted = net.estimate_posterior_dag(
        data.copy(),
        indra_priors.copy(),
        selection="dagma_weighted",
        dagma_lambda1=0.3,
        dagma_w_threshold=0.1,
        verbose=False,
    )

    plain_edges = {(str(u), str(v)) for u, v in y0_plain.directed.edges()}
    weighted_edges = {(str(u), str(v)) for u, v in y0_weighted.directed.edges()}

    assert plain_edges == {("A", "B"), ("C", "D")}
    assert weighted_edges == {("A", "B")}
