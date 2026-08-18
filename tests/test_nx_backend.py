import importlib
import os

import networkx as nx
import pandas as pd

# Ensure src is on path via tests/conftest.py
nx_backend = importlib.import_module("causomic.graph_construction.prior_extraction.nx_backend")


def make_sample_graph():
    G = nx.DiGraph()
    # add nodes with namespace attribute
    for n in ["n1", "n2", "n3", "n4"]:
        G.add_node(n, ns="HGNC")

    # edges with statements list (dicts)
    G.add_edge(
        "n1",
        "n2",
        statements=[
            {"stmt_type": "IncreaseAmount", "evidence_count": 2, "source_counts": {"SRC1": 1}}
        ],
    )
    G.add_edge(
        "n2",
        "n3",
        statements=[
            {
                "stmt_type": "DecreaseAmount",
                "evidence_count": 5,
                "source_counts": {"SRC1": 2, "SRC2": 1},
            }
        ],
    )
    G.add_edge(
        "n3",
        "n4",
        statements=[{"stmt_type": "OtherType", "evidence_count": None, "source_counts": {}}],
    )

    return G


def test_prepare_graph_filters_statements_and_builds_evidence():
    G = make_sample_graph()
    out = nx_backend.prepare_graph(
        G,
        measured_nodes=["n1", "n2", "n3", "n4"],
        node_types=["HGNC"],
        stmt_types=["IncreaseAmount"],
    )
    assert out.has_edge("n1", "n2")
    assert not out.has_edge("n2", "n3")
    stmts = out["n1"]["n2"]["statements"]
    assert all(s.get("stmt_type") == "IncreaseAmount" for s in stmts)
    assert out["n1"]["n2"]["evidence"]["total_evidence"] == 2


def test_add_evidence_info_and_filter_by_evidence_count():
    G = make_sample_graph()
    nx_backend.add_evidence_info(G)
    # evidence should be attached
    assert G["n1"]["n2"].get("evidence") is not None
    # n2->n3 has evidence_count 5 so edge_ok with thr=5 should be True
    assert nx_backend.edge_ok(G, "n2", "n3", thr=5)

    # filter edges with threshold 3 should keep only n2->n3
    f = nx_backend.filter_graph_by_evidence_count(G, 3)
    assert f.number_of_edges() == 1
    assert f.has_edge("n2", "n3")


def test_prepare_graph_filters_measured_nodes():
    G = make_sample_graph()
    prepared = nx_backend.prepare_graph(
        G,
        measured_nodes=["n1", "n2"],
        node_types=["HGNC"],
        stmt_types=["IncreaseAmount", "DecreaseAmount", "OtherType"],
    )
    assert prepared.number_of_edges() == 1
    assert prepared.has_edge("n1", "n2")


def test_query_confounders_returns_dataframe():
    G = make_sample_graph()
    # attach evidence info first
    nx_backend.add_evidence_info(G)
    # create a common confounder node that points to n2 and n3
    G.add_edge(
        "c",
        "n2",
        statements=[
            {"stmt_type": "IncreaseAmount", "evidence_count": 3, "source_counts": {"S1": 1}}
        ],
    )
    G.add_edge(
        "c",
        "n3",
        statements=[
            {"stmt_type": "IncreaseAmount", "evidence_count": 4, "source_counts": {"S1": 1}}
        ],
    )
    nx_backend.add_evidence_info(G)
    df = nx_backend.query_confounders(G, ["n2", "n3"])
    assert isinstance(df, pd.DataFrame)
    # should contain rows for the confounder 'c'
    assert (df["source"] == "c").any()


def test_filtered_paths_and_query_forward_paths():
    G = nx.DiGraph()
    G.add_edges_from(
        [
            (
                "A",
                "B",
                {
                    "evidence": {
                        "total_evidence": 2,
                        "source_evidence": 1,
                        "stmt_type": ["IncreaseAmount"],
                    }
                },
            ),
            (
                "B",
                "C",
                {
                    "evidence": {
                        "total_evidence": 2,
                        "source_evidence": 1,
                        "stmt_type": ["IncreaseAmount"],
                    }
                },
            ),
        ]
    )
    # filtered_paths yields the path A->B->C
    paths = list(nx_backend.filtered_paths(G, "A", "C", cutoff=2, thr=1))
    assert any(path for path in paths if path[0] == "A" and path[-1] == "C")

    # query_forward_paths should return dataframe with the forward edges
    fwd = nx_backend.query_forward_paths(
        G, start_nodes=["A"], end_nodes=["C"], n_mediators=2, med_ev_filter=[1, 1, 1]
    )
    assert isinstance(fwd, pd.DataFrame)
    assert set(["source", "target"]).issubset(fwd.columns)


def test_query_forward_paths_counts_mediators_not_edges():
    G = nx.DiGraph()
    G.add_edges_from(
        [
            (
                "A",
                "B",
                {
                    "evidence": {
                        "total_evidence": 2,
                        "source_evidence": 1,
                        "stmt_type": ["IncreaseAmount"],
                    }
                },
            ),
            (
                "B",
                "C",
                {
                    "evidence": {
                        "total_evidence": 2,
                        "source_evidence": 1,
                        "stmt_type": ["IncreaseAmount"],
                    }
                },
            ),
        ]
    )

    direct_only = nx_backend.query_forward_paths(
        G,
        start_nodes=["A"],
        end_nodes=["C"],
        n_mediators=0,
        med_ev_filter=[1],
    )
    assert direct_only.empty

    one_mediator = nx_backend.query_forward_paths(
        G,
        start_nodes=["A"],
        end_nodes=["C"],
        n_mediators=1,
        med_ev_filter=[1, 1],
    )
    assert set(zip(one_mediator["source"], one_mediator["target"])) == {
        ("A", "B"),
        ("B", "C"),
    }


def _make_weak_chain_graph():
    # A -> B -> C, each edge with weak evidence (2) that only clears a
    # depth-1 (1-mediator) threshold of 1, not a depth-0 (direct) threshold
    # of 5.
    G = nx.DiGraph()
    for u, v in [("A", "B"), ("B", "C")]:
        G.add_edge(
            u,
            v,
            evidence={
                "total_evidence": 2,
                "source_evidence": 1,
                "stmt_type": ["IncreaseAmount"],
            },
        )
    return G


def test_query_forward_paths_default_excludes_other_start_as_mediator():
    # Regression/documentation: when start_nodes == end_nodes (the
    # closed-neighborhood case), the default exclude_other_starts=True drops
    # every other list member as a candidate mediator for a given pair, so a
    # real A->B->C relationship that only clears the threshold at the
    # 1-mediator depth is missed entirely: the (A, B) and (B, C) rounds
    # reject it at their own direct (depth-0) threshold, and the (A, C)
    # round can't route through the excluded node B.
    G = _make_weak_chain_graph()
    nodes = ["A", "B", "C"]

    result = nx_backend.query_forward_paths(
        G,
        start_nodes=nodes,
        end_nodes=nodes,
        n_mediators=1,
        med_ev_filter=[5, 1],
    )
    assert result.empty


def test_query_neighborhood_paths_allows_internal_mediators():
    # Same graph/thresholds as above, but via query_neighborhood_paths
    # (exclude_other_starts=False): B is now a valid mediator between A and
    # C, so both edges on the chain are recovered.
    G = _make_weak_chain_graph()
    nodes = ["A", "B", "C"]

    result = nx_backend.query_neighborhood_paths(
        G,
        nodes,
        n_mediators=1,
        med_ev_filter=[5, 1],
    )
    assert set(zip(result["source"], result["target"])) == {
        ("A", "B"),
        ("B", "C"),
    }

    # Equivalent to calling query_forward_paths directly with the flag off.
    direct = nx_backend.query_forward_paths(
        G,
        start_nodes=nodes,
        end_nodes=nodes,
        n_mediators=1,
        med_ev_filter=[5, 1],
        exclude_other_starts=False,
    )
    assert set(zip(direct["source"], direct["target"])) == set(
        zip(result["source"], result["target"])
    )
