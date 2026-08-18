"""Trim an estimated graph to the part that carries causal signal.

A posterior DAG generally contains nodes irrelevant to the question being
asked -- side branches, downstream readouts of the outcome, disconnected
components. For estimating the effect of ``start_nodes`` on ``end_nodes``, only
nodes lying on some directed path between them can matter. These helpers find
that set and cut the graph down to it, which keeps identification and fitting
focused on the relevant subgraph.
"""

from typing import Iterable, Set

import networkx as nx
from y0.graph import NxMixedGraph


def nodes_on_causal_paths(
    G: NxMixedGraph,
    start_nodes: Iterable[str],
    end_nodes: Iterable[str],
) -> Set[str]:
    """Return the set of nodes that lie on at least one directed path
    from any node in `start_nodes` to any node in `end_nodes`.

    Uses only G.directed for path traversal. Runs in O(V + E) via two
    BFS passes.
    """
    directed = G.directed
    # y0 stores nodes as Variable objects, but callers typically pass plain gene
    # name strings. Match on the node's string name so either form resolves to the
    # actual graph node (otherwise the set intersection is empty and the filter
    # silently drops every node).
    by_name = {getattr(n, "name", str(n)): n for n in directed.nodes}

    def _resolve(names):
        resolved = set()
        for x in names:
            key = getattr(x, "name", str(x))
            if key in by_name:
                resolved.add(by_name[key])
        return resolved

    start_nodes = _resolve(start_nodes)
    end_nodes = _resolve(end_nodes)

    # Forward-reachable from any start node
    forward = set()
    for s in start_nodes:
        forward |= nx.descendants(directed, s)
    forward |= start_nodes

    # Backward-reachable from any end node (traverse reversed graph)
    rev = directed.reverse(copy=False)
    backward = set()
    for e in end_nodes:
        backward |= nx.descendants(rev, e)
    backward |= end_nodes

    return forward & backward


def filter_to_causal_subgraph(
    G: NxMixedGraph,
    start_nodes: Iterable[str],
    end_nodes: Iterable[str],
) -> NxMixedGraph:
    """Return a new NxMixedGraph containing only nodes on directed
    causal paths, preserving all edge types (directed and bidirected)
    between retained nodes.
    """
    keep = nodes_on_causal_paths(G, start_nodes, end_nodes)
    return G.subgraph(keep)
