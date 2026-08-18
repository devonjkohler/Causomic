"""Find where an estimated DAG contradicts the data.

A learned DAG implies a set of conditional independences. Testing them against
the data is the cheapest available check on the structure: every one that fails
is a place the graph is wrong. The most common cause in biological networks is
an unmeasured common cause, which is why a failure here feeds into
:mod:`~causomic.graph_construction.ci_repair.confounders` rather than simply
being reported.

:func:`convert_to_y0_graph` gets an estimated edge list into the y0
``NxMixedGraph`` form the testing machinery needs; :func:`find_failed_tests`
runs the tests and returns the violations.
"""

import networkx as nx
import pandas as pd
from sklearn.impute import KNNImputer
from y0.algorithm.falsification import get_graph_falsifications
from y0.graph import NxMixedGraph


def convert_to_y0_graph(posterior_dag):
    """
    Convert the posterior DAG to a y0 graph format.
    """

    # Confirm index is fine
    posterior_dag = posterior_dag.reset_index(drop=True)

    # Construct NetworkX DiGraph from posterior_dag
    all_nodes = set(posterior_dag["source"]).union(set(posterior_dag["target"]))

    nx_dag = nx.DiGraph()
    for i in range(len(posterior_dag)):
        nx_dag.add_edge(posterior_dag.loc[i, "source"], posterior_dag.loc[i, "target"])

    obs_nodes = all_nodes

    # Set all nodes as observed
    attrs = {
        node: (True if node not in obs_nodes and node != "\\n" else False) for node in all_nodes
    }
    nx.set_node_attributes(nx_dag, attrs, name="hidden")

    # Use y0 to build ADMG
    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(nx_dag, "hidden")

    return y0_graph


def find_failed_tests(
    posterior_dag: NxMixedGraph,
    data: pd.DataFrame,
    max_conditional: int = 2,
    significance_level: float = 0.05,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Run the DAG's implied conditional-independence tests and return the failures.

    Parameters
    ----------
    posterior_dag : NxMixedGraph
        Estimated graph whose implied independences are to be tested.
    data : pd.DataFrame
        Observational data over the graph's nodes. Missing values are KNN-imputed
        first, because the falsification tests cannot accept NaNs; the imputed
        frame is returned so callers score confounder candidates against exactly
        the values the tests saw.
    max_conditional : int, default=2
        Maximum size of the conditioning set for each test.
    significance_level : float, default=0.05
        Threshold applied to multiplicity-adjusted p-values.
    n_neighbors : int, default=5
        Neighbor count for the KNN imputer.

    Returns
    -------
    (failed_tests, imputed_data) : tuple[pd.DataFrame, pd.DataFrame]
        ``failed_tests`` has one row per rejected independence, with 'left',
        'right', and 'given' columns. Tests with an empty conditioning set are
        excluded: a bare marginal dependence between two nodes says nothing
        about a *missing* confounder, only that they are related somehow.
    """
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = pd.DataFrame(
        knn_imputer.fit_transform(data), index=data.index, columns=data.columns
    )

    falsification_results = get_graph_falsifications(
        posterior_dag,
        imputed_data,
        max_given=max_conditional,
        method="pearson",
        verbose=True,
        significance_level=significance_level,
    ).evidence

    failed_tests = falsification_results.loc[
        (falsification_results["p_adj_significant"] == True)  # noqa: E712
        & (falsification_results["given"] != "")
    ].reset_index(drop=True)

    return failed_tests, imputed_data
