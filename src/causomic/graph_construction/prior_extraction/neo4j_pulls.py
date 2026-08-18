"""High-level "pull a prior network from INDRA-CoGEx" convenience wrappers.

Each function here is a three-step composition -- resolve identifiers, run a
Cypher query, format the result -- packaged so callers never have to hold a
CURIE. For anything not covered by these presets, compose
:mod:`~causomic.graph_construction.prior_extraction.identifiers`,
:mod:`~causomic.graph_construction.prior_extraction.neo4j_backend`, and
:mod:`~causomic.graph_construction.prior_extraction.formatting` directly.
"""

from typing import List

import pandas as pd

# indra-cogex is an optional dependency (see causomic._optional)
try:
    from indra_cogex.client import Neo4jClient
except ImportError:
    from causomic._optional import missing_cogex

    Neo4jClient = missing_cogex

from causomic.graph_construction.prior_extraction.formatting import format_query_results
from causomic.graph_construction.prior_extraction.identifiers import resolve_curies
from causomic.graph_construction.prior_extraction.neo4j_backend import (
    compound_query,
    get_neighbor_network,
    mesh_query,
)


def pull_compound_data(compound_ids: List[str], client: Neo4jClient) -> pd.DataFrame:
    """
    Retrieve and format compound-gene interaction data from INDRA database.

    High-level function that combines ID resolution, database querying, and
    result formatting for compound data. Handles the complete workflow from
    compound names/IDs to structured relationship data.

    Parameters
    ----------
    compound_ids : List[str]
        List of compound identifiers or names (e.g., ChEBI IDs, compound names)
    client : Neo4jClient
        Neo4j client instance for database connectivity

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame containing compound-gene relationships with
        standardized columns including source/target information, evidence
        counts, and publication references

    Examples
    --------
    >>> compounds = ["glucose", "CHEBI:15377", "caffeine"]
    >>> df = pull_compound_data(compounds, neo4j_client)
    >>> print(f"Found {len(df)} compound-gene interactions")
    >>> print(df[['source', 'relation', 'target']].head())
    """
    query_ids = resolve_curies(compound_ids, "chebi")
    query_results = compound_query(compounds=query_ids, client=client)
    data = format_query_results(query_results)
    return data


def pull_downstream_network(gene_ids: List[str], client: Neo4jClient) -> pd.DataFrame:
    """
    Retrieve downstream gene network from INDRA database.

    Finds all genes that are downstream targets (regulated by) the specified
    input genes. Uses the neighbor network functionality with downstream-only
    filtering and evidence count thresholding.

    Parameters
    ----------
    gene_ids : List[str]
        List of gene identifiers or names (e.g., HGNC symbols, gene names)
    client : Neo4jClient
        Neo4j client instance for database connectivity

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame containing downstream gene relationships with
        evidence support and relationship metadata

    Examples
    --------
    >>> regulators = ["TP53", "EGFR", "MYC"]
    >>> downstream = pull_downstream_network(regulators, neo4j_client)
    >>> print(f"Found {len(downstream)} downstream relationships")
    """
    query_ids = resolve_curies(gene_ids, "gene")

    query_results = get_neighbor_network(
        nodes=query_ids, client=client, upstream=False, downstream=True, minimum_evidence_count=1
    )

    data = format_query_results(query_results)
    return data


def pull_upstream_network(gene_ids: List[str], client: Neo4jClient) -> pd.DataFrame:
    """
    Retrieve upstream gene network from INDRA database.

    Finds all genes that are upstream regulators (regulate) the specified
    input genes. Useful for identifying potential causal factors and
    regulatory mechanisms affecting genes of interest.

    Parameters
    ----------
    gene_ids : List[str]
        List of gene identifiers or names for which to find upstream regulators
    client : Neo4jClient
        Neo4j client instance for database connectivity

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame containing upstream regulatory relationships with
        evidence support and relationship metadata

    Raises
    ------
    ValueError
        If gene_ids is empty or contains invalid identifiers

    Examples
    --------
    >>> targets = ["BRCA1", "BRCA2", "ATM"]
    >>> upstream = pull_upstream_network(targets, neo4j_client)
    >>> print(f"Found {len(upstream)} upstream regulators")
    >>> print(upstream.groupby('source').size().head())

    Notes
    -----
    Only retrieves relationships with at least one supporting evidence.
    Downstream relationships are excluded to focus on regulatory inputs.
    """
    query_ids = resolve_curies(gene_ids, "gene")

    query_results = get_neighbor_network(
        nodes=query_ids, client=client, upstream=True, downstream=False, minimum_evidence_count=1
    )

    data = format_query_results(query_results)
    return data


def pull_mesh_data(mesh_ids: List[str], client: Neo4jClient) -> pd.DataFrame:
    """
    Retrieve and format gene-disease association data using MeSH terms.

    High-level function that combines MeSH ID formatting, database querying,
    and result formatting for disease-gene associations. Provides a complete
    workflow from MeSH disease terms to structured relationship data.

    Parameters
    ----------
    mesh_ids : List[str]
        List of MeSH (Medical Subject Headings) identifier strings for diseases
        or biological processes, e.g., ["D000544", "D001943"]
    client : Neo4jClient
        Neo4j client instance for database connectivity

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame containing gene-disease associations with
        evidence support, belief scores, and publication references

    Examples
    --------
    >>> diseases = ["D000544", "D001943", "D002292"]  # Alzheimer's, Breast Cancer, Cardiomyopathy
    >>> associations = pull_mesh_data(diseases, neo4j_client)
    >>> print(f"Found {len(associations)} gene-disease associations")
    >>> top_genes = associations.groupby('source')['evidence_count'].sum().sort_values(ascending=False)
    >>> print("Top associated genes:", top_genes.head())

    Notes
    -----
    Automatically converts MeSH IDs to proper namespace format before querying.
    Only returns associations with human gene/protein entities.
    """
    query_ids = [("MESH", mesh_id) for mesh_id in mesh_ids]
    query_results = mesh_query(query_ids=query_ids, client=client)
    data = format_query_results(query_results)
    return data


def pull_go_data(go_ids: List[str], client: Neo4jClient) -> pd.DataFrame:

    query_ids = [("GO", go_id) for go_id in go_ids]
    query_results = mesh_query(query_ids=query_ids, client=client)
    data = format_query_results(query_results)
    return data
