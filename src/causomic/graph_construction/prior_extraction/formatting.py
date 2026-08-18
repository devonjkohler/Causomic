"""Conversion of raw INDRA query results into tidy DataFrames.

Every Neo4j query builder in
:mod:`causomic.graph_construction.prior_extraction.neo4j_backend` returns a list
of INDRA ``Relation`` objects. :func:`format_query_results` is the single point
where those become the ``source``/``target``/``evidence_count`` DataFrame that
the rest of causomic -- prior assembly, scoring, DAGMA -- consumes.
"""

import json
from typing import List

import pandas as pd
from indra.databases.chebi_client import get_chebi_name_from_id
from indra.databases.go_client import get_go_label
from indra.databases.hgnc_client import get_hgnc_name
from indra.databases.mesh_client import get_mesh_name
from indra.databases.uniprot_client import get_gene_name
from indra.statements import Statement

# indra-cogex is an optional dependency (see causomic._optional)
try:
    from indra_cogex.representation import Relation
except ImportError:
    Relation = ()  # isinstance() fallback; always False, unreachable without cogex


def format_query_results(queries: List[Statement]) -> pd.DataFrame:
    """
    Format INDRA statements into a standardized pandas DataFrame.

    Processes INDRA Relation statements to extract key information including
    entity IDs, human-readable names, relationship types, evidence counts,
    belief scores, and publication references. Handles various data structures
    and missing fields gracefully.

    Parameters
    ----------
    queries : List[Statement]
        List of INDRA Relation statements to format

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - source_id: Source entity identifier
        - source: Human-readable source name
        - relation: Relationship/statement type
        - target_id: Target entity identifier
        - target: Human-readable target name
        - stmt_hash: Unique statement hash (if available)
        - evidence_count: Number of supporting evidences
        - belief: Belief score (0-1, if available)
        - source_counts: Aggregated source counts
        - pmid: Publication references dictionary

    Notes
    -----
    Only processes statements where both source and target namespaces
    are supported by the ID mapping functions. Unsupported statements
    are silently filtered out.

    Examples
    --------
    >>> statements = compound_query(compounds=compounds, client=client)
    >>> df = format_query_results(statements)
    >>> print(df.columns.tolist())
    ['source_id', 'source', 'relation', 'target_id', 'target', ...]
    """
    columns = [
        "source_id",
        "source",
        "relation",
        "target_id",
        "target",
        "stmt_hash",
        "evidence_count",
        "belief",
        "source_counts",
        "pmid",
    ]

    # Mapping from namespace to name resolution function
    id_mapper: Dict[str, callable] = {
        "HGNC": get_hgnc_name,
        "CHEBI": get_chebi_name_from_id,
        "MESH": get_mesh_name,
        "UP": get_gene_name,
        "GO": get_go_label,
    }

    # Custom field mapping for different relation types
    relation_mapper: Dict[str, str] = {"gene_disease_association": "papers"}

    # Process INDRA statements and extract relevant data
    rows = []
    for relation in queries:
        # Filter for supported relation types and namespaces
        if (
            isinstance(relation, Relation)
            and relation.source_ns in id_mapper
            and relation.target_ns in id_mapper
        ):

            # Extract statement type or relation type
            stmt_type = relation.data.get("stmt_type") or relation.rel_type

            # Get appropriate evidence count field
            evidence_field = relation_mapper.get(relation.rel_type, "evidence_count")
            evidence_count = relation.data.get(evidence_field)

            # Extract publication references from statement JSON
            pmid = None
            if "stmt_json" in relation.data:
                try:
                    stmt_json = json.loads(relation.data["stmt_json"])
                    if (
                        stmt_json.get("evidence")
                        and len(stmt_json["evidence"]) > 0
                        and "text_refs" in stmt_json["evidence"][0]
                    ):
                        pmid = stmt_json["evidence"][0]["text_refs"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    pmid = None

            # Calculate source counts sum
            source_counts = None
            if "source_counts" in relation.data:
                try:
                    source_counts = json.loads(relation.data["source_counts"])
                except (json.JSONDecodeError, TypeError):
                    source_counts = None

            rows.append(
                (
                    relation.source_id,
                    id_mapper[relation.source_ns](relation.source_id),
                    stmt_type,
                    relation.target_id,
                    id_mapper[relation.target_ns](relation.target_id),
                    relation.data.get("stmt_hash"),
                    evidence_count,
                    relation.data.get("belief"),
                    source_counts,
                    pmid,
                )
            )

    return pd.DataFrame(rows, columns=columns)
