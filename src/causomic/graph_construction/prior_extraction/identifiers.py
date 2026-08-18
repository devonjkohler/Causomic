"""Resolution of entity names and identifiers to INDRA CURIEs.

INDRA-CoGEx addresses every node by a CURIE -- a ``(namespace, id)`` pair such
as ``("hgnc", "391")``. Callers, however, normally hold gene symbols, UniProt
accessions, or chemical names. :func:`resolve_curies` is the single conversion
point between the two, used by every Neo4j query builder in
:mod:`causomic.graph_construction.prior_extraction.neo4j_backend`.
"""

from typing import Iterable, List, Tuple

from indra.databases.chebi_client import get_chebi_id_from_name
from indra.databases.hgnc_client import get_hgnc_id
from protmapper import uniprot_client

SUPPORTED_ID_TYPES = ("gene", "uniprot", "chebi")


def resolve_curies(ids: Iterable[str], id_type: str) -> List[Tuple[str, str]]:
    """Resolve entity names or identifiers to standardized INDRA CURIEs.

    Parameters
    ----------
    ids : Iterable[str]
        Entity names or identifiers to resolve.
    id_type : str
        How to interpret ``ids``. One of:

        - ``"gene"``: HGNC gene symbols (e.g. ``"EGFR"``) -> ``("hgnc", <id>)``
        - ``"uniprot"``: UniProt accessions (e.g. ``"P31749"``) -> ``("hgnc", <id>)``
        - ``"chebi"``: chemical names or ChEBI IDs -> ``("chebi", <id>)``

    Returns
    -------
    List[Tuple[str, str]]
        ``(namespace, identifier)`` pairs for every input that resolved.
        Inputs that fail to resolve are dropped silently rather than raising,
        so the returned list may be shorter than ``ids`` -- callers passing a
        whole dataset's column of protein names should expect some loss.
        Duplicates are collapsed, so order is not preserved.

    Raises
    ------
    ValueError
        If ``id_type`` is not one of :data:`SUPPORTED_ID_TYPES`.

    Examples
    --------
    >>> resolve_curies(["EGFR", "TP53"], "gene")
    [('hgnc', '3236'), ('hgnc', '11998')]
    >>> resolve_curies(["P31749"], "uniprot")
    [('hgnc', '391')]
    """
    if id_type not in SUPPORTED_ID_TYPES:
        raise ValueError(f"Unsupported id_type: {id_type!r}. Supported types: {SUPPORTED_ID_TYPES}")

    resolved = set()

    if id_type == "gene":
        namespace = "hgnc"
        for entity_id in ids:
            hgnc_id = get_hgnc_id(entity_id)
            if hgnc_id:
                resolved.add(hgnc_id)

    elif id_type == "uniprot":
        namespace = "hgnc"
        # set() first: UniProt lookups are the slowest of the three and datasets
        # routinely repeat accessions across rows.
        for uniprot_id in set(ids):
            hgnc_id = uniprot_client.get_hgnc_id(uniprot_id)
            if hgnc_id:
                resolved.add(hgnc_id)

    else:  # chebi
        namespace = "chebi"
        for compound in ids:
            chebi_id = get_chebi_id_from_name(compound)
            if chebi_id:
                resolved.add(chebi_id)

    return [(namespace, entity_id) for entity_id in resolved if entity_id is not None]
