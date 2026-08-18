"""Tests for prior_extraction.identifiers.resolve_curies.

resolve_curies replaces the old get_id/get_ids pair, which overlapped on the
"gene" branch and differed on the other (uniprot vs chebi). get_id also left
hgnc_ids unbound for any unrecognized id_type, so an unsupported type raised
NameError instead of a usable error; the last test here pins the replacement.

The gene and uniprot branches call into INDRA's bundled HGNC/UniProt tables,
which ship with the package and need no network access.
"""

import importlib

import pytest

identifiers = importlib.import_module("causomic.graph_construction.prior_extraction.identifiers")


def test_resolve_curies_gene_symbols():
    curies = identifiers.resolve_curies(["EGFR", "TP53"], "gene")
    assert set(curies) == {("hgnc", "3236"), ("hgnc", "11998")}


def test_resolve_curies_uniprot_accessions():
    # P31749 is AKT1 -> HGNC:391
    assert identifiers.resolve_curies(["P31749"], "uniprot") == [("hgnc", "391")]


def test_resolve_curies_drops_unresolvable_inputs():
    curies = identifiers.resolve_curies(["EGFR", "NOT_A_REAL_GENE_XYZ"], "gene")
    assert curies == [("hgnc", "3236")]


def test_resolve_curies_collapses_duplicates():
    assert identifiers.resolve_curies(["EGFR", "EGFR"], "gene") == [("hgnc", "3236")]


def test_resolve_curies_empty_input():
    assert identifiers.resolve_curies([], "gene") == []


def test_resolve_curies_rejects_unsupported_type():
    # The old get_id raised NameError here (hgnc_ids was never assigned).
    with pytest.raises(ValueError, match="Unsupported id_type"):
        identifiers.resolve_curies(["EGFR"], "ensembl")
