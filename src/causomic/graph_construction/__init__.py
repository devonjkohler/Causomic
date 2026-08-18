"""Construct a causal graph from prior knowledge and experimental data.

Three stages, one per subpackage, run in order:

1. :mod:`~causomic.graph_construction.prior_extraction` -- pull a candidate edge
   set out of INDRA, from either a local ``networkx`` graph (default) or a live
   Neo4j-CoGEx instance.
2. :mod:`~causomic.graph_construction.posterior_estimation` -- learn which of
   those candidate edges the data supports, by constrained hill climbing or
   DAGMA.
3. :mod:`~causomic.graph_construction.ci_repair` -- test the learned graph's
   implied conditional independences and repair the failures, usually by
   identifying a missing confounder.

:mod:`causomic.network` provides one entry point per stage
(:func:`~causomic.network.extract_indra_prior`,
:func:`~causomic.network.estimate_posterior_dag`,
:func:`~causomic.network.repair_confounding`) and is the recommended way in.
Import from the subpackages directly when you need a single step, a
non-default backend, or a component the entry points don't expose.
"""

from causomic.graph_construction import (
    ci_repair,
    posterior_estimation,
    prior_extraction,
)

__all__ = [
    "ci_repair",
    "posterior_estimation",
    "prior_extraction",
]
