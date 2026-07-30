"""Helpers for optional dependencies.

Some features rely on packages that are not installed by default (for example
``indra-cogex``, which is not published on PyPI). Import sites guard these
dependencies so that ``import causomic`` and unrelated functionality keep
working when they are absent; the helpers here provide a clear, actionable
error if such a feature is actually invoked without its dependency installed.
"""

COGEX_INSTALL_HINT = (
    "This feature requires the optional 'indra-cogex' dependency, which is not "
    "available on PyPI. Install it from source with:\n"
    "    pip install git+https://github.com/gyorilab/indra_cogex.git"
)

DAGMA_INSTALL_HINT = (
    "This feature requires the optional 'dagma' dependency. Install it with:\n"
    "    pip install causomic[dagma]\n"
    "or directly with:\n"
    "    pip install dagma"
)


def missing_cogex(*_args, **_kwargs):
    """Placeholder for indra-cogex symbols; raises a helpful error when used."""
    raise ImportError(COGEX_INSTALL_HINT)


def missing_dagma(*_args, **_kwargs):
    """Placeholder for dagma symbols; raises a helpful error when used."""
    raise ImportError(DAGMA_INSTALL_HINT)
