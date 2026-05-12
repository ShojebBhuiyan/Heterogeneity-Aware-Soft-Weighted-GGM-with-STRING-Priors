"""KG wrappers."""

from pgm.kg.prior_matrix import edges_to_sparse_prior, prior_to_networkx
from pgm.kg.string_api import StringAPIClient, fetch_string_for_genes

__all__ = [
    "StringAPIClient",
    "edges_to_sparse_prior",
    "fetch_string_for_genes",
    "prior_to_networkx",
]
