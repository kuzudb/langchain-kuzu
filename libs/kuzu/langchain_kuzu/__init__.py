from importlib import metadata

from langchain_kuzu.chains.graph_qa.kuzu import KuzuQAChain
from langchain_kuzu.graphs.kuzu_graph import KuzuGraph

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "KuzuQAChain",
    "KuzuGraph",
    "__version__",
]
