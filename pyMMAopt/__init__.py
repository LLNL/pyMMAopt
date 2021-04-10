from .__about__ import __version__
from .mma import MMAClient
from .mma_solver import MMASolver
from .constraints import ReducedInequality

__all__ = [
    "__version__",
    "MMAClient",
]
