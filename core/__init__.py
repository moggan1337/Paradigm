"""
Core module for Paradigm AutoML framework.

Contains base classes for search spaces, controllers, and optimizers.
"""

from .search_space import SearchSpace, Architecture, Operation
from .controller import Controller
from .optimizer import Optimizer, OptimizationResult

__all__ = [
    "SearchSpace",
    "Architecture",
    "Operation",
    "Controller",
    "Optimizer",
    "OptimizationResult",
]
