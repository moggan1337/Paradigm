"""
Paradigm: AutoML with Neural Architecture Search

A comprehensive framework for automated machine learning with
state-of-the-art neural architecture search algorithms.
"""

__version__ = "0.1.0"
__author__ = "Paradigm Contributors"

from .core.search_space import SearchSpace, Architecture
from .core.controller import Controller
from .core.optimizer import Optimizer, OptimizationResult
from .search_spaces import CNNSearchSpace, RNNSearchSpace, TransformerSearchSpace
from .controllers import RLController, EvolutionController
from .optimizers import (
    EvolutionaryOptimizer,
    DARTSOptimizer,
    BayesianOptimizer,
    MultiObjectiveOptimizer
)
from .metrics import PerformancePredictor
from .transfer import TransferLearning

# Main search function
def search(
    search_space: SearchSpace,
    controller: Controller = None,
    optimizer: Optimizer = None,
    num_trials: int = 100,
    epochs_per_trial: int = 50,
    dataset: str = "cifar10",
    **kwargs
) -> OptimizationResult:
    """
    Main entry point for architecture search.
    
    Args:
        search_space: Search space definition
        controller: RL controller (for RL-based search)
        optimizer: Optimizer (alternative to controller)
        num_trials: Number of architectures to evaluate
        epochs_per_trial: Training epochs per architecture
        dataset: Dataset name
        **kwargs: Additional arguments
        
    Returns:
        OptimizationResult with best architecture and metrics
    """
    from .core.optimizer import StandardOptimizer
    
    if optimizer is None:
        optimizer = StandardOptimizer(
            search_space=search_space,
            controller=controller,
            **kwargs
        )
    
    return optimizer.search(
        num_trials=num_trials,
        epochs_per_trial=epochs_per_trial,
        dataset=dataset,
        **kwargs
    )


__all__ = [
    # Core
    "SearchSpace",
    "Architecture",
    "Controller",
    "Optimizer",
    "OptimizationResult",
    # Search spaces
    "CNNSearchSpace",
    "RNNSearchSpace",
    "TransformerSearchSpace",
    # Controllers
    "RLController",
    "EvolutionController",
    # Optimizers
    "EvolutionaryOptimizer",
    "DARTSOptimizer",
    "BayesianOptimizer",
    "MultiObjectiveOptimizer",
    # Utilities
    "PerformancePredictor",
    "TransferLearning",
    # Main function
    "search",
]
