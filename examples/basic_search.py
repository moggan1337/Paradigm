"""
Example: Basic NAS Search

This example demonstrates how to use Paradigm for basic
neural architecture search.
"""

import paradigm
from paradigm.search_spaces import CNNSearchSpace
from paradigm.controllers import RLController
from paradigm.optimizers import StandardOptimizer


def main():
    """Run basic NAS search."""
    print("Paradigm: Basic NAS Search Example")
    print("=" * 50)
    
    # Create search space
    print("\n1. Creating CNN search space...")
    search_space = CNNSearchSpace(
        max_nodes=5,
        num_operations=6,
        channels=[16, 32, 64],
        num_cells=6
    )
    print(f"   Search space created with max_nodes={search_space.max_nodes}")
    
    # Create controller
    print("\n2. Creating RL controller...")
    controller = RLController(
        search_space=search_space,
        hidden_size=64,
        num_layers=2
    )
    print("   Controller created")
    
    # Create optimizer
    print("\n3. Creating optimizer...")
    optimizer = StandardOptimizer(
        search_space=search_space,
        controller=controller,
        epochs_per_trial=5,  # Few epochs for quick demo
        batch_size=64
    )
    print("   Optimizer created")
    
    # Run search
    print("\n4. Running search (10 trials)...")
    result = optimizer.search(
        num_trials=10,
        epochs_per_trial=5,
        dataset="cifar10"
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Search Results")
    print("=" * 50)
    print(result.summary())
    
    # Save result
    result.save("results/basic_search_result.json")
    print("\nResults saved to results/basic_search_result.json")


if __name__ == "__main__":
    main()
