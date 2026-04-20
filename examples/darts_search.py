"""
Example: DARTS Search

This example demonstrates how to use DARTS for
differentiable architecture search.
"""

import paradigm
from paradigm.search_spaces import CNNSearchSpace
from paradigm.optimizers import DARTSOptimizer


def main():
    """Run DARTS search."""
    print("Paradigm: DARTS Search Example")
    print("=" * 50)
    
    # Create search space
    print("\n1. Creating CNN search space for DARTS...")
    search_space = CNNSearchSpace(
        max_nodes=5,
        num_operations=6,
        channels=[16, 32, 64],
        num_cells=8
    )
    
    # Create DARTS optimizer
    print("\n2. Creating DARTS optimizer...")
    optimizer = DARTSOptimizer(
        search_space=search_space,
        unrolled_steps=1,
        weight_learning_rate=0.025,
        arch_learning_rate=0.0003,
        first_order=False
    )
    print("   DARTS optimizer created")
    
    # Run search
    print("\n3. Running DARTS search (20 epochs)...")
    result = optimizer.search(
        num_epochs=20,
        dataset="cifar10"
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("DARTS Search Results")
    print("=" * 50)
    print(f"Best Accuracy: {result.best_accuracy:.2f}%")
    print(f"Search Time: {result.search_time:.2f}s")
    
    if result.best_architecture:
        print(f"\nBest Architecture:")
        print(result.best_architecture.visualize())
    
    # Save result
    result.save("results/darts_search_result.json")
    print("\nResults saved to results/darts_search_result.json")


if __name__ == "__main__":
    main()
