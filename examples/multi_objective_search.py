"""
Example: Multi-Objective NAS Search

This example demonstrates multi-objective architecture search
optimizing accuracy, latency, and model size simultaneously.
"""

import paradigm
from paradigm.search_spaces import CNNSearchSpace
from paradigm.optimizers import MultiObjectiveOptimizer
from paradigm.optimizers.multi_objective_optimizer import Objective
from paradigm.metrics import LatencyMetric, ParameterCountMetric


def main():
    """Run multi-objective search."""
    print("Paradigm: Multi-Objective NAS Search Example")
    print("=" * 50)
    
    # Create search space
    print("\n1. Creating CNN search space...")
    search_space = CNNSearchSpace(
        max_nodes=5,
        num_operations=6,
        channels=[16, 32, 64],
        num_cells=6
    )
    
    # Define objectives
    print("\n2. Defining objectives...")
    objectives = [
        Objective("accuracy", weight=1.0, goal="maximize"),
        Objective("latency", weight=0.5, goal="minimize", metric_fn=LatencyMetric()),
        Objective("params", weight=0.3, goal="minimize", metric_fn=ParameterCountMetric()),
    ]
    print("   - Accuracy (maximize)")
    print("   - Latency (minimize)")
    print("   - Parameters (minimize)")
    
    # Create optimizer
    print("\n3. Creating multi-objective optimizer...")
    optimizer = MultiObjectiveOptimizer(
        search_space=search_space,
        objectives=objectives,
        algorithm="nsga2",
        population_size=20,
        num_generations=20
    )
    print("   Optimizer created (NSGA-II)")
    
    # Run search
    print("\n4. Running multi-objective search...")
    result = optimizer.search(
        num_generations=20,
        epochs_per_trial=10
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Multi-Objective Search Results")
    print("=" * 50)
    print(f"Pareto Front Size: {len(result.all_architectures)}")
    print(f"Search Time: {result.search_time:.2f}s")
    
    if result.all_architectures:
        print("\nPareto-Optimal Architectures:")
        for i, arch in enumerate(result.all_architectures[:5]):
            print(f"\n  Architecture {i+1}:")
            print(f"    Accuracy: {result.all_rewards[i]:.2f}%")
    
    # Save result
    result.save("results/multi_objective_result.json")
    print("\nResults saved to results/multi_objective_result.json")


if __name__ == "__main__":
    main()
