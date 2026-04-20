"""
Example: Performance Prediction

This example demonstrates how to use zero-cost metrics
to predict architecture performance without training.
"""

import paradigm
from paradigm.search_spaces import CNNSearchSpace
from paradigm.metrics import PerformancePredictor, SynFlowMetric, GradNormMetric


def main():
    """Run performance prediction demo."""
    print("Paradigm: Performance Prediction Example")
    print("=" * 50)
    
    # Create search space
    print("\n1. Creating search space...")
    search_space = CNNSearchSpace(
        max_nodes=5,
        num_operations=6
    )
    
    # Sample architectures
    print("\n2. Sampling architectures...")
    architectures = [search_space.sample() for _ in range(10)]
    print(f"   Sampled {len(architectures)} architectures")
    
    # Create predictor
    print("\n3. Creating performance predictor...")
    predictor = PerformancePredictor(metrics=[
        SynFlowMetric(),
        GradNormMetric(),
    ])
    print("   Predictor created with SynFlow and GradNorm metrics")
    
    # Score architectures
    print("\n4. Scoring architectures (zero-cost)...")
    scores = predictor.batch_score(
        architectures,
        input_shape=(1, 3, 32, 32)
    )
    
    # Rank architectures
    print("\n5. Ranking architectures...")
    ranked = predictor.rank(
        architectures,
        input_shape=(1, 3, 32, 32)
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Performance Prediction Results")
    print("=" * 50)
    print("\nTop 5 Architectures (by predicted performance):")
    
    for rank, (idx, score) in enumerate(ranked[:5]):
        print(f"  {rank+1}. Architecture {idx}: score = {score:.4f}")
    
    print("\nAll Scores:")
    for i, score in enumerate(scores):
        print(f"  Architecture {i}: {score:.4f}")
    
    print("\nNote: Zero-cost metrics don't require training!")
    print("This enables rapid architecture screening.")


if __name__ == "__main__":
    main()
