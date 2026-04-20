"""
Command-line interface for Paradigm.
"""

import argparse
import sys
from pathlib import Path

from paradigm import (
    CNNSearchSpace,
    RLController,
    DARTSOptimizer,
    MultiObjectiveOptimizer,
    Objective,
    ParameterCountMetric,
    LatencyMetric,
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Paradigm: AutoML with Neural Architecture Search"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Run architecture search")
    search_parser.add_argument(
        "--algorithm",
        type=str,
        default="rl",
        choices=["rl", "darts", "evolution", "bayesian", "multi"],
        help="Search algorithm"
    )
    search_parser.add_argument(
        "--search-space",
        type=str,
        default="cnn",
        choices=["cnn", "rnn", "transformer"],
        help="Search space"
    )
    search_parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials"
    )
    search_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Epochs per trial"
    )
    search_parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show information")
    info_parser.add_argument(
        "--search-space",
        type=str,
        help="Show search space info"
    )
    
    args = parser.parse_args()
    
    if args.command == "search":
        run_search(args)
    elif args.command == "info":
        show_info(args)
    else:
        parser.print_help()


def run_search(args):
    """Run architecture search."""
    print(f"Running {args.algorithm} search on {args.search_space} space")
    print(f"Trials: {args.trials}, Epochs: {args.epochs}")
    
    # Create search space
    if args.search_space == "cnn":
        search_space = CNNSearchSpace()
    else:
        raise NotImplementedError(f"Search space {args.search_space} not implemented")
    
    # Create optimizer
    if args.algorithm == "darts":
        optimizer = DARTSOptimizer(search_space)
        result = optimizer.search(num_epochs=args.epochs, dataset="cifar10")
    elif args.algorithm == "multi":
        optimizer = MultiObjectiveOptimizer(
            search_space,
            objectives=[
                Objective("accuracy", goal="maximize"),
                Objective("latency", goal="minimize", metric_fn=LatencyMetric()),
                Objective("params", goal="minimize", metric_fn=ParameterCountMetric()),
            ]
        )
        result = optimizer.search(
            num_generations=args.trials // 50,
            epochs_per_trial=args.epochs
        )
    else:
        from paradigm import StandardOptimizer
        optimizer = StandardOptimizer(search_space)
        result = optimizer.search(
            num_trials=args.trials,
            epochs_per_trial=args.epochs
        )
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    result.save(output_path / "result.json")
    
    print(f"\nSearch completed!")
    print(f"Best accuracy: {result.best_accuracy:.2f}%")
    print(f"Results saved to {output_path}")


def show_info(args):
    """Show information."""
    if args.search_space:
        print(f"Search space: {args.search_space}")
        # Show search space info
    else:
        print("Paradigm: AutoML with Neural Architecture Search")
        print("\nAvailable commands:")
        print("  search   - Run architecture search")
        print("  info     - Show information")


if __name__ == "__main__":
    main()
