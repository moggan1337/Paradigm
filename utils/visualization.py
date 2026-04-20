"""
Visualization utilities for NAS results.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..core.search_space import Architecture
from ..core.optimizer import OptimizationResult


def plot_pareto_front(
    pareto_front: List[Architecture],
    objectives: List[str],
    fitness_vectors: List[Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Pareto Front"
) -> plt.Figure:
    """
    Plot Pareto front for multi-objective optimization.
    
    Args:
        pareto_front: List of Pareto-optimal architectures
        objectives: List of objective names
        fitness_vectors: Fitness values for each architecture
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if len(objectives) < 2:
        raise ValueError("Need at least 2 objectives for Pareto plot")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract values for each objective
    obj1, obj2 = objectives[0], objectives[1]
    
    x_values = [fv[obj1] for fv in fitness_vectors]
    y_values = [fv[obj2] for fv in fitness_vectors]
    
    # Plot Pareto front
    # Sort by first objective
    sorted_pairs = sorted(zip(x_values, y_values))
    sorted_x = [x for x, y in sorted_pairs]
    sorted_y = [y for x, y in sorted_pairs]
    
    ax.plot(sorted_x, sorted_y, 'b-', alpha=0.5, linewidth=2)
    ax.scatter(sorted_x, sorted_y, c='blue', s=100, zorder=5)
    
    ax.set_xlabel(obj1.capitalize(), fontsize=12)
    ax.set_ylabel(obj2.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_search_history(
    result: OptimizationResult,
    save_path: Optional[str] = None,
    title: str = "Search History"
) -> plt.Figure:
    """
    Plot search history showing accuracy over trials.
    
    Args:
        result: Optimization result
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy over trials
    ax1 = axes[0]
    trials = range(len(result.all_rewards))
    ax1.plot(trials, result.all_rewards, 'b-', alpha=0.5, label='All Trials')
    
    # Add moving average
    window = min(10, len(result.all_rewards))
    if window > 1:
        moving_avg = np.convolve(
            result.all_rewards,
            np.ones(window) / window,
            mode='valid'
        )
        ax1.plot(
            range(window - 1, len(result.all_rewards)),
            moving_avg,
            'r-',
            linewidth=2,
            label=f'Moving Avg (window={window})'
        )
    
    # Mark best
    best_idx = np.argmax(result.all_rewards)
    ax1.scatter([best_idx], [result.all_rewards[best_idx]],
                c='green', s=200, marker='*', zorder=10, label='Best')
    
    ax1.set_xlabel('Trial', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Over Trials', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training history (if available)
    ax2 = axes[1]
    
    if result.history:
        if 'val_accuracy' in result.history[0].get('metrics', {}):
            val_accs = [h['metrics']['val_accuracy'] for h in result.history]
            ax2.plot(range(len(val_accs)), val_accs, 'g-', label='Validation Accuracy')
        
        if 'train_loss' in result.history[0]:
            train_losses = [h['train_loss'] for h in result.history]
            ax2.plot(range(len(train_losses)), train_losses, 'r-', label='Train Loss')
    
    ax2.set_xlabel('Epoch / Trial', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Training Metrics', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_architecture_dag(
    architecture: Architecture,
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot architecture as a DAG.
    
    Args:
        architecture: Architecture to plot
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed, skipping DAG plot")
        return None
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for cell in architecture.cells:
        G.add_node(cell.node_index)
    
    # Add edges
    for edge in architecture.edges:
        G.add_edge(edge.source, edge.target, 
                   label=edge.operation.op_type.value)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000,
                           node_color='lightblue', edgecolors='black')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           arrowsize=20, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    
    # Edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
    
    ax.set_title(title or f"Architecture: {architecture.name}", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comparison(
    results: Dict[str, OptimizationResult],
    metric: str = 'accuracy',
    save_path: Optional[str] = None,
    title: str = "Algorithm Comparison"
) -> plt.Figure:
    """
    Compare multiple optimization results.
    
    Args:
        results: Dictionary of results to compare
        metric: Metric to compare
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = []
    best_values = []
    
    for name, result in results.items():
        algorithms.append(name)
        
        if metric == 'accuracy':
            best_values.append(result.best_accuracy)
        elif metric == 'time':
            best_values.append(result.search_time)
        else:
            # Check metadata
            best_values.append(result.metadata.get(metric, 0.0))
    
    # Bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))
    bars = ax.bar(algorithms, best_values, color=colors)
    
    # Add value labels
    for bar, value in zip(bars, best_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_progression(
    architectures: List[Architecture],
    save_path: Optional[str] = None,
    title: str = "Architecture Progression"
) -> plt.Figure:
    """
    Plot the progression of architectures during search.
    
    Args:
        architectures: List of architectures
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, arch in enumerate(architectures[:10]):
        ax = axes[i]
        
        # Simple representation of architecture complexity
        n_edges = len(arch.edges)
        n_nodes = len(arch.cells)
        depth = arch.depth()
        
        # Bar chart of properties
        properties = ['Edges', 'Nodes', 'Depth']
        values = [n_edges, n_nodes, depth]
        
        ax.bar(properties, values, color=['blue', 'green', 'red'], alpha=0.7)
        ax.set_title(f"Arch {i+1}", fontsize=10)
        ax.set_ylim(0, max(values) * 1.2)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
