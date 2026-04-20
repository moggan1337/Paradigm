"""
Optimizer base class and standard implementations.

This module provides the main optimization loop and various
optimizer implementations for architecture search.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from datetime import datetime
import random

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .search_space import SearchSpace, Architecture, Edge, Cell, Operation, OperationType
from .controller import Controller, RandomController, RLController, EvolutionController


@dataclass
class OptimizationResult:
    """
    Result of architecture search optimization.
    
    Attributes:
        best_architecture: Best architecture found
        best_accuracy: Accuracy of best architecture
        all_architectures: All architectures evaluated
        all_rewards: Rewards for all architectures
        search_time: Total search time in seconds
        history: Search history with metrics per iteration
    """
    best_architecture: Optional[Architecture]
    best_accuracy: float
    all_architectures: List[Architecture] = field(default_factory=list)
    all_rewards: List[float] = field(default_factory=list)
    all_metrics: Dict[str, List[float]] = field(default_factory=dict)
    search_time: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'best_architecture': self.best_architecture.to_dict() if self.best_architecture else None,
            'best_accuracy': self.best_accuracy,
            'num_architectures': len(self.all_architectures),
            'search_time': self.search_time,
            'history': self.history,
            'metadata': self.metadata,
        }
    
    def save(self, path: Union[str, Path]):
        """Save result to file."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 50,
            "Architecture Search Results",
            "=" * 50,
            f"Best Architecture: {self.best_architecture.name if self.best_architecture else 'N/A'}",
            f"Best Accuracy: {self.best_accuracy:.4f}",
            f"Architectures Evaluated: {len(self.all_architectures)}",
            f"Search Time: {self.search_time:.2f}s",
        ]
        
        if self.all_metrics:
            lines.append("\nMetrics:")
            for key, values in self.all_metrics.items():
                if values:
                    lines.append(f"  {key}: mean={np.mean(values):.4f}, max={np.max(values):.4f}")
        
        return "\n".join(lines)


class Optimizer(ABC):
    """
    Abstract base class for architecture optimizers.
    
    Optimizers orchestrate the search process, including:
    - Architecture generation
    - Training and evaluation
    - Controller updates
    - Early stopping
    
    Subclasses implement specific search algorithms:
    - StandardOptimizer: Basic RL/evolutionary search
    - DARTSOptimizer: Differentiable architecture search
    - BayesianOptimizer: Bayesian optimization
    - MultiObjectiveOptimizer: Multi-objective search
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        controller: Optional[Controller] = None,
        epochs_per_trial: int = 50,
        batch_size: int = 128,
        learning_rate: float = 0.025,
        weight_decay: float = 3e-4,
        max_grad_norm: float = 5.0,
        early_stopping_patience: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize optimizer.
        
        Args:
            search_space: Search space to optimize over
            controller: Controller for architecture generation
            epochs_per_trial: Training epochs per architecture
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay
            max_grad_norm: Maximum gradient norm
            early_stopping_patience: Patience for early stopping
            device: Device to use
            seed: Random seed
        """
        self.search_space = search_space
        
        # Use provided controller or create default
        if controller is None:
            controller = RLController(search_space)
        self.controller = controller
        
        # Training configuration
        self.epochs_per_trial = epochs_per_trial
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        
        # Device
        self.device = torch.device(device)
        
        # Seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # State
        self.results: List[OptimizationResult] = []
        self.current_result: Optional[OptimizationResult] = None
    
    @abstractmethod
    def search(
        self,
        num_trials: int = 100,
        dataset: str = "cifar10",
        **kwargs
    ) -> OptimizationResult:
        """
        Run architecture search.
        
        Args:
            num_trials: Number of architectures to evaluate
            dataset: Dataset name
            **kwargs: Additional arguments
            
        Returns:
            OptimizationResult with best architecture
        """
        pass
    
    def _train_architecture(
        self,
        architecture: Architecture,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train a single architecture and return validation accuracy.
        
        Args:
            architecture: Architecture to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (uses default if None)
            
        Returns:
            (validation_accuracy, metrics_dict) tuple
        """
        epochs = epochs or self.epochs_per_trial
        
        # Build model
        model = self.search_space.build(
            architecture,
            input_shape=train_loader.dataset.tensors[0].shape[1:],
            num_classes=10
        )
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            scheduler.step()
            
            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()
            
            val_acc = 100.0 * val_correct / val_total
            best_val_acc = max(best_val_acc, val_acc)
            
            # Early stopping
            if val_acc >= best_val_acc:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break
        
        metrics = {
            'train_accuracy': 100.0 * train_correct / train_total if train_total > 0 else 0,
            'val_accuracy': best_val_acc,
        }
        
        return best_val_acc, metrics
    
    def _evaluate_architecture(
        self,
        architecture: Architecture,
        **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate an architecture without full training.
        
        Used for quick evaluation and early pruning.
        
        Args:
            architecture: Architecture to evaluate
            **kwargs: Additional arguments
            
        Returns:
            (score, metrics_dict) tuple
        """
        # Default: return random score (should be overridden)
        return random.random(), {}
    
    def _create_data_loaders(
        self,
        dataset: str = "cifar10",
        val_split: float = 0.1
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training.
        
        Args:
            dataset: Dataset name
            val_split: Validation split ratio
            
        Returns:
            (train_loader, val_loader) tuple
        """
        # This is a placeholder - in practice, load actual data
        # For now, create synthetic data
        
        # Create dummy data
        num_samples = 10000
        x_train = torch.randn(num_samples, 3, 32, 32)
        y_train = torch.randint(0, 10, (num_samples,))
        
        # Split
        val_size = int(num_samples * val_split)
        indices = torch.randperm(num_samples)
        
        x_val = x_train[indices[:val_size]]
        y_val = y_train[indices[:val_size]]
        x_train = x_train[indices[val_size:]]
        y_train = y_train[indices[val_size:]]
        
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def save_checkpoint(self, path: Union[str, Path]):
        """Save optimizer state."""
        self.controller.save(path)
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load optimizer state."""
        self.controller.load(path)


class StandardOptimizer(Optimizer):
    """
    Standard optimizer using controller-based architecture generation.
    
    This is the main optimizer that uses the controller to generate
    architectures and evaluates them through training.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        controller: Optional[Controller] = None,
        evaluation_budget: int = 50,
        **kwargs
    ):
        """
        Initialize standard optimizer.
        
        Args:
            search_space: Search space
            controller: Controller for generation
            evaluation_budget: Max evaluations per trial
            **kwargs: Base optimizer arguments
        """
        super().__init__(search_space, controller, **kwargs)
        self.evaluation_budget = evaluation_budget
    
    def search(
        self,
        num_trials: int = 100,
        dataset: str = "cifar10",
        epochs_per_trial: Optional[int] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Run architecture search.
        
        Args:
            num_trials: Number of architectures to evaluate
            dataset: Dataset name
            epochs_per_trial: Training epochs per architecture
            **kwargs: Additional arguments
            
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(dataset)
        
        # Initialize result
        result = OptimizationResult(
            best_architecture=None,
            best_accuracy=0.0,
        )
        
        # Search loop
        epochs_per_trial = epochs_per_trial or self.epochs_per_trial
        
        for trial in range(num_trials):
            # Sample architecture
            architectures, metadata = self.controller.sample(batch_size=1)
            architecture = architectures[0]
            
            # Evaluate
            if epochs_per_trial > 0:
                accuracy, metrics = self._train_architecture(
                    architecture, train_loader, val_loader, epochs_per_trial
                )
            else:
                accuracy, metrics = self._evaluate_architecture(architecture)
            
            # Update controller
            self.controller.update([architecture], [accuracy], metrics)
            
            # Record
            result.all_architectures.append(architecture)
            result.all_rewards.append(accuracy)
            
            for key, value in metrics.items():
                if key not in result.all_metrics:
                    result.all_metrics[key] = []
                result.all_metrics[key].append(value)
            
            result.history.append({
                'trial': trial,
                'accuracy': accuracy,
                'metrics': metrics,
                'operator': metadata[0].get('operator', 'controller'),
            })
            
            # Update best
            if accuracy > result.best_accuracy:
                result.best_accuracy = accuracy
                result.best_architecture = architecture
            
            # Progress
            if (trial + 1) % 10 == 0:
                print(f"Trial {trial + 1}/{num_trials}: "
                      f"Best accuracy: {result.best_accuracy:.2f}%, "
                      f"Current accuracy: {accuracy:.2f}%")
        
        result.search_time = time.time() - start_time
        self.current_result = result
        self.results.append(result)
        
        return result


class BatchOptimizer(Optimizer):
    """
    Batch optimizer that evaluates multiple architectures in parallel.
    
    This optimizer generates and evaluates batches of architectures
    together, which can be more efficient on GPU clusters.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        controller: Optional[Controller] = None,
        batch_size: int = 8,
        **kwargs
    ):
        """
        Initialize batch optimizer.
        
        Args:
            search_space: Search space
            controller: Controller for generation
            batch_size: Batch evaluation size
            **kwargs: Base optimizer arguments
        """
        super().__init__(search_space, controller, **kwargs)
        self.eval_batch_size = batch_size
    
    def search(
        self,
        num_trials: int = 100,
        dataset: str = "cifar10",
        epochs_per_trial: Optional[int] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Run batch architecture search.
        
        Args:
            num_trials: Number of architectures to evaluate
            dataset: Dataset name
            epochs_per_trial: Training epochs per architecture
            **kwargs: Additional arguments
            
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(dataset)
        
        # Initialize result
        result = OptimizationResult(
            best_architecture=None,
            best_accuracy=0.0,
        )
        
        epochs_per_trial = epochs_per_trial or self.epochs_per_trial
        
        # Batch evaluation
        for batch_start in range(0, num_trials, self.eval_batch_size):
            batch_end = min(batch_start + self.eval_batch_size, num_trials)
            batch_size = batch_end - batch_start
            
            # Sample batch
            architectures, metadata_list = self.controller.sample(batch_size=batch_size)
            
            # Evaluate batch
            batch_accuracies = []
            batch_metrics_list = []
            
            for architecture in architectures:
                accuracy, metrics = self._train_architecture(
                    architecture, train_loader, val_loader, epochs_per_trial
                )
                batch_accuracies.append(accuracy)
                batch_metrics_list.append(metrics)
                
                # Record
                result.all_architectures.append(architecture)
                result.all_rewards.append(accuracy)
                
                for key, value in metrics.items():
                    if key not in result.all_metrics:
                        result.all_metrics[key] = []
                    result.all_metrics[key].append(value)
                
                # Update best
                if accuracy > result.best_accuracy:
                    result.best_accuracy = accuracy
                    result.best_architecture = architecture
            
            # Update controller with batch
            self.controller.update(architectures, batch_accuracies)
            
            # Record history
            result.history.append({
                'batch_start': batch_start,
                'batch_end': batch_end,
                'mean_accuracy': np.mean(batch_accuracies),
                'max_accuracy': np.max(batch_accuracies),
            })
            
            print(f"Batch {batch_start}-{batch_end}: "
                  f"Mean accuracy: {np.mean(batch_accuracies):.2f}%, "
                  f"Best: {np.max(batch_accuracies):.2f}%")
        
        result.search_time = time.time() - start_time
        self.current_result = result
        self.results.append(result)
        
        return result
