"""
Performance prediction for neural architectures.

This module implements zero-cost and low-cost metrics for predicting
architecture performance without full training.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy import linalg

from ..core.search_space import SearchSpace, Architecture


@dataclass
class ZeroCostMetricResult:
    """Result from a zero-cost metric calculation."""
    
    name: str
    score: float
    normalized_score: Optional[float] = None


class ZeroCostMetric(ABC):
    """
    Abstract base class for zero-cost metrics.
    
    Zero-cost metrics estimate architecture performance without
    training, using properties of the architecture that can be
    computed quickly.
    
    ┌─────────────────────────────────────────────────────────┐
    │              Zero-Cost Metric Framework                 │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │  Input: Architecture (uninitialized weights)            │
    │    ↓                                                    │
    │  Compute Metric-Specific Features:                     │
    │    - SynFlow: Sum of synaptic flows                    │
    │    - Grad-Norm: Gradient norms                         │
    │    - NAS-WOT: Activation patterns                       │
    │    - Fisher: Fisher information                         │
    │    ↓                                                    │
    │  Output: Score (higher = more trainable)               │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, name: str = "metric"):
        """
        Initialize metric.
        
        Args:
            name: Name of the metric
        """
        self.name = name
    
    @abstractmethod
    def calculate(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...],
        device: str = "cuda"
    ) -> float:
        """
        Calculate metric score.
        
        Args:
            architecture: Architecture to evaluate
            input_shape: Input tensor shape
            device: Device to use
            
        Returns:
            Metric score
        """
        pass
    
    def __call__(self, *args, **kwargs) -> float:
        """Allow calling instance directly."""
        return self.calculate(*args, **kwargs)


class SynFlowMetric(ZeroCostMetric):
    """
    Synaptic Flow (SynFlow) metric.
    
    SynFlow estimates trainability by computing the sum of
    absolute synaptic flows through the network.
    
    Reference:
        Tanaka, H., Kunin, D., Yamins, D. L., & Ganguli, S. (2020).
        Pruning neural networks without any data by iteratively
        conserving synaptic flow. NeurIPS 2020.
    """
    
    def __init__(self):
        super().__init__("synflow")
    
    def calculate(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...],
        device: str = "cuda"
    ) -> float:
        """
        Calculate SynFlow score.
        
        Args:
            architecture: Architecture to evaluate
            input_shape: Input shape (e.g., (1, 3, 32, 32))
            device: Device
            
        Returns:
            SynFlow score
        """
        model = self._build_model(architecture, input_shape)
        model = model.to(device)
        model.eval()
        
        # Initialize all weights to 1
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Create input with ones
        x = torch.ones(input_shape, device=device)
        
        # Forward pass
        output = model(x)
        
        # Sum of absolute output
        score = torch.sum(torch.abs(output)).item()
        
        return score
    
    def _build_model(self, architecture: Architecture, input_shape: Tuple[int, ...]):
        """Build model from architecture."""
        # This is a simplified version
        # Full implementation would use the actual architecture
        return nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )


class GradNormMetric(ZeroCostMetric):
    """
    Gradient Norm metric.
    
    Computes the norm of gradients with respect to a random loss
    to estimate trainability.
    """
    
    def __init__(self):
        super().__init__("grad_norm")
    
    def calculate(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...],
        device: str = "cuda"
    ) -> float:
        """
        Calculate gradient norm score.
        
        Args:
            architecture: Architecture to evaluate
            input_shape: Input shape
            device: Device
            
        Returns:
            Gradient norm score
        """
        model = self._build_model(architecture, input_shape)
        model = model.to(device)
        model.train()
        
        # Random input and target
        x = torch.randn(input_shape, device=device)
        y = torch.randint(0, 10, (input_shape[0],), device=device)
        
        # Forward pass
        output = model(x)
        
        # Random loss
        loss = F.cross_entropy(output, y)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        
        return total_norm
    
    def _build_model(self, architecture: Architecture, input_shape: Tuple[int, ...]):
        """Build model from architecture."""
        return nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )


class NASWOTMetric(ZeroCostMetric):
    """
    NAS without Training (NAS-WOT) metric.
    
    Uses activation patterns and network usage to estimate performance.
    
    Reference:
        Chen, W., Gong, X., & Wang, Z. (2021).
        Neural Architecture Search on Efficient Databases via
        Zero-Cost Proxies. arXiv preprint arXiv:2106.03678.
    """
    
    def __init__(self, n_samples: int = 100):
        super().__init__("nas_wot")
        self.n_samples = n_samples
    
    def calculate(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...],
        device: str = "cuda"
    ) -> float:
        """
        Calculate NAS-WOT score.
        
        Args:
            architecture: Architecture to evaluate
            input_shape: Input shape
            device: Device
            
        Returns:
            NAS-WOT score
        """
        model = self._build_model(architecture, input_shape)
        model = model.to(device)
        model.eval()
        
        # Collect activations
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach())
        
        hooks = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                hooks.append(m.register_forward_hook(hook_fn))
        
        # Forward passes with random inputs
        with torch.no_grad():
            for _ in range(self.n_samples):
                x = torch.randn(input_shape, device=device)
                model(x)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Calculate score based on activation patterns
        score = 0.0
        for act in activations:
            # Normalized Hamming distance of binary activations
            binary = (act > 0).float()
            uniqueness = len(torch.unique(binary)) / binary.numel()
            score += uniqueness
        
        return score / len(activations) if activations else 0.0
    
    def _build_model(self, architecture: Architecture, input_shape: Tuple[int, ...]):
        """Build model from architecture."""
        return nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )


class PerformancePredictor:
    """
    Performance predictor using zero-cost metrics.
    
    This class combines multiple zero-cost metrics to predict
    architecture performance.
    
    ┌─────────────────────────────────────────────────────────┐
    │             Performance Predictor                       │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │  Input: Architecture                                     │
    │    ↓                                                    │
    │  ┌──────────────────────────────────┐                  │
    │  │   Zero-Cost Metrics               │                  │
    │  │   - SynFlow                       │                  │
    │  │   - Grad-Norm                     │                  │
    │  │   - NAS-WOT                       │                  │
    │  │   - Fisher                        │                  │
    │  └──────────────┬───────────────────┘                  │
    │                 ↓                                      │
    │  ┌──────────────────────────────────┐                  │
    │  │   Ensemble Scoring                │                  │
    │  │   score = Σ w_i * metric_i        │                  │
    │  └──────────────┬───────────────────┘                  │
    │                 ↓                                      │
    │  Output: Predicted Performance Score                   │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        metrics: List[ZeroCostMetric] = None,
        weights: Dict[str, float] = None
    ):
        """
        Initialize predictor.
        
        Args:
            metrics: List of zero-cost metrics to use
            weights: Weights for each metric
        """
        # Default metrics
        if metrics is None:
            metrics = [
                SynFlowMetric(),
                GradNormMetric(),
                NASWOTMetric(),
            ]
        
        self.metrics = {m.name: m for m in metrics}
        self.weights = weights or {m.name: 1.0 for m in metrics}
    
    def score(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...],
        device: str = "cuda"
    ) -> float:
        """
        Score a single architecture.
        
        Args:
            architecture: Architecture to score
            input_shape: Input shape
            device: Device
            
        Returns:
            Ensemble score
        """
        scores = {}
        
        for name, metric in self.metrics.items():
            scores[name] = metric.calculate(architecture, input_shape, device)
        
        # Weighted ensemble
        total_weight = sum(self.weights.values())
        ensemble_score = sum(
            scores[name] * self.weights[name]
            for name in scores
        ) / total_weight
        
        return ensemble_score
    
    def batch_score(
        self,
        architectures: List[Architecture],
        input_shape: Tuple[int, ...],
        device: str = "cuda",
        batch_size: int = 32
    ) -> List[float]:
        """
        Score multiple architectures.
        
        Args:
            architectures: Architectures to score
            input_shape: Input shape
            device: Device
            batch_size: Batch size
            
        Returns:
            List of scores
        """
        scores = []
        
        for arch in architectures:
            score = self.score(arch, input_shape, device)
            scores.append(score)
        
        return scores
    
    def ensemble_score(
        self,
        architectures: List[Architecture],
        input_shape: Tuple[int, ...],
        device: str = "cuda"
    ) -> Dict[str, List[float]]:
        """
        Get scores from all metrics as an ensemble.
        
        Args:
            architectures: Architectures to score
            input_shape: Input shape
            device: Device
            
        Returns:
            Dictionary mapping metric names to score lists
        """
        results = {name: [] for name in self.metrics}
        
        for arch in architectures:
            for name, metric in self.metrics.items():
                score = metric.calculate(arch, input_shape, device)
                results[name].append(score)
        
        return results
    
    def rank(
        self,
        architectures: List[Architecture],
        input_shape: Tuple[int, ...],
        device: str = "cuda"
    ) -> List[Tuple[int, float]]:
        """
        Rank architectures by predicted performance.
        
        Args:
            architectures: Architectures to rank
            input_shape: Input shape
            device: Device
            
        Returns:
            List of (index, score) tuples sorted by score
        """
        scores = self.batch_score(architectures, input_shape, device)
        
        # Sort by score
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked
