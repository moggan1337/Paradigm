"""
Standard metrics for architecture evaluation.

This module provides common metrics for evaluating neural network
architectures such as latency, parameter count, FLOPs, etc.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Callable
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..core.search_space import Architecture


class Metric(ABC):
    """
    Abstract base class for architecture metrics.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, architecture: Architecture, **kwargs) -> float:
        """Calculate metric value."""
        pass


class LatencyMetric(Metric):
    """
    Latency metric for measuring inference time.
    
    Measures the average inference latency on the specified device.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 1,
        n_warmup: int = 10,
        n_runs: int = 100,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ):
        """
        Initialize latency metric.
        
        Args:
            device: Device for measurement
            batch_size: Batch size for inference
            n_warmup: Number of warmup runs
            n_runs: Number of measurement runs
            input_shape: Input tensor shape
        """
        super().__init__("latency")
        
        self.device = device
        self.batch_size = batch_size
        self.n_warmup = n_warmup
        self.n_runs = n_runs
        self.input_shape = input_shape
    
    def calculate(
        self,
        architecture: Architecture,
        model: Optional[nn.Module] = None,
        **kwargs
    ) -> float:
        """
        Calculate latency.
        
        Args:
            architecture: Architecture (used if model not provided)
            model: Pre-built model (optional)
            
        Returns:
            Average latency in milliseconds
        """
        if model is None:
            return 0.0  # Would need to build model
        
        model = model.to(self.device)
        model.eval()
        
        # Create input
        x = torch.randn(
            (self.batch_size,) + self.input_shape[1:],
            device=self.device
        )
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.n_warmup):
                model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure
        times = []
        
        with torch.no_grad():
            for _ in range(self.n_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                model(x)
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        
        return np.mean(times)


class ParameterCountMetric(Metric):
    """
    Metric for counting model parameters.
    """
    
    def __init__(self):
        super().__init__("params")
    
    def calculate(
        self,
        architecture: Architecture,
        model: Optional[nn.Module] = None,
        **kwargs
    ) -> float:
        """
        Calculate number of parameters in millions.
        
        Args:
            architecture: Architecture (used if model not provided)
            model: Pre-built model (optional)
            
        Returns:
            Number of parameters in millions
        """
        if model is None:
            return architecture.num_parameters() / 1e6
        
        total = 0
        for p in model.parameters():
            total += p.numel()
        
        return total / 1e6


class FLOPsMetric(Metric):
    """
    Metric for counting FLOPs (Floating Point Operations).
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ):
        super().__init__("flops")
        self.input_shape = input_shape
    
    def calculate(
        self,
        architecture: Architecture,
        model: Optional[nn.Module] = None,
        **kwargs
    ) -> float:
        """
        Calculate FLOPs in billions.
        
        Args:
            architecture: Architecture (used if model not provided)
            model: Pre-built model (optional)
            
        Returns:
            FLOPs in billions
        """
        if model is None:
            return 0.0
        
        # Simple FLOP estimation
        total_flops = 0
        
        def count_flops(module, input, output):
            nonlocal total_flops
            
            if isinstance(module, nn.Conv2d):
                batch_size = input[0].size(0)
                output_height, output_width = output.size(2), output.size(3)
                
                kernel_ops = module.kernel_size[0] * module.kernel_size[1]
                output_elements = batch_size * output_height * output_width
                
                # FLOPs = batch_size * output_elements * input_channels * output_channels * kernel_ops / groups
                flops = output_elements * module.in_channels * module.out_channels * kernel_ops / module.groups
                total_flops += flops
            
            elif isinstance(module, nn.Linear):
                batch_size = input[0].size(0)
                flops = batch_size * module.in_features * module.out_features
                total_flops += flops
        
        hooks = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                hooks.append(m.register_forward_hook(count_flops))
        
        # Forward pass
        x = torch.randn(self.input_shape)
        model(x)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        return total_flops / 1e9  # Convert to billions


class MemoryMetric(Metric):
    """
    Metric for measuring memory usage.
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__("memory")
        self.device = device
    
    def calculate(
        self,
        architecture: Architecture,
        model: Optional[nn.Module] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        **kwargs
    ) -> float:
        """
        Calculate peak memory usage in MB.
        
        Args:
            architecture: Architecture
            model: Pre-built model
            input_shape: Input shape
            
        Returns:
            Memory usage in MB
        """
        if model is None:
            return 0.0
        
        if not torch.cuda.is_available():
            return 0.0
        
        torch.cuda.reset_peak_memory_stats()
        
        model = model.to(self.device)
        x = torch.randn(input_shape, device=self.device)
        
        with torch.no_grad():
            model(x)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        return peak_memory


class EdgeLevelMetric(Metric):
    """
    Metric for computing edge-level importance scores.
    
    Used for architecture search with differentiable operations.
    """
    
    def __init__(self):
        super().__init__("edge_importance")
    
    def calculate(
        self,
        architecture: Architecture,
        model: Optional[nn.Module] = None,
        **kwargs
    ) -> Dict[Tuple[int, int], float]:
        """
        Calculate importance scores for each edge.
        
        Returns:
            Dictionary mapping (source, target) to importance score
        """
        if model is None:
            return {}
        
        importance = {}
        
        # Get architecture alphas if available
        if hasattr(model, 'alphas'):
            for key, alpha in model.alphas.items():
                importance[key] = torch.softmax(alpha, dim=0).max().item()
        
        return importance


def compute_accuracy_metric(
    architecture: Architecture,
    model: nn.Module,
    dataloader,
    device: str = "cuda"
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        architecture: Architecture
        model: Trained model
        dataloader: Data loader
        device: Device
        
    Returns:
        Accuracy percentage
    """
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total


def compute_top_k_accuracy(
    model: nn.Module,
    dataloader,
    k: int = 5,
    device: str = "cuda"
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        model: Trained model
        dataloader: Data loader
        k: Top k
        device: Device
        
    Returns:
        Top-k accuracy percentage
    """
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            _, top_k_pred = outputs.topk(k, dim=1, largest=True, sorted=True)
            targets_expanded = targets.view(-1, 1).expand_as(top_k_pred)
            correct += top_k_pred.eq(targets_expanded).sum().item()
            total += targets.size(0)
    
    return 100.0 * correct / total
