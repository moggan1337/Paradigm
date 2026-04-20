"""
DARTS: Differentiable Architecture Search.

This module implements DARTS (Differentiable Architecture Search)
and its variants for gradient-based neural architecture search.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Callable
from copy import deepcopy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import Tensor

from ..core.search_space import SearchSpace, Architecture, Operation, OperationType, Edge
from ..core.optimizer import Optimizer, OptimizationResult


class DARTSOptimizer(Optimizer):
    """
    DARTS (Differentiable Architecture Search) optimizer.
    
    DARTS relaxes the discrete search space into a continuous one,
    allowing gradient-based optimization of architecture parameters.
    
    Algorithm:
    ┌─────────────────────────────────────────────────────────┐
    │                  DARTS Algorithm                        │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │  Initialize:                                            │
    │    - Shared weights w for supernet                     │
    │    - Architecture params α (one per edge)               │
    │                                                          │
    │  Repeat for each epoch:                                │
    │    1. Update weights w on training set:                │
    │       w = w - η_w * ∇_w L_train(w, α)                   │
    │                                                          │
    │    2. Update architecture params α on validation set:   │
    │       α = α - η_α * ∇_α L_val(w, α)                     │
    │                                                          │
    │    3. Derive discrete architecture from α              │
    │                                                          │
    │  Return: Best architecture based on α                   │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    
    Reference:
        Liu, H., Simonyan, K., & Yang, Y. (2018). DARTS: Differentiable
        Architecture Search. ICLR 2019.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        unrolled_steps: int = 1,
        arch_learning_rate: float = 0.0003,
        weight_learning_rate: float = 0.025,
        momentum: float = 0.9,
        weight_decay: float = 3e-4,
        arch_decay: float = 1e-3,
        grad_clip: float = 5.0,
        first_order: bool = False,
        **kwargs
    ):
        """
        Initialize DARTS optimizer.
        
        Args:
            search_space: Search space
            unrolled_steps: Unrolled optimization steps
            arch_learning_rate: Learning rate for architecture params
            weight_learning_rate: Learning rate for weights
            momentum: SGD momentum
            weight_decay: Weight decay
            arch_decay: Architecture parameter decay
            grad_clip: Gradient clipping threshold
            first_order: Use first-order approximation
        """
        super().__init__(search_space, **kwargs)
        
        self.unrolled_steps = unrolled_steps
        self.arch_lr = arch_learning_rate
        self.weight_lr = weight_learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.arch_decay = arch_decay
        self.grad_clip = grad_clip
        self.first_order = first_order
        
        # Architecture parameters (α)
        self.arch_params = None
        
        # Shared weights
        self.model = None
        
        # Optimizers
        self.weight_optimizer = None
        self.arch_optimizer = None
    
    def _init_architecture_params(self):
        """Initialize architecture parameters for each edge."""
        num_ops = len(self.search_space.CNN_OPS) if hasattr(self.search_space, 'CNN_OPS') else 6
        
        # Initialize α parameters
        self.arch_params = nn.ParameterDict()
        
        # For each edge in search space, create learnable α
        # This is simplified - full implementation tracks edges per cell
        num_cells = getattr(self.search_space, 'num_cells', 8)
        max_nodes = self.search_space.max_nodes
        
        for cell_idx in range(num_cells):
            for target in range(1, max_nodes):
                for source in range(target):
                    key = f"cell{cell_idx}_{source}_{target}"
                    self.arch_params[key] = nn.Parameter(
                        torch.ones(num_ops) / num_ops
                    )
    
    def search(
        self,
        num_epochs: int = 50,
        dataset: str = "cifar10",
        train_loader = None,
        val_loader = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Run DARTS architecture search.
        
        Args:
            num_epochs: Number of search epochs
            dataset: Dataset name
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            OptimizationResult with best architecture
        """
        start_time = time.time()
        
        # Create data loaders if not provided
        if train_loader is None or val_loader is None:
            train_loader, val_loader = self._create_data_loaders(dataset)
        
        # Initialize architecture parameters
        self._init_architecture_params()
        
        # Build shared model
        dummy_arch = self.search_space.sample()
        self.model = self.search_space.build(
            dummy_arch,
            input_shape=(3, 32, 32),
            num_classes=10
        )
        self.model = self.model.to(self.device)
        
        # Initialize optimizers
        self.weight_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.weight_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        self.arch_optimizer = torch.optim.Adam(
            self.arch_params.parameters(),
            lr=self.arch_lr,
            betas=(0.5, 0.999),
            weight_decay=self.arch_decay
        )
        
        # Training loop
        best_arch = None
        best_accuracy = 0.0
        history = []
        
        for epoch in range(num_epochs):
            # Training phase (update weights)
            train_loss = self._train_weights(train_loader)
            
            # Validation phase (update architecture)
            val_loss, val_acc = self._train_architecture(val_loader)
            
            # Get current architecture
            current_arch = self._derive_architecture()
            
            # Evaluate current architecture
            if (epoch + 1) % 5 == 0:
                accuracy = self._evaluate_architecture(current_arch, val_loader)
            else:
                accuracy = val_acc
            
            # Update best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_arch = deepcopy(current_arch)
            
            # Record history
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'best_accuracy': best_accuracy,
                'architecture': current_arch.name,
            })
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Acc: {val_acc:.2f}%, "
                      f"Best: {best_accuracy:.2f}%")
        
        # Create result
        result = OptimizationResult(
            best_architecture=best_arch,
            best_accuracy=best_accuracy,
            search_time=time.time() - start_time,
            history=history,
            metadata={
                'algorithm': 'DARTS',
                'num_epochs': num_epochs,
            }
        )
        
        self.current_result = result
        return result
    
    def _train_weights(self, train_loader) -> float:
        """Update shared weights on training data."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.weight_optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip
            )
            
            self.weight_optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _train_architecture(self, val_loader) -> Tuple[float, float]:
        """Update architecture parameters on validation data."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.arch_optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.arch_params.parameters(),
                self.grad_clip
            )
            
            self.arch_optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * total_correct / total_samples
        
        return avg_loss, accuracy
    
    def _derive_architecture(self) -> Architecture:
        """
        Derive discrete architecture from architecture parameters.
        
        For each edge, select the operation with highest α weight.
        
        Returns:
            Discretized Architecture
        """
        arch = Architecture(name=f"darts_derived_{int(time.time())}")
        
        # Get operation set
        ops = self.search_space.CNN_OPS if hasattr(self.search_space, 'CNN_OPS') else [
            OperationType.SEP_CONV_3X3,
            OperationType.SEP_CONV_5X5,
            OperationType.MAX_POOL_3X3,
            OperationType.AVG_POOL_3X3,
            OperationType.SKIP_CONNECT,
            OperationType.NONE,
        ]
        
        num_cells = getattr(self.search_space, 'num_cells', 8)
        max_nodes = self.search_space.max_nodes
        
        for cell_idx in range(num_cells):
            for target in range(1, max_nodes):
                for source in range(target):
                    key = f"cell{cell_idx}_{source}_{target}"
                    
                    if key in self.arch_params:
                        alpha = self.arch_params[key]
                        op_idx = torch.argmax(alpha).item()
                        op_type = ops[op_idx]
                        
                        edge = Edge(
                            source=source,
                            target=target,
                            operation=Operation(op_type=op_type)
                        )
                        arch.edges.append(edge)
        
        # Add cells
        for i in range(max_nodes):
            from ..core.search_space import Cell
            arch.cells.append(Cell(node_index=i))
        
        return arch
    
    def _evaluate_architecture(
        self,
        architecture: Architecture,
        val_loader
    ) -> float:
        """Evaluate a discrete architecture."""
        # Build model for this architecture
        model = self.search_space.build(
            architecture,
            input_shape=(3, 32, 32),
            num_classes=10
        )
        model = model.to(self.device)
        
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.025,
            momentum=0.9,
            weight_decay=3e-4
        )
        
        # Train for a few epochs
        model.train()
        for _ in range(5):  # Quick evaluation
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100.0 * correct / total


class DARTSModel(nn.Module):
    """
    DARTS supernet model with mixed operations.
    
    This model shares weights across all candidate operations
    using the DARTS relaxation.
    """
    
    def __init__(
        self,
        architecture: Architecture,
        num_classes: int = 10,
        channels: List[int] = None,
        num_cells: int = 8
    ):
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.channels = channels or [16, 32, 64]
        self.num_cells = num_cells
        
        # Build supernet
        self._build_supernet()
    
    def _build_supernet(self):
        """Build the supernet with mixed operations."""
        from .cnn_search_space import CNNOperations, DARTSCell
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0])
        )
        
        # Cells
        self.cells = nn.ModuleList()
        
        for i in range(self.num_cells):
            reduce = i >= self.num_cells // 3
            in_channels = self.channels[min(i, len(self.channels) - 1)]
            out_channels = self.channels[min(i + 1 if reduce else i, len(self.channels) - 1)]
            
            cell = DARTSCell(
                architecture=self.architecture,
                in_channels=in_channels,
                out_channels=out_channels,
                is_reduction=reduce
            )
            self.cells.append(cell)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.channels[-1], self.num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through supernet."""
        x = self.stem(x)
        
        for cell in self.cells:
            x = cell([x])  # DARTS cell expects list input
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class PDARTSOptimizer(DARTSOptimizer):
    """
    Progressive DARTS optimizer.
    
    PDARTS progressively increases the depth of the search space
    during the search process, making it more efficient.
    
    Reference:
        Chen, X., Xie, L., Wu, J., & Tian, Q. (2019). Progressive
        Differentiable Architecture Search: Bridging the Depth Gap
        between Search and Evaluation. ICCV 2019.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        depth_epochs: List[int] = None,
        depth_ratios: List[float] = None,
        **kwargs
    ):
        """
        Initialize PDARTS optimizer.
        
        Args:
            search_space: Search space
            depth_epochs: Epochs at each search stage
            depth_ratios: Depth reduction at each stage
        """
        super().__init__(search_space, **kwargs)
        
        self.depth_epochs = depth_epochs or [15, 15, 15]
        self.depth_ratios = depth_ratios or [0.5, 0.7, 1.0]
    
    def search(self, num_epochs: int = 50, **kwargs) -> OptimizationResult:
        """Run progressive DARTS search."""
        # This is a simplified version
        # Full implementation would progressively adjust depth
        return super().search(num_epochs=num_epochs, **kwargs)


class PC-DARTSOptimizer(DARTSOptimizer):
    """
    Partial Channel DARTS optimizer.
    
    PC-DARTS reduces memory usage by sampling a subset of channels
    during architecture search.
    
    Reference:
        Xu, Y., Xie, L., Zhang, X., Chen, X., Qi, G. J., Tian, Q., & Zhang, X. (2019).
        PC-DARTS: Partial Channel Connections for Memory-Efficient
        Architecture Search. NeurIPS 2019.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        channel_ratio: float = 0.5,
        **kwargs
    ):
        """
        Initialize PC-DARTS optimizer.
        
        Args:
            search_space: Search space
            channel_ratio: Ratio of channels to sample
        """
        super().__init__(search_space, **kwargs)
        
        self.channel_ratio = channel_ratio
    
    def search(self, num_epochs: int = 50, **kwargs) -> OptimizationResult:
        """Run PC-DARTS search with partial channels."""
        # Simplified implementation
        return super().search(num_epochs=num_epochs, **kwargs)
