"""
CNN Search Space for image classification architectures.

This module provides search spaces for convolutional neural networks,
including the DARTS-style cell-based search space.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.search_space import (
    SearchSpace, Architecture, Cell, Edge, Operation, OperationType
)


# CNN-specific operations
class CNNOperations:
    """Collection of CNN operations for the search space."""
    
    @staticmethod
    def sep_conv_3x3(in_channels: int, out_channels: int) -> nn.Module:
        """Separable convolution 3x3."""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    @staticmethod
    def sep_conv_5x5(in_channels: int, out_channels: int) -> nn.Module:
        """Separable convolution 5x5."""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    @staticmethod
    def dil_conv_3x3(in_channels: int, out_channels: int, dilation: int = 2) -> nn.Module:
        """Dilated convolution 3x3."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    @staticmethod
    def dil_conv_5x5(in_channels: int, out_channels: int, dilation: int = 2) -> nn.Module:
        """Dilated convolution 5x5."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, padding=dilation * 2, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    @staticmethod
    def max_pool_3x3() -> nn.Module:
        """Max pooling 3x3."""
        return nn.MaxPool2d(3, padding=1, stride=1)
    
    @staticmethod
    def avg_pool_3x3() -> nn.Module:
        """Average pooling 3x3."""
        return nn.AvgPool2d(3, padding=1, stride=1)
    
    @staticmethod
    def skip_connect() -> nn.Module:
        """Identity/skip connection."""
        return nn.Identity()
    
    @staticmethod
    def none() -> nn.Module:
        """No operation (zero)."""
        return nn.ZeroOp() if hasattr(nn, 'ZeroOp') else nn.Identity()


class ZeroOp(nn.Module):
    """Zero operation that outputs zeros."""
    
    def forward(self, x):
        return torch.zeros_like(x)


class CNNSearchSpace(SearchSpace):
    """
    CNN search space for image classification.
    
    This search space defines a cell-based architecture where
    each cell consists of nodes connected by operations.
    
    Search Space:
    ┌─────────────────────────────────────────────────────┐
    │                    CNN Cell                          │
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │   Node 0 ──┬──[sep_conv_3x3]──┬──[sep_conv_5x5]──┐  │
    │            │                  │                  │  │
    │   Node 1 ──┼──[max_pool]──────┼──[avg_pool]──────┤  │
    │            │                  │                  │  │
    │   Node 2 ──┴──[skip_connect]──┴──[none]──────────┘  │
    │                              │                      │
    │                         Output                      │
    │                                                      │
    └─────────────────────────────────────────────────────┘
    
    Available Operations:
    - sep_conv_3x3: Separable convolution 3x3
    - sep_conv_5x5: Separable convolution 5x5
    - dil_conv_3x3: Dilated convolution 3x3
    - dil_conv_5x5: Dilated convolution 5x5
    - max_pool_3x3: Max pooling 3x3
    - avg_pool_3x3: Average pooling 3x3
    - skip_connect: Identity connection
    - none: No operation
    """
    
    # Default CNN operations
    CNN_OPS = [
        OperationType.SEP_CONV_3X3,
        OperationType.SEP_CONV_5X5,
        OperationType.MAX_POOL_3X3,
        OperationType.AVG_POOL_3X3,
        OperationType.SKIP_CONNECT,
        OperationType.NONE,
    ]
    
    def __init__(
        self,
        max_nodes: int = 7,
        num_operations: int = 6,
        channels: List[int] = None,
        stem_channels: int = 16,
        num_cells: int = 8,
        use_stem: bool = True,
        **kwargs
    ):
        """
        Initialize CNN search space.
        
        Args:
            max_nodes: Maximum nodes per cell
            num_operations: Number of available operations
            channels: Channel sizes for each stage
            stem_channels: Channels for stem conv
            num_cells: Number of cells in the network
            use_stem: Whether to use stem convolution
        """
        super().__init__(max_nodes, num_operations, **kwargs)
        
        self.channels = channels or [16, 32, 64]
        self.stem_channels = stem_channels
        self.num_cells = num_cells
        self.use_stem = use_stem
        
        # Limit operations to valid CNN ops
        self.num_operations = min(num_operations, len(self.CNN_OPS))
    
    def sample(self) -> Architecture:
        """
        Sample a random CNN architecture.
        
        Returns:
            Random Architecture
        """
        arch = Architecture(name=f"cnn_{len(self.training_history) if hasattr(self, 'training_history') else 0}")
        
        # Sample edges with random operations
        for target in range(1, self.max_nodes):
            for source in range(target):
                # Random operation
                op_idx = random.randint(0, self.num_operations - 1)
                op_type = self.CNN_OPS[op_idx]
                
                operation = Operation(
                    op_type=op_type,
                    parameters={}
                )
                
                edge = Edge(
                    source=source,
                    target=target,
                    operation=operation
                )
                arch.edges.append(edge)
        
        # Add cell nodes
        for i in range(self.max_nodes):
            arch.cells.append(Cell(node_index=i))
        
        return arch
    
    def mutate(
        self,
        architecture: Architecture,
        mutation_rate: float = 0.1
    ) -> Architecture:
        """
        Mutate a CNN architecture.
        
        Mutations:
        - Change operation on an edge
        - Add/remove skip connections
        - Change channel widths
        
        Args:
            architecture: Architecture to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated Architecture
        """
        arch = deepcopy(architecture)
        arch.name = f"cnn_mutated_{random.randint(0, 10000)}"
        
        # Mutate each edge
        new_edges = []
        for edge in arch.edges:
            if random.random() < mutation_rate:
                # Change operation
                op_idx = random.randint(0, self.num_operations - 1)
                edge.operation = Operation(
                    op_type=self.CNN_OPS[op_idx],
                    parameters={}
                )
            new_edges.append(edge)
        
        arch.edges = new_edges
        return arch
    
    def crossover(
        self,
        parent1: Architecture,
        parent2: Architecture
    ) -> Architecture:
        """
        Crossover two CNN architectures.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child architecture
        """
        # Randomly choose edges from each parent
        child = Architecture(name=f"cnn_cross_{random.randint(0, 10000)}")
        
        # Combine edges
        edges1 = {self._edge_key(e): e for e in parent1.edges}
        edges2 = {self._edge_key(e): e for e in parent2.edges}
        
        all_keys = set(edges1.keys()) | set(edges2.keys())
        
        for key in all_keys:
            if key in edges1 and key in edges2:
                # Both parents have this edge - random choice
                parent_edges = [edges1[key], edges2[key]]
                chosen = random.choice(parent_edges)
            elif key in edges1:
                chosen = edges1[key]
            else:
                chosen = edges2[key]
            
            child.edges.append(deepcopy(chosen))
        
        # Copy cells
        all_nodes = set(c.node_index for c in parent1.cells) | \
                   set(c.node_index for c in parent2.cells)
        for idx in all_nodes:
            child.cells.append(Cell(node_index=idx))
        
        return child
    
    def _edge_key(self, edge: Edge) -> Tuple[int, int]:
        """Get unique key for an edge."""
        return (edge.source, edge.target)
    
    def build(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...] = (3, 32, 32),
        num_classes: int = 10
    ) -> nn.Module:
        """
        Build PyTorch model from architecture.
        
        Args:
            architecture: Architecture to build
            input_shape: Input tensor shape
            num_classes: Number of output classes
            
        Returns:
            PyTorch nn.Module
        """
        return CNNModel(
            architecture=architecture,
            input_shape=input_shape,
            num_classes=num_classes,
            channels=self.channels,
            stem_channels=self.stem_channels,
            num_cells=self.num_cells,
            use_stem=self.use_stem
        )


class CNNModel(nn.Module):
    """
    PyTorch model built from a CNN architecture.
    """
    
    def __init__(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...] = (3, 32, 32),
        num_classes: int = 10,
        channels: List[int] = None,
        stem_channels: int = 16,
        num_cells: int = 8,
        use_stem: bool = True
    ):
        """
        Initialize CNN model.
        
        Args:
            architecture: Architecture specification
            input_shape: Input tensor shape
            num_classes: Number of output classes
            channels: Channel sizes
            stem_channels: Stem convolution channels
            num_cells: Number of cells
            use_stem: Whether to use stem
        """
        super().__init__()
        
        self.architecture = architecture
        self.channels = channels or [16, 32, 64]
        self.stem_channels = stem_channels
        self.num_classes = num_classes
        self.num_cells = num_cells
        self.use_stem = use_stem
        
        # Build model
        self._build_model(input_shape)
    
    def _build_model(self, input_shape: Tuple[int, ...]):
        """Build the model from architecture."""
        in_channels = input_shape[0]
        
        # Stem convolution
        if self.use_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, self.stem_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.stem_channels)
            )
            in_channels = self.stem_channels
        else:
            self.stem = nn.Identity()
        
        # Build cells
        self.cells = nn.ModuleList()
        self.reductions = []
        
        for i in range(self.num_cells):
            # Reduce spatial resolution every few cells
            reduce = i >= self.num_cells // 3
            if reduce:
                self.reductions.append(nn.MaxPool2d(2))
                in_channels = in_channels * 2
            
            # Create cell
            cell = DARTSCell(
                architecture=self.architecture,
                in_channels=in_channels,
                out_channels=self.channels[min(i, len(self.channels) - 1)],
                is_reduction=reduce
            )
            self.cells.append(cell)
            in_channels = self.channels[min(i, len(self.channels) - 1)]
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, self.num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Stem
        x = self.stem(x)
        
        # Cells
        for cell, reduce in zip(self.cells, self.reductions):
            x = cell(x)
            if reduce:
                x = self.reductions[self.reductions.index(reduce)](x)
        
        # Global pooling and classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class DARTSCell(nn.Module):
    """
    DARTS-style differentiable cell.
    
    This cell represents a directed acyclic graph where each
    edge has a mixed operation with learnable weights.
    """
    
    def __init__(
        self,
        architecture: Architecture,
        in_channels: int,
        out_channels: int,
        is_reduction: bool = False
    ):
        """
        Initialize DARTS cell.
        
        Args:
            architecture: Architecture specification
            in_channels: Input channels
            out_channels: Output channels
            is_reduction: Whether this is a reduction cell
        """
        super().__init__()
        
        self.architecture = architecture
        self.is_reduction = is_reduction
        
        # Build operations
        self._build_operations(in_channels, out_channels)
        
        # Initialize alphas (operation weights)
        self._init_alphas()
    
    def _build_operations(
        self,
        in_channels: int,
        out_channels: int
    ):
        """Build mixed operations for each edge."""
        self.ops = nn.ModuleDict()
        
        for edge in self.architecture.edges:
            key = f"{edge.source}_{edge.target}"
            
            # Create mixed operation
            ops = nn.ModuleList([
                self._get_operation(op_type, in_channels, out_channels)
                for op_type in CNNSearchSpace.CNN_OPS
            ])
            
            self.ops[key] = ops
    
    def _get_operation(
        self,
        op_type: OperationType,
        in_channels: int,
        out_channels: int
    ) -> nn.Module:
        """Get operation module for operation type."""
        if op_type == OperationType.SEP_CONV_3X3:
            return CNNOperations.sep_conv_3x3(in_channels, out_channels)
        elif op_type == OperationType.SEP_CONV_5X5:
            return CNNOperations.sep_conv_5x5(in_channels, out_channels)
        elif op_type == OperationType.MAX_POOL_3X3:
            return CNNOperations.max_pool_3x3()
        elif op_type == OperationType.AVG_POOL_3X3:
            return CNNOperations.avg_pool_3x3()
        elif op_type == OperationType.SKIP_CONNECT:
            return CNNOperations.skip_connect()
        elif op_type == OperationType.NONE:
            return CNNOperations.none()
        else:
            return nn.Identity()
    
    def _init_alphas(self):
        """Initialize operation weights."""
        num_ops = len(CNNSearchSpace.CNN_OPS)
        self.alphas = nn.Parameter(
            torch.ones(len(self.architecture.edges), num_ops) / num_ops
        )
    
    def forward(self, x: List[Tensor]) -> Tensor:
        """
        Forward pass through the cell.
        
        Args:
            x: List of input tensors (one per node)
            
        Returns:
            Output tensor
        """
        # This is a simplified version
        # Full implementation would follow DARTS' forward
        states = x if isinstance(x, list) else [x]
        
        for edge_idx, edge in enumerate(self.architecture.edges):
            if edge.source >= len(states) or edge.target >= len(states):
                continue
            
            key = f"{edge.source}_{edge.target}"
            
            if key not in self.ops:
                continue
            
            # Get mixed operation output
            op_weights = F.softmax(self.alphas[edge_idx], dim=0)
            
            for op_idx, op in enumerate(self.ops[key]):
                out = op(states[edge.source])
                if edge_idx == 0 and op_idx == 0:
                    result = op_weights[op_idx] * out
                else:
                    result = result + op_weights[op_idx] * out
            
            # Ensure we have a state for this target
            while len(states) <= edge.target:
                states.append(torch.zeros_like(states[0]))
            
            states[edge.target] = states[edge.target] + result
        
        return states[-1] if states else x


class NASBench101SearchSpace(CNNSearchSpace):
    """
    NAS-Bench-101 specific search space.
    
    NAS-Bench-101 contains 423,624 unique architectures trained
    on CIFAR-10. This search space allows sampling from the
    NAS-Bench-101 search space.
    """
    
    def __init__(
        self,
        max_nodes: int = 7,
        nasbench: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize NAS-Bench-101 search space.
        
        Args:
            max_nodes: Maximum nodes (fixed at 7 for NAS-Bench-101)
            nasbench: NAS-Bench-101 API instance
        """
        super().__init__(max_nodes=7, num_operations=5, **kwargs)
        
        self.nasbench = nasbench
        self.max_nodes = 7
        
        # NAS-Bench-101 specific ops
        self.NAS101_OPS = [
            'conv3x3-bn-relu',
            'conv1x1-bn-relu',
            'maxpool3x3',
        ]
    
    def sample(self) -> Architecture:
        """Sample from NAS-Bench-101."""
        if self.nasbench is None:
            # Return random architecture
            return super().sample()
        
        # Sample from NAS-Bench-101
        unique_hash = self.nasbench.sample_random()
        arch_spec = self.nasbench.get_metrics_from_hash(unique_hash)
        
        # Convert to our architecture format
        arch = Architecture(name=unique_hash)
        
        for i in range(self.max_nodes):
            arch.cells.append(Cell(node_index=i))
        
        # Add edges based on adjacency matrix
        # (This is a simplified conversion)
        
        return arch
    
    def query(self, architecture: Architecture) -> Dict[str, float]:
        """
        Query NAS-Bench-101 for architecture metrics.
        
        Args:
            architecture: Architecture to query
            
        Returns:
            Dictionary of metrics
        """
        if self.nasbench is None:
            return {}
        
        # Query NAS-Bench-101
        # (Implementation depends on NAS-Bench-101 API)
        
        return {}
