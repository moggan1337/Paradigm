"""
Base search space definitions for neural architecture search.

This module provides the foundational classes for defining search spaces
across different neural network architectures (CNN, RNN, Transformer).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np


class OperationType(Enum):
    """Types of operations available in search spaces."""
    
    # CNN Operations
    SEP_CONV_3X3 = "sep_conv_3x3"
    SEP_CONV_5X5 = "sep_conv_5x5"
    DIL_CONV_3X3 = "dil_conv_3x3"
    DIL_CONV_5X5 = "dil_conv_5x5"
    MAX_POOL_3X3 = "max_pool_3x3"
    AVG_POOL_3X3 = "avg_pool_3x3"
    SKIP_CONNECT = "skip_connect"
    NONE = "none"
    
    # RNN Operations
    RNN_CELL = "rnn_cell"
    LSTM_CELL = "lstm_cell"
    GRU_CELL = "gru_cell"
    
    # Attention Operations
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    FULL_ATTENTION = "full_attention"
    
    # Feed-forward Operations
    LINEAR = "linear"
    FFN = "ffn"
    GATED_FFN = "gated_ffn"


@dataclass
class Operation:
    """
    Represents a single operation in the architecture.
    
    Attributes:
        op_type: Type of operation
        parameters: Dictionary of operation-specific parameters
        input_indices: Indices of input nodes
        output_index: Index of output node
    """
    op_type: OperationType
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_indices: Tuple[int, ...] = (0,)
    output_index: int = 1
    
    def __hash__(self):
        return hash((
            self.op_type,
            tuple(sorted(self.parameters.items())),
            self.input_indices,
            self.output_index
        ))
    
    def __eq__(self, other):
        if not isinstance(other, Operation):
            return False
        return (
            self.op_type == other.op_type and
            self.parameters == other.parameters and
            self.input_indices == other.input_indices and
            self.output_index == other.output_index
        )
    
    def encode(self) -> np.ndarray:
        """
        Encode operation as a fixed-size vector.
        
        Returns:
            Numpy array encoding the operation
        """
        op_id = list(OperationType).index(self.op_type)
        params = list(self.parameters.values()) if self.parameters else [0.0]
        
        # Pad or truncate parameters
        max_params = 4
        params = params[:max_params] + [0.0] * (max_params - len(params))
        
        return np.array([op_id] + params + list(self.input_indices) + [self.output_index])
    
    @classmethod
    def decode(cls, encoding: np.ndarray) -> 'Operation':
        """Decode operation from encoding."""
        op_id = int(encoding[0])
        op_type = list(OperationType)[op_id]
        
        params = {}
        for i, key in enumerate(['param1', 'param2', 'param3', 'param4']):
            val = encoding[1 + i]
            if val != 0.0:
                params[key] = val
        
        input_indices = tuple(int(x) for x in encoding[5:8] if x >= 0)
        output_index = int(encoding[8]) if len(encoding) > 8 else 1
        
        return cls(op_type=op_type, parameters=params, 
                   input_indices=input_indices, output_index=output_index)


@dataclass
class Cell:
    """
    Represents a computational cell/node in the architecture.
    
    Attributes:
        node_index: Index of this node
        inputs: List of input edges (from_node_index, operation)
        operation: Operation applied to inputs
        output_shape: Shape of output tensor
    """
    node_index: int
    inputs: List[Tuple[int, Operation]] = field(default_factory=list)
    operation: Optional[Operation] = None
    output_shape: Optional[Tuple[int, ...]] = None
    
    def add_input(self, from_node: int, operation: Operation):
        """Add an input connection to this node."""
        self.inputs.append((from_node, operation))
    
    def get_output_dim(self) -> Optional[int]:
        """Get output dimension if known."""
        if self.output_shape and len(self.output_shape) >= 1:
            return self.output_shape[0]
        return None


@dataclass
class Edge:
    """
    Represents a directed edge between nodes in the DAG.
    
    Attributes:
        source: Source node index
        target: Target node index
        operation: Operation on this edge
        weight: Learnable weight (for DARTS)
    """
    source: int
    target: int
    operation: Operation
    weight: Optional[float] = None


@dataclass
class Architecture:
    """
    Represents a complete neural network architecture.
    
    This is the main data structure that encoders and decoders work with.
    
    Attributes:
        name: Optional name for the architecture
        cells: List of cells in the network
        edges: List of directed edges (DAG structure)
        metadata: Additional metadata about the architecture
    """
    name: Optional[str] = None
    cells: List[Cell] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Cached properties
    _encoding: Optional[np.ndarray] = None
    _hash: Optional[int] = None
    
    def __post_init__(self):
        """Initialize cached properties."""
        self._encoding = None
        self._hash = None
    
    def __hash__(self):
        """Hash the architecture for caching and comparison."""
        if self._hash is None:
            self._hash = hash((
                tuple(sorted(cell.node_index for cell in self.cells)),
                tuple(sorted(
                    (edge.source, edge.target, edge.operation)
                    for edge in self.edges
                ))
            ))
        return self._hash
    
    def __eq__(self, other):
        """Check architecture equality."""
        if not isinstance(other, Architecture):
            return False
        return hash(self) == hash(other)
    
    def num_parameters(self) -> int:
        """Estimate number of parameters in the architecture."""
        total = 0
        for edge in self.edges:
            if edge.operation.op_type == OperationType.SEP_CONV_3X3:
                total += 9  # Simplified estimate
            elif edge.operation.op_type == OperationType.SEP_CONV_5X5:
                total += 25
            elif edge.operation.op_type == OperationType.LINEAR:
                total += edge.operation.parameters.get('out_features', 64) * \
                         edge.operation.parameters.get('in_features', 64)
        return total
    
    def depth(self) -> int:
        """Get maximum depth (longest path) in the DAG."""
        if not self.cells:
            return 0
        max_depth = 0
        for cell in self.cells:
            if not cell.inputs:
                cell_depth = 1
            else:
                cell_depth = 1 + max(
                    self._get_node_depth(edge.source) 
                    for edge in self.edges 
                    if edge.target == cell.node_index
                )
            max_depth = max(max_depth, cell_depth)
        return max_depth
    
    def _get_node_depth(self, node_idx: int) -> int:
        """Get depth of a specific node."""
        for cell in self.cells:
            if cell.node_index == node_idx:
                if not cell.inputs:
                    return 1
                return 1 + max(
                    self._get_node_depth(edge.source)
                    for edge in self.edges
                    if edge.target == cell.node_index
                )
        return 1
    
    def encode(self) -> np.ndarray:
        """
        Encode architecture as a fixed-size vector.
        
        Returns:
            Numpy array representation of the architecture
        """
        if self._encoding is not None:
            return self._encoding
        
        if not self.edges:
            self._encoding = np.zeros(128)
            return self._encoding
        
        # Encode each edge
        edge_encodings = []
        for edge in self.edges:
            enc = edge.operation.encode()
            edge_encodings.append(np.concatenate([
                enc,
                np.array([edge.source, edge.target])
            ]))
        
        # Pad or truncate to fixed size
        max_edges = 20
        encoded = np.zeros((max_edges, 12))
        for i, edge_enc in enumerate(edge_encodings[:max_edges]):
            encoded[i] = edge_enc[:12]
        
        self._encoding = encoded.flatten()
        return self._encoding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary representation."""
        return {
            'name': self.name,
            'cells': [
                {
                    'node_index': cell.node_index,
                    'inputs': [(idx, op.op_type.value) for idx, op in cell.inputs],
                }
                for cell in self.cells
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'operation': edge.operation.op_type.value,
                    'parameters': edge.operation.parameters,
                }
                for edge in self.edges
            ],
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Architecture':
        """Create architecture from dictionary representation."""
        arch = cls(name=data.get('name'))
        
        # Create cells
        node_indices = set()
        for edge_data in data.get('edges', []):
            node_indices.add(edge_data['source'])
            node_indices.add(edge_data['target'])
        
        for idx in node_indices:
            arch.cells.append(Cell(node_index=idx))
        
        # Create edges
        for edge_data in data.get('edges', []):
            op_type = OperationType(edge_data['operation'])
            operation = Operation(
                op_type=op_type,
                parameters=edge_data.get('parameters', {})
            )
            arch.edges.append(Edge(
                source=edge_data['source'],
                target=edge_data['target'],
                operation=operation
            ))
        
        arch.metadata = data.get('metadata', {})
        return arch
    
    def visualize(self) -> str:
        """Generate a simple text visualization of the architecture."""
        lines = [f"Architecture: {self.name or 'unnamed'}"]
        lines.append(f"  Nodes: {len(self.cells)}")
        lines.append(f"  Edges: {len(self.edges)}")
        lines.append(f"  Depth: {self.depth()}")
        lines.append("\nEdges:")
        
        for edge in self.edges:
            lines.append(
                f"  {edge.source} --[{edge.operation.op_type.value}]--> {edge.target}"
            )
        
        return "\n".join(lines)


class SearchSpace(ABC):
    """
    Abstract base class for all search spaces.
    
    A search space defines:
    - The set of possible operations
    - How architectures are sampled
    - How architectures are mutated
    - How architectures are crossed over
    - How to build PyTorch models from architectures
    
    Subclasses should implement:
    - sample(): Sample a random architecture
    - mutate(): Mutate an existing architecture
    - crossover(): Combine two architectures
    - build(): Build PyTorch model
    - encode(): Convert to fixed-size vector
    """
    
    def __init__(
        self,
        max_nodes: int = 7,
        num_operations: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize search space.
        
        Args:
            max_nodes: Maximum number of nodes in the DAG
            num_operations: Number of available operations
            seed: Random seed for reproducibility
        """
        self.max_nodes = max_nodes
        self.num_operations = num_operations or len(OperationType)
        self.rng = np.random.RandomState(seed)
        
        if seed is not None:
            random.seed(seed)
    
    @abstractmethod
    def sample(self) -> Architecture:
        """
        Sample a random architecture from the search space.
        
        Returns:
            A randomly sampled Architecture object
        """
        pass
    
    @abstractmethod
    def mutate(
        self,
        architecture: Architecture,
        mutation_rate: float = 0.1
    ) -> Architecture:
        """
        Mutate an architecture by making small changes.
        
        Args:
            architecture: Architecture to mutate
            mutation_rate: Probability of mutation per element
            
        Returns:
            Mutated Architecture object
        """
        pass
    
    @abstractmethod
    def crossover(
        self,
        parent1: Architecture,
        parent2: Architecture
    ) -> Architecture:
        """
        Combine two architectures through crossover.
        
        Args:
            parent1: First parent architecture
            parent2: Second parent architecture
            
        Returns:
            Child architecture combining traits from both parents
        """
        pass
    
    @abstractmethod
    def build(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...] = (3, 32, 32),
        num_classes: int = 10
    ):
        """
        Build a PyTorch model from an architecture.
        
        Args:
            architecture: Architecture to build
            input_shape: Input tensor shape
            num_classes: Number of output classes
            
        Returns:
            PyTorch nn.Module
        """
        pass
    
    def encode(self, architecture: Architecture) -> np.ndarray:
        """
        Encode architecture as fixed-size vector.
        
        Args:
            architecture: Architecture to encode
            
        Returns:
            Numpy array encoding
        """
        return architecture.encode()
    
    def decode(self, encoding: np.ndarray) -> Architecture:
        """
        Decode architecture from fixed-size vector.
        
        Args:
            encoding: Numpy array encoding
            
        Returns:
            Decoded Architecture
        """
        raise NotImplementedError(
            "Decoding not supported for this search space. "
            "Use the specific search space class."
        )
    
    def get_operation_set(self) -> List[OperationType]:
        """
        Get the set of available operations.
        
        Returns:
            List of OperationType enums
        """
        return list(OperationType)[:self.num_operations]
    
    def render(self, architecture: Architecture) -> str:
        """
        Generate a human-readable representation.
        
        Args:
            architecture: Architecture to render
            
        Returns:
            String representation
        """
        return architecture.visualize()
    
    def get_distance(
        self,
        arch1: Architecture,
        arch2: Architecture
    ) -> float:
        """
        Calculate distance between two architectures.
        
        Args:
            arch1: First architecture
            arch2: Second architecture
            
        Returns:
            Distance metric (0 = identical)
        """
        enc1 = self.encode(arch1)
        enc2 = self.encode(arch2)
        
        # Normalized Euclidean distance
        diff = enc1 - enc2
        return np.sqrt(np.sum(diff ** 2)) / (np.sqrt(np.sum(enc1 ** 2)) + 1e-8)
    
    def is_valid(self, architecture: Architecture) -> bool:
        """
        Check if an architecture is valid.
        
        Args:
            architecture: Architecture to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check for cycles
        if self._has_cycle(architecture):
            return False
        
        # Check connectivity
        if not self._is_connected(architecture):
            return False
        
        return True
    
    def _has_cycle(self, architecture: Architecture) -> bool:
        """Check if the architecture contains a cycle (DAG violation)."""
        if not architecture.edges:
            return False
        
        # Build adjacency list
        adj = {i: [] for i in range(architecture.cells[-1].node_index + 1) if architecture.cells}
        for edge in architecture.edges:
            adj.setdefault(edge.source, []).append(edge.target)
        
        # DFS for cycle detection
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check all nodes
        for node in adj:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def _is_connected(self, architecture: Architecture) -> bool:
        """Check if all nodes are reachable from input nodes."""
        if not architecture.edges or not architecture.cells:
            return False
        
        max_node = max(c.node_index for c in architecture.cells)
        reachable = set()
        
        # Start from node 0 (assumed input)
        stack = [0]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            
            for edge in architecture.edges:
                if edge.source == node:
                    stack.append(edge.target)
        
        return len(reachable) == max_node + 1
    
    def bound(self, architecture: Architecture) -> Architecture:
        """
        Ensure architecture respects search space bounds.
        
        Args:
            architecture: Architecture to bound
            
        Returns:
            Bounded Architecture
        """
        # Remove nodes beyond max_nodes
        max_idx = max(c.node_index for c in architecture.cells) if architecture.cells else 0
        if max_idx >= self.max_nodes:
            # Prune to max_nodes
            architecture.cells = [c for c in architecture.cells if c.node_index < self.max_nodes]
            architecture.edges = [
                e for e in architecture.edges 
                if e.source < self.max_nodes and e.target < self.max_nodes
            ]
        
        return architecture


class MixedOperation(nn.Module):
    """
    Mixed operation for DARTS-style differentiable search.
    
    This module represents a mixture of all candidate operations,
    weighted by learnable parameters α.
    """
    
    def __init__(
        self,
        operations: List[nn.Module],
        alpha: Optional[Tensor] = None
    ):
        super().__init__()
        self.operations = nn.ModuleList(operations)
        num_ops = len(operations)
        
        if alpha is None:
            self.alpha = nn.Parameter(torch.zeros(num_ops))
        else:
            self.alpha = nn.Parameter(alpha)
    
    def forward(self, x: Tensor) -> Tensor:
        # Apply softmax to get weights
        weights = F.softmax(self.alpha, dim=0)
        
        # Sum weighted operations
        result = None
        for op, w in zip(self.operations, weights):
            out = op(x)
            if result is None:
                result = w * out
            else:
                result = result + w * out
        
        return result


# Import torch components here to avoid circular imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
