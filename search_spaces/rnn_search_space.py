"""
RNN Search Space for sequence modeling.

This module provides search spaces for recurrent neural networks,
including LSTM, GRU, and custom recurrent cells.
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


class RNNSearchSpace(SearchSpace):
    """
    RNN search space for sequence modeling.
    
    This search space explores different recurrent cell architectures,
    including the number of layers, hidden sizes, and connectivity patterns.
    
    Search Space Parameters:
    - Number of layers: [1, 4]
    - Hidden size: [64, 512]
    - Dropout: [0.0, 0.5]
    - Use attention: [True, False]
    - Cell type: [RNN, LSTM, GRU]
    
    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                  RNN Encoder                         │
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │   Input → [Embedding] → [RNN Cell × N] → Hidden      │
    │                                    │                │
    │                                    ↓                │
    │   Output ← [Linear] ← [RNN Cell × M] ←──────────────┘
    │                                                      │
    └─────────────────────────────────────────────────────┘
    """
    
    # RNN-specific operations
    RNN_OPS = [
        OperationType.RNN_CELL,
        OperationType.LSTM_CELL,
        OperationType.GRU_CELL,
    ]
    
    def __init__(
        self,
        max_layers: int = 4,
        hidden_sizes: List[int] = None,
        dropout_range: Tuple[float, float] = (0.0, 0.5),
        use_attention: bool = True,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        **kwargs
    ):
        """
        Initialize RNN search space.
        
        Args:
            max_layers: Maximum number of layers
            hidden_sizes: Available hidden sizes
            dropout_range: Min/max dropout rate
            use_attention: Whether to include attention
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
        """
        super().__init__(max_nodes=max_layers + 2, num_operations=3, **kwargs)
        
        self.max_layers = max_layers
        self.hidden_sizes = hidden_sizes or [64, 128, 256, 512]
        self.dropout_range = dropout_range
        self.use_attention = use_attention
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
    
    def sample(self) -> Architecture:
        """
        Sample a random RNN architecture.
        
        Returns:
            Random Architecture
        """
        arch = Architecture(name=f"rnn_{random.randint(0, 10000)}")
        
        # Sample configuration
        num_layers = random.randint(1, self.max_layers)
        hidden_size = random.choice(self.hidden_sizes)
        dropout = random.uniform(*self.dropout_range)
        cell_type = random.choice(self.RNN_OPS)
        
        # Store configuration in metadata
        arch.metadata = {
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'dropout': dropout,
            'cell_type': cell_type.value,
        }
        
        # Build nodes
        for i in range(num_layers + 2):
            arch.cells.append(Cell(node_index=i))
        
        # Add edges (layer connections)
        for layer in range(num_layers):
            # RNN cell operation
            operation = Operation(
                op_type=cell_type,
                parameters={
                    'hidden_size': hidden_size,
                    'layer': layer
                }
            )
            
            edge = Edge(
                source=layer,
                target=layer + 1,
                operation=operation
            )
            arch.edges.append(edge)
        
        # Add optional skip connection
        if random.random() > 0.5:
            skip_op = Operation(
                op_type=OperationType.SKIP_CONNECT,
                parameters={'type': 'residual'}
            )
            arch.edges.append(Edge(
                source=0,
                target=num_layers,
                operation=skip_op
            ))
        
        # Add attention if enabled
        if self.use_attention and random.random() > 0.5:
            attn_op = Operation(
                op_type=OperationType.SELF_ATTENTION,
                parameters={'hidden_size': hidden_size}
            )
            arch.edges.append(Edge(
                source=num_layers - 1,
                target=num_layers + 1,
                operation=attn_op
            ))
        
        return arch
    
    def mutate(
        self,
        architecture: Architecture,
        mutation_rate: float = 0.1
    ) -> Architecture:
        """
        Mutate an RNN architecture.
        
        Mutations:
        - Change hidden size
        - Add/remove layers
        - Change dropout
        - Add skip connections
        
        Args:
            architecture: Architecture to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated Architecture
        """
        arch = deepcopy(architecture)
        arch.name = f"rnn_mutated_{random.randint(0, 10000)}"
        
        # Mutate metadata
        if random.random() < mutation_rate:
            mutation_type = random.choice(['hidden_size', 'layers', 'dropout', 'cell_type'])
            
            if mutation_type == 'hidden_size':
                arch.metadata['hidden_size'] = random.choice(self.hidden_sizes)
            elif mutation_type == 'layers':
                arch.metadata['num_layers'] = max(1, min(
                    self.max_layers,
                    arch.metadata.get('num_layers', 2) + random.choice([-1, 1])
                ))
            elif mutation_type == 'dropout':
                arch.metadata['dropout'] = random.uniform(*self.dropout_range)
            elif mutation_type == 'cell_type':
                arch.metadata['cell_type'] = random.choice(self.RNN_OPS).value
        
        # Mutate edges
        for edge in arch.edges:
            if random.random() < mutation_rate:
                if 'hidden_size' in edge.operation.parameters:
                    edge.operation.parameters['hidden_size'] = arch.metadata['hidden_size']
        
        return arch
    
    def crossover(
        self,
        parent1: Architecture,
        parent2: Architecture
    ) -> Architecture:
        """
        Crossover two RNN architectures.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child architecture
        """
        child = Architecture(name=f"rnn_cross_{random.randint(0, 10000)}")
        
        # Randomly choose metadata from parents
        if random.random() > 0.5:
            child.metadata = deepcopy(parent1.metadata)
        else:
            child.metadata = deepcopy(parent2.metadata)
        
        # Combine edges
        edges1 = {e.source: e for e in parent1.edges}
        edges2 = {e.source: e for e in parent2.edges}
        
        for key in set(edges1.keys()) | set(edges2.keys()):
            if key in edges1 and key in edges2:
                chosen = random.choice([edges1[key], edges2[key]])
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
    
    def build(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...] = None,
        num_classes: int = None
    ) -> nn.Module:
        """
        Build PyTorch RNN model from architecture.
        
        Args:
            architecture: Architecture to build
            input_shape: Input tensor shape (not used for RNN)
            num_classes: Number of output classes
            
        Returns:
            PyTorch nn.Module
        """
        config = architecture.metadata
        
        return RNNModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            num_classes=num_classes or 10,
            dropout=config.get('dropout', 0.0),
            cell_type=config.get('cell_type', 'lstm_cell'),
        )


class RNNModel(nn.Module):
    """
    PyTorch RNN model.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
        cell_type: str = 'lstm_cell'
    ):
        """
        Initialize RNN model.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_size: Hidden size
            num_layers: Number of layers
            num_classes: Number of output classes
            dropout: Dropout rate
            cell_type: Cell type ('rnn_cell', 'lstm_cell', 'gru_cell')
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN
        rnn_class = self._get_rnn_class(cell_type)
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Output
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
        self._init_weights()
    
    def _get_rnn_class(self, cell_type: str):
        """Get RNN class from cell type string."""
        if cell_type == 'rnn_cell':
            return nn.RNN
        elif cell_type == 'gru_cell':
            return nn.GRU
        else:  # lstm_cell
            return nn.LSTM
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: Tensor, lengths: Tensor = None) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len]
            lengths: Sequence lengths
            
        Returns:
            Output logits [batch, num_classes]
        """
        # Embed
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        # Pack padded sequence if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # RNN
        if self.cell_type == 'lstm_cell':
            output, (hidden, cell) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)
        
        # Unpack
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )
        
        # Get last hidden state (bidirectional)
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        # Output
        output = self.dropout(hidden_concat)
        output = self.fc(output)
        
        return output


class RNNCellSearchSpace(SearchSpace):
    """
    Search space for single RNN cells.
    
    This search space explores different internal structures
    of RNN cells, including activation functions, gates, and connections.
    """
    
    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 64,
        num_cell_ops: int = 5,
        **kwargs
    ):
        """
        Initialize RNN cell search space.
        
        Args:
            input_size: Input size
            hidden_size: Hidden size
            num_cell_ops: Number of cell operations
        """
        super().__init__(max_nodes=5, num_operations=num_cell_ops, **kwargs)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
    
    def sample(self) -> Architecture:
        """Sample a random RNN cell architecture."""
        arch = Architecture(name=f"rnn_cell_{random.randint(0, 10000)}")
        
        # Sample internal connections
        for target in range(1, self.max_nodes):
            for source in range(target + 1):
                if random.random() > 0.3:  # 70% probability of connection
                    op_idx = random.randint(0, self.num_operations - 1)
                    op_type = list(OperationType)[op_idx]
                    
                    operation = Operation(
                        op_type=op_type,
                        parameters={'input_size': self.input_size, 'hidden_size': self.hidden_size}
                    )
                    
                    edge = Edge(
                        source=source,
                        target=target,
                        operation=operation
                    )
                    arch.edges.append(edge)
        
        for i in range(self.max_nodes):
            arch.cells.append(Cell(node_index=i))
        
        return arch
    
    def mutate(self, architecture: Architecture, mutation_rate: float = 0.1) -> Architecture:
        """Mutate an RNN cell."""
        return deepcopy(architecture)
    
    def crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Crossover RNN cells."""
        return deepcopy(parent1)
    
    def build(self, architecture: Architecture, input_shape: Tuple[int, ...] = None, num_classes: int = None) -> nn.Module:
        """Build RNN cell."""
        return RNNCell(
            hidden_size=self.hidden_size,
            input_size=self.input_size
        )


class RNNCell(nn.Module):
    """
    Custom RNN cell built from architecture.
    """
    
    def __init__(self, hidden_size: int, input_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Simple default cell
        self.ih = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: Tensor, h: Tensor = None) -> Tensor:
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        h_new = torch.tanh(self.ih(x) + self.hh(h))
        return h_new


class Seq2SeqSearchSpace(RNNSearchSpace):
    """
    Sequence-to-sequence search space.
    
    This search space explores encoder-decoder architectures
    for tasks like translation and summarization.
    """
    
    def __init__(
        self,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        **kwargs
    ):
        """
        Initialize Seq2Seq search space.
        
        Args:
            encoder_layers: Number of encoder layers
            decoder_layers: Number of decoder layers
        """
        super().__init__(max_layers=max(encoder_layers, decoder_layers), **kwargs)
        
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
    
    def sample(self) -> Architecture:
        """Sample a seq2seq architecture."""
        arch = Architecture(name=f"seq2seq_{random.randint(0, 10000)}")
        
        # Encoder configuration
        arch.metadata = {
            'encoder_layers': self.encoder_layers,
            'decoder_layers': self.decoder_layers,
            'hidden_size': random.choice(self.hidden_sizes),
            'dropout': random.uniform(*self.dropout_range),
            'use_attention': random.random() > 0.3,
        }
        
        # Build cells
        for i in range(self.encoder_layers + self.decoder_layers + 2):
            arch.cells.append(Cell(node_index=i))
        
        return arch
    
    def build(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...] = None,
        num_classes: int = None
    ) -> nn.Module:
        """Build seq2seq model."""
        config = architecture.metadata
        
        return Seq2SeqModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=config.get('hidden_size', 128),
            encoder_layers=config.get('encoder_layers', 3),
            decoder_layers=config.get('decoder_layers', 3),
            dropout=config.get('dropout', 0.0),
            use_attention=config.get('use_attention', True),
        )


class Seq2SeqModel(nn.Module):
    """
    Sequence-to-sequence model.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        encoder_layers: int,
        decoder_layers: int,
        dropout: float = 0.0,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        
        # Encoder
        self.encoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim, hidden_size,
            encoder_layers, dropout=dropout if encoder_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder
        self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.LSTM(
            embedding_dim, hidden_size,
            decoder_layers, dropout=dropout if decoder_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention
        if use_attention:
            self.attention = nn.Linear(hidden_size * 2, hidden_size)
            self.decoder_input = nn.Linear(hidden_size + embedding_dim, embedding_dim)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_lengths: Tensor = None
    ) -> Tensor:
        """Forward pass."""
        batch_size = src.size(0)
        
        # Encode
        src_embedded = self.dropout(self.encoder_embedding(src))
        
        if src_lengths is not None:
            src_embedded = nn.utils.rnn.pack_padded_sequence(
                src_embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(src_embedded)
        
        if src_lengths is not None:
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                encoder_outputs, batch_first=True
            )
        
        # Decode
        tgt_embedded = self.dropout(self.decoder_embedding(tgt))
        decoder_outputs, _ = self.decoder(tgt_embedded, (encoder_hidden, encoder_cell))
        
        # Project to vocabulary
        outputs = self.fc(decoder_outputs)
        
        return outputs
