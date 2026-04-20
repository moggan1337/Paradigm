"""
Transformer Search Space for attention-based architectures.

This module provides search spaces for Transformer models,
including encoder-decoder architectures and attention mechanisms.
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


class TransformerSearchSpace(SearchSpace):
    """
    Transformer search space for attention-based architectures.
    
    This search space explores different Transformer configurations,
    including number of layers, attention heads, and feed-forward dimensions.
    
    Search Space Parameters:
    - Number of layers: [2, 12]
    - Number of heads: [4, 16]
    - Head dimension: [32, 128]
    - Feed-forward dimension: [512, 2048]
    - Attention type: [scaled_dot, multihead, linear]
    
    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                  Transformer Block                  │
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │   Input                                              │
    │     ↓                                                │
    │   ┌──────────────────────────────────┐              │
    │   │        Multi-Head Attention       │              │
    │   │   [Head 1] [Head 2] ... [Head N]  │              │
    │   └──────────────┬───────────────────┘              │
    │                  ↓ (Add & Norm)                     │
    │   ┌──────────────────────────────────┐              │
    │   │        Feed-Forward Network       │              │
    │   │   Linear → ReLU → Linear          │              │
    │   └──────────────┬───────────────────┘              │
    │                  ↓ (Add & Norm)                     │
    │   Output                                              │
    │                                                      │
    └─────────────────────────────────────────────────────┘
    """
    
    # Attention operation types
    ATTENTION_OPS = [
        OperationType.SELF_ATTENTION,
        OperationType.CROSS_ATTENTION,
        OperationType.FULL_ATTENTION,
    ]
    
    def __init__(
        self,
        max_layers: int = 12,
        num_heads_options: List[int] = None,
        head_dim_options: List[int] = None,
        ff_dim_options: List[int] = None,
        use_rope: bool = True,
        use_glu: bool = False,
        **kwargs
    ):
        """
        Initialize Transformer search space.
        
        Args:
            max_layers: Maximum number of layers
            num_heads_options: Available number of attention heads
            head_dim_options: Available head dimensions
            ff_dim_options: Available feed-forward dimensions
            use_rope: Whether to use Rotary Position Embedding
            use_glu: Whether to use GLU activation
        """
        super().__init__(max_nodes=max_layers + 2, num_operations=3, **kwargs)
        
        self.max_layers = max_layers
        self.num_heads_options = num_heads_options or [4, 8, 16]
        self.head_dim_options = head_dim_options or [32, 64, 128]
        self.ff_dim_options = ff_dim_options or [512, 1024, 2048]
        self.use_rope = use_rope
        self.use_glu = use_glu
    
    def sample(self) -> Architecture:
        """
        Sample a random Transformer architecture.
        
        Returns:
            Random Architecture
        """
        arch = Architecture(name=f"transformer_{random.randint(0, 10000)}")
        
        # Sample configuration
        num_layers = random.randint(2, self.max_layers)
        num_heads = random.choice(self.num_heads_options)
        head_dim = random.choice(self.head_dim_options)
        ff_dim = random.choice(self.ff_dim_options)
        
        # Store configuration
        model_dim = num_heads * head_dim
        arch.metadata = {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'model_dim': model_dim,
            'ff_dim': ff_dim,
            'use_rope': self.use_rope and random.random() > 0.5,
            'use_glu': self.use_glu and random.random() > 0.5,
            'dropout': random.uniform(0.0, 0.2),
        }
        
        # Build nodes (input, layers, output)
        for i in range(num_layers + 2):
            arch.cells.append(Cell(node_index=i))
        
        # Add transformer layer edges
        for layer in range(num_layers):
            # Self-attention
            attn_op = Operation(
                op_type=OperationType.SELF_ATTENTION,
                parameters={
                    'num_heads': num_heads,
                    'head_dim': head_dim,
                    'layer': layer
                }
            )
            arch.edges.append(Edge(
                source=layer,
                target=layer + 1,
                operation=attn_op
            ))
            
            # Feed-forward (with GLU option)
            ff_op = Operation(
                op_type=OperationType.FFN,
                parameters={
                    'ff_dim': ff_dim,
                    'model_dim': model_dim,
                    'use_glu': arch.metadata['use_glu']
                }
            )
            arch.edges.append(Edge(
                source=layer + 1,
                target=layer + 2,
                operation=ff_op
            ))
        
        # Add skip connections
        if random.random() > 0.3:
            skip_op = Operation(
                op_type=OperationType.SKIP_CONNECT,
                parameters={'type': 'layer_norm'}
            )
            arch.edges.append(Edge(
                source=0,
                target=num_layers + 1,
                operation=skip_op
            ))
        
        return arch
    
    def mutate(
        self,
        architecture: Architecture,
        mutation_rate: float = 0.1
    ) -> Architecture:
        """
        Mutate a Transformer architecture.
        
        Mutations:
        - Change number of layers
        - Change attention heads/dimension
        - Change feed-forward dimension
        - Add/remove RoPE or GLU
        
        Args:
            architecture: Architecture to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated Architecture
        """
        arch = deepcopy(architecture)
        arch.name = f"transformer_mutated_{random.randint(0, 10000)}"
        
        if random.random() < mutation_rate:
            mutation_type = random.choice([
                'num_layers', 'num_heads', 'head_dim', 'ff_dim', 'dropout'
            ])
            
            if mutation_type == 'num_layers':
                arch.metadata['num_layers'] = max(2, min(
                    self.max_layers,
                    arch.metadata.get('num_layers', 6) + random.choice([-1, 1, 2])
                ))
            elif mutation_type == 'num_heads':
                arch.metadata['num_heads'] = random.choice(self.num_heads_options)
                arch.metadata['model_dim'] = arch.metadata['num_heads'] * arch.metadata['head_dim']
            elif mutation_type == 'head_dim':
                arch.metadata['head_dim'] = random.choice(self.head_dim_options)
                arch.metadata['model_dim'] = arch.metadata['num_heads'] * arch.metadata['head_dim']
            elif mutation_type == 'ff_dim':
                arch.metadata['ff_dim'] = random.choice(self.ff_dim_options)
            elif mutation_type == 'dropout':
                arch.metadata['dropout'] = random.uniform(0.0, 0.2)
        
        return arch
    
    def crossover(
        self,
        parent1: Architecture,
        parent2: Architecture
    ) -> Architecture:
        """
        Crossover two Transformer architectures.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child architecture
        """
        child = Architecture(name=f"transformer_cross_{random.randint(0, 10000)}")
        
        # Randomly choose metadata
        if random.random() > 0.5:
            child.metadata = deepcopy(parent1.metadata)
        else:
            child.metadata = deepcopy(parent2.metadata)
        
        # Use parent with more layers
        num_layers = max(
            len(parent1.metadata.get('num_layers', 6)),
            len(parent2.metadata.get('num_layers', 6))
        )
        child.metadata['num_layers'] = num_layers
        
        # Copy cells
        for i in range(num_layers + 2):
            child.cells.append(Cell(node_index=i))
        
        # Copy edges from both parents
        for edge in parent1.edges + parent2.edges:
            if edge.source < num_layers and edge.target < num_layers + 2:
                child.edges.append(deepcopy(edge))
        
        return child
    
    def build(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...] = None,
        num_classes: int = None
    ) -> nn.Module:
        """
        Build PyTorch Transformer model from architecture.
        
        Args:
            architecture: Architecture to build
            input_shape: Input tensor shape
            num_classes: Number of output classes
            
        Returns:
            PyTorch nn.Module
        """
        config = architecture.metadata
        
        return TransformerModel(
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            head_dim=config.get('head_dim', 64),
            ff_dim=config.get('ff_dim', 2048),
            dropout=config.get('dropout', 0.1),
            use_rope=config.get('use_rope', False),
            num_classes=num_classes or 10,
        )


class TransformerModel(nn.Module):
    """
    Transformer model built from architecture.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        num_classes: int = 10,
        vocab_size: int = 50000,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        self.model_dim = num_heads * head_dim
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, self.model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, self.model_dim) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                num_heads=num_heads,
                head_dim=head_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                use_rope=use_rope
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.model_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Forward pass."""
        batch_size, seq_len = x.size()
        
        # Embed
        x = self.embedding(x) * np.sqrt(self.model_dim)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        # Global average pooling and classify
        x = x.mean(dim=1)
        x = self.fc(x)
        
        return x


class TransformerLayer(nn.Module):
    """
    Single Transformer layer.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_rope: bool = False
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.model_dim = num_heads * head_dim
        self.use_rope = use_rope
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            model_dim=self.model_dim,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(self.model_dim)
        self.norm2 = nn.LayerNorm(self.model_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Forward pass with residual connections."""
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.model_dim = num_heads * head_dim
        
        assert self.model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.q_proj = nn.Linear(self.model_dim, self.model_dim)
        self.k_proj = nn.Linear(self.model_dim, self.model_dim)
        self.v_proj = nn.Linear(self.model_dim, self.model_dim)
        
        # Output projection
        self.out_proj = nn.Linear(self.model_dim, self.model_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(head_dim)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        batch_size = query.size(0)
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.model_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network with optional GLU activation.
    """
    
    def __init__(
        self,
        model_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_glu: bool = False
    ):
        super().__init__()
        
        self.use_glu = use_glu
        
        if use_glu:
            # GLU variant: Linear → GLU → Linear
            self.w1 = nn.Linear(model_dim, ff_dim)
            self.w2 = nn.Linear(ff_dim, model_dim)
        else:
            # Standard: Linear → ReLU → Dropout → Linear → Dropout
            self.w1 = nn.Linear(model_dim, ff_dim)
            self.w2 = nn.Linear(ff_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if self.use_glu:
            # GLU activation
            x = self.w1(x)
            x = F.glu(x, dim=-1)
            x = self.dropout(self.w2(x))
        else:
            # Standard FFN
            x = self.dropout(F.gelu(self.w1(x)))
            x = self.w2(x)
            x = self.dropout(x)
        
        return x


class AttentionSearchSpace(TransformerSearchSpace):
    """
    Search space for attention mechanisms specifically.
    
    This space explores different attention variants including:
    - Scaled dot-product attention
    - Multi-head attention
    - Linear (Performer-style) attention
    - Flash attention variants
    """
    
    def __init__(
        self,
        attention_types: List[str] = None,
        **kwargs
    ):
        """
        Initialize attention search space.
        
        Args:
            attention_types: List of attention types to include
            **kwargs: Base Transformer search space arguments
        """
        super().__init__(**kwargs)
        
        self.attention_types = attention_types or [
            'scaled_dot', 'multihead', 'linear', 'flash'
        ]
    
    def sample(self) -> Architecture:
        """Sample an attention architecture."""
        arch = super().sample()
        
        # Choose attention type
        attn_type = random.choice(self.attention_types)
        arch.metadata['attention_type'] = attn_type
        
        return arch


class VisionTransformerSearchSpace(SearchSpace):
    """
    Vision Transformer (ViT) search space.
    
    This space explores different ViT configurations including:
    - Patch embedding sizes
    - Transformer block configurations
    - Class token and positional embedding strategies
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        max_layers: int = 12,
        **kwargs
    ):
        super().__init__(max_nodes=max_layers + 3, num_operations=4, **kwargs)
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
    
    def sample(self) -> Architecture:
        """Sample a ViT architecture."""
        arch = Architecture(name=f"vit_{random.randint(0, 10000)}")
        
        # Sample configuration
        arch.metadata = {
            'num_layers': random.randint(4, 12),
            'num_heads': random.choice([6, 8, 12, 16]),
            'head_dim': random.choice([48, 64, 96]),
            'mlp_dim': random.choice([3072, 4096, 6144]),
            'patch_size': self.patch_size,
            'use_cls_token': random.random() > 0.3,
            'dropout': random.uniform(0.0, 0.2),
        }
        
        num_layers = arch.metadata['num_layers']
        
        # Build cells
        for i in range(num_layers + 3):
            arch.cells.append(Cell(node_index=i))
        
        return arch
    
    def mutate(self, architecture: Architecture, mutation_rate: float = 0.1) -> Architecture:
        """Mutate ViT architecture."""
        return deepcopy(architecture)
    
    def crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Crossover ViT architectures."""
        return deepcopy(parent1)
    
    def build(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        num_classes: int = 1000
    ) -> nn.Module:
        """Build ViT model."""
        config = architecture.metadata
        
        return VisionTransformer(
            image_size=self.image_size,
            patch_size=config['patch_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            head_dim=config['head_dim'],
            mlp_dim=config['mlp_dim'],
            num_classes=num_classes,
            use_cls_token=config['use_cls_token'],
            dropout=config['dropout'],
        )


class VisionTransformer(nn.Module):
    """
    Vision Transformer model.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_dim: int = 3072,
        num_classes: int = 1000,
        use_cls_token: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.use_cls_token = use_cls_token
        
        model_dim = num_heads * head_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, model_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Class token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        
        # Positional embedding
        num_positions = self.num_patches + (1 if use_cls_token else 0)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, model_dim) * 0.02)
        
        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerLayer(
                num_heads=num_heads,
                head_dim=head_dim,
                ff_dim=mlp_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(model_dim)
        self.fc = nn.Linear(model_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.cls_token, std=0.02) if hasattr(self, 'cls_token') else None
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, model_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, model_dim]
        
        # Add cls token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Get cls token output
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        
        x = self.fc(x)
        
        return x
