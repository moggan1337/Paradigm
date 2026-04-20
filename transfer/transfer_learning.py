"""
Transfer learning implementations for neural architecture search.

This module provides various methods for transferring knowledge
across different NAS tasks, including weight inheritance,
performance transfer, and architecture encoding transfer.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.search_space import SearchSpace, Architecture


@dataclass
class TransferConfig:
    """Configuration for transfer learning."""
    
    source_task: str
    target_task: str
    method: str = "weight_inheritance"
    similarity_threshold: float = 0.5
    weight_ratio: float = 1.0


class TransferLearning:
    """
    Base class for transfer learning in NAS.
    
    Transfer learning enables knowledge sharing across different
    NAS tasks, reducing the search cost on new tasks by
    leveraging knowledge from previously explored tasks.
    
    ┌─────────────────────────────────────────────────────────┐
    │              Transfer Learning Framework                 │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │   Source Task (CIFAR-10)                                │
    │   ┌──────────────────────────────────┐                  │
    │   │  Search History                 │                  │
    │   │  - Architectures                 │                  │
    │   │  - Performances                  │                  │
    │   │  - Weights                       │                  │
    │   └──────────────┬───────────────────┘                  │
    │                  │                                        │
    │   Transfer Method:                                       │
    │   - Weight Inheritance                                    │
    │   - Performance Transfer                                  │
    │   - Architecture Encoding                                │
    │                  ↓                                        │
    │   Target Task (ImageNet)                                 │
    │   ┌──────────────────────────────────┐                  │
    │   │  Informed Search                 │                  │
    │   │  - Reduced search space          │                  │
    │   │  - Better initialization          │                  │
    │   │  - Faster convergence            │                  │
    │   └──────────────────────────────────┘                  │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[TransferConfig] = None
    ):
        """
        Initialize transfer learning.
        
        Args:
            search_space: Target search space
            config: Transfer configuration
        """
        self.search_space = search_space
        self.config = config or TransferConfig(
            source_task="unknown",
            target_task="unknown"
        )
        
        # Source task data
        self.source_architectures: List[Architecture] = []
        self.source_performances: Dict[str, float] = {}
        self.source_weights: Dict[str, nn.Module] = {}
    
    def add_source_data(
        self,
        architecture: Architecture,
        performance: float,
        weights: Optional[nn.Module] = None
    ):
        """
        Add data from source task.
        
        Args:
            architecture: Architecture from source task
            performance: Performance on source task
            weights: Trained model weights
        """
        arch_name = architecture.name or f"arch_{len(self.source_architectures)}"
        
        self.source_architectures.append(architecture)
        self.source_performances[arch_name] = performance
        
        if weights is not None:
            self.source_weights[arch_name] = deepcopy(weights)
    
    def transfer(
        self,
        target_architecture: Architecture
    ) -> Optional[Dict[str, Any]]:
        """
        Transfer knowledge to a target architecture.
        
        Args:
            target_architecture: Architecture to transfer to
            
        Returns:
            Transfer metadata
        """
        raise NotImplementedError
    
    def get_similar_architectures(
        self,
        architecture: Architecture,
        top_k: int = 5
    ) -> List[Tuple[Architecture, float]]:
        """
        Find similar architectures from source task.
        
        Args:
            architecture: Target architecture
            top_k: Number of similar architectures to return
            
        Returns:
            List of (architecture, similarity) tuples
        """
        similarities = []
        
        for source_arch in self.source_architectures:
            sim = self.search_space.get_distance(architecture, source_arch)
            similarities.append((source_arch, sim))
        
        # Sort by similarity (lower = more similar for distance)
        similarities.sort(key=lambda x: x[1])
        
        return similarities[:top_k]


class WeightInheritance(TransferLearning):
    """
    Weight inheritance transfer learning.
    
    Transfers weights from similar source architectures to
    initialize target architectures, reducing training time.
    
    Reference:
        Wei, T., Wang, C., Liu, Y., Zhang, B., Sun, J., & Lin, C. (2018).
        Rethinking the Search Space for Neural Architecture
        with Effective Weight Sharing. IJCAI 2019.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        strategy: str = "last_layer",
        similarity_threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize weight inheritance.
        
        Args:
            search_space: Target search space
            strategy: Inheritance strategy
                - "full": Transfer all weights
                - "partial": Transfer weights from similar layers
                - "last_layer": Only transfer final layer weights
            similarity_threshold: Minimum similarity for transfer
        """
        super().__init__(search_space, **kwargs)
        
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
    
    def transfer(
        self,
        target_architecture: Architecture
    ) -> Optional[Dict[str, Any]]:
        """
        Transfer weights to target architecture.
        
        Args:
            target_architecture: Architecture to transfer to
            
        Returns:
            Dictionary with transfer metadata
        """
        # Find most similar architecture
        similar_archs = self.get_similar_architectures(
            target_architecture,
            top_k=1
        )
        
        if not similar_archs:
            return None
        
        best_source, similarity = similar_archs[0]
        
        if similarity > self.similarity_threshold:
            return None  # Too different
        
        # Get source weights
        source_name = best_source.name or str(len(self.source_architectures))
        
        if source_name not in self.source_weights:
            return None
        
        source_weights = self.source_weights[source_name]
        
        # Transfer weights
        transferred_weights = self._transfer_weights(
            source_weights,
            target_architecture
        )
        
        return {
            'source_architecture': best_source.name,
            'similarity': similarity,
            'strategy': self.strategy,
            'transferred_weights': transferred_weights,
        }
    
    def _transfer_weights(
        self,
        source_weights: nn.Module,
        target_architecture: Architecture
    ) -> Dict[str, Tensor]:
        """
        Transfer weights based on strategy.
        
        Args:
            source_weights: Source model weights
            target_architecture: Target architecture
            
        Returns:
            Dictionary of transferred weights
        """
        transferred = {}
        
        if self.strategy == "full":
            # Transfer all weights (requires identical architecture)
            for name, param in source_weights.named_parameters():
                transferred[name] = param.data.clone()
        
        elif self.strategy == "last_layer":
            # Transfer only classifier weights
            for name, param in source_weights.named_parameters():
                if 'classifier' in name or 'fc' in name or 'head' in name:
                    transferred[name] = param.data.clone()
        
        elif self.strategy == "partial":
            # Transfer weights from similar layers
            # Simplified - would need layer matching
            pass
        
        return transferred


class PerformanceTransfer(TransferLearning):
    """
    Performance-based transfer learning.
    
    Uses observed performances from similar architectures
    on the source task to initialize predictions for the
    target task.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        similarity_metric: str = "topology",
        k_neighbors: int = 5,
        **kwargs
    ):
        """
        Initialize performance transfer.
        
        Args:
            search_space: Target search space
            similarity_metric: Metric for measuring architecture similarity
            k_neighbors: Number of neighbors for prediction
        """
        super().__init__(search_space, **kwargs)
        
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
    
    def transfer(
        self,
        target_architecture: Architecture
    ) -> Optional[Dict[str, Any]]:
        """
        Transfer performance predictions to target architecture.
        
        Args:
            target_architecture: Architecture to predict for
            
        Returns:
            Dictionary with performance prediction
        """
        # Find similar architectures
        similar_archs = self.get_similar_architectures(
            target_architecture,
            top_k=self.k_neighbors
        )
        
        if not similar_archs:
            return None
        
        # Weighted average of performances
        total_weight = 0.0
        weighted_perf = 0.0
        
        for source_arch, similarity in similar_archs:
            # Similarity is distance, convert to weight
            weight = 1.0 / (similarity + 1e-8)
            
            source_name = source_arch.name or str(len(self.source_architectures))
            performance = self.source_performances.get(source_name, 0.0)
            
            weighted_perf += weight * performance
            total_weight += weight
        
        predicted_performance = weighted_perf / total_weight if total_weight > 0 else 0.0
        
        return {
            'predicted_performance': predicted_performance,
            'source_architectures': [arch.name for arch, _ in similar_archs],
            'similarities': [sim for _, sim in similar_archs],
            'num_neighbors': len(similar_archs),
        }
    
    def predict_performance(
        self,
        architecture: Architecture
    ) -> float:
        """
        Predict performance on target task.
        
        Args:
            architecture: Architecture to predict for
            
        Returns:
            Predicted performance
        """
        result = self.transfer(architecture)
        
        if result is None:
            return 0.0
        
        return result.get('predicted_performance', 0.0)


class ArchitectureTransfer(TransferLearning):
    """
    Architecture encoding transfer.
    
    Transfers learned architecture patterns or embeddings
    from the source task to guide search on the target task.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        embedding_dim: int = 64,
        alignment: str = "optimal_transport",
        **kwargs
    ):
        """
        Initialize architecture transfer.
        
        Args:
            search_space: Target search space
            embedding_dim: Dimension for architecture embeddings
            alignment: Method for aligning architectures
        """
        super().__init__(search_space, **kwargs)
        
        self.embedding_dim = embedding_dim
        self.alignment = alignment
        
        # Architecture encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.search_space.max_nodes * 12, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def encode_architecture(self, architecture: Architecture) -> Tensor:
        """
        Encode architecture to embedding.
        
        Args:
            architecture: Architecture to encode
            
        Returns:
            Embedding tensor
        """
        encoding = self.search_space.encode(architecture)
        
        # Handle variable-length encodings
        if len(encoding) < self.search_space.max_nodes * 12:
            encoding = np.pad(encoding, (0, self.search_space.max_nodes * 12 - len(encoding)))
        elif len(encoding) > self.search_space.max_nodes * 12:
            encoding = encoding[:self.search_space.max_nodes * 12]
        
        x = torch.tensor(encoding, dtype=torch.float32).unsqueeze(0)
        
        return self.encoder(x).squeeze(0)
    
    def align_embeddings(
        self,
        source_embeddings: List[Tensor],
        target_embeddings: List[Tensor]
    ) -> Tensor:
        """
        Align source and target embeddings.
        
        Args:
            source_embeddings: Embeddings from source task
            target_embeddings: Embeddings from target task
            
        Returns:
            Aligned embeddings
        """
        if self.alignment == "optimal_transport":
            # Simplified optimal transport alignment
            return self._optimal_transport_align(source_embeddings, target_embeddings)
        else:
            # No alignment
            return torch.stack(target_embeddings).mean(dim=0)
    
    def _optimal_transport_align(
        self,
        source_embeddings: List[Tensor],
        target_embeddings: List[Tensor]
    ) -> Tensor:
        """Align using optimal transport."""
        # Simplified implementation
        source_stack = torch.stack(source_embeddings)
        target_stack = torch.stack(target_embeddings)
        
        # Compute transport plan (simplified)
        C = torch.cdist(source_stack, target_stack)
        
        # Sinkhorn algorithm would go here
        # For now, just return target mean
        return target_stack.mean(dim=0)
    
    def transfer(self, target_architecture: Architecture) -> Dict[str, Any]:
        """Get architecture transfer info."""
        # Encode target
        target_embedding = self.encode_architecture(target_architecture)
        
        # Find similar source architectures
        similar = self.get_similar_architectures(target_architecture, top_k=5)
        
        # Get source embeddings
        source_embeddings = []
        for arch, _ in similar:
            emb = self.encode_architecture(arch)
            source_embeddings.append(emb)
        
        # Align
        aligned = self.align_embeddings(source_embeddings, [target_embedding])
        
        return {
            'target_embedding': target_embedding.detach().numpy(),
            'aligned_embedding': aligned.detach().numpy(),
            'similar_architectures': [arch.name for arch, _ in similar],
        }


class MetaNAS:
    """
    Meta-learning for NAS.
    
    Uses meta-learning to learn how to search efficiently
    across different tasks.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        inner_steps: int = 5
    ):
        """
        Initialize MetaNAS.
        
        Args:
            search_space: Search space
            meta_lr: Meta-learning rate
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner loop steps
        """
        self.search_space = search_space
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
        # Meta-learner (predicts architecture reward)
        self.meta_model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        self.optimizer = torch.optim.Adam(
            self.meta_model.parameters(),
            lr=meta_lr
        )
    
    def meta_train(
        self,
        tasks: List[Dict[str, Any]],
        n_epochs: int = 100
    ):
        """
        Meta-train on multiple tasks.
        
        Args:
            tasks: List of task data (each with architectures and performances)
            n_epochs: Number of meta-training epochs
        """
        for epoch in range(n_epochs):
            meta_loss = 0.0
            
            for task in tasks:
                architectures = task['architectures']
                performances = task['performances']
                
                # Inner loop: adapt to task
                adapted_weights = []
                
                for arch, perf in zip(architectures, performances):
                    # Compute gradient for this task
                    encoding = self.search_space.encode(arch)
                    encoding = torch.tensor(encoding, dtype=torch.float32)
                    
                    if len(encoding) < 128:
                        encoding = F.pad(encoding, (0, 128 - len(encoding)))
                    elif len(encoding) > 128:
                        encoding = encoding[:128]
                    
                    pred = self.meta_model(encoding)
                    loss = F.mse_loss(pred.squeeze(), torch.tensor(perf))
                    
                    # Gradient step
                    grad = torch.autograd.grad(loss, self.meta_model.parameters())
                    
                    # Apply gradient (simplified MAML)
                    adapted = []
                    for p, g in zip(self.meta_model.parameters(), grad):
                        adapted.append(p - self.inner_lr * g)
                    adapted_weights.append(adapted)
                
                # Outer loop: compute meta-loss on validation
                meta_loss_task = 0.0
                
                for arch, perf in zip(architectures[-len(architectures)//4:], 
                                       performances[-len(performances)//4:]):
                    encoding = self.search_space.encode(arch)
                    encoding = torch.tensor(encoding, dtype=torch.float32)
                    
                    if len(encoding) < 128:
                        encoding = F.pad(encoding, (0, 128 - len(encoding)))
                    elif len(encoding) > 128:
                        encoding = encoding[:128]
                    
                    # Use first adapted weights (simplified)
                    pred = self.meta_model(encoding)
                    meta_loss_task += F.mse_loss(pred.squeeze(), torch.tensor(perf))
                
                meta_loss += meta_loss_task / len(architectures)
            
            # Update meta-learner
            self.optimizer.zero_grad()
            meta_loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Meta-epoch {epoch + 1}: Meta-loss = {meta_loss.item():.4f}")
    
    def predict(self, architecture: Architecture) -> float:
        """
        Predict performance for an architecture.
        
        Args:
            architecture: Architecture to predict
            
        Returns:
            Predicted performance
        """
        encoding = self.search_space.encode(architecture)
        encoding = torch.tensor(encoding, dtype=torch.float32)
        
        if len(encoding) < 128:
            encoding = F.pad(encoding, (0, 128 - len(encoding)))
        elif len(encoding) > 128:
            encoding = encoding[:128]
        
        with torch.no_grad():
            pred = self.meta_model(encoding)
        
        return pred.item()
