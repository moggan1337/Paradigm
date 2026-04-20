"""
Controller base class for architecture generation.

This module provides the abstract base class for all controllers
used in neural architecture search (RL controllers, evolution, etc.).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from .search_space import SearchSpace, Architecture, Operation, OperationType


@dataclass
class ControllerConfig:
    """Configuration for controller."""
    
    hidden_size: int = 100
    num_layers: int = 2
    learning_rate: float = 0.00035
    entropy_weight: float = 0.01
    baseline_weight: float = 0.99
    max_grad_norm: float = 5.0
    batch_size: int = 32
    temperature: float = 1.0
    use_bias: bool = True
    tanh_c: float = 2.5
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Controller(ABC):
    """
    Abstract base class for architecture controllers.
    
    Controllers are responsible for generating candidate architectures
    based on the search space. Different controller types implement
    different generation strategies:
    - RL Controller: LSTM-based sequential generation
    - Evolutionary Controller: Mutation and crossover
    - Random Controller: Uniform random sampling
    
    Subclasses must implement:
    - sample(): Generate architectures
    - update(): Update controller parameters based on feedback
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[ControllerConfig] = None,
        **kwargs
    ):
        """
        Initialize controller.
        
        Args:
            search_space: Search space to generate architectures from
            config: Controller configuration
            **kwargs: Additional configuration passed to config
        """
        self.search_space = search_space
        
        # Merge config with kwargs
        if config is None:
            config = ControllerConfig(**kwargs)
        self.config = config
        
        # Device
        self.device = torch.device(config.device)
        
        # State tracking
        self.training_history: List[Dict[str, Any]] = []
        self.best_architecture: Optional[Architecture] = None
        self.best_reward: float = float('-inf')
        
        # Initialize
        self._init_controller()
    
    @abstractmethod
    def _init_controller(self):
        """Initialize controller-specific components."""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int = 1) -> List[Tuple[Architecture, Dict[str, Any]]]:
        """
        Sample architectures from the controller.
        
        Args:
            batch_size: Number of architectures to generate
            
        Returns:
            List of (Architecture, metadata) tuples
        """
        pass
    
    @abstractmethod
    def update(
        self,
        architectures: List[Architecture],
        rewards: List[float],
        metrics: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, float]:
        """
        Update controller based on rewards.
        
        Args:
            architectures: Architectures that were evaluated
            rewards: Reward values for each architecture
            metrics: Optional additional metrics
            
        Returns:
            Dictionary of training statistics
        """
        pass
    
    def save(self, path: Union[str, Path]) -> str:
        """
        Save controller state to disk.
        
        Args:
            path: Path to save checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': {
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers,
                'learning_rate': self.config.learning_rate,
                'entropy_weight': self.config.entropy_weight,
                'baseline_weight': self.config.baseline_weight,
            },
            'search_space_class': self.search_space.__class__.__name__,
            'training_history': self.training_history,
            'best_architecture': self.best_architecture.to_dict() if self.best_architecture else None,
            'best_reward': self.best_reward,
        }
        
        # Add controller-specific state
        controller_state = self._get_state()
        if controller_state is not None:
            checkpoint['controller_state'] = controller_state
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        return str(path)
    
    def load(self, path: Union[str, Path]):
        """
        Load controller state from disk.
        
        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        # Restore configuration
        for key, value in checkpoint['config'].items():
            setattr(self.config, key, value)
        
        # Restore training history
        self.training_history = checkpoint['training_history']
        self.best_reward = checkpoint['best_reward']
        
        # Restore best architecture
        if checkpoint.get('best_architecture'):
            self.best_architecture = Architecture.from_dict(
                checkpoint['best_architecture']
            )
        
        # Restore controller state
        if 'controller_state' in checkpoint:
            self._load_state(checkpoint['controller_state'])
    
    @abstractmethod
    def _get_state(self) -> Optional[Dict[str, Any]]:
        """Get controller-specific state for saving."""
        pass
    
    @abstractmethod
    def _load_state(self, state: Dict[str, Any]):
        """Load controller state from dictionary."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.training_history:
            return {
                'num_updates': 0,
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'best_reward': 0.0,
            }
        
        rewards = [h['reward'] for h in self.training_history]
        
        return {
            'num_updates': len(self.training_history),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'best_reward': max(rewards) if rewards else 0.0,
            'latest_reward': rewards[-1] if rewards else 0.0,
            'entropy': np.mean([h.get('entropy', 0) for h in self.training_history]),
        }
    
    def reset(self):
        """Reset controller to initial state."""
        self.training_history = []
        self.best_architecture = None
        self.best_reward = float('-inf')
        self._reset_state()
    
    @abstractmethod
    def _reset_state(self):
        """Reset controller-specific state."""
        pass


class RandomController(Controller):
    """
    Random sampling controller.
    
    This controller samples architectures uniformly at random
    from the search space. Useful as a baseline.
    """
    
    def _init_controller(self):
        """Initialize random controller."""
        pass
    
    def sample(self, batch_size: int = 1) -> List[Tuple[Architecture, Dict[str, Any]]]:
        """Sample random architectures."""
        results = []
        for _ in range(batch_size):
            arch = self.search_space.sample()
            results.append((arch, {'sampled': True}))
        return results
    
    def update(
        self,
        architectures: List[Architecture],
        rewards: List[float],
        metrics: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, float]:
        """No update for random controller."""
        return {'updated': False}
    
    def _get_state(self) -> Optional[Dict[str, Any]]:
        return None
    
    def _load_state(self, state: Dict[str, Any]):
        pass
    
    def _reset_state(self):
        pass


class RLController(Controller):
    """
    Reinforcement Learning controller using LSTM.
    
    This controller uses a multi-layer LSTM to generate
    architecture descriptions sequentially, trained with
    policy gradient (REINFORCE) algorithm.
    
    Architecture:
    ┌─────────────────────────────────────────────────┐
    │                  LSTM Controller                 │
    ├─────────────────────────────────────────────────┤
    │  Input: [prev_action_embedding, prev_hidden]   │
    │    ↓                                             │
    │  ┌─────────┐                                    │
    │  │ LSTM 1  │ → h1, c1                           │
    │  └────┬────┘                                    │
    │       ↓                                          │
    │  ┌─────────┐                                    │
    │  │ LSTM 2  │ → h2, c2                           │
    │  └────┬────┘                                    │
    │       ↓                                          │
    │  ┌─────────┐                                    │
    │  │Linear   │ → logits                           │
    │  │(softmax)│                                    │
    │  └─────────┘                                    │
    │    ↓                                             │
    │  Sample action from distribution               │
    └─────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[ControllerConfig] = None,
        num_choices: int = 10,
        num_input_choices: int = 3,
        **kwargs
    ):
        """
        Initialize RL controller.
        
        Args:
            search_space: Search space
            config: Controller configuration
            num_choices: Number of operation choices
            num_input_choices: Number of input choices per node
            **kwargs: Additional configuration
        """
        self.num_choices = num_choices
        self.num_input_choices = num_input_choices
        super().__init__(search_space, config, **kwargs)
    
    def _init_controller(self):
        """Initialize LSTM controller network."""
        self.num_steps = self.search_space.max_nodes * (self.num_input_choices + 1)
        
        # Embedding for actions
        self.action_embedding = nn.Embedding(
            self.num_choices,
            self.config.hidden_size
        )
        
        # LSTM controller
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.entropy_weight if self.config.num_layers > 1 else 0
        )
        
        # Output linear layer
        self.fc = nn.Linear(
            self.config.hidden_size,
            self.num_choices
        )
        
        # Move to device
        self.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate
        )
        
        # Baseline for REINFORCE
        self.baseline = 0.0
        
        # Hidden state
        self._reset_lstm_state()
    
    def sample(self, batch_size: int = 1) -> List[Tuple[Architecture, Dict[str, Any]]]:
        """
        Sample architectures using the LSTM controller.
        
        Args:
            batch_size: Number of architectures to sample
            
        Returns:
            List of (Architecture, metadata) tuples
        """
        self.train()
        self._reset_lstm_state()
        
        results = []
        
        for _ in range(batch_size):
            self._reset_lstm_state()
            
            # Storage for actions and logits
            actions = []
            log_probs = []
            entropies = []
            
            # Generate architecture step by step
            hidden = self._get_hidden_state()
            
            for step in range(self.num_steps):
                # Get previous action embedding
                if step == 0:
                    prev_embedding = torch.zeros(
                        batch_size, self.config.hidden_size,
                        device=self.device
                    )
                else:
                    prev_action = actions[-1]
                    prev_embedding = self.action_embedding(
                        torch.tensor([prev_action], device=self.device)
                    )
                
                # LSTM step
                lstm_input = prev_embedding.unsqueeze(1)
                lstm_out, hidden = self.lstm(lstm_input, hidden)
                
                # Get logits
                logits = self.fc(lstm_out.squeeze(1))
                
                # Apply temperature and tanh clipping
                logits = logits / self.config.temperature
                logits = self.config.tanh_c * torch.tanh(logits)
                
                # Sample action
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
                # Store for loss computation
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                
                actions.append(action.item())
                log_probs.append(log_prob)
                entropies.append(entropy)
            
            # Build architecture from actions
            architecture, metadata = self._actions_to_architecture(actions)
            
            # Add metadata
            metadata['log_probs'] = log_probs
            metadata['entropies'] = entropies
            metadata['actions'] = actions
            
            results.append((architecture, metadata))
        
        return results
    
    def _actions_to_architecture(
        self,
        actions: List[int]
    ) -> Tuple[Architecture, Dict[str, Any]]:
        """
        Convert action sequence to Architecture object.
        
        Args:
            actions: List of action indices
            
        Returns:
            (Architecture, metadata) tuple
        """
        architecture = Architecture(name=f"arch_{len(self.training_history)}")
        
        # Build DAG from actions
        # Each node has (num_input_choices + 1) actions:
        # - First num_input_choices: select input nodes
        # - Last 1: select operation
        
        nodes_per_step = self.num_input_choices + 1
        
        for node_idx in range(self.search_space.max_nodes):
            step_base = node_idx * nodes_per_step
            
            # Get operation for this node
            if step_base + self.num_input_choices < len(actions):
                op_idx = actions[step_base + self.num_input_choices]
                op_type = list(OperationType)[op_idx % len(OperationType)]
                
                operation = Operation(
                    op_type=op_type,
                    output_index=node_idx + 1
                )
                
                # Add edges from previous nodes
                cell = Cell(node_index=node_idx + 1)
                architecture.cells.append(cell)
                
                for input_idx in range(self.num_input_choices):
                    input_node = actions[step_base + input_idx] % (node_idx + 1)
                    
                    edge = Edge(
                        source=input_node,
                        target=node_idx + 1,
                        operation=Operation(
                            op_type=op_type,
                            parameters={'input_idx': input_idx}
                        )
                    )
                    architecture.edges.append(edge)
        
        # Validate and bound
        architecture = self.search_space.bound(architecture)
        
        metadata = {
            'action_sequence': actions,
        }
        
        return architecture, metadata
    
    def update(
        self,
        architectures: List[Architecture],
        rewards: List[float],
        metrics: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, float]:
        """
        Update controller using REINFORCE algorithm.
        
        Args:
            architectures: Architectures that were evaluated
            rewards: Reward values for each architecture
            
        Returns:
            Dictionary of training statistics
        """
        if not architectures or not rewards:
            return {'updated': False}
        
        self.train()
        
        # Compute discounted rewards (simple baseline subtraction)
        baseline = self.config.baseline_weight * self.baseline + \
                   (1 - self.config.baseline_weight) * np.mean(rewards)
        
        advantages = [r - baseline for r in rewards]
        self.baseline = baseline
        
        # Compute policy gradient loss
        loss = 0.0
        total_entropy = 0.0
        total_log_prob = 0.0
        
        for arch, reward in zip(architectures, rewards):
            # Find corresponding metadata
            for result in self.training_history[-len(architectures):]:
                if result.get('architecture') == arch:
                    metadata = result
                    break
            else:
                continue
            
            log_probs = metadata.get('log_probs', [])
            entropies = metadata.get('entropies', [])
            
            if not log_probs:
                continue
            
            # REINFORCE loss: -log_prob * advantage
            adv = reward - baseline
            
            for log_prob, entropy in zip(log_probs, entropies):
                loss -= log_prob * adv
                loss -= self.config.entropy_weight * entropy
                total_entropy += entropy.item()
                total_log_prob += log_prob.item()
        
        # Normalize by batch size
        loss = loss / len(architectures)
        
        # Gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        # Update best
        for arch, reward in zip(architectures, rewards):
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_architecture = arch
        
        # Record history
        self.training_history.append({
            'reward': np.mean(rewards),
            'best_reward': self.best_reward,
            'entropy': total_entropy / sum(len(r.get('log_probs', [])) 
                                           for r in [architectures[0]]),
            'loss': loss.item(),
            'baseline': baseline,
        })
        
        return {
            'loss': loss.item(),
            'entropy': total_entropy,
            'baseline': baseline,
            'mean_reward': np.mean(rewards),
        }
    
    def _get_hidden_state(self) -> Tuple[Tensor, Tensor]:
        """Get current LSTM hidden state."""
        batch_size = 1
        h = torch.zeros(
            self.config.num_layers, batch_size, self.config.hidden_size,
            device=self.device
        )
        c = torch.zeros(
            self.config.num_layers, batch_size, self.config.hidden_size,
            device=self.device
        )
        return (h, c)
    
    def _reset_lstm_state(self):
        """Reset LSTM hidden state."""
        self._hidden_state = self._get_hidden_state()
    
    def _get_state(self) -> Optional[Dict[str, Any]]:
        """Get controller state for saving."""
        return {
            'lstm_state': self._hidden_state,
            'baseline': self.baseline,
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
    
    def _load_state(self, state: Dict[str, Any]):
        """Load controller state."""
        self.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.baseline = state['baseline']
        self._hidden_state = state['lstm_state']
    
    def _reset_state(self):
        """Reset controller-specific state."""
        self.baseline = 0.0
        self._reset_lstm_state()
    
    def parameters(self):
        """Get trainable parameters."""
        return list(self.action_embedding.parameters()) + \
               list(self.lstm.parameters()) + \
               list(self.fc.parameters())


class EvolutionController(Controller):
    """
    Evolutionary controller for architecture search.
    
    This controller maintains a population of architectures
    and evolves them through mutation and crossover.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[ControllerConfig] = None,
        population_size: int = 50,
        tournament_size: int = 3,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.3,
        **kwargs
    ):
        """
        Initialize evolution controller.
        
        Args:
            search_space: Search space
            config: Controller configuration
            population_size: Size of population
            tournament_size: Tournament size for selection
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        super().__init__(search_space, config, **kwargs)
    
    def _init_controller(self):
        """Initialize evolution controller."""
        self.population: List[Tuple[Architecture, float]] = []
        self.generation = 0
        
        # Initialize with random population
        for _ in range(self.population_size):
            arch = self.search_space.sample()
            self.population.append((arch, 0.0))
    
    def sample(self, batch_size: int = 1) -> List[Tuple[Architecture, Dict[str, Any]]]:
        """
        Sample architectures using evolutionary operators.
        
        Args:
            batch_size: Number of architectures to sample
            
        Returns:
            List of (Architecture, metadata) tuples
        """
        results = []
        
        for _ in range(batch_size):
            # Tournament selection
            parent1 = self._tournament_select()
            
            # Crossover?
            if random.random() < self.crossover_rate and len(self.population) > 1:
                parent2 = self._tournament_select()
                while parent2 == parent1:
                    parent2 = self._tournament_select()
                
                child = self.search_space.crossover(parent1, parent2)
                metadata = {'operator': 'crossover', 'parents': (id(parent1), id(parent2))}
            else:
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self.search_space.mutate(parent1)
                    metadata = {'operator': 'mutation', 'parent': id(parent1)}
                else:
                    child = self.search_space.sample()
                    metadata = {'operator': 'random'}
            
            # Validate
            if not self.search_space.is_valid(child):
                child = self.search_space.sample()
                metadata = {'operator': 'random_fallback'}
            
            results.append((child, metadata))
        
        return results
    
    def _tournament_select(self) -> Architecture:
        """Tournament selection."""
        indices = random.sample(range(len(self.population)), 
                                min(self.tournament_size, len(self.population)))
        
        best_idx = max(indices, key=lambda i: self.population[i][1])
        return self.population[best_idx][0]
    
    def update(
        self,
        architectures: List[Architecture],
        rewards: List[float],
        metrics: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, float]:
        """
        Update population based on rewards.
        
        Args:
            architectures: Architectures that were evaluated
            rewards: Reward values for each architecture
            
        Returns:
            Dictionary of training statistics
        """
        # Add evaluated architectures to population
        for arch, reward in zip(architectures, rewards):
            self.population.append((arch, reward))
        
        # Sort by fitness
        self.population.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top individuals
        self.population = self.population[:self.population_size]
        
        # Update best
        if self.population:
            best_arch, best_reward = self.population[0]
            if best_reward > self.best_reward:
                self.best_reward = best_reward
                self.best_architecture = best_arch
        
        self.generation += 1
        
        # Record history
        if self.population:
            rewards = [f for _, f in self.population]
            self.training_history.append({
                'generation': self.generation,
                'best_reward': self.best_reward,
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards) if len(rewards) > 1 else 0,
            })
        
        return {
            'generation': self.generation,
            'best_reward': self.best_reward,
            'population_size': len(self.population),
        }
    
    def _get_state(self) -> Optional[Dict[str, Any]]:
        """Get controller state for saving."""
        return {
            'population': [(arch.to_dict(), fitness) for arch, fitness in self.population],
            'generation': self.generation,
            'best_reward': self.best_reward,
            'best_architecture': self.best_architecture.to_dict() if self.best_architecture else None,
        }
    
    def _load_state(self, state: Dict[str, Any]):
        """Load controller state."""
        self.population = [
            (Architecture.from_dict(arch_dict), fitness)
            for arch_dict, fitness in state['population']
        ]
        self.generation = state['generation']
        self.best_reward = state['best_reward']
        if state.get('best_architecture'):
            self.best_architecture = Architecture.from_dict(state['best_architecture'])
    
    def _reset_state(self):
        """Reset controller-specific state."""
        self.population = []
        self.generation = 0
