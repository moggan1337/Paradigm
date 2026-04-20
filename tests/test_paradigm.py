"""
Tests for Paradigm AutoML framework.
"""

import pytest
import torch
import numpy as np

from paradigm.core.search_space import (
    SearchSpace, Architecture, Cell, Edge, Operation, OperationType
)
from paradigm.core.controller import Controller, RLController, RandomController
from paradigm.core.optimizer import Optimizer, StandardOptimizer
from paradigm.search_spaces import CNNSearchSpace, RNNSearchSpace
from paradigm.optimizers import DARTSOptimizer, MultiObjectiveOptimizer
from paradigm.metrics import PerformancePredictor, SynFlowMetric
from paradigm.transfer import WeightInheritance


class TestSearchSpace:
    """Tests for search space classes."""
    
    def test_cnn_search_space_sample(self):
        """Test CNN search space sampling."""
        search_space = CNNSearchSpace(max_nodes=5)
        arch = search_space.sample()
        
        assert isinstance(arch, Architecture)
        assert len(arch.cells) > 0
        assert len(arch.edges) > 0
    
    def test_architecture_encoding(self):
        """Test architecture encoding."""
        search_space = CNNSearchSpace(max_nodes=5)
        arch = search_space.sample()
        
        encoding = search_space.encode(arch)
        
        assert isinstance(encoding, np.ndarray)
        assert len(encoding) > 0
    
    def test_architecture_mutation(self):
        """Test architecture mutation."""
        search_space = CNNSearchSpace(max_nodes=5)
        arch1 = search_space.sample()
        arch2 = search_space.mutate(arch1, mutation_rate=0.5)
        
        assert isinstance(arch2, Architecture)
        assert arch1 != arch2  # Should be different after mutation
    
    def test_architecture_crossover(self):
        """Test architecture crossover."""
        search_space = CNNSearchSpace(max_nodes=5)
        arch1 = search_space.sample()
        arch2 = search_space.sample()
        
        child = search_space.crossover(arch1, arch2)
        
        assert isinstance(child, Architecture)
        assert len(child.cells) > 0


class TestArchitecture:
    """Tests for Architecture class."""
    
    def test_architecture_creation(self):
        """Test architecture creation."""
        arch = Architecture(name="test")
        
        assert arch.name == "test"
        assert len(arch.cells) == 0
        assert len(arch.edges) == 0
    
    def test_architecture_to_dict(self):
        """Test architecture serialization."""
        arch = Architecture(name="test")
        arch.cells.append(Cell(node_index=0))
        arch.edges.append(Edge(
            source=0,
            target=1,
            operation=Operation(op_type=OperationType.SKIP_CONNECT)
        ))
        
        d = arch.to_dict()
        
        assert d['name'] == "test"
        assert len(d['cells']) == 1
        assert len(d['edges']) == 1
    
    def test_architecture_from_dict(self):
        """Test architecture deserialization."""
        arch = Architecture(name="test")
        arch.cells.append(Cell(node_index=0))
        
        d = arch.to_dict()
        arch2 = Architecture.from_dict(d)
        
        assert arch2.name == arch.name


class TestController:
    """Tests for controller classes."""
    
    def test_random_controller(self):
        """Test random controller."""
        search_space = CNNSearchSpace(max_nodes=5)
        controller = RandomController(search_space)
        
        architectures, metadata = controller.sample(batch_size=3)
        
        assert len(architectures) == 3
        for arch in architectures:
            assert isinstance(arch, Architecture)
    
    def test_rl_controller_sample(self):
        """Test RL controller sampling."""
        search_space = CNNSearchSpace(max_nodes=5)
        controller = RLController(search_space, hidden_size=32, num_layers=1)
        
        architectures, metadata = controller.sample(batch_size=2)
        
        assert len(architectures) == 2
    
    def test_controller_update(self):
        """Test controller update."""
        search_space = CNNSearchSpace(max_nodes=5)
        controller = RandomController(search_space)
        
        architectures, _ = controller.sample(batch_size=2)
        rewards = [0.8, 0.6]
        
        stats = controller.update(architectures, rewards)
        
        assert 'updated' in stats


class TestOptimizer:
    """Tests for optimizer classes."""
    
    def test_standard_optimizer_init(self):
        """Test standard optimizer initialization."""
        search_space = CNNSearchSpace(max_nodes=5)
        optimizer = StandardOptimizer(
            search_space,
            epochs_per_trial=1,
            controller=RandomController(search_space)
        )
        
        assert optimizer is not None
        assert optimizer.epochs_per_trial == 1


class TestMetrics:
    """Tests for metrics classes."""
    
    def test_synflow_metric(self):
        """Test SynFlow metric."""
        search_space = CNNSearchSpace(max_nodes=5)
        arch = search_space.sample()
        
        metric = SynFlowMetric()
        score = metric.calculate(arch, input_shape=(1, 3, 32, 32))
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_performance_predictor(self):
        """Test performance predictor."""
        search_space = CNNSearchSpace(max_nodes=5)
        architectures = [search_space.sample() for _ in range(3)]
        
        predictor = PerformancePredictor()
        scores = predictor.batch_score(architectures, input_shape=(1, 3, 32, 32))
        
        assert len(scores) == 3


class TestTransfer:
    """Tests for transfer learning classes."""
    
    def test_weight_inheritance_init(self):
        """Test weight inheritance initialization."""
        search_space = CNNSearchSpace(max_nodes=5)
        transfer = WeightInheritance(search_space)
        
        assert transfer is not None
    
    def test_add_source_data(self):
        """Test adding source data."""
        search_space = CNNSearchSpace(max_nodes=5)
        transfer = WeightInheritance(search_space)
        
        arch = search_space.sample()
        transfer.add_source_data(arch, performance=0.95)
        
        assert len(transfer.source_architectures) == 1
        assert transfer.source_performances[arch.name] == 0.95


class TestIntegration:
    """Integration tests."""
    
    def test_full_search_pipeline(self):
        """Test full search pipeline."""
        # Create search space
        search_space = CNNSearchSpace(max_nodes=5)
        
        # Create controller
        controller = RandomController(search_space)
        
        # Create optimizer
        optimizer = StandardOptimizer(
            search_space,
            controller=controller,
            epochs_per_trial=1
        )
        
        # Run search (short version)
        result = optimizer.search(num_trials=3, epochs_per_trial=1)
        
        assert result is not None
        assert len(result.all_architectures) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
