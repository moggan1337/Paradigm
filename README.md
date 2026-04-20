# Paradigm: AutoML with Neural Architecture Search

<p align="center">
  <img src="docs/paradigm_logo.png" alt="Paradigm Logo" width="200"/>
</p>

<p align="center">
  <a href="https://github.com/moggan1337/Paradigm/actions">
    <img src="https://github.com/moggan1337/Paradigm/workflows/CI/badge.svg" alt="CI Status"/>
  </a>
  <a href="https://pypi.org/project/paradigm-nas/">
    <img src="https://img.shields.io/pypi/v/paradigm-nas.svg" alt="PyPI Version"/>
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/>
  </a>
  <a href="https://doi.org/10.5281/zenodo.xxxxxx">
    <img src="https://img.shields.io/badge/DOI-10.5281/zenodo.xxxxxx-green.svg" alt="DOI"/>
  </a>
</p>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Installation](#-installation)
4. [Quick Start](#-quick-start)
5. [Architecture Search Algorithms](#-architecture-search-algorithms)
   - [Reinforcement Learning Controller](#reinforcement-learning-controller)
   - [Evolutionary Algorithms](#evolutionary-algorithms)
   - [DARTS: Gradient-Based NAS](#darts-gradient-based-nas)
   - [Bayesian Optimization](#bayesian-optimization)
6. [Search Spaces](#-search-spaces)
   - [Convolutional Neural Networks](#convolutional-neural-networks)
   - [Recurrent Neural Networks](#recurrent-neural-networks)
   - [Transformers](#transformers)
7. [Multi-Objective Optimization](#-multi-objective-optimization)
8. [Performance Prediction](#-performance-prediction)
9. [Transfer Learning](#-transfer-learning)
10. [Early Pruning Strategies](#-early-pruning-strategies)
11. [Benchmarks](#-benchmarks)
12. [API Reference](#-api-reference)
13. [Examples](#-examples)
14. [Contributing](#-contributing)
15. [Citation](#-citation)
16. [License](#-license)

---

## 🎯 Overview

Paradigm is a comprehensive AutoML framework implementing state-of-the-art Neural Architecture Search (NAS) algorithms. It provides researchers and practitioners with a unified interface to explore, evaluate, and deploy neural network architectures automatically.

### Key Design Principles

- **Modularity**: Each component (search space, controller, optimizer) is designed to be independent and composable
- **Extensibility**: Easy to add new search spaces, controllers, and optimization strategies
- **Efficiency**: Gradient-based methods, early pruning, and performance prediction for fast search
- **Reproducibility**: Complete experiment tracking and configuration management
- **Scalability**: Support for distributed search across multiple GPUs and machines

---

## ✨ Features

### Core Features

| Feature | Description |
|---------|-------------|
| **RL Controller** | LSTM-based controller for sequential architecture generation |
| **Evolutionary Search** | Population-based optimization with mutation and crossover |
| **DARTS** | Differentiable architecture search with shared weights |
| **Bayesian Optimization** | Hyperparameter optimization with Gaussian Processes |
| **Multi-Objective** | Pareto-optimal architectures across multiple objectives |
| **Performance Prediction** | Zero-cost metrics for architecture evaluation |
| **Transfer Learning** | Knowledge transfer across related tasks |
| **Early Pruning** | Efficient search space reduction |

### Search Spaces

| Search Space | Applications |
|--------------|--------------|
| **CNN** | Image classification, object detection, segmentation |
| **RNN** | Sequence modeling, time series, NLP |
| **Transformer** | Language modeling, translation, attention-based tasks |

### Supported Benchmarks

| Benchmark | Dataset | Metrics |
|-----------|---------|---------|
| NAS-Bench-101 | CIFAR-10 | Test accuracy |
| NAS-Bench-201 | CIFAR-10/100, ImageNet-16-120 | Test accuracy |
| NBI101 | NAS-Bench-101 extensions | Multiple metrics |
| DARTS Search Space | CIFAR-10 | Validation accuracy |

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU acceleration)

### From Source

```bash
git clone https://github.com/moggan1337/Paradigm.git
cd Paradigm
pip install -e .
```

### With Optional Dependencies

```bash
# For all optional dependencies
pip install -e ".[all]"

# For visualization
pip install -e ".[viz]"

# For distributed training
pip install -e ".[distributed]"
```

### Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
tqdm>=4.62.0
pyyaml>=5.4.0
joblib>=1.1.0
```

---

## 🚀 Quick Start

### Basic NAS Search

```python
import paradigm
from paradigm.search_spaces import CNNSearchSpace
from paradigm.controllers import RLController
from paradigm.optimizers import EvolutionaryOptimizer

# Define search space
search_space = CNNSearchSpace(
    max_layers=12,
    num_operations=5,
    channels=[16, 32, 64]
)

# Initialize controller
controller = RLController(
    search_space=search_space,
    hidden_size=64,
    num_layers=2
)

# Run search
results = paradigm.search(
    search_space=search_space,
    controller=controller,
    num_trials=100,
    max_epochs_per_trial=50
)

# Get best architecture
best_arch = results.best_architecture()
print(f"Best architecture: {best_arch}")
```

### DARTS Search

```python
from paradigm.optimizers import DARTSOptimizer

# DARTS requires shared weights
optimizer = DARTSOptimizer(
    search_space=search_space,
    unrolled_steps=1,
    log_alpha_every=10
)

# Train to find optimal architecture
best_arch = optimizer.search(
    dataset='cifar10',
    epochs=50,
    batch_size=64
)
```

### Multi-Objective Search

```python
from paradigm.optimizers import MultiObjectiveOptimizer
from paradigm.metrics import LatencyMetric, ParameterCountMetric

optimizer = MultiObjectiveOptimizer(
    search_space=search_space,
    objectives=[
        ('accuracy', 1.0),
        ('latency', 0.5, LatencyMetric()),
        ('params', 0.3, ParameterCountMetric())
    ],
    algorithm='nsga2'
)

pareto_front = optimizer.search(num_generations=100)
```

---

## 🧠 Architecture Search Algorithms

### Reinforcement Learning Controller

The RL-based controller uses a recurrent neural network (LSTM) to generate architecture descriptions sequentially. The controller is trained using policy gradient methods (REINFORCE) to maximize expected validation accuracy.

#### Architecture of the Controller

```
┌─────────────────────────────────────────────────────────┐
│                    LSTM Controller                       │
├─────────────────────────────────────────────────────────┤
│  Input: Previous action embeddings                       │
│    ↓                                                    │
│  ┌─────────┐                                           │
│  │ LSTM 1  │ → hidden state                            │
│  └─────────┘                                           │
│    ↓                                                    │
│  ┌─────────┐                                           │
│  │ LSTM 2  │ → hidden state                            │
│  └─────────┘                                           │
│    ↓                                                    │
│  ┌─────────┐                                           │
│  │Linear   │ → action probabilities                    │
│  │Output   │                                           │
│  └─────────┘                                           │
│    ↓                                                    │
│  Output: Action (operation type, connections, etc.)     │
└─────────────────────────────────────────────────────────┘
```

#### Controller Algorithm

1. **Initialization**: Start with empty architecture state
2. **Action Sampling**: Sample action from policy distribution
3. **Architecture Building**: Update architecture based on action
4. **Evaluation**: Train child network and measure accuracy
5. **Reward Calculation**: Use validation accuracy as reward
6. **Policy Update**: Apply REINFORCE gradient:
   
   $$\nabla_\theta J = \mathbb{E}[R \cdot \nabla_\theta \log \pi_\theta(a|s)]$$

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 100 | LSTM hidden dimension |
| `num_layers` | 2 | Number of LSTM layers |
| `lr` | 0.00035 | Learning rate |
| `entropy_weight` | 0.01 | Entropy regularization |
| `baseline_weight` | 0.99 | Exponential moving average decay |

#### Advantages

- Can handle complex, hierarchical architectures
- Learns to explore promising regions of search space
- No differentiable operations required

#### Limitations

- Requires many evaluations (10,000+ child networks)
- High variance in gradient estimates
- Slow convergence compared to gradient-based methods

### Evolutionary Algorithms

Evolutionary NAS treats architecture search as an optimization problem in the space of neural network architectures. We implement a population-based approach with mutation, crossover, and selection operators.

#### Algorithm Overview

```
┌─────────────────────────────────────────────────────────┐
│               Evolutionary Search Loop                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐                                       │
│  │   Population │  (N architectures)                    │
│  │   [A1,..,AN]  │                                       │
│  └──────┬───────┘                                       │
│         │                                               │
│         ↓                                               │
│  ┌──────────────┐                                       │
│  │  Evaluation  │  (Train & measure fitness)            │
│  │  Fitness Fn  │                                       │
│  └──────┬───────┘                                       │
│         │                                               │
│         ↓                                               │
│  ┌──────────────┐                                       │
│  │  Selection   │  (Tournament selection)              │
│  │  [Parent 1,2]│                                       │
│  └──────┬───────┘                                       │
│         │                                               │
│         ↓                                               │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │   Mutation   │  │   Crossover  │                     │
│  │  (P_mut=0.8)  │  │  (P_cross=0.3)│                    │
│  └──────┬───────┘  └──────┬───────┘                     │
│         │                 │                             │
│         └────────┬────────┘                             │
│                  ↓                                       │
│         ┌──────────────┐                                 │
│         │  Offspring   │                                 │
│         │  [O1, O2...] │                                 │
│         └──────┬───────┘                                 │
│                │                                         │
│                ↓                                         │
│         ┌──────────────┐                                 │
│         │   Survivor   │  (Replace weakest)              │
│         │  Selection   │                                 │
│         └──────────────┘                                 │
│                │                                         │
│         ┌──────┴───────┐                                 │
│         │ Next Gen?    │───Yes──→ Continue              │
│         └──────────────┘                                 │
│                  │                                       │
│                 No                                       │
│                  ↓                                       │
│         ┌──────────────┐                                 │
│         │    Return    │                                 │
│         │  Best Arch   │                                 │
│         └──────────────┘                                 │
└─────────────────────────────────────────────────────────┘
```

#### Mutation Operators

1. **Add Layer**: Add a new operation to a connection
2. **Remove Layer**: Remove an existing operation
3. **Change Operation**: Replace one operation with another
4. **Skip Connection**: Add/remove skip connections
5. **Channel Width**: Modify channel dimensions

#### Crossover Operators

1. **One-Point Crossover**: Exchange architecture subgraphs
2. **Uniform Crossover**: Random gene exchange
3. **Topology Crossover**: Exchange connection patterns

#### Tournament Selection

```python
def tournament_selection(population, fitness, k=3):
    """Select individual using tournament selection."""
    selected = random.sample(range(len(population)), k)
    best = max(selected, key=lambda i: fitness[i])
    return population[best]
```

#### NSGA-II for Multi-Objective

For multi-objective optimization, we use NSGA-II (Non-dominated Sorting Genetic Algorithm II):

1. **Non-dominated Sorting**: Rank individuals by Pareto dominance
2. **Crowding Distance**: Maintain diversity in population
3. **Elitism**: Keep best individuals across generations

```python
# NSGA-II selection criteria
fitness = dominate_rank * α + crowding_distance * β
```

### DARTS: Gradient-Based NAS

DARTS (Differentiable Architecture Search) relaxes the discrete search space into a continuous one, enabling gradient-based optimization of architecture parameters.

#### Key Innovation

Instead of selecting discrete operations, DARTS maintains a mixture of all possible operations with learnable weights:

$$\bar{o}^{(h,w)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(h,w)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(h,w)})} \cdot o(x)$$

#### Search Space Relaxation

```
┌─────────────────────────────────────────────────────────┐
│                   DARTS Search Cell                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│    Input 1 ──┬──[sep_conv_3x3]──┬──[sep_conv_5x5]──┐    │
│              │                  │                  │    │
│              └──[max_pool_3x3]───┼──[avg_pool_3x3]──┤    │
│                                 │                  │    │
│    Input 2 ──[skip_connect]─────┴──[none]──────────┤    │
│                                 │                  │    │
│              All operations active with weights α   │    │
│                                 ↓                  ↓    │
│                              Edge Weights α    Output   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### DARTS Algorithm

1. **Shared Weights**: Single network with all candidate operations
2. **Architecture Parameters**: α for each edge-operation pair
3. **Bi-level Optimization**:
   
   - Inner: Minimize training loss with weights $w$
   - Outer: Minimize validation loss with α

4. **Discretization**: After search, select best operation per edge

#### Optimization Procedure

```python
# Bilevel optimization
for epoch in range(num_epochs):
    # Update weights (inner loop)
    for _ in range(unrolled_steps):
        train_loss = loss_fn(w, α, X_train)
        w = w - lr_w * grad(train_loss, w)
    
    # Update architecture (outer loop)
    val_loss = loss_fn(w, α, X_val)
    grad(α) = grad(val_loss, α)
    α = α - lr_α * grad(α)
```

#### DARTS Variants

| Variant | Description | Memory | Time |
|---------|-------------|--------|------|
| DARTS-V1 | Original formulation | High | Slow |
| DARTS-V2 | Reduced unrolling | Medium | Medium |
| DARTS-1st | First-order approximation | Low | Fast |
| PDARTS | Progressive depth | Medium | Medium |
| PC-DARTS | Partial channels | Low | Fast |

### Bayesian Optimization

For hyperparameter optimization, we use Bayesian Optimization with Gaussian Processes or Tree Parzen Estimators.

#### Acquisition Functions

1. **Expected Improvement (EI)**
   
   $$EI(x) = \mathbb{E}[\max(0, f(x) - f(x^+))]$$

2. **Upper Confidence Bound (UCB)**
   
   $$UCB(x) = \mu(x) + \kappa \sigma(x)$$

3. **Probability of Improvement (PI)**
   
   $$PI(x) = P(f(x) > f(x^+)) = \Phi\left(\frac{\mu(x) - f(x^+)}{\sigma(x)}\right)$$

#### Optimization Loop

```
┌─────────────────────────────────────────────────────────┐
│               Bayesian Optimization Loop                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Initialize with random configurations               │
│     ↓                                                    │
│  2. Train models and collect observations                │
│     ↓                                                    │
│  3. Update surrogate model (GP/TPE)                      │
│     ↓                                                    │
│  4. Optimize acquisition function                        │
│     ↓                                                    │
│  5. Sample next configuration                            │
│     ↓                                                    │
│  6. Evaluate and repeat                                  │
│     ↓                                                    │
│  7. Return best configuration                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔍 Search Spaces

### Convolutional Neural Networks

The CNN search space includes operations commonly used in image classification networks.

#### Operation Set

| Operation | Parameters | Description |
|-----------|------------|-------------|
| `sep_conv_3x3` | stride=1 | Separable convolution 3x3 |
| `sep_conv_5x5` | stride=1 | Separable convolution 5x5 |
| `dil_conv_3x3` | dilation=2 | Dilated convolution 3x3 |
| `dil_conv_5x5` | dilation=2 | Dilated convolution 5x5 |
| `max_pool_3x3` | stride=1 | Max pooling 3x3 |
| `avg_pool_3x3` | stride=1 | Average pooling 3x3 |
| `skip_connect` | - | Identity connection |
| `none` | - | No operation |

#### Cell-Based Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Normal Cell                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   State 0 ─────┐                                         │
│                ├──[op]──┐                               │
│   State 1 ─────┼──[op]──┤──→ State k                    │
│                ├──[op]──┤                               │
│   State 2 ─────┴──[op]──┘                               │
│                                                          │
│   Each state: feature map at different resolution       │
│   Each edge: one operation from operation set           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Encoding Scheme

```python
# Architecture encoding example
{
    'cell_type': 'normal',
    'num_nodes': 4,
    'edges': [
        (0, 2, 'sep_conv_3x3'),
        (1, 3, 'max_pool_3x3'),
        (2, 3, 'skip_connect'),
        (0, 3, 'none'),
    ]
}
```

### Recurrent Neural Networks

The RNN search space explores recurrent cell architectures and connectivity patterns.

#### Recurrent Operations

| Operation | Description |
|-----------|-------------|
| `rnn_cell` | Basic RNN cell |
| `lstm_cell` | LSTM cell |
| `gru_cell` | GRU cell |
| `relu_rnn` | RNN with ReLU activation |
| `tanh_rnn` | RNN with tanh activation |

#### Search Space Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `hidden_size` | [64, 512] | Hidden dimension |
| `num_layers` | [1, 4] | Number of layers |
| `dropout` | [0.0, 0.5] | Dropout rate |
| `use_attention` | [True, False] | Attention mechanism |

#### Sequence-to-Sequence Architecture

```
┌─────────────────────────────────────────────────────────┐
│               RNN Encoder-Decoder                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input → [Embedding] → [RNN Cell × N] → [RNN Cell × M]  │
│                                  ↓                      │
│  Output ← [Linear] ← [RNN Cell × M] ←──┘              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Transformers

The Transformer search space explores attention mechanisms and feed-forward architectures.

#### Attention Variants

| Attention Type | Description |
|----------------|-------------|
| `scaled_dot_attention` | Standard scaled dot-product |
| `multihead_attention` | Multi-head attention |
| `linear_attention` | Linear/complexity attention |
| `performer_attention` | Random feature attention |

#### Search Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `num_heads` | [4, 16] | Number of attention heads |
| `head_dim` | [32, 128] | Dimension per head |
| `ff_dim` | [512, 2048] | Feed-forward dimension |
| `num_layers` | [2, 12] | Number of layers |

#### Transformer Block Structure

```
┌─────────────────────────────────────────────────────────┐
│                  Transformer Block                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input                                                   │
│    ↓                                                    │
│  ┌─────────────────┐                                    │
│  │  Multi-Head     │ ←── Add & Norm                     │
│  │  Attention      │                                    │
│  └────────┬────────┘                                    │
│           ↓                                             │
│  ┌─────────────────┐                                    │
│  │    Feed-Forward │ ←── Add & Norm                     │
│  │    Network      │                                    │
│  └────────┬────────┘                                    │
│           ↓                                             │
│  Output                                                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Multi-Objective Optimization

Multi-objective NAS optimizes architectures across multiple conflicting objectives simultaneously.

### Supported Objectives

| Objective | Metric | Goal |
|-----------|--------|------|
| Accuracy | Top-1/Top-5 | Maximize |
| Latency | ms per inference | Minimize |
| Parameters | Million params | Minimize |
| FLOPs | Billion ops | Minimize |
| Memory | MB | Minimize |
| Energy | Joules | Minimize |

### Pareto Optimality

An architecture is Pareto-optimal if no other architecture is better in all objectives:

$$\nexists x': f_i(x') \geq f_i(x) \forall i \quad \text{and} \quad \exists i: f_i(x') > f_i(x)$$

### Algorithms

1. **NSGA-II**: Non-dominated Sorting Genetic Algorithm II
2. **NSGA-III**: Reference-point based NSGA
3. **MOEA/D**: Multi-objective Evolutionary Algorithm with Decomposition
4. **Scalarization**: Weighted sum of objectives

### Example: Accuracy vs. Latency Trade-off

```
    Accuracy
       ↑
       │     ∙ Pareto Front
       │    ╱ ╲
       │   ╱   ╲  ∙
       │  ╱     ╲   ∙
       │ ╱       ╲    ∙
       │╱         ╲     ∙
       └────────────────→ Latency
       Fast           Slow
```

---

## 📊 Performance Prediction

Performance prediction enables zero-shot architecture evaluation without training.

### Prediction Metrics

| Metric | Description | Speed |
|--------|-------------|-------|
| `synflow` | Synaptic Flow | Fast |
| `grad_norm` | Gradient Norm | Fast |
| `nas_wot` | NAS without Training | Fast |
| `fisher` | Fisher Information | Medium |
| `zen_score` | Zero-shot Ensemble | Fast |

### Synaptic Flow

Synaptic Flow (SynFlow) predicts trainability based on the product of singular values:

```python
def synflow_score(architecture, input_shape):
    """Calculate Synaptic Flow score."""
    model = build_architecture(architecture)
    x = torch.randn(input_shape)
    
    # Forward pass with ones initialization
    model = initialize_with_ones(model)
    output = model(x)
    
    # Sum of absolute values
    score = torch.sum(torch.abs(output)).item()
    
    return score
```

### Prediction-Based Search

```python
from paradigm.metrics import PerformancePredictor

# Initialize predictor
predictor = PerformancePredictor(metric='synflow')

# Score architectures without training
scores = predictor.batch_score(architectures, input_shape=(1, 3, 32, 32))

# Use scores for early pruning
promising = [arch for arch, score in zip(architectures, scores) 
             if score > threshold]
```

---

## 🔄 Transfer Learning

Transfer learning enables knowledge sharing across related NAS tasks.

### Transfer Strategies

#### 1. Weight Inheritance
Transfer weights from parent to child networks:
```python
from paradigm.transfer import WeightInheritance

transfer = WeightInheritance(
    strategy='last_layer',  # 'last_layer', 'partial', 'full'
    similarity_threshold=0.8
)
```

#### 2. Performance Prediction Transfer
Use performance from similar architectures:
```python
from paradigm.transfer import PerformanceTransfer

transfer = PerformanceTransfer(
    similarity_metric='topology',
    k_neighbors=5
)
```

#### 3. Architecture Encoding Transfer
Transfer learned architecture patterns:
```python
from paradigm.transfer import ArchitectureTransfer

transfer = ArchitectureTransfer(
    embedding_dim=64,
    alignment='optimal_transport'
)
```

### Cross-Task Transfer

```
┌─────────────────────────────────────────────────────────┐
│              Cross-Task Transfer Learning                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Task A (CIFAR-10)                                     │
│   ┌──────────────┐                                      │
│   │  Search      │                                      │
│   │  History     │                                      │
│   └──────┬───────┘                                      │
│          │                                              │
│          ↓ (Transfer)                                   │
│   Task B (ImageNet)                                     │
│   ┌──────────────┐                                      │
│   │  Informed    │                                      │
│   │  Search      │                                      │
│   └──────────────┘                                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## ✂️ Early Pruning Strategies

Early pruning reduces search cost by eliminating unpromising architectures early.

### Pruning Criteria

| Strategy | Description | Threshold |
|----------|-------------|-----------|
| Performance Prediction | Use zero-cost metrics | Top-K |
| Architecture Complexity | Penalize complex archs | Threshold |
| Diversity | Maintain population diversity | Min-distance |
| Gradient | Use gradient information | Norm < ε |

### Scheduled Pruning

```python
from paradigm.optimizers import ScheduledPruning

pruner = ScheduledPruning(
    schedule=[
        (0, 1.0),    # Generation 0: keep all
        (20, 0.5),   # Generation 20: keep 50%
        (50, 0.2),   # Generation 50: keep 20%
    ],
    metric='synflow'
)
```

### Confidence-Based Pruning

```python
# Prune based on prediction confidence
for arch in architectures:
    mean, std = predictor.predict_with_uncertainty(arch)
    if std < confidence_threshold:
        # High confidence in poor performance
        if mean < threshold:
            prune(arch)
```

---

## 📈 Benchmarks

### NAS-Bench-101

A benchmark of 423,624 unique neural networks trained on CIFAR-10.

| Metric | Value |
|--------|-------|
| Unique architectures | 423,624 |
| Trained epochs | 108 |
| Dataset | CIFAR-10 |
| Tasks | Classification |

### NAS-Bench-201

Extended benchmark with multiple datasets.

| Dataset | Classes | Image Size |
|---------|---------|------------|
| CIFAR-10 | 10 | 32×32 |
| CIFAR-100 | 100 | 32×32 |
| ImageNet-16-120 | 120 | 16×16 |

### DARTS Benchmark

Evaluation on CIFAR-10 search space.

| Method | Test Error (%) | Params (M) | GPU Days |
|--------|----------------|------------|----------|
| Random Search | 3.90 | 3.2 | - |
| RL Controller | 3.75 | 3.4 | 4.0 |
| Evolution | 3.70 | 3.5 | 4.0 |
| DARTS-V1 | 3.00 | 3.3 | 1.5 |
| DARTS-V2 | 2.82 | 3.3 | 1.0 |

---

## 📚 API Reference

### Core Classes

#### `SearchSpace`

Base class for all search spaces.

```python
class SearchSpace(ABC):
    @abstractmethod
    def sample(self) -> Architecture:
        """Sample a random architecture."""
        pass
    
    @abstractmethod
    def mutate(self, architecture: Architecture) -> Architecture:
        """Mutate an architecture."""
        pass
    
    @abstractmethod
    def crossover(self, a: Architecture, b: Architecture) -> Architecture:
        """Crossover two architectures."""
        pass
    
    @abstractmethod
    def build(self, architecture: Architecture) -> nn.Module:
        """Build PyTorch model from architecture."""
        pass
    
    @abstractmethod
    def encode(self, architecture: Architecture) -> np.ndarray:
        """Encode architecture as fixed-size vector."""
        pass
```

#### `Controller`

Base class for architecture controllers.

```python
class Controller(ABC):
    def __init__(self, search_space: SearchSpace, **kwargs):
        self.search_space = search_space
        self.params = ...  # Controller parameters
    
    @abstractmethod
    def sample(self, batch_size: int = 1) -> List[Architecture]:
        """Sample architectures from controller."""
        pass
    
    @abstractmethod
    def update(self, batch: List[Architecture], rewards: List[float]):
        """Update controller based on rewards."""
        pass
    
    def save(self, path: str):
        """Save controller checkpoint."""
        pass
    
    def load(self, path: str):
        """Load controller checkpoint."""
        pass
```

#### `Optimizer`

Base class for architecture optimizers.

```python
class Optimizer(ABC):
    def __init__(self, search_space: SearchSpace, **kwargs):
        self.search_space = search_space
    
    @abstractmethod
    def search(self, *args, **kwargs) -> OptimizationResult:
        """Run architecture search."""
        pass
    
    def _evaluate(self, architecture: Architecture) -> float:
        """Evaluate single architecture."""
        pass
    
    def _train_child_network(self, architecture: Architecture) -> float:
        """Train child network and return validation accuracy."""
        pass
```

### Configuration

```python
@dataclass
class NASConfig:
    # Search settings
    search_space: str = 'cnn'
    algorithm: str = 'rl_controller'
    num_trials: int = 100
    
    # Training settings
    epochs_per_trial: int = 50
    batch_size: int = 128
    learning_rate: float = 0.025
    
    # Controller settings
    controller_hidden_size: int = 100
    controller_num_layers: int = 2
    controller_lr: float = 0.00035
    
    # Regularization
    entropy_weight: float = 0.01
    baseline_weight: float = 0.99
    
    # Resource limits
    max_gpu_hours: Optional[float] = None
    max_architectures: Optional[int] = None
```

---

## 💡 Examples

### Example 1: Basic NAS Search

```python
import paradigm
from paradigm.search_spaces import CNNSearchSpace
from paradigm.controllers import RLController

# Create search space
search_space = CNNSearchSpace(
    max_layers=8,
    num_operations=8,
    channels=[16, 32, 64]
)

# Create controller
controller = RLController(
    search_space=search_space,
    hidden_size=64,
    num_layers=2
)

# Run search
results = paradigm.search(
    controller=controller,
    num_trials=50,
    epochs_per_trial=25,
    dataset='cifar10'
)

# Print results
print(f"Best architecture: {results.best_architecture}")
print(f"Best accuracy: {results.best_accuracy:.2f}%")
print(f"Total time: {results.total_time:.2f}s")
```

### Example 2: DARTS Search

```python
from paradigm.optimizers import DARTSOptimizer
from paradigm.search_spaces import DARTSSearchSpace

# Create DARTS search space
search_space = DARTSSearchSpace(
    num_nodes=4,
    num_operations=8
)

# Create optimizer
optimizer = DARTSOptimizer(
    search_space=search_space,
    unrolled_steps=1,
    weight_decay=3e-4
)

# Run search
best_arch = optimizer.search(
    dataset='cifar10',
    epochs=50,
    batch_size=64,
    lr=0.025
)

# Build final model
model = search_space.build(best_arch)
```

### Example 3: Multi-Objective Search

```python
from paradigm.optimizers import MultiObjectiveOptimizer
from paradigm.metrics import LatencyMetric, ParameterMetric

# Define objectives
optimizer = MultiObjectiveOptimizer(
    search_space=search_space,
    objectives=[
        ('accuracy', 1.0),       # Primary: maximize accuracy
        ('latency', 0.5, LatencyMetric(device='cuda')),
        ('params', 0.3, ParameterMetric()),
    ],
    algorithm='nsga2',
    population_size=50
)

# Get Pareto front
pareto_front = optimizer.search(num_generations=100)

# Analyze trade-offs
for arch in pareto_front:
    print(f"Acc: {arch.fitness['accuracy']:.2f}, "
          f"Latency: {arch.fitness['latency']:.2f}ms, "
          f"Params: {arch.fitness['params']:.1f}M")
```

### Example 4: Performance Prediction

```python
from paradigm.metrics import PerformancePredictor

# Use multiple zero-cost metrics
predictor = PerformancePredictor(
    metrics=['synflow', 'nas_wot', 'grad_norm']
)

# Score architectures
architectures = [search_space.sample() for _ in range(100)]
scores = predictor.ensemble_score(architectures, input_shape=(1, 3, 32, 32))

# Sort by predicted performance
ranked = sorted(zip(architectures, scores), key=lambda x: x[1], reverse=True)

# Train top 10
top_architectures = [arch for arch, _ in ranked[:10]]
```

---

## 🤝 Contributing

We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/moggan1337/Paradigm.git
cd Paradigm

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black .
isort .
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 Citation

If you use Paradigm in your research, please cite:

```bibtex
@software{paradigm_nas,
  title = {Paradigm: AutoML with Neural Architecture Search},
  author = {Paradigm Contributors},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/moggan1337/Paradigm}}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- DARTS implementation inspired by [DARTS repository](https://github.com/quark0/darts)
- NAS-Bench-101/201 from [NAS-Bench repository](https://github.com/automl/NAS-Bench-101)
- NSGA-II implementation based on [pymoo](https://pymoo.org/)

---

<p align="center">
  Made with ❤️ by the Paradigm Team
</p>
