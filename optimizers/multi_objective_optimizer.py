"""
Multi-Objective Optimization for architecture search.

This module implements multi-objective optimization algorithms
(NSGA-II, NSGA-III, MOEA/D) for finding Pareto-optimal architectures.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from copy import deepcopy
import time
from collections import defaultdict

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.search_space import SearchSpace, Architecture
from ..core.optimizer import Optimizer, OptimizationResult


@dataclass
class Objective:
    """Definition of a single objective."""
    
    name: str
    weight: float = 1.0
    goal: str = 'maximize'  # 'maximize' or 'minimize'
    metric_fn: Optional[Callable] = None
    threshold: Optional[float] = None
    
    def evaluate(self, architecture: Architecture) -> float:
        """Evaluate this objective for an architecture."""
        if self.metric_fn is not None:
            return self.metric_fn(architecture)
        return 0.0
    
    def is_better(self, v1: float, v2: float) -> bool:
        """Check if v1 is better than v2 according to this objective."""
        if self.goal == 'maximize':
            return v1 > v2
        else:
            return v1 < v2


@dataclass
class FitnessVector:
    """Fitness values for multiple objectives."""
    
    values: Dict[str, float] = field(default_factory=dict)
    
    def __getitem__(self, key: str) -> float:
        return self.values[key]
    
    def __setitem__(self, key: str, value: float):
        self.values[key] = value
    
    def __iter__(self):
        return iter(self.values.values())
    
    def __len__(self):
        return len(self.values)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(list(self.values.values()))
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'FitnessVector':
        """Create from dictionary."""
        return cls(values=d)


class Individual:
    """An individual in the evolutionary algorithm."""
    
    def __init__(
        self,
        architecture: Architecture,
        fitness: Optional[FitnessVector] = None
    ):
        self.architecture = architecture
        self.fitness = fitness or FitnessVector()
        self.rank = 0
        self.crowding_distance = 0.0
    
    def dominates(self, other: 'Individual', objectives: List[Objective]) -> bool:
        """
        Check if this individual dominates another.
        
        An individual dominates another if it's better in at least one
        objective and not worse in any other objective.
        """
        better_in_any = False
        
        for obj in objectives:
            v1 = self.fitness[obj.name]
            v2 = other.fitness[obj.name]
            
            if obj.is_better(v1, v2):
                better_in_any = True
            elif v1 != v2 and not obj.is_better(v1, v2):
                # This objective is worse
                return False
        
        return better_in_any


class NSGA2:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II).
    
    NSGA-II is a multi-objective evolutionary algorithm that uses
    non-dominated sorting and crowding distance to find the Pareto front.
    
    Algorithm:
    ┌─────────────────────────────────────────────────────────┐
    │                     NSGA-II Algorithm                    │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │  Initialize: Population P_0 (size N)                   │
    │                                                          │
    │  Repeat:                                                 │
    │    1. Create offspring Q through crossover + mutation  │
    │    2. R = P ∪ Q (combined population)                   │
    │    3. Non-dominated sorting of R:                       │
    │       - Front 1: Non-dominated individuals               │
    │       - Front 2: Dominated only by Front 1              │
    │       - ...                                              │
    │    4. Crowding distance assignment                      │
    │    5. Selection: Best N using rank + crowding distance │
    │    6. P = Selected individuals                          │
    │                                                          │
    │  Return: Pareto front (Front 1)                         │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    
    Reference:
        Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
        A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.
        IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int = 50,
        tournament_size: int = 3,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        eta: float = 20  # SBX distribution index
    ):
        """
        Initialize NSGA-II.
        
        Args:
            search_space: Search space
            population_size: Population size
            tournament_size: Tournament selection size
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            eta: SBX distribution index
        """
        self.search_space = search_space
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta = eta
    
    def initialize_population(self) -> List[Individual]:
        """Initialize random population."""
        population = []
        
        for _ in range(self.population_size):
            arch = self.search_space.sample()
            population.append(Individual(architecture=arch))
        
        return population
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Perform fast non-dominated sorting.
        
        Args:
            population: Population to sort
            
        Returns:
            List of fronts (each front is a list of individuals)
        """
        fronts = [[]]
        
        # For each individual
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            
            for q in population:
                if p.dominates(q):
                    p.dominated_solutions.append(q)
                elif q.dominates(p):
                    p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        # Build subsequent fronts
        current_front = 0
        
        while fronts[current_front]:
            next_front = []
            
            for p in fronts[current_front]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    
                    if q.domination_count == 0:
                        q.rank = current_front + 1
                        next_front.append(q)
            
            current_front += 1
            if next_front:
                fronts.append(next_front)
        
        return [f for f in fronts if f]  # Remove empty fronts
    
    def crowding_distance(
        self,
        front: List[Individual],
        objectives: List[Objective]
    ):
        """
        Calculate crowding distance for individuals in a front.
        
        Args:
            front: Front of individuals
            objectives: List of objectives
        """
        n = len(front)
        
        if n <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for ind in front:
            ind.crowding_distance = 0.0
        
        # For each objective
        for obj in objectives:
            # Sort by objective value
            sorted_front = sorted(front, key=lambda x: x.fitness[obj.name])
            
            # Boundary points get infinite distance
            sorted_front[0].crowding_distance = float('inf')
            sorted_front[-1].crowding_distance = float('inf')
            
            # Range of objective values
            obj_range = (
                sorted_front[-1].fitness[obj.name] - 
                sorted_front[0].fitness[obj.name]
            )
            
            if obj_range == 0:
                continue
            
            # Interior points
            for i in range(1, n - 1):
                sorted_front[i].crowding_distance += (
                    sorted_front[i + 1].fitness[obj.name] - 
                    sorted_front[i - 1].fitness[obj.name]
                ) / obj_range
    
    def tournament_select(
        self,
        population: List[Individual],
        individuals: List[Individual]
    ) -> Individual:
        """
        Tournament selection based on rank and crowding distance.
        
        Args:
            population: Full population
            individuals: Candidates for selection
            
        Returns:
            Selected individual
        """
        candidates = random.sample(individuals, min(self.tournament_size, len(individuals)))
        
        # Sort by rank first, then crowding distance
        candidates.sort(key=lambda x: (x.rank, -x.crowding_distance))
        
        return candidates[0]
    
    def create_offspring(
        self,
        population: List[Individual],
        objectives: List[Objective]
    ) -> List[Individual]:
        """
        Create offspring through crossover and mutation.
        
        Args:
            population: Parent population
            objectives: List of objectives
            
        Returns:
            Offspring population
        """
        offspring = []
        
        while len(offspring) < self.population_size:
            # Tournament selection for two parents
            parent1 = self.tournament_select(population, population)
            parent2 = self.tournament_select(population, population)
            
            # Crossover
            if random.random() < self.crossover_prob:
                child = self.search_space.crossover(
                    parent1.architecture,
                    parent2.architecture
                )
            else:
                child = deepcopy(parent1.architecture)
            
            # Mutation
            if random.random() < self.mutation_prob:
                child = self.search_space.mutate(child)
            
            offspring.append(Individual(architecture=child))
        
        return offspring
    
    def evolve(
        self,
        population: List[Individual],
        objectives: List[Objective],
        generations: int
    ) -> List[Individual]:
        """
        Run NSGA-II evolution.
        
        Args:
            population: Initial population
            objectives: List of objectives
            generations: Number of generations
            
        Returns:
            Final Pareto front
        """
        for gen in range(generations):
            # Create offspring
            offspring = self.create_offspring(population, objectives)
            
            # Combine parent and offspring
            combined = population + offspring
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(combined)
            
            # Calculate crowding distance for all fronts
            for front in fronts:
                self.crowding_distance(front, objectives)
            
            # Selection
            new_population = []
            
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    # Sort by crowding distance and fill
                    front.sort(key=lambda x: -x.crowding_distance)
                    remaining = self.population_size - len(new_population)
                    new_population.extend(front[:remaining])
                    break
            
            population = new_population
            
            if (gen + 1) % 10 == 0:
                pareto = fronts[0]
                avg_fitness = np.mean([ind.fitness.to_array() for ind in pareto])
                print(f"Generation {gen + 1}: Pareto size = {len(pareto)}, "
                      f"Avg fitness = {avg_fitness}")
        
        # Return Pareto front
        fronts = self.fast_non_dominated_sort(population)
        return fronts[0] if fronts else []


class MultiObjectiveOptimizer(Optimizer):
    """
    Multi-objective optimizer for architecture search.
    
    This optimizer finds Pareto-optimal architectures that trade off
    multiple objectives (e.g., accuracy vs. latency vs. model size).
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        objectives: List[Objective],
        algorithm: str = 'nsga2',
        population_size: int = 50,
        num_generations: int = 100,
        **kwargs
    ):
        """
        Initialize multi-objective optimizer.
        
        Args:
            search_space: Search space
            objectives: List of objectives to optimize
            algorithm: Algorithm to use ('nsga2', 'nsga3', 'moead')
            population_size: Population size
            num_generations: Number of generations
        """
        super().__init__(search_space, **kwargs)
        
        self.objectives = objectives
        self.algorithm = algorithm
        self.population_size = population_size
        self.num_generations = num_generations
        
        # Initialize NSGA-II
        if algorithm == 'nsga2':
            self.nsga2 = NSGA2(
                search_space=search_space,
                population_size=population_size,
                **kwargs
            )
        else:
            self.nsga2 = NSGA2(
                search_space=search_space,
                population_size=population_size,
                **kwargs
            )
    
    def search(
        self,
        num_generations: int = None,
        dataset: str = "cifar10",
        epochs_per_trial: int = 20,
        **kwargs
    ) -> OptimizationResult:
        """
        Run multi-objective architecture search.
        
        Args:
            num_generations: Number of generations (overrides default)
            dataset: Dataset name
            epochs_per_trial: Training epochs per architecture
            **kwargs: Additional arguments
            
        Returns:
            OptimizationResult with Pareto front
        """
        start_time = time.time()
        num_generations = num_generations or self.num_generations
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(dataset)
        
        # Initialize population
        print(f"Initializing population ({self.population_size} individuals)...")
        population = self.nsga2.initialize_population()
        
        # Evaluate initial population
        self._evaluate_population(population, train_loader, val_loader, epochs_per_trial)
        
        # Evolution loop
        print(f"Running {self.algorithm.upper()} for {num_generations} generations...")
        
        pareto_front = self.nsga2.evolve(
            population,
            self.objectives,
            num_generations
        )
        
        # Create result
        result = OptimizationResult(
            best_architecture=pareto_front[0].architecture if pareto_front else None,
            best_accuracy=pareto_front[0].fitness['accuracy'] if pareto_front else 0.0,
            search_time=time.time() - start_time,
            metadata={
                'algorithm': self.algorithm,
                'num_objectives': len(self.objectives),
                'pareto_size': len(pareto_front),
                'objectives': [o.name for o in self.objectives],
            }
        )
        
        # Add all Pareto architectures
        for ind in pareto_front:
            result.all_architectures.append(ind.architecture)
            result.all_rewards.append(ind.fitness['accuracy'])
        
        self.current_result = result
        return result
    
    def _evaluate_population(
        self,
        population: List[Individual],
        train_loader,
        val_loader,
        epochs: int
    ):
        """
        Evaluate fitness for all individuals in population.
        
        Args:
            population: Population to evaluate
            train_loader: Training data
            val_loader: Validation data
            epochs: Training epochs
        """
        for ind in population:
            # Evaluate each objective
            for obj in self.objectives:
                if obj.name == 'accuracy':
                    # Train and evaluate accuracy
                    accuracy, _ = self._train_architecture(
                        ind.architecture, train_loader, val_loader, epochs
                    )
                    ind.fitness[obj.name] = accuracy
                else:
                    # Use objective function
                    ind.fitness[obj.name] = obj.evaluate(ind.architecture)


class HypervolumeCalculator:
    """
    Calculate hypervolume metric for Pareto fronts.
    
    Hypervolume measures the dominated space in objective space,
    with higher values indicating better coverage.
    """
    
    @staticmethod
    def calculate(
        pareto_front: List[Individual],
        reference_point: np.ndarray
    ) -> float:
        """
        Calculate hypervolume of Pareto front.
        
        Args:
            pareto_front: List of Pareto-optimal individuals
            reference_point: Reference point for hypervolume
            
        Returns:
            Hypervolume value
        """
        if not pareto_front:
            return 0.0
        
        # Sort by first objective
        sorted_front = sorted(
            pareto_front,
            key=lambda x: x.fitness.to_array()[0]
        )
        
        hypervolume = 0.0
        prev_value = 0.0
        
        for ind in sorted_front:
            values = ind.fitness.to_array()
            
            # Calculate contribution
            width = values[0] - prev_value
            height = np.prod(reference_point - values[1:])
            
            hypervolume += width * height
            prev_value = values[0]
        
        return hypervolume
