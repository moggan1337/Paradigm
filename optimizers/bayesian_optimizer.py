"""
Bayesian Optimization for architecture search.

This module implements Bayesian optimization using Gaussian Processes
for efficient hyperparameter and architecture optimization.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import time
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

from ..core.search_space import SearchSpace, Architecture
from ..core.optimizer import Optimizer, OptimizationResult


@dataclass
class HyperparameterConfig:
    """Configuration for a single hyperparameter."""
    
    name: str
    min_value: float
    max_value: float
    log_scale: bool = False
    categorical: bool = False
    choices: List[Any] = None


class GaussianProcessSurrogate:
    """
    Gaussian Process surrogate model for architecture performance prediction.
    
    The GP models the relationship between architecture encoding
    and performance, enabling efficient acquisition function optimization.
    
    Kernel: Constant * RBF + WhiteKernel for noise
    
    ┌─────────────────────────────────────────────────────────┐
    │            Gaussian Process Surrogate                   │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │   Training Data:                                        │
    │     X = [arch_encoding_1, arch_encoding_2, ...]         │
    │     y = [accuracy_1, accuracy_2, ...]                   │
    │                                                          │
    │   GP Model:                                             │
    │     f(x) ~ GP(m(x), k(x, x'))                          │
    │                                                          │
    │   Prediction:                                           │
    │     μ(x), σ(x) = GP.predict(x)                         │
    │                                                          │
    │   Acquisition:                                          │
    │     EI(x) = Expected Improvement                       │
    │     UCB(x) = Upper Confidence Bound                    │
    │     PI(x) = Probability of Improvement                 │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        kernel_type: str = 'rbf',
        length_scale: float = 1.0,
        noise_std: float = 0.1,
        n_restarts: int = 5
    ):
        """
        Initialize GP surrogate.
        
        Args:
            kernel_type: Type of kernel ('rbf', 'matern')
            length_scale: Initial length scale
            noise_std: Noise standard deviation
            n_restarts: Number of optimizer restarts
        """
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.noise_std = noise_std
        self.n_restarts = n_restarts
        
        self.gp = None
        self.X_train = []
        self.y_train = []
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the GP model.
        
        Args:
            X: Training features [n_samples, n_features]
            y: Training targets [n_samples]
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        # Normalize targets
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-8
        y_normalized = (y - self.y_mean) / self.y_std
        
        # Build kernel
        n_features = X.shape[1]
        
        if self.kernel_type == 'rbf':
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3)) *
                RBF(length_scale=self.length_scale, length_scale_bounds=(1e-2, 1e2))
            )
        else:  # matern
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3)) *
                Matern(length_scale=self.length_scale, nu=2.5)
            )
        
        # Add noise kernel
        kernel = kernel + WhiteKernel(noise_level=self.noise_std ** 2)
        
        # Create and fit GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
            random_state=42
        )
        
        self.gp.fit(X, y_normalized)
        self._fitted = True
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance.
        
        Args:
            X: Query points [n_samples, n_features]
            return_std: Whether to return standard deviation
            
        Returns:
            (mean, std) if return_std=True, else mean only
        """
        if not self._fitted:
            raise ValueError("GP not fitted. Call fit() first.")
        
        X = np.array(X)
        
        if return_std:
            y_pred, y_std = self.gp.predict(X, return_std=True)
            # Denormalize
            y_pred = y_pred * self.y_std + self.y_mean
            y_std = y_std * self.y_std
            return y_pred, y_std
        else:
            y_pred = self.gp.predict(X)
            return y_pred * self.y_std + self.y_mean
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[float, float]:
        """
        Predict with uncertainty for a single point.
        
        Args:
            X: Query point [n_features]
            
        Returns:
            (mean, std) tuple
        """
        X = np.array(X).reshape(1, -1)
        mean, std = self.predict(X, return_std=True)
        return float(mean[0]), float(std[0])


class BayesianOptimizer(Optimizer):
    """
    Bayesian optimization for architecture search.
    
    This optimizer uses a Gaussian Process surrogate model
    to efficiently explore the architecture search space.
    
    Algorithm:
    ┌─────────────────────────────────────────────────────────┐
    │            Bayesian Optimization Loop                  │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │  1. Initialize:                                        │
    │     - Random architectures (n_init)                     │
    │     - Evaluate performance                             │
    │                                                          │
    │  2. Repeat until budget exhausted:                     │
    │     - Fit GP surrogate on observed data                │
    │     - Optimize acquisition function:                   │
    │       x* = argmax_a(x)                                  │
    │     - Evaluate architecture at x*                      │
    │     - Update observations                              │
    │                                                          │
    │  3. Return best architecture                           │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    
    Acquisition Functions:
    - Expected Improvement (EI)
    - Upper Confidence Bound (UCB)
    - Probability of Improvement (PI)
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        acquisition: str = 'ei',
        n_initial_points: int = 10,
        acq_optimizer: str = 'lbfgs',
        exploration_weight: float = 0.01,
        kernel_type: str = 'rbf',
        **kwargs
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            search_space: Search space
            acquisition: Acquisition function ('ei', 'ucb', 'pi')
            n_initial_points: Number of random initial points
            acq_optimizer: Acquisition optimizer ('lbfgs', 'random')
            exploration_weight: Exploration weight for UCB
            kernel_type: GP kernel type
        """
        super().__init__(search_space, **kwargs)
        
        self.acquisition = acquisition
        self.n_initial_points = n_initial_points
        self.acq_optimizer = acq_optimizer
        self.exploration_weight = exploration_weight
        
        # Initialize surrogate
        self.surrogate = GaussianProcessSurrogate(kernel_type=kernel_type)
        
        # Storage
        self.observations: List[Tuple[Architecture, float]] = []
        self.encodings: List[np.ndarray] = []
        self.performance: List[float] = []
    
    def search(
        self,
        num_trials: int = 100,
        dataset: str = "cifar10",
        epochs_per_trial: int = 25,
        **kwargs
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            num_trials: Maximum number of evaluations
            dataset: Dataset name
            epochs_per_trial: Training epochs per architecture
            **kwargs: Additional arguments
            
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(dataset)
        
        # Initialize result
        result = OptimizationResult(
            best_architecture=None,
            best_accuracy=0.0,
        )
        
        # Phase 1: Random initialization
        print(f"Phase 1: Random initialization ({self.n_initial_points} points)")
        
        for i in range(self.n_initial_points):
            # Sample random architecture
            architecture = self.search_space.sample()
            
            # Encode
            encoding = self.search_space.encode(architecture)
            
            # Evaluate
            accuracy, metrics = self._train_architecture(
                architecture, train_loader, val_loader, epochs_per_trial
            )
            
            # Record
            self.observations.append((architecture, accuracy))
            self.encodings.append(encoding)
            self.performance.append(accuracy)
            
            result.all_architectures.append(architecture)
            result.all_rewards.append(accuracy)
            
            if accuracy > result.best_accuracy:
                result.best_accuracy = accuracy
                result.best_architecture = architecture
            
            print(f"  Init {i + 1}/{self.n_initial_points}: Accuracy = {accuracy:.2f}%")
        
        # Phase 2: Bayesian optimization
        print(f"Phase 2: Bayesian optimization ({num_trials - self.n_initial_points} trials)")
        
        # Fit initial surrogate
        self._fit_surrogate()
        
        for trial in range(self.n_initial_points, num_trials):
            # Find next point using acquisition function
            next_encoding = self._optimize_acquisition()
            
            # Decode to architecture
            architecture = self._encoding_to_architecture(next_encoding)
            
            # Evaluate
            accuracy, metrics = self._train_architecture(
                architecture, train_loader, val_loader, epochs_per_trial
            )
            
            # Update observations
            self.observations.append((architecture, accuracy))
            self.encodings.append(next_encoding)
            self.performance.append(accuracy)
            
            # Refit surrogate
            self._fit_surrogate()
            
            # Record
            result.all_architectures.append(architecture)
            result.all_rewards.append(accuracy)
            
            if accuracy > result.best_accuracy:
                result.best_accuracy = accuracy
                result.best_architecture = architecture
            
            result.history.append({
                'trial': trial,
                'accuracy': accuracy,
                'best_accuracy': result.best_accuracy,
                'method': 'bayesian',
            })
            
            print(f"Trial {trial + 1}/{num_trials}: Accuracy = {accuracy:.2f}%, Best = {result.best_accuracy:.2f}%")
        
        result.search_time = time.time() - start_time
        self.current_result = result
        return result
    
    def _fit_surrogate(self):
        """Fit the GP surrogate model."""
        if len(self.encodings) < 2:
            return
        
        X = np.array(self.encodings)
        y = np.array(self.performance)
        
        self.surrogate.fit(X, y)
    
    def _optimize_acquisition(self) -> np.ndarray:
        """
        Optimize acquisition function to find next point.
        
        Returns:
            Encoding of next architecture to evaluate
        """
        if self.acq_optimizer == 'random':
            # Random sampling from search space
            return self.search_space.encode(self.search_space.sample())
        
        # Grid search over random architectures
        best_acq = float('-inf')
        best_encoding = None
        
        n_candidates = 100
        
        for _ in range(n_candidates):
            # Sample candidate
            arch = self.search_space.sample()
            encoding = self.search_space.encode(arch)
            
            # Compute acquisition value
            acq = self._acquisition_function(encoding)
            
            if acq > best_acq:
                best_acq = acq
                best_encoding = encoding
        
        return best_encoding
    
    def _acquisition_function(self, encoding: np.ndarray) -> float:
        """
        Compute acquisition function value.
        
        Args:
            encoding: Architecture encoding
            
        Returns:
            Acquisition value
        """
        encoding = np.array(encoding).reshape(1, -1)
        
        # Get prediction
        mean, std = self.surrogate.predict(encoding, return_std=True)
        mean = float(mean[0])
        std = float(std[0]) + 1e-8
        
        # Best observed value
        best = max(self.performance)
        
        if self.acquisition == 'ei':
            # Expected Improvement
            z = (mean - best - self.exploration_weight) / std
            ei = (mean - best - self.exploration_weight) * norm.cdf(z) + std * norm.pdf(z)
            return ei
        
        elif self.acquisition == 'ucb':
            # Upper Confidence Bound
            return mean + self.exploration_weight * std
        
        elif self.acquisition == 'pi':
            # Probability of Improvement
            z = (mean - best - self.exploration_weight) / std
            return norm.cdf(z)
        
        else:
            return mean
    
    def _encoding_to_architecture(self, encoding: np.ndarray) -> Architecture:
        """
        Convert encoding to architecture.
        
        For Bayesian optimization, we sample architectures and find
        the closest match to the desired encoding.
        
        Args:
            encoding: Target encoding
            
        Returns:
            Architecture with closest encoding
        """
        # Sample candidates and find closest
        best_dist = float('inf')
        best_arch = None
        
        for _ in range(50):
            arch = self.search_space.sample()
            arch_encoding = self.search_space.encode(arch)
            
            dist = np.linalg.norm(encoding - arch_encoding)
            
            if dist < best_dist:
                best_dist = dist
                best_arch = arch
        
        return best_arch


class TPEOptimizer(BayesianOptimizer):
    """
    Tree-structured Parzen Estimator (TPE) optimizer.
    
    TPE is an alternative to GP-based Bayesian optimization
    that models P(x|y) and P(y) using tree-structured density estimators.
    
    Reference:
        Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011).
        Algorithms for Hyper-Parameter Optimization. NeurIPS 2011.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        n_startup_trials: int = 10,
        gamma: float = 0.5,
        **kwargs
    ):
        """
        Initialize TPE optimizer.
        
        Args:
            search_space: Search space
            n_startup_trials: Number of random trials before TPE
            gamma: Fraction of observations for good/bad models
        """
        super().__init__(search_space, **kwargs)
        
        self.n_startup_trials = n_startup_trials
        self.gamma = gamma
    
    def _fit_surrogate(self):
        """Fit TPE models."""
        # Sort observations by performance
        sorted_obs = sorted(zip(self.performance, self.encodings), reverse=True)
        
        # Split into good and bad
        n_good = int(len(sorted_obs) * self.gamma)
        
        if n_good < 1:
            return
        
        good_obs = sorted_obs[:n_good]
        bad_obs = sorted_obs[n_good:]
        
        # Store for TPE sampling
        self.good_encodings = np.array([enc for _, enc in good_obs])
        self.bad_encodings = np.array([enc for _, enc in bad_obs])
        self.good_threshold = good_obs[-1][0] if good_obs else -float('inf')
    
    def _acquisition_function(self, encoding: np.ndarray) -> float:
        """
        Compute TPE acquisition (ratio of likelihoods).
        
        Args:
            encoding: Architecture encoding
            
        Returns:
            Acquisition value
        """
        if not hasattr(self, 'good_encodings') or len(self.good_encodings) == 0:
            return 0.0
        
        encoding = np.array(encoding).reshape(1, -1)
        
        # Compute simple density ratio
        # In practice, use KDE or histograms
        good_mean = np.mean(self.good_encodings, axis=0)
        bad_mean = np.mean(self.bad_encodings, axis=0) if len(self.bad_encodings) > 0 else good_mean
        
        # Distance to good center vs bad center
        dist_good = np.linalg.norm(encoding - good_mean)
        dist_bad = np.linalg.norm(encoding - bad_mean) + 1e-8
        
        # Maximize this ratio
        return 1.0 / (dist_good + 1e-8) - 0.5 / (dist_bad + 1e-8)
