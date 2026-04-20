"""
Logging utilities for Paradigm.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import time

from ..core.optimizer import OptimizationResult


def setup_logger(
    name: str = "paradigm",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


@dataclass
class SearchMetrics:
    """Metrics for a single search trial."""
    
    trial: int
    timestamp: float
    accuracy: float
    loss: float
    epoch_time: float
    lr: float
    metadata: Dict[str, Any]


class MetricsLogger:
    """
    Logger for tracking metrics during architecture search.
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        # Create experiment directory
        self.exp_dir = self.log_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics file
        self.metrics_file = self.exp_dir / "metrics.jsonl"
        
        # Initialize
        self.trial_count = 0
        self.start_time = time.time()
    
    def log_trial(
        self,
        accuracy: float,
        loss: float = 0.0,
        epoch_time: float = 0.0,
        lr: float = 0.0,
        **metadata
    ):
        """
        Log metrics for a single trial.
        
        Args:
            accuracy: Trial accuracy
            loss: Trial loss
            epoch_time: Time per epoch
            lr: Learning rate
            **metadata: Additional metadata
        """
        self.trial_count += 1
        
        metrics = SearchMetrics(
            trial=self.trial_count,
            timestamp=time.time() - self.start_time,
            accuracy=accuracy,
            loss=loss,
            epoch_time=epoch_time,
            lr=lr,
            metadata=metadata
        )
        
        # Write to file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_accuracy: float,
        lr: float,
        **metadata
    ):
        """
        Log metrics for a single epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_accuracy: Validation accuracy
            lr: Learning rate
            **metadata: Additional metadata
        """
        metadata.update({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'lr': lr,
        })
        
        # Write to file
        epoch_file = self.exp_dir / f"epoch_{epoch}.json"
        with open(epoch_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_result(self, result: OptimizationResult):
        """
        Save final optimization result.
        
        Args:
            result: Optimization result
        """
        result_file = self.exp_dir / "result.json"
        
        result_dict = result.to_dict()
        
        # Convert numpy arrays and non-serializable objects
        result_dict['best_architecture'] = result.best_architecture.to_dict() if result.best_architecture else None
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of logged metrics.
        
        Returns:
            Summary dictionary
        """
        if not self.metrics_file.exists():
            return {}
        
        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
        
        if not metrics:
            return {}
        
        accuracies = [m['accuracy'] for m in metrics]
        
        return {
            'num_trials': len(metrics),
            'best_accuracy': max(accuracies),
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'total_time': metrics[-1]['timestamp'] if metrics else 0,
        }


def log_metrics(
    metrics: Dict[str, float],
    prefix: str = "",
    logger: Optional[logging.Logger] = None
):
    """
    Log metrics with optional prefix.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for log message
        logger: Logger instance (uses default if None)
    """
    if logger is None:
        logger = logging.getLogger("paradigm")
    
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    
    message = f"{prefix}: {', '.join(parts)}" if prefix else ', '.join(parts)
    logger.info(message)


class TensorBoardLogger:
    """
    TensorBoard logger for experiment tracking.
    
    Requires tensorboard package.
    """
    
    def __init__(
        self,
        log_dir: str = "./runs",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Name of experiment
        """
        self.log_dir = Path(log_dir)
        
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_dir = self.log_dir / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
            self.enabled = True
        except ImportError:
            self.writer = None
            self.enabled = False
            print("TensorBoard not available. Install with: pip install tensorboard")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalar values."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram of values."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_architecture(self, tag: str, architecture, step: int):
        """Log architecture as text."""
        if self.enabled:
            self.writer.add_text(tag, architecture.visualize(), step)
    
    def close(self):
        """Close the writer."""
        if self.enabled:
            self.writer.close()
