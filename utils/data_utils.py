"""
Data utilities for loading and processing datasets.
"""

from typing import Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms


def get_cifar10_loader(
    batch_size: int = 128,
    data_dir: str = './data',
    val_split: float = 0.1,
    num_workers: int = 4,
    download: bool = True,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 data loaders.
    
    Args:
        batch_size: Batch size
        data_dir: Data directory
        val_split: Validation split ratio
        num_workers: Number of workers
        download: Whether to download data
        augment: Whether to apply data augmentation
        
    Returns:
        (train_loader, val_loader) tuple
    """
    # Transform for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Transform for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load full training set
    full_train_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform if augment else val_transform
    )
    
    # Split into train and val
    n_samples = len(full_train_set)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    indices = torch.randperm(n_samples).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = Subset(full_train_set, train_indices)
    val_dataset = Subset(
        torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False,
            transform=val_transform
        ),
        val_indices
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_cifar100_loader(
    batch_size: int = 128,
    data_dir: str = './data',
    val_split: float = 0.1,
    num_workers: int = 4,
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 data loaders.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    full_train_set = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform
    )
    
    n_samples = len(full_train_set)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    indices = torch.randperm(n_samples).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = Subset(full_train_set, train_indices)
    val_dataset = Subset(
        torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=False,
            transform=val_transform
        ),
        val_indices
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_imagenet_loader(
    batch_size: int = 256,
    data_dir: str = './data',
    val_split: float = 0.1,
    num_workers: int = 8,
    download: bool = False,
    subset_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNet data loaders.
    
    Note: Full ImageNet is large, consider using subset for NAS.
    """
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load data
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"ImageNet not found at {data_dir}")
        print("Please download and extract ImageNet to the data directory")
        return None, None
    
    train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transform)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transform)
    
    # Optional subset
    if subset_size is not None:
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


class TinyImageNetLoader:
    """
    Loader for Tiny ImageNet dataset.
    
    Tiny ImageNet has 200 classes, 100k training images,
    and 10k validation images at 64x64 resolution.
    """
    
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        num_workers: int = 4
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation loaders."""
        normalize = transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975],
            std=[0.2302, 0.2265, 0.2262]
        )
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        train_dir = self.data_dir / 'train'
        val_dir = self.data_dir / 'val'
        
        if not train_dir.exists():
            print(f"Tiny ImageNet not found at {self.data_dir}")
            return None, None
        
        train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transform)
        val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader


def get_dataset(
    name: str,
    data_dir: str = './data',
    batch_size: int = 128,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Get data loader by dataset name.
    
    Args:
        name: Dataset name ('cifar10', 'cifar100', 'imagenet')
        data_dir: Data directory
        batch_size: Batch size
        
    Returns:
        (train_loader, val_loader) tuple
    """
    name = name.lower()
    
    if name == 'cifar10':
        return get_cifar10_loader(batch_size=batch_size, data_dir=data_dir, **kwargs)
    elif name == 'cifar100':
        return get_cifar100_loader(batch_size=batch_size, data_dir=data_dir, **kwargs)
    elif name == 'imagenet':
        return get_imagenet_loader(batch_size=batch_size, data_dir=data_dir, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
