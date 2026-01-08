"""
Data loader for incremental learning tasks
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import List, Tuple
import numpy as np


class IncrementalDataset(Dataset):
    """
    Dataset wrapper for incremental learning scenarios
    """
    
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: Input data (images)
            labels: Target labels
            transform: Optional transform to apply to images
        """
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TaskDataLoader:
    """
    Manages data loading for different tasks in incremental learning
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.current_task = 0
        self.seen_classes = []
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def get_task_data(self, task_id: int) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and test dataloaders for a specific task
        
        Args:
            task_id: Task identifier
        
        Returns:
            train_loader, test_loader: Training and testing data loaders
        """
        # Calculate class range for this task
        start_class = task_id * self.config.NUM_CLASSES_PER_TASK
        end_class = (task_id + 1) * self.config.NUM_CLASSES_PER_TASK
        
        print(f"Loading Task {task_id}: Classes {start_class} to {end_class-1}")
        
        # This is a placeholder - in practice, you would load actual dataset
        # For example, using CIFAR-100, ImageNet, etc.
        train_data, train_labels = self._load_placeholder_data(start_class, end_class, train=True)
        test_data, test_labels = self._load_placeholder_data(start_class, end_class, train=False)
        
        train_dataset = IncrementalDataset(train_data, train_labels, transform=self.transform)
        test_dataset = IncrementalDataset(test_data, test_labels, transform=self.transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )
        
        # Update seen classes
        self.seen_classes.extend(range(start_class, end_class))
        
        return train_loader, test_loader
    
    def _load_placeholder_data(self, start_class: int, end_class: int, train: bool = True):
        """
        Placeholder function for data loading
        In practice, replace this with actual dataset loading logic
        
        Args:
            start_class: Starting class index
            end_class: Ending class index
            train: Whether to load training or test data
        
        Returns:
            data, labels: Image data and corresponding labels
        """
        # Generate dummy data for demonstration
        num_samples = 1000 if train else 200
        data = torch.randn(num_samples, 3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        labels = torch.randint(start_class, end_class, (num_samples,))
        
        return data, labels
    
    def get_memory_data(self, memory_images: torch.Tensor, memory_labels: torch.Tensor) -> DataLoader:
        """
        Create dataloader for memory/exemplar samples
        
        Args:
            memory_images: Tensor of stored exemplar images
            memory_labels: Tensor of corresponding labels
        
        Returns:
            memory_loader: DataLoader for memory samples
        """
        if len(memory_images) == 0:
            return None
        
        memory_dataset = IncrementalDataset(
            memory_images,
            memory_labels,
            transform=self.transform
        )
        
        memory_loader = DataLoader(
            memory_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS
        )
        
        return memory_loader
