"""
Utility functions for SD-based CIL
"""

import os
import torch
import random
import numpy as np
from pathlib import Path


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_directories(config):
    """
    Create necessary directories for the project
    
    Args:
        config: Configuration object
    """
    directories = [
        config.DATA_ROOT,
        config.CHECKPOINT_DIR,
        config.LOG_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Created directories: {', '.join(directories)}")


def save_checkpoint(model, optimizer, epoch, task_id, config, filename=None):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch
        task_id: Current task ID
        config: Configuration object
        filename: Optional custom filename
    """
    if filename is None:
        filename = f"checkpoint_task{task_id}_epoch{epoch}.pth"
    
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    
    checkpoint = {
        'epoch': epoch,
        'task_id': task_id,
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cpu'):
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        filepath: Path to checkpoint file
        device: Device to map checkpoint to
    
    Returns:
        epoch, task_id: Loaded epoch and task ID
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    task_id = checkpoint['task_id']
    
    print(f"Checkpoint loaded from: {filepath}")
    return epoch, task_id


def compute_accuracy(predictions, targets):
    """
    Compute classification accuracy
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
    
    Returns:
        accuracy: Accuracy value
    """
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = 100 * correct / total
    return accuracy
