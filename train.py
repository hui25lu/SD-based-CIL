"""
Main training script for SD-based Class Incremental Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from config import get_config
from models.sd_cil_model import create_model
from data.dataloader import TaskDataLoader
from utils.helpers import (
    set_seed,
    create_directories,
    save_checkpoint,
    compute_accuracy
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average training loss
        avg_acc: Average training accuracy
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predictions = torch.max(outputs, 1)
        correct = (predictions == labels).sum().item()
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / labels.size(0):.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100 * total_correct / total_samples
    
    return avg_loss, avg_acc


def evaluate(model, test_loader, device):
    """
    Evaluate model on test set
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        avg_acc: Average test accuracy
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_acc = 100 * total_correct / total_samples
    return avg_acc


def train_task(model, task_id, train_loader, test_loader, config, device):
    """
    Train model on a single task
    
    Args:
        model: Model to train
        task_id: Task identifier
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration object
        device: Device to train on
    
    Returns:
        model: Trained model
    """
    print(f"\n{'='*50}")
    print(f"Training Task {task_id}")
    print(f"{'='*50}\n")
    
    # Prepare model for new task
    model.prepare_for_task(task_id)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Evaluate
        if epoch % config.EVAL_FREQUENCY == 0:
            test_acc = evaluate(model, test_loader, device)
            print(f"Test Acc: {test_acc:.2f}%")
        
        # Save checkpoint
        if epoch % config.SAVE_FREQUENCY == 0:
            save_checkpoint(model, optimizer, epoch, task_id, config)
    
    # Final evaluation
    test_acc = evaluate(model, test_loader, device)
    print(f"\nTask {task_id} Final Test Accuracy: {test_acc:.2f}%\n")
    
    return model


def main():
    """
    Main training function
    """
    # Load configuration
    config = get_config()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create necessary directories
    create_directories(config)
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create model
    model = create_model(config)
    
    # Initialize data loader
    data_loader = TaskDataLoader(config)
    
    # Incremental learning loop
    for task_id in range(config.NUM_TASKS):
        # Get data for current task
        train_loader, test_loader = data_loader.get_task_data(task_id)
        
        # Train on current task
        model = train_task(model, task_id, train_loader, test_loader, config, device)
        
        # Store exemplars for replay (if enabled)
        if config.USE_REPLAY and task_id < config.NUM_TASKS - 1:
            # In practice, you would select representative samples
            # For now, just use a placeholder
            print(f"Storing exemplars for Task {task_id}...")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    # Save final model
    save_checkpoint(model, None, config.NUM_EPOCHS, config.NUM_TASKS - 1, config, filename="final_model.pth")


if __name__ == "__main__":
    main()
