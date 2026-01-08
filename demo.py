"""
Demo script for SD-based Class Incremental Learning
Runs a quick demonstration with minimal epochs
"""

import torch

from config import Config, get_config
from models.sd_cil_model import create_model
from data.dataloader import TaskDataLoader
from utils.helpers import set_seed, create_directories


class DemoConfig(Config):
    """Demo configuration with reduced settings for quick testing"""
    NUM_TASKS = 2
    NUM_EPOCHS = 2
    BATCH_SIZE = 4
    NUM_CLASSES_PER_TASK = 5
    MEMORY_SIZE = 100


def demo():
    """
    Run a quick demo of the SD-CIL system
    """
    print("\n" + "="*60)
    print("SD-based Class Incremental Learning - Demo")
    print("="*60 + "\n")
    
    # Use demo configuration
    config = DemoConfig()
    
    # Set random seed
    set_seed(42)
    
    # Create directories
    create_directories(config)
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running demo with {config.NUM_TASKS} tasks, {config.NUM_EPOCHS} epochs each\n")
    
    # Create model
    model = create_model(config)
    
    # Initialize data loader
    data_loader = TaskDataLoader(config)
    
    # Quick demonstration of incremental learning
    for task_id in range(config.NUM_TASKS):
        print(f"\n{'='*60}")
        print(f"Demo Task {task_id}")
        print(f"{'='*60}\n")
        
        # Get data
        train_loader, test_loader = data_loader.get_task_data(task_id)
        
        print(f"Training samples: ~{len(train_loader.dataset)}")
        print(f"Test samples: ~{len(test_loader.dataset)}")
        
        # Prepare model for task
        model.prepare_for_task(task_id)
        
        # Show model info
        print(f"Model classes: {model.seen_classes}")
        print(f"Current task: {model.current_task}")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nTo run full training, use: python train.py")
    print("To evaluate a trained model, use: python evaluate.py")


if __name__ == "__main__":
    demo()
