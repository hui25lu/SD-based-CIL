"""
Evaluation script for SD-based Class Incremental Learning
"""

import torch
from tqdm import tqdm

from config import get_config
from models.sd_cil_model import create_model
from data.dataloader import TaskDataLoader
from utils.helpers import set_seed, load_checkpoint


def evaluate_all_tasks(model, data_loader, config, device):
    """
    Evaluate model on all seen tasks
    
    Args:
        model: Trained model
        data_loader: Task data loader
        config: Configuration object
        device: Device to evaluate on
    
    Returns:
        task_accuracies: Dictionary of accuracies per task
    """
    model.eval()
    task_accuracies = {}
    
    print("\n" + "="*50)
    print("Evaluating on all tasks")
    print("="*50 + "\n")
    
    for task_id in range(config.NUM_TASKS):
        _, test_loader = data_loader.get_task_data(task_id)
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Task {task_id}"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = 100 * total_correct / total_samples
        task_accuracies[task_id] = accuracy
        
        print(f"Task {task_id} Accuracy: {accuracy:.2f}%")
    
    # Calculate average accuracy
    avg_accuracy = sum(task_accuracies.values()) / len(task_accuracies)
    print(f"\nAverage Accuracy: {avg_accuracy:.2f}%")
    
    return task_accuracies


def main():
    """
    Main evaluation function
    """
    # Load configuration
    config = get_config()
    
    # Set random seed
    set_seed(42)
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create model
    model = create_model(config)
    
    # Load trained model
    checkpoint_path = f"{config.CHECKPOINT_DIR}/final_model.pth"
    try:
        epoch, task_id = load_checkpoint(model, None, checkpoint_path, device=str(device))
        print(f"Loaded model from epoch {epoch}, task {task_id}\n")
    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    # Initialize data loader
    data_loader = TaskDataLoader(config)
    
    # Evaluate on all tasks
    task_accuracies = evaluate_all_tasks(model, data_loader, config, device)


if __name__ == "__main__":
    main()
