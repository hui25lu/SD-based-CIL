"""
Stable Diffusion-based Class Incremental Learning Model
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class SDCILModel(nn.Module):
    """
    Stable Diffusion-based model for Class Incremental Learning
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        super(SDCILModel, self).__init__()
        self.config = config
        self.current_task = 0
        self.seen_classes = 0
        
        # Initialize base model (placeholder)
        # In practice, you would load a pretrained Stable Diffusion model
        # and adapt it for classification or generation tasks
        print(f"Initializing SD-based CIL model with base: {config.MODEL_NAME}")
        
        # Feature extractor (simplified example)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier head (will be expanded for each task)
        self.classifier = nn.Linear(128, config.NUM_CLASSES_PER_TASK)
        
        # Memory for exemplars (stored as tensors for efficiency)
        self.memory_images = torch.tensor([])
        self.memory_labels = torch.tensor([])
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
        
        Returns:
            Output predictions
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output
    
    def expand_classifier(self, num_new_classes: int):
        """
        Expand classifier to accommodate new classes
        
        Args:
            num_new_classes: Number of new classes to add
        """
        old_weights = self.classifier.weight.data
        old_bias = self.classifier.bias.data
        
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features + num_new_classes
        
        # Create new classifier
        new_classifier = nn.Linear(in_features, out_features)
        
        # Copy old weights (old classes should map to same output dimensions)
        new_classifier.weight.data[:old_weights.size(0), :] = old_weights
        new_classifier.bias.data[:old_bias.size(0)] = old_bias
        
        # Initialize new weights
        nn.init.kaiming_normal_(new_classifier.weight.data[old_weights.size(0):, :])
        nn.init.zeros_(new_classifier.bias.data[old_bias.size(0):])
        
        self.classifier = new_classifier
        self.seen_classes = out_features
        
        print(f"Classifier expanded to {out_features} classes")
    
    def add_memory(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Add samples to memory for rehearsal
        
        Args:
            images: Images to store
            labels: Corresponding labels
        """
        # Store as tensors for efficient access
        if len(self.memory_images) == 0:
            self.memory_images = images.cpu()
            self.memory_labels = labels.cpu()
        else:
            self.memory_images = torch.cat([self.memory_images, images.cpu()], dim=0)
            self.memory_labels = torch.cat([self.memory_labels, labels.cpu()], dim=0)
        
        # Limit memory size
        if len(self.memory_images) > self.config.MEMORY_SIZE:
            self.memory_images = self.memory_images[-self.config.MEMORY_SIZE:]
            self.memory_labels = self.memory_labels[-self.config.MEMORY_SIZE:]
        
        print(f"Memory updated: {len(self.memory_images)} samples stored")
    
    def get_memory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get stored memory samples
        
        Returns:
            memory_images, memory_labels: Stored exemplar samples and labels
        """
        return self.memory_images, self.memory_labels
    
    def prepare_for_task(self, task_id: int):
        """
        Prepare model for a new task
        
        Args:
            task_id: Task identifier
        """
        if task_id > 0:
            # Expand classifier for new classes
            self.expand_classifier(self.config.NUM_CLASSES_PER_TASK)
        
        self.current_task = task_id
        print(f"Model prepared for Task {task_id}")


def create_model(config) -> SDCILModel:
    """
    Factory function to create SD-CIL model
    
    Args:
        config: Configuration object
    
    Returns:
        model: Initialized SDCILModel
    """
    model = SDCILModel(config)
    
    # Move to appropriate device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Model created and moved to {device}")
    return model
