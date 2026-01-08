"""
Configuration settings for SD-based Class Incremental Learning
"""

class Config:
    """Base configuration class"""
    
    # Model settings
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    NUM_CLASSES_PER_TASK = 10
    NUM_TASKS = 5
    
    # Training settings
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    
    # Data settings
    IMAGE_SIZE = 512
    NUM_WORKERS = 4
    
    # Paths
    DATA_ROOT = "./data"
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    
    # Device
    DEVICE = "cuda"
    
    # Incremental learning settings
    MEMORY_SIZE = 2000  # Number of exemplars to store
    USE_REPLAY = True
    
    # Evaluation
    EVAL_FREQUENCY = 1  # Evaluate every N epochs
    SAVE_FREQUENCY = 5  # Save checkpoint every N epochs


def get_config():
    """Returns the default configuration"""
    return Config()
