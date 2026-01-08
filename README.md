# SD-based-CIL

Stable Diffusion-based Class Incremental Learning (CIL) framework for continual learning scenarios where new classes are introduced over time.

## Overview

This project implements a class incremental learning system that leverages Stable Diffusion models to handle the challenge of learning new classes while maintaining performance on previously learned classes. The framework addresses catastrophic forgetting through memory replay and classifier expansion techniques.

## Features

- 🔄 **Incremental Learning**: Support for sequential task learning with new classes
- 🧠 **Memory Replay**: Exemplar-based rehearsal to prevent catastrophic forgetting
- 📊 **Flexible Architecture**: Expandable classifier for accommodating new classes
- ⚙️ **Configurable**: Easy-to-modify configuration for different experimental setups
- 🎯 **Evaluation Tools**: Comprehensive evaluation across all learned tasks

## Project Structure

```
SD-based-CIL/
├── config.py              # Configuration settings
├── train.py              # Main training script
├── evaluate.py           # Evaluation script
├── demo.py               # Quick demonstration script
├── requirements.txt      # Python dependencies
├── models/
│   └── sd_cil_model.py  # SD-based CIL model implementation
├── data/
│   └── dataloader.py    # Data loading utilities
├── utils/
│   └── helpers.py       # Helper functions
├── checkpoints/         # Model checkpoints (created during training)
├── logs/                # Training logs (created during training)
└── data/                # Dataset directory (created during training)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- PyTorch 2.0 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hui25lu/SD-based-CIL.git
cd SD-based-CIL
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run Demo

To quickly test the framework with minimal settings:

```bash
python demo.py
```

### Training

To train the model on incremental tasks:

```bash
python train.py
```

The training script will:
- Load configuration from `config.py`
- Train on multiple sequential tasks
- Save checkpoints to `checkpoints/` directory
- Log training progress

### Evaluation

To evaluate a trained model on all tasks:

```bash
python evaluate.py
```

This will load the trained model and report accuracy on each task and the average accuracy.

## Configuration

Edit `config.py` to customize the training setup:

```python
class Config:
    # Model settings
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    NUM_CLASSES_PER_TASK = 10  # Classes per incremental task
    NUM_TASKS = 5               # Total number of tasks
    
    # Training settings
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    
    # Incremental learning settings
    MEMORY_SIZE = 2000          # Exemplar memory size
    USE_REPLAY = True           # Enable memory replay
```

## Usage Examples

### Custom Training Configuration

```python
from config import Config

# Create custom configuration
config = Config()
config.NUM_TASKS = 3
config.NUM_EPOCHS = 20
config.BATCH_SIZE = 16

# Use in training
# ... training code ...
```

### Loading and Using Trained Model

```python
from models.sd_cil_model import create_model
from utils.helpers import load_checkpoint
from config import get_config

config = get_config()
model = create_model(config)
load_checkpoint(model, None, "checkpoints/final_model.pth")

# Use model for inference
# ... inference code ...
```

## Key Components

### SDCILModel

The main model class that handles:
- Feature extraction using convolutional layers
- Dynamic classifier expansion for new classes
- Memory management for exemplar samples

### TaskDataLoader

Manages data loading for incremental learning:
- Splits data into sequential tasks
- Handles class-specific data filtering
- Provides memory replay functionality

### Training Pipeline

1. **Task Preparation**: Model prepares for new task by expanding classifier
2. **Training**: Model trains on current task data
3. **Memory Update**: Representative samples stored for replay
4. **Evaluation**: Model tested on all seen tasks
5. **Checkpoint Saving**: Model state saved for later use

## Extending the Framework

### Adding Custom Datasets

Modify `data/dataloader.py` to load your dataset:

```python
def _load_placeholder_data(self, start_class, end_class, train=True):
    # Replace with your dataset loading logic
    # Example: Load from CIFAR-100, ImageNet, etc.
    dataset = YourDataset(classes=range(start_class, end_class))
    return dataset.data, dataset.labels
```

### Implementing Custom Models

Extend the `SDCILModel` class in `models/sd_cil_model.py`:

```python
class CustomSDCILModel(SDCILModel):
    def __init__(self, config):
        super().__init__(config)
        # Add custom components
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sd_based_cil,
  title={SD-based Class Incremental Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/hui25lu/SD-based-CIL}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Stable Diffusion models from Hugging Face
- PyTorch team for the deep learning framework
- Class Incremental Learning research community

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.