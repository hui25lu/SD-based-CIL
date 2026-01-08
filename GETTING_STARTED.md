# Getting Started Guide

This guide will help you get started with the SD-based Class Incremental Learning framework.

## Step 1: Installation

First, set up your environment:

```bash
# Clone the repository
git clone https://github.com/hui25lu/SD-based-CIL.git
cd SD-based-CIL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Run the Demo

Test that everything is working:

```bash
python demo.py
```

This will run a quick demonstration with minimal settings (2 tasks, 2 epochs each).

## Step 3: Customize Configuration

Edit `config.py` to match your requirements:

```python
class Config:
    # Adjust these based on your needs
    NUM_CLASSES_PER_TASK = 10  # How many new classes per task
    NUM_TASKS = 5               # Total number of incremental tasks
    NUM_EPOCHS = 10             # Training epochs per task
    BATCH_SIZE = 8              # Batch size for training
    LEARNING_RATE = 1e-4        # Learning rate
```

## Step 4: Prepare Your Dataset

Modify the `_load_placeholder_data` method in `data/dataloader.py` to load your actual dataset:

```python
def _load_placeholder_data(self, start_class, end_class, train=True):
    # Replace with your dataset loading logic
    # Example for CIFAR-100:
    from torchvision.datasets import CIFAR100
    
    dataset = CIFAR100(
        root=self.config.DATA_ROOT,
        train=train,
        download=True
    )
    
    # Filter by class range
    indices = [i for i, (_, label) in enumerate(dataset) 
               if start_class <= label < end_class]
    
    data = [dataset[i][0] for i in indices]
    labels = [dataset[i][1] for i in indices]
    
    return data, labels
```

## Step 5: Train Your Model

Run the training script:

```bash
python train.py
```

This will:
- Train on each task sequentially
- Save checkpoints to `checkpoints/` directory
- Create training logs in `logs/` directory
- Display training progress and accuracy

## Step 6: Evaluate Results

After training, evaluate on all tasks:

```bash
python evaluate.py
```

This will load the trained model and report accuracy on each task plus the average accuracy across all tasks.

## Understanding the Output

### During Training

```
Training Task 0
Classes 0 to 9
Epoch 1/10
Train Loss: 2.3045, Train Acc: 15.32%
Test Acc: 18.45%
...
```

### During Evaluation

```
Evaluating on all tasks
Task 0 Accuracy: 85.32%
Task 1 Accuracy: 78.21%
...
Average Accuracy: 81.76%
```

## Common Configurations

### Small-scale Experiment (Quick Testing)

```python
NUM_TASKS = 2
NUM_EPOCHS = 5
BATCH_SIZE = 4
NUM_CLASSES_PER_TASK = 5
```

### Medium-scale Experiment

```python
NUM_TASKS = 5
NUM_EPOCHS = 10
BATCH_SIZE = 8
NUM_CLASSES_PER_TASK = 10
```

### Large-scale Experiment

```python
NUM_TASKS = 10
NUM_EPOCHS = 20
BATCH_SIZE = 16
NUM_CLASSES_PER_TASK = 10
```

## Troubleshooting

### Out of Memory Error

Reduce batch size in `config.py`:
```python
BATCH_SIZE = 4  # or smaller
```

### CUDA Not Available

The code will automatically fall back to CPU. To force CPU:
```python
DEVICE = "cpu"
```

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

1. **Experiment with different architectures**: Modify `models/sd_cil_model.py`
2. **Try different datasets**: Update `data/dataloader.py`
3. **Tune hyperparameters**: Adjust values in `config.py`
4. **Add new features**: Extend the base classes
5. **Compare methods**: Implement different CIL strategies

## Need Help?

- Check the main `README.md` for detailed documentation
- Review the code comments for implementation details
- Open an issue on GitHub for bugs or questions

Happy learning! 🚀
