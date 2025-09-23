# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a COVID-19 case prediction project using Deep Neural Networks (DNN) for regression. The project implements a PyTorch-based neural network to predict COVID-19 positive cases from epidemiological data.

## Project Structure

```
COVID_Pred/
├── Data/                    # Training and test datasets
│   ├── covid.train.csv     # Training data (2025 samples, 94 features)
│   └── covid.test.csv      # Test data (675 samples, 93 features)
├── Doc/                    # Documentation and assignment materials
├── models/                 # Saved model checkpoints (.pth files)
├── notebooks/              # Jupyter notebooks with implementation
│   └── covid_prediction.ipynb  # Main implementation notebook
├── img/                    # Generated plots and visualizations
└── training_log.md         # Training progress logs
```

## Development Environment

- **Language**: Python
- **Primary Framework**: PyTorch
- **Dependencies**:
  - torch (neural networks)
  - numpy (numerical operations)
  - matplotlib (plotting)
  - csv (data I/O)

No requirements.txt or environment.yml file exists. Dependencies should be installed manually:
```bash
pip install torch numpy matplotlib
```

## Common Commands

### Running the Notebook
```bash
# Start Jupyter notebook
jupyter notebook notebooks/covid_prediction.ipynb
```

### Key Cells to Execute
The notebook is designed to be run sequentially. Key execution points:
- **Cell 18**: Data loading and preprocessing (creates train/dev/test datasets)
- **Cell 20**: Model training (outputs epoch-by-epoch progress)
- **Cell 25**: Test set prediction and CSV output generation

## Main Architecture

### Data Processing
- **COVID19Dataset class**: Custom PyTorch Dataset that:
  - Reads CSV files and extracts features
  - Splits training data into train/validation (90%/10% split using modulo operation)
  - Normalizes temporal features (columns 40+) using training set statistics
  - Supports both full feature set (93 features) and limited feature selection modes
  - **Critical normalization fix**: Ensures train/dev/test sets use consistent normalization parameters

### Neural Network
- **NeuralNet class**: Multi-layer perceptron with:
  - Architecture: input → 16 → 8 → 4 → 1 (with ReLU activations)
  - MSE loss function with L2 regularization penalty
  - Configurable L2 regularization coefficient (default: 0.01)
  - Forward method returns squeezed output for proper dimensionality

### Training Process
- Early stopping mechanism (30 epochs patience)
- Model checkpointing (saves best model based on validation loss)
- Loss tracking for both training and validation sets
- Adam optimizer with learning rate 0.001 and weight decay 0.001

## Key Configuration Parameters

The current optimal configuration:
- Epochs: 300 (with early stopping)
- Batch size: 32
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 0.001
- L2 regularization: 0.01
- Early stopping patience: 30 epochs
- Feature set: All 93 features (not limited selection)

## Model Evolution (TODO Implementations)

The project has gone through several iterations documented in training_log.md:
1. **TODO1**: Feature selection correction (columns 58, 76 for 0-based indexing)
2. **TODO2**: Enhanced network architecture (deeper network with ReLU)
3. **TODO3**: Added L2 regularization
4. **TODO4**: Optimized training parameters (Adam optimizer)
5. **Final improvement**: Fixed normalization consistency across train/dev/test sets

## Key Functions

- `get_device()`: Detects CUDA availability
- `plot_learning_curve()`: Visualizes training progress with train/validation curves
- `plot_pred()`: Shows prediction vs ground truth scatter plot
- `train()`: Main training loop with early stopping and model checkpointing
- `dev()`: Validation evaluation returning average MSE loss
- `test()`: Generate predictions on test set
- `save_pred()`: Save predictions to CSV format

## Critical Implementation Details

### Normalization Handling
The most recent implementation (model_fixed.pth) correctly handles normalization:
- Training set computes and stores normalization statistics
- Validation and test sets use training set's statistics for consistency
- Only temporal features (columns 40+) are normalized

### Data Split Strategy
- Training: indices where `i % 10 != 0` (90% of data)
- Validation: indices where `i % 10 == 0` (10% of data)

### Feature Selection Modes
- `target_only=False`: Uses all 93 available features (current best approach)
- `target_only=True`: Uses only columns 58 and 76 (tested_positive from previous days)

## Output Files

- Model checkpoints: `models/model_fixed.pth` (latest), `models/model_todo4.pth`
- Predictions: `pred_fixed.csv` (latest output with normalized predictions)
- Training logs: Console output with epoch-by-epoch loss tracking
- Visualizations: Learning curves and prediction scatter plots saved as images

## Performance Metrics

Latest model (with normalization fix) achieves:
- Final validation loss: ~1.09
- Prediction range: 12-20 (realistic for COVID case counts)
- Training converged after 288 epochs with early stopping

## Data Format

- **Input features**: 94 columns total (40 state features + 54 daily features across 3 days)
- **Target**: COVID-19 positive case count (final column in training data)
- **Feature normalization**: Applied to temporal features (columns 40+)
- **Train/test split**: Training CSV has 94 columns, test CSV has 93 columns (no target)