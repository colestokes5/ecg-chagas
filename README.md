# ECG Chagas Disease Classification

This project implements and compares different deep learning models for ECG-based Chagas disease classification. The models include CNNLSTM, ViT (Vision Transformer), and HeartGPT architectures.

## Project Structure

```
.
├── data/                  # Data directory
├── models/               # Model implementations and training results
│   ├── CNNLSTM/         # CNNLSTM model results
│   │   └── spec/       # Spectrogram-based CNNLSTM results
│   ├── ViT/            # Vision Transformer results
│   │   └── spec/       # Spectrogram-based ViT results
│   └── HeartGPT/       # HeartGPT model results
│       └── token/      # Token-based HeartGPT results
├── prepare_all_data.py  # Script to prepare all datasets
├── prepare_code15_data.py    # Script for CODE15 dataset preparation
├── prepare_ptbxl_data.py     # Script for PTBXL dataset preparation
├── prepare_samitrop_data.py  # Script for SAMITROP dataset preparation
├── prepare_train_and_val_data.py  # Script for train/val split
├── train_model.py       # Model training script
├── visualize_results.py # Visualization script
├── ecg_datasets.py      # Dataset handling classes
├── helper_code.py       # Utility functions
└── models.py           # Model architecture definitions
```

## Model Comparison Visualization

The `visualize_results.py` script generates interactive visualizations comparing the performance of different models across various metrics:

1. **Training Loss Comparison**: Shows how the training loss evolves over epochs for each model
2. **Validation Accuracy Comparison**: Displays the validation accuracy progression during training
3. **F1 Score Comparison**: Illustrates the F1 score trends across training epochs
4. **Precision-Recall Trade-off**: Shows the relationship between precision and recall for each model
5. **ROC AUC Comparison**: Displays the ROC AUC score progression during training
6. **Best Performance Metrics**: Bar chart comparing the best achieved metrics across models

### How to Generate Visualizations

1. Ensure all required metrics CSV files are present in their respective model directories
2. Run the visualization script:
   ```bash
   python visualize_results.py
   ```
3. Open the generated `model_comparison.html` file in a web browser to view the interactive plots

### Interactive Features

- Hover over data points to see detailed metrics
- Click on legend items to show/hide specific models
- Use the toolbar to zoom, pan, and reset the view
- Export plots as PNG files using the save tool

## Model Training

To train the models:

1. Prepare the datasets using the preparation scripts
2. Run the training script:
   ```bash
   python train_model.py
   ```

## Dependencies

- Python 3.x
- PyTorch
- Bokeh (for visualization)
- Pandas
- NumPy

## Data Preparation

The project includes scripts for preparing different ECG datasets:
- CODE15
- PTBXL
- SAMITROP

Each dataset preparation script handles:
- Data loading
- Preprocessing
- Feature extraction
- Train/validation split

## Model Architectures

### CNNLSTM
- Combines Convolutional Neural Networks with LSTM layers
- Processes spectrogram representations of ECG signals

### Vision Transformer (ViT)
- Applies transformer architecture to ECG spectrograms
- Uses self-attention mechanisms for feature extraction

### HeartGPT
- Transformer-based architecture specifically designed for ECG analysis
- Processes tokenized representations of ECG signals
