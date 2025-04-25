import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from helper_code import load_label

def create_confusion_matrix_plot(cm, classes=['Normal', 'Chagas']):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def train_model(model_name, data_type, num_epochs, learning_rate, use_wandb=False):
    # Create model directory structure
    base_dir = Path(f"models/{model_name}/{data_type}")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project="ecg-chagas",
            config={
                "model": model_name,
                "data_type": data_type,
                "epochs": num_epochs,
                "learning_rate": learning_rate
            }
        )

        # Initialize model based on model_name
        if model_name == "CNNLSTM":
            if data_type == 'raw':
                from models import CNNLSTMRaw
                model = CNNLSTMRaw()
            elif data_type == 'spec':
                from models import CNNLSTMSpec
                model = CNNLSTMSpec()
            criterion = nn.CrossEntropyLoss()
        elif model_name == "ViT":
            if data_type == 'raw':
                raise ValueError(f"Cannot use raw ECG with ViT")
            from models import ViT
            model = ViT(input_channels=12, num_classes=2)
            criterion = nn.CrossEntropyLoss()
        elif model_name == "HeartGPT":
            if data_type != 'token':
                raise ValueError(f"Can not use data type ({data_type}) with HeartGPT")
            from models import HeartGPT
            model = HeartGPT(num_classes=2, vocab_size=101)
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    # Load data
    train_paths = sorted(glob('data/train_data/*.dat'))
    train_files = ['data/train_data/' + os.path.splitext(os.path.basename(p))[0] for p in train_paths]
    train_labels = [load_label(f) for f in train_files]

    val_paths = sorted(glob('data/val_data/*.dat'))
    val_files = ['data/val_data/' + os.path.splitext(os.path.basename(p))[0] for p in val_paths]
    val_labels = [load_label(f) for f in val_files]

    seq_len = 2500

    # Use correct dataset
    if data_type == 'token':
        from ecg_datasets import TokenizedECGDataset as ECGDataset
        resize = False
        seq_len = 500
    elif data_type == 'spec':
        from ecg_datasets import SpectrogramECGDataset as ECGDataset
        resize = (model_name == "ViT")
    elif data_type == 'raw':
        from ecg_datasets import RawECGDataset as ECGDataset
        resize = False
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    train_dataset = ECGDataset(train_files, train_labels, seq_len=seq_len, augment=True, resize=resize)
    val_dataset = ECGDataset(val_files, val_labels, seq_len=seq_len, resize=resize)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Initialize metrics tracking
    metrics_df = pd.DataFrame(columns=[
        'epoch', 'train_loss', 'val_accuracy', 'val_f1', 'val_precision', 
        'val_recall', 'val_auc_roc', 'true_positives', 'false_positives',
        'true_negatives', 'false_negatives'
    ])
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_data, batch_labels in progress_bar:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                outputs = model(batch_data)
                
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    preds = torch.sigmoid(outputs) > 0.5
                else:
                    preds = outputs.argmax(dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        accuracy = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds)
        recall = recall_score(val_labels, val_preds)
        auc_roc = roc_auc_score(val_labels, val_preds)
        
        cm = confusion_matrix(val_labels, val_preds)
        tn, fp, fn, tp = cm.ravel()
        
        # Update metrics DataFrame
        metrics_df.loc[epoch] = [
            epoch + 1,
            np.mean(train_losses),
            accuracy,
            f1,
            precision,
            recall,
            auc_roc,
            tp,
            fp,
            tn,
            fn
        ]
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                'train_loss': np.mean(train_losses),
                'val_accuracy': accuracy,
                'val_f1': f1,
                'val_precision': precision,
                'val_recall': recall,
                'val_auc_roc': auc_roc,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            })
        
        # Update learning rate scheduler
        scheduler.step(np.mean(train_losses))
    
    # Save confusion matrix plot to wandb
    if use_wandb:
        cm_plot = create_confusion_matrix_plot(cm)
        wandb.log({"confusion_matrix": wandb.Image(cm_plot)})
        plt.close()
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = base_dir / f"{model_name}_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    
    # Save metrics
    metrics_save_path = base_dir / f"{model_name}_{timestamp}_metrics.csv"
    metrics_df.to_csv(metrics_save_path, index=False)
    
    if use_wandb:
        wandb.finish()
    
    return model, metrics_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ECG classification model')
    parser.add_argument('--m', type=str, choices=['CNNLSTM', 'ViT', 'HeartGPT'], required=True)
    parser.add_argument('--dt', type=str, choices=['token', 'spec', 'raw'], required=True)
    parser.add_argument('--e', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    
    args = parser.parse_args()
    
    model, metrics = train_model(
        model_name=args.m,
        data_type=args.dt,
        num_epochs=args.e,
        learning_rate=args.lr,
        use_wandb=args.wandb
    ) 