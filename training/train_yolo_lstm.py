"""
Training Script for YOLO + LSTM Violence Detection Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from yolo_lstm_model import SimplifiedYOLOLSTM

# Configuration
SEQUENCES_DIR = Path("../dataset_sequences")
BATCH_SIZE = 2  # Optimized for RTX 3050 6GB
NUM_EPOCHS = 30  # Reduced for faster training
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 2  # Reduced to prevent CPU bottleneck
SAVE_DIR = Path("./models/yolo_lstm")
ACCUMULATE_GRAD = 4  # Gradient accumulation (effective batch = 2*4=8)


class VideoSequenceDataset(Dataset):
    """Dataset for video sequences"""
    def __init__(self, sequences_file):
        with open(sequences_file, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        frames = item['frames']  # (seq_len, H, W, 3)
        label = item['label']
        
        # Convert to torch tensor and permute dimensions
        # From (seq_len, H, W, 3) to (seq_len, 3, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return frames, label


def train_epoch(model, dataloader, criterion, optimizer, device, accumulate_grad=1):
    """Train for one epoch with gradient accumulation"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training")
    for i, (frames, labels) in enumerate(pbar):
        frames = frames.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        # Normalize loss for gradient accumulation
        loss = loss / accumulate_grad
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulate_grad steps
        if (i + 1) % accumulate_grad == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        # Statistics
        running_loss += loss.item() * accumulate_grad
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item() * accumulate_grad:.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for frames, labels in pbar:
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return epoch_loss, accuracy, precision, recall, f1, all_preds, all_labels


def plot_confusion_matrix(labels, preds, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NonViolence', 'Violence'],
                yticklabels=['NonViolence', 'Violence'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_yolo_lstm():
    """Main training function"""
    print("=" * 60)
    print("Training YOLO + LSTM for Violence Detection")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ No GPU available, training on CPU")
    
    # Create save directory
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("\n📁 Loading datasets...")
    train_dataset = VideoSequenceDataset(SEQUENCES_DIR / "train_sequences.pkl")
    val_dataset = VideoSequenceDataset(SEQUENCES_DIR / "val_sequences.pkl")
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model (optimized for RTX 3050 6GB)
    print("\n🔧 Initializing model...")
    model = SimplifiedYOLOLSTM(
        num_classes=2,
        hidden_size=128,  # Reduced for faster training
        num_layers=1,  # Single layer for speed
        dropout=0.3
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...")
    print("-" * 60)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        
        # Train with gradient accumulation
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, ACCUMULATE_GRAD)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, SAVE_DIR / 'best_model.pth')
            print(f"✅ Best model saved! (Val Acc: {val_acc:.4f})")
            
            # Save confusion matrix for best model
            plot_confusion_matrix(val_labels, val_preds, SAVE_DIR / 'confusion_matrix_best.png')
    
    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, SAVE_DIR / 'final_model.pth')
    
    # Save training history
    with open(SAVE_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'training_curves.png')
    plt.close()
    
    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print("=" * 60)
    print(f"\n📁 Models saved in: {SAVE_DIR.absolute()}")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_yolo_lstm()
