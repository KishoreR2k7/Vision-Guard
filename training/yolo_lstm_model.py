"""
YOLO + LSTM Model for Violence Detection
This model combines YOLO features with LSTM for temporal video analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np


class YOLOFeatureExtractor(nn.Module):
    """Extract features from YOLO backbone"""
    def __init__(self, yolo_model_path, freeze_backbone=True):
        super(YOLOFeatureExtractor, self).__init__()
        
        # Load pretrained YOLO model
        self.yolo = YOLO(yolo_model_path)
        self.model = self.yolo.model
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get the backbone (feature extractor)
        # YOLO11 backbone typically has multiple stages
        self.backbone = self.model.model[:10]  # First 10 layers for feature extraction
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, 3, H, W)
        Returns:
            features: Feature tensor
        """
        features = self.backbone(x)
        return features


class YOLOLSTMModel(nn.Module):
    """YOLO + LSTM for video violence detection"""
    def __init__(self, yolo_model_path, hidden_size=512, num_layers=2, num_classes=2, dropout=0.5):
        super(YOLOLSTMModel, self).__init__()
        
        # YOLO feature extractor
        self.feature_extractor = YOLOFeatureExtractor(yolo_model_path, freeze_backbone=True)
        
        # Adaptive pooling to get fixed size features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # We need to determine feature dimension
        # Typically YOLO11n has 512 channels at deeper layers
        self.feature_dim = 512
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, sequence_length, 3, H, W)
        Returns:
            output: Class logits of shape (batch, num_classes)
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Extract features for each frame
        frame_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # (batch, 3, H, W)
            
            with torch.no_grad():
                features = self.feature_extractor(frame)  # (batch, channels, h, w)
            
            # Global average pooling
            features = self.adaptive_pool(features)  # (batch, channels, 1, 1)
            features = features.view(batch_size, -1)  # (batch, channels)
            frame_features.append(features)
        
        # Stack features
        sequence_features = torch.stack(frame_features, dim=1)  # (batch, seq_len, feature_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(sequence_features)  # (batch, seq_len, hidden_size*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
        
        # Classification
        output = self.classifier(context)  # (batch, num_classes)
        
        return output


class SimplifiedYOLOLSTM(nn.Module):
    """Simplified YOLO + LSTM using pretrained CNN instead of full YOLO"""
    def __init__(self, num_classes=2, hidden_size=128, num_layers=1, dropout=0.3):
        super(SimplifiedYOLOLSTM, self).__init__()
        
        # Use ResNet18 as feature extractor (much lighter and faster than ResNet50)
        from torchvision import models
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final FC layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_dim = 512  # ResNet18 output dimension (vs 2048 for ResNet50)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, sequence_length, 3, H, W)
        Returns:
            output: Class logits of shape (batch, num_classes)
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Extract features for each frame
        frame_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # (batch, 3, H, W)
            
            with torch.no_grad():
                features = self.feature_extractor(frame)  # (batch, 2048, 1, 1)
            
            features = features.view(batch_size, -1)  # (batch, 2048)
            frame_features.append(features)
        
        # Stack features
        sequence_features = torch.stack(frame_features, dim=1)  # (batch, seq_len, 512)
        
        # LSTM
        lstm_out, _ = self.lstm(sequence_features)  # (batch, seq_len, hidden_size*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
        
        # Classification
        output = self.classifier(context)  # (batch, num_classes)
        
        return output


if __name__ == "__main__":
    # Test the model
    print("Testing SimplifiedYOLOLSTM model...")
    
    model = SimplifiedYOLOLSTM(num_classes=2, hidden_size=256, num_layers=2)
    
    # Create dummy input
    batch_size = 2
    seq_len = 16
    dummy_input = torch.randn(batch_size, seq_len, 3, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✅ Model test successful!")
