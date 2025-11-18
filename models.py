# models.py
import torch.nn as nn
import torch

class DifficultyEstimator(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.temperature = 0.5  # Helps diversify output

    def forward(self, x):
        logits = self.linear(x) / self.temperature
        return logits

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
    
class ImprovedClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),  # increased capacity
            nn.ReLU(),
            nn.BatchNorm1d(256),        # stabilizes training
            nn.Dropout(dropout_rate),   # regularization

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)