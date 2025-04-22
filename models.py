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
    
class DifficultyRegressor(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # optional
        )
    def forward(self, x):
        return self.model(x)