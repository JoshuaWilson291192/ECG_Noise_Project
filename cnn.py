import torch 
from torch import nn

class ECG_CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(

        # Convolutional layer 1
            nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=3),
            nn.BacthNorm1d(32),
            nn.ReLU(),

        # Convolutional layer 2
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BacthNorm1d(64),
            nn.ReLU(),

        # Convolutional layer 3 
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BacthNorm1d(128),
            nn.ReLU(),

        # Prediction layer
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.network(x)
