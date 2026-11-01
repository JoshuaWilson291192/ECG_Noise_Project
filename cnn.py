import torch 
from torch import nn

class ECG_CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(

        # Convolutional layer 1
            nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

        # Convolutional layer 2
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

        # Convolutional layer 3 
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

        # Prediction layer
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.network(x)

#
if __name__ == "__main__":
    model = ECG_CNN()

    example_input = torch.zeros((4, 12, 5000))  # batch size of 4, 12 leads, 5000 samples

    model.eval()
    with torch.no_grad():
        output = model(example_input)

    print("input shape:", tuple(example_input.shape))
    print("output shape:", tuple(output.shape))
