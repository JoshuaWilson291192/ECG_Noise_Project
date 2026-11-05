import torch 
from torch import nn

class ECG_CNN(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
        # Conv layer 1
            nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        # Conv layer 2
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        # Conv layer 3 
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


if __name__ == "__main__":
    from data_loader import create_dataloaders 

    train_loader, val_loader, test_loader = create_dataloaders(batch_size=4)

    signal, labels, ids = next(iter(train_loader))

    model = ECG_CNN()

    model.eval()

    with torch.no_grad():
        output = model(signal)
        probabilities = torch.sigmoid(output)

    print("Input shape:", signal.shape)
    print("Output shape:", output.shape)
    print("First ECG ID:", ids[0].item())
    print("Labels:", labels[0].tolist())
    print("Predicted probabilities:", probabilities[0].tolist())
