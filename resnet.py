
import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main path: two convolutional layers
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=5, stride=stride,
                      padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Conv1d(out_channels, out_channels,
                      kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        # Shortcut path
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.layers(x) + self.shortcut(x))


class ECG_ResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # Initial convolution
            nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Three residual blocks
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),

            # Prediction layer
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = ECG_ResNet()
    model.eval()

    example_input = torch.zeros(4, 12, 5000)

    with torch.no_grad():
        output = model(example_input)

    assert output.shape == (4, 5)

    print("Input shape:", tuple(example_input.shape))
    print("Output shape:", tuple(output.shape))
    print("Trainable parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("ResNet test: PASSED")
