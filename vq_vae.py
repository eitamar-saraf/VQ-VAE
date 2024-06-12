import torch
from torch import nn
import numpy as np


class ResidualLayer(nn.Module):

    def __init__(self, input_channels: int = 128, res_output_channels: int = 32):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_channels, out_channels=res_output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=res_output_channels, out_channels=input_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    def __init__(self, input_channels: int = 128, n_res_layers: int = 2, res_output_channels: int = 32):
        super(ResidualStack, self).__init__()
        self.number_of_residual_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(input_channels=input_channels,
                           res_output_channels=res_output_channels)] * self.number_of_residual_layers)
        self.last_activation = nn.ReLU()

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = self.last_activation(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_channels: int = 3, output_channels: int = 128, n_res_layers: int = 2,
                 res_output_channels: int = 32):
        super(Encoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels // 2, kernel_size=4, stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels // 2, out_channels=output_channels, kernel_size=4, stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
            ResidualStack(input_channels=output_channels, n_res_layers=n_res_layers,
                          res_output_channels=res_output_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        self.encoder = Encoder()

    def forward(self, x):
        pass


def test():
    x = np.random.random_sample((64, 3, 128, 128))
    x = torch.tensor(x).float().to('cuda')
    encoder = Encoder(input_channels=3, output_channels=128, n_res_layers=2, res_output_channels=32)
    encoder = encoder.to('cuda')
    encoder_out = encoder(x)


if __name__ == "__main__":
    test()
