import torch
from torch import nn
from torch.nn import functional as F


Secs = int


class UNetConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self._model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._model(X)


class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self._model = nn.Sequential(
            nn.MaxPool1d(2),
            UNetConv(in_channels, out_channels),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._model(X)


class UNetUp(nn.Module):
    def __init__(
            self,
            in_channels: int,
            in_channels_skip: int,
            out_channels: int,
    ) -> None:
        super(UNetUp, self).__init__()
        self._up = nn.ConvTranspose1d(
            in_channels,
            in_channels,
            kernel_size=8,
            stride=2,
            padding=3,
        )
        self._model = UNetConv(in_channels + in_channels_skip, out_channels)

    def forward(self, X_skip: torch.Tensor, X: torch.Tensor) -> None:
        X = self._up(X)
        diff = X_skip.size()[2] - X.size()[2]
        X = F.pad(X, (diff // 2, diff - diff // 2))

        return self._model(torch.cat([X_skip, X], dim=1))


class UNet1d(nn.Module):
    def __init__(self, channel_multiplier: int = 4) -> None:
        super().__init__()

        self._input = UNetConv(1, channel_multiplier)
        self._down1 = UNetDown(channel_multiplier, 2*channel_multiplier)
        self._down2 = UNetDown(2*channel_multiplier, 4*channel_multiplier)
        self._down3 = UNetDown(4*channel_multiplier, 8*channel_multiplier)
        self._down4 = UNetDown(8*channel_multiplier, 16*channel_multiplier)
        self._up1 = UNetUp(
            16*channel_multiplier, 8*channel_multiplier, 8*channel_multiplier,
        )
        self._up2 = UNetUp(
            8*channel_multiplier, 4*channel_multiplier, 4*channel_multiplier,
        )
        self._up3 = UNetUp(
            4*channel_multiplier, 2*channel_multiplier, 2*channel_multiplier,
        )
        self._up4 = UNetUp(2*channel_multiplier, channel_multiplier, channel_multiplier)
        self._output = nn.Conv1d(channel_multiplier, 1, kernel_size=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x1 = self._input(X)
        x2 = self._down1(x1)
        x3 = self._down2(x2)
        x4 = self._down3(x3)
        x = self._down4(x4)
        x = self._up1(x4, x)
        x = self._up2(x3, x)
        x = self._up3(x2, x)
        x = self._up4(x1, x)

        return self._output(x)


if __name__ == '__main__':
    from torchinfo import summary

    print(summary(UNet1d(), input_size=(1, 1, 5000)))
