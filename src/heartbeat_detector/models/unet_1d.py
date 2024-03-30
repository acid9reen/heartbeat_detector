import torch
from torch import nn
from torch.nn import functional as F


class UNetConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 9,
            padding: int = 4,
            stride: int = 1,
    ) -> None:
        super().__init__()
        self._model = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                padding_mode='reflect',
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                padding_mode='reflect',
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._model(X)


class UNetDown(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 9,
            padding: int = 4,
            stride: int = 1,
    ) -> None:
        super().__init__()
        self._model = nn.Sequential(
            nn.MaxPool1d(2),
            UNetConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._model(X)


class UNetUp(nn.Module):
    def __init__(
            self,
            in_channels: int,
            in_channels_skip: int,
            out_channels: int,
            kernel_size: int = 8,
            padding: int = 3,
            stride: int = 2,
    ) -> None:
        super(UNetUp, self).__init__()
        self._up = nn.ConvTranspose1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self._model = UNetConv(in_channels + in_channels_skip, out_channels)

    def forward(self, X_skip: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        X = self._up(X)
        diff = X_skip.size()[2] - X.size()[2]
        X = F.pad(X, (diff // 2, diff - diff // 2))

        return self._model(torch.cat([X_skip, X], dim=1))


class UNet1d(nn.Module):
    def __init__(self, channel_multiplier: int = 4) -> None:
        super().__init__()

        self._input = UNetConv(1, channel_multiplier)
        self._down1 = UNetDown(channel_multiplier, 2 * channel_multiplier, 17, 8)
        self._down2 = UNetDown(2 * channel_multiplier, 4 * channel_multiplier, 17, 8)
        self._down3 = UNetDown(4 * channel_multiplier, 8 * channel_multiplier, 29, 14)
        self._down4 = UNetDown(8 * channel_multiplier, 16 * channel_multiplier, 29, 14)
        self._down5 = UNetDown(16 * channel_multiplier, 32 * channel_multiplier, 29, 14)
        self._down6 = UNetDown(32 * channel_multiplier, 64 * channel_multiplier, 41, 20)

        self._up1 = UNetUp(64 * channel_multiplier, 32 * channel_multiplier, 32 * channel_multiplier)
        self._up2 = UNetUp(32 * channel_multiplier, 16 * channel_multiplier, 16 * channel_multiplier, 14, 6)
        self._up3 = UNetUp(16 * channel_multiplier, 8 * channel_multiplier, 8 * channel_multiplier, 14, 6)
        self._up4 = UNetUp(8 * channel_multiplier, 4 * channel_multiplier, 4 * channel_multiplier, 22, 10)
        self._up5 = UNetUp(4 * channel_multiplier, 2 * channel_multiplier, 2 * channel_multiplier, 22, 10)
        self._up6 = UNetUp(2 * channel_multiplier, channel_multiplier, channel_multiplier, 30, 14)

        self._output = nn.Sequential(
            nn.Conv1d(channel_multiplier, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x1 = self._input(X)
        x2 = self._down1(x1)
        x3 = self._down2(x2)
        x4 = self._down3(x3)
        x5 = self._down4(x4)
        x6 = self._down5(x5)
        x = self._down6(x6)

        x = self._up1(x6, x)
        x = self._up2(x5, x)
        x = self._up3(x4, x)
        x = self._up4(x3, x)
        x = self._up5(x2, x)
        x = self._up6(x1, x)

        return self._output(x)


if __name__ == '__main__':
    from torchinfo import summary

    print(summary(UNet1d(), input_size=(1, 1, 10_000)))
