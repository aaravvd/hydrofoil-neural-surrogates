from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int, activation: str = "gelu"):
        super().__init__()
        act: type[nn.Module] = nn.GELU if activation == "gelu" else nn.Tanh
        layers: list[nn.Module] = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), act()])
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointDNN(MLP):
    pass


class PINN(MLP):
    """Point model trained with supervised loss plus steady incompressible residuals."""


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, nx, ny = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batch, self.out_channels, nx, ny // 2 + 1, dtype=torch.cfloat, device=x.device)
        mx = min(self.modes_x, nx)
        my = min(self.modes_y, ny // 2 + 1)
        out_ft[:, :, :mx, :my] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :mx, :my], self.weights[:, :, :mx, :my])
        return torch.fft.irfft2(out_ft, s=(nx, ny))


class FNO2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int = 48, depth: int = 4, modes_x: int = 16, modes_y: int = 16):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes_x, modes_y) for _ in range(depth)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for spec, local in zip(self.spectral, self.local):
            h = F.gelu(spec(h) + local(h))
        return self.project(h)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4 if out_channels >= 4 else 1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4 if out_channels >= 4 else 1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int = 32, depth: int = 3):
        super().__init__()
        widths = [width * 2**i for i in range(depth)]
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for channels in widths:
            self.down_blocks.append(ConvBlock(prev, channels))
            self.pools.append(nn.MaxPool2d(2))
            prev = channels
        self.bottleneck = ConvBlock(prev, prev * 2)
        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        prev = prev * 2
        for channels in reversed(widths):
            self.up_transpose.append(nn.ConvTranspose2d(prev, channels, 2, stride=2))
            self.up_blocks.append(ConvBlock(channels * 2, channels))
            prev = channels
        self.head = nn.Conv2d(prev, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for block, pool in zip(self.down_blocks, self.pools):
            h = block(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)
        for up, block, skip in zip(self.up_transpose, self.up_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = block(torch.cat([h, skip], dim=1))
        return self.head(h)


class DeepONet(nn.Module):
    def __init__(self, param_dim: int, coord_dim: int, out_dim: int, width: int = 128, depth: int = 3, basis: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.basis = basis
        self.branch = MLP(param_dim, out_dim * basis, width, depth)
        self.trunk = MLP(coord_dim, out_dim * basis, width, depth)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = x[:, :2]
        params = x[:, 2:-2]
        branch = self.branch(params).view(-1, self.out_dim, self.basis)
        trunk = self.trunk(coords).view(-1, self.out_dim, self.basis)
        return (branch * trunk).sum(dim=-1) / math.sqrt(self.basis) + self.bias


def build_model(name: str, in_dim: int, out_dim: int, args) -> nn.Module:
    if name == "dnn":
        return PointDNN(in_dim, out_dim, args.width, args.depth)
    if name in {"unet", "cnn_unet"}:
        return UNet2d(in_channels=in_dim, out_channels=out_dim, width=args.width, depth=args.depth)
    if name == "pinn":
        return PINN(in_dim, out_dim, args.width, args.depth, activation="tanh")
    if name == "deeponet":
        return DeepONet(param_dim=in_dim - 4, coord_dim=2, out_dim=out_dim, width=args.width, depth=args.depth, basis=args.basis)
    if name == "fno":
        return FNO2d(in_channels=in_dim, out_channels=out_dim, width=args.width, depth=args.depth, modes_x=args.modes, modes_y=args.modes)
    raise ValueError(f"Unknown model: {name}")
