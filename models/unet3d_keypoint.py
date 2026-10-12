"""Standard 3D U-Net for volumetric keypoint heatmap detection.

The model predicts one heatmap channel per keypoint from an input volume with
shape ``[B, C, D, H, W]``. Coordinates are returned in ``(z, y, x)`` order when
``return_dict=True``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tuple(value: int | Sequence[int], length: int) -> Tuple[int, ...]:
    if isinstance(value, int):
        return tuple([value] * length)
    if len(value) != length:
        raise ValueError(f"Expected {length} values, got {value}")
    return tuple(int(v) for v in value)


def soft_argmax_3d(heatmap_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute differentiable coordinates from 3D heatmap logits.

    Args:
        heatmap_logits: Tensor with shape ``[B, K, D, H, W]``.
        temperature: Spatial softmax temperature.

    Returns:
        Tensor with shape ``[B, K, 3]`` in ``(z, y, x)`` order.
    """
    if heatmap_logits.ndim != 5:
        raise ValueError(f"Expected heatmap logits [B,K,D,H,W], got {tuple(heatmap_logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    b, k, d, h, w = heatmap_logits.shape
    logits = (heatmap_logits / float(temperature)).float().clamp(min=-50.0, max=50.0)
    prob = F.softmax(logits.flatten(2), dim=-1).view(b, k, d, h, w)

    z = torch.arange(d, device=heatmap_logits.device, dtype=prob.dtype).view(1, 1, d, 1, 1)
    y = torch.arange(h, device=heatmap_logits.device, dtype=prob.dtype).view(1, 1, 1, h, 1)
    x = torch.arange(w, device=heatmap_logits.device, dtype=prob.dtype).view(1, 1, 1, 1, w)
    coords = torch.stack(
        [
            (prob * z).sum(dim=(2, 3, 4)),
            (prob * y).sum(dim=(2, 3, 4)),
            (prob * x).sum(dim=(2, 3, 4)),
        ],
        dim=-1,
    )
    return coords


class ConvBlock3D(nn.Module):
    """Two-convolution 3D U-Net block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "instance",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        channels = [int(in_channels), int(out_channels), int(out_channels)]
        for idx in range(2):
            layers.append(nn.Conv3d(channels[idx], channels[idx + 1], kernel_size=3, padding=1, bias=False))
            layers.append(_make_norm(norm, channels[idx + 1]))
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout3d(p=float(dropout)))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _make_norm(norm: str, channels: int) -> nn.Module:
    norm = norm.lower()
    if norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    if norm == "group":
        groups = 8
        while channels % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, channels)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm}")


class UNet3DKeypointDetector(nn.Module):
    """3D U-Net keypoint detector with heatmap output.

    Args:
        in_channels: Number of input image channels.
        num_keypoints: Number of keypoint heatmap channels.
        base_channels: Number of channels in the first encoder stage.
        channel_multipliers: Multipliers for each encoder stage.
        norm: ``instance``, ``batch``, ``group``, or ``none``.
        dropout: Dropout probability inside convolution blocks.
        coord_temperature: Temperature used by soft-argmax coordinate output.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_keypoints: int = 12,
        base_channels: int = 32,
        channel_multipliers: Iterable[int] = (1, 2, 4, 8),
        norm: str = "instance",
        dropout: float = 0.0,
        coord_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        multipliers = tuple(int(m) for m in channel_multipliers)
        if len(multipliers) < 2:
            raise ValueError("channel_multipliers must contain at least two stages")

        self.in_channels = int(in_channels)
        self.num_keypoints = int(num_keypoints)
        self.coord_temperature = float(coord_temperature)
        channels = [int(base_channels) * m for m in multipliers]

        self.encoders = nn.ModuleList()
        prev_channels = self.in_channels
        for out_channels in channels:
            self.encoders.append(ConvBlock3D(prev_channels, out_channels, norm=norm, dropout=dropout))
            prev_channels = out_channels

        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for idx in range(len(channels) - 1, 0, -1):
            self.upconvs.append(nn.ConvTranspose3d(channels[idx], channels[idx - 1], kernel_size=2, stride=2))
            self.decoders.append(
                ConvBlock3D(channels[idx - 1] * 2, channels[idx - 1], norm=norm, dropout=dropout)
            )

        self.head = nn.Conv3d(channels[0], self.num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor, return_dict: bool = True):
        if x.ndim != 5:
            raise ValueError(f"Expected input [B,C,D,H,W], got {tuple(x.shape)}")

        skips = []
        out = x
        for idx, encoder in enumerate(self.encoders):
            out = encoder(out)
            skips.append(out)
            if idx != len(self.encoders) - 1:
                out = self.downsample(out)

        skips = skips[:-1][::-1]
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, skips):
            out = upconv(out)
            out = _center_pad_or_crop(out, skip.shape[-3:])
            out = torch.cat([skip, out], dim=1)
            out = decoder(out)

        heatmaps = self.head(out)
        if not return_dict:
            return heatmaps

        pred_coords = soft_argmax_3d(heatmaps, temperature=self.coord_temperature)
        return {
            "heatmaps": heatmaps,
            "response_maps": heatmaps,
            "pred_coords": pred_coords,
            "coarse_coords": pred_coords,
        }


def _center_pad_or_crop(x: torch.Tensor, spatial_size: Sequence[int]) -> torch.Tensor:
    target = _as_tuple(spatial_size, 3)
    current = tuple(int(v) for v in x.shape[-3:])

    pads = []
    slices = []
    for cur, tgt in zip(reversed(current), reversed(target)):
        if cur < tgt:
            total = tgt - cur
            pads.extend([total // 2, total - total // 2])
            slices.append(slice(None))
        else:
            pads.extend([0, 0])
            start = (cur - tgt) // 2
            slices.append(slice(start, start + tgt))

    if any(pads):
        x = F.pad(x, pads)
    slices = tuple(reversed(slices))
    return x[(slice(None), slice(None)) + slices]


def get_unet3d_keypoint_detector(
    in_channels: int = 1,
    num_keypoints: int = 12,
    base_channels: int = 32,
    channel_multipliers: Iterable[int] = (1, 2, 4, 8),
    norm: str = "instance",
    dropout: float = 0.0,
    coord_temperature: float = 1.0,
) -> UNet3DKeypointDetector:
    return UNet3DKeypointDetector(
        in_channels=in_channels,
        num_keypoints=num_keypoints,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        norm=norm,
        dropout=dropout,
        coord_temperature=coord_temperature,
    )


__all__ = [
    "ConvBlock3D",
    "UNet3DKeypointDetector",
    "get_unet3d_keypoint_detector",
    "soft_argmax_3d",
]
