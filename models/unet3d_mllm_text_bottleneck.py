"""3D U-Net with frozen MLLM bottleneck and fixed landmark text tokens.

Visual bottleneck tokens are concatenated with fixed Qwen3-VL text embeddings
for 12 knee landmarks, then passed through one frozen Qwen3-VL language decoder
layer. Only the visual tokens are projected back to the 3D U-Net decoder.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet3d_keypoint import ConvBlock3D, _center_pad_or_crop, soft_argmax_3d
from .unet3d_mllm_bottleneck import FrozenMLLMBottleneckAdapter


class SpatialPositionalEncoding3D(nn.Module):
    """3D positional encoding for pooled visual tokens."""

    def __init__(self, dim: int, spatial_size: Tuple[int, int, int]) -> None:
        super().__init__()
        self.dim = int(dim)
        self.spatial_size = tuple(int(v) for v in spatial_size)

        dim_d = self.dim // 3
        dim_h = self.dim // 3
        dim_w = self.dim - dim_d - dim_h
        self.dim_splits = (dim_d, dim_h, dim_w)

        d, h, w = self.spatial_size
        self.pos_d = nn.Parameter(torch.zeros(d, dim_d))
        self.pos_h = nn.Parameter(torch.zeros(h, dim_h))
        self.pos_w = nn.Parameter(torch.zeros(w, dim_w))
        nn.init.trunc_normal_(self.pos_d, std=0.02)
        nn.init.trunc_normal_(self.pos_h, std=0.02)
        nn.init.trunc_normal_(self.pos_w, std=0.02)

    def forward(self, tokens: torch.Tensor, spatial_size: Tuple[int, int, int]) -> torch.Tensor:
        d, h, w = tuple(int(v) for v in spatial_size)
        if (d, h, w) != self.spatial_size:
            raise ValueError(f"Expected pooled spatial size {self.spatial_size}, got {(d, h, w)}.")

        d_idx = torch.arange(d, device=tokens.device)
        h_idx = torch.arange(h, device=tokens.device)
        w_idx = torch.arange(w, device=tokens.device)
        dd, hh, ww = torch.meshgrid(d_idx, h_idx, w_idx, indexing="ij")
        pos = torch.cat(
            [
                self.pos_d[dd.reshape(-1)],
                self.pos_h[hh.reshape(-1)],
                self.pos_w[ww.reshape(-1)],
            ],
            dim=-1,
        )
        return tokens + pos.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)


class TextConditionedMLLMBottleneckAdapter(FrozenMLLMBottleneckAdapter):
    def __init__(
        self,
        in_channels: int,
        llm_model_name: Optional[str],
        text_embedding_path: str,
        llm_layer_idx: int = 17,
        llm_dim: Optional[int] = None,
        spatial_pool_size: Tuple[int, int, int] = (6, 6, 6),
        freeze_llm: bool = True,
        text_first: bool = False,
        use_3d_position_encoding: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            llm_model_name=llm_model_name,
            llm_layer_idx=llm_layer_idx,
            llm_dim=llm_dim,
            spatial_pool_size=spatial_pool_size,
            freeze_llm=freeze_llm,
        )
        self.text_first = bool(text_first)
        self.use_3d_position_encoding = bool(use_3d_position_encoding)
        self.spatial_pos_embed = (
            SpatialPositionalEncoding3D(self.llm_dim, self.spatial_pool_size)
            if self.use_3d_position_encoding
            else nn.Identity()
        )
        text_embeddings = self._load_text_embeddings(text_embedding_path)
        if text_embeddings.shape[-1] != self.llm_dim:
            raise ValueError(
                f"Text embedding dim {text_embeddings.shape[-1]} does not match MLLM dim {self.llm_dim}."
            )
        self.register_buffer("fixed_text_tokens", text_embeddings.unsqueeze(0), persistent=True)

    @staticmethod
    def _load_text_embeddings(path: str) -> torch.Tensor:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if "embeddings" not in obj:
                raise KeyError(f"{path} is a dict but does not contain an 'embeddings' key.")
            embeddings = obj["embeddings"]
        else:
            embeddings = obj
        if not torch.is_tensor(embeddings):
            raise TypeError(f"Expected tensor embeddings in {path}, got {type(embeddings)}.")
        if embeddings.ndim != 2:
            raise ValueError(f"Expected text embeddings with shape [T, C], got {tuple(embeddings.shape)}.")
        return embeddings.detach().float()

    def forward_with_text_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, d, h, w = x.shape
        pooled = F.adaptive_avg_pool3d(x, self.spatial_pool_size)
        pd, ph, pw = pooled.shape[-3:]
        visual_tokens = pooled.flatten(2).transpose(1, 2)
        visual_tokens = self.proj_in(visual_tokens)
        if self.use_3d_position_encoding:
            visual_tokens = self.spatial_pos_embed(visual_tokens, (pd, ph, pw))

        text_tokens = self.fixed_text_tokens.to(device=visual_tokens.device, dtype=visual_tokens.dtype)
        text_tokens = text_tokens.expand(b, -1, -1)
        num_visual_tokens = visual_tokens.shape[1]

        if self.text_first:
            tokens_in = torch.cat([text_tokens, visual_tokens], dim=1)
            tokens_llm = self._run_llm_layer(tokens_in)
            text_tokens_llm = tokens_llm[:, : text_tokens.shape[1]]
            visual_tokens_llm = tokens_llm[:, -num_visual_tokens:]
        else:
            tokens_in = torch.cat([visual_tokens, text_tokens], dim=1)
            tokens_llm = self._run_llm_layer(tokens_in)
            visual_tokens_llm = tokens_llm[:, :num_visual_tokens]
            text_tokens_llm = tokens_llm[:, num_visual_tokens:]

        if visual_tokens_llm.device != x.device or visual_tokens_llm.dtype != x.dtype:
            visual_tokens_llm = visual_tokens_llm.to(device=x.device, dtype=x.dtype)
        if text_tokens_llm.device != x.device or text_tokens_llm.dtype != x.dtype:
            text_tokens_llm = text_tokens_llm.to(device=x.device, dtype=x.dtype)

        visual_tokens_out = self.proj_out(visual_tokens_llm)
        out = visual_tokens_out.transpose(1, 2).reshape(b, c, pd, ph, pw)
        out = F.interpolate(out, size=(d, h, w), mode="trilinear", align_corners=False)
        return out, text_tokens_llm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_with_text_tokens(x)
        return out


class UNet3DMLLMTextBottleneckKeypointDetector(nn.Module):
    """3D U-Net with fixed text tokens interacting inside a frozen MLLM layer."""

    def __init__(
        self,
        in_channels: int = 1,
        num_keypoints: int = 12,
        base_channels: int = 32,
        channel_multipliers: Iterable[int] = (1, 2, 4, 8),
        norm: str = "instance",
        dropout: float = 0.0,
        coord_temperature: float = 1.0,
        llm_model_name: Optional[str] = None,
        text_embedding_path: str = "",
        llm_layer_idx: int = 17,
        llm_dim: Optional[int] = None,
        spatial_pool_size: Tuple[int, int, int] = (6, 6, 6),
        freeze_llm: bool = True,
        text_first: bool = False,
        use_3d_position_encoding: bool = True,
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
        self.mllm_adapter = TextConditionedMLLMBottleneckAdapter(
            in_channels=channels[-1],
            llm_model_name=llm_model_name,
            text_embedding_path=text_embedding_path,
            llm_layer_idx=llm_layer_idx,
            llm_dim=llm_dim,
            spatial_pool_size=spatial_pool_size,
            freeze_llm=freeze_llm,
            text_first=text_first,
            use_3d_position_encoding=use_3d_position_encoding,
        )

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

        out = self.mllm_adapter(out)

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


def get_unet3d_mllm_text_bottleneck_keypoint_detector(
    in_channels: int = 1,
    num_keypoints: int = 12,
    base_channels: int = 32,
    channel_multipliers: Iterable[int] = (1, 2, 4, 8),
    norm: str = "instance",
    dropout: float = 0.0,
    coord_temperature: float = 1.0,
    llm_model_name: Optional[str] = None,
    text_embedding_path: str = "",
    llm_layer_idx: int = 17,
    llm_dim: Optional[int] = None,
    spatial_pool_size: Tuple[int, int, int] = (6, 6, 6),
    freeze_llm: bool = True,
    text_first: bool = False,
    use_3d_position_encoding: bool = True,
) -> UNet3DMLLMTextBottleneckKeypointDetector:
    return UNet3DMLLMTextBottleneckKeypointDetector(
        in_channels=in_channels,
        num_keypoints=num_keypoints,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        norm=norm,
        dropout=dropout,
        coord_temperature=coord_temperature,
        llm_model_name=llm_model_name,
        text_embedding_path=text_embedding_path,
        llm_layer_idx=llm_layer_idx,
        llm_dim=llm_dim,
        spatial_pool_size=spatial_pool_size,
        freeze_llm=freeze_llm,
        text_first=text_first,
        use_3d_position_encoding=use_3d_position_encoding,
    )


__all__ = [
    "TextConditionedMLLMBottleneckAdapter",
    "SpatialPositionalEncoding3D",
    "UNet3DMLLMTextBottleneckKeypointDetector",
    "get_unet3d_mllm_text_bottleneck_keypoint_detector",
]
