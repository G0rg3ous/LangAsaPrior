"""3D U-Net with a frozen MLLM layer inserted at the bottleneck.

This model keeps the encoder, decoder, skip connections, and heatmap head
identical to ``UNet3DKeypointDetector``. The only architectural change is a
feature-conversion adapter between encoder and decoder:

    bottleneck feature -> pooled spatial tokens -> linear to MLLM dim
    -> one frozen MLLM layer -> linear back to CNN channels -> upsample

No keypoint tokens or text prompts are used.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet3d_keypoint import ConvBlock3D, _center_pad_or_crop, soft_argmax_3d


class FrozenMLLMBottleneckAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        llm_model_name: Optional[str],
        llm_layer_idx: int = 17,
        llm_dim: Optional[int] = None,
        spatial_pool_size: Tuple[int, int, int] = (6, 6, 6),
        freeze_llm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.spatial_pool_size = tuple(int(v) for v in spatial_pool_size)
        self.llm_model_name = llm_model_name

        if not llm_model_name:
            self.llm_dim = self.in_channels if llm_dim is None else int(llm_dim)
            self.llm_layer = nn.Identity()
            self.llm_rotary_emb = None
            self.proj_in = nn.Identity() if self.llm_dim == self.in_channels else nn.Linear(self.in_channels, self.llm_dim)
            self.proj_out = nn.Identity() if self.llm_dim == self.in_channels else nn.Linear(self.llm_dim, self.in_channels)
            return

        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError("transformers is required for the MLLM bottleneck model.") from exc

        llm_config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
        try:
            llm_model = AutoModel.from_pretrained(llm_model_name, trust_remote_code=True, dtype=torch.float16)
        except Exception:
            llm_model = AutoModel.from_pretrained(llm_model_name, trust_remote_code=True)

        if llm_dim is None:
            if hasattr(llm_config, "text_config") and hasattr(llm_config.text_config, "hidden_size"):
                llm_dim = int(llm_config.text_config.hidden_size)
            elif hasattr(llm_config, "hidden_size"):
                llm_dim = int(llm_config.hidden_size)
            else:
                raise ValueError("Cannot infer MLLM hidden size. Please pass llm_dim.")
        self.llm_dim = int(llm_dim)

        if hasattr(llm_model, "language_model"):
            language_model = llm_model.language_model
            if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
                llm_layers = language_model.model.layers
            elif hasattr(language_model, "layers"):
                llm_layers = language_model.layers
            else:
                raise ValueError("Cannot find language_model layers in MLLM.")
        elif hasattr(llm_model, "model") and hasattr(llm_model.model, "layers"):
            llm_layers = llm_model.model.layers
        elif hasattr(llm_model, "layers"):
            llm_layers = llm_model.layers
        else:
            raise ValueError("Cannot find transformer layers in MLLM.")

        if llm_layer_idx >= len(llm_layers):
            raise ValueError(f"llm_layer_idx={llm_layer_idx} out of range for {len(llm_layers)} layers.")
        self.llm_layer = llm_layers[int(llm_layer_idx)]
        self.llm_rotary_emb = None
        if hasattr(llm_model, "language_model") and hasattr(llm_model.language_model, "rotary_emb"):
            self.llm_rotary_emb = llm_model.language_model.rotary_emb
        elif hasattr(llm_model, "model") and hasattr(llm_model.model, "rotary_emb"):
            self.llm_rotary_emb = llm_model.model.rotary_emb
        elif hasattr(llm_model, "rotary_emb"):
            self.llm_rotary_emb = llm_model.rotary_emb

        if freeze_llm:
            for param in self.llm_layer.parameters():
                param.requires_grad = False
            if self.llm_rotary_emb is not None:
                for param in self.llm_rotary_emb.parameters():
                    param.requires_grad = False

        self.proj_in = nn.Sequential(
            nn.Linear(self.in_channels, self.llm_dim),
            nn.LayerNorm(self.llm_dim),
            nn.GELU(),
        )
        self.proj_out = nn.Sequential(
            nn.Linear(self.llm_dim, self.in_channels),
            nn.LayerNorm(self.in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        pooled = F.adaptive_avg_pool3d(x, self.spatial_pool_size)
        pd, ph, pw = pooled.shape[-3:]
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens_in = self.proj_in(tokens)

        tokens_llm = self._run_llm_layer(tokens_in)
        if tokens_llm.device != x.device or tokens_llm.dtype != x.dtype:
            tokens_llm = tokens_llm.to(device=x.device, dtype=x.dtype)

        tokens_out = self.proj_out(tokens_llm)
        out = tokens_out.transpose(1, 2).reshape(b, c, pd, ph, pw)
        out = F.interpolate(out, size=(d, h, w), mode="trilinear", align_corners=False)
        return out

    def _run_llm_layer(self, tokens: torch.Tensor) -> torch.Tensor:
        if isinstance(self.llm_layer, nn.Identity):
            return tokens

        try:
            first_param = next(self.llm_layer.parameters())
            llm_device = first_param.device
            llm_dtype = first_param.dtype
        except StopIteration:
            llm_device = tokens.device
            llm_dtype = tokens.dtype

        hidden = tokens.to(device=llm_device, dtype=llm_dtype)
        b, seq_len, _ = hidden.shape
        position_ids = torch.arange(seq_len, device=llm_device).unsqueeze(0).expand(b, -1)

        kwargs = {
            "hidden_states": hidden,
            "position_ids": position_ids,
            "attention_mask": None,
            "past_key_values": None,
            "use_cache": False,
            "output_attentions": False,
        }
        if self.llm_rotary_emb is not None:
            kwargs["position_embeddings"] = self.llm_rotary_emb(hidden, position_ids)

        try:
            out = self.llm_layer(**kwargs)
        except TypeError:
            kwargs.pop("past_key_values", None)
            out = self.llm_layer(**kwargs)

        if isinstance(out, tuple):
            return out[0]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out


class UNet3DMLLMBottleneckKeypointDetector(nn.Module):
    """UNet3DKeypointDetector with a frozen MLLM bottleneck adapter."""

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
        llm_layer_idx: int = 17,
        llm_dim: Optional[int] = None,
        spatial_pool_size: Tuple[int, int, int] = (6, 6, 6),
        freeze_llm: bool = True,
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
        self.mllm_adapter = FrozenMLLMBottleneckAdapter(
            in_channels=channels[-1],
            llm_model_name=llm_model_name,
            llm_layer_idx=llm_layer_idx,
            llm_dim=llm_dim,
            spatial_pool_size=spatial_pool_size,
            freeze_llm=freeze_llm,
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


def get_unet3d_mllm_bottleneck_keypoint_detector(
    in_channels: int = 1,
    num_keypoints: int = 12,
    base_channels: int = 32,
    channel_multipliers: Iterable[int] = (1, 2, 4, 8),
    norm: str = "instance",
    dropout: float = 0.0,
    coord_temperature: float = 1.0,
    llm_model_name: Optional[str] = None,
    llm_layer_idx: int = 17,
    llm_dim: Optional[int] = None,
    spatial_pool_size: Tuple[int, int, int] = (6, 6, 6),
    freeze_llm: bool = True,
) -> UNet3DMLLMBottleneckKeypointDetector:
    return UNet3DMLLMBottleneckKeypointDetector(
        in_channels=in_channels,
        num_keypoints=num_keypoints,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        norm=norm,
        dropout=dropout,
        coord_temperature=coord_temperature,
        llm_model_name=llm_model_name,
        llm_layer_idx=llm_layer_idx,
        llm_dim=llm_dim,
        spatial_pool_size=spatial_pool_size,
        freeze_llm=freeze_llm,
    )


__all__ = [
    "FrozenMLLMBottleneckAdapter",
    "UNet3DMLLMBottleneckKeypointDetector",
    "get_unet3d_mllm_bottleneck_keypoint_detector",
]
