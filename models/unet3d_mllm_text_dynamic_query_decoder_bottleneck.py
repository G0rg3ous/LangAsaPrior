"""3D U-Net with MLLM-enhanced text queries and dynamic heatmap decoding."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .unet3d_keypoint import ConvBlock3D, _center_pad_or_crop, soft_argmax_3d
from .unet3d_mllm_text_bottleneck import TextConditionedMLLMBottleneckAdapter


class MultiScaleDynamicLandmarkDecoder(nn.Module):
    """Decode landmark heatmaps from text queries and multi-scale visual features."""

    def __init__(
        self,
        feature_channels: Sequence[int],
        text_dim: int,
        num_keypoints: int,
        output_channels: int,
        query_dim: int = 128,
        num_heads: int = 4,
        pool_sizes: Sequence[Tuple[int, int, int]] = ((6, 6, 6), (6, 6, 6), (8, 8, 8), (12, 12, 12)),
        dropout: float = 0.0,
        fusion_init: float = 0.1,
    ) -> None:
        super().__init__()
        channels = tuple(int(v) for v in feature_channels)
        pools = tuple(tuple(int(v) for v in size) for size in pool_sizes)
        if len(channels) != len(pools):
            raise ValueError(f"feature_channels and pool_sizes must have equal length, got {len(channels)} and {len(pools)}.")

        self.num_keypoints = int(num_keypoints)
        self.query_dim = int(query_dim)
        self.output_channels = int(output_channels)
        self.pool_sizes = pools

        self.text_proj = nn.Sequential(
            nn.Linear(int(text_dim), self.query_dim),
            nn.LayerNorm(self.query_dim),
            nn.GELU(),
        )
        self.visual_projs = nn.ModuleList(
            [nn.Conv3d(channel, self.query_dim, kernel_size=1, bias=False) for channel in channels]
        )
        self.visual_norms = nn.ModuleList([nn.LayerNorm(self.query_dim) for _ in channels])
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.query_dim,
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in channels
            ]
        )
        self.query_norms = nn.ModuleList([nn.LayerNorm(self.query_dim) for _ in channels])
        self.query_ffn = nn.Sequential(
            nn.Linear(self.query_dim, self.query_dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.query_dim * 4, self.query_dim),
        )
        self.final_query_norm = nn.LayerNorm(self.query_dim)
        self.kernel_generator = nn.Linear(self.query_dim, self.output_channels)
        self.bias_generator = nn.Linear(self.query_dim, 1)
        self.semantic_scale = nn.Parameter(torch.tensor(float(fusion_init)))

    def forward(self, features: Sequence[torch.Tensor], text_tokens: torch.Tensor) -> torch.Tensor:
        if len(features) != len(self.visual_projs):
            raise ValueError(f"Expected {len(self.visual_projs)} feature scales, got {len(features)}.")
        if text_tokens.shape[1] != self.num_keypoints:
            raise ValueError(f"Expected {self.num_keypoints} text tokens, got {text_tokens.shape[1]}.")

        queries = self.text_proj(text_tokens)
        for feature, visual_proj, visual_norm, cross_attn, query_norm, pool_size in zip(
            features,
            self.visual_projs,
            self.visual_norms,
            self.cross_attns,
            self.query_norms,
            self.pool_sizes,
        ):
            pooled = torch.nn.functional.adaptive_avg_pool3d(feature, pool_size)
            visual_tokens = visual_proj(pooled).flatten(2).transpose(1, 2)
            visual_tokens = visual_norm(visual_tokens)
            attended, _ = cross_attn(query=queries, key=visual_tokens, value=visual_tokens, need_weights=False)
            queries = query_norm(queries + attended)

        queries = self.final_query_norm(queries + self.query_ffn(queries))
        kernels = self.kernel_generator(queries)
        biases = self.bias_generator(queries)

        output_feature = features[-1]
        b, c, d, h, w = output_feature.shape
        if c != self.output_channels:
            raise ValueError(f"Expected final decoder feature with {self.output_channels} channels, got {c}.")
        dynamic_heatmaps = torch.bmm(kernels, output_feature.flatten(2))
        dynamic_heatmaps = dynamic_heatmaps + biases
        dynamic_heatmaps = dynamic_heatmaps.reshape(b, self.num_keypoints, d, h, w)
        return self.semantic_scale * dynamic_heatmaps


class UNet3DMLLMTextDynamicQueryDecoderBottleneckKeypointDetector(nn.Module):
    """3D U-Net with an MLLM-enhanced multi-scale dynamic landmark decoder."""

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
        query_dim: int = 128,
        query_heads: int = 4,
        query_pool_sizes: Sequence[Tuple[int, int, int]] = ((6, 6, 6), (6, 6, 6), (8, 8, 8), (12, 12, 12)),
        query_dropout: float = 0.0,
        semantic_fusion_init: float = 0.1,
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
        decoder_channels = []
        for idx in range(len(channels) - 1, 0, -1):
            self.upconvs.append(nn.ConvTranspose3d(channels[idx], channels[idx - 1], kernel_size=2, stride=2))
            self.decoders.append(
                ConvBlock3D(channels[idx - 1] * 2, channels[idx - 1], norm=norm, dropout=dropout)
            )
            decoder_channels.append(channels[idx - 1])

        self.head = nn.Conv3d(channels[0], self.num_keypoints, kernel_size=1)
        self.dynamic_decoder = MultiScaleDynamicLandmarkDecoder(
            feature_channels=[channels[-1], *decoder_channels],
            text_dim=self.mllm_adapter.llm_dim,
            num_keypoints=self.num_keypoints,
            output_channels=channels[0],
            query_dim=query_dim,
            num_heads=query_heads,
            pool_sizes=query_pool_sizes,
            dropout=query_dropout,
            fusion_init=semantic_fusion_init,
        )

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

        out, enhanced_text_tokens = self.mllm_adapter.forward_with_text_tokens(out)
        multi_scale_features = [out]

        skips = skips[:-1][::-1]
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, skips):
            out = upconv(out)
            out = _center_pad_or_crop(out, skip.shape[-3:])
            out = torch.cat([skip, out], dim=1)
            out = decoder(out)
            multi_scale_features.append(out)

        cnn_heatmaps = self.head(out)
        dynamic_heatmaps = self.dynamic_decoder(multi_scale_features, enhanced_text_tokens)
        heatmaps = cnn_heatmaps + dynamic_heatmaps
        if not return_dict:
            return heatmaps

        pred_coords = soft_argmax_3d(heatmaps.detach(), temperature=self.coord_temperature)
        return {
            "heatmaps": heatmaps,
            "response_maps": heatmaps,
            "pred_coords": pred_coords,
            "coarse_coords": pred_coords,
            "cnn_heatmaps": cnn_heatmaps.detach(),
            "dynamic_heatmaps": dynamic_heatmaps.detach(),
        }


def get_unet3d_mllm_text_dynamic_query_decoder_bottleneck_keypoint_detector(
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
    query_dim: int = 128,
    query_heads: int = 4,
    query_pool_sizes: Sequence[Tuple[int, int, int]] = ((6, 6, 6), (6, 6, 6), (8, 8, 8), (12, 12, 12)),
    query_dropout: float = 0.0,
    semantic_fusion_init: float = 0.1,
) -> UNet3DMLLMTextDynamicQueryDecoderBottleneckKeypointDetector:
    return UNet3DMLLMTextDynamicQueryDecoderBottleneckKeypointDetector(
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
        query_dim=query_dim,
        query_heads=query_heads,
        query_pool_sizes=query_pool_sizes,
        query_dropout=query_dropout,
        semantic_fusion_init=semantic_fusion_init,
    )


__all__ = [
    "MultiScaleDynamicLandmarkDecoder",
    "UNet3DMLLMTextDynamicQueryDecoderBottleneckKeypointDetector",
    "get_unet3d_mllm_text_dynamic_query_decoder_bottleneck_keypoint_detector",
]
