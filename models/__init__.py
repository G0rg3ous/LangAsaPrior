"""Models required by the AFIDs baseline and MLLM dynamic-query framework."""

from .unet3d_keypoint import (
    ConvBlock3D,
    UNet3DKeypointDetector,
    get_unet3d_keypoint_detector,
    soft_argmax_3d,
)
from .unet3d_mllm_text_dynamic_query_decoder_bottleneck import (
    MultiScaleDynamicLandmarkDecoder,
    UNet3DMLLMTextDynamicQueryDecoderBottleneckKeypointDetector,
    get_unet3d_mllm_text_dynamic_query_decoder_bottleneck_keypoint_detector,
)

__all__ = [
    "ConvBlock3D",
    "UNet3DKeypointDetector",
    "get_unet3d_keypoint_detector",
    "soft_argmax_3d",
    "MultiScaleDynamicLandmarkDecoder",
    "UNet3DMLLMTextDynamicQueryDecoderBottleneckKeypointDetector",
    "get_unet3d_mllm_text_dynamic_query_decoder_bottleneck_keypoint_detector",
]
