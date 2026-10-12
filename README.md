# AFIDs 3D U-Net Landmark Baseline

This repository contains the code required to train and evaluate both the plain
3D U-Net baseline and the MLLM-enhanced dynamic landmark-query framework on the
AFIDs 1 mm, `192 x 192 x 192` dataset split used in this project.

The repository provides implementation code and split metadata for reproducibility.
**The AFIDs dataset is not distributed through this repository.** Researchers
should obtain the imaging data and annotations directly from the official
[AFIDs data repository](https://github.com/afids/afids-data), subject to the
original datasets' access requirements. Checkpoints, TensorBoard logs, and
derived prediction files are not redistributed.

## Contents

- `train_unet3d_keypoint_afids_1mm_192.py`: AFIDs-specific distributed training entry point.
- `train_unet3d_keypoint.py`: shared train/validation/test loop.
- `models/unet3d_keypoint.py`: 3D U-Net heatmap detector and soft-argmax.
- `models/unet3d_mllm_bottleneck.py`: frozen Qwen3-VL language-layer adapter.
- `models/unet3d_mllm_text_bottleneck.py`: fixed text-token and 3D visual-token fusion.
- `models/unet3d_mllm_text_dynamic_query_decoder_bottleneck.py`: multi-scale
  text-conditioned dynamic landmark-query decoder.
- `train_unet3d_mllm_text_dynamic_query_decoder_bottleneck_afids_1mm_192.py`:
  AFIDs MLLM training entry point.
- `fixed_text_embeddings_qwen3vl_afids32_names_only.pt`: fixed, mean-pooled
  32-landmark text tokens; this is not a Qwen checkpoint.
- `scripts/prepare_text_embeddings.py`: optional offline text-token preparation.
- `dataset/`: AFIDs NIfTI/FCSV loading and Gaussian heatmap construction.
- `utils/`: loss, metrics, and split helpers.
- `configs/`: baseline and AFIDs-192 configuration.
- `splits/split_afids_7_1_2.json`: fixed 92/13/27 train/validation/test split.

## AFIDs dataset availability

**We do not directly provide or redistribute the AFIDs dataset**, including its
MRI volumes or ground-truth landmark annotations. The dataset should be obtained
from the official [AFIDs data repository](https://github.com/afids/afids-data).
Please follow the access instructions and applicable data-use terms provided by
the original dataset maintainers. Some underlying imaging datasets, including
HCP and OASIS, have their own access conditions and data-use agreements.

This repository supplies code and experiment split metadata only; access to the
original AFIDs data must be arranged separately. After obtaining the data,
prepare them locally in the format described below.

### Knee CT dataset

The knee CT dataset used in this study is a private research dataset and is
not publicly available at present. We are currently unable to release these
data. We will consider making the dataset publicly available in future work,
subject to the necessary data-sharing permissions and approvals. No release
timeline is currently specified.

The loader expects a prepared data root with the following layout:

```text
${AFIDS_DATA_DIR}/
  AFIDs-HCP/sub-103111/anat/*T1w.nii.gz
  AFIDs-HCP/derivatives/afids_groundtruth/sub-103111/anat/*groundtruth_afids.fcsv
  AFIDs-OASIS/...
  LHSCPD/...
  SNSX/...
```

For this experiment, every image must already be 1 mm isotropic,
`192 x 192 x 192`, and normalized to `[0, 1]`. The code validates these
conditions before training. The original local preprocessing utility was not
part of the source tree used for this release, so it is not represented as an
unverified replacement here.

## Environment

```bash
conda create -n afids-unet python=3.9 -y
conda activate afids-unet
pip install -r requirements.txt
```

Install a PyTorch build appropriate for the local CUDA driver before, or in
place of, the generic PyTorch requirement in `requirements.txt`.

## Training

Set the prepared-data root and optional output root, then launch one process per
GPU:

```bash
export AFIDS_DATA_DIR=/absolute/path/to/datasets_1mm_192
export AFIDS_OUTPUT_DIR=/absolute/path/to/outputs

python train_unet3d_keypoint_afids_1mm_192.py --gpus 0,1,2,3
```

The script uses the fixed 7/1/2 split in `splits/split_afids_7_1_2.json`,
trains for 300 epochs by default, selects the best validation checkpoint, and
evaluates the held-out test set at the end of training.

### MLLM dynamic-query framework

Download Qwen3-VL-2B-Thinking separately and set its local path. The Qwen
weights are not included in this repository. The released AFIDs text-token
cache is already compatible with the expected 2048-dimensional text hidden
space.

```bash
export AFIDS_DATA_DIR=/absolute/path/to/datasets_1mm_192
export AFIDS_OUTPUT_DIR=/absolute/path/to/outputs
export QWEN3_VL_MODEL_DIR=/absolute/path/to/Qwen3-VL-2B-Thinking

python train_unet3d_mllm_text_dynamic_query_decoder_bottleneck_afids_1mm_192.py \
  --gpus 0,1,2,3
```

The framework pools the U-Net bottleneck to 3D visual tokens, concatenates
them before the 32 fixed landmark text tokens, passes them through frozen
Qwen3-VL language layer index 17, and uses the resulting landmark tokens in a
four-scale dynamic heatmap decoder.

## Coordinate convention

NIfTI data are loaded in `(x, y, z)` voxel order, converted to the network's
`(z, y, x)` order, and FCSV physical coordinates are mapped through the inverse
NIfTI affine. The 32 output heatmaps follow the AFIDs labels 1--32.

## Reproducibility scope

This release includes the AFIDs plain 3D U-Net baseline and the MLLM dynamic
query-decoder framework used in the corresponding experiments. It excludes
unrelated alternative backbones, ablation-only variants, checkpoints, logs, and
private development artifacts.
