# HyperResASHS

[![arXiv](https://img.shields.io/badge/arXiv-2508.17171-b31b1b.svg)](https://doi.org/10.48550/arXiv.2508.17171)

**HyperResASHS** is a deep learning pipeline for isotropic segmentation of medial temporal lobe (MTL) subregions from multi-modality 3T MRI (T1w and T2w). This repository implements the method described in our paper for achieving high-resolution, isotropic segmentation of brain structures.

## Overview

This project addresses the challenge of segmenting MTL subregions from anisotropic MRI data by:
1. Building an isotropic atlas using Implicit Neural Representations (INR)
2. Training a multi-modality segmentation model with nnU-Net
3. Performing inference on test data with automatic preprocessing

The pipeline handles the entire workflow from raw multi-modality MRI images to final segmentation results, including registration, ROI extraction, upsampling, and model inference.

## Requirements
* **NVidia GPU** is needed to run training and inference. It is possible to run inference on CPU but runtimes will be over an hour per case, vs. ~3 minutes on GPU.
* A Python (version 3.10 or greater) environment with **conda** package manager and is recommended

## Easy Setup (Most Users)

Follow these steps to set up the repository in a fresh environment:

### 1. Create a Conda Environment

Create a new `conda` environment with Python 3.10 or higher:

```bash
conda create -n hyperresashs python=3.10
conda activate hyperresashs
```

An alternative to using conda is to create a Python virtual environment

```bash
cd /my/project/path
python -v env .venv
source .venv/bin/activate
```

### 2. Install hyperresashs from PyPi

Install the package and all dependencies

```bash
pip install hyperresashs
```

Check that the install was successful

```bash
hrashs check
```

The command should report that HyperResASHS was successfully installed and print the software version. If you see errors/warnings related to PyTorch and CUDA, please follow more detailed instructions for installing PyTorch under *Advanced Setup* below.

## Advanced Setup (Developers)

Follow these steps to set up the repository in a fresh environment:

### 1. Create a Conda Environment

Create a new conda environment with Python 3.10 or higher:

```bash
conda create -n hyperresashs python=3.10
conda activate hyperresashs
```

### 2. Clone the Repository

Clone the repository with submodules:

```bash
git clone --recursive https://github.com/pyushkevich/HyperResASHS.git
cd HyperResASHS
```

If you've already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

### 3. Install Python Dependencies

**Important**: PyTorch version compatibility is critical. This pipeline requires PyTorch 2.5.x (tested with 2.5.1). Newer versions (e.g., 2.9) may cause compatibility issues.

For optimal performance, you should install PyTorch library optimized for your CUDA version. To check your CUDA version, type

```bash
nvidia-smi | grep "CUDA Version"
```

The output will be like this. If you get an error, then CUDA is probably not configured on your machine.
```
| NVIDIA-SMI 535.288.01             Driver Version: 535.288.01   CUDA Version: 12.2     |
```

Then enter the following command, with `cu118` changed to match your cuda version (e.g., `cu122`):

```bash
# For CUDA 11.8:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

Then install the package and remaining dependencies:

```bash
# Then install the package and remaining dependencies
pip install -e .
```

### 6. Verify Installation

Verify that the package and dependencies are installed correctly:

```bash
hrashs check
```

## Basic Usage

HyperResASHS provides a command-line interface for running segmentation on new subjects using pre-trained models.

### 1. List Available Models

View all pre-trained models available for download:

```bash
hrashs list
```

This displays model IDs, names, and brief descriptions of each available model.

### 2. View Model Details

Get detailed information about a specific model including required inputs and expected outputs:

```bash
hrashs desc <model_id>
```

Example:
```bash
hrashs desc HSG_Body_Atlas
```

### 3. Run Segmentation

Run segmentation on your subject data:

```bash
hrashs run -a <model_id> -g <t1_image> -f <t2_image> -w <workdir>
```

**Required arguments:**
- `-a, --atlas`: Model ID or path to atlas config file
- `-g, --t1`: Path to T1-weighted (MPRAGE) image
- `-f, --t2`: Path to T2-weighted (TSE) image  
- `-w, --workdir`: Output directory for results

**Optional arguments:**
- `-I, --subject`: Subject ID, which will be prefixed to all output filenames
- `-t, --threads`: Number of parallel threads for registration, etc. [default: 1]
- `--device`: Device for computation ("cuda" or "cpu") [default: auto]
- `-N, --no-overwrite`: Do not overwrite existing results
- `-L, --no-links`: Copy files instead of creating symlinks
- `-T, --tidy`: Reduce intermediate files
- `-k, --disable-ssl-verification`: Disable SSL verification for Hugging Face

**Example:**
```bash
hrashs run -a HSG_Body_Atlas -g mprage.nii.gz -f tse.nii.gz -I sub01 -w ./output -t 4
```

**Outputs:** Segmentation results are saved in the `final` subfolder of the output directory. Two types of images are generated:
* `[subject]_tse_patch_hyperres_seg_[side].nii.gz`: These are HyperResASHS segmentations with nearly isotropic resolution (e.g., 0.4x0.4x0.4mm^3). We recommend using these images for volumetry, thickness computation, running CRASHS, etc. 
* `[subject]_tse_native_seg_[side].nii.gz`: These segmentations have been downsampled to original T2-MRI space. They can be used for visualization or computing similarity metrics with manual segmentations. However, these segmentations suffer from step edge artifacts when the T2-MRI is anisotropic.
* `[subject]_hyperres_volumes.csv`: A CSV file with subfield volumes (computed from hyper-resolution segmentations).

**QC Images:** Registration and segmentation QC images in .png format are placed in the `qc` subfolder of the output directory. 

## Training

Train a custom HyperResASHS model on your own dataset:

```bash
hrashs train -c <config> -m <manifest> -l <labels> -w <workdir>
```

**Required arguments:**
- `-c, --config`: Path to training configuration YAML file (see format below)
- `-m, --manifest`: Path to manifest CSV describing the training data
- `-l, --labels`: Path to ITK-SNAP label description file
- `-w, --workdir`: Output directory for training artifacts

**Optional arguments:**
- `-x, --xval`: Cross-validation fold file (if not provided, random 5-fold split is used)
- `-s, --stage`: Run specific stage(s): `1` (preprocess), `2` (INR), `3` (nnU-Net prep), `4` (nnU-Net train), or ranges like `1-3`
- `-F, --filter`: Regex to filter subjects/folds (stage-dependent)
- `-R, --inr-random-seed`: Random seed for INR optimization
- `--inr-batch-size`: INR batch size [default: 2000]
- `--inr-epochs`: INR training epochs [default: 60]
- `-t, --threads`, `--device`, `-N`, `-L`, `-T`, `-k`: Same as `run` command

**Example:**
```bash
hrashs train -c atlas.yaml -m manifest.csv -l labels.txt -w ./training -s 1-2 -t 8
```

### Configuration File Format (`-c`)

YAML file with atlas metadata and training parameters.

```yaml
metadata:
  id: 'MyAtlas'
  name: 'My Custom Atlas'
  version: '1.0.0'
config:
  EXP_NUM: 401                    # Unique experiment number
  MODEL_NAME: 'IsotropicSeg'      # Model name
  TRAINER: 'ModAugUNetTrainer'    # nnU-Net trainer class
  CONDITION: 'in_vivo'            # Condition identifier
  UPSAMPLING_METHOD: 'INRUpsampling'
```

### Manifest File Format (`-m`)

CSV file listing training cases. Paths can be absolute or relative to the manifest file:

```csv
id,date,mprage,tse,seg_left,seg_right
subj01,2024-01-15,subj01/mprage.nii.gz,subj01/tse.nii.gz,subj01/seg_left.nii.gz,subj01/seg_right.nii.gz
subj02,,subj02/mprage.nii.gz,subj02/tse.nii.gz,subj02/seg_left.nii.gz,subj02/seg_right.nii.gz
```

Required columns: `id`, `mprage`, `tse`, `seg_left`, `seg_right`. The `date` column is optional.

### Label File Format (`-l`)

ITK-SNAP label description file format:

```
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
   0     0    0    0    0    0   0    "Clear Label"
   1   255    0    0    1    1   1    "CA1"
   2     0  255    0    1    1   1    "CA2"
```

### Cross-Validation File Format (`-x`)

Optional text file where each line lists subject IDs for one held-out fold (space-separated):

```
subj01 subj02 subj03
subj04 subj05 subj06
subj07 subj08 subj09
```

### Output Structure

```
workdir/
├── preproc/           # Preprocessed images per subject
├── inr_training/      # INR training data and outputs
├── nnunet_training/   # nnU-Net datasets and trained models
└── final/             # Final trained atlas ready for deployment
```

For detailed configuration information, see [Configuration Guide](docs/configuration.md).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{hyperresashs2024,
  title={HyperResASHS: Isotropic Segmentation of MTL Subregions from Multi-modality 3T MRI},
  author={[Authors]},
  journal={arXiv preprint arXiv:2508.17171},
  year={2024},
  url={https://doi.org/10.48550/arXiv.2508.17171}
}
```

## Changelog

### 01/14/2026
- Replaced `trim_neck.sh` shell script with Python implementation using `picsl_c3d` package
- Removed ITK-SNAP installation requirement (no longer needed)
- Updated `multi_contrast_inr` submodule to latest version
- Modified `--config_id` argument to accept both integer ID and full file path
- Added config validation checks: ID consistency, conflict detection, and nnUNet dataset existence
- Made `FILE_NAME_CONFIG` optional with automatic defaults based on stage (test vs. other stages)
- Renamed `scripts/` folder to `shell/` for better clarity
- Added optional `--subject_id` argument for test stage to test specific subjects
- Updated `.gitignore` to exclude generated config and script files while keeping template files tracked
- Fixed config validation to skip checks when stage is `test`

### 01/07/2026
- Added `requirements.txt` and `setup.py` with pinned package versions for reproducible installation
- Added nnU-Net environment variables setup instructions
- Added Python stages for INR upsampling (`stage = run_inr`) and nnU-Net training (`stage = train`)
- Created comprehensive documentation in `docs/` folder:
  - Configuration guide with training and test config details
  - INR upsampling shell script guide
  - nnU-Net training shell script guide
- Updated README to reference documentation files for better organization
- Added `trim_neck.sh` script for neck trimming
- Updated testing documentation with config_test details and test data structure
- Changed default pipeline stage from `prepare_inr` to `prepare`

### 01/04/2026
- Refactored pipeline to support linear execution order
- Added separate `prepare` stage for patch data preparation
- Simplified `execute()` method in preprocessing to remove conditional checks
- Updated pipeline documentation to reflect linear execution flow
- Added `.gitignore` to exclude Python cache files and build artifacts

### 12/24/2025
- Added INR and modified nnUNet as git submodules
- Added INR preparation module with config generation
- Added INR upsampling script template and generation
- Added nnUNet training script template and generation
- Updated README with submodule setup and pipeline documentation

### 12/22/2025
- Updated README.md with comprehensive documentation
- Added data structure documentation slots for atlas and test data

### 11/20/2025
- Added main pipeline of preprocessing

### 10/27/2025
- Initial release of isotropic segmentation pipeline for MTL subregions
- Support for 3T-T2w and 3T-T1w multi-modality MRI

## Contact

For questions or support, please open an issue or contact [liyue3780@gmail.com](mailto:liyue3780@gmail.com).
