# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

This is a YOLOv11 repository extended with attention mechanisms and architectural improvements. It builds on the Ultralytics framework (v8.3.185) and adds 16 attention module variants plus custom building blocks (SPDConv, MSFF, A2C2f, ADown) for YOLO11 object detection models. The primary research focus is small-object detection on the DUO underwater dataset and VOC dataset.

## Development Environment

This project uses a conda environment on Windows:
```bash
conda activate yolov11-attention    # Environment at F:\anaconda3\envs\yolov11-attention
```

## Key Commands

### Installation
```bash
pip install -e .            # Development install (editable mode)
pip install -e ".[dev]"     # With dev dependencies (pytest, mkdocs, etc.)
```

### Training
```bash
# CLI usage
yolo detect train data=coco8.yaml model=yolo11n.yaml epochs=100 imgsz=640
yolo detect train data=coco8.yaml model=yolo11n.pt epochs=10        # Fine-tune from pretrained

# Train with attention modules (SE, CBAM, CA, ECA, etc.)
yolo detect train data=coco8.yaml model=ultralytics/cfg/models/11/Att_yaml/yolo11-SE.yaml epochs=100

# DUO dataset training (small object detection)
yolo detect train model=ultralytics/cfg/models/11/Att_yaml/yolo11-SE.yaml data=DUO.yaml epochs=100

# Common training flags
yolo detect train ... batch=16 imgsz=640 workers=0 device=0          # Windows: set workers=0
yolo detect train ... pretrained=weights/yolo11s.pt cache=True      # Use pretrained weights
yolo detect train ... close_mosaic=10                                # Disable mosaic last N epochs

# Python API
from ultralytics import YOLO
model = YOLO("yolo11n.yaml")
model.train(data="coco8.yaml", epochs=100)
```

### Validation & Prediction
```bash
yolo detect val model=yolo11n.pt data=coco8.yaml
yolo detect predict model=yolo11n.pt source="path/to/images"
```

### Export
```bash
yolo export model=yolo11n.pt format=onnx
```

### Testing
```bash
pytest                           # Run all tests
pytest tests/test_python.py      # Run specific test file
pytest -x tests/test_python.py::test_model_forward  # Run single test
pytest --slow                    # Include slow tests
pytest --doctest-modules         # Run doctests
```

## Architecture

### Core Structure
- `ultralytics/nn/modules/` — Neural network building blocks: Conv, C3k2, SPPF, C2PSA, detection heads, plus custom additions (SPDConv, MSFF, A2C2f, ADown, AConv)
- `ultralytics/nn/Attmodules/` — **Custom attention modules** (16 files, see below)
- `ultralytics/nn/tasks.py` — Model classes and the critical `parse_model()` function that maps YAML to modules
- `ultralytics/engine/` — Core engine: `Model` base class, `Trainer`, `Predictor`, `Validator`, `Exporter`
- `ultralytics/models/yolo/` — Task-specific implementations (detect, segment, classify, pose, obb)
- `ultralytics/cfg/` — Configuration: `default.yaml` (hyperparameters), model YAMLs, dataset definitions
- `ultralytics/data/` — Data loading, augmentation, dataset classes
- `ultralytics/utils/loss.py` — Loss functions (custom MPDIoU loss added in A4 experiment series)

### Model Architecture (YOLO11)
Models are defined in YAML files with `backbone` and `head` sections:
```yaml
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]    # Layer 0
  - [-1, 2, C3k2, [256, False]]  # Layer 1
  ...
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]    # Concatenate with layer 6
  ...
```

Multiple model version directories exist under `ultralytics/cfg/models/`: `v3/`, `v5/`, `v6/`, `v8/`, `v9/`, `v10/`, `11/`, `v12/`, `rt-detr/`. Only `11/` has been extended with custom YAMLs.

### Attention Module Integration (Critical Path)

**Registration in `parse_model()`** (`ultralytics/nn/tasks.py`):

1. **Import** (line 2): `from .Attmodules import *` — imports all attention classes into `tasks.py`'s namespace
2. **Frozenset** (lines 1672-1674): `attn_modules` frozenset enumerates all attention classes — this is how `parse_model()` distinguishes attention modules from other custom modules
3. **Name resolution** (lines 1676-1683): When a YAML module name isn't `nn.*` or `torchvision.ops.*`, it's resolved via `globals()[m]`, which finds attention classes thanks to step 1
4. **Channel injection** (lines 1727-1736):
   - **Standard attention** (SE, CBAM, ECA, CA, SimAM, etc.): `args = [c1, *args]`, output channels `c2 = c1` — identity channel mapping, drop-in anywhere
   - **MSBlock family** (MSBlock, InceptionDWConvBlock, C2f_MSBlock, MSBlock_S): `args = [c1, c2, *args]` — allows channel dimension changes where `c2` comes from YAML `args[0]`

To add a new attention module:
1. Create the class file in `ultralytics/nn/Attmodules/`
2. Add `from .NewModule import *` to `Attmodules/__init__.py`
3. Add the class to the `attn_modules` frozenset in `tasks.py`
4. Create a model YAML referencing it by class name

### Custom Non-Attention Building Blocks

These are integrated directly into `ultralytics/nn/modules/` (block.py, conv.py) and registered in `parse_model()`'s `base_modules` frozenset:

| Module | File | Purpose |
|--------|------|---------|
| `SPDConv` | conv.py:104 | Space-to-depth convolution — replaces strided Conv for small-object detection |
| `MSFF` | conv.py:168 | Multi-Scale Feature Fusion — fuses features across scales |
| `A2C2f` | block.py:1765 | A2-Net attention wrapped inside a C2f block structure |
| `ADown` | block.py:977 | Alternative downsampling block |
| `AConv` | block.py:957 | Attention-augmented convolution |

### Available Attention Modules

**Channel Attention:**
| Class | File | Description |
|-------|------|-------------|
| `SE` | SE.py | Squeeze-and-Excitation |
| `ECA` | ECA.py | Efficient Channel Attention (1D conv, no FC) |
| `SCSA` | SCSA.py | Spatial and Channel Squeeze Attention |

**Spatial Attention:**
| Class | File | Description |
|-------|------|-------------|
| `SimAM` | SimAM.py | Simple, parameter-free attention (energy function) |
| `SLAM` | SLAM.py | Spatial Lightweight Attention Module |
| `ELA` | ELA.py | Efficient Local Attention |

**Hybrid (Channel + Spatial):**
| Class | File | Description |
|-------|------|-------------|
| `CBAM` | CBAM.py | Convolutional Block Attention Module |
| `BAM` | BAM.py | Bottleneck Attention Module |
| `GAM` | GAM.py | Global Attention Module |
| `TripletAttention` | TripletAttention.py | 3-branch rotation-based attention |
| `EMA` | EMA.py | Efficient Multi-scale Attention |

**Coordinate / Position-Aware:**
| Class | File | Description |
|-------|------|-------------|
| `CA` | CA.py | Coordinate Attention (2D positional encoding) |

**Kernel / Multi-Scale:**
| Class | File | Description |
|-------|------|-------------|
| `SK` | SK.py | Selective Kernel (dynamic kernel selection) |
| `ACmix` | ACmix.py | Conv + Self-Attention fusion |

**Multi-Scale Blocks (MS family):**
| Class | File | Description |
|-------|------|-------------|
| `MSBlock` | MS.py | Multi-scale feature extraction block |
| `InceptionDWConvBlock` | MS.py | Inception-style depthwise conv block |
| `C2f_MSBlock` | MS.py | C2f wrapper around MSBlock |
| `MSBlock_S` | MS.py | Smaller variant of MSBlock |

**Double Attention:**
| Class | File | Description |
|-------|------|-------------|
| `A2` | A2.py | A2-Net double attention (second-order pooling) |

All attention modules take `channels` (or `c1`) as their first constructor argument and return a tensor of the same shape as input (identity-mapped), making them drop-in replacements at any backbone position. The MSBlock family is the exception — it can change channel dimensions.

### Attention Model YAMLs

Located in `ultralytics/cfg/models/11/Att_yaml/` (24 YAML files). Key patterns:
- **Standard**: `yolo11-SE.yaml`, `yolo11-CBAM.yaml`, `yolo11-CA.yaml`, etc. — attention at default backbone positions
- **Layer-specific**: `yolo11-SE-layer1.yaml`, `yolo11-SE-layer5.yaml`, `yolo11-SE-layer7.yaml`, `yolo11-SE-layer7-v2.yaml`, `yolo11-SE-layer10.yaml` — attention at specific backbone layers for ablation
- **MS family**: `yolo11-MS.yaml`, `yolo11-MS-full.yaml`, `yolo11-C2f-MS.yaml`
- **Test**: `yolo11-test.yaml` — development/testing config

Additional experiment YAMLs in `ultralytics/cfg/models/11/` (root):
- `yolo11n-spd-a1.yaml`, `yolo11n-spd-a1-p2only.yaml` — SPDConv experiments
- `yolo11n-msff-a2-lite.yaml`, `yolo11n-msff-a2-full.yaml` — MSFF experiments
- `yolo11n-a5-lite-spd-msff.yaml` — combined SPDConv + MSFF

### Task System
Five supported tasks: `detect`, `segment`, `classify`, `pose`, `obb`

Each task has corresponding model, trainer, validator, and predictor classes mapped in `YOLO.task_map()`.

### Scale Variants
Models support scales: n (nano), s (small), m (medium), l (large), x (extra-large) via depth/width/max_channels multipliers in YAML. Scale-specific logic in `parse_model()` adjusts C3k2 and A2C2f configurations.

## CLI Entry Points
- `yolo` or `ultralytics` — Both point to `ultralytics.cfg:entrypoint`
- Modes: `train`, `val`, `predict`, `export`, `track`, `benchmark`
- Tasks: `detect`, `segment`, `classify`, `pose`, `obb`

## Key Training Defaults

From `ultralytics/cfg/default.yaml`:
- `epochs: 100`, `patience: 100` (no early stopping by default)
- `batch: 16`, `imgsz: 640`
- `optimizer: auto` (AdamW for small models, SGD for large)
- `lr0: 0.01`, `lrf: 0.01` (final LR = lr0 × lrf = 1e-4)
- `momentum: 0.937`, `weight_decay: 0.0005`
- `warmup_epochs: 3.0`
- `cos_lr: False`
- `close_mosaic: 10` (disable mosaic for last 10 epochs)
- `amp: True`
- Loss gains: `box: 7.5`, `cls: 0.5`, `dfl: 1.5`
- Augmentation: `mosaic: 1.0`, `mixup: 0.0`, `translate: 0.1`, `scale: 0.5`, `fliplr: 0.5`

Important: On Windows, always set `workers=0` to avoid multiprocessing issues.

## Dataset Scripts
- `duo2yolo.py` — Converts DUO underwater dataset to YOLO format
- `voc_to_yolo_dataset_preparation.py` — Converts VOC dataset to YOLO format
- `DUO/` — DUO underwater dataset directory
- `DUO.yaml` — Dataset config (referenced in training commands)

## Experiment Context

The root directory contains experiment reports (A0-A5) documenting a progressive research pipeline on the DUO underwater dataset:
- **A0**: Baseline YOLO11n (mAP@50=0.849)
- **A1**: SPDConv experiments (space-to-depth replacing strided Conv for small objects)
- **A2**: MSFF experiments (Multi-Scale Feature Fusion)
- **A4**: MPDIoU loss function experiments (code changes in `ultralytics/utils/metrics.py` and `loss.py`)
- **A5**: Combined SPDConv + MSFF Lite

These reports contain training parameters, per-class metrics, and decision criteria for each experiment. Refer to them when working on related modifications.

## Pre-trained Weights
- `weights/` — Contains pre-trained model weights (e.g., `yolo11s.pt`)
- Use `pretrained=weights/yolo11s.pt` for transfer learning from larger models

## Code Style
- Line length: 120 characters (configured in pyproject.toml for ruff, yapf, isort)
- Docstring convention: Google style (pydocstyle)
- Format: `yapf` with PEP8 base style
- Lint/format tools are configured in `pyproject.toml` under `[tool.*]` sections
