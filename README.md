# Multimodal YOLO11

This project performs object detection using RGB, infrared, and depth images. Data, model, and output paths are relative to the project root.

## Environment Setup

Choose the default CPU installation or the optional NVIDIA GPU installation. The examples pin PyTorch 2.8.0 and torchvision 0.23.0, which support Python 3.9.

### Create a Virtual Environment

Run the following commands in Windows PowerShell:

```powershell
cd D:\FLProjects\MMD_baseline
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

The execution policy change applies only to the current PowerShell session. If `.venv` already exists and is active, skip creating it again.

### Default: CPU Installation

No NVIDIA GPU is required. Install PyTorch explicitly from the CPU package index:

```powershell
python -m pip install torch==2.8.0+cpu torchvision==0.23.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

Set the shared device in `multimodal_config.py` for training, validation, and testing:

```python
DEVICE = "cpu"
```

CPU training is slow. For an initial check, use `epochs=1`, `batch=1`, `imgsz=640`, and `workers=0`.

### Optional: NVIDIA GPU Installation

A supported NVIDIA GPU and compatible driver are required. Check the GPU model and the CUDA version supported by the driver:

```powershell
nvidia-smi
```

The following example uses CUDA 12.6. The `CUDA Version` reported by `nvidia-smi` indicates the highest CUDA version supported by the driver and should be 12.6 or later for this example. The GPU model must also be supported by this PyTorch/CUDA combination.

```powershell
python -m pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
```

If the CPU versions above are already installed, the GPU installation command replaces them. Recreating the virtual environment is unnecessary. A separate CUDA Toolkit installation is usually unnecessary.

For other GPUs or drivers, consult the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and [previous version instructions](https://pytorch.org/get-started/previous-versions/). Choose compatible PyTorch, torchvision, Python, and CUDA versions together; changing only the CUDA index suffix may select an unsupported combination.

After installation, you can check the installed version and GPU availability:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

The GPU example should report `2.8.0+cu126`, `12.6`, and `True`. The CPU example should report `2.8.0+cpu`, `None`, and `False`.

Once the GPU is available, set `DEVICE = "0"` in `multimodal_config.py` for the first GPU. This is the current default; CPU installations must change it to `"cpu"`.

`pip install -e .` installs this project and its remaining dependencies so training uses the local modified `ultralytics` source. It does not automatically select CUDA-enabled PyTorch based on your GPU, so install the desired PyTorch version first.

## Data Layout

```text
data/
|-- Train2000/
|   |-- visible/
|   |-- infrared/
|   |-- depth/
|   `-- labels/
`-- Test1000/
    |-- visible/
    |-- infrared/
    |-- depth/
    `-- labels/
```

Images from all three modalities must have identical filenames, including extensions. Labels must share the same filename stem and use the `.txt` extension. Training and testing index files in `visible`; the loader replaces `visible` with `infrared` and `depth` to locate matching images.

Depth images are min-max normalized and converted to Jet color maps. RGB, infrared, and depth each contribute three channels, producing a nine-channel input.

## Dataset Configuration

The configuration file is `ultralytics/cfg/datasets/multimodal.yaml`:

```yaml
path: .
train: data/Train2000/visible/
val: data/Test1000/visible/
channels: 9
```

Run scripts from the project root. Place `Train2000` and `Test1000` directly under the project's `data` directory.

## Training

The three-modal training entry point is `train_yolo_3modal.py`:

```bash
python train_yolo_3modal.py
```

Before training, the script checks the training and validation data paths. If a path is missing, an image directory is empty, or an image list file is empty, it prints an English message and exits normally without starting training or generating weights. Add the missing data and run the script again. Other problems, such as invalid configuration or corrupt images, still raise errors.

The current settings use 50 epochs, an image size of 1024, and a batch size of 16. Results are saved to:

```text
runs/rgbdt/rgbdt-50epoch/
```

The best weights are saved to:

```text
runs/rgbdt/rgbdt-50epoch/weights/best.pt
```

The model configuration is `ultralytics/cfg/models/multimodal/rgbdt-fusion-yolo11.yaml`. Its first layer, `ThreeModalFusion`, extracts features from each modality and combines them using channel and spatial attention.

`multimodal_config.py` defines the shared device, experiment name, checkpoint path, and absolute output paths. Training refuses an existing experiment directory to prevent overwriting results or silently selecting a numbered directory. Change `RUN_NAME` for a new experiment; validation and testing will then use that experiment's checkpoint automatically. The RGB-only training entry point has been removed.

The corrected spatial attention layer outputs three modality weights. Checkpoints created with the previous implementation are incompatible with the corrected layer; retrain the model before validation or testing.

## Validation

The validation entry point is `val_yolo.py`. It loads the following weights by default:

```text
runs/rgbdt/rgbdt-50epoch/weights/best.pt
```

Validation results are saved under `runs/val/`. The dataset configuration is passed explicitly, with `split="val"` evaluating `Test1000` in the current configuration. This uses the test set for model selection; reserve an independent test set if you need an unbiased final evaluation.

## Testing

The test entry point is `test_yolo.py`:

```bash
python test_yolo.py
```

The script reads `data/Test1000/visible/`, pairs the infrared and depth images, saves predictions in YOLO format with confidence scores, and archives the prediction label directory under:

```text
runs/detect/test/rgbdt-test1000/
```

Testing uses the dataset configuration's `test` path, falling back to `val` when `test` is empty. Configure the device and output name in `multimodal_config.py`. Repeated predictions use automatically numbered output directories, and label archives are created in the actual prediction directory. Predictions are streamed to avoid retaining all nine-channel test images in memory.

## Data Checks

`dataset_tools/check_overlap.py` checks whether files in the `visible`, `infrared`, `depth`, and `labels` directories under `data/Test1000/` correspond to each other.

Other `dataset_tools` scripts also use paths under the project's `data` directory. Scripts that move, delete, or swap files modify the data directly; back up the data before using them.

## Notes

- Training and testing require RGB, infrared, and depth images.
- All three modalities must be spatially aligned and use matching filename stems.
- `ultralytics/data/build.py` currently selects the three-modal dataset, so inputs are processed as nine channels.
- Run scripts from the project root on both Windows and Linux to resolve relative paths correctly.
