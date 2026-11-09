from pathlib import Path
import random

import yaml

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import DATASETS_DIR, YAML

from multimodal_config import DATA_CONFIG, DEVICE, MODEL_CONFIG, RUN_NAME, TRAIN_DIR, TRAIN_PROJECT


def prepare_train_val_split(data_file: Path, train_ratio: float = 0.7, seed: int = 0) -> Path:
    """Create deterministic train/validation lists from the labeled training source."""
    config = YAML.load(data_file)
    root = Path(config.get("path") or data_file.parent)
    if not root.exists() and not root.is_absolute():
        root = DATASETS_DIR / root

    sources = config["train"]
    sources = sources if isinstance(sources, list) else [sources]
    image_files = []
    for source in sources:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = (root / source_path).resolve()
        if not source_path.exists() and str(source).startswith("../"):
            source_path = (root / str(source)[3:]).resolve()
        if source_path.is_dir():
            image_files.extend(
                path.resolve()
                for path in source_path.rglob("*")
                if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS
            )
        elif source_path.is_file() and source_path.suffix.lower() == ".txt":
            for line in source_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                image_path = Path(line)
                if not image_path.is_absolute():
                    image_path = (source_path.parent / image_path).resolve()
                image_files.append(image_path)
        elif source_path.is_file() and source_path.suffix[1:].lower() in IMG_FORMATS:
            image_files.append(source_path.resolve())

    image_files = sorted(set(image_files))
    if len(image_files) < 2:
        raise ValueError(f"At least two labeled training images are required, found {len(image_files)}.")

    random.Random(seed).shuffle(image_files)
    split_index = min(max(round(len(image_files) * train_ratio), 1), len(image_files) - 1)
    split_dir = data_file.parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_list = split_dir / "multimodal_train_70.txt"
    val_list = split_dir / "multimodal_val_30.txt"
    train_list.write_text("\n".join(path.as_posix() for path in image_files[:split_index]) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(path.as_posix() for path in image_files[split_index:]) + "\n", encoding="utf-8")

    split_config = dict(config)
    split_config["train"] = train_list.resolve().as_posix()
    split_config["val"] = val_list.resolve().as_posix()
    split_config_path = split_dir / "multimodal_70_30.yaml"
    split_config_path.write_text(yaml.safe_dump(split_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return split_config_path


def has_training_data(data_file):
    """Skip training when a configured image directory is missing or empty."""
    config = YAML.load(data_file)
    root = Path(config.get("path") or data_file.parent)
    if not root.exists() and not root.is_absolute():
        root = DATASETS_DIR / root

    for split in ("train", "val"):
        sources = config[split]
        sources = sources if isinstance(sources, list) else [sources]
        if not sources:
            print(f"No {split} data configured. Training skipped; exiting normally.")
            return False
        for source in sources:
            if not source:
                print(f"No {split} data configured. Training skipped; exiting normally.")
                return False
            path = (root / source).resolve()
            if not path.exists() and source.startswith("../"):
                path = (root / source[3:]).resolve()
            if not path.exists():
                print(f"Missing {split} data: {path}. Training skipped; exiting normally.")
                return False
            if path.is_dir() and not any(
                file.is_file() and file.suffix[1:].lower() in IMG_FORMATS for file in path.rglob("*")
            ):
                print(f"No images in {split} directory: {path}. Training skipped; exiting normally.")
                return False
            if path.is_file() and path.suffix.lower() == ".txt" and not path.read_text(encoding="utf-8").strip():
                print(f"Empty {split} image list: {path}. Training skipped; exiting normally.")
                return False
    return True


def main():
    """Check local data before constructing the model and starting training."""
    data_file = prepare_train_val_split(DATA_CONFIG)
    if not has_training_data(data_file):
        return

    if TRAIN_DIR.exists():
        raise FileExistsError(f"Training directory already exists: {TRAIN_DIR}. Change RUN_NAME in multimodal_config.py.")

    model = YOLO(str(MODEL_CONFIG))
    return model.train(
        data=str(data_file),
        epochs=50,
        imgsz=1024,
        batch=16,
        workers=1,
        device=DEVICE,
        project=str(TRAIN_PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        close_mosaic=10,
        pretrained=False,
        plots=False,
        amp=True,
    )


if __name__ == "__main__":
    results = main()
