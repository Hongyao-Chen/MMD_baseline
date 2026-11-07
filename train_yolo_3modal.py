from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import DATASETS_DIR, YAML

from multimodal_config import DATA_CONFIG, DEVICE, MODEL_CONFIG, RUN_NAME, TRAIN_DIR, TRAIN_PROJECT


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
    data_file = DATA_CONFIG
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
        device=DEVICE,
        project=str(TRAIN_PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        close_mosaic=10,
        pretrained=True,
        show_labels=True,
        amp=True,
    )


if __name__ == "__main__":
    results = main()
