import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from multimodal_config import DATA_CONFIG, DEVICE, TEST_NAME, TEST_PROJECT, WEIGHTS


def main():
    """Predict on the configured test images and archive the actual output labels."""
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS}. Complete training first.")
    data = check_det_dataset(str(DATA_CONFIG), autodownload=False)
    source = data.get("test") or data["val"]
    model = YOLO(str(WEIGHTS))
    for _ in model(
        source=source,
        stream=True,
        imgsz=1024,
        device=DEVICE,
        project=str(TEST_PROJECT),
        name=TEST_NAME,
        save=False,
        visualize=False,
        save_txt=True,
        save_conf=True,
        save_crop=False,
        conf=0.25,
    ):
        pass

    output_dir = Path(model.predictor.save_dir)
    labels_dir = output_dir / "labels"
    if labels_dir.is_dir():
        archive = shutil.make_archive(
            base_name=str(output_dir / f"{output_dir.name}_labels"),
            format="zip",
            root_dir=str(labels_dir),
        )
        print(f"Archive created: {archive}")
    else:
        print(f"No prediction labels generated: {labels_dir}")


if __name__ == "__main__":
    main()
