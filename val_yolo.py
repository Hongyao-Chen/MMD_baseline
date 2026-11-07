from ultralytics import YOLO

from multimodal_config import DATA_CONFIG, DEVICE, TEST_NAME, VAL_PROJECT, WEIGHTS


def main():
    """Evaluate the configured checkpoint using the test set mapped to val."""
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS}. Complete training first.")
    model = YOLO(str(WEIGHTS))
    return model.val(
        data=str(DATA_CONFIG),
        split="val",
        device=DEVICE,
        project=str(VAL_PROJECT),
        name=TEST_NAME,
        conf=0.001,
        save_txt=True,
        save_conf=True,
    )


if __name__ == "__main__":
    metrics = main()
