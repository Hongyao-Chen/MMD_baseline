"""Shared paths and device settings for the multimodal entry points."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_CONFIG = ROOT / "ultralytics/cfg/datasets/multimodal.yaml"
MODEL_CONFIG = ROOT / "ultralytics/cfg/models/multimodal/rgbdt-fusion-yolo11.yaml"
DEVICE = "0"
RUN_NAME = "rgbdt-50epoch"
TRAIN_PROJECT = ROOT / "runs/rgbdt"
TRAIN_DIR = TRAIN_PROJECT / RUN_NAME
WEIGHTS = TRAIN_DIR / "weights/best.pt"
VAL_PROJECT = ROOT / "runs/val"
TEST_PROJECT = ROOT / "runs/detect/test"
TEST_NAME = "rgbdt-test1000"
