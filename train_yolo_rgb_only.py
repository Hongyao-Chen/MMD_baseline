from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.graph")  # Ignore selected warnings.

model = YOLO("ultralytics/cfg/models/JNU/jnu-yolo11.yaml")

results = model.train(
    data="ultralytics/cfg/datasets/JNU_MMDet.yaml",
    epochs=50,
    imgsz=1024,
    batch=16,
    device="0",
    project="runs/rgb-only",
    name="rgb-only-1",
    close_mosaic=10,
    pretrained=True,
    show_labels=True,
    amp=True,
)
