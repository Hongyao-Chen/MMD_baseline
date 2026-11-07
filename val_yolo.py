# Evaluate the model on the validation set.
from ultralytics import YOLO

model = YOLO("runs/rgbdt/rgbdt-50epoch/weights/best.pt")
metrics = model.val(
    device="0",
    project="runs/val",
    name="rgbdt-test1000",
    conf=0.25,
    save_txt=True,
    save_conf=True,
)
