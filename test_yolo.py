import os
import shutil

from ultralytics import YOLO

task = "detect"
project = "test"
name = "rgbdt-test1000"
output_dir = os.path.join("runs", task, project, name)

if os.path.exists(output_dir):
    raise FileExistsError(f"Output directory already exists: {output_dir}. Change 'name' or remove the old directory.")

model = YOLO("runs/rgbdt/rgbdt-50epoch/weights/best.pt")
result = model(
    source="data/Test1000/visible/",
    imgsz=1024,
    device="3",
    project=project,
    name=name,
    save=False,
    visualize=False,
    save_txt=True,
    save_conf=True,
    save_crop=False,
    conf=0.25,
)

result_labels_path = os.path.join(output_dir, "labels")
zip_file_name = os.path.join(output_dir, f"{name}_labels")
if os.path.exists(result_labels_path):
    print(f"Creating archive: {zip_file_name}.zip ...")
    shutil.make_archive(base_name=zip_file_name, format="zip", root_dir=result_labels_path)
    print(f"Archive created: {os.path.abspath(zip_file_name + '.zip')}")
else:
    print("Labels directory not found.")
