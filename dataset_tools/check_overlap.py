import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "Test1000")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def get_image_names(folder):
    return {
        os.path.splitext(name)[0]
        for name in os.listdir(folder)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    }


def get_label_names(folder):
    return {
        os.path.splitext(name)[0]
        for name in os.listdir(folder)
        if name.endswith(".txt")
    }


def main():
    visible = get_image_names(os.path.join(DATA_ROOT, "visible"))
    infrared = get_image_names(os.path.join(DATA_ROOT, "infrared"))
    depth = get_image_names(os.path.join(DATA_ROOT, "depth"))
    labels = get_label_names(os.path.join(DATA_ROOT, "labels"))

    groups = {"visible": visible, "infrared": infrared, "depth": depth, "labels": labels}
    for name, values in groups.items():
        print(f"{name}: {len(values)}")

    reference = visible
    for name, values in groups.items():
        missing = reference - values
        extra = values - reference
        if missing:
            print(f"Missing in {name}: {len(missing)}")
        if extra:
            print(f"Extra in {name}: {len(extra)}")


if __name__ == "__main__":
    main()
