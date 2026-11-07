import os
import shutil
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# =========================
# ?
# =========================

new_label_root = os.path.join(DATA_ROOT, "Test1000", "labels")

# =========================
# ?
# =========================

datasets = [
    os.path.join(DATA_ROOT, "Test1000"),
    os.path.join(DATA_ROOT, "Test1000"),
    os.path.join(DATA_ROOT, "Test1000")
]


# =========================
# 
# =========================

def update_labels(dataset_root):

    label_dir = os.path.join(dataset_root, "labels")

    files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    print(f"\nUpdating {dataset_root}")
    print("samples:", len(files))

    for f in tqdm(files):

        src = os.path.join(new_label_root, f)
        dst = os.path.join(label_dir, f)

        if os.path.exists(src):

            shutil.copy2(src, dst)

        else:

            print("missing label:", f)


# =========================
# ?
# =========================

def main():

    for ds in datasets:

        update_labels(ds)

    print("\nAll labels updated.")


if __name__ == "__main__":
    main()


