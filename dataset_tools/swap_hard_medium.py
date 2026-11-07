import os
import shutil
import random
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

random.seed(0)

# =========================
# 
# =========================

root_easy = os.path.join(DATA_ROOT, "Test1000")
root_medium = os.path.join(DATA_ROOT, "Test1000")
root_hard = os.path.join(DATA_ROOT, "Test1000")

# =========================
# ?
# =========================

MODES = ["visible", "depth", "infrared", "labels"]

IMG_EXT = [".jpg", ".png", ".jpeg", ".bmp"]

# =========================
# 
# =========================

SAMPLE_NUM = 200


# =========================
# basename
# =========================

def get_names(root):

    label_dir = os.path.join(root, "labels")

    names = []

    for f in os.listdir(label_dir):

        if f.endswith(".txt"):
            names.append(os.path.splitext(f)[0])

    return names


# =========================
# 
# =========================

def find_img(path_base):

    for ext in IMG_EXT:
        if os.path.exists(path_base + ext):
            return path_base + ext

    return None


# =========================
# ?
# =========================

def copy_sample(name, src_root, dst_root):

    for mode in MODES:

        src_dir = os.path.join(src_root, mode)
        dst_dir = os.path.join(dst_root, mode)

        os.makedirs(dst_dir, exist_ok=True)

        if mode == "labels":

            src = os.path.join(src_dir, name + ".txt")

            if os.path.exists(src):

                shutil.copy2(src, os.path.join(dst_dir, name + ".txt"))

        else:

            src = find_img(os.path.join(src_dir, name))

            if src:

                shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


# =========================
# 
# =========================

def move_all(src_root, dst_root):

    for mode in MODES:

        src_dir = os.path.join(src_root, mode)
        dst_dir = os.path.join(dst_root, mode)

        os.makedirs(dst_dir, exist_ok=True)

        for f in os.listdir(src_dir):

            shutil.move(
                os.path.join(src_dir, f),
                os.path.join(dst_dir, f)
            )


# =========================
# ?
# =========================

def main():

    hard_names = get_names(root_hard)

    print("hard samples:", len(hard_names))

    # =====================
    # Step1: 200
    # =====================

    sampled = random.sample(hard_names, SAMPLE_NUM)

    print("sampled:", len(sampled))

    for name in tqdm(sampled):

        copy_sample(name, root_hard, root_medium)

    # =====================
    # Step2: medium ?hard
    # =====================

    temp_root = os.path.join(DATA_ROOT, "tmp_swap")

    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)

    os.makedirs(temp_root)

    print("\nSwapping medium and hard...")

    move_all(root_medium, temp_root)
    move_all(root_hard, root_medium)
    move_all(temp_root, root_hard)

    shutil.rmtree(temp_root)

    print("\nDone.")


if __name__ == "__main__":
    main()


