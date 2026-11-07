import os
import shutil
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# ===============================
# 
# ===============================

root_src = os.path.join(DATA_ROOT, "Test1000")

visible_dir = os.path.join(root_src, "visible")
depth_dir = os.path.join(root_src, "depth")
infrared_dir = os.path.join(root_src, "infrared")
label_dir = os.path.join(root_src, "labels")

# ===============================
# 
# ===============================

root_easy = os.path.join(DATA_ROOT, "Test1000")
root_medium = os.path.join(DATA_ROOT, "Test1000")
root_hard = os.path.join(DATA_ROOT, "Test1000")

# ===============================
# txt
# ===============================

easy_txt = os.path.join(PROJECT_ROOT, "easy.txt")
medium_txt = os.path.join(PROJECT_ROOT, "medium.txt")
hard_txt = os.path.join(PROJECT_ROOT, "hard.txt")

# ===============================
# 
# ===============================

IMG_EXT = [".jpg", ".png", ".jpeg", ".bmp"]


# ===============================
# 
# ===============================

def create_dirs(root):

    for d in ["visible", "depth", "infrared", "labels"]:
        os.makedirs(os.path.join(root, d), exist_ok=True)


# ===============================
# 
# ===============================

def find_image(path_base):

    for ext in IMG_EXT:
        if os.path.exists(path_base + ext):
            return path_base + ext

    return None


# ===============================
# txt
# ===============================

def load_list(txt):

    with open(txt) as f:
        names = [line.strip() for line in f if line.strip()]

    return names


# ===============================
# 
# ===============================

def copy_subset(name_list, target_root):

    print(f"\ncopy to {target_root}")

    for name in tqdm(name_list):

        # visible
        src = find_image(os.path.join(visible_dir, name))
        if src:
            dst = os.path.join(target_root, "visible", os.path.basename(src))
            shutil.copy2(src, dst)

        # depth
        src = find_image(os.path.join(depth_dir, name))
        if src:
            dst = os.path.join(target_root, "depth", os.path.basename(src))
            shutil.copy2(src, dst)

        # infrared
        src = find_image(os.path.join(infrared_dir, name))
        if src:
            dst = os.path.join(target_root, "infrared", os.path.basename(src))
            shutil.copy2(src, dst)

        # label
        label_src = os.path.join(label_dir, name + ".txt")

        if os.path.exists(label_src):

            dst = os.path.join(target_root, "labels", name + ".txt")

            shutil.copy2(label_src, dst)


# ===============================
# ?
# ===============================

def main():

    easy = load_list(easy_txt)
    medium = load_list(medium_txt)
    hard = load_list(hard_txt)

    print("split size")
    print("easy:", len(easy))
    print("medium:", len(medium))
    print("hard:", len(hard))

    # 
    create_dirs(root_easy)
    create_dirs(root_medium)
    create_dirs(root_hard)

    # 
    copy_subset(easy, root_easy)
    copy_subset(medium, root_medium)
    copy_subset(hard, root_hard)

    print("\nDone.")


if __name__ == "__main__":
    main()


