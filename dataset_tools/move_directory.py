import os
import shutil
import random
from tqdm import tqdm

random.seed(0)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# ?
# =========================

root = os.path.join(PROJECT_ROOT, "data")

train_src = os.path.join(root, "Train2000")
test_src = os.path.join(root, "Test1000")

easy_src = os.path.join(root, "Test1000")
medium_src = os.path.join(root, "Test1000")
hard_src = os.path.join(root, "Test1000")

target_root = root

# =========================
# easy 
# =========================

EASY_SAMPLE = 400

# =========================
# ?
# =========================

MODES = ["visible", "depth", "infrared", "labels"]

IMG_EXT = [".jpg", ".png", ".jpeg", ".bmp"]

# =========================
# 
# =========================

os.makedirs(target_root, exist_ok=True)

# =========================
# 1 ?
# =========================

print("Moving Train2000...")
shutil.move(train_src, os.path.join(target_root, "Train2000"))

print("Moving Test1000...")
shutil.move(test_src, os.path.join(target_root, "Test1000"))

print("Moving Test1000...")
shutil.move(medium_src, os.path.join(target_root, "Test1000"))

print("Moving Test1000...")
shutil.move(hard_src, os.path.join(target_root, "Test1000"))

# =========================
# 2  easy
# =========================

print("\nSampling from Test1000")

easy_labels = os.path.join(easy_src, "labels")

names = [f[:-4] for f in os.listdir(easy_labels) if f.endswith(".txt")]

print("total easy:", len(names))

sampled = random.sample(names, EASY_SAMPLE)

print("sampled:", len(sampled))

#  easy
easy_target = os.path.join(target_root, "Test1000")

for m in MODES:
    os.makedirs(os.path.join(easy_target, m), exist_ok=True)


# =========================
# ?
# =========================

def find_img(path_base):

    for ext in IMG_EXT:
        p = path_base + ext
        if os.path.exists(p):
            return p

    return None


# =========================
# 
# =========================

for name in tqdm(sampled):

    for mode in MODES:

        src_dir = os.path.join(easy_src, mode)
        dst_dir = os.path.join(easy_target, mode)

        if mode == "labels":

            src = os.path.join(src_dir, name + ".txt")

            if os.path.exists(src):
                shutil.move(src, os.path.join(dst_dir, name + ".txt"))

        else:

            img = find_img(os.path.join(src_dir, name))

            if img:
                shutil.move(img, os.path.join(dst_dir, os.path.basename(img)))


print("\nDone.")



