import os
import shutil
import random
from tqdm import tqdm  #  tqdm ip install tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")


def find_image_file(directory, basename):
    """?""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
    for ext in extensions:
        file_path = os.path.join(directory, f"{basename}{ext}")
        if os.path.exists(file_path):
            return f"{basename}{ext}"
    return None


def copy_sampled_dataset():
    # --- 1.  ---
    src_root = os.path.join(DATA_ROOT, "Train2000")
    dst_root = os.path.join(DATA_ROOT, "Train2000")
    modalities = ["visible", "infrared", "depth"]
    sample_size = 2000

    # --- 2. ?---
    label_src_dir = os.path.join(src_root, "labels")
    if not os.path.exists(label_src_dir):
        print(f":  {label_src_dir}")
        return

    all_names = [f.replace(".txt", "") for f in os.listdir(label_src_dir) if f.endswith(".txt")]
    total_available = len(all_names)

    if total_available < sample_size:
        print(f": ?{total_available}  {sample_size} ?)
        sample_size = total_available

    # --- 3.  ---
    sampled_names = random.sample(all_names, sample_size)
    print(f"?{total_available} ?{sample_size} ?)

    # --- 4.  ---
    print(f": {dst_root} ...")

    # ?
    success_count = 0

    #  tqdm  range?
    for name in tqdm(sampled_names, desc="Copying Samples"):
        # A. 
        l_src = os.path.join(label_src_dir, f"{name}.txt")
        l_dst_dir = os.path.join(dst_root, "labels")
        os.makedirs(l_dst_dir, exist_ok=True)
        shutil.copy2(l_src, os.path.join(l_dst_dir, f"{name}.txt"))

        # B. ?
        sample_complete = True
        for m in modalities:
            m_src_dir = os.path.join(src_root, m)
            m_dst_dir = os.path.join(dst_root, m)
            os.makedirs(m_dst_dir, exist_ok=True)

            img_fullname = find_image_file(m_src_dir, name)
            if img_fullname:
                shutil.copy2(os.path.join(m_src_dir, img_fullname),
                             os.path.join(m_dst_dir, img_fullname))
            else:
                # 
                print(f"\n[]  {name} ? {m}")
                sample_complete = False

        if sample_complete:
            success_count += 1

    # --- 5.  ---
    print("\n" + "=" * 30)
    print(f"?)
    print(f": {sample_size}")
    print(f": {success_count}")
    print(f": {dst_root}")
    print("=" * 30)


if __name__ == "__main__":
    copy_sampled_dataset()

