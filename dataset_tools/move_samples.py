import os
import shutil
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "")

def count_dataset_samples(root_path):
    """?labels ?""
    label_dir = os.path.join(root_path, "labels")
    if not os.path.exists(label_dir):
        return 0
    return len([f for f in os.listdir(label_dir) if f.endswith('.txt')])


def find_image_file(directory, basename):
    """?""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
    for ext in extensions:
        file_path = os.path.join(directory, f"{basename}{ext}")
        if os.path.exists(file_path):
            return f"{basename}{ext}"
    return None


def safe_distribute_data():
    # --- 1. ?---
    base_train = os.path.join(DATA_ROOT, "Train2000")
    targets = {"test": {"path": os.path.join(DATA_ROOT, "Test1000"), "need": 1000}}
    modalities = ["visible", "infrared", "depth"]

    # --- 2. ?---
    print("=== [ 1/5] ?===")
    print(f"Train2000 ? {count_dataset_samples(base_train)}")
    for k, v in targets.items():
        print(f"Test-{k} ? {count_dataset_samples(v['path'])}")
    print("-" * 40)

    # --- 3.  ---
    label_dir = os.path.join(base_train, "labels")
    all_names = [f.replace(".txt", "") for f in os.listdir(label_dir) if f.endswith(".txt")]

    if len(all_names) < 1000:
        print(f"CRITICAL ERROR: Train ?1000 ?)
        return

    #  1000 ?
    sampled_names = random.sample(all_names, 1000)

    # ?
    name_with_counts = []
    for name in sampled_names:
        l_path = os.path.join(label_dir, f"{name}.txt")
        with open(l_path, 'r') as f:
            count = len([line for line in f if line.strip()])
        name_with_counts.append((name, count))

    # 
    name_with_counts.sort(key=lambda x: x[1])

    # 
    # 0-600 (? -> Easy; 600-900 -> Medium; 900-1000 (? -> Hard
    dist_map = {"test": [x[0] for x in name_with_counts]}

    # --- 4.  ---
    print("=== [ 2/5] ?(? ===")
    for key, names in dist_map.items():
        dest_root = targets[key]['path']
        print(f"?{key}  {len(names)} ?..")

        for name in names:
            # A.  (txt)
            label_src = os.path.join(base_train, "labels", f"{name}.txt")
            label_dst_dir = os.path.join(dest_root, "labels")
            os.makedirs(label_dst_dir, exist_ok=True)
            shutil.move(label_src, os.path.join(label_dst_dir, f"{name}.txt"))

            # B.  ()
            for m in modalities:
                m_src_dir = os.path.join(base_train, m)
                m_dst_dir = os.path.join(dest_root, m)
                os.makedirs(m_dst_dir, exist_ok=True)

                # 
                img_fullname = find_image_file(m_src_dir, name)
                if img_fullname:
                    shutil.move(os.path.join(m_src_dir, img_fullname),
                                os.path.join(m_dst_dir, img_fullname))
                else:
                    print(f"Warning:  {name} ?{m} ?)

    # --- 5. ?---
    print("-" * 40)
    print("=== [ 3/5] ?===")
    print(f"Train2000 ? {count_dataset_samples(base_train)}")
    for k, v in targets.items():
        final_count = count_dataset_samples(v['path'])
        print(f"Test-{k} : {final_count} (: 1000)")
    print("===  ===")


if __name__ == "__main__":
    safe_distribute_data()



