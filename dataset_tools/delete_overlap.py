import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
root_medium = os.path.join(DATA_ROOT, "Test1000")
root_hard = os.path.join(DATA_ROOT, "Test1000")

MODES = ["visible", "depth", "infrared", "labels"]
IMG_EXT = [".jpg", ".png", ".jpeg", ".bmp"]


# =========================
# ?
# =========================
def get_names(root):

    label_dir = os.path.join(root, "labels")

    names = []

    for f in os.listdir(label_dir):

        if f.endswith(".txt"):
            names.append(os.path.splitext(f)[0])

    return set(names)


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
# ?
# =========================
def remove_sample(name, root):

    for mode in MODES:

        dir_path = os.path.join(root, mode)

        if mode == "labels":

            p = os.path.join(dir_path, name + ".txt")

            if os.path.exists(p):
                os.remove(p)

        else:

            img = find_img(os.path.join(dir_path, name))

            if img:
                os.remove(img)


# =========================
# ?
# =========================
def main():

    medium_names = get_names(root_medium)
    hard_names = get_names(root_hard)

    overlap = medium_names & hard_names

    print("medium:", len(medium_names))
    print("hard:", len(hard_names))
    print("overlap:", len(overlap))

    #  medium 
    for name in overlap:

        remove_sample(name, root_medium)

    print("\nremove finished")

    # ?
    medium_names = get_names(root_medium)
    hard_names = get_names(root_hard)

    print("\nAfter clean:")
    print("medium:", len(medium_names))
    print("hard:", len(hard_names))
    print("overlap:", len(medium_names & hard_names))


if __name__ == "__main__":
    main()


