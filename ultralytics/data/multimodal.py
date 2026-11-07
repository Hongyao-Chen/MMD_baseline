"""Read spatially aligned visible, infrared, and depth images consistently."""

from pathlib import Path

import cv2
import numpy as np


def read_three_modal_image(filename):
    """Load nine BGR channels, replacing only the visible directory component."""
    visible = Path(filename)
    parts = list(visible.parts)
    if "visible" not in parts[:-1]:
        raise ValueError(f"Expected an image inside a 'visible' directory: {visible}")
    index = max(i for i, part in enumerate(parts[:-1]) if part == "visible")
    images = []
    for modality in ("visible", "infrared", "depth"):
        paired = parts.copy()
        paired[index] = modality
        path = Path(*paired)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {modality} image: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED if modality == "depth" else cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode {modality} image: {path}")
        if modality == "depth":
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[..., 0]
            if image.ndim != 2:
                raise ValueError(f"Expected a single-channel depth image: {path}")
            image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            image = cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_JET)
        if images and image.shape[:2] != images[0].shape[:2]:
            raise ValueError(f"Modality image dimensions do not match: {path}")
        images.append(image)
    return np.concatenate(images, axis=-1)
