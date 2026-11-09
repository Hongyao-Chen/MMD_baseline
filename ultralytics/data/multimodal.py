"""Read spatially aligned visible, infrared, and depth images consistently."""

from pathlib import Path

import cv2
import numpy as np


def read_three_modal_image(filename):
    """Load aligned RGB, infrared, and depth data as a seven-channel uint8 image."""
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
                image = image[:, :, 0]
            elif image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.ndim != 2:
                raise ValueError(f"Unsupported depth image shape {image.shape}: {path}")
            if image.dtype != np.uint8:
                if image.dtype == np.uint16:
                    # Convert in place to avoid a full-resolution float32 temporary array.
                    np.right_shift(image, 8, out=image)
                    image = image.astype(np.uint8)
                elif np.issubdtype(image.dtype, np.integer):
                    max_value = np.iinfo(image.dtype).max
                    image //= max(1, max_value // 255)
                    image = np.clip(image, 0, 255).astype(np.uint8)
                else:
                    np.clip(image, 0.0, 255.0, out=image)
                    image = image.astype(np.uint8)
            image = image[..., None]
        images.append(image)
    return np.concatenate(images, axis=-1)
