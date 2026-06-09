"""Image preprocessing for Planktoscope ROIs.

Planktoscope ROIs are tight crops with highly variable aspect ratio (round
cells vs. elongated chains) on a light (~245) background. A naive
``Resize((224, 224))`` stretches a chain into a blob and destroys morphology
that is diagnostic for classification. We pad to a square first, then resize,
so shape is preserved.
"""

import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Planktoscope ROIs sit on a light background (measured corner mean ~245).
DEFAULT_FILL = (255, 255, 255)


class PadToSquare:
    """Pad a PIL image to a centered square without resizing.

    The fill should match the ROI background so the padding does not introduce
    a spurious edge the model could key on.
    """

    def __init__(self, fill=DEFAULT_FILL):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        canvas = Image.new(img.mode, (side, side), self.fill)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas


def build_transform(image_size: int = 224, train: bool = False,
                    fill=DEFAULT_FILL) -> T.Compose:
    """Build a preprocessing pipeline.

    Parameters
    ----------
    image_size : int
        Output side length. 224 is a multiple of 14, so it is valid for both
        EfficientNet-B0 and DINOv2 ViT-S/14 (patch size 14).
    train : bool
        When True, add label-preserving augmentation. Plankton have no
        canonical orientation, so flips and full rotations are free augments;
        this is especially valuable for the rare classes.
    """
    steps = [PadToSquare(fill=fill)]
    if train:
        steps += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180, fill=fill),
            T.Resize((image_size, image_size)),
        ]
    else:
        steps += [T.Resize((image_size, image_size))]
    steps += [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(steps)
