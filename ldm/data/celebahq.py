import os
from PIL import Image
from torchvision.datasets import ImageFolder
import numpy as np


class CelebAHQBase(ImageFolder):
    def __init__(self, datadir, size, transform=None, **kwargs):
        # self.root = root
        self.size = size
        self.transform = transform
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        super().__init__(root=os.path.join(cachedir, datadir), **kwargs)

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        img = np.array(image).astype(np.uint8)

        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[
            (h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2
        ]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size))
        image = np.array(image).astype(np.uint8)

        example = {"image": (image / 127.5 - 1.0).astype(np.float32)}

        return example


class CelebAHQTrain(CelebAHQBase):
    def __init__(self, **kwargs):
        super().__init__(datadir="CelebAHQ", **kwargs)


class CelebAHQTest(CelebAHQBase):
    def __init__(self, **kwargs):
        super().__init__(datadir="CelebAHQ", **kwargs)
