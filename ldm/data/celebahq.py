from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose
import numpy as np

# datapath = "/home/saiyuel/Downloads/celebahq256_imgs/train"

# CelebAHQTrain = ImageFolder(
#     root="/home/saiyuel/Downloads/celebahq256_imgs/train",
#     transform=Compose([Resize(config.size), CenterCrop(config.size), ToTensor()]
#                       )
#
# CelebAHQTest = ImageFolder(
#     root="/home/saiyuel/Downloads/celebahq256_imgs/valid",
#     transform=Compose([Resize(config.size), CenterCrop(config.size), ToTensor()]
#                       )


class CelebAHQBase(ImageFolder):
    def __init__(self, root, size, transform=None, **kwargs):
        # self.root = root
        self.size = size
        self.transform = transform
        super().__init__(
            root=root,
            **kwargs)

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)

        img = np.array(image).astype(np.uint8)

        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)  ## convert array to Image
        if self.size is not None:
            image = image.resize((self.size, self.size))

        image = np.array(image).astype(np.uint8)

        # print('final size is : ', image.shape) final size is :  (32, 32, 3)
        breakpoint()

        example = {"image": (image / 127.5 - 1.0).astype(np.float32)}

        return example


class CelebAHQTrain(CelebAHQBase):
    def __init__(self, **kwargs):
        super().__init__(root="/home/saiyuel/Downloads/celebahq256_imgs/train", **kwargs)


class CelebAHQTest(CelebAHQBase):
    def __init__(self, **kwargs):
        super().__init__(root="/home/saiyuel/Downloads/celebahq256_imgs/valid", **kwargs)



