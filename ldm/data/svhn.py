import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets

# tran_transform = transforms.Compose([
#     transforms.Resize(self.config.data.image_size),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor()
# ])
# test_transform = transforms.Compose([
#     transforms.Resize(self.config.data.image_size),
#     transforms.ToTensor()
# ])


class SVHNBase(datasets.SVHN):
    def __init__(self, datadir="SVHN", size=None, interpolation="bicubic", flip_p=0.5, **kwargs):
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        super().__init__(root=os.path.join(cachedir, datadir), **kwargs)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        img = np.array(image).astype(np.uint8)  ## convert Image to array and assert uint 8
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)  ## convert array to Image
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image) ## does test data need flip?
        image = np.array(image).astype(np.uint8)

        example = {} # as in the origin codes, exmaple is coded as dict
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["class_label"] = label

        # example = ...
        return example

        # image, label = self.data[index], int(self.labels[index])
        # image = np.transpose(image, (1, 2, 0))
        #
        # if self.transform is not None:
        #     transformed = self.transform(image=image)
        #     image = transformed["image"]
        #
        # print('enter svhnbase')
        # # breakpoint()
        # return image, label


class SVHNTrain(SVHNBase):
    def __init__(self, **kwargs):
        # datasets = [
        #     SVHNBase(root="~/data/svhn", split="train", download=True, transform=transform),
        #     # SVHNSearchDataset(root=root, split="extra", download=download, transform=transform),
        # ]
        # print('enter svhntrain')
        super().__init__(datadir="SVHN_Train", split="train", download=True, **kwargs)


class SVHNVal(SVHNBase):
    # def __init__(self, root, download, transform=None):
    def __init__(self, **kwargs):
        # datasets = [
        #     SVHNDataset(root="~/data/svhn", split="test", download=True, transform=transform),
        #     # SVHNSearchDataset(root=root, split="extra", download=download, transform=transform),
        # ]
        super().__init__(datadir="SVHN_Val", split="test", download=True, **kwargs)
