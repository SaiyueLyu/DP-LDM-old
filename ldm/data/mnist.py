import os
import numpy as np
from PIL import Image
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

#### check https://pytorch.org/vision/main/_modules/torchvision/datasets/svhn.html#SVHN.__getitem__

class MNISTBase(datasets.MNIST):
    def __init__(self, datadir, size=None, **kwargs):
        self.size = size
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        super().__init__(root=os.path.join(cachedir, datadir), **kwargs)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        img = np.array(image).astype(np.uint8)
        img_three_channels = np.stack((img, img, img), axis=2)

        # print('now it should has three channels with size : ', img_three_channels.shape) now it should has three channels with size :  (28, 28, 3)

        img = img_three_channels


        # img = np.array(image).astype(np.uint8)  ## convert Image to array and assert uint 8
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)  ## convert array to Image
        if self.size is not None:
            image = image.resize((self.size, self.size))

        image = np.array(image).astype(np.uint8)

        # print('final size is : ', image.shape) final size is :  (32, 32, 3)


        example = { "image" : (image / 127.5 - 1.0).astype(np.float32)}
        example["class_label"] = label

        return example



class MNISTTrain(MNISTBase):
    def __init__(self, **kwargs):
        super().__init__(datadir="MNIST_Train", train=True, download=True, **kwargs)


class MNISTVal(MNISTBase):
    def __init__(self, **kwargs):
        super().__init__(datadir="MNIST_Val", train=False, download=True, **kwargs)
