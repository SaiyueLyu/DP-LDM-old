"""
This script computes the mean and covariance of the InceptionV3 activations on a
given dataset. A separate script should be used for computing the same
statistics on generated samples. To compute the FID, these stats can be combined
using a third script.
"""
import argparse
import os

from pytorch_fid.inception import InceptionV3
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fid.cifar10_fid_stats_pytorch_fid import stats_from_dataloader, set_seeds


class DatasetWrapper(Dataset):
    def __init__(self, images):
        self.images = images
    def __len__(self):
        return self.images.shape[0]
    def __getitem__(self, i):
        image = self.images[i]
        assert image.shape[0] == 3 and image.shape[1] == image.shape[2], \
               f"Samples not in CxHxW format, instead got {image.shape}"

        image = image / 2 + 0.5
        image = image.clamp(min=0, max=1)
        return image


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    images = torch.load(args.samples)["image"]
    dataset = DatasetWrapper(images)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    inception_model = InceptionV3().to(device)
    mu, sigma = stats_from_dataloader(dataloader, inception_model, device)

    if args.fid_dir:
        if not args.save_name:
            dataset_name = args.dataset.split('.')[-1]
            args.save_name = "stats_" + dataset_name
        file_path = os.path.join(args.fid_dir, args.save_name + '.npz')
        np.savez(file_path, mu=mu, sigma=sigma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size per GPU')
    parser.add_argument('--samples', type=str, help='Path to samples class')
    parser.add_argument('--fid_dir', type=str, help='Directory to store fid stats')
    parser.add_argument('--save_name', type=str, help='File name of fid stats')
    args = parser.parse_args()

    if not args.fid_dir:
        print("--fid_dir not provided, generated stats will not be saved")

    set_seeds(0, 0)

    main(args)