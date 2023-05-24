"""
This script computes the mean and covariance of the InceptionV3 activations on a
given dataset. A separate script should be used for computing the same
statistics on generated samples. To compute the FID, these stats can be combined
using a third script.
"""
import argparse
import os

from omegaconf import OmegaConf
from pytorch_fid.inception import InceptionV3
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fid.cifar10_fid_stats_pytorch_fid import stats_from_dataloader, set_seeds
from ldm.util import instantiate_from_config


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        item = super().__getitem__(i)
        return item["image"]


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_config = OmegaConf.create({ "target": args.dataset })
    if args.data_size:
        OmegaConf.update(dataset_config, "params.config.size", args.data_size)
    dataset = instantiate_from_config(dataset_config)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    inception_model = InceptionV3(normalize_input=False).to(device)
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
    parser.add_argument('--dataset', type=str, help='Path to dataset class')
    parser.add_argument('--data_size', type=int, help='Size of images')
    parser.add_argument('--fid_dir', type=str, help='Directory to store fid stats')
    parser.add_argument('--save_name', type=str, help='File name of fid stats')
    args = parser.parse_args()

    if not args.fid_dir:
        print("--fid_dir not provided, generated stats will not be saved")

    set_seeds(0, 0)

    main(args)