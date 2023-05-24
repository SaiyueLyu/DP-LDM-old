# Enviroment :
- `conda env create -f environment.yaml`
- `conda activate dp-ldm`

# Prepare model checkpoints :
- download the pretrained models (we will update this soon)
- change the 'ckpt_path` in the corresponding yaml files
- change the `batch_size` you want in the corresponding yaml files to fit your machine.

# Pretrain autoencoder (step 1) :
To train autoencoder, the comman is :
```
CUDA_VISIBLE_DEVICES=0 python main.py --base <path to autoencoder yaml> -t --gpus 0,
```

# Pretrain dm (step 2):
To train diffusion model, the command is :
```
CUDA_VISIBLE_DEVICES=0 python main.py --base <path to dm yaml> -t --gpus 0,
```

# Fine Tune with differential privacy constraints (step 3) :
```
CUDA_VISIBLE_DEVICES=0 python main.py --base <path to fine-tune yaml> -t --gpus 0,
```

# Sampling (step 4) :
To have conditional sampling (for MNIST, FMNIST, CIFAR10), the command is :
```
python sampling/cond_sampling_test.py -y <path to the yaml you used> -ckpt <path to the ckpt you want to sample from> -n 10 -c 0 1 2 3 4 -bs 2
```

To have unconditional sampling (for CelebA), the command is :
```
python sampling/unonditional_sampling.py --yaml <path to the yaml you used> --ckpt <path to the ckpt you want to sample from> -num_samples 10 --batch_size 2
```

# Downstreaming accuracy :
For MNIST, to compute the accuracy, the command is :
```
python scripts/dpdm_downstreaming_classifier_mnist.py --train <path of your saved samples for train split> --test <path of your saved samples for test split>
```
We also provide a script to do sampling and accuracy without saving and reading the generated samples for MNIST, so it will generate 50000 training split and 10000 test split, then compute the accuracy, the command is :
```
python scrtips/mnist_sampling_and_acc.py --yaml <path to the yaml you used> --ckpt <path to the ckpt you want to sample from> -bs 800
```


# FID computation

## CIFAR10 stats
 - Real data: 
```
 python cifar10_fid_stats.py --fid_dir <path you want to save your real data stats>
```
 - Generated data:
 
## Imagenet32 stats

 - Real val data: 
```
 python imagenet32_fid_stats.py --fid_dir <path you want to save your real data stats>
```


# Comments
Our code is based on [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion), thanks for open sourcing.
Our code is also based on [Transferring Pretrained Diffusion Probabilistic Models](https://openreview.net/forum?id=8u9eXwu5GAb), thanks the authors for sending the codes before making it public! 
