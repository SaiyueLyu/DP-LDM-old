# Load data:

Add it (at least fot Imagenet) 

- for CelebAHQ-256, data downloaded [here](https://www.kaggle.com/datasets/denislukovnikov/celebahq256-images-only)

# To run LDM code :

- make sure you installed the required package using `conda env create -f environment.yaml`
- make sure you downloaded the pre-trained autoencoder or LDM
- change the 'ckpt_path` in the corresponding yaml files
- change the `batch_size` you want in the corresponding yaml files
- To train autoencoder, the yaml you want is [configs/autoencoder/svhn_autoencoder_kl_4×4×3.yaml](https://github.com/SaiyueLyu/ldmtrans/blob/saiyue/latent-diffusion-main/configs/autoencoder/svhn_autoencoder_kl_4x4x3.yaml), the command is :
```
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/autoencoder/svhn_autoencoder_kl_4x4x3.yaml -t --gpus 0,
```
- To train LDM, the yaml you want is [configs/latent-diffusion/svhn-ldm-kl.yaml](https://github.com/SaiyueLyu/ldmtrans/blob/saiyue/latent-diffusion-main/configs/latent-diffusion/svhn-ldm-kl.yaml), the command is :
```
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/svhn-ldm-kl.yaml -t --gpus 0,
```
- To train fine tune with MNIST, the yaml you want is [configs/trans-diffusion/mnist-transldm-kl.yaml](https://github.com/SaiyueLyu/ldmtrans/blob/saiyue/latent-diffusion-main/configs/trans-diffusion/mnist-transldm-kl.yaml), the command is :
```
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/trans-diffusion/mnist-transldm-kl.yaml -t --gpus 0,
```
- Uncondition Sampling, the command is :
```
CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py -r models/transldm/mnist/mnistfinal.ckpt -l samples -n 20 --batch_size 8 -c 250 -e 1
```
will generate 20 samples in `samples/mnist/<global_step>/<time>/img`, currently mnistfinal.ckpt is the same as `logs/2023-04-17T20-44-17_mnist-transldm-kl/checkpoints/epoch=000074.ckpt`, the `models/transldm/mnist/config.yaml` is the one you used to train with mnist, which is the same as `configs/trans-diffusion/mnist-transldm-kl.yaml`

The old logs : 
- rgb svhn autoencoder : /home/saiyuel/Downloads/latent-diffusion-main/logs/2023-04-08T16-13-17_svhn_autoencoder_kl_4x4x3/checkpoints/epoch=000011.ckpt
- rgb svhn conditional ldm : /home/saiyuel/Downloads/latent-diffusion-main/logs/2023-04-25T11-51-58_svhn-ldm-condition-kl/checkpoints/epoch=000029.ckpt
- rgb mnist conditonal transldm : /home/saiyuel/Downloads/latent-diffusion-main/logs/2023-04-25T18-20-31_mnist-conditional-transldm-kl/checkpoints/epoch=000162.ckpt

The log to use :
- rgb svhn autoencoder : /home/saiyuel/Downloads/latent-diffusion-main/logs/2023-04-29T22-31-35_svhn_autoencoder_kl_4x4x3/checkpoints/epoch=000015.ckpt or last.ckpt
- rgb svhn conditional ldm : /home/saiyuel/Downloads/latent-diffusion-main/logs/2023-04-30T21-02-49_svhn-ldm-condition-kl
- rgb mnist conditonal transldm :

## Imagenet32 to Cifar10

1. Train autoencoder:

2. Pre-train ldm:

3. Fine-tune:

4. Sample: 
```
python sampling/cond_sampling_test.py  --yaml /home/mvinaroz/Downloads/ldmtrans/latent-diffusion-main/configs/trans-diffusion/cifar10-conditional-transldm-kl.yaml --ckpt_path /home/mvinaroz/Downloads/ldmtrans/latent-diffusion-main/logs/2023-05-03T13-20-04_cifar10-conditional-transldm-kl/checkpoints/epoch=000119.ckpt --num_samples 50000 --classes 0 1 2 3 4 5 6 7 8 9 --batch_size 200

```

# FID computation

## CIFAR10 stats
 - Real data: 
```
 python cifar10_fid_stats.py --fid_dir /home/mvinaroz/Downloads/ldmtrans/latent-diffusion-main/fid/stats
```
 - Generated data:
 
## Imagenet32 stats

 - Real val data: 
```
 python imagenet32_fid_stats.py --fid_dir /home/mvinaroz/Downloads/ldmtrans/latent-diffusion-main/fid/stats
```
