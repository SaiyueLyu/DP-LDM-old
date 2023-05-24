import os
import argparse
import torch as pt
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
import numpy as np
from einops import rearrange

from tqdm import tqdm


from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = pt.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def stats_from_dataloader(dataloader, model, device='cpu', save_memory=False):
  """
  Returns:
  -- mu    : The mean over samples of the activations of the pool_3 layer of
             the inception model.
  -- sigma : The covariance matrix of the activations of the pool_3 layer of
             the inception model.
  """
  model.eval()

  pred_list = []
  if not save_memory:  # compute in single pass, store all embeddings
    for batch in tqdm(dataloader):
        x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
        x = x.to(device)
        with pt.no_grad():
            pred = model(x)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = pt.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_list.append(pred)

    pred_arr = np.concatenate(pred_list, axis=0)
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma
  else:  # compute in two passes, no need to store all embeddings
    # first pass: calculate mean
    mu_acc = None
    n_samples = 0
    for batch in tqdm(dataloader):
      x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
      x = x.to(device)


      with pt.no_grad():
        pred = model(x)[0]
      if pred.size(2) != 1 or pred.size(3) != 1:
        pred = pt.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

      n_samples += pred.shape[0]
      pred = pt.sum(pred.squeeze(3).squeeze(2), dim=0)
      mu_acc = mu_acc + pred if mu_acc is not None else pred

    mu = mu_acc / n_samples
    sigma_acc = None
    for batch in tqdm(dataloader):
      x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
      x = x.to(device)
      with pt.no_grad():
        pred = model(x)[0]
      if pred.size(2) != 1 or pred.size(3) != 1:
        pred = pt.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

      pred = pred.squeeze(3).squeeze(2)
      pred_cent = pred - mu
      sigma_batch = pt.matmul(pt.t(pred_cent), pred_cent)
      sigma_acc = sigma_acc + sigma_batch if sigma_acc is not None else sigma_batch

    sigma = sigma_acc / (n_samples - 1)
    return mu.cpu().numpy(), sigma.cpu().numpy()

path_to_ldms = '/scratch/cifar10_ldm_results/ablation_results'

all_models=os.listdir(path_to_ldms)

def main(args):
    device = 'cuda' if pt.cuda.is_available() else 'cpu'

    cur_path=os.getcwd()

    real_stats = np.load(args.path_real_stats)

    ddim_steps = 200
    ddim_eta = 1
    scale = 1
    n_samples_per_class = 500
    batch_size_sampling = 250
    classes=list(np.arange(10))
    bs_inception=128
    iter = int(n_samples_per_class/ batch_size_sampling)

    path_ckpt=os.path.join(args.path_model, "checkpoints")
    yaml_path = os.path.join(path_to_ldms, "configs")
    checkpoints = os.listdir(path_ckpt)
    yaml_list = os.listdir(yaml_path)
    for yaml_item in yaml_list:
            if 'project' in yaml_item:
                yaml_pr = yaml_item
    yaml_pr = os.path.join(yaml_path, yaml_pr)
    config = OmegaConf.load(yaml_pr)
    config['model']['params']['ckpt_path'] = args.path_pre_train_ldm
    for ckpt in checkpoints: 
        if ckpt == 'last.ckpt':
            model = load_model_from_config(config, os.path.join(path_ckpt, ckpt))
            sampler = DDIMSampler(model)

            shape = [model.model.diffusion_model.in_channels,
                model.model.diffusion_model.image_size,
                model.model.diffusion_model.image_size]

            all_samples = list()

            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = pt.tensor(n_samples_per_class * [class_label])

                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

                uc = None
                for idx in range(iter):
                    c_batch = c[ idx*batch_size_sampling: (idx+1)*batch_size_sampling]

                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                     conditioning=c_batch,
                                                     batch_size=batch_size_sampling,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta)
                    
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = pt.clamp((x_samples_ddim+1.0)/2.0,
                                             min=0.0, max=1.0)
                    

                    all_samples.append(x_samples_ddim)

            grid = pt.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')


            synth_dataloader = pt.utils.data.DataLoader(
                grid,
                batch_size=bs_inception,
                shuffle=True,
                drop_last=False,
                num_workers=0
            )

            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            model = InceptionV3([block_idx]).to(device)

            mu, sigma = stats_from_dataloader(synth_dataloader, model, device)

            fid = calculate_frechet_distance(real_stats["mu"], real_stats["sigma"], mu, sigma)

            print("fid =", fid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--path_real_stats', type=str, default='', help='Path directory to real cifar10 stats and file must be in npz format')
    parser.add_argument('--path_model', type=str, default='', help='Path directory of model to evaluate fid')
    parser.add_argument('--path_pre_train_ldm', type=str, default='', help='Path directory to pre-trained ldm checkpoint')
    parser.add_argument('--fid_dir', type=str, default='', help='Directory to store fid stats')
    parser.add_argument('--batch_size',type=int, default=128)
    args = parser.parse_args()


    main(args)