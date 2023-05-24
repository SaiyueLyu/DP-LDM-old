import torch
import numpy as np
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/imagenet32_ldm_kl_8.yaml")  
    model = load_model_from_config(config, "logs/2023-04-12T11-57-04_imagenet32_ldm_kl_8/checkpoints/last.ckpt")
    return model
     


model = get_model()
sampler = DDIMSampler(model)

ddim_steps = 200
ddim_eta = 1.0
scale = 3.0 

num_samples=200




samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=None,
                                             batch_size=num_samples,
                                             shape=[3, 8, 8],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
#                                             unconditional_conditioning=uc, 
                                             eta=ddim_eta)

print("samples_ddim.shape=", samples_ddim.shape)
x_samples_ddim = model.decode_first_stage(samples_ddim)
print("x_samples_ddim.shape=", x_samples_ddim.shape)

x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                         min=0.0, max=1.0)
x_samples_plot=x_samples_ddim[0:18]
print("x_samples_plot.shape=", x_samples_plot.shape)
all_samples = list(x_samples_plot)

grid = torch.stack(all_samples, 0)
#grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=6)

# to image
grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
plotted_imgs=Image.fromarray(grid.astype(np.uint8))
im1 = plotted_imgs.save("test.jpg")
