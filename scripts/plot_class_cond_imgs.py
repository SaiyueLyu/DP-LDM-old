import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image

#from ldm.modules.encoders.modules import ClassEmbedder

#class ClassEmbedder(nn.Module):
#    def __init__(self, embed_dim, n_classes=1000, key='class'):
#        super().__init__()
#        self.key = key
#        self.embedding = nn.Embedding(n_classes, embed_dim)

#    def forward(self, batch, key=None):
#        if key is None:
#            key = self.key
        # this is for use in crossattn
#        c = batch[:, None]
#        print("c in ClassEmbedder=", c)
#        print("c.shape in ClassEmbedder=", c.shape)
#        c = self.embedding(c)
#        return c


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
    config = OmegaConf.load("configs/latent-diffusion/imagenet32_ldm_class_cond_kl_8.yaml")  
    model = load_model_from_config(config, "logs/2023-04-17T13-13-53_imagenet32_ldm_class_cond_kl_8/checkpoints/epoch=000001.ckpt")
    return model
     


model = get_model()
sampler = DDIMSampler(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ddim_steps = 200
ddim_eta = 1.0
scale = 3.0 

num_samples=20
classes=[11,99]
n_samples_per_class=int(num_samples / len(classes))


all_samples = list()

#emb_class=ClassEmbedder(embed_dim=512)

with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
            )
        print("uc=", uc)
        
        for class_label in classes:
            print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor(n_samples_per_class*[class_label])
#            print("xc=", xc)
#            print("xc.shape=", xc.shape)
#            c = emb_class(xc)
#            print("c= ", c)
#            print("c.shape=", c.shape)
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
            
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples_per_class,
                                             shape=[3, 8, 8],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc, 
                                             eta=ddim_eta)
#            print("samples_ddim.shape=", samples_ddim.shape)

#            x_samples_ddim = model.decode_first_stage(samples_ddim)
#            print("x_samples_ddim.shape=", x_samples_ddim.shape)
#            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
#                                         min=0.0, max=1.0)
            
            #all_samples.append(x_samples_ddim)


#print("all_samples=", all_samples)
#print("len(all_samples)=", len(all_samples))



#x_samples_plot=x_samples_ddim[0:18]
#print("x_samples_plot.shape=", x_samples_plot.shape)
#all_samples = list(x_samples_plot)

#grid = torch.stack(all_samples, 0)
#grid = rearrange(grid, 'n b c h w -> (n b) c h w')
#grid = make_grid(grid, nrow=6)

# to image
#grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
#plotted_imgs=Image.fromarray(grid.astype(np.uint8))
#im1 = plotted_imgs.save("test.jpg")
