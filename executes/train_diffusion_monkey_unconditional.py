#!/usr/bin/env python
# coding: utf-8

# ## Unconditional Diffusion Training on Monkey Neural Data
# 
# Trains an unconditional diffusion model on latent representations of monkey neural data. The notebook loads a pretrained autoencoder, creates latent datasets from spike data, and trains a denoiser using DDPM. Training includes EMA model updates and periodic sample visualization. Final model is saved for downstream conditional training or evaluation.
# 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# append parent directory to path (../notebooks -> ..)
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))

import accelerate
import lovely_tensors as lt
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import yaml
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from ldns.networks import AutoEncoder, CountWrapper
from ldns.utils.plotting_utils import *
from ldns.networks import Denoiser
from diffusers.training_utils import EMAModel
from diffusers.schedulers import DDPMScheduler

lt.monkey_patch()
matplotlib.rc_file('matplotlibrc') # mackelab plotting style


# In[2]:


## load config and model path

cfg_ae = OmegaConf.load("conf/autoencoder-monkey_z=16.yaml")

cfg_yaml = """
denoiser_model:
  C_in: 16
  C: 256
  kernel: s4
  num_blocks: 6
  bidirectional: True
  num_train_timesteps: 1000
training:
  lr: 0.001
  weight_decay: 0.0
  num_epochs: 2000
  num_warmup_epochs: 50
  batch_size: 512
  random_seed: 42
  precision: bf16
exp_name: diffusion_monkey_unconditional
"""

cfg = OmegaConf.create(yaml.safe_load(cfg_yaml))
cfg.dataset = cfg_ae.dataset


# In[ ]:


from ldns.data.monkey import get_monkey_dataloaders

# initialize autoencoder model    
ae_model = AutoEncoder(
    C_in=cfg_ae.model.C_in,
    C=cfg_ae.model.C,
    C_latent=cfg_ae.model.C_latent,
    L=cfg_ae.dataset.signal_length,
    num_blocks=cfg_ae.model.num_blocks,
    num_blocks_decoder=cfg_ae.model.get("num_blocks_decoder", cfg_ae.model.num_blocks),
    num_lin_per_mlp=cfg_ae.model.get("num_lin_per_mlp", 2),  # default 2
    bidirectional=cfg_ae.model.get("bidirectional", True),
)

ae_model = CountWrapper(ae_model)

# load pretrained autoencoder
ae_model.load_state_dict(torch.load(f"exp/{cfg_ae.exp_name}/model.pt"))

# set random seeds
torch.manual_seed(cfg.training.random_seed)
np.random.seed(cfg.training.random_seed)

# get dataloaders
train_dataloader, val_dataloader, test_dataloader = get_monkey_dataloaders(
        cfg_ae.dataset.task, cfg_ae.dataset.datapath, bin_width=5, batch_size=cfg_ae.training.batch_size
    )

# setup accelerator
accelerator = accelerate.Accelerator(
    mixed_precision=cfg_ae.training.get("precision", "no"),
)

# prepare model and data for training
ae_model = accelerator.prepare(ae_model)

(
    train_dataloader,
    val_dataloader,
    test_dataloader,
) = accelerator.prepare(
    train_dataloader,
    val_dataloader,
    test_dataloader,
)


# In[14]:


save_path = f'exp/{cfg.exp_name}'
os.makedirs(save_path, exist_ok=True)


# In[4]:




accelerator.load_state(f"exp/epoch_140/epoch_140") # best checkpoint in our case, after this, it overfits on val Poisson loss

(
    train_dataloader,
    val_dataloader,
    test_dataloader,
) = accelerator.prepare(
    train_dataloader,
    val_dataloader,
    test_dataloader,
)


# In[ ]:


# Create dataset containing behavior, behavior angle, spike dataset, latents from ae

from ldns.data.monkey import LatentMonkeyDataset

latent_dataset_train = LatentMonkeyDataset(train_dataloader, ae_model, clip=False)

latent_dataset_val = LatentMonkeyDataset(
    val_dataloader,
    ae_model,
    latent_means=latent_dataset_train.latent_means,
    latent_stds=latent_dataset_train.latent_stds,
    clip=False,
)

latent_dataset_test = LatentMonkeyDataset(
    test_dataloader,
    ae_model,
    latent_means=latent_dataset_train.latent_means,
    latent_stds=latent_dataset_train.latent_stds,
    clip=False,
)


# In[16]:


# set up dataloaders for diffusion training

train_latent_dataloader = torch.utils.data.DataLoader(
    latent_dataset_train,
    batch_size=cfg.training.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_latent_dataloader = torch.utils.data.DataLoader(
    latent_dataset_val,
    batch_size=cfg.training.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

test_latent_dataloader = torch.utils.data.DataLoader(
    latent_dataset_test,
    batch_size=cfg.training.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

num_batches = len(train_latent_dataloader)

# check if signal length is power of 2
if cfg.dataset.signal_length & (cfg.dataset.signal_length - 1) != 0:
    cfg.training.precision = "no"  # torch.fft doesnt support half if L!=2^x

# prepare the denoiser model and dataset
(
    train_latent_dataloader,
    val_latent_dataloader,
    test_latent_dataloader,
) = accelerator.prepare(
    train_latent_dataloader,
    val_latent_dataloader,
    test_latent_dataloader,
)


# ## initialize (unconditional) denoiser
# 

# In[ ]:



denoiser = Denoiser(
    C_in=cfg.denoiser_model.C_in,
    C=cfg.denoiser_model.C,
    L=cfg.dataset.signal_length,
    num_blocks=cfg.denoiser_model.num_blocks,
    bidirectional=cfg.denoiser_model.get("bidirectional", True),
)

# initial values may be way off, scaling down the output layer makes training faster
denoiser.conv_out.weight.data = denoiser.conv_out.weight.data * 0.1
denoiser.conv_out.bias.data = denoiser.conv_out.bias.data * 0.1

scheduler = DDPMScheduler(
    num_train_timesteps=cfg.denoiser_model.num_train_timesteps,
    clip_sample=False,
    beta_schedule="linear"
)


optimizer = torch.optim.AdamW(
    denoiser.parameters(), lr=cfg.training.lr
)  # default wd=0.01 for now



num_batches = len(train_latent_dataloader)
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=num_batches * cfg.training.num_warmup_epochs,  # warmup for 10% of epochs
    num_training_steps=num_batches * cfg.training.num_epochs * 1.3,  # total number of steps until 0 lr, we use 1.3 to no go all the way to 0 lr
)

# prepare the denoiser model and dataset
(
    denoiser,
    optimizer,
    lr_scheduler,
) = accelerator.prepare(
    denoiser,
    optimizer,
    lr_scheduler,
)

ema_model = EMAModel(denoiser)


# In[20]:


# some helper functions for checking perf during training of diffusion model

def sample(
    ema_denoiser,
    scheduler,
    cfg,
    batch_size=1,
    generator=None,
    device="cuda",
    signal_length=None
):
    """Sample latents from the diffusion model.

    Args:
        ema_denoiser: EMA model wrapper around denoiser
        scheduler: DDPM noise scheduler
        cfg: Config dictionary containing model parameters
        batch_size: Number of samples to generate
        generator: Random number generator for reproducibility
        device: Device to run sampling on
        signal_length: Length of signal to generate. If None, uses cfg.dataset.signal_length

    Returns:
        Sampled latent tensors of shape (batch_size, C_in, signal_length)
    """
    if signal_length is None:
        signal_length = cfg.dataset.signal_length

    # sample initial noise
    z_t = torch.randn(
        (batch_size, cfg.denoiser_model.C_in, signal_length),
        device=device
    )

    # get averaged ema model
    ema_denoiser_avg = ema_denoiser.averaged_model
    ema_denoiser_avg.eval()

    # set up sampling timesteps
    scheduler.set_timesteps(cfg.denoiser_model.num_train_timesteps)

    # iteratively denoise
    for t in tqdm(scheduler.timesteps, desc="Sampling DDPM"):
        with torch.no_grad():
            model_output = ema_denoiser_avg(
                z_t, 
                torch.tensor([t] * batch_size, device=device).long()
            )
        z_t = scheduler.step(
            model_output, t, z_t, generator=generator, return_dict=False
        )[0]

    return z_t


def sample_spikes(
    ema_denoiser,
    scheduler,
    ae,
    cfg,
    batch_size=1, 
    device="cuda"
):
    """Sample spike trains from the diffusion model.

    Args:
        ema_denoiser: EMA model wrapper around denoiser
        scheduler: DDPM noise scheduler
        ae: Autoencoder model for decoding latents to rates
        cfg: Config dictionary containing model parameters
        batch_size: Number of samples to generate
        device: Device to run sampling on

    Returns:
        Tuple containing:
            - Sampled spike trains of shape (batch_size, C_in, signal_length)
            - Firing rates used to generate spikes
    """
    # sample initial noise
    z_t = torch.randn(
        (batch_size, cfg.denoiser_model.C_in, cfg.dataset.signal_length),
        device=device
    )

    # get averaged ema model
    ema_denoiser_avg = ema_denoiser.averaged_model
    ema_denoiser_avg.eval()

    # set up sampling timesteps
    scheduler.set_timesteps(cfg.denoiser_model.num_train_timesteps)

    # iteratively denoise
    for t in tqdm(scheduler.timesteps, desc="Sampling DDPM"):
        with torch.no_grad():
            model_output = ema_denoiser_avg(
                z_t,
                torch.tensor([t] * batch_size, device=device).long()
            )
        z_t = scheduler.step(model_output, t, z_t, return_dict=False)[0]

    # unnormalize latents
    z_t = z_t * latent_dataset_train.latent_stds.to(z_t.device) +           latent_dataset_train.latent_means.to(z_t.device)

    # decode latents to rates and sample spikes
    with torch.no_grad():
        rates = ae.decode(z_t).cpu()
    spikes = torch.poisson(rates)

    return spikes, rates


# # Diffusion training loop

# In[ ]:


# use smooth l1 loss for faster convergence than mse
loss_fn = torch.nn.SmoothL1Loss(beta=0.04, reduction="mean")

# training loop

# sampled rates will be very high at the beginning,
# then converge to inferred rates during training
# this is expected behavior


# flags for different eval plots
plot_sample_comparison = True
plot_spike_count_dist = False 
plot_per_neuron_dist = False

pbar = tqdm(range(0, cfg.training.num_epochs), desc="epochs")
for epoch in pbar:
    for i, batch in enumerate(train_latent_dataloader):
        denoiser.train()
        optimizer.zero_grad()

        # get batch and add noise
        z = batch["latent"] 
        t = torch.randint(0, cfg.denoiser_model.num_train_timesteps, (z.shape[0],), device="cpu").long()
        noise = torch.randn_like(z)
        noisy_z = scheduler.add_noise(z, noise, t)
        
        # predict noise and compute loss
        noise_pred = denoiser(noisy_z, t)
        loss = loss_fn(noise, noise_pred)

        # optimization step
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        # update progress bar
        if i % 10 == 0:
            pbar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})

        # update ema model
        ema_model.step(denoiser)

    # evaluation and plotting
    if (epoch) % 100 == 0 and plot_sample_comparison:
        denoiser.eval()


        # generate samples
        sampled_latents = sample(ema_denoiser=ema_model, scheduler=scheduler, 
                               cfg=cfg, batch_size=2, device="cuda")
        
        # unnormalize samples
        sampled_latents = sampled_latents * latent_dataset_train.latent_stds.to(sampled_latents.device) +                          latent_dataset_train.latent_means.to(sampled_latents.device)

        # get real samples for comparison
        real_latents = latent_dataset_train.latents[:2].cuda()
        real_latents = real_latents * latent_dataset_train.latent_stds.to(real_latents.device) +                       latent_dataset_train.latent_means.to(real_latents.device)


        # decode latents to rates
        with torch.no_grad():
            sampled_rates = ae_model.decode(sampled_latents).cpu()
            decoded_rates_from_real_latents = ae_model.decode(real_latents).cpu()
        # plot comparison between sampled and real rates
        fig, ax = plt.subplots(1, 2, figsize=(6,4))
        
        # plot sampled rates
        im = ax[0].imshow(sampled_rates[0], aspect="auto")
        ax[0].set_title("sampled rates")
        fig.colorbar(im, ax=ax[0], orientation="vertical", fraction=0.046, pad=0.04)

        # plot real rates
        im = ax[1].imshow(decoded_rates_from_real_latents[0], aspect="auto")
        ax[1].set_title("real rates") 
        fig.colorbar(im, ax=ax[1], orientation="vertical", fraction=0.046, pad=0.04)
        
        fig.tight_layout()
        plt.show()

pbar.close()


# In[ ]:


# save model
torch.save(accelerator.unwrap(ema_model.averaged_model), f"exp/{cfg.exp_name}/model.pt")


# To train conditional diffusion head to `notebooks/train_diffusion_monkey_angle_conditioned.ipynb` and train conditional diffusion head to `notebooks/train_diffusion_monkey_velocity_conditioned.ipynb`.
# 
# To evaluate the model, go to `notebooks/plotting_diffusion_monkey_unconditional.ipynb`.
# 
# To train a spike history model, go to `notebooks/train_with_spike_history_monkey.ipynb`.
