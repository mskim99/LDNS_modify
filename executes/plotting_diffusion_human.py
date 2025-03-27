#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys

# change path to parent directory for paths
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))

import accelerate
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import yaml
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from einops import rearrange

from ldns.networks import AutoEncoder, CountWrapper
from ldns.utils.plotting_utils import *
from ldns.losses import latent_regularizer
from ldns.networks import Denoiser
from diffusers.training_utils import EMAModel
from diffusers.schedulers import DDPMScheduler

import lovely_tensors as lt

lt.monkey_patch()

matplotlib.rc_file("matplotlibrc")
import warnings

# suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)
# suppress all font manager warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
warnings.filterwarnings("ignore", message="findfont: Generic family")


# In[2]:


# load config and model path

cfg_ae = OmegaConf.load("conf/autoencoder-human.yaml")
cfg = OmegaConf.load("conf/diffusion_human.yaml")

cfg.dataset = cfg_ae.dataset


# In[ ]:


from ldns.data.human import get_human_dataloaders

# set seed
torch.manual_seed(cfg.training.random_seed)
np.random.seed(cfg.training.random_seed)

train_dataloader, val_dataloader, test_dataloader = get_human_dataloaders(
    cfg_ae.dataset.datapath,
    batch_size=cfg_ae.training.batch_size,
    shuffle_train=False,  # for eval
)


# In[ ]:


ae_model = AutoEncoder(
    C_in=cfg_ae.model.C_in,
    C=cfg_ae.model.C,
    C_latent=cfg_ae.model.C_latent,
    L=cfg_ae.dataset.max_seqlen,
    num_blocks=cfg_ae.model.num_blocks,
    num_blocks_decoder=cfg_ae.model.get("num_blocks_decoder", cfg_ae.model.num_blocks),
    num_lin_per_mlp=cfg_ae.model.get("num_lin_per_mlp", 2),  # default 2
    bidirectional=cfg_ae.model.get("bidirectional", False),
)

ae_model = CountWrapper(ae_model)
ae_model.load_state_dict(torch.load(f"exp/stored_models/{cfg_ae.exp_name}/model.pt"))

accelerator = accelerate.Accelerator(mixed_precision="no")

ae_model = accelerator.prepare(ae_model)
print(cfg_ae.exp_name)

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


from ldns.data.human import LatentHumanDataset

# create training dataset
latent_dataset_train = LatentHumanDataset(train_dataloader, ae_model, clip=False)

# create validation and test datasets using training set statistics
latent_dataset_val = LatentHumanDataset(
    val_dataloader,
    ae_model,
    latent_means=latent_dataset_train.latent_means,
    latent_stds=latent_dataset_train.latent_stds,
    clip=False,
)
latent_dataset_test = LatentHumanDataset(
    test_dataloader,
    ae_model,
    latent_means=latent_dataset_train.latent_means,
    latent_stds=latent_dataset_train.latent_stds,
    clip=False,
)


# In[ ]:


display(latent_dataset_train[0])
display(latent_dataset_train[1])

element = latent_dataset_train[0]


# In[ ]:


plt.plot(element["latent"][1])
plt.plot(element["mask"][0])

# mask and (padded) latents visualized


# In[9]:


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

num_batches = len(train_latent_dataloader)

# check if signal length is power of 2
if cfg.dataset.max_seqlen & (cfg.dataset.max_seqlen - 1) != 0:
    cfg.training.precision = "no"  # torch.fft doesnt support half if L!=2^x

# prepare the denoiser model and dataset
(
    train_latent_dataloader,
    val_latent_dataloader,
) = accelerator.prepare(
    train_latent_dataloader,
    val_latent_dataloader,
)


# In[10]:


## initialize denoiser

denoiser = Denoiser(
    C_in=cfg.denoiser_model.C_in + 1,  # 1 for mask (length of required latent)
    C=cfg.denoiser_model.C,
    L=cfg.dataset.max_seqlen,
    num_blocks=cfg.denoiser_model.num_blocks,
    bidirectional=cfg.denoiser_model.get("bidirectional", True),
)

denoiser.load_state_dict(torch.load(f"exp/stored_models/{cfg.exp_name}/model.pt"))  # load after training

scheduler = DDPMScheduler(
    num_train_timesteps=cfg.denoiser_model.num_train_timesteps,
    clip_sample=False,
    beta_schedule="linear",
)

# prepare the denoiser model
denoiser = accelerator.prepare(denoiser)


# In[11]:


def sample_spikes_with_mask(denoiser, scheduler, ae, cfg, lengths=None, batch_size=1, device="cuda"):
    """Sample spike trains from the diffusion model with variable length masks.

    Uses DDPM sampling with EMA-averaged model to generate latents, which are then
    decoded to rates and spikes.
    Args:
        denoiser: denoiser model
        scheduler: DDPM scheduler for sampling
        ae: Trained autoencoder for decoding latents to rates
        cfg: Config object with model parameters
        lengths: Optional list/tensor of sequence lengths for masks
        batch_size: Number of samples to generate
        device: Device to run sampling on

    Returns:
        dict containing:
            rates: Generated firing rates
            spikes: Sampled spike trains
            latents: Generated latents
            masks: Generated masks
            mask_lengths: Lengths used for masks
    """
    # start with random noise matching model input shape
    z_t = torch.randn((batch_size, cfg.denoiser_model.C_in, cfg.dataset.max_seqlen)).to(device)

    # generate lengths for masks if not provided
    if lengths is None:
        lengths = torch.linspace(100, 512, batch_size).long().to(device)
    else:
        if isinstance(lengths, int):
            lengths = torch.tensor([lengths] * batch_size).to(device)
        elif isinstance(lengths, list):
            lengths = torch.tensor(lengths).long().to(device)

    # create masks with 1s in center and 0s for padding
    masks = torch.zeros(batch_size, cfg.dataset.max_seqlen).to(device)
    for i, l in enumerate(lengths):
        padding_left = (cfg.dataset.max_seqlen - l) // 2
        masks[i, padding_left : padding_left + l] = 1.0

    masks = masks.unsqueeze(1)

    # get EMA model and prepare for inference
    denoiser.eval()
    scheduler.set_timesteps(cfg.denoiser_model.num_train_timesteps)

    # iteratively denoise using DDPM
    for t in tqdm(scheduler.timesteps, desc="Sampling DDPM (different masks)"):
        with torch.no_grad():
            model_output = denoiser(torch.cat([z_t, masks], dim=1), torch.tensor([t] * batch_size).to(device).long())[
                :, :-1
            ]
        z_t = scheduler.step(model_output, t, z_t, return_dict=False)[0]

    # scale latents back to original range
    z_t = z_t * latent_dataset_train.latent_stds.to(z_t.device) + latent_dataset_train.latent_means.to(z_t.device)

    # decode latents to rates and sample spikes
    with torch.no_grad():
        rates = ae.decode(z_t).cpu()

    spikes = torch.poisson(rates)

    return {
        "rates": rates,
        "spikes": spikes,
        "latents": z_t.cpu(),
        "masks": masks.cpu(),
        "mask_lengths": lengths,
    }


def reconstruct_spikes(model, dataloader):
    """Reconstruct spikes from a trained autoencoder model.

    Args:
        model: Trained autoencoder model
        dataloader: DataLoader containing batches of spike data

    Returns:
        dict: Dictionary containing:
            - latents: Encoded latent vectors (batch_size, latent_dim)
            - spikes: Original spike trains (batch_size, n_channels, seq_len)
            - rec_spikes: Reconstructed spike trains (batch_size, n_channels, seq_len)
            - signal_masks: Masks indicating valid timesteps (batch_size, 1, seq_len)
    """
    # set model to eval mode
    model.eval()

    # initialize lists to store outputs
    latents = []
    spikes = []
    rec_spikes = []
    signal_masks = []

    # process each batch
    for batch in dataloader:
        signal = batch["signal"]
        signal_mask = batch["mask"]

        # get model outputs without gradients
        with torch.no_grad():
            output_rates, z = model(signal)
            z = z.cpu()

        # store outputs
        latents.append(z)
        spikes.append(signal.cpu())
        rec_spikes.append(torch.poisson(output_rates.cpu()) * signal_mask.cpu())
        signal_masks.append(signal_mask.cpu())

    # concatenate all batches
    return {
        "latents": torch.cat(latents, 0),
        "spikes": torch.cat(spikes, 0),
        "rec_spikes": torch.cat(rec_spikes, 0),
        "signal_masks": torch.cat(signal_masks, 0),
    }


# In[12]:


def plot_real_vs_sampled_rates_and_spikes(
    real_rates,
    sampled_rates,
    real_spikes,
    sampled_spikes,
    real_masks,
    sampled_masks,
    batch_idx=0,
):
    """Plot real and sampled rates and spikes side by side for comparison.

    Args:
        real_rates (torch.Tensor): Ground truth firing rates [batch, channels, length]
        sampled_rates (torch.Tensor): Model generated firing rates [batch, channels, length]
        real_spikes (torch.Tensor): Ground truth spike trains [batch, channels, length]
        sampled_spikes (torch.Tensor): Model generated spike trains [batch, channels, length]
        real_masks (torch.Tensor): Masks for real data [batch, 1, length]
        sampled_masks (torch.Tensor): Masks for sampled data [batch, 1, length]
        batch_idx (int): Which batch element to plot. Defaults to 0.
    """
    B, C, L = real_rates.shape

    # create 2x2 subplot with appropriate size
    fig, axs = plt.subplots(2, 2, figsize=cm2inch(12, 8), dpi=300)

    # select single batch element to plot
    real_rates = real_rates[batch_idx]
    sampled_rates = sampled_rates[batch_idx]
    real_spikes = real_spikes[batch_idx]
    sampled_spikes = sampled_spikes[batch_idx]
    real_masks = real_masks[batch_idx]
    sampled_masks = sampled_masks[batch_idx]

    # get indices where mask is 1 (non-padding)
    real_mask_idx_with_1 = torch.arange(real_masks[0].nonzero().flatten().numel())
    sampled_mask_idx_with_1 = sampled_masks[0].nonzero().flatten()

    # plot real rates with colorbar
    im = axs[0, 0].imshow(real_rates[:, real_mask_idx_with_1], cmap="viridis", alpha=1.0, aspect="auto")
    axs[0, 0].set_title("Real rates")
    fig.colorbar(im, ax=axs[0, 0], orientation="vertical", fraction=0.046, pad=0.04)

    # plot sampled rates with colorbar
    im = axs[0, 1].imshow(sampled_rates[:, sampled_mask_idx_with_1], cmap="viridis", alpha=1.0, aspect="auto")
    axs[0, 1].set_title("Sampled rates")
    fig.colorbar(im, ax=axs[0, 1], orientation="vertical", fraction=0.046, pad=0.04)

    # plot real spikes with colorbar
    im = axs[1, 0].imshow(real_spikes[:, real_mask_idx_with_1], cmap="Greys", alpha=1.0, aspect="auto")
    axs[1, 0].set_title("Real spikes")
    fig.colorbar(im, ax=axs[1, 0], orientation="vertical", fraction=0.046, pad=0.04)

    # plot sampled spikes with colorbar
    im = axs[1, 1].imshow(sampled_spikes[:, sampled_mask_idx_with_1], cmap="Greys", alpha=1.0, aspect="auto")
    axs[1, 1].set_title("Sampled spikes")
    fig.colorbar(im, ax=axs[1, 1], orientation="vertical", fraction=0.046, pad=0.04)

    # print shapes for debugging
    print(
        real_rates[:, real_mask_idx_with_1].shape,
        sampled_rates[:, sampled_mask_idx_with_1].shape,
        real_spikes[:, real_mask_idx_with_1].shape,
        sampled_spikes[:, sampled_mask_idx_with_1].shape,
    )

    # remove y-ticks from right plots for cleaner visualization
    for i, ax in enumerate(axs.flatten()):
        if i % 2 != 0:
            ax.set_yticks([])

    fig.tight_layout()
    plt.show()


# ##  Evaluation
# 
# We first generate spikes from the trained model using the `sample_spikes_with_mask` function, and then compare with the ground truth spikes.

# In[ ]:


ret_dict = sample_spikes_with_mask(
    denoiser,
    scheduler,
    ae_model,
    cfg,
    batch_size=train_latent_dataloader.dataset.train_spike_masks[::6].shape[0],  # 1/6 of training data
    lengths=[
        l.sum() for l in train_latent_dataloader.dataset.train_spike_masks[::6, 0]
    ],  # corresponding to 1/6 of training data
    device="cuda",
)


# In[14]:


# get training spikes and masks, using only 1/6 of the data for faster evaluation
(
    train_spikes,
    train_masks,
) = (
    train_latent_dataloader.dataset.train_spikes[::6],  # 1/6 of training data
    train_latent_dataloader.dataset.train_spike_masks[::6],  # 1/6 of training data
)

# trim spikes to only include timesteps where mask is 1
train_spikes_trimmed = []

for i in range(len(train_spikes)):
    # get indices where mask is 1 for this sequence
    nonzero_mask = train_masks[i, 0].nonzero().flatten()
    spike = train_spikes[i]
    # select only timesteps with mask=1
    spike_ = spike[:, nonzero_mask]

    train_spikes_trimmed.append(spike_)


# In[ ]:


# visualizing trial length distribution
fig = plt.figure(figsize=cm2inch(4, 1.5), dpi=300)

# calculate the length of training spikes in seconds
spikes_train_len = np.array([t.shape[-1] for t in train_spikes_trimmed])

# plot histogram of trial lengths, converting to seconds
_1, bins, _2 = plt.hist(spikes_train_len / 50, color="grey", alpha=0.99, bins=40)
plt.yticks([])
plt.xticks([2, 6, 10])
plt.gca().spines["left"].set_visible(False)
plt.xlabel("trial length (s)")

# print average trial length in seconds
print(f"{spikes_train_len.mean()/50}")


# In[16]:


# get spikes and masks from diffusion model output
sampled_spikes, sampled_masks = ret_dict["spikes"], ret_dict["masks"]

# list to store trimmed spike sequences
sampled_spikes_trimmed = []

# iterate through each sequence
for i in range(len(sampled_spikes)):
    # get indices where mask is 1 for this sequence
    nonzero_mask = sampled_masks[i, 0].nonzero().flatten()
    # get spike sequence
    spike = sampled_spikes[i]
    # select only timesteps with mask=1
    spike_ = spike[:, nonzero_mask]

    # append trimmed sequence to list
    sampled_spikes_trimmed.append(spike_)


# In[17]:


# we do the same thing for the autoencoder reconstruction

rec_dict = reconstruct_spikes(ae_model, train_dataloader)

rec_train_spikes = rec_dict["rec_spikes"][::6]
rec_train_spike_masks = rec_dict["signal_masks"][::6]

# list to store trimmed reconstructed spike sequences
rec_train_spikes_trimmed = []

# iterate through each sequence
for i in range(len(rec_train_spikes)):
    # get indices where mask is 1 for this sequence
    nonzero_mask = rec_train_spikes[i, 0].nonzero().flatten()
    # get spike sequence
    spike = sampled_spikes[i]
    # select only timesteps with mask=1
    spike_ = spike[:, nonzero_mask]

    # append trimmed sequence to list
    rec_train_spikes_trimmed.append(spike_)


# In[18]:


# concatenate all trimmed spike sequences along time dimension
# this gives us one long sequence per neuron
sampled_spikes_trimmed_cat = torch.cat(sampled_spikes_trimmed, dim=-1)  # sampled spikes from diffusion model
train_spikes_trimmed_cat = torch.cat(train_spikes_trimmed, dim=-1)  # original training data spikes
rec_train_spikes_trimmed_cat = torch.cat(rec_train_spikes_trimmed, dim=-1)  # reconstructed spikes from autoencoder


# In[19]:


# plotting correlation matrix, compute neuron-neuron correlation

corrcoefs_train = np.corrcoef(train_spikes_trimmed_cat)
corrcoefs_sampled = np.corrcoef(sampled_spikes_trimmed_cat)
corrcoefs_rec = np.corrcoef(rec_train_spikes_trimmed_cat)
np.fill_diagonal(corrcoefs_train, 0)
np.fill_diagonal(corrcoefs_sampled, 0)
np.fill_diagonal(corrcoefs_rec, 0)


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=cm2inch(9.8, 3), dpi=300)

ax[0].axis("off")
ax[1].axis("off")
ax[2].axis("off")

# Get vmin/vmax from corrcoefs_train
vmin = corrcoefs_train.min()
vmax = corrcoefs_train.max()

im = ax[0].imshow(
    corrcoefs_train,
    cmap="Reds",
    vmin=vmin,
    vmax=vmax,
    aspect="auto",
)
cbar = plt.colorbar(
    im,
    ax=ax[0],
    orientation="vertical",
    fraction=0.046,
    pad=0.04,
)

# all colorbars have 0.0 and 0.5 tick labels
cbar.set_ticks([0.0, 0.5])
cbar.set_ticklabels([0.0, 0.5])
im = ax[1].imshow(
    corrcoefs_rec,
    cmap="Reds",
    vmin=vmin,
    vmax=vmax,
    aspect="auto",
)
cbar = plt.colorbar(
    im,
    ax=ax[1],
    orientation="vertical",
    fraction=0.046,
    pad=0.04,
)

# all colorbars have 0.0 and 0.5 tick labels
cbar.set_ticks([0.0, 0.5])
cbar.set_ticklabels([0.0, 0.5])
im = ax[2].imshow(
    corrcoefs_sampled,
    cmap="Reds",
    vmin=vmin,
    vmax=vmax,
    aspect="auto",
)
cbar = plt.colorbar(
    im,
    ax=ax[2],
    orientation="vertical",
    fraction=0.046,
    pad=0.04,
)

ax[0].set_title("gt neuron corr")
ax[1].set_title("ldns (ae) neuron corr")
ax[2].set_title("ldns (diff) neuron corr")

# all colorbars have 0.0 and 0.5 tick labels
cbar.set_ticks([0.0, 0.5])
cbar.set_ticklabels([0.0, 0.5])

fig.tight_layout()
plt.show()


print(
    corrcoefs_sampled.min(),
    corrcoefs_sampled.max(),
    corrcoefs_train.min(),
    corrcoefs_train.max(),
)


# In[ ]:


fig = plt.figure(figsize=cm2inch(4, 4), dpi=300)

# Scatter plot comparing ground truth, diffusion, and autoencoder correlations
plt.scatter(
    corrcoefs_train.flatten()[::1],
    corrcoefs_sampled.flatten()[::1],
    alpha=0.1,
    s=1,
    color="darkblue",
    rasterized=True,
    label="ldns diffusion",
)
plt.scatter(
    corrcoefs_train.flatten()[::1],
    corrcoefs_rec.flatten()[::1],
    alpha=0.1,
    s=1,
    color="darkred",
    rasterized=True,
    label="ldns ae",
)

# Determine global min and max for axis scaling
min_global = min(corrcoefs_train.min(), corrcoefs_sampled.min(), corrcoefs_rec.min())
max_global = max(corrcoefs_train.max(), corrcoefs_sampled.max(), corrcoefs_rec.max())

# Set labels and title
plt.xlabel("gt")
plt.ylabel("ldns")
plt.title("neuron vs neuron correlation")

# Add legend
plt.legend()

# Add diagonal reference line
x = np.linspace(min_global, max_global, 10)
plt.plot(x, x, "k--", zorder=-10, alpha=0.99)


# In[22]:


# for population spike count
summed_spikes_train = np.concatenate([t.sum(0) for t in train_spikes_trimmed])
summed_spikes_sampled = np.concatenate([t.sum(0) for t in sampled_spikes_trimmed])


# In[28]:


# for spike stats (isi), convert to spiketrain. This might be a bit slow.

from ldns.utils.eval_utils import counts_to_spike_trains_ragged

fps = 1000 / 20
spike_trains_train_spiketrain = counts_to_spike_trains_ragged(
    [t.permute(1, 0).numpy() for t in train_spikes_trimmed], fps=fps
)
spike_trains_sampled_spiketrain = counts_to_spike_trains_ragged(
    [t.permute(1, 0).numpy() for t in sampled_spikes_trimmed], fps=fps
)


# In[29]:


# compute spike stats
from ldns.utils.eval_utils import compute_spike_stats_per_neuron
from ldns.utils.plotting_utils import cm2inch

spike_stats_gt = compute_spike_stats_per_neuron(
    spike_trains_train_spiketrain,
    n_samples=len(train_spikes_trimmed),
    n_neurons=train_spikes_trimmed[0].shape[0],
    mean_output=False,
)
spike_stats_sampled = compute_spike_stats_per_neuron(
    spike_trains_sampled_spiketrain,
    n_samples=len(sampled_spikes_trimmed),
    n_neurons=sampled_spikes_trimmed[0].shape[0],
    mean_output=False,
)


# In[ ]:


from scipy.stats import gaussian_kde

# create figure with 4 subplots (pop spike count, neuron-neuron corr, mean isi, std isi)
fig, ax = plt.subplots(2, 2, figsize=cm2inch(9, 8))
ax = ax.flatten()

# clip spike counts to reasonable range
gt_spikes = torch.tensor(summed_spikes_train).clip(0, 220)

# compute kernel density estimates
kde_gt = gaussian_kde(gt_spikes.flatten())
kde_sampled = gaussian_kde(sampled_spikes_trimmed_cat.flatten())

# evaluate densities on common grid
x_grid = np.linspace(0, 220, 220 + 1)

density_model = kde_sampled(x_grid)
density_gt = kde_gt(x_grid)
# normalize densities
density_model /= density_model.sum()
density_gt /= density_gt.sum()

# plot population spike count distributions
bins_psc = np.arange(-0.5, 220 - 0.5, 1)
ax[0].hist(gt_spikes, bins=bins_psc, density=True, alpha=0.5, label="data", color="grey", rasterized=True)
ax[0].hist(
    sampled_spikes_trimmed_cat, bins=bins_psc, density=True, alpha=0.5, label="ldns", color="darkred", rasterized=True
)
ax[0].plot(x_grid, density_gt, ".-", label="data kde", color="black", rasterized=False)
ax[0].plot(x_grid, density_model, ".-", label="ldns kde", color="darkred", rasterized=False)

ax[0].set_xlim(40, 160)
ax[0].set_yticks([])
ax[0].spines["left"].set_visible(False)
ax[0].legend(fontsize=7)
ax[0].set_xlabel("spike count")

# plot correlation structure comparison
C_model = corrcoefs_sampled
C_gt = corrcoefs_train
np.fill_diagonal(C_model, 0)
np.fill_diagonal(C_gt, 0)
C_model = np.tril(C_model, k=-1)
C_gt = np.tril(C_gt, k=-1)

ax[1].plot(C_gt.flatten(), C_model.flatten(), ".", alpha=0.3, color="darkred", ms=2, rasterized=True)
data_limits = [min(C_gt.min(), C_model.min()), max(C_gt.max(), C_model.max())]
ax[1].plot([data_limits[0], data_limits[1]], [data_limits[0], data_limits[1]], "--", color="black")
ax[1].set_xlabel("gt")
ax[1].set_ylabel("ldns")
ax[1].set_aspect("equal")
data_limis_ax = [data_limits[0] - 0.15 * np.abs(data_limits[0]), data_limits[1] + 0.15 * np.abs(data_limits[1])]
ax[1].set_xlim(data_limis_ax)
ax[1].set_ylim(data_limis_ax)

# plot mean ISI comparison
ax[2].plot(
    spike_stats_gt["mean_isi"].flatten(),
    spike_stats_sampled["mean_isi"].flatten(),
    ".",
    alpha=0.5,
    color="darkred",
    ms=2,
    rasterized=True,
)
data_limits = [
    min(spike_stats_gt["mean_isi"].flatten().min(), spike_stats_sampled["mean_isi"].flatten().min()),
    max(spike_stats_gt["mean_isi"].flatten().max(), spike_stats_sampled["mean_isi"].flatten().max()),
]
ax[2].plot([data_limits[0], data_limits[1]], [data_limits[0], data_limits[1]], "--", color="black")
ax[2].set_xlabel("gt mean isi")
ax[2].set_ylabel("ldns mean isi")
ax[2].set_aspect("equal")
data_limis_ax = [data_limits[0] - 0.15 * np.abs(data_limits[0]), data_limits[1] + 0.15 * np.abs(data_limits[1])]
ax[2].set_xlim(data_limis_ax)
ax[2].set_ylim(data_limis_ax)

# plot ISI std comparison
ax[3].plot(
    spike_stats_gt["std_isi"].flatten(),
    spike_stats_sampled["std_isi"].flatten(),
    ".",
    alpha=0.5,
    color="darkred",
    ms=2,
    rasterized=True,
)
data_limits = [
    min(spike_stats_gt["std_isi"].flatten().min(), spike_stats_sampled["std_isi"].flatten().min()),
    max(spike_stats_gt["std_isi"].flatten().max(), spike_stats_sampled["std_isi"].flatten().max()),
]
ax[3].plot([data_limits[0], data_limits[1]], [data_limits[0], data_limits[1]], "--", color="black")
ax[3].set_xlabel("gt std isi")
ax[3].set_ylabel("ldns std isi")
ax[3].set_aspect("equal")
data_limis_ax = [data_limits[0] - 0.15 * np.abs(data_limits[0]), data_limits[1] + 0.15 * np.abs(data_limits[1])]
ax[3].set_xlim(data_limis_ax)
ax[3].set_ylim(data_limis_ax)
ax[3].set_xticks([0.05, 0.15])
ax[3].set_yticks([0.05, 0.15])


# In[ ]:


# create figure with two subplots for comparing correlation matrices
fig, ax = plt.subplots(1, 2, figsize=cm2inch(6.5, 3), dpi=300)

# plot correlation matrix for sampled data
im = ax[0].imshow(corrcoefs_sampled, cmap="Reds", vmin=-0.2, vmax=0.35, aspect="auto")
ax[0].axis("off")
cbar = plt.colorbar(im, ax=ax[0], orientation="vertical", fraction=0.046, pad=0.04)
ax[0].set_title("ldns neuron corr")

# plot correlation matrix for ground truth data
im = ax[1].imshow(corrcoefs_train, cmap="Reds", vmin=-0.2, vmax=0.35, aspect="auto")
cbar = plt.colorbar(im, ax=ax[1], orientation="vertical", fraction=0.046, pad=0.04)
ax[1].set_title("gt neuron corr")
ax[1].axis("off")

# adjust spacing between subplots
fig.tight_layout()


# In[ ]:


# create figure with two subplots for comparing correlation matrices
fig, ax = plt.subplots(1, 2, figsize=cm2inch(6.5, 3), dpi=300)

# plot correlation matrix for sampled data
im = ax[0].imshow(corrcoefs_sampled, cmap="Reds", vmin=-0.2, vmax=0.35, aspect="auto")
ax[0].axis("off")
cbar = plt.colorbar(im, ax=ax[0], orientation="vertical", fraction=0.046, pad=0.04)
ax[0].set_title("ldns neuron corr")

# plot correlation matrix for ground truth data
im = ax[1].imshow(corrcoefs_train, cmap="Reds", vmin=-0.2, vmax=0.35, aspect="auto")
cbar = plt.colorbar(im, ax=ax[1], orientation="vertical", fraction=0.046, pad=0.04)
ax[1].set_title("gt neuron corr")
ax[1].axis("off")

# adjust spacing between subplots
fig.tight_layout()


# In[ ]:


# plot two examples of ground truth spike trains
fig = plt.figure(figsize=cm2inch(6, 2), dpi=300)

# plot first example with consistent vmin/vmax for comparison
vmin, vmax = 0, 5
im = plt.imshow(train_spikes[5, :], cmap="Greys", alpha=1.0, aspect="auto", vmin=vmin, vmax=vmax)

# clean up axes
plt.xticks([])
plt.gca().spines["bottom"].set_visible(False)
plt.yticks([])
plt.ylabel("neurons")
plt.show()

# plot second example
fig = plt.figure(figsize=cm2inch(6, 2), dpi=300)

im = plt.imshow(train_spikes[0, :], cmap="Greys", alpha=1.0, aspect="auto", vmin=vmin, vmax=vmax)

# add time axis labels
plt.xticks([0, 500])
plt.gca().set_xticklabels([0, 10])
plt.xlabel("time (s)")

# clean up y axis
plt.yticks([])
plt.ylabel("neurons")
plt.show()


# In[ ]:


# plot three examples of sampled spike trains
fig = plt.figure(figsize=cm2inch(6, 2), dpi=300)
vmax = 4

# get indices of non-zero mask values for first example
sampled_mask_idx_with_11 = sampled_masks[248, 0].nonzero().flatten()

# pad sampled spikes with zeros to match dimensions
sampled_spikes_padded2 = torch.cat(
    (sampled_spikes[248, :, sampled_mask_idx_with_11], torch.zeros(128, 512 - len(sampled_mask_idx_with_11))), dim=-1
)
print(sampled_spikes_padded2.shape)

# plot first example
plt.imshow(sampled_spikes_padded2, cmap="Greys", alpha=1.0, aspect="auto", vmin=0, vmax=vmax)
plt.xticks([])
plt.gca().spines["bottom"].set_visible(False)
plt.yticks([])
plt.ylabel("neurons")
plt.show()

# plot second example
fig = plt.figure(figsize=cm2inch(6, 2), dpi=300)
sampled_mask_idx_with_12 = sampled_masks[16, 0].nonzero().flatten()
sampled_spikes_padded2 = torch.cat(
    (sampled_spikes[16, :, sampled_mask_idx_with_12], torch.zeros(128, 512 - len(sampled_mask_idx_with_12))), dim=-1
)
plt.imshow(sampled_spikes_padded2, cmap="Greys", alpha=1.0, aspect="auto", vmin=0, vmax=vmax)
plt.xticks([])
plt.gca().spines["bottom"].set_visible(False)
plt.yticks([])
plt.ylabel("neurons")
plt.show()

# plot third example with time axis
fig = plt.figure(figsize=cm2inch(6, 2), dpi=300)
sampled_mask_idx_with_13 = sampled_masks[89, 0].nonzero().flatten()
sampled_spikes_padded2 = torch.cat(
    (sampled_spikes[89, :, sampled_mask_idx_with_13], torch.zeros(128, 512 - len(sampled_mask_idx_with_13))), dim=-1
)
plt.imshow(sampled_spikes_padded2, cmap="Greys", alpha=1.0, aspect="auto", vmin=0, vmax=vmax)
plt.xticks([0, 500])
plt.gca().set_xticklabels([0, 10])
plt.xlabel("time (s)")
plt.yticks([])
plt.ylabel("neurons")
plt.show()


# ### PCA of smoothed spikes, data vs sampled
# 
# We will train PCA on one subset of smoothed spikes and transform the other subset to compare the latent spaces
# 
# We will also compare this to the latent space of sampled spikes from the diffusion model
# 

# In[ ]:


# import required packages for PCA and signal processing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import signal

# define parameters for smoothing window
time_per_bin = 0.02  # 20 ms bin size for spike data
win_len = 8  # window length in number of bins (8 * 20ms = 160ms total)
win_std = 0.06  # standard deviation of gaussian window in seconds
num_bins_std = int(win_std / time_per_bin)  # convert std from seconds to bins

# create gaussian smoothing window
smo_window = signal.windows.gaussian(int(win_len * num_bins_std), num_bins_std)
smo_window /= smo_window.sum()  # normalize window to sum to 1


# In[ ]:


# get ground truth spike data
ground_truth_spikes = train_spikes_trimmed

# convert to numpy arrays for processing
ground_truth_np = [t.numpy() for t in ground_truth_spikes]
print(ground_truth_np[0].shape, ground_truth_np[1].shape, len(ground_truth_np))

# smooth each spike train with gaussian window
smoothed_spikes = []
for spike_train in ground_truth_np:
    # apply convolution along time axis for each neuron
    smoothed_spikes.append(
        np.apply_along_axis(lambda m: np.convolve(m, smo_window, mode="same"), axis=1, arr=spike_train)
    )

print(smoothed_spikes[0].shape, smoothed_spikes[1].shape, len(smoothed_spikes))

# randomly split smoothed data into two halves for cross-validation
split_indices = np.random.choice(len(smoothed_spikes), len(smoothed_spikes) // 2, replace=False)
smoothed_train = [smoothed_spikes[i] for i in split_indices]
# get complementary indices for test set
smoothed_test = [smoothed_spikes[i] for i in range(len(smoothed_spikes)) if i not in split_indices]

print(len(smoothed_train), len(smoothed_test))


# In[ ]:


# train PCA on one subset, transform the other
smoothed_train_reshaped = [rearrange(t, "c l -> l c") for t in smoothed_train]
smoothed_train_concat = np.concatenate(smoothed_train_reshaped, axis=0)
print(smoothed_train_concat.shape)


# In[ ]:


# get number of test samples
num_smoothed_test = len(smoothed_test)

# get length of each test sequence
smoothed_test_lengths = [t.shape[-1] for t in smoothed_test]

# reshape test data from (channels, length) to (length, channels)
smoothed_test_reshaped = [rearrange(t, "c l -> l c") for t in smoothed_test]

# concatenate all test sequences into single array
smoothed_test_concat = np.concatenate(smoothed_test_reshaped, axis=0)
print(smoothed_test_concat.shape)


# In[ ]:


# standardize training data by removing mean and scaling to unit variance
scaler = StandardScaler()
train_data_standardized = scaler.fit_transform(smoothed_train_concat)

# fit PCA to reduce dimensionality to 4 components
pca = PCA(n_components=4)
pca.fit(train_data_standardized)


# In[ ]:


from einops import pack, unpack

# transform ground truth data using fitted PCA and scaler
gt_transformed = pca.transform(scaler.transform(smoothed_test_concat))

# split transformed data back into original sequence lengths
gt_transformed_sequences = []
for i, length in enumerate(np.cumsum(smoothed_test_lengths)):
    seq_start = length - smoothed_test_lengths[i]
    gt_transformed_sequences.append(gt_transformed[seq_start:length])


# In[ ]:


# transform sampled spikes data similar to ground truth data
# convert pytorch tensors to numpy arrays
sampled_spikes_np = [t.numpy() for t in sampled_spikes_trimmed]

# smooth each sequence using convolution with smoothing window
smoothed_sampled = []
for spikes in sampled_spikes_np:
    smoothed_sampled.append(np.apply_along_axis(lambda m: np.convolve(m, smo_window, mode="same"), axis=1, arr=spikes))

# get length of each smoothed sequence
smoothed_sampled_lengths = [t.shape[-1] for t in smoothed_sampled]

# reshape from (channels, length) to (length, channels) like ground truth
smoothed_reshaped = [rearrange(t, "c l -> l c") for t in smoothed_sampled]

# concatenate all sequences into single array
smoothed_sampled_concat = np.concatenate(smoothed_reshaped, axis=0)


# In[ ]:


# standardize smoothed sampled data using fitted scaler
sampled_spikes_trimmed_smoothed_standardized = scaler.transform(smoothed_sampled_concat)

# transform standardized data using fitted PCA
sampled_spikes_trimmed_smoothed_transformed = pca.transform(sampled_spikes_trimmed_smoothed_standardized)

# split transformed data back into original sequence lengths
sampled_transformed = []
for i, length in enumerate(np.cumsum(smoothed_sampled_lengths)):
    sampled_transformed.append(
        sampled_spikes_trimmed_smoothed_transformed[length - smoothed_sampled_lengths[i] : length]
    )


# In[ ]:


# create figure with 2x2 subplots for comparing ground truth vs sampled trajectories in PC space
fig, ax = plt.subplots(2, 2, figsize=cm2inch(6, 10), dpi=300, sharey=True)

# plot every 50th trajectory from ground truth and sampled data
for i in range(len(gt_transformed_sequences) // 50):
    # plot ground truth trajectories in grey
    ax[0, 0].plot(gt_transformed_sequences[i * 50][:, 0], alpha=0.2, color="grey")  # pc1
    ax[1, 0].plot(gt_transformed_sequences[i * 50][:, 1], alpha=0.2, color="grey")  # pc2

    # plot sampled trajectories in red
    ax[0, 1].plot(sampled_transformed[i * 50][:, 0], alpha=0.2, color="#A44A3F")  # pc1
    ax[1, 1].plot(sampled_transformed[i * 50][:, 1], alpha=0.2, color="#A44A3F")  # pc2

# add y-axis labels for principal components
ax[0, 0].set_ylabel("PC1")
ax[1, 0].set_ylabel("PC2")

# add x-axis labels for time
ax[1, 0].set_xlabel("time (s)")
ax[1, 1].set_xlabel("time (s)")

# set consistent x-axis limits and ticks for all subplots
for ax_ in ax.flatten():
    ax_.set_xlim(-10, 500)
    ax_.set_xticks([0, 250, 500])

# remove x-tick labels from top row
for ax_ in ax[0]:
    ax_.set_xticklabels([])

# convert x-ticks from samples to seconds for bottom row
for ax_ in ax[1]:
    ax_.set_xticklabels([0 / 50, 250 / 50, 500 / 50])

# add titles to distinguish ground truth from sampled data
ax[0, 0].set_title("gt")
ax[0, 1].set_title("ldns")

plt.tight_layout()
plt.show()

