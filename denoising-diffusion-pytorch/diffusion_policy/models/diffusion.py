"""
Gaussian Diffusion for 1D conditional generation.

This module contains the diffusion process implementation for policy learning,
supporting both DDPM and DDIM sampling.
"""
import math
from functools import partial
from collections import namedtuple
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from einops import reduce

from ..config import DiffusionConfig


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    """Extract values from a based on indices t."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def normalize_to_neg_one_to_one(img):
    """Normalize from [0, 1] to [-1, 1]."""
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    """Unnormalize from [-1, 1] to [0, 1]."""
    return (t + 1) * 0.5


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Linear beta schedule."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(nn.Module):
    """
    Gaussian Diffusion model for 1D conditional generation.

    This implements the diffusion process for policy learning, supporting
    both DDPM and DDIM sampling strategies.
    """

    def __init__(
        self,
        model: nn.Module,
        seq_length: int,
        timesteps: int = 1000,
        sampling_timesteps: Optional[int] = None,
        objective: str = 'pred_noise',
        beta_schedule: str = 'cosine',
        ddim_sampling_eta: float = 0.0,
        auto_normalize: bool = True,
    ):
        super().__init__()

        self.model = model
        self.channels = self.model.out_channels
        self.seq_length = seq_length
        self.objective = objective

        assert objective == 'pred_noise', "Only pred_noise objective is supported"

        # Setup beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = int(timesteps)

        # Sampling parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # Helper function to register buffers as float32
        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Diffusion q(x_t | x_{t-1}) calculations
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior q(x_{t-1} | x_t, x_0) calculations
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        # Loss weight (uniform for pred_noise)
        register_buffer('loss_weight', torch.ones_like(alphas_cumprod / (1 - alphas_cumprod)))

        # Normalization
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """Predict noise from x_t and x_0."""
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute posterior distribution q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        clip_x_start: bool = False,
        rederive_pred_noise: bool = False,
    ) -> ModelPrediction:
        """Get model predictions for noise and x_start."""
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity

        model_output = self.model(x, t, local_cond, global_cond)

        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ):
        """Compute mean and variance for p(x_{t-1} | x_t)."""
        preds = self.model_predictions(x, t, local_cond, global_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ):
        """Sample from p(x_{t-1} | x_t)."""
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times,
            local_cond=local_cond, global_cond=global_cond,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0
        pred = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: tuple,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DDPM sampling loop."""
        batch, device = shape[0], self.betas.device

        x = torch.randn(shape, device=device)
        if local_cond is not None:
            local_cond = local_cond.to(device)
        if global_cond is not None:
            global_cond = global_cond.to(device)

        x_start = None
        for t in reversed(range(0, self.num_timesteps)):
            x, x_start = self.p_sample(x, t, local_cond, global_cond)

        x = self.unnormalize(x)
        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """DDIM sampling loop."""
        batch, device = shape[0], self.betas.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)
        if local_cond is not None:
            local_cond = local_cond.to(device)
        if global_cond is not None:
            global_cond = global_cond.to(device)

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                x, time_cond, local_cond, global_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        x = self.unnormalize(x)
        return x

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from the model."""
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length), local_cond, global_cond)

    @autocast('cuda', enabled=False)
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from q(x_t | x_0)."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute training loss."""
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Add noise to x_start
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict noise
        model_out = self.model(x, t, local_cond, global_cond)

        # Compute loss
        target = noise
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss.mean()

    def forward(
        self,
        actions: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            actions: Action tensor [batch, action_dim, seq_length]
            local_cond: Local conditioning [batch, local_cond_dim, seq_length]
            global_cond: Global conditioning [batch, global_cond_dim]

        Returns:
            Training loss
        """
        b, c, n, device = *actions.shape, actions.device
        assert n == self.seq_length, f'Sequence length must be {self.seq_length}'

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        actions = self.normalize(actions)

        return self.p_losses(actions, t, local_cond, global_cond, *args, **kwargs)

    @classmethod
    def from_config(cls, model: nn.Module, config: DiffusionConfig) -> "GaussianDiffusion1D":
        """Create a GaussianDiffusion1D from a DiffusionConfig."""
        return cls(
            model=model,
            seq_length=config.seq_length,
            timesteps=config.timesteps,
            sampling_timesteps=config.sampling_timesteps,
            objective=config.objective,
            beta_schedule=config.beta_schedule,
            ddim_sampling_eta=config.ddim_sampling_eta,
            auto_normalize=config.auto_normalize,
        )
