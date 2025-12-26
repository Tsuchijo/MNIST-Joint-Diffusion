"""
MNIST Classification via Joint Diffusion of Images and Labels

This model learns P(image, label) by diffusing both modalities jointly with
cross-attention between them. At inference time, we hold the image constant
and only denoise the label, effectively sampling from P(label | image) by
marginalizing over the prior as described in the theoretical framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ============================================================================
# Sinusoidal Time Embeddings
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embeddings for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ============================================================================
# Shared Embedding Processor
# ============================================================================

class SharedProcessor(nn.Module):
    """
    Processes concatenated embeddings from both modalities.
    Allows information sharing through shared layers.
    """
    def __init__(self, hidden_dim=256, num_layers=3):
        super().__init__()

        # Process concatenated embeddings (2 * hidden_dim input)
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.SiLU(),
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.SiLU(),
                ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, hidden_dim * 2) - concatenated embeddings
        Returns:
            (B, hidden_dim * 2) - processed shared embeddings
        """
        return self.layers(x) + x  # Residual connection


# ============================================================================
# Image Denoiser Path
# ============================================================================

class ImageDenoiser(nn.Module):
    """
    Processes noisy images and produces a fixed-size embedding vector.
    """
    def __init__(self, time_dim=64, hidden_dim=256):
        super().__init__()

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input convolution (1 -> 64 channels)
        self.input_conv = nn.Conv2d(1, 64, 3, padding=1)

        # Early conv layers (28x28 -> 14x14)
        self.early_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # Downsample
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        # Middle conv layers (14x14 -> 7x7)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # Downsample
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )

        # Global pooling and projection to embedding
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.to_embedding = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, t):
        """
        Args:
            x: (B, 1, 28, 28) - noisy images
            t: (B,) - timesteps
        Returns:
            (B, hidden_dim) - image embedding
        """
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)  # (B, hidden_dim)

        # Process image through convolutions
        h = F.silu(self.input_conv(x))
        h = self.early_conv(h)
        h = self.middle_conv(h)  # (B, 256, 7, 7)

        # Global average pooling
        h = self.pool(h).squeeze(-1).squeeze(-1)  # (B, 256)

        # Project to embedding and add time
        h = self.to_embedding(h)  # (B, hidden_dim)
        h = h + t_emb

        return h


class ImageReconstructor(nn.Module):
    """
    Reconstructs image from shared embeddings.
    Predicts the noise that was added to the image.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()

        # Project from shared embedding to spatial features
        self.from_shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256 * 7 * 7),
            nn.SiLU(),
        )

        # Upsampling path (7x7 -> 14x14 -> 28x28)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )

        # Output projection
        self.output_conv = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, h):
        """
        Args:
            h: (B, hidden_dim * 2) - shared embeddings
        Returns:
            (B, 1, 28, 28) - predicted noise in image space
        """
        B = h.size(0)

        # Project to spatial features
        h = self.from_shared(h)  # (B, 256 * 7 * 7)
        h = h.reshape(B, 256, 7, 7)

        # Upsample
        h = self.up1(h)  # (B, 128, 14, 14)
        h = self.up2(h)  # (B, 64, 28, 28)

        # Output
        return self.output_conv(h)  # (B, 1, 28, 28)


# ============================================================================
# Label Denoiser Path
# ============================================================================

class LabelDenoiser(nn.Module):
    """
    Processes noisy label vectors and produces a fixed-size embedding.
    """
    def __init__(self, label_dim=10, time_dim=64, hidden_dim=256):
        super().__init__()

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(label_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Processing layers
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x, t):
        """
        Args:
            x: (B, 10) - noisy label vectors
            t: (B,) - timesteps
        Returns:
            (B, hidden_dim) - label embedding
        """
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)  # (B, hidden_dim)

        # Project label
        h = self.input_proj(x)  # (B, hidden_dim)

        # Add time embedding
        h = h + t_emb

        # Process
        h = self.layers(h)

        return h


class LabelReconstructor(nn.Module):
    """
    Reconstructs label vector from shared embeddings.
    Predicts the noise that was added to the label.
    """
    def __init__(self, label_dim=10, hidden_dim=256):
        super().__init__()

        # Process shared embeddings
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Output projection to label space
        self.output_proj = nn.Linear(hidden_dim, label_dim)

    def forward(self, h):
        """
        Args:
            h: (B, hidden_dim * 2) - shared embeddings
        Returns:
            (B, 10) - predicted noise in label space
        """
        h = self.layers(h)
        return self.output_proj(h)  # (B, 10)


# ============================================================================
# Joint Diffusion Model
# ============================================================================

class JointDenoiser(nn.Module):
    """
    Joint denoiser that processes both images and labels by concatenating
    their embeddings and processing through shared layers.
    """
    def __init__(self, time_dim=64, hidden_dim=256, num_shared_layers=3):
        super().__init__()

        # Separate encoders for each modality
        self.image_denoiser = ImageDenoiser(time_dim, hidden_dim)
        self.label_denoiser = LabelDenoiser(label_dim=10, time_dim=time_dim, hidden_dim=hidden_dim)

        # Shared processor for concatenated embeddings
        self.shared_processor = SharedProcessor(hidden_dim, num_layers=num_shared_layers)

        # Separate decoders for each modality
        self.image_reconstructor = ImageReconstructor(hidden_dim)
        self.label_reconstructor = LabelReconstructor(label_dim=10, hidden_dim=hidden_dim)

    def forward(self, img_noisy, label_noisy, t):
        """
        Args:
            img_noisy: (B, 1, 28, 28) - noisy images
            label_noisy: (B, 10) - noisy label vectors
            t: (B,) - timesteps
        Returns:
            predicted_img_noise: (B, 1, 28, 28)
            predicted_label_noise: (B, 10)
        """
        # Get embeddings from each modality
        img_emb = self.image_denoiser(img_noisy, t)  # (B, hidden_dim)
        label_emb = self.label_denoiser(label_noisy, t)  # (B, hidden_dim)

        # Concatenate embeddings
        shared_emb = torch.cat([img_emb, label_emb], dim=-1)  # (B, hidden_dim * 2)

        # Process through shared layers
        shared_emb = self.shared_processor(shared_emb)  # (B, hidden_dim * 2)

        # Decode to noise predictions
        predicted_img_noise = self.image_reconstructor(shared_emb)
        predicted_label_noise = self.label_reconstructor(shared_emb)

        return predicted_img_noise, predicted_label_noise


# ============================================================================
# Complete Joint Diffusion Model
# ============================================================================

class JointDiffusion(nn.Module):
    """
    Complete joint diffusion model for P(image, label).
    During training: diffuses both modalities jointly
    During inference: holds image constant, only denoises label to sample from P(label|image)
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()

        self.num_timesteps = num_timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))

        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # Joint denoiser network
        self.denoiser = JointDenoiser(time_dim=64, hidden_dim=128, num_shared_layers=3)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        # Handle different shapes (images vs labels)
        while len(sqrt_alpha.shape) < len(x_0.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def p_losses(self, images, labels):
        """
        Compute training loss with conditional diffusion support.
        Randomly trains on three scenarios:
        1. Both modalities noised (joint diffusion)
        2. Image noised, label clean (P(image|label))
        3. Image clean, label noised (P(label|image) - classification)

        Args:
            images: (B, 1, 28, 28) MNIST images (normalized)
            labels: (B,) integer labels
        """
        batch_size = images.size(0)
        device = images.device

        # Prepare label vectors: convert to one-hot and scale to [-1, 1]
        label_vectors = F.one_hot(labels, num_classes=10).float()
        label_vectors = label_vectors * 2 - 1

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Sample noise for both modalities
        img_noise = torch.randn_like(images)
        label_noise = torch.randn_like(label_vectors)

        # Randomly choose training mode: 0=both noised, 1=img noised, 2=label noised
        mode = torch.randint(0, 3, (1,)).item()

        if mode == 0:
            # Both modalities noised (joint diffusion)
            img_noisy = self.q_sample(images, t, img_noise)
            label_noisy = self.q_sample(label_vectors, t, label_noise)
            pred_img_noise, pred_label_noise = self.denoiser(img_noisy, label_noisy, t.float())
            img_loss = F.mse_loss(pred_img_noise, img_noise)
            label_loss = F.mse_loss(pred_label_noise, label_noise)
            total_loss = img_loss + label_loss

        elif mode == 1:
            # Image noised, label clean (P(image|label))
            img_noisy = self.q_sample(images, t, img_noise)
            label_noisy = label_vectors  # Clean label
            pred_img_noise, pred_label_noise = self.denoiser(img_noisy, label_noisy, t.float())
            img_loss = F.mse_loss(pred_img_noise, img_noise)
            label_loss = torch.tensor(0.0, device=device)  # No label loss
            total_loss = img_loss

        else:  # mode == 2
            # Image clean, label noised (P(label|image) - classification)
            img_noisy = images  # Clean image
            label_noisy = self.q_sample(label_vectors, t, label_noise)
            pred_img_noise, pred_label_noise = self.denoiser(img_noisy, label_noisy, t.float())
            img_loss = torch.tensor(0.0, device=device)  # No image loss
            label_loss = F.mse_loss(pred_label_noise, label_noise)
            total_loss = label_loss

        return total_loss, img_loss.item(), label_loss.item()

    @torch.no_grad()
    def p_sample_step(self, img_t, label_t, t, denoise_image=True, denoise_label=True):
        """
        Single denoising step.

        Args:
            img_t: (B, 1, 28, 28) - noisy image
            label_t: (B, 10) - noisy label
            t: (B,) - timesteps
            denoise_image: whether to denoise the image (False for classification)
            denoise_label: whether to denoise the label
        """
        # Predict noise
        pred_img_noise, pred_label_noise = self.denoiser(img_t, label_t, t.float())

        # Get coefficients for this timestep
        t_idx = t[0].long()

        beta_t = self.betas[t_idx]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_idx]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_idx]

        # Denoise image if requested
        if denoise_image:
            img_mean = sqrt_recip_alpha_t * (
                img_t - beta_t / sqrt_one_minus_alpha_cumprod_t * pred_img_noise
            )
            if t_idx > 0:
                img_noise = torch.randn_like(img_t)
                posterior_variance_t = self.posterior_variance[t_idx]
                img_next = img_mean + torch.sqrt(posterior_variance_t) * img_noise
            else:
                img_next = img_mean
        else:
            img_next = img_t  # Keep image frozen

        # Denoise label if requested
        if denoise_label:
            label_mean = sqrt_recip_alpha_t * (
                label_t - beta_t / sqrt_one_minus_alpha_cumprod_t * pred_label_noise
            )
            if t_idx > 0:
                label_noise = torch.randn_like(label_t)
                posterior_variance_t = self.posterior_variance[t_idx]
                label_next = label_mean + torch.sqrt(posterior_variance_t) * label_noise
            else:
                label_next = label_mean
        else:
            label_next = label_t  # Keep label frozen

        return img_next, label_next

    @torch.no_grad()
    def sample_joint(self, batch_size, device, num_inference_steps=50):
        """
        Sample from the joint distribution P(image, label).
        Starts from pure noise for both modalities.
        """
        # Start from pure noise
        img = torch.randn(batch_size, 1, 28, 28, device=device)
        label = torch.randn(batch_size, 10, device=device)

        # Create timestep schedule
        step_size = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device)
            img, label = self.p_sample_step(img, label, t_batch,
                                           denoise_image=True, denoise_label=True)

        # Convert label to probabilities using softmax
        label = F.softmax(label, dim=-1)

        return img, label

    @torch.no_grad()
    def classify(self, images, num_inference_steps=50):
        """
        Classify images by holding them constant and denoising only labels.
        This implements P(label | image) via marginalization over the prior.
        """
        device = images.device
        batch_size = images.size(0)

        # Start from pure noise in label space
        label = torch.randn(batch_size, 10, device=device)

        # Hold image constant (no noise added)
        img = images

        # Create timestep schedule
        step_size = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        # Denoise only the label, keeping image fixed
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device)
            img, label = self.p_sample_step(img, label, t_batch,
                                           denoise_image=False,  # Keep image frozen
                                           denoise_label=True)   # Only denoise label

        # Convert to probabilities using softmax
        label = F.softmax(label, dim=-1)

        return label

    @torch.no_grad()
    def predict(self, images, num_inference_steps=50):
        """Get class predictions."""
        probs = self.classify(images, num_inference_steps)
        return probs.argmax(dim=-1)


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_classification(model, dataloader, device, num_samples=16, num_inference_steps=50):
    """
    Visualize classification results: show images with predicted vs true labels.
    """
    model.eval()

    # Get a batch of images
    images, labels = next(iter(dataloader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    # Get predictions
    with torch.no_grad():
        probs = model.classify(images, num_inference_steps)
        predictions = probs.argmax(dim=-1)

    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].cpu().numpy()[0]
        pred = predictions[i].item()
        true = labels[i].item()
        conf = probs[i, pred].item()

        axes[i].imshow(img, cmap='gray')
        color = 'green' if pred == true else 'red'
        axes[i].set_title(f'Pred: {pred} (True: {true})\nConf: {conf:.2f}',
                         color=color, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def visualize_joint_generation(model, device, num_samples=16, num_inference_steps=50):
    """
    Generate samples from the joint distribution P(image, label).
    Shows what the model has learned about the joint space.
    """
    model.eval()

    with torch.no_grad():
        images, label_probs = model.sample_joint(num_samples, device, num_inference_steps)
        labels = label_probs.argmax(dim=-1)

    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].cpu().numpy()[0]
        label = labels[i].item()
        conf = label_probs[i, label].item()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Generated: {label}\nConf: {conf:.2f}', fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def visualize_conditional_generation(model, dataloader, device, num_samples=8, num_inference_steps=50):
    """
    Visualize conditional generation in both directions:
    1. Given image -> generate label (classification)
    2. Show the denoising process for labels
    """
    model.eval()

    # Get real images
    images, true_labels = next(iter(dataloader))
    images, true_labels = images[:num_samples].to(device), true_labels[:num_samples].to(device)

    # Classify (denoise labels while holding images constant)
    with torch.no_grad():
        label_probs = model.classify(images, num_inference_steps)
        pred_labels = label_probs.argmax(dim=-1)

    # Create figure showing image and its predicted label distribution
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 4))

    for i in range(num_samples):
        # Show image
        img = images[i].cpu().numpy()[0]
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'True: {true_labels[i].item()}', fontsize=9)
        axes[0, i].axis('off')

        # Show predicted label distribution
        probs = label_probs[i].cpu().numpy()
        axes[1, i].bar(range(10), probs, color='steelblue')
        axes[1, i].set_ylim([0, 1])
        axes[1, i].set_xlabel('Digit', fontsize=8)
        axes[1, i].set_ylabel('Prob', fontsize=8)
        axes[1, i].set_title(f'Pred: {pred_labels[i].item()}', fontsize=9)
        axes[1, i].tick_params(labelsize=7)

    plt.tight_layout()
    return fig


def visualize_denoising_process(model, dataloader, device, num_steps_to_show=8):
    """
    Visualize the denoising process for both images and labels.
    Shows how noise gradually becomes structured data.
    """
    model.eval()

    # Get a real image for reference
    images, labels = next(iter(dataloader))
    image, label = images[0:1].to(device), labels[0:1].to(device)

    # Start from noise for both modalities
    with torch.no_grad():
        batch_size = 1
        img_t = torch.randn(batch_size, 1, 28, 28, device=device)
        label_t = torch.randn(batch_size, 10, device=device)

        # Create timestep schedule
        total_steps = 10
        step_size = model.num_timesteps // total_steps
        timesteps = list(range(0, model.num_timesteps, step_size))[::-1]

        # Collect snapshots at regular intervals
        snapshot_interval = len(timesteps) // num_steps_to_show
        img_snapshots = []
        label_snapshots = []

        for idx, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device)
            img_t, label_t = model.p_sample_step(img_t, label_t, t_batch,
                                                denoise_image=True, denoise_label=True)

            if idx % snapshot_interval == 0 or idx == len(timesteps) - 1:
                img_snapshots.append(img_t.clone())
                label_probs = F.softmax(label_t.clone(), dim=-1)
                label_snapshots.append(label_probs)

    # Create figure
    fig, axes = plt.subplots(2, len(img_snapshots), figsize=(16, 4))

    for i, (img, label_prob) in enumerate(zip(img_snapshots, label_snapshots)):
        # Show image denoising
        axes[0, i].imshow(img[0, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Step {i}', fontsize=9)
        axes[0, i].axis('off')

        # Show label distribution denoising
        probs = label_prob[0].cpu().numpy()
        axes[1, i].bar(range(10), probs, color='steelblue')
        axes[1, i].set_ylim([0, 1])
        axes[1, i].set_xlabel('Digit', fontsize=8)
        axes[1, i].tick_params(labelsize=7)
        if i == len(img_snapshots) - 1:
            pred = probs.argmax()
            axes[1, i].set_title(f'Final: {pred}', fontsize=9)

    axes[0, 0].set_ylabel('Image', fontsize=10)
    axes[1, 0].set_ylabel('Label Prob', fontsize=10)

    plt.tight_layout()
    return fig


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_img_loss = 0
    total_label_loss = 0

    # Train over the dataset 10 times per epoch
    num_repeats = 10
    total_batches = len(dataloader) * num_repeats

    pbar = tqdm(total=total_batches, desc=f"Epoch {epoch}")
    global_batch_idx = 0

    for repeat in range(num_repeats):
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss, img_loss, label_loss = model.p_losses(images, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_img_loss += img_loss
            total_label_loss += label_loss

            # Log batch metrics
            if global_batch_idx % 100 == 0:
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/img_loss": img_loss,
                    "batch/label_loss": label_loss,
                    "batch/step": (epoch - 1) * total_batches + global_batch_idx,
                })

            # Update progress bar
            pbar.set_postfix({
                'repeat': f'{repeat + 1}/{num_repeats}',
                'loss': f'{loss.item():.4f}',
                'img': f'{img_loss:.4f}',
                'label': f'{label_loss:.4f}'
            })
            pbar.update(1)
            global_batch_idx += 1

    pbar.close()
    n = total_batches
    return total_loss / n, total_img_loss / n, total_label_loss / n


@torch.no_grad()
def evaluate(model, dataloader, device, num_inference_steps=50):
    model.eval()
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)

        predictions = model.predict(images, num_inference_steps)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ============================================================================
# Main
# ============================================================================

def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 100
    learning_rate = 1e-3
    num_timesteps = 10
    num_inference_steps = 10
    viz_every_n_epochs = 5  # Visualize every N epochs

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    wandb.init(
        project="mnist-joint-diffusion",
        config={
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "num_timesteps": num_timesteps,
            "num_inference_steps": num_inference_steps,
            "architecture": "joint_diffusion_with_concatenation_and_conditional_training",
        }
    )

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = JointDiffusion(num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    wandb.config.update({"model_parameters": num_params})

    # Training
    best_acc = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, img_loss, label_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        print(f"Train Loss: {train_loss:.4f} (img: {img_loss:.4f}, label: {label_loss:.4f})")

        # Evaluate every epoch
        test_acc = evaluate(model, test_loader, device, num_inference_steps)
        print(f"Test Accuracy: {test_acc:.4f}")

        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/img_loss": img_loss,
            "train/label_loss": label_loss,
            "test/accuracy": test_acc,
            "learning_rate": scheduler.get_last_lr()[0],
        })

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_joint_model.pt')
            print(f"New best accuracy: {best_acc:.4f}")
            wandb.run.summary["best_accuracy"] = best_acc

        # Periodic visualization
        if (epoch + 1) % viz_every_n_epochs == 0 or epoch == 0:
            print("Generating visualizations...")

            # 1. Classification results
            fig_class = visualize_classification(model, test_loader, device, num_samples=16,
                                                num_inference_steps=num_inference_steps)
            wandb.log({"visualizations/classification": wandb.Image(fig_class)})
            plt.close(fig_class)

            # 2. Joint generation (sample from P(image, label))
            fig_joint = visualize_joint_generation(model, device, num_samples=16,
                                                  num_inference_steps=num_inference_steps)
            wandb.log({"visualizations/joint_generation": wandb.Image(fig_joint)})
            plt.close(fig_joint)

            # 3. Conditional generation (image -> label with distributions)
            fig_cond = visualize_conditional_generation(model, test_loader, device, num_samples=8,
                                                       num_inference_steps=num_inference_steps)
            wandb.log({"visualizations/conditional_generation": wandb.Image(fig_cond)})
            plt.close(fig_cond)

            # 4. Denoising process
            fig_denoise = visualize_denoising_process(model, test_loader, device, num_steps_to_show=8)
            wandb.log({"visualizations/denoising_process": wandb.Image(fig_denoise)})
            plt.close(fig_denoise)

        scheduler.step()

    print(f"\nTraining complete. Best test accuracy: {best_acc:.4f}")

    # Final detailed evaluation
    print("\nFinal evaluation with more inference steps...")
    model.load_state_dict(torch.load('best_joint_model.pt'))
    final_acc = evaluate(model, test_loader, device, num_inference_steps=100)
    print(f"Final Test Accuracy (100 steps): {final_acc:.4f}")

    wandb.log({"test/final_accuracy_100steps": final_acc})

    # Final visualizations with more samples
    print("Generating final visualizations...")
    fig_final_class = visualize_classification(model, test_loader, device, num_samples=16,
                                              num_inference_steps=100)
    wandb.log({"final/classification": wandb.Image(fig_final_class)})
    plt.close(fig_final_class)

    fig_final_joint = visualize_joint_generation(model, device, num_samples=16,
                                                num_inference_steps=100)
    wandb.log({"final/joint_generation": wandb.Image(fig_final_joint)})
    plt.close(fig_final_joint)

    wandb.finish()


if __name__ == "__main__":
    main()
