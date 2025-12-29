"""
Conditional-Only Training for Joint Flow Matching Architecture

This script trains the same joint flow matching architecture but ONLY on the
conditional image generation task P(image | label), rather than the full joint
distribution P(image, label).

Key differences from joint training:
- Labels are always kept at t=0 (clean, one-hot encoded)
- Only the image branch flows from noise to data
- Loss is computed only on image velocity prediction
- This is a pure conditional diffusion model

This approach can be useful for:
1. Faster training (single modality flow)
2. Better conditional generation quality
3. Comparing against joint training
4. When classification is not needed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

# Import model architecture
from joint_diffusion_classifier import JointFlowMatching


# ============================================================================
# Conditional-Only Wrapper
# ============================================================================

class ConditionalOnlyFlowMatching(nn.Module):
    """
    Wrapper around JointFlowMatching that trains only on P(image|label).

    During training:
    - Labels are always at t=0 (clean)
    - Only images flow from t=1 (noise) to t=0 (data)
    - Only image velocity is predicted and trained
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.model = JointFlowMatching(hidden_dim=hidden_dim)

    def compute_loss(self, images, labels):
        """
        Conditional flow matching loss: P(image | label).

        The label is always at t=0 (clean), only the image flows.
        """
        batch_size = images.size(0)
        device = images.device

        # Convert labels to one-hot at t=0 (clean, no noise)
        label_vectors = F.one_hot(labels, num_classes=10).float()
        label_vectors = label_vectors * 2 - 1  # Scale to [-1, 1]

        # Sample time only for images
        t_img = torch.rand(batch_size, device=device)

        # Labels always at t=0 (no noise)
        t_label = torch.zeros(batch_size, device=device)

        # Sample noise for images only
        noise_img = torch.randn_like(images)

        # Interpolate images: x_t = (1-t)*x_0 + t*noise
        img_t = self.model.interpolate(images, noise_img, t_img)

        # Target velocity for images: v = noise - x_0
        target_v_img = noise_img - images

        # Predict velocity (label_t is just clean labels, no noise)
        pred_v_img, pred_v_label = self.model.velocity_field(
            img_t, label_vectors, t_img, t_label
        )

        # Loss only on image velocity
        img_loss = F.mse_loss(pred_v_img, target_v_img)

        # Note: We ignore pred_v_label since labels don't flow
        return img_loss

    @torch.no_grad()
    def generate(self, labels, num_steps=50):
        """Generate images conditioned on labels."""
        return self.model.generate_from_label(labels, num_steps)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# ============================================================================
# Visualization
# ============================================================================

def visualize_conditional_generation(model, device, num_per_class=8, num_steps=50):
    """Generate images conditioned on each digit class."""
    model.eval()

    fig, axes = plt.subplots(num_per_class, 10, figsize=(20, 2*num_per_class))

    for digit in range(10):
        labels = torch.full((num_per_class,), digit, device=device)
        images = model.generate(labels, num_steps)

        for j in range(num_per_class):
            axes[j, digit].imshow(images[j, 0].cpu().numpy(), cmap='gray')
            axes[j, digit].axis('off')
            if j == 0:
                axes[j, digit].set_title(f'{digit}', fontsize=12, fontweight='bold')

    plt.suptitle(f'Conditional Generation: P(image | label) [{num_steps} steps]',
                 fontsize=14)
    plt.tight_layout()
    return fig


def visualize_generation_grid(model, device, num_samples=64, num_steps=50):
    """Generate a grid of random conditional samples."""
    model.eval()

    # Random labels
    labels = torch.randint(0, 10, (num_samples,), device=device)
    images = model.generate(labels, num_steps)

    # Create grid
    rows = int(torch.sqrt(torch.tensor(num_samples)).item())
    cols = (num_samples + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2.5))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].cpu().numpy()[0]
        label = labels[i].item()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{label}', fontsize=10)
        axes[i].axis('off')

    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Random Conditional Samples [{num_steps} steps]', fontsize=14)
    plt.tight_layout()
    return fig


def visualize_diversity(model, device, digit, num_samples=16, num_steps=50):
    """Show diversity in generation for a single digit."""
    model.eval()

    labels = torch.full((num_samples,), digit, device=device)
    images = model.generate(labels, num_steps)

    rows = 4
    cols = num_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].cpu().numpy()[0]
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    plt.suptitle(f'Generation Diversity - Digit {digit} [{num_steps} steps]',
                 fontsize=14)
    plt.tight_layout()
    return fig


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_repeats = 10

    total_iters = len(dataloader) * num_repeats
    pbar = tqdm(total=total_iters, desc=f"Epoch {epoch}")

    for _ in range(num_repeats):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(images, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            pbar.update(1)

    pbar.close()
    n = len(dataloader) * num_repeats
    return total_loss / n


@torch.no_grad()
def compute_fid_proxy(model, dataloader, device, num_steps=50):
    """
    Compute a simple proxy for generation quality.
    Generate images for each label in the batch and measure reconstruction similarity.
    """
    model.eval()
    total_similarity = 0
    total_samples = 0

    for images, labels in tqdm(dataloader, desc="Computing quality metric", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Generate images conditioned on true labels
        generated = model.generate(labels, num_steps)

        # Measure similarity (negative MSE as proxy)
        similarity = -F.mse_loss(generated, images, reduction='none').mean(dim=[1,2,3])
        total_similarity += similarity.sum().item()
        total_samples += images.size(0)

    return total_similarity / total_samples


# ============================================================================
# Main
# ============================================================================

def main():
    # Config
    batch_size = 2048
    num_epochs = 100
    learning_rate = 1e-3
    hidden_dim = 128
    num_steps_train_eval = 50  # Steps for evaluation during training
    num_steps_final = 100  # Steps for final evaluation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    wandb.init(
        project="mnist-conditional-flow",
        config={
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "num_steps_eval": num_steps_train_eval,
            "architecture": "conditional_only_flow_matching",
            "training_mode": "conditional_only",
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
    model = ConditionalOnlyFlowMatching(hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Training loop
    best_quality = float('-inf')
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch}: loss={train_loss:.4f}")

        # Evaluate generation quality
        quality = compute_fid_proxy(model, test_loader, device, num_steps_train_eval)
        print(f"Quality metric: {quality:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "eval/quality": quality,
            "lr": scheduler.get_last_lr()[0],
        })

        if quality > best_quality:
            best_quality = quality
            torch.save(model.state_dict(), 'best_conditional_model.pt')
            print(f"New best quality: {best_quality:.4f}")

        # Visualize periodically
        if epoch % 5 == 0 or epoch == 1:
            fig_all = visualize_conditional_generation(model, device, 8, num_steps_train_eval)
            wandb.log({"viz/all_digits": wandb.Image(fig_all)})
            plt.close(fig_all)

            fig_grid = visualize_generation_grid(model, device, 64, num_steps_train_eval)
            wandb.log({"viz/random_samples": wandb.Image(fig_grid)})
            plt.close(fig_grid)

            fig_div = visualize_diversity(model, device, 7, 16, num_steps_train_eval)
            wandb.log({"viz/diversity": wandb.Image(fig_div)})
            plt.close(fig_div)

        scheduler.step()

    # Final evaluation
    print(f"\nBest quality metric during training: {best_quality:.4f}")
    print(f"Final evaluation with {num_steps_final} steps...")

    model.load_state_dict(torch.load('best_conditional_model.pt'))
    final_quality = compute_fid_proxy(model, test_loader, device, num_steps_final)
    print(f"Final quality ({num_steps_final} steps): {final_quality:.4f}")

    # Generate final visualizations
    print("Generating final visualizations...")
    fig_final = visualize_conditional_generation(model, device, 10, num_steps_final)
    wandb.log({"final/conditional_generation": wandb.Image(fig_final)})
    plt.close(fig_final)

    wandb.log({"eval/final_quality": final_quality})
    wandb.finish()

    print("\nTraining complete!")
    print(f"Best model saved to: best_conditional_model.pt")


if __name__ == "__main__":
    main()
