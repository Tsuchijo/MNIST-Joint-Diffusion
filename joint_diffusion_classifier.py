import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt


# ============================================================================
# Sinusoidal Time Embeddings (now for continuous t ∈ [0,1])
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: (B,) values in [0, 1]"""
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Scale t to get good frequency coverage
        emb = (t * 1000)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ============================================================================
# Cross-Attention Module
# ============================================================================

class CrossAttention(nn.Module):
    """Bidirectional cross-attention between two modalities."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_a = nn.Linear(dim, dim)
        self.k_b = nn.Linear(dim, dim)
        self.v_b = nn.Linear(dim, dim)
        self.out_a = nn.Linear(dim, dim)
        
        self.q_b = nn.Linear(dim, dim)
        self.k_a = nn.Linear(dim, dim)
        self.v_a = nn.Linear(dim, dim)
        self.out_b = nn.Linear(dim, dim)
        
        # Zero init for residual-friendly behavior
        nn.init.zeros_(self.out_a.weight)
        nn.init.zeros_(self.out_a.bias)
        nn.init.zeros_(self.out_b.weight)
        nn.init.zeros_(self.out_b.bias)
        
    def forward(self, a, b):
        B = a.size(0)
        
        # A attends to B
        q_a = self.q_a(a).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_b = self.k_b(b).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v_b = self.v_b(b).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_a = (q_a @ k_b.transpose(-2, -1)) * self.scale
        attn_a = attn_a.softmax(dim=-1)
        a_update = (attn_a @ v_b).transpose(1, 2).reshape(B, -1)
        a_out = a + self.out_a(a_update)
        
        # B attends to A
        q_b = self.q_b(b).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_a = self.k_a(a).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v_a = self.v_a(a).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_b = (q_b @ k_a.transpose(-2, -1)) * self.scale
        attn_b = attn_b.softmax(dim=-1)
        b_update = (attn_b @ v_a).transpose(1, 2).reshape(B, -1)
        b_out = b + self.out_b(b_update)
        
        return a_out, b_out


# ============================================================================
# Image Branch
# ============================================================================

class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim=256, time_dim=64):
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encoder with skip connections (U-Net style)
        # Layer 1: 28x28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.SiLU(),
        )

        # Layer 2: 14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        # Layer 3: 7x7 (bottleneck)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, hidden_dim)

    def forward(self, x, t):
        """
        x: (B, 1, 28, 28) interpolated image
        t: (B,) continuous time in [0, 1]

        Returns:
            h: (B, hidden_dim) - bottleneck representation
            skips: List of skip connection features
        """
        t_emb = self.time_mlp(self.time_embed(t))

        # Encoder with skip connections
        skip1 = self.conv1(x)         # (B, 64, 28, 28)
        skip2 = self.conv2(skip1)      # (B, 128, 14, 14)
        skip3 = self.conv3(skip2)      # (B, 256, 7, 7)

        # Bottleneck
        h = self.pool(skip3).squeeze(-1).squeeze(-1)  # (B, 256)
        h = self.proj(h)  # (B, hidden_dim)
        h = h + t_emb

        return h, [skip1, skip2, skip3]


class ImageDecoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        self.proj = nn.Linear(hidden_dim, 256 * 7 * 7)

        # Decoder with skip connections (U-Net style)
        # Upsample from 7x7 to 14x14 (concatenate with skip2: 128 channels)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        # After concat: 128 + 128 = 256 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        # Upsample from 14x14 to 28x28 (concatenate with skip1: 64 channels)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        # After concat: 64 + 64 = 128 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )

        # Final output layer
        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, h, skips):
        """
        h: (B, hidden_dim) - bottleneck features
        skips: [skip1, skip2, skip3] - skip connection features
               skip1: (B, 64, 28, 28)
               skip2: (B, 128, 14, 14)
               skip3: (B, 256, 7, 7)
        """
        skip1, skip2, skip3 = skips
        B = h.size(0)

        # Project and reshape to spatial
        h = self.proj(h).view(B, 256, 7, 7)  # (B, 256, 7, 7)

        # Upsample to 14x14 and concatenate with skip2
        h = self.up1(h)                      # (B, 128, 14, 14)
        h = torch.cat([h, skip2], dim=1)     # (B, 256, 14, 14)
        h = self.conv1(h)                    # (B, 128, 14, 14)

        # Upsample to 28x28 and concatenate with skip1
        h = self.up2(h)                      # (B, 64, 28, 28)
        h = torch.cat([h, skip1], dim=1)     # (B, 128, 28, 28)
        h = self.conv2(h)                    # (B, 64, 28, 28)

        # Output
        return self.out(h)                   # (B, 1, 28, 28)


# ============================================================================
# Label Branch  
# ============================================================================

class LabelEncoder(nn.Module):
    def __init__(self, label_dim=10, hidden_dim=256, time_dim=64):
        super().__init__()
        
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(label_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x, t):
        """
        x: (B, 10) interpolated label vector
        t: (B,) continuous time in [0, 1]
        """
        t_emb = self.time_mlp(self.time_embed(t))
        h = self.mlp(x)
        return h + t_emb


class LabelDecoder(nn.Module):
    def __init__(self, label_dim=10, hidden_dim=256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, label_dim),
        )
        
    def forward(self, h):
        return self.mlp(h)


# ============================================================================
# Joint Velocity Field with Cross-Attention
# ============================================================================

class JointVelocityField(nn.Module):
    """
    Predicts velocity field v(x_t, t) for both modalities.
    Uses cross-attention for information sharing between branches.
    """
    def __init__(self, hidden_dim=256, num_cross_attn_layers=3):
        super().__init__()
        
        self.img_encoder = ImageEncoder(hidden_dim)
        self.label_encoder = LabelEncoder(hidden_dim=hidden_dim)
        
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(hidden_dim) for _ in range(num_cross_attn_layers)
        ])
        
        self.img_refine = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_cross_attn_layers)
        ])
        
        self.label_refine = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_cross_attn_layers)
        ])
        
        self.img_decoder = ImageDecoder(hidden_dim)
        self.label_decoder = LabelDecoder(hidden_dim=hidden_dim)
        
    def forward(self, img_t, label_t, t_img, t_label):
        """
        Predict velocities for both modalities.

        Args:
            img_t: (B, 1, 28, 28) - image at time t_img
            label_t: (B, 10) - label at time t_label
            t_img: (B,) - continuous time for image [0, 1]
            t_label: (B,) - continuous time for label [0, 1] (INDEPENDENT)

        Returns:
            v_img: (B, 1, 28, 28) - predicted velocity for image
            v_label: (B, 10) - predicted velocity for label
        """
        h_img, img_skips = self.img_encoder(img_t, t_img)
        h_label = self.label_encoder(label_t, t_label)

        for cross_attn, img_ref, label_ref in zip(
            self.cross_attn_layers, self.img_refine, self.label_refine
        ):
            h_img, h_label = cross_attn(h_img, h_label)
            h_img = h_img + img_ref(h_img)
            h_label = h_label + label_ref(h_label)

        v_img = self.img_decoder(h_img, img_skips)
        v_label = self.label_decoder(h_label)

        return v_img, v_label


# ============================================================================
# Joint Flow Matching Model
# ============================================================================

class JointFlowMatching(nn.Module):
    """
    Joint flow matching with independent time sampling.
    
    Path: x_t = (1-t)*x_0 + t*noise  (linear optimal transport)
    Velocity: v = dx/dt = noise - x_0
    
    Training: sample t_img, t_label independently, predict velocity
    Inference: integrate ODE from t=1 (noise) to t=0 (data)
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.velocity_field = JointVelocityField(hidden_dim=hidden_dim)
        
    def interpolate(self, x_0, noise, t):
        """
        Linear interpolation: x_t = (1-t)*x_0 + t*noise
        
        Args:
            x_0: clean data
            noise: sampled noise
            t: (B,) time values in [0, 1]
        """
        # Reshape t for broadcasting
        t_shape = t
        while len(t_shape.shape) < len(x_0.shape):
            t_shape = t_shape.unsqueeze(-1)
        
        return (1 - t_shape) * x_0 + t_shape * noise
    
    def compute_loss(self, images, labels):
        """
        Flow matching loss with independent time sampling.
        
        Target velocity is simply: v = noise - x_0
        (derivative of linear interpolation path)
        """
        batch_size = images.size(0)
        device = images.device
        
        # Convert labels to scaled one-hot
        label_vectors = F.one_hot(labels, num_classes=10).float()
        label_vectors = label_vectors * 2 - 1  # Scale to [-1, 1]
        
        # Sample independent times for each modality
        t_img = torch.rand(batch_size, device=device)
        t_label = torch.rand(batch_size, device=device)
        
        # Sample noise
        noise_img = torch.randn_like(images)
        noise_label = torch.randn_like(label_vectors)
        
        # Interpolate to get x_t
        img_t = self.interpolate(images, noise_img, t_img)
        label_t = self.interpolate(label_vectors, noise_label, t_label)
        
        # Target velocity: v = noise - x_0
        target_v_img = noise_img - images
        target_v_label = noise_label - label_vectors
        
        # Predict velocity
        pred_v_img, pred_v_label = self.velocity_field(img_t, label_t, t_img, t_label)
        
        # MSE loss on velocities
        img_loss = F.mse_loss(pred_v_img, target_v_img)
        label_loss = F.mse_loss(pred_v_label, target_v_label)
        
        total_loss = img_loss + label_loss
        
        return total_loss, img_loss.item(), label_loss.item()
    
    @torch.no_grad()
    def integrate_ode(self, x_init, t_modality, fixed_modality, fixed_value, 
                      num_steps=50, modality='label'):
        """
        Integrate ODE from t=1 to t=0 using Euler method.
        
        Args:
            x_init: initial value (noise) for the modality being integrated
            t_modality: which modality we're integrating ('img' or 'label')
            fixed_modality: the other modality held at t=0
            fixed_value: value of the fixed modality (clean data)
            num_steps: number of integration steps
            modality: 'img' or 'label'
        """
        device = x_init.device
        batch_size = x_init.size(0)
        
        x = x_init
        dt = 1.0 / num_steps
        
        # Fixed modality stays at t=0
        t_fixed = torch.zeros(batch_size, device=device)
        
        for i in range(num_steps):
            t = 1.0 - i * dt  # t goes from 1 → 0
            t_current = torch.full((batch_size,), t, device=device)
            
            if modality == 'label':
                # Integrating label, image is fixed
                v_img, v_label = self.velocity_field(
                    fixed_value, x, t_fixed, t_current
                )
                x = x - v_label * dt  # Euler step (minus because going from 1→0)
            else:
                # Integrating image, label is fixed
                v_img, v_label = self.velocity_field(
                    x, fixed_value, t_current, t_fixed
                )
                x = x - v_img * dt
        
        return x
    
    @torch.no_grad()
    def classify(self, images, num_steps=50):
        """
        Classification: P(label | image)
        
        Hold image at t=0 (clean), integrate label from t=1 (noise) to t=0.
        """
        device = images.device
        batch_size = images.size(0)
        
        # Start label from pure noise
        label_noise = torch.randn(batch_size, 10, device=device)
        
        # Integrate label ODE with image fixed
        label_final = self.integrate_ode(
            x_init=label_noise,
            t_modality='label',
            fixed_modality='img',
            fixed_value=images,
            num_steps=num_steps,
            modality='label'
        )
        
        return label_final
    
    @torch.no_grad()
    def generate_from_label(self, labels, num_steps=50):
        """
        Conditional generation: P(image | label)
        
        Hold label at t=0 (clean), integrate image from t=1 to t=0.
        """
        device = labels.device
        batch_size = labels.size(0)
        
        # Convert to one-hot if needed
        if labels.dim() == 1:
            label_vectors = F.one_hot(labels, num_classes=10).float()
            label_vectors = label_vectors * 2 - 1
        else:
            label_vectors = labels
        
        # Start image from pure noise
        img_noise = torch.randn(batch_size, 1, 28, 28, device=device)
        
        # Integrate image ODE with label fixed
        img_final = self.integrate_ode(
            x_init=img_noise,
            t_modality='img',
            fixed_modality='label',
            fixed_value=label_vectors,
            num_steps=num_steps,
            modality='img'
        )
        
        return img_final
    
    @torch.no_grad()
    def sample_joint(self, batch_size, device, num_steps=50):
        """
        Sample from joint P(image, label).
        
        Integrate both modalities together from t=1 to t=0.
        """
        img = torch.randn(batch_size, 1, 28, 28, device=device)
        label = torch.randn(batch_size, 10, device=device)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = 1.0 - i * dt
            t_batch = torch.full((batch_size,), t, device=device)
            
            v_img, v_label = self.velocity_field(img, label, t_batch, t_batch)
            
            img = img - v_img * dt
            label = label - v_label * dt
        
        return img, label
    
    @torch.no_grad()
    def predict(self, images, num_steps=50):
        probs = self.classify(images, num_steps)
        return probs.argmax(dim=-1)


# ============================================================================
# Visualization
# ============================================================================

def visualize_classification(model, dataloader, device, num_samples=16, num_steps=50):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)
    
    probs = model.classify(images, num_steps)
    predictions = probs.argmax(dim=-1)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = images[i].cpu().numpy()[0]
        pred = predictions[i].item()
        true = labels[i].item()
        conf = probs[i, pred].item()
        
        axes[i].imshow(img, cmap='gray')
        color = 'green' if pred == true else 'red'
        axes[i].set_title(f'Pred: {pred} (True: {true})\nConf: {conf:.2f}', color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_joint_generation(model, device, num_samples=16, num_steps=50):
    model.eval()
    images, label_probs = model.sample_joint(num_samples, device, num_steps)
    labels = label_probs.argmax(dim=-1)
    
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


def visualize_conditional_generation(model, device, num_per_class=2, num_steps=50):
    """Generate images conditioned on each digit class."""
    model.eval()
    
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    
    for digit in range(10):
        labels = torch.full((num_per_class,), digit, device=device)
        images = model.generate_from_label(labels, num_steps)
        
        for j in range(num_per_class):
            axes[j, digit].imshow(images[j, 0].cpu().numpy(), cmap='gray')
            axes[j, digit].axis('off')
            if j == 0:
                axes[j, digit].set_title(f'{digit}', fontsize=12)
    
    plt.suptitle('Conditional Generation: P(image | label)', fontsize=14)
    plt.tight_layout()
    return fig


def visualize_flow_trajectory(model, dataloader, device, num_snapshots=8, num_steps=50):
    """Visualize the flow trajectory during classification."""
    model.eval()
    
    images, true_labels = next(iter(dataloader))
    image = images[0:1].to(device)
    true_label = true_labels[0].item()
    
    # Track trajectory
    batch_size = 1
    label = torch.randn(batch_size, 10, device=device)
    t_fixed = torch.zeros(batch_size, device=device)
    
    dt = 1.0 / num_steps
    snapshot_interval = num_steps // num_snapshots
    
    trajectories = [label.clone()]
    times = [1.0]
    
    for i in range(num_steps):
        t = 1.0 - i * dt
        t_current = torch.full((batch_size,), t, device=device)
        
        _, v_label = model.velocity_field(image, label, t_fixed, t_current)
        label = label - v_label * dt
        
        if (i + 1) % snapshot_interval == 0:
            trajectories.append(label.clone())
            times.append(t - dt)
    
    # Plot
    fig, axes = plt.subplots(1, len(trajectories), figsize=(20, 3))
    
    for i, (traj, t) in enumerate(zip(trajectories, times)):
        probs = traj[0].detach().cpu().numpy()
        axes[i].bar(range(10), probs, color='steelblue')
        axes[i].set_ylim([0, 1])
        axes[i].set_xlabel('Digit')
        axes[i].set_title(f't={t:.2f}')
        if i == len(trajectories) - 1:
            pred = probs.argmax()
            axes[i].set_title(f't={t:.2f}\nPred: {pred} (True: {true_label})')
    
    axes[0].set_ylabel('Probability')
    plt.suptitle('Label flow trajectory during classification', fontsize=12)
    plt.tight_layout()
    return fig


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_img_loss = 0
    total_label_loss = 0
    num_repeats = 10

    total_iters = len(dataloader) * num_repeats
    pbar = tqdm(total=total_iters, desc=f"Epoch {epoch}")

    for _ in range(num_repeats):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss, img_loss, label_loss = model.compute_loss(images, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_img_loss += img_loss
            total_label_loss += label_loss

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'img': f'{img_loss:.4f}',
                'label': f'{label_loss:.4f}'
            })
            pbar.update(1)

    pbar.close()
    n = len(dataloader) * num_repeats
    return total_loss / n, total_img_loss / n, total_label_loss / n


@torch.no_grad()
def evaluate(model, dataloader, device, num_steps=50):
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        predictions = model.predict(images, num_steps)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return correct / total


# ============================================================================
# Main
# ============================================================================

def main():
    # Config
    batch_size = 128
    num_epochs = 100
    learning_rate = 1e-3
    hidden_dim = 256
    num_steps_train_eval = 50  # Steps for evaluation during training
    num_steps_final = 100  # Steps for final evaluation
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    wandb.init(
        project="mnist-flow-matching",
        config={
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "num_steps_eval": num_steps_train_eval,
            "architecture": "joint_flow_matching_independent_times",
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
    model = JointFlowMatching(hidden_dim=hidden_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Training loop
    best_acc = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, img_loss, label_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch}: loss={train_loss:.4f}, img={img_loss:.4f}, label={label_loss:.4f}")
        
        # Evaluate
        test_acc = evaluate(model, test_loader, device, num_steps_train_eval)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/img_loss": img_loss,
            "train/label_loss": label_loss,
            "test/accuracy": test_acc,
            "lr": scheduler.get_last_lr()[0],
        })
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_flow_model.pt')
            print(f"New best: {best_acc:.4f}")
        
        # Visualize periodically
        if epoch % 5 == 0 or epoch == 1:
            fig_class = visualize_classification(model, test_loader, device, 16, num_steps_train_eval)
            wandb.log({"viz/classification": wandb.Image(fig_class)})
            plt.close(fig_class)
            
            fig_joint = visualize_joint_generation(model, device, 16, num_steps_train_eval)
            wandb.log({"viz/joint_generation": wandb.Image(fig_joint)})
            plt.close(fig_joint)
            
            fig_cond = visualize_conditional_generation(model, device, 2, num_steps_train_eval)
            wandb.log({"viz/conditional_generation": wandb.Image(fig_cond)})
            plt.close(fig_cond)
            
            fig_traj = visualize_flow_trajectory(model, test_loader, device, 8, num_steps_train_eval)
            wandb.log({"viz/flow_trajectory": wandb.Image(fig_traj)})
            plt.close(fig_traj)
        
        scheduler.step()
    
    # Final evaluation with more steps
    print(f"\nBest accuracy during training: {best_acc:.4f}")
    print(f"Final evaluation with {num_steps_final} steps...")
    
    model.load_state_dict(torch.load('best_flow_model.pt'))
    final_acc = evaluate(model, test_loader, device, num_steps_final)
    print(f"Final accuracy ({num_steps_final} steps): {final_acc:.4f}")
    
    wandb.log({"test/final_accuracy": final_acc})
    wandb.finish()


if __name__ == "__main__":
    main()