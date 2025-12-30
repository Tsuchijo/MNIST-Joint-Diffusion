"""
Joint Generation Trajectory Animation

This script creates an animated GIF showing how both the image and label
evolve together during joint generation from pure noise to final samples.

Visualizes:
- Left: The image evolving from noise (t=1) to data (t=0)
- Right: The label probabilities evolving from noise to a distribution
- Shows the full ODE integration trajectory

This demonstrates the joint flow matching process in action!
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import imageio
from PIL import Image
from io import BytesIO

from joint_diffusion_classifier import JointFlowMatching


def generate_joint_trajectory(model, device, num_steps=100, sample_interval=1):
    """
    Generate a single joint sample while tracking the full trajectory.

    Args:
        model: Trained JointFlowMatching model
        device: torch device
        num_steps: Number of ODE integration steps
        sample_interval: Save trajectory every N steps (1=every step)

    Returns:
        trajectories: Dict with 'images', 'labels', 'times'
    """
    model.eval()

    # Start from pure noise
    batch_size = 1
    img = torch.randn(batch_size, 1, 28, 28, device=device)
    label = torch.randn(batch_size, 10, device=device)

    dt = 1.0 / num_steps

    # Track trajectories
    trajectories = {
        'images': [],
        'labels': [],
        'times': []
    }

    # Save initial state (t=1, pure noise)
    trajectories['images'].append(img.clone())
    trajectories['labels'].append(label.clone())
    trajectories['times'].append(1.0)

    print(f"Integrating ODE with {num_steps} steps...")
    for i in tqdm(range(num_steps), desc="ODE integration"):
        t = 1.0 - i * dt
        t_batch = torch.full((batch_size,), t, device=device)

        # Predict velocities
        v_img, v_label = model.velocity_field(img, label, t_batch, t_batch)

        # Euler step (negative because going from t=1 to t=0)
        img = img - v_img * dt
        label = label - v_label * dt

        # Save trajectory
        if (i + 1) % sample_interval == 0:
            trajectories['images'].append(img.clone())
            trajectories['labels'].append(label.clone())
            trajectories['times'].append(t - dt)

    return trajectories


def create_frame(image_tensor, label_tensor, time, final_pred=None):
    """
    Create a single frame showing both image and label distribution.

    Args:
        image_tensor: (1, 1, 28, 28) image tensor
        label_tensor: (1, 10) label tensor
        time: Current time value
        final_pred: Optional final prediction for comparison

    Returns:
        PIL Image
    """
    fig = plt.figure(figsize=(8, 8))

    # Top: Image
    ax_img = plt.subplot(2, 1, 1)
    img = image_tensor[0, 0].cpu().detach().numpy()
    ax_img.imshow(img, cmap='gray', vmin=-3, vmax=3)  # Fixed scale for consistency
    ax_img.set_title(f'Image (t={time:.3f})', fontsize=16, fontweight='bold')
    ax_img.axis('off')

    # Bottom: Label confidence scores (normalized logits)
    ax_label = plt.subplot(2, 1, 2)
    logits = label_tensor[0].detach().cpu().numpy()

    # Normalize logits to [0, 1] range (shows true confidence evolution)
    # Shift so minimum is 0, then scale so maximum is 1
    logits_shifted = logits - logits.min()
    if logits_shifted.max() > 0:
        confidence = logits_shifted / logits_shifted.max()
    else:
        confidence = logits_shifted

    colors = ['steelblue' if i != final_pred else 'orange' for i in range(10)]
    bars = ax_label.bar(range(10), confidence, color=colors, alpha=0.8, edgecolor='black')

    ax_label.set_ylim([0, 1.15])
    ax_label.set_xlabel('Digit', fontsize=14)
    ax_label.set_ylabel('Confidence', fontsize=14)
    ax_label.set_title(f'Label Confidence (t={time:.3f})', fontsize=16, fontweight='bold')
    ax_label.set_xticks(range(10))
    ax_label.grid(axis='y', alpha=0.3, linestyle='--')

    # Add max confidence annotation
    max_idx = confidence.argmax()
    max_conf = confidence[max_idx]
    ax_label.text(max_idx, max_conf + 0.05, f'{max_conf:.3f}',
                 ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Convert to PIL Image using BytesIO (handles high-DPI displays correctly)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    pil_img = Image.open(buf).copy()  # .copy() to allow buf to be closed
    buf.close()

    plt.close(fig)

    return pil_img


def create_trajectory_gif(trajectories, output_path, fps=10, final_pause_frames=20):
    """
    Create animated GIF from trajectory.

    Args:
        trajectories: Dict with 'images', 'labels', 'times'
        output_path: Where to save the GIF
        fps: Frames per second
        final_pause_frames: Number of extra frames to hold on final result
    """
    frames = []

    # Get final prediction for highlighting
    final_label = trajectories['labels'][-1]
    final_pred = F.softmax(final_label[0], dim=-1).argmax().item()

    print(f"Creating {len(trajectories['images'])} frames...")
    for img, label, time in tqdm(
        zip(trajectories['images'], trajectories['labels'], trajectories['times']),
        total=len(trajectories['images']),
        desc="Rendering frames"
    ):
        frame = create_frame(img, label, time, final_pred=final_pred)
        frames.append(frame)

    # Add extra frames at the end to pause on final result
    print(f"Adding {final_pause_frames} pause frames at end...")
    for _ in range(final_pause_frames):
        frames.append(frames[-1])

    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    duration = 1000 / fps  # milliseconds per frame
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # Loop forever
    )

    print(f"✓ GIF saved successfully!")
    print(f"  - Total frames: {len(frames)}")
    print(f"  - Duration: {len(frames) / fps:.1f} seconds")
    print(f"  - Final prediction: {final_pred}")


def create_side_by_side_comparison(trajectories, output_path, num_snapshots=8):
    """
    Create a static image showing key snapshots from the trajectory.

    Args:
        trajectories: Dict with 'images', 'labels', 'times'
        output_path: Where to save the image
        num_snapshots: Number of snapshots to show
    """
    num_frames = len(trajectories['images'])
    indices = np.linspace(0, num_frames - 1, num_snapshots, dtype=int)

    fig, axes = plt.subplots(2, num_snapshots, figsize=(2.5*num_snapshots, 6))

    final_label = trajectories['labels'][-1]
    final_pred = F.softmax(final_label[0], dim=-1).argmax().item()

    print(f"Creating snapshot comparison with {num_snapshots} frames...")
    for col_idx, frame_idx in enumerate(tqdm(indices, desc="Creating snapshots")):
        img = trajectories['images'][frame_idx][0, 0].cpu().detach().numpy()
        label = trajectories['labels'][frame_idx]
        time = trajectories['times'][frame_idx]
        logits = label[0].detach().cpu().numpy()

        # Normalize logits to [0, 1] range
        logits_shifted = logits - logits.min()
        if logits_shifted.max() > 0:
            confidence = logits_shifted / logits_shifted.max()
        else:
            confidence = logits_shifted

        # Top row: Images
        axes[0, col_idx].imshow(img, cmap='gray', vmin=-3, vmax=3)
        axes[0, col_idx].set_title(f't={time:.3f}', fontsize=11)
        axes[0, col_idx].axis('off')

        # Bottom row: Label confidence
        colors = ['steelblue' if i != final_pred else 'orange' for i in range(10)]
        axes[1, col_idx].bar(range(10), confidence, color=colors, alpha=0.8)
        axes[1, col_idx].set_ylim([0, 1])
        axes[1, col_idx].set_xticks(range(10))
        axes[1, col_idx].tick_params(labelsize=8)
        axes[1, col_idx].grid(axis='y', alpha=0.3)

        if col_idx == 0:
            axes[1, col_idx].set_ylabel('Confidence', fontsize=10)

    axes[0, 0].text(-0.1, 0.5, 'Image', transform=axes[0, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90,
                   va='center', ha='right')
    axes[1, 0].text(-0.1, 0.5, 'Confidence', transform=axes[1, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90,
                   va='center', ha='right')

    plt.suptitle(f'Joint Generation Trajectory (Final prediction: {final_pred})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved snapshot comparison to {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate animated GIF of joint generation trajectory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate GIF with 100 steps
  python generate_joint_trajectory_gif.py --steps 100 --fps 10

  # High-quality with more steps
  python generate_joint_trajectory_gif.py --steps 200 --sample-interval 2 --fps 15

  # Quick preview
  python generate_joint_trajectory_gif.py --steps 50 --fps 5
        """
    )

    parser.add_argument('--model-path', type=str, default='models/best_flow_model.pt',
                       help='Path to trained model')

    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Model hidden dimension')

    parser.add_argument('--steps', type=int, default=100,
                       help='Number of ODE integration steps')

    parser.add_argument('--sample-interval', type=int, default=1,
                       help='Save trajectory every N steps (1=every step)')

    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for GIF')

    parser.add_argument('--final-pause', type=int, default=20,
                       help='Number of frames to pause on final result')

    parser.add_argument('--output-dir', type=str, default='trajectory_gifs',
                       help='Output directory')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of different trajectories to generate')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("Joint Generation Trajectory Animation")
    print("="*70)
    print(f"Device: {device}")
    print(f"ODE steps: {args.steps}")
    print(f"Sample interval: {args.sample_interval}")
    print(f"FPS: {args.fps}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = JointFlowMatching(hidden_dim=args.hidden_dim).to(device)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: Model file '{args.model_path}' not found!")
        return

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Generate trajectories
    for sample_idx in range(args.num_samples):
        print("="*70)
        print(f"Generating trajectory {sample_idx + 1}/{args.num_samples}")
        print("="*70)

        # Set different seed for each sample
        torch.manual_seed(args.seed + sample_idx)

        trajectories = generate_joint_trajectory(
            model, device, args.steps, args.sample_interval
        )

        # Create GIF
        gif_path = output_dir / f'joint_trajectory_{sample_idx}.gif'
        create_trajectory_gif(
            trajectories, gif_path, args.fps, args.final_pause
        )

        # Create static comparison
        static_path = output_dir / f'joint_trajectory_{sample_idx}_snapshots.png'
        create_side_by_side_comparison(trajectories, static_path, num_snapshots=8)

        print()

    print("="*70)
    print("All trajectories generated! 🎬")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*70)
    print("\nGenerated files:")
    for sample_idx in range(args.num_samples):
        print(f"  - joint_trajectory_{sample_idx}.gif")
        print(f"  - joint_trajectory_{sample_idx}_snapshots.png")


if __name__ == "__main__":
    main()
