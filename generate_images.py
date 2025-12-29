"""
Image Generation Script for Joint Flow Matching Model

This script generates high-quality images using the trained flow matching model.
Supports both conditional generation P(image|label) and joint generation P(image, label).

Usage examples:
    # Generate images for all digits (8 samples per digit)
    python generate_images.py --mode conditional --num-per-class 8

    # Generate 64 joint samples
    python generate_images.py --mode joint --num-samples 64

    # Generate only specific digits with high quality
    python generate_images.py --mode conditional --digits 0 1 7 --num-per-class 16 --steps 500

    # Quick generation for testing
    python generate_images.py --mode conditional --steps 50
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from datetime import datetime

# Import model architecture
from joint_diffusion_classifier import JointFlowMatching


def generate_conditional_images(model, device, digits, num_per_class, num_steps, output_path):
    """
    Generate images conditioned on specific digit labels.

    Args:
        model: Trained JointFlowMatching model
        device: torch device
        digits: List of digit classes to generate (0-9)
        num_per_class: Number of samples per digit class
        num_steps: Number of ODE integration steps
        output_path: Where to save the output image
    """
    model.eval()

    num_digits = len(digits)
    fig, axes = plt.subplots(num_per_class, num_digits,
                            figsize=(2*num_digits, 2*num_per_class))

    # Handle single digit case
    if num_digits == 1:
        axes = axes.reshape(-1, 1)
    if num_per_class == 1:
        axes = axes.reshape(1, -1)

    print(f"Generating {num_per_class} samples for each of {num_digits} digit(s) with {num_steps} steps...")

    for col_idx, digit in enumerate(tqdm(digits, desc="Generating digits")):
        labels = torch.full((num_per_class,), digit, device=device)
        images = model.generate_from_label(labels, num_steps)

        for row_idx in range(num_per_class):
            axes[row_idx, col_idx].imshow(images[row_idx, 0].cpu().numpy(), cmap='gray')
            axes[row_idx, col_idx].axis('off')

            # Title on first row
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f'Digit {digit}',
                                                 fontsize=14, fontweight='bold')

    plt.suptitle(f'Conditional Generation: P(image | label) [{num_steps} ODE steps]',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def generate_joint_samples(model, device, num_samples, num_steps, output_path):
    """
    Generate samples from the joint distribution P(image, label).

    Args:
        model: Trained JointFlowMatching model
        device: torch device
        num_samples: Total number of samples to generate
        num_steps: Number of ODE integration steps
        output_path: Where to save the output image
    """
    model.eval()

    print(f"Generating {num_samples} joint samples with {num_steps} steps...")
    images, label_probs = model.sample_joint(num_samples, device, num_steps)
    labels = label_probs.argmax(dim=-1)

    # Arrange in grid
    rows = int(np.sqrt(num_samples))
    cols = (num_samples + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2.5*rows))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].cpu().numpy()[0]
        label = labels[i].item()
        conf = label_probs[i, label].item()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{label}\n({conf:.3f})', fontsize=11)
        axes[i].axis('off')

    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Joint Generation: P(image, label) [{num_steps} ODE steps]',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def generate_grid_all_digits(model, device, num_per_class, num_steps, output_path):
    """
    Generate a large grid showing all digits in a single organized visualization.

    Args:
        model: Trained JointFlowMatching model
        device: torch device
        num_per_class: Number of samples per digit class
        num_steps: Number of ODE integration steps
        output_path: Where to save the output image
    """
    model.eval()

    print(f"Generating comprehensive grid: {num_per_class} × 10 = {num_per_class * 10} images with {num_steps} steps...")

    # Generate all images at once for efficiency
    all_images = []
    for digit in tqdm(range(10), desc="Generating all digits"):
        labels = torch.full((num_per_class,), digit, device=device)
        images = model.generate_from_label(labels, num_steps)
        all_images.append(images)

    # Create compact grid
    fig, axes = plt.subplots(10, num_per_class, figsize=(num_per_class*1.5, 15))

    if num_per_class == 1:
        axes = axes.reshape(-1, 1)

    for digit in range(10):
        for sample_idx in range(num_per_class):
            img = all_images[digit][sample_idx, 0].cpu().numpy()
            axes[digit, sample_idx].imshow(img, cmap='gray')
            axes[digit, sample_idx].axis('off')

            # Label first column with digit
            if sample_idx == 0:
                axes[digit, sample_idx].set_ylabel(f'{digit}',
                                                   fontsize=20,
                                                   fontweight='bold',
                                                   rotation=0,
                                                   labelpad=20)

    plt.suptitle(f'Conditional Image Generation - All Digits [{num_steps} ODE steps]',
                 fontsize=18, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def generate_diversity_showcase(model, device, digit, num_samples, num_steps, output_path):
    """
    Generate many samples of a single digit to show diversity.

    Args:
        model: Trained JointFlowMatching model
        device: torch device
        digit: Which digit to generate
        num_samples: Number of samples to generate
        num_steps: Number of ODE integration steps
        output_path: Where to save the output image
    """
    model.eval()

    print(f"Generating {num_samples} diverse samples of digit {digit} with {num_steps} steps...")

    labels = torch.full((num_samples,), digit, device=device)
    images = model.generate_from_label(labels, num_steps)

    # Arrange in grid
    rows = int(np.sqrt(num_samples))
    cols = (num_samples + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i, 0].cpu().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Diversity Showcase - Digit {digit} [{num_steps} ODE steps]',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate images using trained Joint Flow Matching model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All digits, 8 samples each
  python generate_images.py --mode conditional --num-per-class 8

  # Specific digits with high quality
  python generate_images.py --mode conditional --digits 0 1 7 --num-per-class 16 --steps 500

  # Joint generation
  python generate_images.py --mode joint --num-samples 64 --steps 300

  # Diversity showcase for single digit
  python generate_images.py --mode diversity --digit 7 --num-samples 64 --steps 200

  # Compact grid of all digits
  python generate_images.py --mode grid --num-per-class 10 --steps 200
        """
    )

    parser.add_argument('--mode', type=str,
                       choices=['conditional', 'joint', 'grid', 'diversity'],
                       default='conditional',
                       help='Generation mode (default: conditional)')

    parser.add_argument('--model-path', type=str, default='best_flow_model.pt',
                       help='Path to trained model checkpoint (default: best_flow_model.pt)')

    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Model hidden dimension (default: 128)')

    parser.add_argument('--steps', type=int, default=200,
                       help='Number of ODE integration steps (default: 200)')

    parser.add_argument('--digits', type=int, nargs='+', default=list(range(10)),
                       help='Which digits to generate for conditional mode (default: all)')

    parser.add_argument('--num-per-class', type=int, default=8,
                       help='Samples per digit class for conditional/grid mode (default: 8)')

    parser.add_argument('--num-samples', type=int, default=64,
                       help='Total samples for joint/diversity mode (default: 64)')

    parser.add_argument('--digit', type=int, default=7,
                       help='Specific digit for diversity mode (default: 7)')

    parser.add_argument('--output-dir', type=str, default='generated_images',
                       help='Output directory (default: generated_images)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("Joint Flow Matching - Image Generation")
    print("="*70)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"ODE steps: {args.steps}")
    print(f"Random seed: {args.seed}")
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
        print("Please train the model first using joint_diffusion_classifier.py")
        return

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate based on mode
    if args.mode == 'conditional':
        output_path = output_dir / f'conditional_{timestamp}.png'
        generate_conditional_images(
            model, device, args.digits, args.num_per_class, args.steps, output_path
        )

    elif args.mode == 'joint':
        output_path = output_dir / f'joint_{timestamp}.png'
        generate_joint_samples(
            model, device, args.num_samples, args.steps, output_path
        )

    elif args.mode == 'grid':
        output_path = output_dir / f'grid_all_digits_{timestamp}.png'
        generate_grid_all_digits(
            model, device, args.num_per_class, args.steps, output_path
        )

    elif args.mode == 'diversity':
        output_path = output_dir / f'diversity_digit{args.digit}_{timestamp}.png'
        generate_diversity_showcase(
            model, device, args.digit, args.num_samples, args.steps, output_path
        )

    print()
    print("="*70)
    print("Generation complete! 🎨")
    print(f"Output saved to: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()
