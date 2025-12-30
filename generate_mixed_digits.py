"""
Mixed Digit Generation - Interpolating Between Classes

This script generates "in-between" digits by conditioning the flow model on
mixed class labels instead of pure one-hot encodings.

Examples:
- 50/50 mix of 3 and 7: What does a "3.5 + 7.5" look like?
- 30/70 mix of 1 and 7: Morphing from 1 towards 7
- Smooth transitions: Animate the full interpolation path

This explores the model's learned conditional manifold between digit classes.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from joint_diffusion_classifier import JointFlowMatching


def create_mixed_label(digit_a, digit_b, alpha, num_classes=10):
    """
    Create a mixed label vector.

    Args:
        digit_a: First digit class (0-9)
        digit_b: Second digit class (0-9)
        alpha: Mixing weight for digit_a (0=pure B, 1=pure A, 0.5=50/50 mix)
        num_classes: Number of classes (default 10)

    Returns:
        Mixed label vector scaled to [-1, 1] range
    """
    # Create one-hot vectors
    one_hot_a = F.one_hot(torch.tensor(digit_a), num_classes=num_classes).float()
    one_hot_b = F.one_hot(torch.tensor(digit_b), num_classes=num_classes).float()

    # Mix them
    mixed = alpha * one_hot_a + (1 - alpha) * one_hot_b

    # Scale to [-1, 1] range (same as training)
    mixed = mixed * 2 - 1

    return mixed


def generate_mixed_digit(model, device, digit_a, digit_b, alpha, num_steps=200):
    """Generate a single mixed digit."""
    model.eval()

    mixed_label = create_mixed_label(digit_a, digit_b, alpha).unsqueeze(0).to(device)
    image = model.generate_from_label(mixed_label, num_steps)

    return image[0, 0].cpu().numpy()


def visualize_pairwise_mixing(model, device, digit_pairs, num_steps=200,
                              save_path='mixed_digits_pairwise.png'):
    """
    Visualize 50/50 mixes for multiple digit pairs.

    Args:
        digit_pairs: List of (digit_a, digit_b) tuples
        num_steps: ODE integration steps
    """
    model.eval()

    num_pairs = len(digit_pairs)
    num_samples = 8  # Samples per pair

    fig, axes = plt.subplots(num_pairs, num_samples + 2, figsize=(2*(num_samples+2), 2*num_pairs))

    if num_pairs == 1:
        axes = axes.reshape(1, -1)

    print(f"Generating pairwise mixes with {num_steps} steps...")

    for row_idx, (digit_a, digit_b) in enumerate(tqdm(digit_pairs, desc="Digit pairs")):
        # Generate pure digit A
        label_a = create_mixed_label(digit_a, digit_a, 1.0).unsqueeze(0).to(device)
        img_a = model.generate_from_label(label_a, num_steps)[0, 0].cpu().numpy()
        axes[row_idx, 0].imshow(img_a, cmap='gray')
        axes[row_idx, 0].set_title(f'{digit_a}', fontsize=12, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Generate mixed digits (50/50)
        for sample_idx in range(num_samples):
            mixed_label = create_mixed_label(digit_a, digit_b, 0.5).unsqueeze(0).to(device)
            img_mixed = model.generate_from_label(mixed_label, num_steps)[0, 0].cpu().numpy()
            axes[row_idx, sample_idx + 1].imshow(img_mixed, cmap='gray')
            if sample_idx == num_samples // 2 - 1:
                axes[row_idx, sample_idx + 1].set_title(f'{digit_a}+{digit_b}\n(50/50)',
                                                        fontsize=10)
            axes[row_idx, sample_idx + 1].axis('off')

        # Generate pure digit B
        label_b = create_mixed_label(digit_b, digit_b, 1.0).unsqueeze(0).to(device)
        img_b = model.generate_from_label(label_b, num_steps)[0, 0].cpu().numpy()
        axes[row_idx, num_samples + 1].imshow(img_b, cmap='gray')
        axes[row_idx, num_samples + 1].set_title(f'{digit_b}', fontsize=12, fontweight='bold')
        axes[row_idx, num_samples + 1].axis('off')

    plt.suptitle('Mixed Digit Generation: 50/50 Class Label Mixing', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def visualize_interpolation_path(model, device, digit_a, digit_b, num_steps_interp=11,
                                 num_steps_ode=200, num_samples=5,
                                 save_path='interpolation_path.png'):
    """
    Visualize smooth interpolation from digit A to digit B.

    Args:
        digit_a: Starting digit
        digit_b: Ending digit
        num_steps_interp: Number of interpolation steps (including endpoints)
        num_steps_ode: ODE integration steps
        num_samples: Number of sample paths to generate
    """
    model.eval()

    # Create interpolation alphas from 1 (pure A) to 0 (pure B)
    alphas = np.linspace(1, 0, num_steps_interp)

    fig, axes = plt.subplots(num_samples, num_steps_interp,
                            figsize=(1.5*num_steps_interp, 1.5*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    print(f"Generating interpolation paths from {digit_a} to {digit_b} with {num_steps_ode} steps...")

    for sample_idx in tqdm(range(num_samples), desc="Sample paths"):
        for alpha_idx, alpha in enumerate(alphas):
            mixed_label = create_mixed_label(digit_a, digit_b, alpha).unsqueeze(0).to(device)
            image = model.generate_from_label(mixed_label, num_steps_ode)[0, 0].cpu().numpy()

            axes[sample_idx, alpha_idx].imshow(image, cmap='gray')
            axes[sample_idx, alpha_idx].axis('off')

            # Add title on first row
            if sample_idx == 0:
                if alpha_idx == 0:
                    title = f'{digit_a}\n(α=1.0)'
                elif alpha_idx == num_steps_interp - 1:
                    title = f'{digit_b}\n(α=0.0)'
                else:
                    title = f'α={alpha:.2f}'
                axes[sample_idx, alpha_idx].set_title(title, fontsize=9)

    plt.suptitle(f'Class Label Interpolation: {digit_a} → {digit_b}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def visualize_mixing_grid(model, device, num_steps=200, save_path='mixing_grid.png'):
    """
    Create a large grid showing all possible digit pair mixtures.

    For each (i, j) where i < j, show a 50/50 mix of digits i and j.
    """
    model.eval()

    # Create list of unique pairs
    pairs = [(i, j) for i in range(10) for j in range(i+1, 10)]

    print(f"Generating mixing grid for all {len(pairs)} digit pairs with {num_steps} steps...")

    # Create grid: 10x10 where upper triangle shows mixes
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))

    for i in range(10):
        for j in range(10):
            if i == j:
                # Diagonal: pure digits
                label = create_mixed_label(i, i, 1.0).unsqueeze(0).to(device)
                image = model.generate_from_label(label, num_steps)[0, 0].cpu().numpy()
                axes[i, j].imshow(image, cmap='gray')
                axes[i, j].set_title(f'{i}', fontsize=14, fontweight='bold')
            elif i < j:
                # Upper triangle: i+j mix
                mixed_label = create_mixed_label(i, j, 0.5).unsqueeze(0).to(device)
                image = model.generate_from_label(mixed_label, num_steps)[0, 0].cpu().numpy()
                axes[i, j].imshow(image, cmap='gray')
                axes[i, j].set_title(f'{i}+{j}', fontsize=10)
            else:
                # Lower triangle: empty
                axes[i, j].axis('off')
                continue

            axes[i, j].axis('off')

    plt.suptitle('Complete Digit Mixing Grid (50/50 mixes)', fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def visualize_triple_mix(model, device, digits, num_steps=200, num_samples=16,
                        save_path='triple_mix.png'):
    """
    Mix THREE digits together equally (33.3% each).

    Args:
        digits: List of 3 digits to mix
        num_steps: ODE integration steps
        num_samples: Number of samples to generate
    """
    model.eval()

    assert len(digits) == 3, "Must provide exactly 3 digits"

    print(f"Generating triple mix {digits[0]}+{digits[1]}+{digits[2]} with {num_steps} steps...")

    # Create triple mix: equal weights
    one_hot_a = F.one_hot(torch.tensor(digits[0]), num_classes=10).float()
    one_hot_b = F.one_hot(torch.tensor(digits[1]), num_classes=10).float()
    one_hot_c = F.one_hot(torch.tensor(digits[2]), num_classes=10).float()

    mixed = (one_hot_a + one_hot_b + one_hot_c) / 3.0
    mixed = mixed * 2 - 1  # Scale to [-1, 1]

    # Generate multiple samples
    rows = 4
    cols = num_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()

    for i in tqdm(range(num_samples), desc="Generating samples"):
        mixed_label = mixed.unsqueeze(0).to(device)
        image = model.generate_from_label(mixed_label, num_steps)[0, 0].cpu().numpy()
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')

    plt.suptitle(f'Triple Mix: {digits[0]}+{digits[1]}+{digits[2]} (33.3% each)',
                 fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate mixed digits by interpolating class labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate pairwise 50/50 mixes
  python generate_mixed_digits.py --mode pairwise

  # Smooth interpolation from 3 to 8
  python generate_mixed_digits.py --mode interpolate --digits 3 8

  # Complete mixing grid
  python generate_mixed_digits.py --mode grid

  # Mix three digits equally
  python generate_mixed_digits.py --mode triple --digits 1 4 7
        """
    )

    parser.add_argument('--mode', type=str,
                       choices=['pairwise', 'interpolate', 'grid', 'triple'],
                       default='pairwise',
                       help='Generation mode')

    parser.add_argument('--model-path', type=str, default='models/best_flow_model.pt',
                       help='Path to trained model')

    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Model hidden dimension')

    parser.add_argument('--steps', type=int, default=200,
                       help='Number of ODE integration steps')

    parser.add_argument('--digits', type=int, nargs='+',
                       help='Digits to mix (mode dependent)')

    parser.add_argument('--output-dir', type=str, default='mixed_digits',
                       help='Output directory')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("Mixed Digit Generation - Class Label Interpolation")
    print("="*70)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"ODE steps: {args.steps}")
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

    print()

    # Generate based on mode
    if args.mode == 'pairwise':
        # Interesting digit pairs to mix
        pairs = [
            (3, 8),  # Curvy digits
            (1, 7),  # Lines and angles
            (6, 9),  # Rotational relationship
            (0, 8),  # Both have loops
            (4, 9),  # Structural similarity
        ]
        visualize_pairwise_mixing(
            model, device, pairs, args.steps,
            save_path=output_dir / 'pairwise_mixing.png'
        )

    elif args.mode == 'interpolate':
        if args.digits is None or len(args.digits) != 2:
            print("Error: --mode interpolate requires exactly 2 digits")
            print("Example: --mode interpolate --digits 3 8")
            return

        visualize_interpolation_path(
            model, device, args.digits[0], args.digits[1],
            num_steps_interp=11, num_steps_ode=args.steps, num_samples=5,
            save_path=output_dir / f'interpolate_{args.digits[0]}to{args.digits[1]}.png'
        )

    elif args.mode == 'grid':
        visualize_mixing_grid(
            model, device, args.steps,
            save_path=output_dir / 'complete_mixing_grid.png'
        )

    elif args.mode == 'triple':
        if args.digits is None or len(args.digits) != 3:
            print("Error: --mode triple requires exactly 3 digits")
            print("Example: --mode triple --digits 1 4 7")
            return

        visualize_triple_mix(
            model, device, args.digits, args.steps, num_samples=16,
            save_path=output_dir / f'triple_mix_{args.digits[0]}_{args.digits[1]}_{args.digits[2]}.png'
        )

    print()
    print("="*70)
    print("Generation complete! 🎨")
    print(f"Output saved to: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()
