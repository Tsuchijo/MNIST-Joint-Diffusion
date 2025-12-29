"""
Compare Joint vs Conditional-Only Training

This script loads both models and compares their conditional generation quality:
1. Joint model (trained on P(image, label))
2. Conditional-only model (trained only on P(image|label))

Generates side-by-side visualizations to compare:
- Generation quality
- Diversity
- Consistency across digits
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

from joint_diffusion_classifier import JointFlowMatching
from train_conditional_only import ConditionalOnlyFlowMatching


def visualize_comparison(joint_model, cond_model, device, num_per_class=8, num_steps=200,
                        save_path='model_comparison.png'):
    """Generate side-by-side comparison of both models."""

    joint_model.eval()
    cond_model.eval()

    # Create figure with two sections
    fig = plt.figure(figsize=(24, 2.5*num_per_class))
    gs = fig.add_gridspec(num_per_class, 20, hspace=0.3, wspace=0.2)

    print(f"Generating comparison with {num_steps} steps...")
    for digit in tqdm(range(10), desc="Generating digits"):
        labels = torch.full((num_per_class,), digit, device=device)

        # Generate from both models
        joint_images = joint_model.generate_from_label(labels, num_steps)
        cond_images = cond_model.generate(labels, num_steps)

        for sample_idx in range(num_per_class):
            # Joint model (left half)
            ax_joint = fig.add_subplot(gs[sample_idx, digit])
            ax_joint.imshow(joint_images[sample_idx, 0].cpu().numpy(), cmap='gray')
            ax_joint.axis('off')
            if sample_idx == 0:
                ax_joint.set_title(f'{digit}', fontsize=14, fontweight='bold')

            # Conditional model (right half)
            ax_cond = fig.add_subplot(gs[sample_idx, digit + 10])
            ax_cond.imshow(cond_images[sample_idx, 0].cpu().numpy(), cmap='gray')
            ax_cond.axis('off')
            if sample_idx == 0:
                ax_cond.set_title(f'{digit}', fontsize=14, fontweight='bold')

    # Add section labels
    fig.text(0.25, 0.98, 'Joint Training P(image, label)',
             ha='center', fontsize=18, fontweight='bold')
    fig.text(0.75, 0.98, 'Conditional-Only Training P(image|label)',
             ha='center', fontsize=18, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison to {save_path}")
    plt.close()


def visualize_diversity_comparison(joint_model, cond_model, device, digit=7,
                                   num_samples=32, num_steps=200,
                                   save_path='diversity_comparison.png'):
    """Compare diversity in generation for a single digit."""

    joint_model.eval()
    cond_model.eval()

    labels = torch.full((num_samples,), digit, device=device)

    print(f"Generating diversity comparison for digit {digit} with {num_steps} steps...")
    joint_images = joint_model.generate_from_label(labels, num_steps)
    cond_images = cond_model.generate(labels, num_steps)

    # Create grid
    rows = 8
    cols = num_samples // rows

    fig, axes = plt.subplots(rows, cols*2 + 1, figsize=(2*(cols*2 + 1), 2*rows))

    for i in range(num_samples):
        row = i // cols
        col = i % cols

        # Joint model (left side)
        axes[row, col].imshow(joint_images[i, 0].cpu().numpy(), cmap='gray')
        axes[row, col].axis('off')

        # Conditional model (right side)
        axes[row, col + cols + 1].imshow(cond_images[i, 0].cpu().numpy(), cmap='gray')
        axes[row, col + cols + 1].axis('off')

    # Hide middle column
    for row in range(rows):
        axes[row, cols].axis('off')

    # Add labels
    fig.text(0.25, 0.98, f'Joint Training - Digit {digit}',
             ha='center', fontsize=16, fontweight='bold')
    fig.text(0.75, 0.98, f'Conditional-Only - Digit {digit}',
             ha='center', fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved diversity comparison to {save_path}")
    plt.close()


def compute_statistics(joint_model, cond_model, device, num_samples=1000, num_steps=200):
    """Compute generation statistics for both models."""

    joint_model.eval()
    cond_model.eval()

    print(f"\nComputing statistics over {num_samples} samples with {num_steps} steps...")

    stats = {
        'joint': {'mean': [], 'std': []},
        'cond': {'mean': [], 'std': []}
    }

    for digit in tqdm(range(10), desc="Computing stats per digit"):
        labels = torch.full((num_samples,), digit, device=device)

        # Generate samples
        joint_images = joint_model.generate_from_label(labels, num_steps)
        cond_images = cond_model.generate(labels, num_steps)

        # Compute statistics
        stats['joint']['mean'].append(joint_images.mean().item())
        stats['joint']['std'].append(joint_images.std().item())
        stats['cond']['mean'].append(cond_images.mean().item())
        stats['cond']['std'].append(cond_images.std().item())

    return stats


def plot_statistics(stats, save_path='statistics_comparison.png'):
    """Plot generation statistics comparison."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    digits = list(range(10))

    # Mean comparison
    axes[0].plot(digits, stats['joint']['mean'], 'o-', label='Joint Training', linewidth=2)
    axes[0].plot(digits, stats['cond']['mean'], 's-', label='Conditional-Only', linewidth=2)
    axes[0].set_xlabel('Digit', fontsize=12)
    axes[0].set_ylabel('Mean Pixel Value', fontsize=12)
    axes[0].set_title('Generated Image Mean', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Std comparison
    axes[1].plot(digits, stats['joint']['std'], 'o-', label='Joint Training', linewidth=2)
    axes[1].plot(digits, stats['cond']['std'], 's-', label='Conditional-Only', linewidth=2)
    axes[1].set_xlabel('Digit', fontsize=12)
    axes[1].set_ylabel('Std Dev', fontsize=12)
    axes[1].set_title('Generated Image Variance', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved statistics to {save_path}")
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("Model Comparison: Joint vs Conditional-Only Training")
    print("="*70)
    print(f"Device: {device}\n")

    # Create output directory
    output_dir = Path('comparison_outputs')
    output_dir.mkdir(exist_ok=True)

    # Load models
    print("Loading models...")
    hidden_dim = 128

    joint_model = JointFlowMatching(hidden_dim=hidden_dim).to(device)
    try:
        joint_model.load_state_dict(torch.load('best_flow_model.pt', map_location=device))
        print("✓ Loaded joint model (best_flow_model.pt)")
    except FileNotFoundError:
        print("✗ Error: best_flow_model.pt not found!")
        return

    cond_model = ConditionalOnlyFlowMatching(hidden_dim=hidden_dim).to(device)
    try:
        cond_model.load_state_dict(torch.load('best_conditional_model.pt', map_location=device))
        print("✓ Loaded conditional-only model (best_conditional_model.pt)")
    except FileNotFoundError:
        print("✗ Error: best_conditional_model.pt not found!")
        print("Please train the conditional-only model first using train_conditional_only.py")
        return

    print()

    # Settings
    num_steps = 200
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Side-by-side generation comparison
    print("="*70)
    print("1. Generation Quality Comparison")
    print("="*70)
    visualize_comparison(
        joint_model, cond_model, device,
        num_per_class=8, num_steps=num_steps,
        save_path=output_dir / 'generation_comparison.png'
    )
    print()

    # 2. Diversity comparison
    print("="*70)
    print("2. Diversity Comparison")
    print("="*70)
    visualize_diversity_comparison(
        joint_model, cond_model, device,
        digit=7, num_samples=32, num_steps=num_steps,
        save_path=output_dir / 'diversity_comparison.png'
    )
    print()

    # 3. Statistical comparison
    print("="*70)
    print("3. Statistical Analysis")
    print("="*70)
    stats = compute_statistics(
        joint_model, cond_model, device,
        num_samples=100, num_steps=num_steps
    )
    plot_statistics(stats, save_path=output_dir / 'statistics_comparison.png')
    print()

    # 4. High-quality showcase
    print("="*70)
    print("4. High-Quality Showcase (500 steps)")
    print("="*70)
    visualize_comparison(
        joint_model, cond_model, device,
        num_per_class=10, num_steps=500,
        save_path=output_dir / 'generation_comparison_hq.png'
    )
    print()

    print("="*70)
    print("Comparison complete! 🎯")
    print(f"All outputs saved to: {output_dir.absolute()}")
    print("="*70)
    print("\nGenerated files:")
    print("  - generation_comparison.png      : Side-by-side generation (200 steps)")
    print("  - generation_comparison_hq.png   : Side-by-side generation (500 steps)")
    print("  - diversity_comparison.png       : Diversity for digit 7")
    print("  - statistics_comparison.png      : Statistical analysis")


if __name__ == "__main__":
    main()
