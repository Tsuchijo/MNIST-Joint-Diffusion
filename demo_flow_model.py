"""
Demo and Evaluation Script for Joint Flow Matching Model

This script provides a comprehensive evaluation of the trained model:
1. Classification accuracy with confusion matrix
2. Conditional generation (P(image|label)) for all digits
3. Joint generation (P(image, label)) samples
4. Flow trajectory visualization during classification

Uses high-quality settings with many integration steps for best results.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from pathlib import Path

# Import model architecture
from joint_diffusion_classifier import JointFlowMatching


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Model settings
    model_path = 'models/best_flow_model.pt'
    hidden_dim = 256

    # Evaluation settings
    num_steps_high_quality = 200  # High quality integration
    num_steps_ultra_quality = 500  # Ultra quality for final showcase

    # Data settings
    test_subset_size = 1000  # Size of test subset for detailed evaluation
    batch_size = 256

    # Generation settings
    num_samples_per_class = 8  # For conditional generation
    num_joint_samples = 64  # For joint generation

    # Visualization
    random_seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_confusion_matrix(model, dataloader, device, num_steps):
    """Compute confusion matrix for classification."""
    model.eval()
    num_classes = 10
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    print(f"Computing predictions with {num_steps} integration steps...")
    for images, labels in tqdm(dataloader, desc="Computing confusion matrix"):
        images, labels = images.to(device), labels.to(device)
        predictions = model.predict(images, num_steps)

        for true_label, pred_label in zip(labels, predictions):
            confusion[true_label.item(), pred_label.item()] += 1

    return confusion.numpy()


def evaluate_accuracy(model, dataloader, device, num_steps):
    """Evaluate classification accuracy."""
    model.eval()
    correct = 0
    total = 0

    print(f"Evaluating accuracy with {num_steps} integration steps...")
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        predictions = model.predict(images, num_steps)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_confusion_matrix(confusion, accuracy, num_steps, save_path='confusion_matrix.png'):
    """Plot confusion matrix with accuracy."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize by row (true labels) to get percentages
    confusion_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

    sns.heatmap(confusion_normalized, annot=True, fmt='.2f', cmap='Blues',
                square=True, cbar_kws={'label': 'Proportion'}, ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.4f}, Steps: {num_steps})',
                 fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()

    return fig


def plot_classification_examples(model, dataloader, device, num_steps,
                                 num_examples=32, save_path='classification_examples.png'):
    """Visualize classification examples with predictions and confidence."""
    model.eval()

    images, labels = next(iter(dataloader))
    images, labels = images[:num_examples].to(device), labels[:num_examples].to(device)

    probs = model.classify(images, num_steps)
    predictions = probs.argmax(dim=-1)

    # Create grid
    rows = int(np.sqrt(num_examples))
    cols = (num_examples + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2.5))
    axes = axes.flatten()

    for i in range(num_examples):
        img = images[i].cpu().numpy()[0]
        pred = predictions[i].item()
        true = labels[i].item()
        conf = probs[i, pred].item()

        axes[i].imshow(img, cmap='gray')

        # Color code: green for correct, red for incorrect
        color = 'green' if pred == true else 'red'
        axes[i].set_title(f'True: {true}\nPred: {pred} ({conf:.3f})',
                         color=color, fontsize=9)
        axes[i].axis('off')

    # Hide extra subplots
    for i in range(num_examples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Classification Examples ({num_steps} steps)', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved classification examples to {save_path}")
    plt.close()

    return fig


def plot_conditional_generation(model, device, num_steps, num_per_class=8,
                                save_path='conditional_generation.png'):
    """Generate and visualize images conditioned on each digit class."""
    model.eval()

    fig, axes = plt.subplots(num_per_class, 10, figsize=(20, 2*num_per_class))

    print(f"Generating conditional samples with {num_steps} integration steps...")
    for digit in tqdm(range(10), desc="Conditional generation"):
        labels = torch.full((num_per_class,), digit, device=device)
        images = model.generate_from_label(labels, num_steps)

        for j in range(num_per_class):
            axes[j, digit].imshow(images[j, 0].cpu().numpy(), cmap='gray')
            axes[j, digit].axis('off')
            if j == 0:
                axes[j, digit].set_title(f'{digit}', fontsize=14, fontweight='bold')

    plt.suptitle(f'Conditional Generation: P(image | label) [{num_steps} steps]',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved conditional generation to {save_path}")
    plt.close()

    return fig


def plot_joint_generation(model, device, num_steps, num_samples=64,
                         save_path='joint_generation.png'):
    """Generate and visualize samples from joint distribution."""
    model.eval()

    print(f"Generating joint samples with {num_steps} integration steps...")
    images, label_probs = model.sample_joint(num_samples, device, num_steps)
    labels = label_probs.argmax(dim=-1)

    # Create grid
    rows = int(np.sqrt(num_samples))
    cols = (num_samples + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2.5))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].cpu().numpy()[0]
        label = labels[i].item()
        conf = label_probs[i, label].item()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{label} ({conf:.3f})', fontsize=10)
        axes[i].axis('off')

    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Joint Generation: P(image, label) [{num_steps} steps]',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved joint generation to {save_path}")
    plt.close()

    return fig


def plot_flow_trajectory(model, dataloader, device, num_steps,
                        num_snapshots=10, save_path='flow_trajectory.png'):
    """Visualize the flow trajectory during classification."""
    model.eval()

    images, true_labels = next(iter(dataloader))

    # Select 4 diverse examples
    selected_indices = [0, 1, 2, 3]

    fig, axes = plt.subplots(4, num_snapshots + 1, figsize=(2.5*(num_snapshots+1), 10))

    print(f"Computing flow trajectories with {num_steps} integration steps...")
    for row_idx, sample_idx in enumerate(tqdm(selected_indices, desc="Flow trajectories")):
        image = images[sample_idx:sample_idx+1].to(device)
        true_label = true_labels[sample_idx].item()

        # Show original image in first column
        axes[row_idx, 0].imshow(images[sample_idx, 0].numpy(), cmap='gray')
        axes[row_idx, 0].set_title(f'Input\n(True: {true_label})', fontsize=10)
        axes[row_idx, 0].axis('off')

        # Track trajectory
        batch_size = 1
        label = torch.randn(batch_size, 10, device=device)
        t_fixed = torch.zeros(batch_size, device=device)

        dt = 1.0 / num_steps
        snapshot_interval = num_steps // num_snapshots

        trajectories = []
        times = []

        for i in range(num_steps):
            t = 1.0 - i * dt
            t_current = torch.full((batch_size,), t, device=device)

            _, v_label = model.velocity_field(image, label, t_fixed, t_current)
            label = label - v_label * dt

            if (i + 1) % snapshot_interval == 0:
                trajectories.append(label.clone())
                times.append(t - dt)

        # Plot probability distributions
        for col_idx, (traj, t) in enumerate(zip(trajectories, times), start=1):
            probs = F.softmax(traj[0], dim=-1).detach().cpu().numpy()

            axes[row_idx, col_idx].bar(range(10), probs, color='steelblue', alpha=0.7)
            axes[row_idx, col_idx].set_ylim([0, 1])
            axes[row_idx, col_idx].set_xticks(range(10))

            if col_idx == len(trajectories):
                pred = probs.argmax()
                color = 'green' if pred == true_label else 'red'
                axes[row_idx, col_idx].set_title(f't={t:.2f}\nPred: {pred}',
                                                fontsize=9, color=color)
            else:
                axes[row_idx, col_idx].set_title(f't={t:.2f}', fontsize=9)

            if row_idx == 3:  # Bottom row
                axes[row_idx, col_idx].set_xlabel('Digit', fontsize=8)
            else:
                axes[row_idx, col_idx].set_xticklabels([])

            if col_idx == 1:  # First prob column
                axes[row_idx, col_idx].set_ylabel('Probability', fontsize=8)

    plt.suptitle(f'Flow Trajectory During Classification [{num_steps} steps]',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved flow trajectory to {save_path}")
    plt.close()

    return fig


def plot_per_class_accuracy(confusion, save_path='per_class_accuracy.png'):
    """Plot per-class accuracy bar chart."""
    # Calculate per-class accuracy
    per_class_acc = np.diag(confusion) / confusion.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(per_class_acc)
    bars = ax.bar(range(10), per_class_acc, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Classification Accuracy', fontsize=14, pad=20)
    ax.set_xticks(range(10))
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-class accuracy to {save_path}")
    plt.close()

    return fig


# ============================================================================
# Main Demo
# ============================================================================

def main():
    config = Config()

    # Set random seed for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    print("="*70)
    print("Joint Flow Matching Model - Demo & Evaluation")
    print("="*70)
    print(f"Device: {config.device}")
    print(f"Model path: {config.model_path}")
    print()

    # Create output directory
    output_dir = Path('demo_outputs')
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    print("Loading MNIST test data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Create subset for detailed evaluation
    test_subset = Subset(test_dataset, range(config.test_subset_size))
    test_subset_loader = DataLoader(test_subset, batch_size=config.batch_size,
                                   shuffle=False, num_workers=4)

    # Full test set for final accuracy
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=4)

    # Load model
    print("Loading model...")
    model = JointFlowMatching(hidden_dim=config.hidden_dim).to(config.device)

    try:
        model.load_state_dict(torch.load(config.model_path, map_location=config.device))
        print(f"✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: Model file '{config.model_path}' not found!")
        print("Please train the model first using joint_diffusion_classifier.py")
        return

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # ========================================================================
    # 1. Classification Evaluation
    # ========================================================================
    print("="*70)
    print("1. CLASSIFICATION EVALUATION")
    print("="*70)

    # Compute confusion matrix on subset
    confusion = compute_confusion_matrix(
        model, test_subset_loader, config.device, config.num_steps_high_quality
    )

    subset_accuracy = np.diag(confusion).sum() / confusion.sum()
    print(f"\nSubset Accuracy ({config.test_subset_size} samples): {subset_accuracy:.4f}")

    # # Full test set accuracy
    # full_accuracy = evaluate_accuracy(
    #     model, test_loader, config.device, config.num_steps_high_quality
    # )
    # print(f"Full Test Accuracy (10000 samples): {full_accuracy:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(confusion, subset_accuracy, config.num_steps_high_quality,
                         save_path=output_dir / 'confusion_matrix.png')

    # Plot per-class accuracy
    plot_per_class_accuracy(confusion, save_path=output_dir / 'per_class_accuracy.png')

    # Plot classification examples
    plot_classification_examples(
        model, test_subset_loader, config.device, config.num_steps_high_quality,
        num_examples=32, save_path=output_dir / 'classification_examples.png'
    )

    print()

    # ========================================================================
    # 2. Conditional Generation
    # ========================================================================
    print("="*70)
    print("2. CONDITIONAL GENERATION: P(image | label)")
    print("="*70)

    # High quality
    plot_conditional_generation(
        model, config.device, config.num_steps_high_quality,
        num_per_class=config.num_samples_per_class,
        save_path=output_dir / 'conditional_generation_hq.png'
    )

    # # Ultra quality
    # plot_conditional_generation(
    #     model, config.device, config.num_steps_ultra_quality,
    #     num_per_class=config.num_samples_per_class,
    #     save_path=output_dir / 'conditional_generation_ultra.png'
    # )

    print()

    # ========================================================================
    # 3. Joint Generation
    # ========================================================================
    print("="*70)
    print("3. JOINT GENERATION: P(image, label)")
    print("="*70)

    # High quality
    plot_joint_generation(
        model, config.device, config.num_steps_high_quality,
        num_samples=config.num_joint_samples,
        save_path=output_dir / 'joint_generation_hq.png'
    )

    # # Ultra quality
    # plot_joint_generation(
    #     model, config.device, config.num_steps_ultra_quality,
    #     num_samples=config.num_joint_samples,
    #     save_path=output_dir / 'joint_generation_ultra.png'
    # )

    print()

    # ========================================================================
    # 4. Flow Trajectory Visualization
    # ========================================================================
    print("="*70)
    print("4. FLOW TRAJECTORY VISUALIZATION")
    print("="*70)

    plot_flow_trajectory(
        model, test_subset_loader, config.device, config.num_steps_high_quality,
        num_snapshots=10, save_path=output_dir / 'flow_trajectory.png'
    )

    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("="*70)
    print("SUMMARY")
    print("="*70)
    #print(f"Test Accuracy: {full_accuracy:.4f}")
    print(f"Integration steps (high quality): {config.num_steps_high_quality}")
    #print(f"Integration steps (ultra quality): {config.num_steps_ultra_quality}")
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")
    print()

    # Print per-class accuracy
    per_class_acc = np.diag(confusion) / confusion.sum(axis=1)
    print("Per-class accuracy:")
    for digit in range(10):
        print(f"  Digit {digit}: {per_class_acc[digit]:.4f}")

    print()
    print("Demo complete! 🎉")
    print("="*70)


if __name__ == "__main__":
    main()
