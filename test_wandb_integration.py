"""Test wandb integration with visualization functions."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from joint_diffusion_classifier import (
    JointDiffusion,
    visualize_classification,
    visualize_joint_generation,
    visualize_conditional_generation,
    visualize_denoising_process
)
import wandb
import matplotlib.pyplot as plt

def test_wandb_integration():
    """Test that all visualization functions work and can be logged to wandb."""

    # Initialize wandb in offline mode for testing
    wandb.init(
        project="mnist-joint-diffusion-test",
        mode="offline",  # Don't sync to cloud during test
        config={"test": True}
    )

    print("Creating model...")
    model = JointDiffusion(num_timesteps=100)
    device = torch.device('cpu')
    model = model.to(device)

    # Create small test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("\nTesting visualization functions...")

    # Test 1: Classification
    print("1. Testing classification visualization...")
    fig = visualize_classification(model, test_loader, device, num_samples=4, num_inference_steps=5)
    wandb.log({"test/classification": wandb.Image(fig)})
    plt.close(fig)
    print("   ✓ Classification visualization works")

    # Test 2: Joint generation
    print("2. Testing joint generation visualization...")
    fig = visualize_joint_generation(model, device, num_samples=4, num_inference_steps=5)
    wandb.log({"test/joint_generation": wandb.Image(fig)})
    plt.close(fig)
    print("   ✓ Joint generation visualization works")

    # Test 3: Conditional generation
    print("3. Testing conditional generation visualization...")
    fig = visualize_conditional_generation(model, test_loader, device, num_samples=4, num_inference_steps=5)
    wandb.log({"test/conditional_generation": wandb.Image(fig)})
    plt.close(fig)
    print("   ✓ Conditional generation visualization works")

    # Test 4: Denoising process
    print("4. Testing denoising process visualization...")
    fig = visualize_denoising_process(model, test_loader, device, num_steps_to_show=4)
    wandb.log({"test/denoising_process": wandb.Image(fig)})
    plt.close(fig)
    print("   ✓ Denoising process visualization works")

    # Test 5: Metrics logging
    print("5. Testing metrics logging...")
    wandb.log({
        "train/loss": 1.234,
        "train/img_loss": 0.567,
        "train/label_loss": 0.667,
        "test/accuracy": 0.123,
    })
    print("   ✓ Metrics logging works")

    wandb.finish()
    print("\n✅ All wandb integration tests passed!")

if __name__ == "__main__":
    test_wandb_integration()
