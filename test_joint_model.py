"""Quick test to verify the joint diffusion model works."""

import torch
from joint_diffusion_classifier import JointDiffusion

def test_model():
    print("Creating model...")
    model = JointDiffusion(num_timesteps=100)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))

    print("Running p_losses...")
    loss, img_loss, label_loss = model.p_losses(images, labels)
    print(f"Loss: {loss.item():.4f} (img: {img_loss:.4f}, label: {label_loss:.4f})")

    # Test backward pass
    print("\nTesting backward pass...")
    loss.backward()
    print("Backward pass successful!")

    # Test classification
    print("\nTesting classification...")
    model.eval()
    with torch.no_grad():
        predictions = model.predict(images, num_inference_steps=10)
    print(f"Predictions: {predictions}")
    print(f"True labels: {labels}")

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_model()
