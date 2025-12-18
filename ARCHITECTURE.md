# Joint Diffusion Architecture

## Overview

This document explains the key differences between the original conditional diffusion approach and the new joint diffusion approach, and how it implements the theoretical framework.

## Theoretical Foundation

### Original Approach: Conditional Diffusion
The original model learned `P(label | image)` directly by:
1. Extracting fixed image embeddings via CNN
2. Diffusing only in label space, conditioned on these embeddings
3. Injecting image information via additive conditioning

**Limitation**: This doesn't model the full joint distribution and treats the image as a fixed condition rather than part of the generative process.

### New Approach: Joint Diffusion with Marginalization
The new model learns `P(image, label)` and derives `P(label | image)` via Bayes' rule:

```
P(label | image) = P(image, label) / P(image)
                 = P(image, label) / Σ_l P(image, l)
```

The key insight: **diffusion models learn score functions**, which are gradients of log-probability. When we differentiate the conditional:

```
∇_label log P(label | image) = ∇_label log P(image, label) - ∇_label log P(image)
                              = ∇_label log P(image, label)  [P(image) doesn't depend on label!]
```

This means our joint model already knows the conditional score. At inference:
- **Hold the image constant** (no noise, no denoising)
- **Denoise only the label** starting from pure noise
- The model naturally samples from `P(label | image)` without computing the intractable denominator

## Architectural Differences

### 1. Separate Denoising Paths

**Original**: Single denoiser for labels only
```python
LabelDenoiser(noisy_labels, t, image_embedding)
```

**New**: Dual denoisers for both modalities
```python
ImageDenoiser(noisy_images, t) -> image_features
LabelDenoiser(noisy_labels, t) -> label_features
```

### 2. Cross-Attention vs Additive Conditioning

**Original**: Additive injection in middle layers
```python
h = layer(h + img_cond)  # Simple addition
```

**New**: Bidirectional cross-attention
```python
# Both modalities attend to each other
img_feats = CrossAttention(query=img_feats, context=label_feats)
label_feats = CrossAttention(query=label_feats, context=img_feats)
```

This allows the network to:
- Learn what information to exchange between modalities
- Build richer joint representations
- Capture complex dependencies between images and labels

### 3. Training Process

**Original**: Diffuse labels only
```python
label_noisy = add_noise(one_hot_labels, t)
image_embed = CNN(clean_images)  # Fixed, no gradient flow to image processing
predicted_noise = denoiser(label_noisy, t, image_embed)
loss = MSE(predicted_noise, actual_noise)
```

**New**: Diffuse both modalities jointly
```python
img_noisy = add_noise(images, t)
label_noisy = add_noise(one_hot_labels, t)
pred_img_noise, pred_label_noise = denoiser(img_noisy, label_noisy, t)
loss = MSE(pred_img_noise, img_noise) + MSE(pred_label_noise, label_noise)
```

### 4. Inference Process

**Original**: Denoise from noise → labels, always conditioned on image
```python
label_t = random_noise()
for t in timesteps:
    image_embed = CNN(image)  # Computed once
    label_t = denoise_step(label_t, t, image_embed)
```

**New**: Joint denoising with selective freezing
```python
label_t = random_noise()
img_t = clean_image  # No noise!
for t in timesteps:
    # Only denoise label, image stays clean
    img_t, label_t = denoise_step(
        img_t, label_t, t,
        denoise_image=False,  # Freeze image
        denoise_label=True     # Only update label
    )
```

## Key Implementation Details

### Cross-Attention Module
The cross-attention layers implement multi-head attention:
```python
class CrossAttention:
    def forward(x, context):
        # x: queries from one modality
        # context: keys/values from other modality
        Q = project(x)
        K, V = project(context)
        attention = softmax(Q @ K^T / √d)
        output = attention @ V
        return LayerNorm(x + output)  # Residual connection
```

### Spatial Features for Images
Images are processed through convolutions and converted to spatial sequences:
```
(B, 1, 28, 28) → conv → (B, 256, 7, 7) → flatten → (B, 49, 256)
```

Each of the 49 positions represents a 4×4 patch of the original image, allowing fine-grained cross-attention with label features.

### Label Feature Expansion
Labels are expanded from 10-d vectors to sequences:
```
(B, 10) → project → (B, 8, 256)
```

This creates multiple "positions" in label space that can attend to different spatial locations in the image.

### Inference: Marginalizing Over the Prior
At inference time for classification:

1. **Start**: `label_t ~ N(0, I)`, `img_t = clean_image`
2. **Iterate**: Denoise only label using full joint model
3. **Result**: `label_0` converges to high-probability label for that image

The magic: by keeping the image clean, we're effectively asking "given this specific image, what labels have high joint probability?" The model denoises toward `P(label | image)` automatically.

## Why This Matters

### Advantages of Joint Modeling

1. **Richer Representations**: Cross-attention allows the model to learn complex image-label relationships
2. **True Generative Model**: Can sample novel (image, label) pairs, not just classify
3. **Theoretical Elegance**: Directly implements the score-based marginalization framework
4. **Flexibility**: Same model can:
   - Generate images given labels (denoise image, freeze label)
   - Classify images (denoise label, freeze image)
   - Generate joint samples (denoise both)

### When to Use Each Approach

**Original (Conditional Diffusion)**:
- Simpler, faster to train
- Lower memory requirements
- Good when you only need classification

**New (Joint Diffusion)**:
- More powerful for complex domains
- Enables multi-task learning (generation + classification)
- Better aligns with world model theory
- Essential when you need the full joint distribution

## Scaling to Complex Domains

For larger problems (e.g., Minecraft as mentioned in the blog post):

```python
# Same framework, different modalities
game_state_noisy = add_noise(game_state, t)
action_noisy = add_noise(action, t)

# Cross-attention between state and action
state_feats = StateEncoder(game_state_noisy, t)
action_feats = ActionEncoder(action_noisy, t)

state_feats = CrossAttention(state_feats, action_feats)
action_feats = CrossAttention(action_feats, state_feats)

# At inference: freeze observed state, denoise action
```

This is the foundation of world model agents like Dreamer, which then add:
- Recurrent state for temporal coherence
- Value functions for planning
- Policy gradients for optimization
- Model-based rollouts for imagination

But the core insight remains: **learn the joint distribution, then marginalize** to answer conditional queries without explicit normalization.
