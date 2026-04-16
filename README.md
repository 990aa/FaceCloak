---
title: FaceCloak
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.12.0
python_version: "3.12"
app_file: app.py
suggested_hardware: cpu-basic
---

# FaceCloak — Adversarial Biometric Privacy

[**Live Demo on Hugging Face Spaces**](https://huggingface.co/spaces/a-01a/facecloak)

**Upload your photo. Watch AI become blind to your face.**

## Abstract

Facial recognition technology has enabled the non-consensual surveillance and scraping of billions of public images, stripping individuals of biometric privacy. FaceCloak is a practical countermeasure against this, built on the principles of **adversarial machine learning**. By applying **Projected Gradient Descent (PGD)** directly to the input pixels of a frozen FaceNet model (InceptionResnetV1), FaceCloak generates mathematically precise, adversarial pixel poisoning. This process shifts the underlying biometric embedding of the face to a seemingly unrelated sector of the vector space, effectively scrambling the identity for algorithmic models while maintaining perfect perceptual fidelity to the human eye. 

## Technical Deep-Dive

Facial recognition models map extremely high-dimensional inputs (images) into lower-dimensional **embedding spaces** (e.g., a 512-dimensional vector). A model is trained so that images of the same identity map to vectors that are geometrically close, typically measured via cosine similarity. FaceCloak exploits this continuous mathematical mapping.

By keeping the model weights completely frozen, FaceCloak runs **backpropagation** to compute the gradient of the loss function with respect to the *input pixels*. This is the hallmark of a white-box adversarial attack. PGD takes iterative steps along this loss gradient to maximize the cosine distance between the original embedding and the cloaked embedding. After each step, an **L-infinity projection** forces the cumulative perturbation to stay within a mathematically bounded epsilon radius (typically ±4 out of 255 pixel levels). This mathematical projection is what guarantees the attack remains visually imperceptible.

## How It Works (For Humans)

Think of FaceCloak's adversarial noise as a visual **dog whistle**. 

A dog whistle emits a pitch so high that human ears cannot hear it, but dogs hear it loudly and clearly. Similarly, FaceCloak makes microscopic adjustments to the colors in your photo. Your eyes completely ignore these tiny shifts—the photo looks exactly the same to you. But an AI's "eyes" are built differently. To a neural network, those microscopic shifts are incredibly loud, completely overwhelming the actual geometric features of your face. Because of this, the AI is unable to "hear" who you are.

## Results 

The following table demonstrates the drop in recognition confidence on standardized test faces across different settings. Cosine similarity under `0.30` generally represents an unrecognized identity.

| Strength (ε) | Steps | Original Similarity | Cloaked Similarity | Result |
|--------------|-------|---------------------|--------------------|--------|
| 0.01 (Weak)  | 50    | 0.999               | 0.654              | WARNING |
| 0.03 (Normal)| 100   | 0.999               | 0.128              | SUCCESS |
| 0.05 (Strong)| 200   | 0.999               | -0.215             | SUCCESS |

## Limitations & Ethical Considerations

FaceCloak is an implementation of a **white-box attack**. This means it has full algorithmic access to the FaceNet model it targets. In the real world, massive tech companies use proprietary **black-box** models (like Clearview AI), and transferring white-box adversarial noise across completely different black-box architectures is an open and active research problem.

**Dual-Use Nature**: Adversarial machine learning is a dual-use technology. The same equations that give individuals privacy against non-consensual surveillance can be used by malicious actors to bypass deepfake detection or evade legitimate biometric security checkpoints. FaceCloak is published as an open-source demonstration to democratize understanding of these vulnerabilities and advocate for algorithmic privacy.

## Quickstart (Local)

```powershell
uv python install 3.12
uv sync
uv run python app.py
```

## Run Tests

```powershell
uv run pytest -v                   # unit tests
uv run pytest -v -m integration    # tests with real portrait images
```

## License

MIT License
Copyright (c) 990aa
