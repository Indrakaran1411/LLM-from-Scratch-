# 📘 Build a Large Language Model from Scratch
## Lecture 21 — GELU Activation Function & Feed-Forward Network
> **Video:** https://youtu.be/d_PiwZe8UF4

---

## 🗺️ What This Lecture Covers
1. What is an Activation Function and why we need one
2. ReLU — the common activation, and its problem
3. GELU — what it is and why GPT uses it
4. GELU vs ReLU — key differences
5. The Feed-Forward Network inside the Transformer Block
6. Full FeedForward class in PyTorch

---

## 1. What is an Activation Function and Why Do We Need One?

A neural network without activation functions is just a stack of linear transformations. No matter how many layers you add, the whole network collapses into a single linear equation — it can't learn complex patterns.

**Activation functions add non-linearity.** This is what allows neural networks to learn the complex patterns found in language.

```
Without activation:  output = W3 × (W2 × (W1 × input))
                            = (W3 × W2 × W1) × input
                            = single_matrix × input   ← just one linear layer!

With activation:     output = W2 × activation(W1 × input)
                            ← genuinely multi-layer, can learn complex patterns
```

---

## 2. ReLU — The Common Activation and Its Problem

**ReLU (Rectified Linear Unit)** is the most widely known activation function:

```
ReLU(x) = max(0, x)

   x < 0  →  output = 0
   x ≥ 0  →  output = x
```

```
     |        /
     |       /
     |      /
     |     /
─────|────/────→  x
     |   (flat at 0 for negatives)
```

### The Problem with ReLU — "Dead Neurons"
When a neuron's input is negative, ReLU outputs exactly 0. If a neuron gets stuck outputting 0 for all training examples, its gradient is also 0 — meaning it **never gets updated**. It's permanently dead.

In very deep networks like GPT (12 layers × many neurons), this becomes a real problem. Many neurons can die during training, wasting capacity.

---

## 3. GELU — What GPT Actually Uses

**GELU (Gaussian Error Linear Unit)** is the activation function used inside GPT's feed-forward network.

### The Formula
```
GELU(x) = x × Φ(x)

where Φ(x) = cumulative distribution function of the standard normal distribution
```

### Practical Approximation (what's actually coded)
```
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

Don't memorize this formula — the key intuition is what matters.

### The Shape of GELU

```
         |         /
         |        /
         |       /  ← smooth curve, like ReLU
    ─────|──────/────→  x
         | ↗
         |/ ← small negative values allowed (not exactly 0)
```

Unlike ReLU, GELU:
- **Smoothly approaches 0** for negative values (doesn't hard-cut to 0)
- **Allows small negative outputs** for negative inputs
- Has a **smooth gradient everywhere** — no sharp corner at 0

### Why This Matters
The smooth curve means gradients flow through even negative inputs. No neurons ever "die" completely. Training is more stable and efficient in deep networks.

---

## 4. GELU vs ReLU — Side by Side

| Feature | ReLU | GELU |
|---------|------|------|
| For x > 0 | output = x | output ≈ x (slightly less) |
| For x < 0 | output = 0 (hard cut) | output ≈ small negative value |
| Gradient at x < 0 | Always 0 | Small but non-zero |
| Dead neurons? | ✅ Yes, can happen | ❌ Much less likely |
| Smooth curve? | ❌ Sharp corner at 0 | ✅ Smooth everywhere |
| Used in GPT | ❌ No | ✅ Yes |
| Used in simple nets | ✅ Yes | Less common |

> The smooth gradient of GELU is especially important in very deep networks like GPT where gradients have to flow through 12 Transformer blocks without dying.

---

## 5. GELU in Code

```python
import torch
import torch.nn as nn
import math

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
        ))
```

### Quick test

```python
import matplotlib.pyplot as plt

gelu = GELU()
relu = nn.ReLU()

x = torch.linspace(-4, 4, 100)

gelu_out = gelu(x)
relu_out = relu(x)

# GELU is smooth; ReLU has a sharp bend at 0
# GELU allows small negative outputs; ReLU outputs exactly 0 for all negatives
```

---

## 6. The Feed-Forward Network Inside Each Transformer Block

Every Transformer block contains a small **2-layer neural network** after the multi-head attention. This is called the Feed-Forward Network (FFN).

### Structure

```
Input [tokens × 768]
      ↓
Linear layer: 768 → 3072   (expand by 4×)
      ↓
GELU activation
      ↓
Linear layer: 3072 → 768   (compress back)
      ↓
Output [tokens × 768]
```

**Why expand to 3072 first?**
The intermediate dimension is 4 × embedding_dim. This wider layer gives the network more capacity to learn complex transformations before compressing back. It's a common pattern in Transformers.

### What does the FFN actually do?
The attention mechanism figures out **which tokens to focus on**. The FFN then transforms those representations — learning things like facts, patterns, and relationships that attention alone can't capture. Together they're complementary.

---

## 7. Full FeedForward Class in PyTorch

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 768 → 3072
            GELU(),                                           # non-linearity
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 3072 → 768
        )

    def forward(self, x):
        return self.layers(x)
```

### Testing it

```python
GPT2_CONFIG = {"emb_dim": 768}

ffn = FeedForward(GPT2_CONFIG)

# Simulate input: batch of 2, 4 tokens, 768-dim each
x = torch.rand(2, 4, 768)
out = ffn(x)
print(out.shape)   # torch.Size([2, 4, 768]) — same shape in and out
```

Input and output shapes are identical — the FFN transforms the representations without changing their dimensions.

---

## 8. Where FFN + GELU Fit in the Transformer Block

```
Input Embeddings [batch, tokens, 768]
        ↓
Layer Norm
        ↓
Masked Multi-Head Attention   ← figures out which tokens relate to which
        ↓
Dropout + Shortcut connection
        ↓
Layer Norm
        ↓
Feed-Forward Network          ← transforms each token's representation
  [Linear 768→3072 → GELU → Linear 3072→768]
        ↓
Dropout + Shortcut connection
        ↓
Output [batch, tokens, 768]
```

The attention and FFN work as a team:
- **Attention** = communication between tokens (which tokens matter for which)
- **FFN** = computation per token (what to do with that information)

---

## 🔑 Key Terms

| Term | Meaning |
|------|---------|
| **Activation Function** | Non-linear function applied after each layer — enables complex learning |
| **ReLU** | `max(0, x)` — simple but causes dead neuron problem |
| **Dead Neuron** | A neuron stuck at 0 output with 0 gradient — never learns again |
| **GELU** | Smooth activation used in GPT — no dead neurons, smooth gradients |
| **Feed-Forward Network (FFN)** | 2-layer MLP inside each Transformer block |
| **4× expansion** | FFN intermediate layer is 4× the embedding dim (768→3072 for GPT-2) |
| **Non-linearity** | What activation functions add — enables learning beyond linear patterns |

---

## ⚡ Quick Summary

1. **Activation functions** add non-linearity — without them, deep networks collapse to a single linear operation
2. **ReLU** is simple but causes dead neurons (gradient = 0 for all negative inputs)
3. **GELU** is smooth — allows small negative outputs, gradients never fully die
4. **GPT uses GELU** because dead neurons are especially harmful in 12-layer deep networks
5. **FFN structure:** Linear(768→3072) → GELU → Linear(3072→768)
6. **Why 4× expansion?** Gives the network more capacity for complex transformations
7. **Attention + FFN together:** Attention handles token relationships; FFN handles per-token computation
8. Both input and output of FFN have the same shape `[batch, tokens, 768]`

---

## 📌 What's Coming Next

- **Shortcut (Residual) Connections** — why they're critical for training deep networks
- **Putting the Transformer Block together** — combining LayerNorm + Attention + FFN + Shortcuts
- **Full GPT Model** — assembling all 12 blocks

---

*Notes — Vizuara: "Build a Large Language Model from Scratch"*
