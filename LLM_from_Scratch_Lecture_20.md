# 📘 Build a Large Language Model from Scratch
## Lecture 20 — Layer Normalization in the LLM Architecture
> **Video:** https://youtu.be/G3W-LT79LSI

---

## 🗺️ What This Lecture Covers
1. Why Layer Normalization is needed (the gradient problem)
2. What Layer Normalization does — the math
3. Where it appears in the GPT architecture
4. Implementation from scratch
5. The LayerNorm class with scale & shift
6. Layer Norm vs Batch Norm — key difference

---

## 1. Why Layer Normalization is Needed

When you stack many layers in a deep neural network (like GPT's 12 Transformer blocks), training becomes unstable for two reasons:

### Problem 1 — Vanishing / Exploding Gradients
- During backpropagation, gradients are multiplied across layers
- If any layer's output is **too large** → gradient explodes → unstable training
- If any layer's output is **too small** → gradient vanishes → learning stops
- **The cause:** layer outputs with uncontrolled magnitude

### Problem 2 — Internal Covariate Shift
- As training progresses, the input distribution to each layer keeps changing
- This makes it hard for each layer to learn — it keeps adapting to a moving target
- **Result:** slow convergence, poor performance

### How Layer Normalization Solves Both
> Layer Normalization forces the output of every layer to have **mean = 0** and **variance = 1**.

This keeps gradients in a stable range AND makes input distributions consistent across training.

---

## 2. What Layer Normalization Does — The Math

### Simple Example
Say a layer produces 4 outputs: `[1.1, -0.8, 2.3, 4.4]`

**Step 1: Compute mean**
```
μ = (1.1 + (-0.8) + 2.3 + 4.4) / 4 = 2.15 / 4 = 1.75 (approx)
```

**Step 2: Compute variance**
```
σ² = mean of [(x - μ)²] for each x
```

**Step 3: Normalize**
```
x_normalized = (x - μ) / √σ²
```

After this, the 4 values will have **mean ≈ 0** and **variance ≈ 1**.

> Think of it like "resetting the scale" — no matter how large or small the outputs were, they always come out in the same stable range.

---

## 3. Where Layer Normalization Appears in GPT

Layer Norm appears **3 times** in the GPT architecture:

```
Input Embeddings
       ↓
┌──────────────────────────────┐
│      TRANSFORMER BLOCK       │
│                              │
│  → Layer Norm (1st)         │  ← BEFORE multi-head attention
│  → Masked Multi-Head Attn   │
│  → Dropout + Shortcut       │
│  → Layer Norm (2nd)         │  ← BEFORE feed-forward network
│  → Feed Forward Network     │
│  → Dropout + Shortcut       │
└──────────────────────────────┘
       ↓
→ Layer Norm (3rd)              ← AFTER all Transformer blocks, before output
       ↓
Output Head → Logits
```

Since LayerNorm is used in multiple places, it makes sense to **define it as a reusable class**.

---

## 4. Step-by-Step Implementation

### First, let's understand it with a concrete example

```python
import torch
import torch.nn as nn

# Two batches, each with 5 inputs going through a layer with 6 neurons
batch_example = torch.randn(2, 5)   # shape: [2, 5]

layer = nn.Sequential(
    nn.Linear(5, 6),
    nn.ReLU()
)

output = layer(batch_example)
print(output.shape)   # torch.Size([2, 6])
# Row 0 = 6 outputs for batch 1
# Row 1 = 6 outputs for batch 2
```

### Apply normalization manually

```python
# dim=-1 means operate along the LAST dimension (columns)
# keepdim=True preserves the shape as [2, 1] instead of collapsing to [2]

mean = output.mean(dim=-1, keepdim=True)
# shape: [2, 1] — one mean per batch

var = output.var(dim=-1, keepdim=True, unbiased=False)
# shape: [2, 1] — one variance per batch

# Normalize: subtract mean, divide by std
output_norm = (output - mean) / torch.sqrt(var)

# Verify
print(output_norm.mean(dim=-1))   # ≈ 0.0 for each batch ✅
print(output_norm.var(dim=-1))    # ≈ 1.0 for each batch ✅
```

**Why `dim=-1`?** We normalize across features (columns) within each sample (row). Each row represents one batch's outputs.

**Why `keepdim=True`?** Without it, the mean tensor shape collapses from `[2, 1]` to `[2]`, and the subtraction `output - mean` won't broadcast correctly.

**Why `unbiased=False`?** This uses `n` in the denominator instead of `n-1`. It matches GPT-2's original implementation and TensorFlow's default. (The difference is negligible at GPT's embedding sizes of 768+.)

---

## 5. The LayerNorm Class (with scale & shift)

In practice, layer normalization adds two **trainable parameters**:
- **Scale (γ)** — multiplied element-wise with the normalized output
- **Shift (β)** — added element-wise to the result

These let the model learn to adjust the normalized values if needed. They start at 1 and 0 respectively (no change), and get optimized during training.

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5   # small value to prevent division by zero

        # Trainable parameters, same size as the embedding dimension
        self.scale = nn.Parameter(torch.ones(emb_dim))   # starts at 1 (no scaling)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # starts at 0 (no shift)

    def forward(self, x):
        # x shape: [batch, tokens, emb_dim]

        # Step 1: Compute mean across the embedding dimension
        mean = x.mean(dim=-1, keepdim=True)

        # Step 2: Compute variance across the embedding dimension
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Step 3: Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Step 4: Apply trainable scale and shift
        return self.scale * x_norm + self.shift
```

### Testing it

```python
batch_example = torch.randn(2, 5)

ln = LayerNorm(emb_dim=5)
output_ln = ln(batch_example)

print(output_ln.mean(dim=-1))   # ≈ 0.0 for each row ✅
print(output_ln.var(dim=-1))    # ≈ 1.0 for each row ✅
```

### Input and output shapes don't change

```
Input:   [batch, tokens, 768]
Output:  [batch, tokens, 768]   ← same shape, values normalized
```

LayerNorm only adjusts the **values**, never the **dimensions**.

---

## 6. Layer Norm vs Batch Norm — Key Difference

| | **Layer Normalization** | **Batch Normalization** |
|--|------------------------|------------------------|
| Normalizes across | Feature dimension (columns) per sample | Batch dimension (rows) per feature |
| Depends on batch size? | ❌ No | ✅ Yes |
| Works with small batches? | ✅ Yes | ❌ Unstable |
| Used in GPT/Transformers? | ✅ Yes | ❌ No |
| Flexible for hardware? | ✅ Very flexible | ❌ Limited by hardware |

**Why GPT uses Layer Norm:**
If you're training on limited hardware, you might have to use a small batch size. Batch Norm's quality degrades with small batches. Layer Norm is independent of batch size — it always works the same way.

---

## 🔑 Key Terms

| Term | Meaning |
|------|---------|
| **Layer Normalization** | Normalizing each layer's output to have mean=0 and variance=1 |
| **Vanishing Gradient** | Gradients become too small during backprop → no learning |
| **Exploding Gradient** | Gradients become too large → unstable training |
| **Internal Covariate Shift** | Input distributions to layers keep changing during training |
| **Scale (γ)** | Trainable multiplier applied after normalization |
| **Shift (β)** | Trainable additive offset applied after normalization |
| **`dim=-1`** | Apply operation along the last dimension (columns = features) |
| **`keepdim=True`** | Preserve tensor shape after reduction operation |
| **`unbiased=False`** | Divide by n (not n-1) in variance — matches GPT-2's implementation |
| **eps** | Small constant (1e-5) added to variance to prevent division by zero |

---

## ⚡ Quick Summary

1. **Why LayerNorm?** Deep networks suffer from vanishing/exploding gradients and shifting input distributions
2. **What it does:** Forces each layer's output to have **mean = 0, variance = 1**
3. **How:** Subtract mean, divide by standard deviation — per sample, across features
4. **Where in GPT:** 3 times — before attention, before feed-forward, and after all Transformer blocks
5. **Scale & Shift** are two trainable parameters added on top — let the model fine-tune the normalization
6. **Layer Norm ≠ Batch Norm** — LayerNorm doesn't depend on batch size, making it better for LLMs

---

## 📌 What's Coming Next

- **GELU Activation Function** — the non-linearity used inside GPT's feed-forward network
- **Feed-Forward Network** — the two-layer MLP inside each Transformer block
- **Shortcut (Residual) Connections** — why they're essential for deep networks

---

*Notes — Vizuara: "Build a Large Language Model from Scratch"*
