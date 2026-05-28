# 📘 Build a Large Language Model from Scratch
## Lecture 23 — Coding the Full Transformer Block
> **Video:** Lecture 23 in the series

---

## 🗺️ What This Lecture Covers
1. Where the Transformer Block fits in the full GPT stack
2. The 5 components inside the Transformer Block (recap)
3. Key design principle — dimensionality is preserved
4. Pre-LayerNorm vs Post-LayerNorm
5. Full TransformerBlock class in PyTorch
6. Testing: input shape = output shape

---

## 1. The Transformer Block in the GPT Stack

```
Raw text → Tokenize → Token IDs
        ↓
Token Embeddings + Positional Embeddings
        ↓
Dropout
        ↓
┌──────────────────────────────┐
│      TRANSFORMER BLOCK       │  ← This lecture
│    (repeated 12× in GPT-2)  │
└──────────────────────────────┘
        ↓
Final Layer Normalization
        ↓
Output Head → Logits → Next word
```

GPT-2 Small repeats the Transformer Block **12 times**. Each block takes the same shape as input and produces the same shape as output — which is exactly what allows stacking.

---

## 2. The 5 Components Inside the Transformer Block

All 5 were covered in previous lectures. Here's a quick recap before we combine them:

### Component 1 — Masked Multi-Head Attention
- Takes input embedding vectors → produces **context vectors**
- Context vectors are richer: they encode not just word meaning but how each word relates to all other words
- Uses causal masking so position N can't see positions N+1, N+2...

### Component 2 — Layer Normalization
- Normalizes outputs to mean=0, variance=1
- Prevents vanishing/exploding gradients
- Prevents internal covariate shift
- Applied **twice** inside the block (before attention AND before FFN)

### Component 3 — Dropout
- Randomly zeros out some values during training
- Prevents neurons from becoming lazy / prevents overfitting
- Applied after attention and after FFN

### Component 4 — Feed-Forward Network (with GELU)
- Two linear layers: expand 768→3072, then compress 3072→768
- GELU activation between them (no dead neuron problem)
- Processes each token independently (unlike attention which looks across tokens)

### Component 5 — Shortcut Connections
- Add the input of a sub-block directly to its output
- Ensures gradients never vanish during backprop
- Applied after both the attention sub-block and the FFN sub-block

---

## 3. The Most Important Design Principle: Shape is Preserved

> **The Transformer Block's input and output have the exact same shape.**

```
Input:  [batch=2, tokens=4, emb_dim=768]
Output: [batch=2, tokens=4, emb_dim=768]   ← identical shape
```

Why does this matter?
- You can stack 12 blocks in a row without any shape mismatches
- The output of block N feeds directly into block N+1 without any reshaping
- Each block just **refines** the representations without changing their structure

The content changes dramatically (embedding vectors become increasingly context-aware), but the shape stays constant.

---

## 4. Pre-LayerNorm vs Post-LayerNorm

Original Transformer (2017): applied LayerNorm **after** attention and FFN → called **Post-LayerNorm**

GPT and modern LLMs: apply LayerNorm **before** attention and FFN → called **Pre-LayerNorm**

```
POST-LayerNorm (old):        PRE-LayerNorm (GPT, modern):
x → Attention → LN → +      x → LN → Attention → +
x → FFN → LN → +            x → LN → FFN → +
```

**Why Pre-LayerNorm is better:** Research showed Post-LayerNorm causes unstable training dynamics in deep networks. Pre-LayerNorm leads to smoother gradient flow and faster convergence.

---

## 5. The Full TransformerBlock Class

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Multi-Head Attention (from previous lectures)
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        # Feed-Forward Network with GELU
        self.ff = FeedForward(cfg)

        # Two Layer Normalization layers (Pre-LN)
        self.norm1 = LayerNorm(cfg["emb_dim"])   # before attention
        self.norm2 = LayerNorm(cfg["emb_dim"])   # before FFN

        # Dropout for shortcut paths
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # ── First sub-block: Attention ──────────────────────────
        shortcut = x                        # save input for shortcut

        x = self.norm1(x)                   # Pre-LayerNorm
        x = self.att(x)                     # Multi-Head Attention
        x = self.drop_shortcut(x)           # Dropout
        x = x + shortcut                    # Add shortcut ← "+" in diagram

        # ── Second sub-block: Feed-Forward ──────────────────────
        shortcut = x                        # save input for shortcut

        x = self.norm2(x)                   # Pre-LayerNorm
        x = self.ff(x)                      # Feed-Forward Network
        x = self.drop_shortcut(x)           # Dropout
        x = x + shortcut                    # Add shortcut ← "+" in diagram

        return x   # same shape as input ✅
```

### Reading the forward method against the diagram

```
                 x (saved as shortcut)
                 ↓
              norm1(x)         ← Pre-LayerNorm
                 ↓
              att(x)           ← Masked Multi-Head Attention
                 ↓
         drop_shortcut(x)      ← Dropout
                 ↓
           x + shortcut  ──────────────────← shortcut added here
                 ↓
          (saved as new shortcut)
                 ↓
              norm2(x)         ← Pre-LayerNorm
                 ↓
               ff(x)           ← Feed-Forward Network
                 ↓
         drop_shortcut(x)      ← Dropout
                 ↓
           x + shortcut  ──────────────────← shortcut added here
                 ↓
              return x         ← same shape as input
```

---

## 6. Testing: Input Shape = Output Shape

```python
GPT2_CONFIG = {
    "vocab_size":     50257,
    "context_length": 1024,
    "emb_dim":        768,
    "n_heads":        12,
    "n_layers":       12,
    "drop_rate":      0.1,
    "qkv_bias":       False
}

# Simulate a batch: 2 sentences, 4 tokens each, 768-dim embeddings
x = torch.rand(2, 4, 768)

block = TransformerBlock(GPT2_CONFIG)
output = block(x)

print(f"Input  shape: {x.shape}")       # torch.Size([2, 4, 768])
print(f"Output shape: {output.shape}")  # torch.Size([2, 4, 768])  ✅
```

Shape is preserved exactly. The content of each vector has been enriched — every token now encodes contextual information about all other tokens — but the structure is unchanged.

---

## 7. What Happens to the Data Inside One Block

Let's trace what happens to a single batch `[2, 4, 768]` step by step:

| Step | Operation | Shape | What changed |
|------|-----------|-------|-------------|
| Input | — | [2, 4, 768] | Raw embeddings |
| norm1 | LayerNorm | [2, 4, 768] | Each row: mean=0, var=1 |
| att | Multi-Head Attention | [2, 4, 768] | Embedding → context vectors |
| dropout | Zeros 10% randomly | [2, 4, 768] | Some values zeroed |
| + shortcut | Add original input | [2, 4, 768] | Gradient path preserved |
| norm2 | LayerNorm | [2, 4, 768] | Each row: mean=0, var=1 |
| ff | Feed-Forward | [2, 4, 768] | Richer per-token representation |
| dropout | Zeros 10% randomly | [2, 4, 768] | Some values zeroed |
| + shortcut | Add input | [2, 4, 768] | Gradient path preserved |
| Output | — | [2, 4, 768] | Rich context-aware vectors |

Shape never changes. The representations just get better and better.

---

## 🔑 Key Terms

| Term | Meaning |
|------|---------|
| **Transformer Block** | The repeating core unit of GPT — contains attention + FFN + LayerNorm + dropout + shortcuts |
| **Pre-LayerNorm** | Applying LayerNorm before each sub-block (used in GPT, modern LLMs) |
| **Post-LayerNorm** | Applying LayerNorm after each sub-block (older approach, less stable) |
| **Dimension preservation** | Input and output of the Transformer Block have identical shapes |
| **Sub-block** | Either the attention portion or the FFN portion — each has its own shortcut |
| **Context vector** | Output of the attention mechanism — encodes how each token relates to all others |
| **n_layers = 12** | 12 Transformer Blocks stacked in GPT-2 Small |

---

## ⚡ Quick Summary

1. The Transformer Block combines **5 components**: LayerNorm, Masked Multi-Head Attention, Dropout, Feed-Forward Network (GELU), Shortcut Connections
2. **Order inside the block:** LN → Attention → Dropout → Shortcut, then LN → FFN → Dropout → Shortcut
3. **Pre-LayerNorm** (GPT's approach) = normalize before each sub-block → more stable training than Post-LayerNorm
4. **Shape is always preserved:** `[batch, tokens, 768]` in, `[batch, tokens, 768]` out
5. This shape preservation is what allows **12 blocks to be stacked** easily
6. Shortcut connections appear twice: once around attention, once around FFN
7. The output contains **context vectors** — same shape as input but much richer content

---

## 📌 What's Coming Next

- **Full GPT Model** — stacking 12 Transformer Blocks + adding the output head
- How to convert the Transformer Block's output into **next-word predictions**
- The output head that projects from 768D → 50257D (vocabulary size)

---

*Notes — Vizuara: "Build a Large Language Model from Scratch"*
