# 📘 Build a Large Language Model from Scratch
## Lecture 16 — Causal Self-Attention Mechanism
> **Video:** https://youtu.be/h94TQOK7NRA

---

## 🗺️ What This Lecture Covers
1. What is Causal Attention and why we need it
2. The Masking Idea — Lower Triangular Matrix
3. Naive Masking Approach (and its problem — Data Leakage)
4. Smart Masking — The -∞ Trick (correct approach)
5. Dropout in Attention
6. Full CausalAttention Class with Batches

---

## 1. What is Causal Attention?

### The Problem with Normal Self-Attention
In normal self-attention, when computing the context vector for "journey", the model looks at **ALL** words — including future words like "with", "one", "step".

But remember: **GPT is trained to predict the NEXT word.** So when predicting word at position 3, it should NOT be allowed to look at words at positions 4, 5, 6. That would be cheating!

### What Causal Attention Does
> **Causal Attention** = Self-attention where each token can only attend to **itself and tokens that came before it**. Future tokens are blocked (masked).

Also called: **Masked Attention**

### Visual Example

```
Sentence: your  journey  starts  with  one  step
Position:   1      2        3       4    5     6

When computing context vector for "journey" (position 2):
  ✅ Can attend to: your (pos 1), journey (pos 2)
  ❌ Must ignore:   starts, with, one, step (pos 3,4,5,6)

When computing context vector for "with" (position 4):
  ✅ Can attend to: your, journey, starts, with (pos 1,2,3,4)
  ❌ Must ignore:   one, step (pos 5,6)

When computing context vector for "step" (position 6):
  ✅ Can attend to: ALL previous words (pos 1,2,3,4,5,6)
  ❌ Nothing to ignore
```

### What the Masked Attention Weight Matrix Looks Like

```
Before masking (normal self-attention):
         your  journey  starts  with  one  step
your   [ 0.21,  0.18,   0.19,  0.12, 0.09, 0.21 ]
journey[ 0.14,  0.24,   0.23,  0.12, 0.11, 0.16 ]
starts [ 0.13,  0.18,   0.25,  0.12, 0.12, 0.20 ]
with   [ 0.15,  0.18,   0.18,  0.21, 0.10, 0.18 ]
one    [ 0.12,  0.16,   0.18,  0.16, 0.22, 0.16 ]
step   [ 0.11,  0.15,   0.19,  0.15, 0.15, 0.25 ]

After causal masking (zeros above diagonal):
         your  journey  starts  with  one  step
your   [ 1.00,  0.00,   0.00,  0.00, 0.00, 0.00 ]
journey[ 0.37,  0.63,   0.00,  0.00, 0.00, 0.00 ]
starts [ 0.25,  0.36,   0.39,  0.00, 0.00, 0.00 ]
with   [ 0.21,  0.26,   0.26,  0.27, 0.00, 0.00 ]
one    [ 0.17,  0.23,   0.26,  0.23, 0.31, 0.00 ]
step   [ 0.11,  0.15,   0.19,  0.15, 0.15, 0.25 ]
```

**Key pattern:** Everything above the diagonal = 0. This is a **lower triangular matrix**.

---

## 2. Masking Using the Lower Triangular Matrix

### Two Types of Triangular Matrices

```python
import torch

# Lower Triangular (torch.tril) — zeros ABOVE diagonal
torch.tril(torch.ones(4, 4))
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])

# Upper Triangular (torch.triu) — zeros BELOW diagonal
torch.triu(torch.ones(4, 4))
# tensor([[1., 1., 1., 1.],
#         [0., 1., 1., 1.],
#         [0., 0., 1., 1.],
#         [0., 0., 0., 1.]])
```

To zero out future tokens → we need **zeros above the diagonal** → use `torch.tril`.

---

## 3. Naive Masking Approach (has a problem!)

### Step-by-Step

```python
context_length = 6

# Step 1: Build a lower triangular mask (1s on and below diagonal, 0s above)
mask_simple = torch.tril(torch.ones(context_length, context_length))
# [[1,0,0,0,0,0],
#  [1,1,0,0,0,0],
#  ...
#  [1,1,1,1,1,1]]

# Step 2: Multiply attention weights by the mask (zeros out future tokens)
masked = attn_weights * mask_simple
# All elements above diagonal are now 0

# Step 3: Re-normalize each row to sum to 1
row_sums = masked.sum(dim=-1, keepdim=True)
masked_normalized = masked / row_sums
# Now each row sums to 1 again ✅
```

### The Problem — Data Leakage ⚠️

This approach has a subtle but serious flaw:

- We computed attention weights using **softmax first** (which included future tokens in its denominator)
- Then we zeroed out the future tokens
- But the damage is already done — the softmax denominator already "saw" the future tokens
- So the remaining values are **slightly influenced by future tokens** they shouldn't have seen

> This is called **data leakage** — information from the future has leaked into the past.

---

## 4. Smart Masking — The -∞ Trick ✅ (Correct Approach)

### The Key Insight

Instead of: `softmax first → mask → re-normalize`

Do this: `mask with -∞ first → softmax once`

**Why it works:**
```
e^(-∞) = 0

So when softmax sees -∞, it outputs exactly 0 for those positions.
And softmax already normalizes everything, so rows automatically sum to 1.
One step, no leakage!
```

### Implementation

```python
# Step 1: Get attention scores (before softmax)
attn_scores = queries @ keys.transpose(-2, -1)

# Step 2: Create upper triangular mask (1s ABOVE diagonal)
#         We want to REPLACE those positions with -inf
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
# [[False, True,  True,  True,  True,  True ],
#  [False, False, True,  True,  True,  True ],
#  ...
#  [False, False, False, False, False, False]]

# Step 3: Replace positions above diagonal with -infinity
attn_scores_masked = attn_scores.masked_fill(mask, float('-inf'))
# [[s11,  -inf, -inf, -inf, -inf, -inf],
#  [s21,   s22, -inf, -inf, -inf, -inf],
#  ...
#  [s61,   s62,  s63,  s64,  s65,  s66]]

# Step 4: Scale and apply softmax ONCE
d_k = keys.shape[-1]
attn_weights = torch.softmax(attn_scores_masked / d_k**0.5, dim=-1)
# e^(-inf) = 0, so all masked positions become 0
# Each row automatically sums to 1 ✅ No leakage ✅
```

### Why Upper Triangular for -∞ but Lower Triangular for zeroing?

| Goal | Which mask | Why |
|------|-----------|-----|
| Zero out above diagonal | `torch.tril` (lower triangular) | Multiplying by 0 kills those values |
| Replace above diagonal with -∞ | `torch.triu` (upper triangular) | We need to find the positions above diagonal and fill them |

Both achieve the same result — all future positions become 0 in the final attention weights.

---

## 5. Dropout in Attention

### What is Dropout?
A deep learning technique that **randomly switches off some neurons** during training.

**Why?** Some neurons become "lazy" — they let other neurons do all the work. Dropout forces all neurons to contribute by randomly disabling some at each training step.

### Two Benefits
1. **Prevents overfitting** — model can't rely too heavily on specific attention patterns
2. **Better generalization** — performs better on unseen data

### How Dropout Works in Attention

```python
dropout = torch.nn.Dropout(p=0.5)  # 50% dropout rate

# Before dropout:
# [[0.37, 0.63, 0.00, 0.00, 0.00, 0.00],
#  [0.25, 0.36, 0.39, 0.00, 0.00, 0.00], ...]

attn_weights_dropped = dropout(attn_weights)

# After dropout (50% randomly zeroed, survivors scaled by 2):
# [[0.74, 0.00, 0.00, 0.00, 0.00, 0.00],   ← 0.63 dropped, 0.37 × 2 = 0.74
#  [0.50, 0.00, 0.39, 0.00, 0.00, 0.00], ...]   ← 0.36 dropped, 0.25 × 2 = 0.50
```

### Important Notes
- Dropout is **probabilistic** — not exactly 50% every row, but 50% on average
- Survivors are **scaled up** by `1/(1-p)` to keep the total "energy" the same
- In production GPT: dropout rate is small like `0.1` or `0.2`, not 0.5
- Applied **after computing attention weights**, before multiplying with values

---

## 6. Full CausalAttention Class with Batches

### Handling Batches
Real LLM training processes **multiple sentences at once** (a batch). So our class must handle a 3D input tensor:

```
Input shape: [batch_size, num_tokens, embedding_dim]
             [    2,          6,           3        ]
```

### Complete CausalAttention Class

```python
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_rate):
        super().__init__()
        self.d_out = d_out

        # Trainable weight matrices (no bias needed)
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key   = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

        # Register the causal mask as a buffer (not a trainable parameter)
        # Buffers automatically move to GPU/CPU with the model
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # x shape: [batch_size, num_tokens, d_in]
        b, num_tokens, d_in = x.shape

        # Step 1: Compute Q, K, V
        queries = self.W_query(x)   # [b, num_tokens, d_out]
        keys    = self.W_key(x)     # [b, num_tokens, d_out]
        values  = self.W_value(x)   # [b, num_tokens, d_out]

        # Step 2: Attention Scores
        # transpose(1, 2) swaps the num_tokens and d_out dims for matrix multiply
        attn_scores = queries @ keys.transpose(1, 2)   # [b, num_tokens, num_tokens]

        # Step 3: Apply causal mask (replace future positions with -inf)
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens].bool(), float('-inf')
        )

        # Step 4: Scale + Softmax → Attention Weights
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)

        # Step 5: Apply Dropout
        attn_weights = self.dropout(attn_weights)

        # Step 6: Compute Context Vectors
        context_vecs = attn_weights @ values   # [b, num_tokens, d_out]
        return context_vecs
```

### Testing with a Batch of 2 Sentences

```python
# Two sentences, each with 6 tokens, each token = 3D vector
batch = torch.stack([inputs, inputs])   # shape: [2, 6, 3]

# Create model
ca = CausalAttention(d_in=3, d_out=2, context_length=6, dropout_rate=0.0)

# Forward pass
context_vecs = ca(batch)
print(context_vecs.shape)   # torch.Size([2, 6, 2])
# 2 batches × 6 context vectors × 2D each
```

---

## 7. Why `register_buffer`?

The causal mask is a **fixed matrix** — it never changes during training. We never backpropagate through it.

```python
# WRONG: Using a regular tensor
self.mask = torch.triu(...)   # This stays on CPU even if model moves to GPU!

# CORRECT: Using register_buffer
self.register_buffer('mask', torch.triu(...))
# This automatically moves with the model to any device (CPU/GPU)
```

> Rule: Any fixed matrix that's NOT trained → use `register_buffer`.

---

## 8. Complete Flow — Causal Attention Summary

```
Input [batch, tokens, d_in]
        ↓ × WQ, WK, WV
Q, K, V [batch, tokens, d_out]
        ↓ Q @ Kᵀ
Attention Scores [batch, tokens, tokens]
        ↓ masked_fill(upper_tri_mask, -inf)
Masked Scores [batch, tokens, tokens]
        ↓ softmax( / √dk )
Attention Weights [batch, tokens, tokens]  ← rows sum to 1, future = 0
        ↓ dropout
Dropped Weights [batch, tokens, tokens]
        ↓ × V
Context Vectors [batch, tokens, d_out]  ✅
```

---

## 🔑 Key Terms

| Term | Meaning |
|------|---------|
| **Causal Attention** | Self-attention where each token only sees itself and past tokens |
| **Masked Attention** | Another name for causal attention |
| **Causal Mask** | The upper triangular pattern of -∞ values blocking future tokens |
| **Data Leakage** | When future token info sneaks into past token calculations |
| **-∞ Trick** | Replace future positions with -inf before softmax → clean masking |
| **torch.tril** | Lower triangular matrix (zeros above diagonal) |
| **torch.triu** | Upper triangular matrix (zeros below diagonal) |
| **Dropout** | Randomly zeros out neurons during training to prevent overfitting |
| **register_buffer** | PyTorch way to store a fixed (non-trainable) tensor that moves with the model |
| **Batch** | Multiple input sequences processed together |

---

## ⚡ Quick Summary

1. **Why causal attention?** GPT predicts the next word — it must NOT see future words
2. **How?** Mask (zero out) all attention weights above the diagonal
3. **Naive approach problem:** Applying softmax first causes data leakage
4. **Smart approach:** Replace future positions with **-∞ BEFORE softmax** — clean, no leakage
5. **e^(-∞) = 0** — so those positions become exactly 0 after softmax
6. **Dropout** randomly zeros attention weights during training to prevent overfitting
7. **Batches** add a dimension: input shape becomes `[batch, tokens, dim]`
8. **register_buffer** = store the fixed mask so it moves with the model to GPU/CPU

---

## 📌 What's Coming Next

**Multi-Head Attention** — the actual mechanism used in GPT:
- Run multiple causal attention heads **in parallel**
- Each head focuses on different aspects of the input
- Concatenate all head outputs → final context vectors

---

*Notes — Vizuara: "Build a Large Language Model from Scratch"*
