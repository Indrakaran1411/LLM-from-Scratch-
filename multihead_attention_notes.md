# Multi-Head Attention — Complete Notes
### From Scratch: Weight Splits, Dimensions & Full Mathematics

> Based on: *Build Large Language Models from Scratch* — Lecture 18 (Part 2)  
> Topic: Efficient Multi-Head Attention with Weight Splits

---

## Table of Contents

1. [Why Weight Splits? The Problem with Naive MHA](#1-why-weight-splits-the-problem-with-naive-mha)
2. [Core Idea: One Multiplication, Then Split](#2-core-idea-one-multiplication-then-split)
3. [Architecture Overview](#3-architecture-overview)
4. [Step-by-Step Walkthrough with a Concrete Example](#4-step-by-step-walkthrough-with-a-concrete-example)
   - [Step 1 — Define the Input](#step-1--define-the-input)
   - [Step 2 — Set Hyperparameters](#step-2--set-hyperparameters)
   - [Step 3 — Initialize Trainable Weight Matrices](#step-3--initialize-trainable-weight-matrices)
   - [Step 4 — Compute Q, K, V (3D tensors)](#step-4--compute-q-k-v-3d-tensors)
   - [Step 5 — Reshape to 4D (Introduce Head Dimension)](#step-5--reshape-to-4d-introduce-head-dimension)
   - [Step 6 — Transpose to Group by Heads](#step-6--transpose-to-group-by-heads)
   - [Step 7 — Compute Attention Scores](#step-7--compute-attention-scores)
   - [Step 8 — Apply Causal Mask + Scale + Softmax → Attention Weights](#step-8--apply-causal-mask--scale--softmax--attention-weights)
   - [Step 9 — Multiply Attention Weights × Values → Context Vectors](#step-9--multiply-attention-weights--values--context-vectors)
   - [Step 10 — Transpose Back](#step-10--transpose-back)
   - [Step 11 — Flatten Heads → Final Context Matrix](#step-11--flatten-heads--final-context-matrix)
5. [Full Dimension Trace Summary](#5-full-dimension-trace-summary)
6. [Complete PyTorch Code with Comments](#6-complete-pytorch-code-with-comments)
7. [Key Formulas Reference](#7-key-formulas-reference)
8. [Intuition Behind Each Design Choice](#8-intuition-behind-each-design-choice)
9. [GPT Model Scale Reference](#9-gpt-model-scale-reference)
10. [Common Mistakes / FAQs](#10-common-mistakes--faqs)

---

## 1. Why Weight Splits? The Problem with Naive MHA

In the **naive** multi-head attention approach (Part 1), for every attention head you create *separate* weight matrices:

```
Head 1:  W_Q1 (D_in × D_out),  W_K1,  W_V1
Head 2:  W_Q2 (D_in × D_out),  W_K2,  W_V2
...
Head h:  W_Qh,                  W_Kh,  W_Vh
```

**Cost:** For `h` heads, you need `h` matrix multiplications just to get Q, `h` for K, and `h` for V.

> GPT-3 uses **96 attention heads**, which means **96 × 3 = 288 separate matrix multiplications** just to get Q, K, V — extremely wasteful.

**Weight Splits Solution:**  
Use **one large** weight matrix for Q, one for K, one for V.  
Multiply once → then split the result into `h` parts.

| Approach | Matrix Multiplications for Q, K, V |
|---|---|
| Naive (separate heads) | `3 × h` multiplications |
| Weight Splits | `3` multiplications (1 each for Q, K, V) |

---

## 2. Core Idea: One Multiplication, Then Split

```
Naive:      X @ W_Q1 → Q1
            X @ W_Q2 → Q2        (h separate multiplications)

Weight Split:   X @ W_Q  → Q     (one big matrix multiplication)
                split Q  → Q1, Q2  (cheap reshape/view operation)
```

The key insight: **D_out = num_heads × head_dim**

So a single large weight matrix `W_Q` of shape `(D_in, D_out)` implicitly encodes all `h` heads. After multiplying, you simply *view* the last dimension as `(num_heads, head_dim)`.

---

## 3. Architecture Overview

```
Input X  ──────────────────────────────────────────────────────
(B, T, D_in)                                                   │
     │                                                         │
     ├── @ W_Q (D_in, D_out) ──► Q (B, T, D_out)              │
     ├── @ W_K (D_in, D_out) ──► K (B, T, D_out)              │
     └── @ W_V (D_in, D_out) ──► V (B, T, D_out)              │
                                                               │
         ↓  reshape (view)                                     │
     Q: (B, T, h, head_dim)                                    │
     K: (B, T, h, head_dim)                                    │
     V: (B, T, h, head_dim)                                    │
                                                               │
         ↓  transpose(1, 2) — group by heads                   │
     Q: (B, h, T, head_dim)                                    │
     K: (B, h, T, head_dim)                                    │
     V: (B, h, T, head_dim)                                    │
                                                               │
         ↓  Attention Score = Q @ K^T / √head_dim              │
     scores: (B, h, T, T)                                      │
                                                               │
         ↓  Causal Mask + Softmax                              │
     weights: (B, h, T, T)                                     │
                                                               │
         ↓  Context = weights @ V                              │
     context: (B, h, T, head_dim)                              │
                                                               │
         ↓  transpose(1, 2)                                    │
     context: (B, T, h, head_dim)                              │
                                                               │
         ↓  reshape → flatten heads                            │
     context: (B, T, D_out)    ◄──────────────────────────────
                                                               │
         ↓  Optional Output Projection W_O                     │
     output: (B, T, D_out)
```

**Notation:**
- `B` = batch size
- `T` = number of tokens (sequence length)
- `D_in` = input embedding dimension
- `D_out` = output embedding dimension
- `h` = number of attention heads
- `head_dim` = `D_out / h`

---

## 4. Step-by-Step Walkthrough with a Concrete Example

**Running Example throughout:** Sentence = `["the", "cat", "sleeps"]`

---

### Step 1 — Define the Input

Each word (token) is converted to a **vector embedding** of dimension `D_in`.

```
Sentence:   ["the",  "cat",  "sleeps"]
Tokens:       t_1     t_2      t_3

Input X shape:  (B=1, T=3, D_in=6)
```

Visual (ignoring batch dimension for clarity):

```
        col→  0     1     2     3     4     5
row↓
t_1 "the"  [ 0.1,  0.5, -0.3,  0.8,  0.2, -0.1 ]
t_2 "cat"  [ 0.4, -0.2,  0.7,  0.3, -0.5,  0.9 ]
t_3 "slp"  [-0.6,  0.8,  0.1, -0.4,  0.7,  0.2 ]
```

Shape: `3 rows × 6 cols` → 3 tokens, each a 6-dimensional vector.

---

### Step 2 — Set Hyperparameters

```python
D_in      = 6    # input embedding dimension
D_out     = 6    # output embedding dimension (same as D_in in GPT-style)
num_heads = 2    # number of attention heads
head_dim  = D_out // num_heads  # = 6 // 2 = 3
```

**Key Relationship:**
$$D_{out} = \text{num\_heads} \times \text{head\_dim}$$
$$6 = 2 \times 3 \checkmark$$

Each head is responsible for a 3-dimensional subspace of the full 6-dimensional output.

---

### Step 3 — Initialize Trainable Weight Matrices

Three matrices are created once and *learned during training* via backpropagation:

$$W_Q, W_K, W_V \in \mathbb{R}^{D_{in} \times D_{out}} = \mathbb{R}^{6 \times 6}$$

```python
# In PyTorch (inside __init__):
self.W_query = nn.Linear(D_in, D_out, bias=False)  # shape: (6, 6)
self.W_key   = nn.Linear(D_in, D_out, bias=False)  # shape: (6, 6)
self.W_value = nn.Linear(D_in, D_out, bias=False)  # shape: (6, 6)
```

Why `nn.Linear`? It uses optimized weight initialization (Kaiming uniform by default), which is better for gradient flow during training than random initialization.

> **Why D_out and not D_out/h?**  
> Because we want the *full* large weight matrix so that a single multiplication gives us the combined output for all heads, which we then split.

---

### Step 4 — Compute Q, K, V (3D tensors)

Multiply the input by each weight matrix:

$$Q = X \cdot W_Q, \quad K = X \cdot W_K, \quad V = X \cdot W_V$$

**Dimension check:**
$$X: (1, 3, 6) \quad \times \quad W_Q: (6, 6) \quad \rightarrow \quad Q: (1, 3, 6)$$

```python
Q = self.W_query(x)   # (B, T, D_out) = (1, 3, 6)
K = self.W_key(x)     # (B, T, D_out) = (1, 3, 6)
V = self.W_value(x)   # (B, T, D_out) = (1, 3, 6)
```

At this point, each row (token) has a 6-dimensional Query, Key, and Value vector. The head dimension is *not yet separated* — it's still hidden inside the 6 columns.

---

### Step 5 — Reshape to 4D (Introduce Head Dimension)

We "unroll" the last dimension `D_out=6` into `(num_heads=2, head_dim=3)`:

$$Q: (1, 3, 6) \quad \xrightarrow{\text{view}} \quad Q: (1, 3, 2, 3)$$

```python
B, T, _ = Q.shape
Q = Q.view(B, T, num_heads, head_dim)  # (1, 3, 2, 3)
K = K.view(B, T, num_heads, head_dim)  # (1, 3, 2, 3)
V = V.view(B, T, num_heads, head_dim)  # (1, 3, 2, 3)
```

**How to read a 4D tensor `(1, 3, 2, 3)`:**

```
Dimension 0: Batch       → 1 sample
Dimension 1: Token       → 3 tokens (the, cat, sleeps)
Dimension 2: Head        → 2 attention heads
Dimension 3: Head dim    → 3 values per head
```

Visual for Q after reshape:

```
Token "the" (row 0):
    Head 1:  [q1_1, q1_2, q1_3]   ← 3-dim query for head 1
    Head 2:  [q2_1, q2_2, q2_3]   ← 3-dim query for head 2

Token "cat" (row 1):
    Head 1:  [q1_1, q1_2, q1_3]
    Head 2:  [q2_1, q2_2, q2_3]

Token "sleeps" (row 2):
    Head 1:  [q1_1, q1_2, q1_3]
    Head 2:  [q2_1, q2_2, q2_3]
```

Current grouping: **by token** (outer) → heads (inner).

---

### Step 6 — Transpose to Group by Heads

Currently grouped as: `(B, Token, Head, HeadDim)`  
We need: `(B, Head, Token, HeadDim)`

Why? Because attention scores are computed **per head**. We want to process Head 1's Q with Head 1's K, and Head 2's Q with Head 2's K — independently.

$$Q: (1, 3, 2, 3) \quad \xrightarrow{\text{transpose(1,2)}} \quad Q: (1, 2, 3, 3)$$

```python
Q = Q.transpose(1, 2)   # (B, num_heads, T, head_dim) = (1, 2, 3, 3)
K = K.transpose(1, 2)   # (1, 2, 3, 3)
V = V.transpose(1, 2)   # (1, 2, 3, 3)
```

> `transpose(1, 2)` swaps index 1 (tokens) and index 2 (heads).  
> Python uses 0-based indexing: index 0 = batch, index 1 = tokens, index 2 = heads.

Visual after transpose:

```
Head 1 block (index 0):
    "the"    [q1_1, q1_2, q1_3]   ← row 0
    "cat"    [q1_1, q1_2, q1_3]   ← row 1
    "sleeps" [q1_1, q1_2, q1_3]   ← row 2

Head 2 block (index 1):
    "the"    [q2_1, q2_2, q2_3]
    "cat"    [q2_1, q2_2, q2_3]
    "sleeps" [q2_1, q2_2, q2_3]
```

Now each head's Q, K, V matrices are `(T × head_dim) = (3 × 3)` — ready for independent attention computation.

---

### Step 7 — Compute Attention Scores

The attention score between query token `i` and key token `j` is their dot product.

$$\text{scores} = Q \cdot K^T \quad \text{(transpose last two dims)}$$

```python
# K.transpose(-2, -1) transposes the last two dims: (B, h, head_dim, T)
scores = Q @ K.transpose(-2, -1)   # (B, h, T, T) = (1, 2, 3, 3)
```

**Dimension check:**
$$Q: (1, 2, 3, 3) \quad \times \quad K^T: (1, 2, 3, 3) \quad \rightarrow \quad \text{scores}: (1, 2, 3, 3)$$

The last two dims: `(T, head\_dim) × (head\_dim, T) → (T, T)` ✓

**Interpreting the (T × T) score matrix (for one head):**

```
              "the"   "cat"  "sleeps"   ← KEY tokens (columns)
              ──────────────────────
"the"    →  [ s11,    s12,    s13  ]   ← How much "the" attends to each
"cat"    →  [ s21,    s22,    s23  ]   ← How much "cat" attends to each
"sleeps" →  [ s31,    s32,    s33  ]   ← How much "sleeps" attends to each
```

`s_ij` = raw attention score between query token `i` and key token `j`.

---

### Step 8 — Apply Causal Mask + Scale + Softmax → Attention Weights

This step has three sub-steps:

#### 8a. Causal (Auto-Regressive) Masking

Language models are **auto-regressive**: a token can only attend to *itself and previous tokens*, not future ones. This is enforced by masking future positions with `−∞`.

```
Before mask:                After mask:
[ s11,  s12,  s13 ]        [ s11,  -∞,   -∞  ]
[ s21,  s22,  s23 ]   →    [ s21,  s22,  -∞  ]
[ s31,  s32,  s33 ]        [ s31,  s32,  s33  ]
```

Upper triangle (above diagonal) = future tokens → replaced by `−∞`.

```python
# Create upper-triangular mask
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

Why `−∞`? Because `softmax(−∞) = 0`, which automatically zeros out those positions.

#### 8b. Scaling by √head_dim

$$\text{scores\_scaled} = \frac{\text{scores}}{\sqrt{\text{head\_dim}}}$$

```python
scores = scores / (head_dim ** 0.5)   # divide by √3 ≈ 1.732
```

**Why scale?**  
When `head_dim` is large, the dot products `Q·K^T` grow in magnitude (variance scales with dimension). Large values cause softmax to become very "peaked" (close to one-hot), which leads to vanishing gradients during backpropagation.

Dividing by `√head_dim` keeps the variance of the dot products approximately 1, stabilizing training.

> **Formal reasoning:** If Q and K entries have variance 1, then `Q·K^T` has variance `head_dim`. Dividing by `√head_dim` brings variance back to 1.

#### 8c. Softmax → Attention Weights

$$\text{weights}_{ij} = \frac{e^{\text{score}_{ij}}}{\sum_k e^{\text{score}_{ik}}}$$

```python
weights = torch.softmax(scores, dim=-1)   # (B, h, T, T)
```

`dim=-1` means softmax is applied along the last dimension (key dimension), so **each row sums to 1**.

Result (example values, head 1):

```
              "the"  "cat"  "sleeps"
"the"    →  [ 1.00,  0.00,   0.00 ]   ← can only see itself
"cat"    →  [ 0.96,  0.04,   0.00 ]   ← 96% on "the", 4% on "cat"
"sleeps" →  [ 0.04,  0.26,   0.69 ]   ← distributes across all 3
```

These are now **interpretable probabilities**: "sleeps pays 69% attention to itself, 26% to 'cat', and 4% to 'the'."

#### Optional: Dropout

```python
weights = dropout(weights)   # randomly zeros some weights during training
```

---

### Step 9 — Multiply Attention Weights × Values → Context Vectors

$$\text{context} = \text{weights} \cdot V$$

```python
context = weights @ V   # (B, h, T, T) @ (B, h, T, head_dim) = (B, h, T, head_dim)
```

**Dimension check:**
$$\text{weights}: (1, 2, 3, 3) \quad \times \quad V: (1, 2, 3, 3) \quad \rightarrow \quad \text{context}: (1, 2, 3, 3)$$

Last two dims: `(T, T) × (T, head_dim) → (T, head_dim)` ✓

**Intuition:** The context vector for token `i` is a **weighted sum of Value vectors**:
$$\text{context}_i = \sum_j \text{weight}_{ij} \cdot V_j$$

The model "mixes" information from all relevant tokens proportional to their attention weights.

Shape after this step: `(B=1, h=2, T=3, head_dim=3)`

```
Head 1:
    context for "the"    → [c1, c2, c3]   (3-dim)
    context for "cat"    → [c1, c2, c3]
    context for "sleeps" → [c1, c2, c3]

Head 2:
    context for "the"    → [c1, c2, c3]
    context for "cat"    → [c1, c2, c3]
    context for "sleeps" → [c1, c2, c3]
```

---

### Step 10 — Transpose Back

Undo the transpose from Step 6 to bring tokens back to dimension 1:

$$\text{context}: (1, 2, 3, 3) \quad \xrightarrow{\text{transpose(1,2)}} \quad \text{context}: (1, 3, 2, 3)$$

```python
context = context.transpose(1, 2)   # (B, T, num_heads, head_dim) = (1, 3, 2, 3)
```

Now grouped **by token** again (outer) with heads inside — ready to merge.

```
Token "the":
    Head 1 output: [c1, c2, c3]
    Head 2 output: [c1, c2, c3]

Token "cat":
    Head 1 output: [c1, c2, c3]
    Head 2 output: [c1, c2, c3]

Token "sleeps":
    Head 1 output: [c1, c2, c3]
    Head 2 output: [c1, c2, c3]
```

---

### Step 11 — Flatten Heads → Final Context Matrix

Concatenate the outputs of all heads for each token by reshaping:

$$\text{context}: (1, 3, 2, 3) \quad \xrightarrow{\text{reshape}} \quad \text{context}: (1, 3, 6)$$

```python
context = context.contiguous().view(B, T, D_out)   # (1, 3, 6)
```

- `.contiguous()` ensures the tensor data is laid out in contiguous memory before reshape (required after transpose).
- `.view(B, T, D_out)` merges `(num_heads, head_dim)` → `D_out`.

**Final output interpretation:**

```
Row 0 → context vector for "the"    → 6-dim vector
Row 1 → context vector for "cat"    → 6-dim vector
Row 2 → context vector for "sleeps" → 6-dim vector

Shape: (1, 3, 6) = (B, T, D_out) ✓
```

Each token now has a **rich contextual representation** that encodes information from all relevant tokens (weighted by attention), across all heads.

#### Optional: Output Projection

```python
output = self.out_proj(context)   # Linear(D_out, D_out) → same shape
```

A learned linear transformation applied after combining heads — allows the model to mix head outputs in a learnable way.

---

## 5. Full Dimension Trace Summary

| Step | Operation | Shape |
|------|-----------|-------|
| Input | — | `(B, T, D_in)` = `(1, 3, 6)` |
| After W_Q / W_K / W_V | Linear projection | `(B, T, D_out)` = `(1, 3, 6)` |
| After `.view()` | Split heads | `(B, T, h, head_dim)` = `(1, 3, 2, 3)` |
| After `.transpose(1,2)` | Group by heads | `(B, h, T, head_dim)` = `(1, 2, 3, 3)` |
| Attention scores | `Q @ K^T` | `(B, h, T, T)` = `(1, 2, 3, 3)` |
| After mask + scale + softmax | Attention weights | `(B, h, T, T)` = `(1, 2, 3, 3)` |
| After `weights @ V` | Context per head | `(B, h, T, head_dim)` = `(1, 2, 3, 3)` |
| After `.transpose(1,2)` | Group by tokens | `(B, T, h, head_dim)` = `(1, 3, 2, 3)` |
| After `.view()` | Merge heads | `(B, T, D_out)` = `(1, 3, 6)` |

---

## 6. Complete PyTorch Code with Comments

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, D_in, D_out, num_heads, context_length, dropout=0.0):
        super().__init__()
        assert D_out % num_heads == 0, "D_out must be divisible by num_heads"

        self.D_out = D_out
        self.num_heads = num_heads
        self.head_dim = D_out // num_heads  # dimension per head

        # --- Trainable weight matrices ---
        # Single large matrix for all heads (weight split approach)
        self.W_query = nn.Linear(D_in, D_out, bias=False)
        self.W_key   = nn.Linear(D_in, D_out, bias=False)
        self.W_value = nn.Linear(D_in, D_out, bias=False)

        # Optional output projection after combining heads
        self.out_proj = nn.Linear(D_out, D_out, bias=False)

        self.dropout = nn.Dropout(dropout)

        # --- Causal mask (upper triangular) ---
        # Registered as buffer so it moves with model.to(device)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        B, T, D_in = x.shape                 # (batch, tokens, input_dim)

        # Step 4: Compute Q, K, V — single matrix multiplication each
        Q = self.W_query(x)                  # (B, T, D_out)
        K = self.W_key(x)                    # (B, T, D_out)
        V = self.W_value(x)                  # (B, T, D_out)

        # Step 5: Reshape — split D_out into (num_heads, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim)  # (B, T, h, head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim)

        # Step 6: Transpose — group by heads for parallel attention computation
        Q = Q.transpose(1, 2)   # (B, h, T, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Step 7: Attention scores = Q @ K^T
        scores = Q @ K.transpose(-2, -1)     # (B, h, T, T)

        # Step 8a: Apply causal mask (block future tokens)
        scores = scores.masked_fill(
            self.mask[:T, :T].bool(), float('-inf')
        )

        # Step 8b: Scale by sqrt(head_dim) to stabilize gradients
        scores = scores / (self.head_dim ** 0.5)

        # Step 8c: Softmax — each row of (T x T) sums to 1
        weights = torch.softmax(scores, dim=-1)  # (B, h, T, T)
        weights = self.dropout(weights)

        # Step 9: Context = weighted sum of Values
        context = weights @ V                    # (B, h, T, head_dim)

        # Step 10: Transpose back — group by tokens
        context = context.transpose(1, 2)        # (B, T, h, head_dim)

        # Step 11: Flatten heads → merge into D_out
        context = context.contiguous().view(B, T, self.D_out)  # (B, T, D_out)

        # Optional output projection
        output = self.out_proj(context)          # (B, T, D_out)
        return output


# ─── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":
    # Sentence: ["the", "cat", "sleeps"] — batch of 2
    x1 = torch.randn(3, 6)   # 3 tokens, 6-dim embedding
    x2 = torch.randn(3, 6)
    x  = torch.stack([x1, x2])  # batch: (2, 3, 6)

    mha = MultiHeadAttention(
        D_in=6, D_out=6,
        num_heads=2,
        context_length=6,
        dropout=0.0
    )

    out = mha(x)
    print(out.shape)   # Expected: torch.Size([2, 3, 6])
```

---

## 7. Key Formulas Reference

### Scaled Dot-Product Attention (per head)

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where `d_k = head_dim`.

### Head Dimension

$$\text{head\_dim} = \frac{D_{out}}{\text{num\_heads}}$$

### Multi-Head Attention Output (combined)

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W_O$$

$$\text{where} \quad \text{head}_i = \text{Attention}(QW_Q^{(i)}, KW_K^{(i)}, VW_V^{(i)})$$

In the **weight split** implementation, the concatenation + W_O is replaced by:
1. A single projection through large `W_Q, W_K, W_V`
2. A reshape + transpose (split)
3. A final reshape (concatenate) + optional `W_O`

### Variance Justification for Scaling

If `Q_i, K_i ~ N(0, 1)` independently, then:
$$\text{Var}(Q \cdot K) = \sum_{i=1}^{d_k} \text{Var}(Q_i K_i) = d_k$$
$$\text{Std}(Q \cdot K) = \sqrt{d_k}$$

Dividing by `√d_k` normalizes variance to 1 → prevents softmax saturation.

---

## 8. Intuition Behind Each Design Choice

| Design Choice | Why It's Done |
|---|---|
| **Single large W_Q, W_K, W_V** | Reduces from `h` matrix multiplications to 1; much faster |
| **Split via `.view()` not separate matrices** | `view` is a zero-copy operation (no new memory), just reinterprets shape |
| **Transpose before attention** | PyTorch's `@` operator applies matrix multiply to the last two dims; transpose ensures correct head-wise multiplication |
| **Causal mask with −∞** | `softmax(−∞) = 0` kills future positions cleanly; doesn't require extra branching |
| **Scale by √head_dim** | Prevents dot products from exploding in magnitude as head_dim grows; stabilizes softmax and gradients |
| **`.contiguous()` before `.view()`** | After transpose, memory layout may not be contiguous; `.view()` requires contiguous memory |
| **Output projection W_O** | Learnable mixing of multi-head outputs; gives the model an extra degree of freedom |

---

## 9. GPT Model Scale Reference

| Model | Heads (`h`) | D_model | head_dim |
|---|---|---|---|
| **GPT-2 small** | 12 | 768 | 64 |
| **GPT-2 large** | 16 | 1024 | 64 |
| **GPT-2 XL** | 25 | 1600 | 64 |
| **GPT-3 (175B)** | 96 | 12288 | 128 |

Everything we built with `h=2, D_out=6` scales directly — just plug in larger numbers.

With weight splits, GPT-3 only needs **3** large matrix multiplications (Q, K, V), not **288**.

---

## 10. Common Mistakes / FAQs

**Q: Why can't I just skip the transpose and use 3D tensors?**  
A: The PyTorch `@` matmul operator acts on the last two dimensions. Without transpose, head 1's Q would be mixed with head 2's K. The transpose isolates each head's computation.

**Q: Why do we need `.contiguous()` before `.view()`?**  
A: `.transpose()` doesn't move data in memory — it just changes how strides are interpreted. `.view()` requires the tensor to be stored as a contiguous block. `.contiguous()` copies data to a new contiguous block if needed.

**Q: What's the difference between attention *scores* and attention *weights*?**  
A: Scores = raw dot products `Q·K^T` (can be any real number). Weights = after softmax (each row sums to 1, all values in [0, 1]). Only weights are interpretable as probabilities.

**Q: Why is D_in = D_out in GPT models?**  
A: In transformers, the residual connection adds the MHA output back to the input. For this to work, their shapes must match: `x = x + MHA(x)`. This requires `D_in = D_out`.

**Q: Does each head learn something different?**  
A: Yes — that's the whole point of multi-head attention. Different heads learn to attend to different patterns (e.g., one head learns syntactic relations, another learns positional proximity). The model discovers these specializations through training.

**Q: What does the output projection W_O actually do?**  
A: After concatenating all head outputs, `W_O` is a learned linear transformation that allows the model to combine the head outputs in a flexible, task-optimized way. Without it, heads are simply concatenated with no additional mixing.

---

*End of Notes — Multi-Head Attention with Weight Splits*
