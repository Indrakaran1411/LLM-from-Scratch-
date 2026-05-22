# Multi-Head Attention — Part 1 Notes
### Basics, Intuition & Python Implementation (Wrapper Approach)

> Based on: *Build Large Language Models from Scratch* — Lecture 17  
> Prerequisites: Simplified Self-Attention → Self-Attention with Trainable Weights → Causal Attention

---

## Table of Contents

1. [Learning Path So Far](#1-learning-path-so-far)
2. [Quick Recap: Causal Attention](#2-quick-recap-causal-attention)
3. [What is Multi-Head Attention?](#3-what-is-multi-head-attention)
4. [Core Idea: Why Multiple Heads?](#4-core-idea-why-multiple-heads)
5. [Architecture: From Single Head to Multi-Head](#5-architecture-from-single-head-to-multi-head)
6. [Step-by-Step with Concrete Example](#6-step-by-step-with-concrete-example)
7. [Code: Multi-Head Attention Wrapper](#7-code-multi-head-attention-wrapper)
8. [Output Shape Explained](#8-output-shape-explained)
9. [Problem with This Approach](#9-problem-with-this-approach)
10. [Key Formulas & Dimension Table](#10-key-formulas--dimension-table)
11. [Summary Comparison: Single Head vs Multi-Head](#11-summary-comparison-single-head-vs-multi-head)

---

## 1. Learning Path So Far

The lectures build on each other deliberately. You cannot understand MHA without the earlier layers:

```
Simplified Self-Attention
        ↓
Self-Attention with Trainable Weights   (introduces W_Q, W_K, W_V)
        ↓
Causal Attention                        (introduces masking, prevents future leakage)
        ↓
Multi-Head Attention Part 1             (stack multiple causal attention heads) ← YOU ARE HERE
        ↓
Multi-Head Attention Part 2             (weight splits, efficient implementation)
```

If you jumped here directly, go back to causal attention first.

---

## 2. Quick Recap: Causal Attention

This is the building block. Multi-head attention is literally just "run this multiple times and concatenate."

### What it does

Takes input token embeddings → produces enriched **context vectors**.

| Vector | What it captures |
|---|---|
| Input embedding | Semantic meaning of the word itself |
| Context vector | Semantic meaning + how this word relates to all other words |

### The full causal attention pipeline

```
Input X  (B, T, D_in)
    │
    ├── × W_Q  →  Q  (B, T, D_out)
    ├── × W_K  →  K  (B, T, D_out)
    └── × W_V  →  V  (B, T, D_out)
         │
         ▼
  Attention Scores = Q @ K^T         (B, T, T)
         │
         ▼
  Apply Causal Mask                  (zero out above-diagonal = future tokens)
         │
         ▼
  Apply Softmax → Attention Weights  (each row sums to 1)
         │
         ▼
  Context = Weights @ V              (B, T, D_out)
```

### The Causal Mask — Why?

Language models are **auto-regressive**: predict the next word using only past words. So when computing the context for token `i`, it must not "see" tokens `i+1, i+2, ...`

```
Sentence: "your journey begins with one step"
           t1     t2      t3     t4    t5   t6

Attention weights matrix (rows = queries, cols = keys):

              t1   t2   t3   t4   t5   t6
t1 "your" → [✓    ✗    ✗    ✗    ✗    ✗ ]   ← can only see itself
t2 "jour" → [✓    ✓    ✗    ✗    ✗    ✗ ]
t3 "begi" → [✓    ✓    ✓    ✗    ✗    ✗ ]
t4 "with" → [✓    ✓    ✓    ✓    ✗    ✗ ]
t5 "one"  → [✓    ✓    ✓    ✓    ✓    ✗ ]
t6 "step" → [✓    ✓    ✓    ✓    ✓    ✓ ]   ← can see everything

✓ = kept,  ✗ = set to 0 (future tokens masked out)
```

### Causal Attention Output Shape

```
Input:   (B=2, T=6, D_in=3)
Output:  (B=2, T=6, D_out=2)

One context vector per token, for each batch.
```

---

## 3. What is Multi-Head Attention?

**One-line definition:**  
Run causal attention multiple times in parallel, each with its own weight matrices, then concatenate all the outputs.

The word **"head"** = one complete run of the attention mechanism (one set of W_Q, W_K, W_V).

```
Single Head:   1 × W_Q,  1 × W_K,  1 × W_V  →  1 context vector matrix
Multi-Head:    h × W_Q,  h × W_K,  h × W_V  →  h context vector matrices → concatenate
```

If `h = 2`:
- Head 1 produces context matrix of shape `(T, D_out)`
- Head 2 produces context matrix of shape `(T, D_out)`
- Concatenated output: `(T, 2 × D_out)`

---

## 4. Core Idea: Why Multiple Heads?

### Human analogy

When you read *"The cat that my neighbor owns sleeps all day"*, your brain simultaneously tracks:

- **Who** is the subject? → "cat"
- **What** is happening? → "sleeps"
- **Whose** cat? → "neighbor's"
- **When?** → "all day"

A single attention head is like using one "lens" to understand the sentence. Multiple heads = multiple lenses, each potentially focusing on different types of relationships.

### What research shows heads learn

After training, different heads in real LLMs specialize in:

| Head type | What it attends to |
|---|---|
| Positional head | The immediately previous token |
| Syntactic head | Subject-verb agreement |
| Coreference head | Pronoun → noun it refers to |
| Rare-word head | Unusual or important words |

> **Key point:** You don't program what each head learns. The model discovers specializations through gradient descent. You just provide the structure (number of heads).

### Why it improves performance

A single head computes one "summary" of the context. Multiple heads compute several different summaries — the concatenated output is richer and allows the model to simultaneously encode multiple types of relationships.

---

## 5. Architecture: From Single Head to Multi-Head

### Single Head (Causal Attention)

```
         W_Q    W_K    W_V
          │      │      │
Input X ──┼──────┼──────┤
          ▼      ▼      ▼
          Q      K      V
          │
          └──→ Q @ K^T → mask → softmax → weights @ V → Context (T × D_out)
```

### Multi-Head Attention (h = 2 heads)

```
         W_Q1   W_K1   W_V1
          │      │      │
Input X ──┼──────┼──────┤  →  Head 1  →  Context_1  (T × D_out)
          │                                   │
         W_Q2   W_K2   W_V2                   │  concatenate
          │      │      │                     │  along columns
Input X ──┼──────┼──────┤  →  Head 2  →  Context_2  (T × D_out)
                                              │
                                              ▼
                                      Final Context  (T × 2·D_out)
```

Each head has its **own independent** W_Q, W_K, W_V — they are not shared.

---

## 6. Step-by-Step with Concrete Example

**Setup:**
```
Sentence: "your journey begins with one step"
Tokens:    t1     t2      t3     t4    t5   t6

T       = 6     (number of tokens)
D_in    = 3     (input embedding dimension)
D_out   = 2     (output dimension per head)
h       = 2     (number of heads)
Batch   = 2     (two sentences processed together)
```

### Step 1: Input

```
Input X shape: (B=2, T=6, D_in=3)

Batch 1, token embeddings:
  "your"    →  [0.43, 0.15, 0.89]
  "journey" →  [0.55, 0.87, 0.66]
  "begins"  →  [0.57, 0.85, 0.64]
  "with"    →  [0.22, 0.58, 0.33]
  "one"     →  [0.77, 0.25, 0.11]
  "step"    →  [0.05, 0.80, 0.23]
```

### Step 2: Head 1 computes its context matrix

Weight matrices for Head 1:
- `W_Q1: (3×2)`, `W_K1: (3×2)`, `W_V1: (3×2)`

```
Q1 = X @ W_Q1  →  shape (6, 2)
K1 = X @ W_K1  →  shape (6, 2)
V1 = X @ W_V1  →  shape (6, 2)

Scores_1  = Q1 @ K1^T      →  shape (6, 6)
Apply causal mask           →  upper triangle = 0
Apply softmax               →  each row sums to 1
Context_1 = weights @ V1   →  shape (6, 2)
```

### Step 3: Head 2 computes its context matrix (independently)

Weight matrices for Head 2:
- `W_Q2: (3×2)`, `W_K2: (3×2)`, `W_V2: (3×2)`

Same procedure → `Context_2` of shape `(6, 2)`

### Step 4: Concatenate along columns

```
Context_1:  (6, 2)
                    } concatenate → Final Context: (6, 4)
Context_2:  (6, 2)
```

Visually:

```
Context_1           Context_2          Final (concatenated)
┌────────────┐    ┌────────────┐    ┌──────────────────────┐
│ c11  c12   │    │ c21  c22   │    │ c11  c12  c21  c22   │  ← "your"
│ c11  c12   │    │ c21  c22   │    │ c11  c12  c21  c22   │  ← "journey"
│ c11  c12   │    │ c21  c22   │    │ c11  c12  c21  c22   │  ← "begins"
│ c11  c12   │ ++ │ c21  c22   │ =  │ c11  c12  c21  c22   │  ← "with"
│ c11  c12   │    │ c21  c22   │    │ c11  c12  c21  c22   │  ← "one"
│ c11  c12   │    │ c21  c22   │    │ c11  c12  c21  c22   │  ← "step"
└────────────┘    └────────────┘    └──────────────────────┘
  6 rows × 2       6 rows × 2          6 rows × 4
```

Each token's final context vector now has dimension `4 = D_out × num_heads`.

---

## 7. Code: Multi-Head Attention Wrapper

### The Causal Attention Class (building block)

```python
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, D_in, D_out, context_length, dropout=0.0, bias=False):
        super().__init__()
        self.D_out = D_out
        self.W_query = nn.Linear(D_in, D_out, bias=bias)
        self.W_key   = nn.Linear(D_in, D_out, bias=bias)
        self.W_value = nn.Linear(D_in, D_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        B, T, D_in = x.shape
        Q = self.W_query(x)                                      # (B, T, D_out)
        K = self.W_key(x)
        V = self.W_value(x)

        scores  = Q @ K.transpose(-2, -1)                        # (B, T, T)
        scores  = scores.masked_fill(self.mask[:T,:T].bool(), float('-inf'))
        scores  = scores / (self.D_out ** 0.5)                   # scale
        weights = torch.softmax(scores, dim=-1)                  # (B, T, T)
        weights = self.dropout(weights)

        context = weights @ V                                    # (B, T, D_out)
        return context
```

### The Multi-Head Attention Wrapper

```python
class MultiHeadAttentionWrapper(nn.Module):
    """
    Naive multi-head attention:
    Creates h independent CausalAttention instances,
    runs them sequentially, concatenates outputs.
    """
    def __init__(self, D_in, D_out, context_length, dropout, num_heads, bias=False):
        super().__init__()

        # nn.ModuleList properly registers all heads as submodules
        # so their parameters are included in model.parameters()
        self.heads = nn.ModuleList([
            CausalAttention(D_in, D_out, context_length, dropout, bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # Run each head independently, concatenate along last dim (columns)
        # Each head(x) → (B, T, D_out)
        # After cat on dim=-1 → (B, T, D_out × num_heads)
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

### Running it

```python
# Input: "your journey begins with one step"
inputs = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.11],
    [0.05, 0.80, 0.23],
])

batch = torch.stack([inputs, inputs])   # (2, 6, 3) — batch of 2

mha = MultiHeadAttentionWrapper(
    D_in=3,
    D_out=2,
    context_length=6,
    dropout=0.0,
    num_heads=2
)

context_vecs = mha(batch)
print(context_vecs.shape)   # torch.Size([2, 6, 4])
```

---

## 8. Output Shape Explained

```
Output shape: (B, T, D_out × num_heads)
              (2, 6, 4)

Dimension 0 → B = 2           : two batches
Dimension 1 → T = 6           : six tokens
Dimension 2 → D_out × h = 4   : 2 per head × 2 heads
```

**General formula:**

$$\text{Output embedding dim} = D_{out} \times \text{num\_heads}$$

| D_out | num_heads | Final dim | Model |
|-------|-----------|-----------|-------|
| 2 | 2 | 4 | (toy example) |
| 64 | 12 | 768 | GPT-2 small |
| 64 | 16 | 1024 | GPT-2 large |
| 64 | 25 | 1600 | GPT-2 XL |
| 128 | 96 | 12288 | GPT-3 (175B) |

---

## 9. Problem with This Approach

This wrapper approach works correctly but is **sequentially inefficient**.

### The bottleneck

```python
# Runs heads ONE BY ONE in a Python for-loop
return torch.cat([head(x) for head in self.heads], dim=-1)
```

With 96 heads (GPT-3): run head 1, wait, run head 2, wait, ..., run head 96.

Modern GPUs can process thousands of operations simultaneously — this serial approach wastes almost all of that capacity.

### The fix: Part 2 (Weight Splits)

Instead of separate weight matrices per head, use one large matrix and split the result:

```
Wrapper (Part 1):
  h × (X @ W_Qi) → h separate matrix mults → concatenate

Weight Split (Part 2):
  1 × (X @ W_Q_large) → reshape → process all heads simultaneously
```

The weight split approach reduces h matrix multiplications down to just 1, enabling full GPU parallelism. That's what Lecture 18 covers.

---

## 10. Key Formulas & Dimension Table

### Attention Score (for token i attending to token j)

$$\text{score}_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{D_{out}}}$$

### Context Vector (causal — only past tokens)

$$\text{context}_i = \sum_{j \leq i} \text{softmax}\!\left(\frac{Q K^T}{\sqrt{D_{out}}}\right)_{ij} \cdot \mathbf{v}_j$$

### Final Output Dimension

$$\text{Final context dim} = D_{out} \times \text{num\_heads}$$

### Shape Trace (full forward pass)

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input X | `(B, T, D_in)` | Raw token embeddings |
| W_Q, W_K, W_V (per head) | `(D_in, D_out)` | Trainable |
| Q, K, V | `(B, T, D_out)` | After projection |
| Attention scores | `(B, T, T)` | `Q @ K^T` |
| After mask | `(B, T, T)` | Upper triangle = −∞ |
| Attention weights | `(B, T, T)` | After softmax, rows sum to 1 |
| Context (1 head) | `(B, T, D_out)` | `weights @ V` |
| Context (h heads, final) | `(B, T, D_out × h)` | After concatenation |

---

## 11. Summary Comparison: Single Head vs Multi-Head

| Property | Causal Attention | MHA Wrapper (Part 1) | MHA Weight Split (Part 2) |
|---|---|---|---|
| Weight matrices | 1 × each | h × each | 1 large × each |
| Context output dim | `D_out` | `D_out × h` | `D_out` (D_out already = h × head_dim) |
| Relationships captured | One type | Multiple types | Multiple types |
| Computation | 1 pass | h sequential passes | 1 pass + reshape |
| GPU efficiency | Good | Poor (serial loop) | Excellent (parallel) |
| Used in GPT? | No | Conceptually yes | Yes (actual implementation) |

---

## Appendix: Parameter Count

For one `MultiHeadAttentionWrapper`:

$$\text{Parameters} = \text{num\_heads} \times 3 \times D_{in} \times D_{out}$$

For our toy example (`h=2, D_in=3, D_out=2`):
$$= 2 \times 3 \times 3 \times 2 = 36 \text{ parameters}$$

For GPT-2 small (`h=12, D_{in}=768, D_{out}=64`):
$$= 12 \times 3 \times 768 \times 64 = 1,769,472 \text{ parameters per MHA layer}$$

---

*End of Notes — Multi-Head Attention Part 1 (Wrapper Approach)*  
*Next: Part 2 — Weight Splits for Efficient Parallel MHA*
