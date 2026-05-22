# 📘 Build a Large Language Model from Scratch
## Lecture 15 — Self-Attention with Key, Query & Value Matrices
> **Source:** Vizuara YouTube Channel | **Instructor:** Dr. Raj Dandekar
> **Video:** https://youtu.be/UjdRN80c6p8
> **Series:** Stage 1 — Attention Mechanism

---

## 🗺️ What This Lecture Covers

1. Recap — Simplified Self-Attention (Lecture 14)
2. Why Trainable Weights are needed
3. The Three Weight Matrices — Query (WQ), Key (WK), Value (WV)
4. Step 1 — Convert Input Embeddings → Q, K, V Matrices
5. Step 2 — Compute Attention Scores (Q × Kᵀ)
6. Step 3 — Scale by √dk then Softmax → Attention Weights
7. Why √dk Scaling? Two Deep Reasons
8. Step 4 — Compute Context Vectors (Attention Weights × V)
9. Full Self-Attention Class in PyTorch (V1 and V2)
10. Intuitive Meaning of Query, Key, Value

---

## Attention Mechanism Roadmap

```
✅ Simplified Self-Attention (no trainable weights)  ← Lecture 14
🔥 Self-Attention with trainable weights (K, Q, V)   ← THIS LECTURE
⬜ Causal Attention (masking future tokens)
⬜ Multi-Head Attention (the actual GPT mechanism)
```

---

## 1. Recap — What We Did in Lecture 14

### The 3-Step Process (No Trainable Weights)

```
Step 1: Attention Scores  = dot_product(query, each_input_vector)
Step 2: Attention Weights = softmax(attention_scores)  ← sum to 1
Step 3: Context Vector    = Σ(attention_weight_i × input_vector_i)
```

### The Limitation
- Attention was based **only on semantic similarity** (dot product of embeddings)
- Example: "warm" should attend to "mat" (the mat is warm) — but their embeddings are perpendicular!
- Without trainable weights, this contextual relationship is missed entirely

---

## 2. The Solution — Trainable Weight Matrices

### The Core Idea
> Instead of using raw input embeddings directly for attention, we first **transform** them using learnable weight matrices. These matrices are optimized during training so the model learns meaningful attention patterns.

### Three Trainable Weight Matrices

| Matrix | Symbol | Produces | Purpose |
|--------|--------|---------|---------|
| Query Weight Matrix | **WQ** | Queries (Q) | "What am I looking for?" |
| Key Weight Matrix | **WK** | Keys (K) | "What do I contain?" |
| Value Weight Matrix | **WV** | Values (V) | "What do I return if selected?" |

---

## 3. Step 1 — Convert Input Embeddings → Q, K, V Matrices

### The Math

```
Q = Input × WQ    (Queries matrix)
K = Input × WK    (Keys matrix)
V = Input × WV    (Values matrix)
```

### Dimensions

```
Input Matrix:   [6 × 3]    (6 tokens, 3D embeddings)
WQ, WK, WV:    [3 × 2]    (3D in, 2D out)

Q = [6×3] × [3×2] = [6×2]
K = [6×3] × [3×2] = [6×2]
V = [6×3] × [3×2] = [6×2]
```

Each row of Q, K, V = one token's transformed representation.

> 💡 In GPT, D_in = D_out (dimensions are preserved). Here we use D_in=3, D_out=2 for illustration.

### Code

```python
import torch
import torch.nn as nn

# Input: 6 tokens, 3D embeddings
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # your     (X1)
    [0.55, 0.87, 0.66],  # journey  (X2)
    [0.57, 0.85, 0.64],  # starts   (X3)
    [0.22, 0.58, 0.33],  # with     (X4)
    [0.77, 0.25, 0.10],  # one      (X5)
    [0.05, 0.80, 0.55],  # step     (X6)
])

d_in, d_out = 3, 2

# Initialize trainable weight matrices (random, to be optimized later)
W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# For one token (journey = index 1)
x2 = inputs[1]
query_2 = x2 @ W_query   # shape: [2]
key_2   = x2 @ W_key     # shape: [2]
value_2 = x2 @ W_value   # shape: [2]

# For ALL tokens at once
queries = inputs @ W_query   # shape: [6, 2]
keys    = inputs @ W_key     # shape: [6, 2]
values  = inputs @ W_value   # shape: [6, 2]

print(queries.shape)  # torch.Size([6, 2])
print(keys.shape)     # torch.Size([6, 2])
print(values.shape)   # torch.Size([6, 2])
```

---

## 4. Step 2 — Compute Attention Scores

### For One Query (Journey)

```python
# Attention scores for journey = dot product of query_2 with ALL keys
# query_2 shape: [2]   keys shape: [6, 2]   keys.T shape: [2, 6]
# [1×2] @ [2×6] = [1×6]  → 6 attention scores

attn_scores_2 = query_2 @ keys.T   # shape: [6]
```

### For ALL Queries at Once

```python
# queries: [6×2]   keys.T: [2×6]
# [6×2] @ [2×6] = [6×6]

attn_scores = queries @ keys.T   # shape: [6, 6]
```

### What the 6×6 Attention Score Matrix Means

```
             your  journey  starts  with  one   step
your:       [s11,  s12,    s13,    s14,  s15,  s16]
journey:    [s21,  s22,    s23,    s24,  s25,  s26]  ← Row 2
starts:     [s31,  ...]
with:       [...]
one:        [...]
step:       [s61,  s62,    s63,    s64,  s65,  s66]

Row i = attention scores between query_i and ALL keys
sij = how much token i should attend to token j
```

---

## 5. Step 3 — Scale by √dk, then Softmax → Attention Weights

### The Formula

```
Attention Weights = softmax( (Q × Kᵀ) / √dk )
```

Where **dk** = dimension of the key vectors (= 2 in our example)

### Code

```python
d_k = keys.shape[-1]   # = 2

# Scale by sqrt(dk)
attn_scores_scaled = attn_scores / (d_k ** 0.5)

# Apply softmax (dim=-1 means normalize across each row)
attn_weights = torch.softmax(attn_scores_scaled, dim=-1)   # shape: [6, 6]

# Verify each row sums to 1
print(attn_weights.sum(dim=-1))
# tensor([1., 1., 1., 1., 1., 1.]) ✅
```

### Interpretable Attention Weights for "Journey"

```
Row 2 of attn_weights (journey query):
  your:    15%  attention
  journey: 22%  attention  ← highest (query = itself)
  starts:  22%  attention
  with:    13%  attention
  one:      9%  attention  ← lowest
  step:    18%  attention
```

---

## 6. Why Scale by √dk? — Two Deep Reasons

### Reason 1 — Prevent Peaked Softmax

When inputs to softmax are large, the output becomes extremely **peaked** (overconfident):

```python
x = torch.tensor([0.1, -2.0, 3.0, -2.0, 0.5])

# Normal softmax: diffuse distribution ✅
print(torch.softmax(x, dim=0))
# tensor([0.13, 0.02, 0.56, 0.02, 0.20])

# If we multiply by 8 first (large values):
print(torch.softmax(x * 8, dim=0))
# tensor([0.00, 0.00, 0.98, 0.00, 0.00])  ← extreme peak!
```

When softmax is peaked:
- The model is **overconfident** in one key
- Gradients become very small for all other keys
- Training becomes **unstable**

Dividing by √dk keeps values small → **diffuse, stable softmax**.

---

### Reason 2 — Control Variance of Dot Products

When you take a dot product Q·K, the **variance grows with dimension**:

```python
# Experiment: dot product variance vs dimension
import torch

for dim in [5, 20, 100]:
    var_list = []
    for _ in range(1000):
        q = torch.randn(dim)
        k = torch.randn(dim)
        dot = torch.dot(q, k)
        var_list.append(dot.item())
    variance = torch.tensor(var_list).var().item()
    print(f"dim={dim:3d}  variance before scaling: {variance:.1f}")

# Output:
# dim=  5  variance before scaling: 5.1
# dim= 20  variance before scaling: 19.8
# dim=100  variance before scaling: 101.3
```

**Without scaling:** variance ≈ dk (the dimension itself!)

**With √dk scaling:**
```python
    scaled_dot = dot / (dim ** 0.5)
    # variance ≈ 1.0 regardless of dimension!
```

> 💡 **Why variance ≈ 1 matters:** Backpropagation and gradient descent work best when values have consistent scale. Large variance → gradients explode or vanish → unstable training.

### Summary: Why √dk Specifically?

| Approach | Variance after scaling |
|----------|----------------------|
| No scaling | ≈ dk (grows with dimension) |
| Divide by dk | < 1 (shrinks too much) |
| **Divide by √dk** | **≈ 1.0 (stable!)** ✅ |

> This is why the mechanism is called **Scaled Dot-Product Attention**.

---

## 7. Step 4 — Compute Context Vectors

### The Formula

```
Context Vectors = Attention Weights × V
[6×6] × [6×2] = [6×2]
```

### Visual Intuition for "Journey"

```
Context Vector for Journey = weighted sum of ALL value vectors:

  V_your    × 0.15   ←  15% contribution
+ V_journey × 0.22   ←  22% contribution
+ V_starts  × 0.22   ←  22% contribution
+ V_with    × 0.13   ←  13% contribution
+ V_one     × 0.09   ←   9% contribution
+ V_step    × 0.18   ←  18% contribution
─────────────────────────────────────────
= Context Vector Z2   ← enriched representation of "journey"
```

### Code

```python
# Context vector for journey only
context_vec_2 = attn_weights[1] @ values   # shape: [2]

# Context vectors for ALL tokens
all_context_vecs = attn_weights @ values   # shape: [6, 2]
# Row 1 = context vector for "journey"

print(all_context_vecs.shape)  # torch.Size([6, 2])
```

### Why Context Vectors Are Richer than Embeddings

| Representation | Contains |
|----------------|---------|
| Input Embedding | Semantic meaning of the word itself |
| Context Vector | Semantic meaning + how this word relates to ALL other words |

---

## 8. Full Self-Attention Class in PyTorch

### Version 1 — Using nn.Parameter

```python
class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # Trainable weight matrices (initialized randomly)
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # Step 1: Compute Q, K, V
        queries = x @ self.W_query   # [n_tokens, d_out]
        keys    = x @ self.W_key     # [n_tokens, d_out]
        values  = x @ self.W_value   # [n_tokens, d_out]

        # Step 2: Attention Scores
        attn_scores = queries @ keys.T   # [n_tokens, n_tokens]

        # Step 3: Scale + Softmax → Attention Weights
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)

        # Step 4: Context Vectors
        context_vecs = attn_weights @ values   # [n_tokens, d_out]
        return context_vecs

# Usage
sa_v1 = SelfAttentionV1(d_in=3, d_out=2)
context_vecs = sa_v1(inputs)
print(context_vecs.shape)   # torch.Size([6, 2])
```

### Version 2 — Using nn.Linear (Preferred)

```python
class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        # nn.Linear has better weight initialization than nn.Parameter
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        context_vecs = attn_weights @ values
        return context_vecs

sa_v2 = SelfAttentionV2(d_in=3, d_out=2)
context_vecs = sa_v2(inputs)
print(context_vecs.shape)   # torch.Size([6, 2])
```

### V1 vs V2 — Why nn.Linear is Better

| | nn.Parameter | nn.Linear |
|--|-------------|-----------|
| Initialization | Uniform random | Kaiming/Xavier (optimized) |
| Training stability | Less stable | More stable |
| Used in practice | Rarely | Always |
| Bias support | Manual | Built-in (disabled here) |

> 💡 `nn.Linear` with `bias=False` is just a weight matrix multiplication — same as `nn.Parameter` but with better initialization.

---

## 9. Complete 4-Step Flow Summary

```
INPUT EMBEDDINGS [6×3]
        ↓ multiply by WQ, WK, WV [3×2]
        ↓
Q [6×2]    K [6×2]    V [6×2]
        ↓
STEP 2: Attention Scores = Q @ Kᵀ
        [6×2] @ [2×6] = [6×6]
        ↓
STEP 3: Scale by 1/√dk, then Softmax(dim=-1)
        → Attention Weights [6×6]  (each row sums to 1)
        ↓
STEP 4: Context Vectors = Attention Weights @ V
        [6×6] @ [6×2] = [6×2]
        ↓
CONTEXT VECTORS [6×2]  ← one enriched vector per token ✅
```

---

## 10. Intuitive Meaning of Query, Key, Value

### The Database Analogy

| Component | Analogy | In Attention |
|-----------|---------|-------------|
| **Query (Q)** | Search query | The current token asking "what should I focus on?" |
| **Key (K)** | Database index | Each token saying "here's what I'm about" |
| **Value (V)** | Database content | The actual information to retrieve if selected |

### Simple Intuition

```
When processing "journey":
  Query  = "What context am I looking for?"
  Keys   = Each word answering: "I'm about X"
  Scores = How well each key matches the query
  Values = What information to actually mix in

Result = Context Vector that knows:
  - "journey" means travel/process
  - "starts" is highly related (22%)
  - "one" is less related (9%)
```

---

## 🔑 Key Terms to Know

| Term | Simple Meaning |
|------|---------------|
| **WQ, WK, WV** | Trainable weight matrices that transform input embeddings |
| **Query (Q)** | Transformed version of the current focus token |
| **Key (K)** | Transformed version used for matching with queries |
| **Value (V)** | Transformed version used for computing the final output |
| **Attention Score** | Q × Kᵀ — raw alignment between query and key |
| **Scaling by √dk** | Divides attention scores by √(key dimension) for stability |
| **Attention Weight** | Normalized (softmax) attention score — sums to 1 |
| **Context Vector** | Weighted sum of value vectors — the final enriched output |
| **Scaled Dot-Product Attention** | The official name for this mechanism (because of √dk scaling) |
| **nn.Parameter** | PyTorch tensor marked as a trainable parameter |
| **nn.Linear** | PyTorch linear layer — better initialization than nn.Parameter |
| **requires_grad** | If True, gradients computed for backpropagation (training) |
| **Variance** | Statistical spread of values — kept ≈ 1 by √dk scaling |

---

## ⚡ Quick Summary

1. **Problem with Lecture 14:** No trainable weights → only semantic similarity, misses contextual importance
2. **Solution:** Three trainable weight matrices WQ, WK, WV transform inputs into Q, K, V
3. **Step 1:** Q = Input×WQ, K = Input×WK, V = Input×WV
4. **Step 2:** Attention Scores = Q × Kᵀ (dot product measures alignment)
5. **Step 3:** Scale by 1/√dk → Softmax → Attention Weights (sum to 1)
6. **Why √dk?** (1) Prevents peaked softmax (2) Keeps dot product variance ≈ 1
7. **Step 4:** Context Vectors = Attention Weights × V
8. **Context vector** = enriched embedding containing both meaning AND contextual relationships
9. `nn.Linear(bias=False)` preferred over `nn.Parameter` for better initialization
10. **This is Scaled Dot-Product Attention** — the core mechanism of GPT

---

## 🧠 Key Takeaways to Remember

- 🔹 **Q, K, V are all learned** — WQ, WK, WV are randomly initialized and optimized during training
- 🔹 **Q × Kᵀ** = attention scores | **softmax(Q×Kᵀ / √dk)** = attention weights
- 🔹 **√dk scaling** = prevents variance explosion AND prevents peaked softmax
- 🔹 **Context Vector = attention_weights × V** — a weighted mix of all value vectors
- 🔹 **Query** = "what I'm looking for" | **Key** = "what I offer" | **Value** = "what I contain"
- 🔹 **"Scaled Dot-Product Attention"** = the official name because of the √dk scaling step
- 🔹 **dim=-1 in softmax** = normalize across columns so each row sums to 1
- 🔹 Without trainable weights → only semantic similarity; **with K, Q, V → learns context**

---

## 📌 What's Coming Next

- **Lecture 16: Causal Attention**
- Prevents the model from "seeing the future" — masking tokens that come AFTER the current position
- Essential for GPT's next-word prediction task
- Implements upper-triangular masking in the attention weight matrix

---

*Notes prepared from Vizuara YouTube Series: "Build a Large Language Model from Scratch"*
*GitHub-ready markdown — push directly to your notes repo* 🚀
