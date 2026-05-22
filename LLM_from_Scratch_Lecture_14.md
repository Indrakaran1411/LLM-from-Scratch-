# 📘 Build a Large Language Model from Scratch
## Lecture 14 — Simplified Self-Attention Mechanism (No Trainable Weights)
> **Source:** Vizuara YouTube Channel | **Instructor:** Dr. Raj Dandekar
> **Video:** https://youtu.be/eSRhpYLerw4
> **Series:** Stage 1 — Attention Mechanism (Part 1 of 4)

---

## 🗺️ What This Lecture Covers

1. Why Attention is needed — beyond vector embeddings
2. The Goal: Embedding Vectors → Context Vectors
3. Step 1 — Attention Scores (dot product)
4. Step 2 — Normalization → Attention Weights (softmax)
5. Step 3 — Context Vector (weighted sum)
6. Extending to ALL tokens — the Attention Weight Matrix
7. Matrix multiplication shortcut for efficiency
8. Why trainable weights are needed (preview of next lecture)
9. Full Python implementation from scratch

---

## 1. Why Attention is Needed — Beyond Embeddings

### The Setup
Sentence: **"Your journey starts with one step"**

After pre-processing:
```
Tokens:       your    journey  starts  with   one    step
Token IDs:    [101,   204,     39,     567,   88,    92]
Embeddings:   [x1,    x2,      x3,     x4,    x5,    x6]   ← 3D vectors
```

### The Problem with Just Embeddings

The embedding vector for **"journey"** captures the meaning of "journey" — but it tells us **nothing** about:
- How "journey" relates to "starts"
- How "journey" relates to "step"
- Which words are most important **in this specific sentence context**

> 💡 Vector embeddings capture *what* a word means.
> But they don't capture *how* a word relates to all other words in its context.

### The Solution — Context Vectors

> **Context Vector** = An enriched embedding vector that captures both the word's own meaning AND its relationship to every other word in the sentence.

```
Embedding Vector (x2):    captures meaning of "journey" only
Context Vector (z2):      captures meaning of "journey" + how it relates to all other words
```

The **goal of every attention mechanism** — simplified or complex — is the same:
> **Convert embedding vectors → context vectors**

---

## 2. Notation & Setup

| Symbol | Meaning |
|--------|---------|
| `x1, x2, ..., x6` | Input embedding vectors (one per token) |
| `x2` | Embedding for "journey" (our focus token = **query**) |
| `z2` | Context vector for "journey" (what we want to compute) |
| **Query** | The token we are currently computing a context vector for |
| **Attention weights** | How much attention to pay to each token when computing a context vector |

### Input Tensor (3D embeddings for demo)

```python
import torch

inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # x1: "your"
    [0.55, 0.87, 0.66],  # x2: "journey"  ← our query
    [0.57, 0.85, 0.64],  # x3: "starts"
    [0.22, 0.58, 0.33],  # x4: "with"
    [0.77, 0.25, 0.10],  # x5: "one"
    [0.05, 0.80, 0.55],  # x6: "step"
])
# Shape: [6, 3] — 6 tokens, each a 3D vector
```

> 💡 Real GPT uses 768–12288 dimensional vectors. We use 3D here for visual clarity.

---

## 3. Step 1 — Attention Scores (Dot Product)

### The Core Question
> When computing the context vector for "journey", how much importance should each of the other words receive?

### Why Dot Product?

Think about it geometrically:
- If two vectors are **aligned** (small angle) → they are similar → high attention
- If two vectors are **perpendicular** (90° angle) → they are unrelated → low attention

The **dot product** captures exactly this:

```
dot product = |A| × |B| × cos(θ)

θ = 0°   (aligned)      → cos(0) = 1   → dot product = MAXIMUM
θ = 90°  (perpendicular) → cos(90) = 0 → dot product = ZERO
```

> 💡 **Key Insight:** Higher dot product = more aligned vectors = more attention should be paid

### Computing Attention Scores

```python
query = inputs[1]          # x2 = "journey" vector
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)  # dot product

print(attn_scores_2)
# tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
```

### Interpreting the Scores

| Token | Attention Score | Interpretation |
|-------|----------------|----------------|
| "your" | 0.9544 | Medium — somewhat aligned with "journey" |
| "journey" | 1.4950 | **Highest** — the query itself, perfectly aligned |
| "starts" | 1.4754 | **High** — closely aligned in meaning |
| "with" | 0.8434 | Lower — less related |
| "one" | 0.7070 | **Lowest** — nearly perpendicular to "journey" |
| "step" | 1.0865 | Medium-high — fairly aligned |

✅ This matches our intuition: "starts" and "step" should get more attention than "one" when looking at "journey"

---

## 4. Step 2 — Normalization → Attention Weights

### Why Normalize?

**Goal:** Convert attention scores to **attention weights** that sum to 1.0

**Reason 1 — Interpretability:**
```
Without normalization:  scores = [0.95, 1.49, 1.47, 0.84, 0.70, 1.08]  ← hard to interpret
With normalization:     weights = [0.13, 0.24, 0.23, 0.12, 0.10, 0.15]  ← 13%, 24%, 23%...
```
Now you can say: "Pay 24% attention to 'journey', 23% to 'starts', only 10% to 'one'"

**Reason 2 — Training stability:** Values between 0 and 1 help backpropagation converge.

---

### Method 1 — Simple Sum Normalization (Naive)

```python
attn_weights_2_naive = attn_scores_2 / attn_scores_2.sum()
print(attn_weights_2_naive.sum())  # tensor(1.0000) ✅
```

**Problem with naive normalization for extreme values:**

```
Scores: [1, 2, 3, 400]

Naive:    [1/406, 2/406, 3/406, 400/406] = [0.002, 0.005, 0.007, 0.985]
Ideal:    [~0,    ~0,    ~0,    ~1.0]    ← extreme value should dominate completely
```

The naive method doesn't push small values close enough to zero. This confuses the optimizer during backpropagation.

---

### Method 2 — Softmax Normalization ✅ (Preferred)

**Softmax formula:**
```
softmax(xi) = exp(xi) / Σ exp(xj)
```

With extreme values:
```
exp(400) ≈ ∞  → normalized ≈ 1.0   ✅
exp(1)   ≈ 2.7 → normalized ≈ 0.0  ✅
```

Softmax **amplifies differences** — high scores become close to 1, low scores become close to 0.

**Additional properties of softmax:**
- All outputs are **always positive** (due to exp)
- All outputs **sum to 1**
- Great for interpretability

### PyTorch Softmax Implementation

PyTorch's softmax subtracts the maximum value first to prevent numerical overflow:
```
softmax_pytorch(xi) = exp(xi - max) / Σ exp(xj - max)
```

Mathematically equivalent (the max cancels out) but **numerically stable**.

```python
# Naive softmax (works here, may have overflow for large values)
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_naive = softmax_naive(attn_scores_2)

# PyTorch softmax (always use this in practice)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print(attn_weights_2)
# tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])

print(attn_weights_2.sum())  # tensor(1.0000) ✅
```

### Interpreting Attention Weights

| Token | Weight | Attention % |
|-------|--------|------------|
| "your" | 0.1385 | 13.85% |
| "journey" | 0.2379 | **23.79%** ← highest (it's the query) |
| "starts" | 0.2333 | **23.33%** ← closely related |
| "with" | 0.1240 | 12.40% |
| "one" | 0.1082 | **10.82%** ← lowest (least related) |
| "step" | 0.1581 | 15.81% |

---

## 5. Step 3 — Context Vector (Weighted Sum)

### The Idea
> Scale each input embedding by its attention weight, then sum them all up.

```
z2 = 0.1385 × x1   +   0.2379 × x2   +   0.2333 × x3
   + 0.1240 × x4   +   0.1082 × x5   +   0.1581 × x6
```

**Visual intuition:**
- "journey" (0.2379) contributes the MOST to the context vector
- "starts" (0.2333) contributes nearly as much
- "one" (0.1082) contributes the LEAST

### Code

```python
context_vec_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(context_vec_2)
# tensor([0.4419, 0.6515, 0.5683])
```

### Why Context Vectors Are Powerful

```
x2 (embedding for "journey"):  [0.55, 0.87, 0.66]  ← only "journey" info
z2 (context vector):            [0.44, 0.65, 0.57]  ← "journey" + all other words
```

The context vector is an **enriched** version of the embedding — it now contains information about "journey" AND how it relates to every other word in the sentence.

---

## 6. Extending to ALL Tokens — The Attention Weight Matrix

We need context vectors for ALL 6 tokens, not just "journey".

### The 3-Step Process (same for every token)

```
For EACH token (query):
  Step 1: Compute attention scores  → dot product with all tokens
  Step 2: Normalize scores          → softmax → attention weights
  Step 3: Weighted sum              → context vector
```

### The Attention Weight Matrix

A 6×6 matrix where:
- **Row i** = attention weights for token i as the query
- **Entry [i, j]** = how much attention token i pays to token j

```
            your  journey  starts  with   one   step
your      [ 0.21,  0.18,   0.19,   0.17,  0.12,  0.13 ]
journey   [ 0.14,  0.24,   0.23,   0.12,  0.11,  0.16 ]  ← our row
starts    [ 0.13,  0.22,   0.24,   0.13,  0.10,  0.18 ]
with      [ 0.18,  0.16,   0.17,   0.21,  0.15,  0.13 ]
one       [ 0.20,  0.14,   0.15,   0.16,  0.22,  0.13 ]
step      [ 0.14,  0.19,   0.20,   0.13,  0.11,  0.23 ]
```

Each row sums to 1.0 ✅

---

## 7. Matrix Multiplication — The Efficient Way

### Naive Approach (slow — two for loops)
```python
attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
```

### Fast Approach (matrix multiplication)
```python
# Step 1: All attention scores at once
attn_scores = inputs @ inputs.T        # [6,3] × [3,6] = [6,6]

# Step 2: Softmax normalization (dim=-1 = normalize across columns of each row)
attn_weights = torch.softmax(attn_scores, dim=-1)   # [6,6]

# Step 3: All context vectors at once
all_context_vecs = attn_weights @ inputs             # [6,6] × [6,3] = [6,3]
```

**Why `dim=-1` in softmax?**
- For a 2D tensor, `dim=-1` = last dimension = columns
- This normalizes **across each row** so each row sums to 1

**Verify:**
```python
print(attn_weights.sum(dim=-1))
# tensor([1., 1., 1., 1., 1., 1.])  ✅ every row sums to 1

print(all_context_vecs[1])
# tensor([0.4419, 0.6515, 0.5683])  ✅ matches our earlier manual calculation for "journey"
```

### Why Matrix Multiplication Works

For the "journey" row (row 2), the matrix product computes:
```
context_vec[2] = 0.1385 × x1 + 0.2379 × x2 + 0.2333 × x3
               + 0.1240 × x4 + 0.1082 × x5 + 0.1581 × x6
```

This is **exactly** the weighted sum we computed manually — matrix multiplication just does it for all 6 tokens simultaneously and much faster!

---

## 8. Complete Implementation Summary

```python
import torch

# Input embeddings (6 tokens, 3D vectors)
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # "your"
    [0.55, 0.87, 0.66],  # "journey"
    [0.57, 0.85, 0.64],  # "starts"
    [0.22, 0.58, 0.33],  # "with"
    [0.77, 0.25, 0.10],  # "one"
    [0.05, 0.80, 0.55],  # "step"
])

# Step 1: Attention scores (dot products)
attn_scores = inputs @ inputs.T          # [6, 6]

# Step 2: Normalize → attention weights (softmax)
attn_weights = torch.softmax(attn_scores, dim=-1)   # [6, 6]

# Step 3: Context vectors (weighted sum)
all_context_vecs = attn_weights @ inputs             # [6, 3]

print("Attention Weights:\n", attn_weights)
print("\nContext Vectors:\n", all_context_vecs)

# Verify: context vector for "journey" (row index 1)
print("\nContext vector for 'journey':", all_context_vecs[1])
# tensor([0.4419, 0.6515, 0.5683])
```

---

## 9. Why We Still Need Trainable Weights

### The Limitation of This Approach

Currently we only pay attention to words that are **semantically similar** (aligned vectors).

But consider:
> "The cat sat on the mat **because it is warm**."

Query = **"warm"**

| Word | Semantic similarity to "warm" | Should we attend? |
|------|------------------------------|-------------------|
| "it" | Low — not similar in meaning | Yes! ("it" = the mat) |
| "mat" | Low — not similar in meaning | **YES!** (the mat is warm) |
| "warm" | Very high — same word | Yes (query itself) |

**Without trainable weights:**
- Only "warm" gets high attention (it's similar in meaning)
- "mat" gets low attention (not semantically related to "warm")
- But "mat" is the most important word in context!

**With trainable weights:**
- The model can **learn** from training data that "warm" should attend to "mat" in sentences like these
- This is how LLMs capture **long-range dependencies**

> 💡 Without trainable weights → only semantic similarity matters
> With trainable weights → the model learns contextual importance from data

This is exactly what we'll add in the next lecture with **Key, Query, Value** matrices!

---

## 🔑 Key Terms to Know

| Term | Simple Meaning |
|------|---------------|
| **Query** | The current token we're computing a context vector for |
| **Attention Score** | Dot product between query vector and another token's vector |
| **Attention Weight** | Normalized attention score (all weights sum to 1) |
| **Context Vector** | Enriched embedding — weighted sum of all input vectors |
| **Dot Product** | Mathematical operation that measures how aligned two vectors are |
| **Softmax** | Normalization function that outputs positive values summing to 1 |
| **Attention Weight Matrix** | 6×6 matrix of all attention weights (row i = weights for token i as query) |
| **dim=-1** | PyTorch parameter meaning "last dimension" — normalizes across columns |
| **Overflow** | Numerical error when values get too large for computer memory |
| **Underflow** | Numerical error when values get too small (rounded to zero) |
| **Naive Softmax** | Simple softmax without overflow protection |
| **PyTorch Softmax** | Numerically stable softmax — subtracts max before exponentiation |

---

## ⚡ Quick Summary

1. **Embedding vectors** capture word meaning but not word relationships
2. **Context vectors** = enriched embeddings that capture both meaning AND relationships
3. **Goal of attention:** Convert embedding vectors → context vectors
4. **Step 1 — Attention Scores:** Dot product between query and each input vector (higher dot product = more aligned = more important)
5. **Step 2 — Attention Weights:** Apply softmax to attention scores (all sum to 1, all positive)
6. **Step 3 — Context Vector:** Weighted sum of all input embeddings using attention weights
7. **Softmax vs naive normalization:** Softmax handles extreme values better, prevents overflow
8. **Matrix multiplication shortcut:** `attn_scores = inputs @ inputs.T` → `attn_weights = softmax(attn_scores)` → `context_vecs = attn_weights @ inputs`
9. **`dim=-1` in softmax:** Normalize across columns so each row sums to 1
10. **Limitation:** Without trainable weights, only semantically similar words get high attention — context-based importance is missed

---

## 🧠 Key Takeaways to Remember

- 🔹 **Attention converts embeddings → context vectors** (the entire point)
- 🔹 **Dot product = alignment measure** between two vectors (foundation of attention)
- 🔹 **Higher dot product = more aligned = higher attention score**
- 🔹 **Attention weights = normalized scores via softmax** (sum to 1, always positive)
- 🔹 **Context vector = weighted sum** of all input embeddings
- 🔹 **Matrix multiplication** does this efficiently for all tokens at once: `attn_weights @ inputs`
- 🔹 **Always use PyTorch softmax** — not naive — for numerical stability
- 🔹 **dim=-1** = normalize across last dimension (columns) so each row sums to 1
- 🔹 **Limitation:** no trainable weights = only semantic similarity captured, not context

---

## 📌 What's Coming Next — Lecture 15

**Self-Attention with Trainable Weights (Key, Query, Value)**
- Introduce **K, Q, V matrices** — the core of real attention in GPT
- Trainable weights allow the model to learn contextual importance from data
- "mat" can learn to attend to "warm" even without semantic similarity
- This is how long-range dependencies are truly captured

---

*Notes prepared from Vizuara YouTube Series: "Build a Large Language Model from Scratch"*
*GitHub-ready markdown — push directly to your notes repo* 🚀
