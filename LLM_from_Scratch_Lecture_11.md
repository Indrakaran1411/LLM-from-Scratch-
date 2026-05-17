# 📘 Build a Large Language Model from Scratch
## Lecture 11 — Positional Embeddings
> **Source:** Vizuara YouTube Channel | **Instructor:** Dr. Raj Dandekar
> **Video:** https://youtu.be/ufrPLpKnapU
> **Series:** Stage 1 — Data Preparation & Sampling

---

## 🗺️ What This Lecture Covers

1. Recap — Token Embeddings & the full pipeline so far
2. Why Positional Embeddings are needed
3. Absolute vs Relative Positional Embeddings
4. Which one does GPT use?
5. Hands-On: Building the full Input Embedding pipeline in PyTorch
6. Tensor Dimensions explained step by step

---

## 1. Recap — The LLM Input Pipeline So Far

```
Step 1: Raw Text → Tokens
        "The cat sat" → ["The", "cat", "sat"]

Step 2: Tokens → Token IDs (Vocabulary lookup)
        ["The", "cat", "sat"] → [5, 2, 3]

Step 3: Token IDs → Token Embeddings (semantic vectors)
        [5, 2, 3] → [[0.2, 0.8, ...], [0.5, 0.1, ...], ...]

Step 3.5: Token Embeddings + Positional Embeddings → Input Embeddings
        (THIS LECTURE)
              ↓
        Final Input to LLM Training
```

> 💡 **Token Embedding** captures *what* a word means.
> **Positional Embedding** captures *where* a word appears.
> Together → **Input Embedding** = the complete input to the Transformer.

---

## 2. Why Positional Embeddings Are Needed

### The Problem — Same Word, Different Position = Same Vector

Consider these two sentences:
```
Sentence 1: "The cat sat on the mat"
Sentence 2: "On the mat the cat sat"
```

- The word **"cat"** appears in both sentences
- It gets the same Token ID → same Token Embedding vector
- But **"cat"** is in **position 2** in Sentence 1 and **position 5** in Sentence 2
- **Position matters** — it changes meaning and context!

### What the Token Embedding Layer Fails to Capture

```
Token ID 5 (cat) in Sentence 1 → vector [0.5, 0.3, 0.8]
Token ID 5 (cat) in Sentence 2 → vector [0.5, 0.3, 0.8]  ← IDENTICAL!
```

The same token ID always maps to the same vector — **position is completely ignored**.

### The Solution — Add Positional Embeddings

> For each position in the input sequence, add a **unique positional vector** to the token's embedding vector.

```
Final Input Embedding = Token Embedding + Positional Embedding
```

Now:
```
cat at position 2 → token_embedding(cat) + positional_embedding(2)
cat at position 5 → token_embedding(cat) + positional_embedding(5)
```

These two are now **different vectors** — position is encoded! ✅

### Why Positional Embeddings Must Have the Same Dimension as Token Embeddings
- We need to **add** them together
- You can only add vectors of the **same dimension**
- So if token embeddings are 256-dimensional → positional embeddings must also be 256-dimensional

---

## 3. Absolute vs Relative Positional Embeddings

### Type 1 — Absolute Positional Embedding

> Each position in the sequence gets its own **unique embedding vector**.

**How it works:**
```
Position 1 → positional vector P1
Position 2 → positional vector P2
Position 3 → positional vector P3
Position 4 → positional vector P4
```

**Visual Example (context size = 4, vector dim = 3):**

| Token | Token Embedding | Position | Positional Embedding | Final Input Embedding |
|-------|----------------|----------|--------------------|--------------------|
| "the" | [1.0, 1.0, 1.0] | 1 | [1.1, 1.2, 1.3] | [2.1, 2.2, 2.3] |
| "cat" | [1.0, 1.0, 1.0] | 2 | [2.1, 2.2, 2.3] | [3.1, 3.2, 3.3] |
| "sat" | [1.0, 1.0, 1.0] | 3 | [3.1, 3.2, 3.3] | [4.1, 4.2, 4.3] |
| "mat" | [1.0, 1.0, 1.0] | 4 | [4.1, 4.2, 4.3] | [5.1, 5.2, 5.3] |

*(Token embeddings shown as identical for simplicity — in reality they differ)*

Even though token embeddings were the same, the **final input embeddings are all different** because different positional vectors were added. ✅

**Used by:**
- ✅ GPT-1, GPT-2, GPT-3, GPT-4
- ✅ Original Transformer ("Attention Is All You Need")

---

### Type 2 — Relative Positional Embedding

> Instead of encoding the exact position, encode the **distance/relationship between tokens**.

**How it works:**
- The model learns "token A is 3 positions before token B"
- Exact position (1st, 2nd, 3rd...) doesn't matter — only the **gap** matters

**When it's better:**
- Very **long sequences** (paragraphs, documents)
- When the **same phrase repeats** in different parts of the text
- When the model needs to generalize to sequence lengths it hasn't seen during training

**Example:**
- Absolute: "cat is at position 47" — model struggles if it only trained up to position 40
- Relative: "cat is 3 tokens before mat" — works regardless of where in the document

---

### Comparison Table

| Feature | Absolute | Relative |
|---------|---------|---------|
| What is encoded | Exact position (1st, 2nd, 3rd...) | Distance between tokens |
| Handles variable length | ❌ Struggles | ✅ Handles well |
| Used by GPT | ✅ Yes | ❌ No |
| Used by original Transformer | ✅ Yes (sinusoidal formula) | ❌ No |
| Best for | Fixed-length, sequence generation | Long sequences, repeated phrases |
| How values are set | Optimized during training (GPT) OR sinusoidal formula (original Transformer) | Learned during training |

---

### How Positional Embedding Values Are Determined

**Original Transformer paper ("Attention Is All You Need"):**
- Used a **sinusoidal/cosine formula** to compute fixed positional values
- Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d))

**GPT Models:**
- Did **NOT** use a formula
- Positional embedding values are initialized randomly and **optimized during training** — just like token embedding weights
- These are learned parameters, updated via backpropagation

---

## 4. Hands-On PyTorch Implementation

### Setup — Parameters

```python
import torch
import torch.nn as nn

vocab_size   = 50257   # GPT-2 vocabulary size (BPE tokenizer)
output_dim   = 256     # Vector dimension (GPT-2 small uses 768; we use 256 for demo)
context_size = 4       # Max tokens per input sequence
batch_size   = 8       # Number of sequences per batch
```

---

### Step 1 — Create the DataLoader (get input batches)

```python
# Using the data loader from Lecture 9
dataloader = create_dataloader(
    raw_text,
    batch_size=8,
    max_length=4,    # context_size = 4
    stride=4,
    shuffle=False
)

# Get one batch
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print(inputs.shape)   # torch.Size([8, 4])
# 8 sequences, each with 4 token IDs
```

**Input batch visualization:**
```
inputs = [
    [10,  8, 20, 21],   ← sequence 1: 4 token IDs
    [22,  3,  7, 15],   ← sequence 2: 4 token IDs
    [ 5, 19, 11,  2],   ← sequence 3: 4 token IDs
    ...                  ← 8 rows total
]
Shape: [8, 4]
```

---

### Step 2 — Create Token Embedding Layer

```python
# Token embedding: vocab_size rows, output_dim columns
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
# Matrix shape: [50257, 256] — randomly initialized

# Get token embeddings for the entire batch
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)   # torch.Size([8, 4, 256])
```

**Why 8 × 4 × 256?**
```
8   = batch size (8 sequences)
4   = context size (4 tokens per sequence)
256 = vector dimension per token

For each of the 8×4 = 32 token IDs → a 256-dimensional vector
```

**3D Tensor visualization:**
```
Token Embeddings [8, 4, 256]:

Batch 1, Token 1: [0.23, -0.11, 0.54, ..., 0.88]  ← 256 values
Batch 1, Token 2: [0.71,  0.33, -0.22, ..., 0.45] ← 256 values
Batch 1, Token 3: [-0.15, 0.67, 0.12, ..., -0.33] ← 256 values
Batch 1, Token 4: [0.44, -0.88, 0.91, ..., 0.21]  ← 256 values
Batch 2, Token 1: [...]
...
```

---

### Step 3 — Create Positional Embedding Layer

```python
# Positional embedding: context_size rows, output_dim columns
positional_embedding_layer = nn.Embedding(context_size, output_dim)
# Matrix shape: [4, 256] — one vector per position, randomly initialized

# Create position indices: [0, 1, 2, 3]
positions = torch.arange(context_size)  # tensor([0, 1, 2, 3])

# Get positional embeddings for all 4 positions
positional_embeddings = positional_embedding_layer(positions)
print(positional_embeddings.shape)   # torch.Size([4, 256])
```

**Why only 4 rows (not 8×4)?**
- We only have 4 **positions** (1st, 2nd, 3rd, 4th token)
- The same positional vectors apply to ALL sequences in the batch
- Position 1 always gets the same positional vector, regardless of which sequence it's in

---

### Step 4 — Add Token + Positional Embeddings = Input Embeddings

```python
input_embeddings = token_embeddings + positional_embeddings
print(input_embeddings.shape)   # torch.Size([8, 4, 256])
```

**How does Python add [8, 4, 256] + [4, 256]?**

> Python uses **broadcasting** — it automatically copies the [4, 256] positional embedding 8 times to match the [8, 4, 256] shape.

```
token_embeddings:      [8, 4, 256]
positional_embeddings: [4, 256]  → broadcast to [8, 4, 256]

Result: Each row of the batch gets the SAME 4 positional vectors added to it
```

**What broadcasting looks like:**
```
Batch 1: token_emb[1] + pos_emb   ← same pos_emb for all rows
Batch 2: token_emb[2] + pos_emb
Batch 3: token_emb[3] + pos_emb
...
Batch 8: token_emb[8] + pos_emb
```

---

### Final Dimension Summary

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `inputs` | [8, 4] | 8 sequences × 4 token IDs each |
| `token_embeddings` | [8, 4, 256] | Each token ID → 256-dim vector |
| `positional_embeddings` | [4, 256] | Each of 4 positions → 256-dim vector |
| `input_embeddings` | [8, 4, 256] | Final input to the LLM |

---

### Complete Code

```python
import torch
import torch.nn as nn

# Parameters
vocab_size   = 50257
output_dim   = 256
context_size = 4
batch_size   = 8

# Step 1: Get input batch from DataLoader
# inputs shape: [8, 4]

# Step 2: Token Embedding Layer
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
# Shape: [8, 4, 256]

# Step 3: Positional Embedding Layer
positional_embedding_layer = nn.Embedding(context_size, output_dim)
positions = torch.arange(context_size)          # [0, 1, 2, 3]
positional_embeddings = positional_embedding_layer(positions)
# Shape: [4, 256]

# Step 4: Add together → Input Embeddings
input_embeddings = token_embeddings + positional_embeddings
# Shape: [8, 4, 256]  ← final input to LLM
```

---

## 5. The Embedding Weight Matrices — What Gets Optimized

During LLM training, **two embedding matrices** are optimized:

| Matrix | Shape | What it learns |
|--------|-------|---------------|
| **Token Embedding Matrix** | [50257, 256] | Semantic meaning of each token |
| **Positional Embedding Matrix** | [4, 256] | Meaning of each position |

Both are:
1. Initialized with **random values**
2. Updated via **backpropagation** during training
3. Gradually learn meaningful representations

---

## 🔑 Key Terms to Know

| Term | Simple Meaning |
|------|---------------|
| **Positional Embedding** | A vector added to the token embedding to encode position in the sequence |
| **Input Embedding** | Token Embedding + Positional Embedding — the final input to the LLM |
| **Absolute Positional Embedding** | Each position gets a unique fixed vector |
| **Relative Positional Embedding** | Encodes the distance/gap between tokens, not their exact positions |
| **Context Size / Context Length** | Max number of tokens in one input sequence |
| **Broadcasting** | Python automatically extends a smaller tensor to match a larger one for addition |
| **Sinusoidal Encoding** | Fixed formula used in the original Transformer paper for positional encoding |
| **torch.arange(n)** | Creates tensor [0, 1, 2, ..., n-1] — used to generate position indices |
| **3D Tensor** | A tensor with 3 dimensions, e.g. [batch, sequence, embedding_dim] |

---

## ⚡ Quick Summary

1. **Token embeddings** alone are insufficient — the same word gets the same vector regardless of position
2. Position matters — *"cat sat on mat"* ≠ *"mat sat on cat"*
3. **Positional Embedding** = a unique vector for each position, added to the token embedding
4. **Input Embedding = Token Embedding + Positional Embedding**
5. Positional vectors must have the **same dimension** as token vectors (to enable addition)
6. **Absolute** (exact position) — used by GPT, original Transformer
7. **Relative** (distance between tokens) — better for long sequences
8. GPT learns positional embedding values via **training** (unlike original Transformer's fixed formula)
9. Token embeddings: **[8, 4, 256]** | Positional embeddings: **[4, 256]** → broadcast + add → **[8, 4, 256]**
10. Both token and positional embedding matrices are **randomly initialized** and **optimized during training**

---

## 🧠 Key Takeaways to Remember

- 🔹 **Why positional embeddings?** Same word at different positions needs different vectors
- 🔹 **Formula:** Input Embedding = Token Embedding + Positional Embedding
- 🔹 **Must have same dimension** — you can only add equal-sized vectors
- 🔹 **Absolute** = exact position (GPT uses this) | **Relative** = distance between tokens
- 🔹 **GPT** learns positional values via training | **Original Transformer** uses sinusoidal formula
- 🔹 **Dimensions:** [batch=8, context=4, dim=256] — understand this and you understand everything
- 🔹 **Broadcasting** = Python copies the [4, 256] positional vector to match [8, 4, 256] automatically
- 🔹 Both embedding matrices start **random** and become meaningful through **backpropagation**

---

## 📌 What's Coming Next

- **Attention Mechanism** — the heart of the Transformer
- Understanding **Keys, Queries, and Values**
- Coding **Self-Attention from scratch** in Python

---

*Notes prepared from Vizuara YouTube Series: "Build a Large Language Model from Scratch"*
*GitHub-ready markdown — push directly to your notes repo* 🚀
