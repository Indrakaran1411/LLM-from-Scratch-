# Lecture 26 — Measuring the LLM Loss Function
### Series: Build Large Language Models from Scratch | Stage 2: Training

---

## Table of Contents

1. [Why Loss Functions Matter in LLMs](#1-why-loss-functions-matter-in-llms)
2. [The 7-Step Training Pipeline](#2-the-7-step-training-pipeline)
3. [GPT Model Configuration Deep Dive](#3-gpt-model-configuration-deep-dive)
4. [GPT Architecture Recap](#4-gpt-architecture-recap)
5. [Text Generation Recap — How the Model Produces Output](#5-text-generation-recap--how-the-model-produces-output)
6. [Inputs and Targets — The Foundation of Supervised Training](#6-inputs-and-targets--the-foundation-of-supervised-training)
7. [The Probability Output Matrix — Explained with 7-Token Vocabulary](#7-the-probability-output-matrix--explained-with-7-token-vocabulary)
8. [Why the Untrained Model Produces Garbage](#8-why-the-untrained-model-produces-garbage)
9. [Deriving the Loss Function from First Principles](#9-deriving-the-loss-function-from-first-principles)
10. [PyTorch Implementation — Step by Step](#10-pytorch-implementation--step-by-step)
11. [Cross-Entropy: The Mathematical Foundation](#11-cross-entropy-the-mathematical-foundation)
12. [Perplexity — The Interpretable Loss Metric](#12-perplexity--the-interpretable-loss-metric)
13. [Full End-to-End Pipeline Summary](#13-full-end-to-end-pipeline-summary)
14. [Common Bugs and Misconceptions](#14-common-bugs-and-misconceptions)
15. [Senior Engineer Mental Models](#15-senior-engineer-mental-models)
16. [What's Coming Next (Lectures 27–30)](#16-whats-coming-next-lectures-2730)
17. [Cheat Sheet + Flashcards](#17-cheat-sheet--flashcards)

---

## 1. Why Loss Functions Matter in LLMs

### The Problem with Traditional Loss Functions

In classical ML:

| Model Type | Loss Function | Why It Works |
|---|---|---|
| Regression | Mean Squared Error (MSE) | Output is continuous; penalizes Euclidean distance |
| Binary Classification | Binary Cross-Entropy | Output is a probability over 2 classes |
| Multi-class Classification | Categorical Cross-Entropy | Output is a probability distribution over N classes |

For **Large Language Models**, none of the above directly apply in the naive sense because:

- The output is a **distribution over 50,000+ vocabulary tokens**
- The task is **autoregressive** — each token prediction feeds into the next
- There are **multiple prediction tasks per single forward pass** (one per input token position)
- The model is generating **sequences**, not single labels

### The Fundamental Question

> "My LLM is giving random outputs. How do I turn the qualitative intuition of 'this is bad' into a quantitative number I can minimize with gradient descent?"

That number is the **loss function**. Once you have a differentiable scalar loss, you can:

1. Compute gradients with respect to all model parameters via backpropagation
2. Use gradient descent (Adam, AdamW, etc.) to update weights
3. Repeat until the model produces good outputs

This is exactly the same optimization loop used in all neural networks — the key insight of this lecture is that **training an LLM reduces to a problem we've already solved**.

---

## 2. The 7-Step Training Pipeline

This lecture covers **Steps 1 and 2** of the full training pipeline:

```
┌──────────────────────────────────────────────────────────────────┐
│              COMPLETE LLM TRAINING PIPELINE                      │
├────┬─────────────────────────────────────┬───────────────────────┤
│ 1  │ Text Generation                     │ ← THIS LECTURE (recap)│
│ 2  │ Text Evaluation / Loss Function     │ ← THIS LECTURE (main) │
│ 3  │ Feed full dataset → get logits      │   Next lecture        │
│ 4  │ Compute train & validation loss     │   Next lecture        │
│ 5  │ Build training loop (backprop)      │   Future lecture      │
│ 6  │ Tune hyperparameters                │   Future lecture      │
│ 7  │ Load pre-trained OpenAI weights     │   End of series       │
└────┴─────────────────────────────────────┴───────────────────────┘
```

### Where We Are in the Full Series

```
STAGE 1 — Architecture (COMPLETE)
  └── Data Prep & Sampling
  └── Attention Mechanism (Multi-head Self-Attention)
  └── GPT Architecture (Transformer blocks, LayerNorm, etc.)

STAGE 2 — Training (CURRENT)
  └── Lecture 26: Loss Function ← YOU ARE HERE
  └── Lecture 27: Dataset + Train/Val Split
  └── Lecture 28: Training Loop
  └── ...
```

---

## 3. GPT Model Configuration Deep Dive

```python
GPT_CONFIG_124M = {
    "vocab_size":     50257,   # tiktoken BPE vocabulary
    "context_length": 256,     # reduced from 1024 for local training
    "emb_dim":        768,     # token embedding dimension
    "n_heads":        12,      # multi-head attention heads
    "n_layers":       12,      # number of transformer blocks stacked
    "drop_rate":      0.1,     # dropout probability
    "qkv_bias":       False    # no bias in Q, K, V weight matrices
}
```

### Why Each Parameter Exists

**`vocab_size = 50257`**
- This is the exact vocabulary size used by GPT-2 and all early OpenAI models
- Uses `tiktoken` (tick-token) with **Byte Pair Encoding (BPE)**
- BPE creates sub-word tokens: a single word can be 1 token or split into multiple tokens
- This is why "1 word ≠ 1 token" — important for understanding sequence lengths

**`context_length = 256`** (GPT-2 uses 1024)
- The maximum number of tokens the model can "see" at once
- Reduced to 256 here so you can train on a laptop in **2–3 minutes**
- The architecture is identical at 1024 — you just need more RAM/VRAM
- Determines the size of positional embeddings

**`emb_dim = 768`**
- Each token ID maps to a 768-dimensional float vector
- Higher dimensions capture more semantic meaning
- This is the "width" of the network

**`n_heads = 12`**
- Number of parallel self-attention heads inside each transformer block
- Each head attends to different relationships in the sequence
- Must divide evenly into `emb_dim`: 768 / 12 = 64 dims per head

**`n_layers = 12`**
- Number of stacked transformer blocks
- Depth of the network — more layers = more abstract representations
- GPT-2 small = 12 layers, GPT-2 XL = 48 layers

**`drop_rate = 0.1`**
- 10% of activations randomly set to 0 during training
- Prevents overfitting; set to 0 during inference

**`qkv_bias = False`**
- Whether to add a bias term when computing Query, Key, Value matrices
- GPT-2 originally used bias; many modern models omit it for cleanliness

---

## 4. GPT Architecture Recap

```
INPUT: Token IDs  [16833, 3626, 610]
         │
         ▼
┌─────────────────────┐
│  Token Embedding    │  → maps each token ID to a 768-dim vector
│  (nn.Embedding)     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Positional Embedding│  → adds position-aware signal (learned)
│  (nn.Embedding)     │  → same dim: 768
└─────────────────────┘
         │  [token emb + pos emb]
         ▼
┌─────────────────────┐
│     Dropout         │  drop_rate = 0.1
└─────────────────────┘
         │
         ▼
┌─────────────────────┐  ┐
│  Transformer Block  │  │
│  ┌───────────────┐  │  │
│  │  LayerNorm    │  │  │
│  │  Multi-Head   │  │  │  × n_layers (12)
│  │  Attention    │  │  │
│  │  Dropout      │  │  │
│  │  LayerNorm    │  │  │
│  │  Feed Forward │  │  │
│  │  Dropout      │  │  │
│  └───────────────┘  │  │
└─────────────────────┘  ┘
         │
         ▼
┌─────────────────────┐
│  Final LayerNorm    │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Linear Head        │  → projects 768-dim → 50257-dim (vocab size)
│  (nn.Linear)        │
└─────────────────────┘
         │
         ▼
OUTPUT: LOGITS  shape = (batch, seq_len, 50257)
```

**Key output: LOGITS** — raw unnormalized scores, NOT probabilities yet. The final Linear layer has no activation function applied.

---

## 5. Text Generation Recap — How the Model Produces Output

### The `generate_text_simple` Function

From the previous lecture, this function:
1. Takes the model + an input token sequence
2. Autoregressively generates `max_new_tokens` new tokens
3. At each step: forward pass → logits → argmax → append to sequence

```python
# Example call from lecture:
output_text = generate_text_simple(
    model=model,
    idx=text_to_token_ids("every effort moves you"),
    max_new_tokens=10,
    context_size=GPT_CONFIG["context_length"]
)
# Result: "every effort moves you <GARBAGE TOKENS>"
# The model outputs random tokens because it hasn't been trained
```

### The Two Utility Functions Defined

```python
def text_to_token_ids(text, tokenizer):
    """Converts a string to a tensor of token IDs"""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # adds batch dim
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """Converts token IDs back to human-readable string"""
    flat = token_ids.squeeze(0)  # removes batch dim
    return tokenizer.decode(flat.tolist())
```

These are inverses of each other. You can verify:
```python
text = "every effort moves you"
assert token_ids_to_text(text_to_token_ids(text)) == text  # True
```

---

## 6. Inputs and Targets — The Foundation of Supervised Training

### Why This is Supervised Learning

The LLM training task is framed as **supervised learning** even though we don't hand-label data. The labels (targets) come **for free** from the text itself — the next token in the sequence is always the label for the current position.

### The Input Tensor

```python
inputs = torch.tensor([
    [16833, 3626,  610],   # Batch 1: "every", "effort", "moves"
    [   40, 1107,  588]    # Batch 2: "I",     "really", "like"
])
# Shape: (2, 3)  →  (num_batches, context_length)
```

### The Target Tensor — Critical Insight

Every input position generates a prediction task. For N input tokens, there are N prediction tasks. This is the **causal language modeling** objective.

```
BATCH 1: "every effort moves"

  Position 0: INPUT="every"              → PREDICT: "effort"   → Target ID: 3626
  Position 1: INPUT="every effort"       → PREDICT: "moves"    → Target ID:  610
  Position 2: INPUT="every effort moves" → PREDICT: "U"        → Target ID:  345

BATCH 2: "I really like"

  Position 0: INPUT="I"                  → PREDICT: "really"   → Target ID: 1107
  Position 1: INPUT="I really"           → PREDICT: "like"     → Target ID:  588
  Position 2: INPUT="I really like"      → PREDICT: "chocolate"→ Target ID: 11311
```

```python
targets = torch.tensor([
    [ 3626,   610,   345],   # Batch 1 targets
    [ 1107,   588, 11311]    # Batch 2 targets
])
# Shape: (2, 3)  — same shape as inputs
```

### The Shift Relationship (IMPORTANT)

```
inputs  = [t₀,   t₁,   t₂  ]
targets = [t₁,   t₂,   t₃  ]
              ↑
         targets = inputs shifted LEFT by 1 + 1 new token
```

In practice, if you have a sequence `[t₀, t₁, t₂, t₃]`:
- `inputs  = [t₀, t₁, t₂]`
- `targets = [t₁, t₂, t₃]`

This means **targets are just the next tokens in the original sequence** — no labeling required. This is the magic of self-supervised learning.

---

## 7. The Probability Output Matrix — Explained with 7-Token Vocabulary

The lecture uses a simplified 7-word vocabulary to illustrate what's happening inside the model. Real vocab = 50,257, but concept is identical.

### Simplified Vocabulary

```
Index: 0    1       2      3        4      5   6
Token: "a" "effort" "every" "forward" "moves" "U" "Zoo"
```

### Probability Matrix After Softmax

```
                 a      effort  every  forward  moves   U     Zoo
                ──────────────────────────────────────────────────
Row 0 (every) → [0.10,  0.60,  0.20,   0.05,  0.02, 0.01, 0.02]
Row 1 (effort)→ [0.00,  0.10,  0.00,   0.00,  0.70, 0.10, 0.10]
Row 2 (moves) → [0.00,  0.00,  0.00,   0.00,  0.00, 0.80, 0.20]
```

### Reading the Matrix

- **Each row** = one prediction task
- **Each column** = probability of that vocabulary token being next
- **All values in a row** sum to 1.0 (because softmax)
- **argmax per row** = the model's predicted next token

```
Row 0 argmax → index 1 → "effort"  ✓ (correct! given "every")
Row 1 argmax → index 4 → "moves"   ✓ (correct! given "every effort")
Row 2 argmax → index 5 → "U"       ✓ (correct! given "every effort moves")
```

> This idealized matrix is what a **trained model** would look like. An **untrained model** produces near-uniform random probabilities — the argmax will be meaningless.

### Actual Dimensions in Our Code

```
Shape of logits:       (batch=2, tokens=3, vocab=50257)
Shape after softmax:   (batch=2, tokens=3, vocab=50257)
```

For each of the 6 token positions (2 batches × 3 tokens), there is a **vector of 50,257 probabilities**.

---

## 8. Why the Untrained Model Produces Garbage

Before training:
- All weights are **randomly initialized**
- The model has no knowledge of language, grammar, or semantics
- For `inputs = "every effort moves"`, the output might be: `"armed H Netflix"`
- This is **expected** — it's the starting point for training

The loss function quantifies exactly how far this garbage output is from the correct output. Minimizing the loss IS the training process.

```
Before Training:
  Input:  "every effort moves"
  Output: "armed H Netflix"       ← garbage
  Loss:   10.79                   ← very high
  Perplexity: 48,725              ← model is nearly random

After Training:
  Input:  "every effort moves"
  Output: "you forward progress"  ← coherent, contextually appropriate
  Loss:   < 2.0                   ← low
  Perplexity: < 10                ← model is confident and correct
```

---

## 9. Deriving the Loss Function from First Principles

### Step 1 — Identify What We Want to Maximize

For each of the 6 prediction tasks, we want the probability assigned to the **correct** (target) token to be as high as possible, ideally = 1.0.

Let's name these target probabilities:

```
Batch 1: P₁₁, P₁₂, P₁₃   ← probabilities at target token indices
Batch 2: P₂₁, P₂₂, P₂₃
```

**Goal:** All Pᵢⱼ → 1.0

### Step 2 — Why Not Maximize Directly?

We could maximize `mean(P₁₁, P₁₂, ..., P₂₃)`. But this has problems:
- Probabilities are multiplied in likelihood: we actually want to maximize the **joint probability** P₁₁ × P₁₂ × ... × P₂₃
- Multiplying many small numbers causes **numerical underflow** (floating point goes to 0)

### Step 3 — Apply Log (The Standard Trick)

Taking log converts the product into a sum:

```
log(P₁₁ × P₁₂ × ... × P₂₃)
  = log(P₁₁) + log(P₁₂) + ... + log(P₂₃)
```

No more underflow. And since log is monotonically increasing, maximizing `log(product)` is the same as maximizing the product.

**This is called Log-Likelihood.**

### Step 4 — Normalize (Take the Mean)

```
Log-Likelihood = (1/6) × [log(P₁₁) + log(P₁₂) + log(P₁₃) + log(P₂₁) + log(P₂₂) + log(P₂₃)]
```

### Step 5 — Negate (Convention: Minimize Instead of Maximize)

Since log(p) is always negative for p ∈ (0, 1):

```
Negative Log-Likelihood (NLL) = -Log-Likelihood
                               = -(1/6) × Σ log(Pᵢⱼ)
```

Now we have a **positive loss** that we **minimize**. This is the universal convention in deep learning.

### Why the Loss Curve Looks the Way It Does

```
NLL Loss
   ↑
∞  │ ·
   │  ·
   │   ·
   │    ·
   │     ·
   │       ·
   │          ·
   │              ·
   │                    ·           ·
 0 └─────────────────────────────────────→ P (target probability)
   0                                    1
   ↑                                    ↑
Worst case                         Best case
(model assigns                   (model is 100%
 0% to correct token)             confident and correct)
```

As P → 1 (model becomes more confident on the correct token), NLL Loss → 0.
As P → 0 (model is completely wrong), NLL Loss → ∞.

### Mathematical Equivalence to Cross-Entropy

Cross-entropy between the true distribution Q (one-hot: 1 at target index, 0 elsewhere) and the predicted distribution P is:

```
H(Q, P) = -Σ Q(x) × log(P(x))
```

Since Q is one-hot (only 1 at the target index), all terms are 0 except the target:

```
H(Q, P) = -log(P_target)
```

Averaged over all positions:

```
Cross-Entropy Loss = -(1/N) × Σ log(P_target_i)
```

**This is exactly the NLL loss.** Cross-entropy = NLL for one-hot classification. They are the same thing.

---

## 10. PyTorch Implementation — Step by Step

### Setup

```python
import torch
import torch.nn as nn
import tiktoken

# Initialize tokenizer (same BPE as GPT-2)
tokenizer = tiktoken.get_encoding("gpt2")

# Initialize model
model = GPTModel(GPT_CONFIG_124M)
model.eval()  # no dropout during forward pass for loss computation
```

### Define Inputs and Targets

```python
# Token IDs for: "every effort moves" and "I really like"
inputs = torch.tensor([
    [16833, 3626,  610],   # every, effort, moves
    [   40, 1107,  588]    # I, really, like
])

# Targets = next tokens at each position
targets = torch.tensor([
    [ 3626,   610,   345],  # effort, moves, U(you)
    [ 1107,   588, 11311]   # really, like, chocolate
])
```

### Forward Pass → Logits

```python
with torch.no_grad():  # no gradient computation during evaluation
    logits = model(inputs)

print(logits.shape)  # torch.Size([2, 3, 50257])
```

### Convert to Probabilities (for Inspection Only)

```python
probas = torch.softmax(logits, dim=-1)
print(probas.shape)  # torch.Size([2, 3, 50257])
```

> **Note:** You don't actually need to compute `probas` explicitly for the loss. `F.cross_entropy` handles softmax internally. Computing `probas` here is for understanding.

### Get Predicted Token IDs

```python
# argmax along vocab dimension (dim=-1)
predicted_token_ids = torch.argmax(probas, dim=-1)
print(predicted_token_ids)
# tensor([[XXXXX, YYYYY, ZZZZZ],   ← batch 1 predictions (garbage before training)
#         [AAAAA, BBBBB, CCCCC]])  ← batch 2 predictions (garbage before training)
```

### Manual Loss Computation (Long Way — for Understanding)

```python
# Step 1: Extract target probabilities for batch 1
# probas[batch_idx, token_position, target_token_id]
target_probs_batch1 = probas[0, [0, 1, 2], targets[0]]
# → tensor([P₁₁, P₁₂, P₁₃])

# Step 2: Extract target probabilities for batch 2
target_probs_batch2 = probas[1, [0, 1, 2], targets[1]]
# → tensor([P₂₁, P₂₂, P₂₃])

# Step 3: Concatenate all 6 target probabilities
all_target_probs = torch.cat([target_probs_batch1, target_probs_batch2])
# → tensor([P₁₁, P₁₂, P₁₃, P₂₁, P₂₂, P₂₃])

# Step 4: Log of each
log_probs = torch.log(all_target_probs)
# → all values negative (log of number < 1)

# Step 5: Mean
mean_log_prob = torch.mean(log_probs)

# Step 6: Negate → Loss
loss = -mean_log_prob
print(f"Manual loss: {loss:.4f}")  # ~10.79
```

### ⭐ One-Line Loss Computation (Production Way)

```python
# Step 1: Flatten logits from (2, 3, 50257) → (6, 50257)
logits_flat = logits.flatten(0, 1)
# flatten(0, 1) merges dimension 0 and 1 together

# Step 2: Flatten targets from (2, 3) → (6,)
targets_flat = targets.flatten()

# Step 3: Compute cross-entropy loss
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(f"Loss: {loss:.4f}")  # ~10.79
```

### What `F.cross_entropy` Does Internally

```
logits_flat (6, 50257)
       │
       ▼  log_softmax(dim=-1)    ← numerically stable: log(softmax(x))
       │                            equivalent to softmax then log
       │                            but avoids numerical overflow
       ▼
For each of 6 positions: pick the value at the target index
       │
       ▼  [log(P₁₁), log(P₁₂), log(P₁₃), log(P₂₁), log(P₂₂), log(P₂₃)]
       │
       ▼  mean()
       │
       ▼  negate
       │
       ▼
SCALAR LOSS  (e.g., 10.79)
```

> **Important:** PyTorch's `F.cross_entropy` uses `log_softmax` (not `softmax` then `log`) for numerical stability. The result is mathematically identical but numerically safer.

### Why Flatten? — The Shape Contract

`F.cross_entropy` has a strict API:
```
logits:  (N, C)  — N samples, C classes
targets: (N,)    — N integer class indices
```

Our logits are `(batch=2, seq=3, vocab=50257)`. We need to collapse batch and seq into one "N":
```
(2, 3, 50257) → flatten(0,1) → (6, 50257)   ✓ matches (N=6, C=50257)
(2, 3)        → flatten()    → (6,)          ✓ matches (N=6,)
```

---

## 11. Cross-Entropy: The Mathematical Foundation

### What Cross-Entropy Actually Measures

Cross-entropy measures the **distance between two probability distributions**:
- **Q**: the true distribution (one-hot: 1 at target token, 0 everywhere else)
- **P**: the model's predicted distribution (softmax output over vocab)

```
H(Q, P) = -Σₓ Q(x) × log P(x)
```

Because Q is one-hot, this collapses to:

```
H(Q, P) = -log P(target_token)
```

**Intuition:** The model's predicted probability of the correct answer, on a log scale, negated. Higher loss = model was less confident about the right answer.

### Cross-Entropy vs MSE — Why Not MSE for LLMs?

MSE computes `(predicted - target)²`. But:
- Token IDs are **categorical** (token 3626 is not "3626 units away" from token 3625)
- MSE assumes a continuous ordinal output space — inappropriate for token IDs
- Cross-entropy is correct for **probability distributions over discrete categories**

### The Baseline Loss

For an untrained model with vocab size V, the expected loss is:

```
Baseline Loss = log(V) = log(50257) ≈ 10.82
```

This is because a random model assigns uniform probability `1/V` to each token:
```
Loss = -log(1/V) = log(V)
```

The lecture's reported loss of **10.79** is very close to `log(50257) = 10.82` — confirming the untrained model is essentially random.

> **Engineer's sanity check:** Always verify your initial loss ≈ `log(vocab_size)`. If it's wildly off, something is wrong with your implementation.

---

## 12. Perplexity — The Interpretable Loss Metric

### Definition

Perplexity is defined as the exponent of the cross-entropy loss:

```
Perplexity = e^(Loss)
```

Or equivalently, if loss = average NLL over N tokens:

```
Perplexity = e^(-(1/N) Σ log P(tᵢ))
           = (∏ P(tᵢ))^(-1/N)
           = (geometric mean of token probabilities)^(-1)
```

### Interpretation

Perplexity answers: **"How many tokens is the model effectively choosing uniformly between at each step?"**

If Perplexity = K, the model behaves as if it randomly selects the next token from K equally likely options.

```
Perplexity  ≈ 1       → model is perfect (always picks the right token)
Perplexity  ≈ 2       → model is almost always correct
Perplexity  ≈ 10      → good model, some uncertainty
Perplexity  ≈ 100     → weak model
Perplexity  ≈ 1,000   → very poor
Perplexity  ≈ 50,000  → essentially random (untrained)
```

### From This Lecture

```
Loss       = 10.79
Perplexity = e^10.79 ≈ 48,725

Interpretation: The untrained model, when given "every effort moves",
is as uncertain as if it had to pick from 48,725 tokens randomly.
Vocab = 50,257. So essentially all tokens are equally likely — pure noise.
```

### Why Perplexity is Better for Communication

| Metric | Value | Understanding |
|--------|-------|---------------|
| Cross-Entropy Loss | 10.79 | Hard to interpret — what does "10.79" mean? |
| Perplexity | 48,725 | Clear — the model is randomly choosing from ~48k tokens |

When reporting model quality to a non-ML audience (or even to yourself during debugging), perplexity is the preferred metric because it's directly interpretable in terms of the vocabulary size.

### Computing Perplexity in Code

```python
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
perplexity = torch.exp(loss)
print(f"Loss: {loss:.2f}")           # 10.79
print(f"Perplexity: {perplexity:.0f}")  # 48725
```

---

## 13. Full End-to-End Pipeline Summary

```
RAW TEXT
  "every effort moves you"
  "I really like chocolate"
        │
        ▼
TOKENIZER (tiktoken BPE)
  [16833, 3626, 610, 345]
  [40, 1107, 588, 11311]
        │
        ├──────────────────────────────────────────────────────────┐
        ▼                                                          ▼
   INPUTS (no last token)                               TARGETS (no first token)
   Batch 1: [16833, 3626,  610]                         Batch 1: [3626,  610,  345]
   Batch 2: [   40, 1107,  588]                         Batch 2: [1107,  588, 11311]
   Shape: (2, 3)                                        Shape: (2, 3)
        │
        ▼
GPT MODEL FORWARD PASS
  Token Emb + Pos Emb → Dropout → 12× Transformer Blocks → LayerNorm → Linear
        │
        ▼
LOGITS
  Shape: (2, 3, 50257)  ← raw unnormalized scores
        │
        ▼
FLATTEN LOGITS + TARGETS
  logits_flat:  (6, 50257)
  targets_flat: (6,)
        │
        ▼
F.cross_entropy(logits_flat, targets_flat)
  Internally: log_softmax → index by target → mean → negate
        │
        ▼
SCALAR LOSS: 10.79
        │
        ▼
PERPLEXITY: e^10.79 = 48,725
        │
        ▼ (next lectures)
BACKPROPAGATION
  loss.backward() → compute ∂loss/∂weights for every parameter
        │
        ▼
OPTIMIZER STEP
  optimizer.step() → update all weights
        │
        ▼
REPEAT until Loss → ~1.5–2.5 (well-trained model)
```

---

## 14. Common Bugs and Misconceptions

### Bug 1: Not Flattening Before `cross_entropy`

```python
# WRONG — cross_entropy expects (N, C), not (B, T, C)
loss = F.cross_entropy(logits, targets)           # Shape error!

# CORRECT
loss = F.cross_entropy(logits.flatten(0,1), targets.flatten())
```

### Bug 2: Applying Softmax Before `cross_entropy`

```python
# WRONG — F.cross_entropy applies log_softmax internally
probas = torch.softmax(logits_flat, dim=-1)
loss = F.cross_entropy(probas, targets_flat)       # Double softmax!

# CORRECT — pass raw logits
loss = F.cross_entropy(logits_flat, targets_flat)
```

### Bug 3: Confusing `dim=-1` in argmax and softmax

```python
# dim=-1 = "along the last dimension" = along the vocab dimension
probas = torch.softmax(logits, dim=-1)     # ✓ normalizes over vocab
token_ids = torch.argmax(probas, dim=-1)   # ✓ picks best token in vocab
```

### Misconception 1: "Targets should only have 1 value per batch"

Wrong. Because the model makes N predictions per input sequence (one per token position), targets must have the same shape as inputs: `(batch_size, seq_len)`.

### Misconception 2: "1 word = 1 token"

Wrong. BPE creates subword tokens. "chocolate" might be 1 or 2 tokens. "ChatGPT" might be 3 tokens. Always use the tokenizer to know the actual token count.

### Misconception 3: "High loss = model is broken"

Not at initialization. Loss ≈ `log(vocab_size)` is expected for a randomly initialized model. It means the model is treating all tokens as equally likely. This is the correct starting point — training will reduce it.

---

## 15. Senior Engineer Mental Models

### Mental Model 1: Loss as an Information-Theoretic Measure

Cross-entropy loss measures **how many bits of information** the model needs beyond its predictions to communicate the truth. A perfect model needs 0 extra bits. A random model needs `log₂(V)` bits.

### Mental Model 2: The Loss Function is the Learning Signal

Without a loss function, there is no gradient, and without a gradient, there is no learning. **Every design choice in training (architecture, data, optimizer) exists to serve the loss function.** Understanding the loss = understanding what the model is being trained to do.

### Mental Model 3: Perplexity Measures "Branching Factor"

Think of the model's decision at each token as a tree with branching factor = Perplexity. A model with perplexity 2 has a binary choice at each step. A model with perplexity 50,000 has no idea what comes next.

### Mental Model 4: The Baseline Check

Before training anything, always verify:
```
Initial Loss ≈ log(vocab_size)    ← sanity check #1
Perplexity   ≈ vocab_size         ← sanity check #2
```

If not, your loss function, targets, or model output has a bug.

### Mental Model 5: Why Self-Supervised?

The LLM requires no human-annotated labels. Any text document automatically provides infinite (input, target) pairs via the shifting trick. This is why LLMs can be trained at massive scale — labels are free.

---

## 16. What's Coming Next (Lectures 27–30)

| Lecture | Topic | What You'll Learn |
|---------|-------|-------------------|
| 27 | Full Dataset Training | Load "The Verdict" book; tokenize the full text; create DataLoader with input/target pairs at scale |
| 28 | Train/Val Loss | Compute loss over entire dataset; plot training vs validation loss curves |
| 29 | Training Loop | `loss.backward()`, `optimizer.step()`, gradient clipping, learning rate scheduling |
| 30 | Pre-trained Weights | Load OpenAI GPT-2 weights into our architecture; verify perplexity improves immediately |

---

## 17. Cheat Sheet + Flashcards

### Formulas

```
Cross-Entropy Loss  = -(1/N) × Σ log(P_target_i)
Negative Log-Likelihood = same thing
Perplexity          = e^(Loss)
Baseline Loss       = log(vocab_size) = log(50257) ≈ 10.82
```

### Shape Reference

```
Input tensor:         (batch_size, seq_len)              e.g. (2, 3)
Target tensor:        (batch_size, seq_len)              e.g. (2, 3)
Logits:               (batch_size, seq_len, vocab_size)  e.g. (2, 3, 50257)
Probabilities:        (batch_size, seq_len, vocab_size)  e.g. (2, 3, 50257)
logits_flat:          (batch_size × seq_len, vocab_size) e.g. (6, 50257)
targets_flat:         (batch_size × seq_len,)            e.g. (6,)
Loss:                 scalar                             e.g. 10.79
Perplexity:           scalar                             e.g. 48725
```

### One-Line Glossary

| Term | Definition |
|---|---|
| **Logits** | Raw unnormalized model output before softmax |
| **Softmax** | Converts logits to probabilities summing to 1 |
| **log_softmax** | Numerically stable version of log(softmax(x)) |
| **argmax** | Returns the index of the maximum value |
| **NLL** | Negative Log-Likelihood — the loss we minimize |
| **Cross-Entropy** | NLL for categorical distributions; same as NLL for one-hot targets |
| **Perplexity** | e^loss — interpretable measure of model uncertainty |
| **Target** | True next token at each position (ground truth label) |
| **Causal LM** | Predicting next token using only past context (GPT style) |
| **BPE** | Byte Pair Encoding — sub-word tokenization algorithm used by tiktoken |

### Flashcards

```
Q: What is the shape of the logit tensor for batch=2, seq=3, vocab=50257?
A: (2, 3, 50257)

Q: What shape does F.cross_entropy expect for logits?
A: (N, C) where N = total samples, C = num classes (vocab size)

Q: How do you flatten logits from (2,3,50257) for cross_entropy?
A: logits.flatten(0, 1)  →  (6, 50257)

Q: What does F.cross_entropy do internally to the logits?
A: Applies log_softmax, indexes by target, takes mean, negates

Q: Should you apply softmax BEFORE passing logits to F.cross_entropy?
A: NO — F.cross_entropy applies it internally (log_softmax). Double-applying is a bug.

Q: What is the expected loss of a randomly initialized model? (vocab=50257)
A: log(50257) ≈ 10.82

Q: Formula for perplexity?
A: e^(loss)

Q: What is perplexity of 48,725 telling you?
A: The model behaves as if choosing next token uniformly from 48,725 options — essentially random.

Q: Why do targets have N values per batch instead of 1?
A: Because each of the N input token positions generates one prediction task (causal LM).

Q: Why take the log of probabilities in the loss function?
A: (1) Converts product to sum — numerically stable. (2) Penalizes low-confidence predictions heavily.

Q: Why negate the log-likelihood?
A: log(p) ∈ (-∞, 0) for p ∈ (0,1). Negating gives a positive loss we can minimize.

Q: What is the relationship between targets and inputs?
A: targets = inputs shifted left by 1 position (+ next token appended)

Q: What does dim=-1 mean in torch.argmax(probas, dim=-1)?
A: Look along the last dimension (vocab dimension) — find which token has max probability.

Q: Why is cross-entropy the right loss for LLMs (not MSE)?
A: Token IDs are categorical (not ordinal). Cross-entropy is the correct loss for probability distributions over discrete categories.

Q: What is the baseline perplexity sanity check for an untrained model?
A: Perplexity ≈ vocab_size (≈ 50,257 for GPT-2 vocab)
```

---

*Notes by: AI Engineering Study Series*
*Source: "Measuring the LLM Loss Function" — Build LLMs from Scratch, Lecture 26*
*Covers: 100% of lecture transcript content + senior engineer depth additions*
