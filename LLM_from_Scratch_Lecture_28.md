# Lecture 28 — Coding the Entire LLM Pre-training Loop
### Series: Build Large Language Models from Scratch | Stage 2: Training

---

## Table of Contents

1. [Where We Are — Series Checkpoint](#1-where-we-are--series-checkpoint)
2. [Dataset: The Verdict — Setup and Splits](#2-dataset-the-verdict--setup-and-splits)
3. [DataLoader: How Input-Target Pairs Are Built](#3-dataloader-how-input-target-pairs-are-built)
4. [Loss Function Recap — The Bridge from Lecture 26](#4-loss-function-recap--the-bridge-from-lecture-26)
5. [The Pre-training Loop — Conceptual Schematic](#5-the-pre-training-loop--conceptual-schematic)
6. [Parameter Count Deep Dive — 162 Million Parameters](#6-parameter-count-deep-dive--162-million-parameters)
7. [Backpropagation in One Line — loss.backward()](#7-backpropagation-in-one-line--lossbackward)
8. [The Optimizer — AdamW](#8-the-optimizer--adamw)
9. [Full Pre-training Loop Code — Line by Line](#9-full-pre-training-loop-code--line-by-line)
10. [Evaluation: evaluate_model Function](#10-evaluation-evaluate_model-function)
11. [Monitoring Output: generate_and_print_sample](#11-monitoring-output-generate_and_print_sample)
12. [Training Results — What Actually Happened](#12-training-results--what-actually-happened)
13. [Overfitting — Diagnosis and Root Cause](#13-overfitting--diagnosis-and-root-cause)
14. [Loss Curves — How to Read Them](#14-loss-curves--how-to-read-them)
15. [What Real-World Training Looks Like vs This Lecture](#15-what-real-world-training-looks-like-vs-this-lecture)
16. [Common Bugs and Misconceptions](#16-common-bugs-and-misconceptions)
17. [Senior Engineer Mental Models](#17-senior-engineer-mental-models)
18. [What's Coming Next — Lecture 29: Decoding Strategies](#18-whats-coming-next--lecture-29-decoding-strategies)
19. [Cheat Sheet + Flashcards](#19-cheat-sheet--flashcards)

---

## 1. Where We Are — Series Checkpoint

```
STAGE 1 — Architecture (COMPLETE)
  └── Data Prep, Tokenization, Sampling
  └── Attention Mechanism (Causal Multi-Head Self-Attention)
  └── Full GPT Architecture (Transformer blocks, LayerNorm, Output head)

STAGE 2 — Training (CURRENT)
  └── Lecture 26: Loss Function (Cross-Entropy, Perplexity)
  └── Lecture 27: Dataset loading, Train/Val split, DataLoader, calc_loss functions
  └── Lecture 28: Full Pre-training Loop ← YOU ARE HERE
  └── Lecture 29: Decoding Strategies (temperature, top-k)
  └── Lecture 30: Load Pre-trained OpenAI Weights
```

### The Accumulated Knowledge Required to Get Here

The instructor makes an important point: the 5-minute training loop summary in this lecture required **20–25 prior lectures** of groundwork:

```
Tokenization (tiktoken BPE)
    ↓
Positional Embeddings
    ↓
Multi-Head Causal Self-Attention
    ↓
Dropout, LayerNorm, Feed-Forward Networks
    ↓
Full GPT Architecture (forward pass)
    ↓
DataLoader (input-target pair construction)
    ↓
Cross-Entropy Loss Function
    ↓
THIS LECTURE: Backward pass + Parameter Updates
```

Every one of those layers was necessary. Without a differentiable loss function, `loss.backward()` is meaningless. Without the architecture, there is no loss. This is why ML engineering is sequential — you cannot skip layers.

---

## 2. Dataset: The Verdict — Setup and Splits

### About the Dataset

| Property | Value |
|---|---|
| Book title | "The Verdict" |
| Written | 1906 (public domain) |
| Characters | ~20,000 |
| Tokens (approx.) | ~5,000–5,500 |
| Source | Open source, freely downloadable |
| License | Public domain |

> **Why so small?** The goal is educational — you can train on a laptop in under 10 minutes and still observe real training dynamics (loss decreasing, overfitting, text coherence improving). Real LLM datasets are billions of tokens.

### Train/Validation Split

```python
train_ratio = 0.9   # 90% training, 10% validation

split_idx = int(train_ratio * len(raw_text))

train_data = raw_text[:split_idx]   # first 90% ≈ 4,500–5,000 tokens
val_data   = raw_text[split_idx:]   # last  10% ≈  500  tokens
```

### Why Split at All?

- **Training loss** measures how well the model fits the data it's trained on — can always be driven toward 0 by memorization
- **Validation loss** measures generalization — how well the model performs on data it has **never seen**
- If training loss decreases but validation loss stagnates or increases → **overfitting**
- The split must happen **before** any training — if the model sees validation data, the evaluation is invalid

---

## 3. DataLoader: How Input-Target Pairs Are Built

### The Core Concept

LLMs have **no external labels**. Labels come directly from the text itself via the **next-token prediction** (causal language modeling) objective.

```
Raw text:  "I had always thought Jack was..."
              ↓  tokenize
Token IDs: [40, 550, 2037, 3579, 6628, 373, ...]
              ↓  sliding window with stride
Input:  [40,  550, 2037, 3579]   →   "I had always thought"
Target: [550, 2037, 3579, 6628]  →   "had always thought Jack"
```

Targets = Inputs shifted left by 1 position. This is the self-supervised signal.

### Stride and Context Size

```
context_size = 4   (simplified; real code uses 256)
stride       = 4   (no overlap between samples)

Dataset: [t0, t1, t2, t3, t4, t5, t6, t7, t8, ...]

Sample 1:  Input  = [t0, t1, t2, t3]
           Target = [t1, t2, t3, t4]

Sample 2:  Input  = [t4, t5, t6, t7]   ← stride=4, no overlap
           Target = [t5, t6, t7, t8]

Sample 3:  Input  = [t8, t9, t10, t11]
           ...
```

**When stride = context_size:** No overlap. Each token appears in exactly one training sample. Efficient but each sample provides less context diversity.

**When stride < context_size:** Overlapping windows. More training samples but higher redundancy. Often used for better learning signal density.

### What One Batch Looks Like

```
Input Batch (batch_size=2, context_size=4):
  [[40,   550,  2037,  3579],   ← Sample 1: "I had always thought"
   [6628,  373,  1234,  5678]]  ← Sample 2: next 4 tokens

Target Batch (batch_size=2, context_size=4):
  [[550,  2037,  3579, 6628],   ← Sample 1 targets (shifted by 1)
   [373,  1234,  5678,  910]]   ← Sample 2 targets (shifted by 1)
```

> In the actual code, context_size = 256, so each sample has 256 tokens and each batch contains `batch_size × 256` tokens.

### Token Count Per Batch

```
tokens_per_batch = batch_size × context_size
                 = 2 × 256
                 = 512 tokens processed per forward pass
```

This is why the lecture tracks "tokens seen" — it's the standard unit of training progress in LLM research.

---

## 4. Loss Function Recap — The Bridge from Lecture 26

Quick recap of what's already implemented before the training loop:

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    """Compute cross-entropy loss for one batch."""
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)                         # (B, T, V)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),    # (B*T, V)
        target_batch.flatten()   # (B*T,)
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Compute average loss over an entire DataLoader."""
    total_loss = 0.0
    if num_batches is None:
        num_batches = len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
```

These two functions are the foundation. Everything in the training loop calls these.

---

## 5. The Pre-training Loop — Conceptual Schematic

```
FOR each Epoch (1 to num_epochs):
│
├── FOR each Batch in training_loader:
│   │
│   ├── [1] FORWARD PASS
│   │       input_batch → GPT model → logits → cross_entropy → loss (scalar)
│   │
│   ├── [2] BACKWARD PASS  ← THE CRITICAL STEP
│   │       loss.backward()
│   │       Computes ∂loss/∂θ for ALL 162M parameters simultaneously
│   │
│   ├── [3] PARAMETER UPDATE
│   │       optimizer.step()
│   │       θ_new = θ_old - lr × ∂loss/∂θ
│   │
│   ├── [4] RESET GRADIENTS
│   │       optimizer.zero_grad()
│   │       (Must happen before next batch — else gradients accumulate)
│   │
│   └── [5] EVALUATION (every eval_freq batches)
│           evaluate_model() → print train_loss, val_loss
│
└── [6] TEXT SAMPLE (after each Epoch)
        generate_and_print_sample() → show what LLM generates right now
```

### Why This Order Matters

The order `forward → backward → step → zero_grad` is non-negotiable:

1. **Forward first** — you need a loss before you can differentiate it
2. **Backward second** — computes all gradients in one pass
3. **Step third** — uses the just-computed gradients to update weights
4. **Zero grad after** — clears the gradient buffer so the next batch starts fresh

> **Critical bug**: If you call `optimizer.zero_grad()` BEFORE `optimizer.step()`, you erase the gradients before using them. The model won't learn.

> **Another critical bug**: If you forget `optimizer.zero_grad()` entirely, gradients from previous batches accumulate, and your effective gradient is the sum of all past batches — completely wrong.

---

## 6. Parameter Count Deep Dive — 162 Million Parameters

Understanding where 162M parameters come from is important for understanding what `loss.backward()` is actually differentiating through.

### Breakdown by Component

#### A. Embedding Layer

```
Token Embeddings:     vocab_size × emb_dim  =  50257 × 768  =  38,597,376
Positional Embeddings: context_len × emb_dim =   256 × 768  =     196,608
                                                         Total: ~38.4 million
```

> These are **learned** parameters — the model learns what each token "means" as a vector, and what each position "means."

#### B. Each Transformer Block

One transformer block contains:

```
Multi-Head Attention:
  Q weight matrix:    768 × 768  =  589,824
  K weight matrix:    768 × 768  =  589,824
  V weight matrix:    768 × 768  =  589,824
  Output projection:  768 × 768  =  589,824
  Subtotal MHA:                  = 2,359,296  ≈ 2.36M

Feed-Forward Network (expansion-contraction):
  Layer 1 (expand):   768 → 4×768 = 768 × 3072  =  2,359,296
  Layer 2 (contract): 4×768 → 768 = 3072 × 768  =  2,359,296
  Subtotal FFN:                   = 4,718,592    ≈ 4.72M

LayerNorm parameters (γ, β):      small, ~3,072 per block

Total per Transformer Block:  ≈ 2.36M + 4.72M = 7.08M
```

**Why 4× expansion in FFN?**
This is a standard design from the original "Attention is All You Need" paper. The expansion creates a higher-dimensional "thinking space" for each token representation before projecting back down.

#### C. 12 Transformer Blocks

```
12 × 7.08M = 84.96M ≈ 85.2M parameters
```

#### D. Final Output Head (Linear layer)

```
emb_dim × vocab_size = 768 × 50257 = 38,597,376 ≈ 38.4M
```

This projects from the 768-dimensional hidden state back to vocabulary-sized logits.

#### E. Total

```
Embeddings:         38.4M
Transformer blocks: 85.2M
Output head:        38.4M
─────────────────────────
TOTAL:             162.0M parameters
```

### Why GPT-2 Has Only 124M (Weight Tying)

GPT-2 uses **weight tying**: the token embedding matrix and the output head matrix are the **same** matrix (shared weights).

```
Without weight tying:  38.4M (embeddings) + 38.4M (output head) = 76.8M
With weight tying:     38.4M (shared)                           = 38.4M
Savings:               38.4M parameters
```

```
Our model:  162M
GPT-2:      162M - 38.4M ≈ 124M  ← weight tying saves 38.4M parameters
```

Weight tying also tends to improve performance because the model must learn token representations that work both as inputs **and** as classifiers.

---

## 7. Backpropagation in One Line — loss.backward()

### What loss.backward() Actually Does

```python
loss.backward()
```

This single line:
1. Traverses the **computational graph** of the entire forward pass in reverse
2. Applies the **chain rule** of calculus at every operation node
3. Accumulates `∂loss/∂θ` in the `.grad` attribute of **every parameter tensor** in the model
4. Does this for all **162 million parameters** simultaneously

### The Computational Graph

Every PyTorch operation on tensors with `requires_grad=True` creates a node in a directed acyclic graph (DAG). The forward pass builds this graph automatically.

```
input_tokens
    │
    ▼ embedding lookup
token_embeddings + pos_embeddings
    │
    ▼ dropout
    │
    ▼ transformer_block_1
    │   ├── layernorm_1
    │   ├── multi_head_attention (Q, K, V matmul, softmax, output proj)
    │   ├── dropout
    │   ├── residual add
    │   ├── layernorm_2
    │   ├── feed_forward (linear → GELU → linear)
    │   ├── dropout
    │   └── residual add
    │
    ▼ ... (×12 transformer blocks)
    │
    ▼ final layernorm
    │
    ▼ output_head (linear projection)
    │
    ▼ logits  →  F.cross_entropy(logits_flat, targets_flat)
                         │
                         ▼
                       LOSS (scalar)
                         │
                         ▼  loss.backward()
         ∂loss/∂(every parameter in the graph above)
```

### Why This is "Pretty Awesome"

Before PyTorch and TensorFlow, computing gradients for a 162M parameter model by hand would require:
- Writing symbolic derivatives for every layer type
- Managing numerical precision across millions of operations
- Weeks of engineering per new architecture

Now: one line. PyTorch's automatic differentiation (autograd) handles all of it.

### The Chain Rule (Why It Works)

For a composition of functions `Loss = f(g(h(x)))`, the chain rule gives:

```
∂Loss/∂x = (∂Loss/∂f) × (∂f/∂g) × (∂g/∂h) × (∂h/∂x)
```

PyTorch traverses the computational graph backward and multiplies these partial derivatives together, layer by layer. This is exactly **backpropagation** — just automated.

---

## 8. The Optimizer — AdamW

### Why Not Vanilla Gradient Descent?

Vanilla gradient descent (SGD):
```
θ_new = θ_old - lr × ∂loss/∂θ
```

Problems with vanilla SGD for LLMs:
- Same learning rate for all parameters — some need large steps, others tiny steps
- Sensitive to gradient scale differences across layers
- Prone to getting stuck in local minima or saddle points
- Slow convergence for deep networks

### Adam (Adaptive Moment Estimation)

Adam maintains two running averages for each parameter:

```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t        ← 1st moment (mean of gradients)
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²       ← 2nd moment (variance of gradients)

θ_new = θ_old - lr × m̂_t / (√v̂_t + ε)
```

Where:
- `g_t` = gradient at step t
- `β₁ = 0.9` (default) — how much to weight past gradient directions
- `β₂ = 0.999` (default) — how much to weight past gradient magnitudes
- `ε = 1e-8` — numerical stability constant
- `m̂_t, v̂_t` = bias-corrected estimates

**Effect:** Each parameter gets its own adaptive learning rate. Parameters with consistently large gradients get smaller effective steps; parameters with small, noisy gradients get larger relative steps.

### AdamW (Adam with Weight Decay)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,          # learning rate (hyperparameter to tune)
    weight_decay=0.1    # L2 regularization strength (hyperparameter to tune)
)
```

**Weight decay** adds a regularization term that penalizes large parameter values:

```
Loss_total = Loss_CE + λ × Σ(θ²)
```

This prevents individual parameters from growing too large, which helps generalization and prevents overfitting. AdamW applies weight decay **correctly** (directly to weights, not through the gradient), which standard Adam does not.

**Why AdamW for LLMs?**
- Adaptive learning rates help navigate the complex loss landscape of deep transformers
- Weight decay improves generalization — critical when training data is limited
- Adam avoids many local minima that trap SGD
- Industry standard: GPT-2, GPT-3, LLaMA, and essentially all modern LLMs use Adam or AdamW

### Key Hyperparameters to Tune

| Hyperparameter | Value Used | Typical Range | Effect |
|---|---|---|---|
| `lr` (learning rate) | 0.0004 | 1e-5 to 1e-3 | Step size; too high → divergence, too low → slow |
| `weight_decay` | 0.1 | 0.0 to 0.1 | Regularization strength; helps prevent overfitting |
| `num_epochs` | 10 | 1–100+ | More epochs → more training but more overfitting risk |
| `batch_size` | 2 | 8–4096 | Larger = more stable gradients, more memory |
| `context_size` | 256 | 128–8192+ | Larger = more context, more compute |

---

## 9. Full Pre-training Loop Code — Line by Line

```python
import torch
import time

def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,          # evaluate every N batches
    eval_iter,          # how many batches to use when evaluating
    start_context,      # initial text for sample generation
    tokenizer
):
    # Track losses and token counts over training
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # ── OUTER LOOP: iterate over epochs ──────────────────────────────────
    for epoch in range(num_epochs):

        model.train()   # enable dropout, gradient tracking

        # ── INNER LOOP: iterate over batches ─────────────────────────────
        for input_batch, target_batch in train_loader:

            # [1] RESET GRADIENTS from previous batch
            optimizer.zero_grad()

            # [2] FORWARD PASS: compute loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # [3] BACKWARD PASS: compute all gradients in one line
            loss.backward()

            # [4] PARAMETER UPDATE: apply gradient descent step
            optimizer.step()

            # Track how many tokens have been processed
            tokens_seen += input_batch.numel()
            global_step += 1

            # [5] EVALUATION (every eval_freq batches)
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        # [6] GENERATE TEXT SAMPLE after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
```

### Annotated Line-by-Line Explanation

```python
model.train()
```
Puts the model in training mode. This enables:
- **Dropout**: randomly zeros activations (regularization)
- **Gradient computation**: `requires_grad=True` is active
Without this, calling `.eval()` previously would have disabled dropout.

```python
optimizer.zero_grad()
```
Clears the `.grad` buffer of every parameter. **Must be called before the backward pass**, not after. If you skip this, gradients from the previous batch accumulate — the effective gradient becomes the sum of all previous gradients, causing incorrect updates.

```python
loss = calc_loss_batch(input_batch, target_batch, model, device)
```
Full forward pass: input → GPT model → logits → cross_entropy → scalar loss.

```python
loss.backward()
```
Full backward pass. PyTorch traverses the computational graph in reverse, applying the chain rule at every node. After this line, every parameter `p` in the model has `p.grad` populated with `∂loss/∂p`.

```python
optimizer.step()
```
Uses the `.grad` values to update all parameters according to the AdamW update rule. After this, the model weights have changed — hopefully in a direction that reduces future loss.

```python
tokens_seen += input_batch.numel()
```
`numel()` returns the total number of elements in the tensor: `batch_size × context_size`. This is the token throughput counter.

```python
if global_step % eval_freq == 0:
```
Every `eval_freq` (=5) batches, run evaluation. We don't evaluate every step because evaluation is expensive (requires a full pass over the dataset) and slows training.

### The Main Setup Block

```python
# Model configuration
GPT_CONFIG_124M = {
    "vocab_size":     50257,
    "context_length": 256,    # reduced from 1024 for local training
    "emb_dim":        768,
    "n_heads":        12,
    "n_layers":       12,
    "drop_rate":      0.1,
    "qkv_bias":       False
}

# Instantiate model
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,
    weight_decay=0.1
)

# Training run
num_epochs = 10
eval_freq  = 5       # print losses every 5 batches
eval_iter  = 5       # use 5 batches to estimate loss during evaluation
start_context = "Every effort moves you"

start_time = time.time()
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader,
    optimizer, device,
    num_epochs, eval_freq, eval_iter,
    start_context, tokenizer
)
end_time = time.time()

print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")
```

---

## 10. Evaluation: evaluate_model Function

```python
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Compute average loss on train and validation sets.
    Disables dropout and gradient computation during evaluation.
    """
    model.eval()                    # disable dropout
    with torch.no_grad():           # disable gradient computation (saves memory)
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()                   # re-enable training mode after evaluation
    return train_loss, val_loss
```

### Why `model.eval()` and `torch.no_grad()`?

**`model.eval()`:**
- Disables dropout (so evaluation is deterministic and not noisier than necessary)
- Puts BatchNorm layers (if any) into inference mode
- Does NOT disable gradient computation by itself

**`torch.no_grad()`:**
- Tells PyTorch not to build the computational graph during this forward pass
- Saves memory (~50% memory reduction for the forward pass)
- Speeds up evaluation (no graph construction overhead)
- You don't need gradients just to evaluate loss

**`model.train()` at the end:**
- Critical — must switch back to training mode after evaluation
- Forgetting this is a subtle bug: dropout stays disabled for the rest of training → no regularization → model overfits faster

### `eval_iter` Parameter

```python
# eval_iter = 5 means: use only 5 batches to estimate loss
# (not the entire dataset, which would be slow)

train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
```

This is an approximation — you don't evaluate the exact loss on the entire training set at every eval step; you sample 5 batches as an estimate. Faster, but noisier. For final reporting you'd use `num_batches=None` (full dataset).

---

## 11. Monitoring Output: generate_and_print_sample

```python
def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    After each epoch, generate 50 tokens starting from start_context
    and print them. Purely for qualitative monitoring.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]  # get context length from model
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
```

### Purpose

This function answers: **"Right now, after this many epochs, what does the model actually say?"**

It takes `"Every effort moves you"` as the seed and generates 50 new tokens. This is purely **qualitative** monitoring — there's no loss calculation here. You're just reading the output and asking: "Does this look better than last epoch?"

### Why 50 Tokens?

Short enough to read quickly, long enough to see whether the model is generating coherent structure (punctuation, word spacing, sentence-like flow) vs pure noise (comma comma comma).

---

## 12. Training Results — What Actually Happened

### Loss Numbers (from the Lecture Run)

Training was done on a MacBook Air. Total time: **6.6 minutes** for 10 epochs over a ~5,000 token dataset with 162M parameters.

```
Training Loss:
  Start (Epoch 1, Step 0):    9.781
  End   (Epoch 10, last step): 0.391   ← dramatic reduction

Validation Loss:
  Start:   9.933
  End:     ~6.37                        ← much higher than train loss → overfitting
```

### Qualitative Output Per Epoch

This is one of the most instructive parts of the lecture — watching what the model generates as training progresses:

```
Epoch 1:  ", , , , , , , , ,"
          → Model has learned nothing. Default filler tokens.

Epoch 2:  "U , and and and and and"
          → Starting to pick up high-frequency tokens from the corpus.

Epoch 3:  "and I had been"
          → Beginning to form token sequences that appear in training data.

Epoch 4:  "you know the I had the donkey and I had the"
          → Repetitive, but actual words from the dataset appearing.

Epoch 7:  "every effort moves you know was one of the picture for nothing
           I told Mrs"
          → Words from the dataset, some sentence structure emerging.

Epoch 9:  "every effort moves you ? yes quite insensible to the irony
           she wanted him a Vindicated and by me"
          → Direct memorization! "quite insensible to the irony" is a
            verbatim quote from the training data.

Epoch 10: "every effort moves you was one of the xmc laid down across the
           SE and silver of an exquisitely appointed luncheon table I had
           run over from Monte Carlo and Mrs J"
          → Grammatically coherent, rich vocabulary, but clearly
            recycling training text verbatim.
```

### Interpretation of the Progression

| Phase | Epochs | Behavior | Explanation |
|---|---|---|---|
| Random | 1–2 | Commas, and's | Model is near-random; defaults to high-frequency tokens |
| Proto-learning | 3–4 | Short phrases | Starting to capture common bigrams/trigrams |
| Pattern capture | 5–7 | Coherent fragments | Capturing syntactic patterns and vocabulary |
| Memorization | 8–10 | Verbatim excerpts | Overfitting — model is memorizing not generalizing |

---

## 13. Overfitting — Diagnosis and Root Cause

### Definition

**Overfitting** occurs when a model performs well on training data but fails to generalize to unseen data.

### Evidence in This Lecture

```
Training loss: 9.781 → 0.391    (reduced by 96%)
Validation loss: 9.933 → 6.37  (reduced by only 36%)

Gap = val_loss - train_loss = 6.37 - 0.39 = 5.98   ← large gap = overfit
```

The model memorized "quite insensible to the irony she wanted him indicated and by me" — a verbatim copy from the training set. This is the clearest possible demonstration of memorization.

### Root Cause: Dataset Size

```
Training data: ~5,000 tokens
Model parameters: 162,000,000

Ratio of parameters to training tokens = 162M / 5K = 32,400 parameters per token
```

This ratio is astronomically high. The model has 32,400 parameters for every single training token. It can trivially memorize the entire dataset because it has vastly more capacity than the dataset has information.

For comparison:
```
GPT-3:   175B parameters / ~300B training tokens ≈ 0.58 parameters per token
```

GPT-3 has **less than 1 parameter per training token** — the opposite extreme. This is why GPT-3 generalizes and our toy model memorizes.

### Strategies to Reduce Overfitting (Not Covered Until Later)

1. **More training data** — the real fix; more data always helps most
2. **Stronger regularization** — increase `weight_decay`, increase `drop_rate`
3. **Fewer epochs** — stop before the model memorizes (early stopping)
4. **Smaller model** — reduce `n_layers`, `emb_dim` to reduce capacity
5. **Decoding strategies** — temperature, top-k sampling (next lecture) help with output diversity

### Why Overfitting is Acceptable Here

The instructor correctly notes: overfitting on a tiny dataset is **expected and acceptable** for this educational exercise. The goal is to verify the training loop works (loss decreases), understand overfitting conceptually, and set up the infrastructure for future improvements.

---

## 14. Loss Curves — How to Read Them

```python
import matplotlib.pyplot as plt

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot losses vs epochs (top x-axis)
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="--", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Second x-axis: tokens seen
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # invisible, just sets scale
    ax2.set_xlabel("Tokens seen")

    plt.tight_layout()
    plt.savefig("loss_curves.pdf")
    plt.show()
```

### What the Curves Tell You

```
Loss
  ↑
10│\
  │ \  ← Train loss: rapidly drops in early epochs
  │  \
  │   \______
6 │          -----___  ← Val loss: drops then plateaus
  │                  ‾‾‾‾‾‾‾‾  (stagnation = overfitting)
0 │──────────────────────────────→ Epochs / Tokens Seen
  1    2    3    4    5    6    7    8    9    10
              ↑
          Divergence point (~Epoch 2)
          After this, train and val losses separate
          → Model starts memorizing instead of generalizing
```

### Key Diagnostic Patterns

| Pattern | Meaning | Action |
|---|---|---|
| Both losses high and decreasing | Model is still learning, underfitting | Keep training |
| Train loss low, val loss high and diverging | Overfitting | More data, more regularization, fewer epochs |
| Both losses plateau at the same value | Model has converged (good) or is stuck (bad) | Try different lr, check data |
| Val loss lower than train loss | Data leakage in split, or evaluation noise | Check your train/val split |
| Loss oscillates wildly | Learning rate too high | Reduce lr |
| Loss barely moves | Learning rate too low, or gradient vanishing | Increase lr, check architecture |

---

## 15. What Real-World Training Looks Like vs This Lecture

### Key Differences

| Aspect | This Lecture | Real LLM Training |
|---|---|---|
| Dataset size | ~5,000 tokens | 300B–15T tokens |
| Epochs | 10 (over same data) | Usually 1 epoch (data seen once) |
| Batch size | 2 | 512–4096 |
| Context length | 256 | 2048–128k+ |
| Hardware | MacBook Air (CPU/MPS) | Thousands of A100/H100 GPUs |
| Training time | 6.6 minutes | Weeks to months |
| Parameters | 162M | 7B–405B+ |
| Overfitting | Yes (tiny data) | Rare (data >> model capacity) |
| Optimizer | AdamW (same) | AdamW (same) |
| Loss function | Cross-entropy (same) | Cross-entropy (same) |

> The training loop logic is **identical**. What scales is the hardware, data, and model size — not the algorithm.

### Why Train for Only 1 Epoch on Large Datasets?

When your dataset has 300 billion tokens and your model has 70 billion parameters, you have 4.3 tokens per parameter — a reasonable ratio. Training for more than 1 epoch would mean showing the model the same data twice, which risks:
- Memorization of specific text (just like our toy model)
- Wasted compute (better to use a larger dataset for a 2nd epoch's worth of compute)

Modern LLMs use essentially every internet token once, then the training is done.

---

## 16. Common Bugs and Misconceptions

### Bug 1: Wrong order of zero_grad, backward, step

```python
# WRONG — loses gradients before using them
loss.backward()
optimizer.zero_grad()   # ← erases gradients!
optimizer.step()        # ← no gradients to use

# CORRECT
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Bug 2: Forgetting to switch back to train mode

```python
# During evaluate_model:
model.eval()
# ... compute val loss ...
# WRONG: forgot to switch back
# model.train()  ← missing!

# Now the training loop continues with dropout DISABLED
# Model trains without regularization → overfits faster
```

### Bug 3: Applying softmax before cross_entropy

```python
# WRONG: F.cross_entropy applies log_softmax internally
probs = torch.softmax(logits, dim=-1)
loss = F.cross_entropy(probs, targets)   # double-softmax → wrong loss

# CORRECT: pass raw logits
loss = F.cross_entropy(logits.flatten(0,1), targets.flatten())
```

### Bug 4: Not detaching loss before logging

```python
# WRONG: keeps computational graph in memory → memory leak
train_losses.append(loss)

# CORRECT: detach from graph, get scalar
train_losses.append(loss.item())   # .item() detaches and converts to Python float
```

### Bug 5: Evaluating with gradients enabled

```python
# WRONG: wastes memory building graph during evaluation
train_loss = calc_loss_loader(train_loader, model, device)

# CORRECT: disable gradient computation during evaluation
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
```

### Misconception 1: "Training loss should reach 0"

Not a goal. Training loss reaching 0 means the model has perfectly memorized the training data. In practice, you want a **low but nonzero** training loss with a **similar validation loss** — that indicates learning, not memorization.

### Misconception 2: "More epochs always helps"

After a certain point, more epochs hurt generalization. The model begins memorizing training examples. The optimal epoch count depends on dataset size, model capacity, and regularization.

### Misconception 3: "Validation loss should always decrease"

Validation loss can plateau or even increase after the model starts overfitting. This is a signal to stop training, not a bug.

---

## 17. Senior Engineer Mental Models

### Mental Model 1: The Training Loop is the Same Everywhere

The four-line core of training:
```python
optimizer.zero_grad()
loss = criterion(model(inputs), targets)
loss.backward()
optimizer.step()
```
is identical whether you are training a linear regression, a CNN image classifier, or a 400B parameter LLM. The scale changes. The algorithm does not.

### Mental Model 2: loss.backward() is Magic — But Understandable Magic

`loss.backward()` implements reverse-mode automatic differentiation. The "magic" is that PyTorch recorded every operation in the forward pass as a node in a graph, and now it traverses that graph in reverse, applying the chain rule at each node. It's not actually magic — it's a very elegant software engineering solution to a calculus problem.

### Mental Model 3: Train/Val Loss Gap = Generalization Gap

The gap between training and validation loss is the most important number to watch during training. A small gap means the model generalizes. A large gap means it memorizes. This is the core diagnostic signal in all of supervised learning.

### Mental Model 4: Tokens Seen as the Universal Training Currency

In LLM research, the x-axis of training curves is always "tokens seen" (or "training FLOPs"), never "epochs." This is because:
- Epochs are dataset-relative (meaningless across different dataset sizes)
- Tokens seen is absolute — you can compare runs across different datasets and batch sizes
- Research papers report training cost in tokens seen

### Mental Model 5: Small Dataset + Large Model = Guaranteed Overfitting

If `model_parameters >> dataset_tokens`, the model will memorize. This is fundamental capacity theory. The only fixes are: get more data, shrink the model, or add regularization (dropout, weight decay, early stopping).

### Mental Model 6: The Training Time Sanity Check

```
~6 minutes for 10 epochs × 5,000 tokens × 162M parameters on a MacBook Air
```

This gives you a baseline for estimating training time for larger experiments. Scaling laws are roughly:
```
Training time ∝ (num_params × num_tokens) / (hardware_flops)
```

If you 10× the dataset, expect ~10× the training time on the same hardware.

---

## 18. What's Coming Next — Lecture 29: Decoding Strategies

The training loop is now complete, but the model currently uses **greedy decoding** (always pick the argmax token). This causes:
- Repetitive output (model gets stuck in loops)
- No creativity or diversity
- Over-representation of high-frequency tokens

Lecture 29 introduces:

| Strategy | Mechanism | Effect |
|---|---|---|
| **Temperature scaling** | Divide logits by T before softmax | T<1: more focused/deterministic; T>1: more random/creative |
| **Top-k sampling** | Zero out all but top-k logits, then sample | Prevents low-probability tokens; controls randomness |
| **Nucleus (top-p) sampling** | Keep smallest set of tokens whose probability sum ≥ p | Dynamic top-k based on probability mass |

These strategies are **inference-time only** — the model itself doesn't change, only how you sample from its output distribution.

---

## 19. Cheat Sheet + Flashcards

### Key Numbers from This Lecture

```
Dataset:           "The Verdict" — ~5,000 tokens, ~20,000 characters
Train/Val split:   90% / 10%
Batch size:        2 samples per batch
Context size:      256 tokens per sample
Tokens per batch:  2 × 256 = 512
Total parameters:  162M (vs GPT-2's 124M due to weight tying)
Epochs:            10
Eval frequency:    every 5 batches
Training time:     6.6 minutes on MacBook Air
Train loss:        9.781 → 0.391
Val loss:          9.933 → 6.37 (stagnated → overfitting)
```

### Parameter Count Reference

```
Token embeddings:     50257 × 768 = 38.6M
Positional embeddings:  256 × 768 =  0.2M
12 Transformer blocks:              85.2M   (7.08M × 12)
  └─ Each block MHA:     2.36M
  └─ Each block FFN:     4.72M
Output head:          768 × 50257 = 38.6M
────────────────────────────────────────
TOTAL:                             162.0M
GPT-2 (weight tying):              124.0M  (saves 38.4M)
```

### One-Line Glossary

| Term | Definition |
|---|---|
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | A subset of training samples processed together in one forward/backward pass |
| **Forward pass** | Computing model output and loss from input |
| **Backward pass** | Computing gradients of loss w.r.t. all parameters via backpropagation |
| **Gradient** | ∂loss/∂θ — direction and magnitude of steepest loss increase |
| **optimizer.step()** | Updates all parameters using their computed gradients |
| **optimizer.zero_grad()** | Clears gradient buffers before computing new gradients |
| **AdamW** | Adam optimizer + correct weight decay; standard for LLM training |
| **Weight tying** | Sharing embedding and output head parameters to reduce parameter count |
| **Overfitting** | Model performs well on training data but poorly on validation data |
| **Generalization gap** | val_loss - train_loss; the core diagnostic metric |
| **Tokens seen** | Total number of tokens processed during training; universal training currency |
| **model.eval()** | Disables dropout; switches model to inference mode |
| **model.train()** | Enables dropout; switches model back to training mode |
| **torch.no_grad()** | Disables gradient computation during evaluation; saves memory |
| **loss.item()** | Extracts scalar value from loss tensor; detaches from computational graph |

### Flashcards

```
Q: What are the four core lines of any PyTorch training loop?
A: optimizer.zero_grad() / loss = criterion(model(x), y) / loss.backward() / optimizer.step()

Q: What does loss.backward() actually compute?
A: ∂loss/∂θ for every learnable parameter θ in the model, using reverse-mode automatic differentiation.

Q: Why must optimizer.zero_grad() be called before loss.backward()?
A: PyTorch accumulates gradients by default. Without zeroing, gradients from previous batches add to current batch gradients → wrong update.

Q: What is weight tying and how does it reduce GPT-2's parameter count?
A: Sharing the token embedding and output head weight matrix. Saves 38.4M parameters (768×50257). GPT-2: 162M - 38.4M = 124M.

Q: Why does our model overfit on "The Verdict"?
A: 162M parameters vs ~5,000 training tokens = 32,400 parameters per token. Model has far more capacity than data.

Q: What is the difference between model.eval() and torch.no_grad()?
A: model.eval() disables dropout (deterministic mode). torch.no_grad() disables gradient computation (saves memory). Both needed during evaluation.

Q: What does "tokens seen" measure and why is it preferred over "epochs"?
A: Total tokens processed during training. Preferred because it's absolute and comparable across different dataset sizes and batch sizes.

Q: Why does AdamW have weight_decay and what does it do?
A: Adds L2 regularization penalty (λ×Σθ²) to the loss, penalizing large weights and preventing overfitting. AdamW applies it correctly, decoupled from the gradient.

Q: What is eval_freq and why not evaluate every batch?
A: Number of batches between evaluations. Evaluation is expensive (full dataset pass). Every-batch evaluation would slow training significantly.

Q: What does a large gap between train_loss and val_loss indicate?
A: Overfitting — model has learned to memorize training data rather than generalize.

Q: What is the baseline loss for an untrained model with vocab_size=50257?
A: log(50257) ≈ 10.82. The lecture reports 9.781 (training) and 9.933 (validation) at step 0 — consistent with this.

Q: In real LLM training, why is typically only 1 epoch used?
A: Dataset is so large (hundreds of billions of tokens) that 1 pass provides enough signal. Multiple epochs risk memorization and waste compute that could train on more diverse data.

Q: What is the computational complexity difference between forward and backward pass?
A: Backward pass is typically 2–3× the compute of the forward pass (it traverses the same graph but computes and stores gradients at each node).

Q: What happens to model output quality as training progresses in this lecture?
A: Epoch 1: random commas. Epochs 2-4: high-frequency words. Epochs 5-7: coherent fragments. Epochs 8-10: grammatically correct but verbatim memorized text from training data.
```

---

*Notes by: AI Engineering Study Series*
*Source: "Coding the Entire LLM Pre-training Loop" — Build LLMs from Scratch, Lecture 28*
*Covers: 100% of lecture transcript content + senior engineer depth additions*
*Companion to: LLM_from_Scratch_Lecture_26.md (Loss Functions)*
