# GPT Text Generation — From Output Tensor to Predicted Words
### Lecture Notes | Build LLMs from Scratch Series — Lecture 25

> **Context:** This is the final lecture of the GPT Architecture module. All previous lectures built up the full GPT model — tokenization, embeddings, positional encoding, multi-head attention, transformer blocks, layer norm, and the output head. You now have a working `GPTModel` class that returns a logits tensor. This lecture answers one question: **"How do we turn that logits tensor into actual generated text?"**

---

## What You Already Have — Quick Anchor

By the end of the previous lecture, your `GPTModel`:
- Accepts input token IDs of shape `[batch_size, num_tokens]`
- Internally runs: token embeddings → positional embeddings → dropout → N transformer blocks → layer norm → linear output head
- Returns a **logits tensor** of shape `[batch_size, num_tokens, vocab_size]` where `vocab_size = 50,257`

Every single value in that output tensor is a raw, unnormalized score (a logit) representing how likely a particular vocabulary token is to appear next. The challenge: **how do you go from 50,257 numbers per token position to one actual predicted word?**

---

## 1. How LLMs Generate Text — Autoregressive Generation

### The fundamental mechanism

LLMs do **not** generate an entire sentence in one shot. They generate **one token at a time**, in a loop. Each new token is appended to the input, and the full extended sequence is fed back into the model for the next prediction. This is called **autoregressive generation**.

"Auto" = self. "Regressive" = fed back. The model feeds its own output back to itself.

### Step-by-step walkthrough

```
Iteration 1:
  Input tokens:  ["Hello", "I", "am"]          → token IDs: [15496, 314, 716]
  Model output:  logits for next token
  Predicted:     "a"  (token ID: 257)
  
Iteration 2:
  Input tokens:  ["Hello", "I", "am", "a"]     → token IDs: [15496, 314, 716, 257]
  Model output:  logits for next token
  Predicted:     "model"  (token ID: 2746)

Iteration 3:
  Input tokens:  ["Hello", "I", "am", "a", "model"]
  Model output:  logits for next token
  Predicted:     "ready"

... continues ...

After 6 iterations (max_new_tokens = 6):
  Final output:  "Hello I am a model ready to help."
```

Each iteration, the input sequence grows by exactly one token. The model re-processes the entire sequence from scratch each time (it has no internal memory between calls — each forward pass is independent).

### Why does each new token get appended?

Because the model uses **causal (masked) self-attention** — every token can only attend to tokens that came before it. So the last token in the sequence attends to all previous tokens, which means it has the most context and makes the most informed prediction about what comes next. By appending and re-running, we always give the model the fullest possible context for each new prediction.

### The context window — a hard limit

The model has a **maximum context size** — the maximum number of tokens it can process in one forward pass. For GPT-2 this is **1,024 tokens**.

If your input + generated tokens ever exceeds this:
- You **cannot** just pass all tokens in — the positional embedding layer only has embeddings for positions 0 to 1023
- You must **crop** the input to the last `context_size` tokens before each forward pass

```python
idx_cond = idx[:, -context_size:]  # Keep only last 1024 tokens
```

Tokens that fall outside the window are simply not visible to the model for that prediction. This is a fundamental limitation of transformer architectures.

---

## 2. The Output Tensor — What Every Row and Column Means

### Shape breakdown

```
Output logits tensor shape: [batch_size, num_tokens, vocab_size]
                                  2            4          50,257

Example with batch=2, input="every effort moves you" (4 tokens):
Shape = [2, 4, 50257]
```

- **Dimension 0 (batch_size = 2):** Two separate input sequences processed in parallel
- **Dimension 1 (num_tokens = 4):** One set of logits for each input token position
- **Dimension 2 (vocab_size = 50,257):** For each position, a score for every token in the vocabulary

### What each row in dimension 1 actually represents

This is the most important insight. The transformer processes all 4 input tokens simultaneously, but internally it computes predictions for every prefix of the sequence simultaneously. This is how it trains efficiently — multiple prediction tasks in one forward pass.

For input "every effort moves you":

| Token position | Token | This row predicts... | Expected answer |
|----------------|-------|----------------------|-----------------|
| Position 0 | `every` | What comes after just "every"? | `effort` |
| Position 1 | `effort` | What comes after "every effort"? | `moves` |
| Position 2 | `moves` | What comes after "every effort moves"? | `you` |
| Position 3 | `you` | What comes after "every effort moves you"? | **← This is the one we want** |

**During inference, you only care about the last row** — position 3 — because that's the prediction given the complete input sequence.

During training, all four rows are used to compute four separate losses simultaneously, which makes training much more efficient (you get 4 training signals from one forward pass).

### Why vocab_size = 50,257?

This is the size of the BPE (Byte Pair Encoding) vocabulary used by GPT-2, implemented via `tiktoken`. Every possible subword token the model knows about has one entry. The 50,257th entry is the special `<|endoftext|>` token assigned the largest ID (50,256).

---

## 3. From Logits to a Predicted Word — The 5-Step Process

### Step 1 — Get the output logits tensor

```python
with torch.no_grad():
    logits = model(idx_cond)
# Shape: [batch_size, num_tokens, vocab_size]
# e.g.: [2, 4, 50257]
```

`torch.no_grad()` tells PyTorch not to build the computation graph for gradient tracking — we don't need gradients during inference, so this saves memory and speeds things up.

### Step 2 — Extract only the last token's logits

```python
logits = logits[:, -1, :]
# Shape: [batch_size, vocab_size]
# e.g.: [2, 50257]
```

`[:, -1, :]` in plain English:
- `:` → keep all batches (don't filter any)
- `-1` → take only the last token position (the one with full context)
- `:` → keep all vocab columns

You've now reduced from 3D to 2D. Each row in the result is a vector of 50,257 logit scores — one score per vocabulary token — representing the model's belief about what token should come next after the full input.

### Step 3 — Convert logits to probabilities using Softmax

```python
probas = torch.softmax(logits, dim=-1)
# Shape: [batch_size, vocab_size]
# All 50,257 values per row now sum to exactly 1.0
```

**What softmax does mathematically:**

```
softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)   for all j in vocabulary
```

It exponentiates every logit (making all values positive) and then normalises by the sum. The result is a valid probability distribution — all values between 0 and 1, summing to 1.

**Why softmax and not just normalise by dividing by the sum?**

The exponential function does two things:
1. Makes all values positive (logits can be negative; probabilities can't)
2. Amplifies differences — a logit that is 2x larger than another doesn't just get 2x the probability, it gets exponentially more. This makes the distribution more decisive.

**Important note — softmax is monotonic:**

Softmax preserves the ordering of values. The token with the highest logit will always have the highest probability after softmax. This means for greedy decoding (just picking the max), softmax is technically redundant — `argmax(logits) == argmax(softmax(logits))` always.

We apply it anyway because:
- Probabilities are interpretable — you can say "the model is 57% confident the next word is 'forward'"
- Downstream sampling strategies (temperature scaling, top-k) require probability values, not raw logits

### Step 4 — Select the next token ID

```python
idx_next = torch.argmax(probas, dim=-1, keepdim=True)
# Shape: [batch_size, 1]
```

`torch.argmax` scans all 50,257 probability values and returns the **index** (not the value) of the maximum. That index is the **token ID**.

`keepdim=True` keeps the result as a 2D tensor `[batch, 1]` instead of a 1D tensor `[batch]` — this matters for the concatenation in the next step.

This strategy — always picking the single highest-probability token — is called **greedy decoding**. It's deterministic: same input always gives same output.

**Example:** If `idx_next = 3` for batch 0, you look up token ID 3 in the vocabulary using `tokenizer.decode([3])` and get the actual word (e.g., "forward").

### Step 5 — Append the new token and loop

```python
idx = torch.cat((idx, idx_next), dim=1)
# Shape grows: [batch, n_tokens] → [batch, n_tokens + 1]
```

`torch.cat` concatenates tensors along dimension 1 (the token dimension). The newly predicted token ID is added to the end of the running sequence.

On the next loop iteration, this extended sequence `idx` becomes the new input to the model. The model now sees one more token of context and makes its next prediction.

This loop repeats exactly `max_new_tokens` times.

---

## 4. The Complete `generate_text_simple` Function

All 5 steps above packaged into one clean, reusable function:

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text autoregressively using greedy decoding.

    Args:
        model:          An instance of GPTModel
        idx:            Input token IDs, shape [batch_size, n_tokens]
                        Each row is a separate sequence being generated
        max_new_tokens: How many new tokens to generate (number of loop iterations)
        context_size:   Maximum tokens the model can process (GPT-2 = 1024)
                        Used to crop the input if it grows too long

    Returns:
        idx:  Extended token ID tensor, shape [batch_size, n_tokens + max_new_tokens]
    """
    for _ in range(max_new_tokens):

        # --- CROP: ensure we never exceed the context window ---
        # If idx has more columns than context_size, take only the last context_size tokens
        # idx[:, -context_size:] when len(idx) < context_size just returns idx unchanged
        idx_cond = idx[:, -context_size:]

        # --- FORWARD PASS: run the full GPT model ---
        # No gradients needed — we are not training, just generating
        with torch.no_grad():
            logits = model(idx_cond)
        # logits shape: [batch_size, n_tokens, vocab_size]

        # --- EXTRACT: focus only on the last token's prediction ---
        # The last token has attended to all previous tokens and makes
        # the most informed prediction about what comes next
        logits = logits[:, -1, :]
        # logits shape: [batch_size, vocab_size]

        # --- SOFTMAX: convert raw scores to a probability distribution ---
        # Every value now between 0 and 1, all 50,257 values sum to 1 per row
        probas = torch.softmax(logits, dim=-1)
        # probas shape: [batch_size, vocab_size]

        # --- ARGMAX: greedy decoding — pick the single highest-probability token ---
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # idx_next shape: [batch_size, 1]

        # --- APPEND: extend the input sequence with the new token ---
        # This extended sequence becomes the input for the next iteration
        idx = torch.cat((idx, idx_next), dim=1)
        # idx shape: [batch_size, n_tokens + 1] (grows by 1 each iteration)

    return idx
    # Final shape: [batch_size, original_n_tokens + max_new_tokens]
```

### Parameters deep dive

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `model` | `GPTModel` | — | The trained (or initialised) GPT model instance |
| `idx` | `torch.Tensor` (long) | `[batch, n_tokens]` | Token IDs of the input prompt. Use `unsqueeze(0)` to add the batch dimension for a single prompt |
| `max_new_tokens` | `int` | — | Number of loop iterations = number of new tokens generated |
| `context_size` | `int` | — | From `GPT_CONFIG_124M["context_length"]` = 1024 for GPT-2 |

---

## 5. Running End-to-End — Full Pipeline

```python
import torch
import tiktoken

# --- Setup ---
tokenizer = tiktoken.get_encoding("gpt2")

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)  # 124M parameter model with random weights

# --- Step 1: Text → Token IDs ---
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
# encoded = [15496, 11, 314, 716]  ← these are the actual GPT-2 token IDs

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# unsqueeze(0) adds batch dimension: shape goes [4] → [1, 4]
# The model always expects a 2D input [batch_size, n_tokens]

# --- Step 2: Switch to eval mode ---
model.eval()
# CRITICAL: this disables dropout layers
# During training, dropout randomly zeros out neurons (prevents overfitting)
# During inference, we want all neurons active for deterministic, full output

# --- Step 3: Generate ---
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
# out shape: [1, 10]  (4 original tokens + 6 new tokens)
print("Output token IDs:", out)
print("Output length:", len(out[0]))  # → 10

# --- Step 4: Token IDs → Text ---
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# squeeze(0) removes batch dimension: [1, 10] → [10]
# .tolist() converts tensor to Python list for the tokenizer
print(decoded_text)
# → "Hello, I am Featureiman Byeswickcrit..." (gibberish — untrained model)
```

### Why `.eval()` is critical

```python
# Training mode (default after model = GPTModel(...))
model.train()
# Dropout is ACTIVE → randomly zeros neurons → non-deterministic output
# BatchNorm uses batch statistics → output changes with batch size

# Evaluation/Inference mode
model.eval()
# Dropout is DISABLED → all neurons active → deterministic output
# BatchNorm uses running statistics → consistent output

# Always call model.eval() before generating text or evaluating
# Always call model.train() before training
```

### Why `torch.no_grad()` is critical

```python
# Without no_grad():
logits = model(idx_cond)
# PyTorch builds a computation graph tracking every operation
# This graph is needed for backpropagation during training
# Memory cost: ~2-3x the model size just for storing the graph

# With no_grad():
with torch.no_grad():
    logits = model(idx_cond)
# No computation graph built
# Significantly less memory usage
# Faster forward pass (no gradient bookkeeping overhead)
```

During inference, you never call `.backward()`, so you never need the graph. Always use `torch.no_grad()` during generation.

---

## 6. Tracing a Single Token Through the Complete Pipeline

Let's trace exactly what happens when we predict the first new token after "Hello I am":

```
Input text: "Hello, I am"

Step 1 — Tokenize:
  tokenizer.encode("Hello, I am") = [15496, 11, 314, 716]
  encoded_tensor shape: [1, 4]  (batch=1, tokens=4)

Step 2 — Crop check:
  4 < 1024 (context_size), so no cropping needed
  idx_cond = idx = [[15496, 11, 314, 716]]

Step 3 — Forward pass through GPTModel:
  tok_embeds = embedding_layer([[15496, 11, 314, 716]])
  → shape: [1, 4, 768]  (each token → 768-dim vector)
  
  pos_embeds = position_embedding([0, 1, 2, 3])
  → shape: [1, 4, 768]  (position 0,1,2,3 → 768-dim each)
  
  x = tok_embeds + pos_embeds  → [1, 4, 768]
  x = dropout(x)
  x = transformer_block_1(x)   → [1, 4, 768]
  x = transformer_block_2(x)   → [1, 4, 768]
  ...  (12 transformer blocks total)
  x = transformer_block_12(x)  → [1, 4, 768]
  x = layer_norm(x)            → [1, 4, 768]
  logits = output_head(x)      → [1, 4, 50257]
  
  Shape: [batch=1, tokens=4, vocab=50257]

Step 4 — Extract last token:
  logits = logits[:, -1, :]   → [1, 50257]
  (50,257 raw scores for what comes after "Hello, I am")

Step 5 — Softmax:
  probas = softmax(logits)    → [1, 50257]
  (50,257 probabilities, all summing to 1.0)
  
  Example values (random weights, so these are arbitrary):
  probas[0][0]     = 0.000019  (probability for token "!")
  probas[0][716]   = 0.000043  (probability for token " am")
  probas[0][27018] = 0.000091  ← this happens to be highest (random)
  ...

Step 6 — Argmax:
  idx_next = argmax(probas)   → [[27018]]
  Shape: [1, 1]

Step 7 — Decode:
  tokenizer.decode([27018])   → some word (random with untrained weights)

Step 8 — Append:
  idx = cat([[15496, 11, 314, 716]], [[27018]], dim=1)
      = [[15496, 11, 314, 716, 27018]]
  Shape: [1, 5]

Next iteration uses [[15496, 11, 314, 716, 27018]] as input → predicts token 6...
```

---

## 7. The Multiple Prediction Tasks — Why the Model Learns Efficiently

This is a deep insight into why transformers train so efficiently.

When the model processes "every effort moves you" (4 tokens), it doesn't just make 1 prediction. It simultaneously makes **4 predictions**:

```
Input seen → Expected next token    (used during training)
──────────────────────────────────
"every"                   → "effort"
"every effort"            → "moves"
"every effort moves"      → "you"
"every effort moves you"  → "forward"   ← this is inference time
```

This is possible because of **causal masking** in multi-head attention: token at position i can only attend to positions 0 through i. So:
- Position 0 (`every`) can only see itself → predicts `effort`
- Position 1 (`effort`) sees `every effort` → predicts `moves`
- Position 2 (`moves`) sees `every effort moves` → predicts `you`
- Position 3 (`you`) sees all 4 tokens → predicts `forward`

During training, all 4 prediction errors contribute to the loss simultaneously — you get 4x the training signal per forward pass compared to an RNN that processes tokens one at a time. This is a major reason transformers train faster than their predecessors.

---

## 8. Why the Output is Gibberish — And Why That's Correct

```python
# With random weights, output looks like:
"Hello, I am Featureiman Byeswickcrit..."

# After training on text data, output will look like:
"Hello, I am a language model ready to help..."
```

**Random weights → random output. This is expected and correct.**

The model architecture is fully functional. Every layer is running, every computation is happening. The problem is that the 124 million weight parameters contain random numbers from `torch.manual_seed(123)`. The attention heads haven't learned what tokens relate to each other. The feed-forward layers haven't learned any language patterns. The output head maps embeddings to the vocabulary in a completely arbitrary way.

Think of it like a calculator with the circuit board fully wired but no program loaded yet. The hardware works — it just doesn't know what to compute.

### What training will fix

| Component | Before training | After training |
|-----------|----------------|----------------|
| Attention heads | Random attention patterns | Learns to focus on relevant context |
| Feed-forward layers | Random transformations | Learns syntactic/semantic patterns |
| Output head weights | Random mapping to vocab | Maps context to likely next tokens |
| Token embeddings | Random 768-dim vectors | Similar words get similar vectors |

---

## 9. What's Coming Next — The Training Module

The full architecture pipeline is now in place:

```
Text → Tokenize → Token IDs → GPTModel → Logits → Probabilities → Next Token ID → Text
```

The next module focuses on **making the weights meaningful** through training:

1. **Loss function:** Cross-entropy loss comparing predicted token probabilities against the actual next token in the training data
2. **Backpropagation:** Computing gradients of the loss with respect to all 124M parameters
3. **Optimizer (AdamW):** Updating all parameters to reduce the loss
4. **Training loop:** Repeating this over millions of text examples until the model learns language patterns
5. **Better decoding strategies:**
   - **Temperature scaling** — divide logits by T before softmax to control randomness
   - **Top-k sampling** — restrict sampling to only the k most probable tokens
   - These require softmax probabilities — which is exactly why we kept the softmax step in `generate_text_simple` even though argmax doesn't technically need it

---

## 10. Complete Inference Flow — Visual Summary

```
"Hello, I am"                          ← Input text
        │
        ▼
[15496, 11, 314, 716]                  ← tokenizer.encode()
        │
        ▼
tensor([[15496, 11, 314, 716]])         ← .unsqueeze(0) adds batch dim [1,4]
        │
        ├─────────────────────────────────────────────────┐
        │           LOOP (max_new_tokens times)           │
        │                                                 │
        ▼                                                 │
idx[:, -context_size:]                 ← crop if needed  │
        │                                                 │
        ▼                                                 │
GPTModel forward pass                                     │
  tok_emb + pos_emb → dropout → 12x TransformerBlock     │
  → LayerNorm → Linear(768→50257)                        │
        │                                                 │
        ▼                                                 │
logits  [1, n_tokens, 50257]           ← full output     │
        │                                                 │
        ▼                                                 │
logits[:, -1, :]  [1, 50257]           ← last token only │
        │                                                 │
        ▼                                                 │
softmax(logits)   [1, 50257]           ← probabilities   │
        │                                                 │
        ▼                                                 │
argmax(probas)    [1, 1]               ← token ID        │
        │                                                 │
        ▼                                                 │
torch.cat(idx, idx_next, dim=1)        ← append ─────────┘
        │
        ▼  (after loop ends)
[15496, 11, 314, 716, t1, t2, t3, t4, t5, t6]
        │
        ▼
tokenizer.decode(...)                  ← back to text
        │
        ▼
"Hello, I am [6 new tokens]"           ← output text
```

---

## Key Concepts to Remember

- **Autoregressive generation** — the model generates one token at a time, appending each new token to the input before the next prediction; this loops until `max_new_tokens` is reached
- **Context window (1024 for GPT-2)** — hard limit on input length per forward pass; inputs longer than this must be cropped using `idx[:, -context_size:]`
- **Output tensor shape `[batch, num_tokens, vocab_size]`** — the model produces one 50,257-dim logit vector per input token position per batch
- **Only the last row matters at inference** — `logits[:, -1, :]` extracts predictions given the full input context
- **Multiple prediction tasks per forward pass** — during training, all token positions contribute to the loss simultaneously (4 tokens = 4 training signals); during inference only the last matters
- **Softmax is monotonic** — argmax(logits) == argmax(softmax(logits)); softmax isn't needed for greedy decoding but is kept because future sampling strategies (temperature, top-k) require probabilities
- **Greedy decoding** — always picks the single highest-probability token via `torch.argmax`; deterministic, same output every run; basis for all more advanced strategies
- **`model.eval()`** — disables dropout and training-only layers; must be called before inference
- **`torch.no_grad()`** — disables computation graph building; must be used during inference to save memory and speed up forward pass
- **Gibberish output from untrained model is correct** — the architecture is working; the weights just haven't learned anything yet; training fills in the knowledge

---

*Lecture 25 | LLM from Scratch Series | Next: GPT Model Training*
