# GPT Architecture — Text Generation from Output Tokens
### Lecture Notes | Build LLMs from Scratch Series

> **Context:** This is the final lecture of the GPT Architecture module. Previous lectures covered tokenization, embeddings, multi-head attention, transformer blocks, layer normalization, feed-forward networks, and the full GPT model class. These notes focus **only on what's new** — converting the GPT output tensor into actual generated text.

---

## What You Already Know (Quick Anchor)

By the end of the previous lecture, you had a working `GPTModel` that:
- Takes token IDs as input (shape: `[batch_size, num_tokens]`)
- Passes them through embeddings → dropout → transformer blocks → layer norm → output head
- Returns a **logits tensor** of shape `[batch_size, num_tokens, vocab_size]` (vocab_size = 50,257)

The question this lecture answers: **"How do we go from that big logits tensor to actual predicted words?"**

---

## 1. How LLMs Generate Text — The Core Mechanism

LLMs generate text **one token at a time**, iteratively. This is called **autoregressive generation**.

### The Loop

```
Input: "Hello I am"
  → Model predicts next token → "a"
  → Append "a" to input
  
Input: "Hello I am a"
  → Model predicts next token → "model"
  → Append "model" to input

Input: "Hello I am a model"
  → Model predicts next token → "ready"
  → ... continues until max_new_tokens is reached
```

Each new token becomes part of the next iteration's input. The model never goes back — it only appends forward.

### Key Parameter: Context Size
The model has a **maximum context size** (GPT-2 = 1,024 tokens). If your accumulated input ever exceeds this, you must **crop it** to the last `context_size` tokens before feeding the model. The model simply cannot "see" beyond its context window.

---

## 2. From Logits Tensor to a Predicted Word — 5 Steps

The output logits tensor has shape `[batch, num_tokens, vocab_size]`. Here's exactly how you extract the next word:

### Step 1 — Understand What the Output Tensor Contains

Each **row** in the token dimension represents a prediction task:

| Row | Input seen so far | Predicts |
|-----|-------------------|----------|
| Row 1 (`every`) | `every` | what comes after `every` |
| Row 2 (`effort`) | `every effort` | what comes after `every effort` |
| Row 3 (`moves`) | `every effort moves` | what comes after `every effort moves` |
| Row 4 (`you`) | `every effort moves you` | **what comes next — this is what we want** |

**You only care about the last row** — it represents the prediction given the full input.

### Step 2 — Extract the Last Token's Logit Vector

```python
logits = logits[:, -1, :]
# Shape goes from [batch, num_tokens, vocab_size]
#              to [batch, vocab_size]
```

`[:, -1, :]` means: for every batch, take only the last token's row, keep all vocab columns.

### Step 3 — Convert Logits to Probabilities (Softmax)

```python
probas = torch.softmax(logits, dim=-1)
# Shape: [batch, vocab_size]
# All values now sum to 1.0 per row
```

Each of the 50,257 values now represents the probability that the next token is that vocabulary entry.

> **Important note on softmax here:** Technically, softmax is **not required** to find the highest-scoring token, because softmax is *monotonic* — it preserves order. `argmax(logits) == argmax(softmax(logits))` always. We apply softmax here anyway because:
> - It gives you interpretable probabilities (useful for understanding confidence)
> - In the training module, you'll use these probabilities for **sampling strategies** (temperature, top-k) that *require* probability values, not raw logits

### Step 4 — Pick the Token with Highest Probability

```python
idx_next = torch.argmax(probas, dim=-1, keepdim=True)
# Shape: [batch, 1]
```

This gives you the **token ID** (an integer like `57`) of the most probable next word. You then look this up in the tokenizer vocabulary to get the actual word.

This strategy — always picking the highest-probability token — is called **greedy decoding**.

### Step 5 — Append and Repeat

```python
idx = torch.cat((idx, idx_next), dim=1)
# Shape grows: [batch, n_tokens] → [batch, n_tokens + 1]
```

The new token is appended to the running input sequence. Next iteration, this extended sequence goes back into the model. Repeat until `max_new_tokens` is reached.

---

## 3. The `generate_text_simple` Function

Everything above is packaged into one clean function:

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        
        # Crop if input exceeds context window
        idx_cond = idx[:, -context_size:]
        
        # Forward pass (no gradient needed — we're not training)
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Extract last token's logits → [batch, vocab_size]
        logits = logits[:, -1, :]
        
        # Softmax → probabilities
        probas = torch.softmax(logits, dim=-1)
        
        # Greedy pick: highest probability token
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
```

### Parameters explained

| Parameter | What it is |
|-----------|------------|
| `model` | Your `GPTModel` instance |
| `idx` | Input token IDs, shape `[batch, n_tokens]` |
| `max_new_tokens` | How many new tokens to generate (controls loop count) |
| `context_size` | Max tokens the model can see — used for cropping |

---

## 4. Running It End-to-End

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# Step 1: Encode input text
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dim → [1, 3]

# Step 2: Set model to eval mode (disables dropout)
model.eval()

# Step 3: Generate
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]  # 1024
)

# Step 4: Decode back to text
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
```

### Why `.eval()` matters
During training, **dropout** randomly zeros out neurons to prevent overfitting. During inference (generation), you want deterministic, full output. `model.eval()` disables dropout and any other training-only layers. Always call this before generating text.

### Why `torch.no_grad()`
Gradient computation is only needed during training (backpropagation). Wrapping the forward pass in `torch.no_grad()` skips building the computation graph, saving significant memory and compute.

---

## 5. Why the Output is Gibberish Right Now

When you run this with a freshly initialized model, you'll get nonsense output like:

```
Hello, I am Featureiman Byeswickcrit
```

**This is expected and correct behavior.** The 124M parameters are initialized with random weights. The model's architecture is complete and functional — it's just that those weights haven't learned anything yet.

After training (next module), the weights will be tuned so that:
- The attention heads learn what tokens relate to each other
- The feed-forward layers learn language patterns
- The output head learns to map embeddings to meaningful vocabulary distributions

The pipeline from input → output is now fully in place. Training is what fills it with knowledge.

---

## 6. What's Coming Next — The Training Module

Now that the architecture is complete, the next module will:

1. Define a **loss function** (cross-entropy over predicted vs actual next tokens)
2. Run **backpropagation** to compute gradients across all 124M parameters
3. Use an **optimizer** (typically AdamW) to update the weights
4. Introduce **better sampling strategies** beyond greedy decoding:
   - **Temperature scaling** — controls randomness of predictions
   - **Top-k sampling** — picks randomly from only the top-k probable tokens
   - These require the softmax probabilities — which is why we kept that step

---

## 7. Summary — What Happens at Inference

```
Text Input
    ↓  tokenizer.encode()
Token IDs  [batch, n_tokens]
    ↓  (loop starts)
Crop to context_size  →  idx[:, -context_size:]
    ↓  model forward pass
Logits  [batch, n_tokens, vocab_size]
    ↓  [:, -1, :]
Last token logits  [batch, vocab_size]
    ↓  softmax
Probabilities  [batch, vocab_size]
    ↓  argmax
Next token ID  [batch, 1]
    ↓  torch.cat
Extended sequence  [batch, n_tokens + 1]
    ↓  (repeat until max_new_tokens)
Final token ID sequence
    ↓  tokenizer.decode()
Generated Text
```

---

## Key Concepts to Remember

- **Autoregressive generation** — output tokens are fed back as input, one at a time
- **Context window** — the hard limit on how many tokens a model can "see"; inputs are cropped to this
- **Greedy decoding** — always pick the single highest-probability token (simplest strategy; used here)
- **Softmax is monotonic** — not strictly needed for greedy decoding, but needed for sampling strategies later
- **`model.eval()`** — must be called before inference to disable dropout
- **`torch.no_grad()`** — must wrap inference to skip gradient tracking and save memory
- **Gibberish output = untrained weights**, not a broken architecture

---

*These notes cover Lecture 7 (final) of the GPT Architecture module. Next: Training the GPT Model.*
