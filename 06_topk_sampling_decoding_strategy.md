# Top-K Sampling — LLM Decoding Strategy

> **Series:** Build LLMs from Scratch — Pre-training  
> **Topic:** Top-K Sampling + Temperature Scaling (Combined Decoding Strategy)  
> **Prerequisite:** Temperature Scaling lecture (previous lecture in series)

---

## Table of Contents
1. [Why Decoding Strategies Exist](#1-why-decoding-strategies-exist)
2. [Quick Recap — Temperature Scaling](#2-quick-recap--temperature-scaling)
3. [The Problem with Temperature Alone](#3-the-problem-with-temperature-alone)
4. [What is Top-K Sampling?](#4-what-is-top-k-sampling)
5. [How Top-K Works Step by Step](#5-how-top-k-works-step-by-step)
6. [Combining Top-K with Temperature Scaling](#6-combining-top-k-with-temperature-scaling)
7. [The Complete Decoding Workflow](#7-the-complete-decoding-workflow)
8. [Code — generate_with_topk() Function](#8-code--generate_with_topk-function)
9. [Choosing the Right Values for K and Temperature](#9-choosing-the-right-values-for-k-and-temperature)
10. [Why This Prevents Overfitting](#10-why-this-prevents-overfitting)
11. [Summary](#11-summary)

---

## 1. Why Decoding Strategies Exist

The simplest way to predict the next token is **greedy decoding** — always pick the token with the highest probability. This seems logical but has a critical flaw.

```python
# Greedy (naive) decoding — always picks max probability token
next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
```

**Problems with greedy decoding:**
- The model always picks the same deterministic token for the same input
- The output becomes repetitive and memorized from training data — a classic sign of overfitting
- There is zero creativity or diversity in the output

> **Example:** Input: `"Every effort moves you"` → Greedy output: `"one of the xmc laid down across the seers and silver of an exquisitely appointed"` — repetitive, nonsensical, clearly memorized.

Decoding strategies solve this by making next-token selection **probabilistic** instead of deterministic.

---

## 2. Quick Recap — Temperature Scaling

Temperature scaling makes next-token selection probabilistic by:

1. Taking the **logit tensor** from the model output
2. **Dividing logits by a temperature value** before applying Softmax
3. **Sampling from a multinomial distribution** instead of using argmax

```python
# Temperature scaling — probabilistic sampling
logits = logits / temperature          # scale logits
probs = torch.softmax(logits, dim=-1)  # convert to probabilities
next_token_id = torch.multinomial(probs, num_samples=1)  # sample
```

**Effect of temperature on the distribution:**

| Temperature | Effect | Output Style |
|---|---|---|
| Very low (e.g. 0.1) | Sharper distribution — one token dominates | Conservative, repetitive |
| ~1.0 | Balanced distribution | Moderate creativity |
| High (e.g. 5.0) | Uniform distribution — all tokens roughly equal | Very random, nonsensical |

> The multinomial distribution samples tokens **proportional to their probability**. The highest probability token still wins most often, but other tokens get a chance — enabling creativity.

---

## 3. The Problem with Temperature Alone

Even with temperature scaling, there is still an issue.

When temperature is high (e.g. `temperature = 5`), the distribution becomes nearly uniform. This means **completely unrelated tokens** get a real chance of being selected as the next token.

**Example:**
- Input: `"Every effort moves you"`
- At temperature = 5, the token `"pizza"` has ~4% chance of being the next token
- There is no logical reason `"pizza"` should ever appear here

**Root cause:** Temperature scaling lowers probabilities of bad tokens but never fully removes them from the candidate pool. Every single token in the vocabulary (50,257 tokens for GPT-2) remains a candidate.

**Solution:** We need to **eliminate clearly wrong tokens** from the candidate pool entirely before applying temperature. That is exactly what Top-K Sampling does.

---

## 4. What is Top-K Sampling?

Top-K Sampling restricts the next-token selection pool to only the **K most likely tokens**. All other tokens are completely excluded before temperature scaling is applied.

**Intuition:**  
Instead of choosing the next token from 50,257 options (most of which are irrelevant), we first shortlist the top K plausible tokens and then sample from just those K options.

- K is small (e.g. 25, 50) relative to vocab size (50,257)
- Tokens outside the top K get **zero probability** — they cannot be selected
- Within the top K, multinomial sampling still applies — so selection remains probabilistic

---

## 5. How Top-K Works Step by Step

### Step 1 — Get the logit tensor from the model

```python
# Example logit tensor (simplified)
next_token_logits = tensor([4.51, 0.89, -1.20, 6.75, 0.34, -2.10, 1.45, 6.28])
#                   idx:     0     1      2     3     4      5     6     7
```

### Step 2 — Find the top K logits using `torch.topk()`

```python
K = 3
top_logits, top_indices = torch.topk(next_token_logits, k=K)
# top_logits  → tensor([6.75, 6.28, 4.51])
# top_indices → tensor([3, 7, 0])
```

`torch.topk()` returns the K highest values and their positions in the original tensor.

### Step 3 — Replace all non-top-K values with negative infinity

```python
# Find the minimum value among top K logits
min_top_k_logit = top_logits[-1]  # 4.51 (the smallest among top K)

# Replace everything below the threshold with -inf
new_logits = torch.where(
    next_token_logits < min_top_k_logit,
    torch.tensor(float('-inf')),
    next_token_logits
)
# Result: tensor([4.51, -inf, -inf, 6.75, -inf, -inf, -inf, 6.28])
```

**Why negative infinity?**  
When Softmax processes `-inf`, it outputs exactly `0.0`. So these tokens get zero probability and can never be sampled.

### Step 4 — Apply Softmax — only top K tokens survive

```python
probs = torch.softmax(new_logits, dim=-1)
# Result: tensor([0.21, 0.0, 0.0, 0.57, 0.0, 0.0, 0.0, 0.22])
# Only positions 0, 3, 7 have non-zero probability — the top 3
# Probabilities sum to 1.0 across only these 3 tokens
```

---

## 6. Combining Top-K with Temperature Scaling

Top-K and temperature are applied **in sequence**:

1. Apply Top-K first → restrict candidates to K tokens (replace rest with `-inf`)
2. Divide logits by temperature → control sharpness of the distribution among those K tokens
3. Apply Softmax → convert to probabilities
4. Sample from multinomial → probabilistic selection

```python
K = 3
temperature = 1.4

# Step 1: Apply Top-K mask
top_logits, _ = torch.topk(logits, k=K)
min_val = top_logits[:, -1].unsqueeze(-1)
logits = torch.where(logits < min_val, torch.tensor(float('-inf')), logits)

# Step 2: Temperature scaling
logits = logits / temperature

# Step 3: Softmax
probs = torch.softmax(logits, dim=-1)

# Step 4: Sample from multinomial
next_token_id = torch.multinomial(probs, num_samples=1)
```

**What each component contributes:**

| Component | Role |
|---|---|
| Top-K | Removes nonsensical tokens entirely. Pizza never gets a chance. |
| Temperature | Controls creativity among the remaining K good tokens. |
| Multinomial | Keeps selection probabilistic — avoids deterministic/repetitive output. |

---

## 7. The Complete Decoding Workflow

```
LLM Output
    │
    ▼
Logit Tensor  (shape: [vocab_size] = [50,257])
    │
    ▼
Top-K Sampling  ──► Keep top K logits, replace rest with -inf
    │
    ▼
Temperature Scaling  ──► Divide logits by temperature value
    │
    ▼
Softmax  ──► Convert to probability distribution (sums to 1)
    │
    ▼
Multinomial Sampling  ──► Sample next token probabilistically
    │
    ▼
Next Token ID
```

---

## 8. Code — `generate_with_topk()` Function

```python
def generate_with_topk(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=1.0,
    top_k=None,
    eos_id=None
):
    """
    Generates new tokens using Top-K sampling + Temperature scaling.

    Args:
        model        : Instance of GPTModel
        idx          : Input token IDs — shape (batch, seq_len)
        max_new_tokens: Number of new tokens to generate
        context_size : Max context length the model supports
        temperature  : Controls randomness. Higher = more random.
                       Set to 0 or None to use greedy (argmax).
        top_k        : Number of top tokens to consider. None = no restriction.
        eos_id       : End-of-sequence token ID. Generation stops if encountered.
    """
    for _ in range(max_new_tokens):

        # Crop context to the model's max context window
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Focus on the last token's logits only
        logits = logits[:, -1, :]  # shape: (batch, vocab_size)

        # --- Step 1: Apply Top-K mask ---
        if top_k is not None:
            top_logits, _ = torch.topk(logits, k=top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf'), device=logits.device),
                logits
            )

        # --- Step 2 & 3 & 4: Temperature + Softmax + Multinomial ---
        if temperature and temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        # --- Fallback: Greedy decoding if no temperature ---
        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        # --- Early stopping on end-of-sequence token ---
        if eos_id is not None and (next_token_id == eos_id).any():
            break

        # Append new token and continue
        idx = torch.cat([idx, next_token_id], dim=1)

    return idx
```

### Running the function

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
input_text = "Every effort moves you"

# Encode input
input_ids = tokenizer.encode(input_text)
input_tensor = torch.tensor(input_ids).unsqueeze(0)  # add batch dim

# Generate
output_ids = generate_with_topk(
    model=model,
    idx=input_tensor,
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    temperature=1.4,   # good balance of creativity and coherence
    top_k=25,          # restrict to top 25 tokens out of 50,257
)

# Decode output
output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
print(output_text)
# "Every effort moves you stand to work on surprise one of us had gone with random"
# — new words not present in training text = no overfitting ✓
```

---

## 9. Choosing the Right Values for K and Temperature

### Top-K value

| K value | Behavior |
|---|---|
| Very small (e.g. 1–5) | Very conservative. Almost like greedy. Low diversity. |
| Moderate (e.g. 10–50) | Good balance. Blocks random tokens, allows creativity. |
| Very large (e.g. 500+) | Too permissive. Bad tokens start sneaking back in. |

**Recommended starting point:** `top_k = 25` or `top_k = 50`

### Temperature value

| Temperature | Behavior |
|---|---|
| 0.0 | Pure greedy (argmax). Deterministic, repetitive. |
| 0.5–0.8 | More focused, coherent output. Lower creativity. |
| 1.0 | Balanced — default for most use cases. |
| 1.2–1.5 | More diverse and creative output. Some risk of incoherence. |
| > 2.0 | Very random. Output likely nonsensical. |

**Recommended starting point:** `temperature = 1.4` for creative generation with a GPT-2 scale model.

> Always treat these as hyperparameters and tune them based on your use case. A creative story generator needs higher temperature than a code completion model.

---

## 10. Why This Prevents Overfitting

**Overfitting in text generation** means the model just regurgitates training text verbatim instead of generating new content.

**Why greedy decoding causes overfitting:**  
The deterministic argmax always picks the exact same next token → the same sequence of tokens → the model effectively reproduces memorized training passages.

**Why Top-K + Temperature prevents it:**
- Multinomial sampling means the same input does not always produce the same output
- Top-K ensures only plausible tokens are in the running — so outputs stay meaningful
- Temperature controls how much the model explores beyond its top prediction

**Verification check:**  
After generating with Top-K + Temperature, confirm generated tokens are NOT in the training text:
```python
original_text = "..."  # your training passage

generated_tokens = ["stand", "to", "work", "on", "surprise"]
for word in generated_tokens:
    if word not in original_text:
        print(f"'{word}' is new — not memorized ✓")
```

---

## 11. Summary

```
Problem 1: Greedy decoding
  → Deterministic, repetitive, memorizes training text (overfitting)
  → Fix: Multinomial sampling from probability distribution

Problem 2: Temperature scaling alone
  → Nonsensical tokens (like "pizza") still have a chance
  → Fix: Top-K sampling — eliminate all but top K candidates first

Combined solution: Top-K + Temperature + Multinomial
  ┌─────────────────────────────────────────────────────────────┐
  │ Logits → Top-K mask → ÷ Temperature → Softmax → Multinomial │
  └─────────────────────────────────────────────────────────────┘
  ✓ Removes nonsensical tokens (Top-K)
  ✓ Controls creativity (Temperature)
  ✓ Keeps selection probabilistic (Multinomial)
  ✓ Prevents overfitting (all three together)
```

### Two decoding strategies learned so far

| Strategy | What it controls | How |
|---|---|---|
| Temperature Scaling | Sharpness of the probability distribution | Divide logits by T before Softmax |
| Top-K Sampling | Which tokens are even eligible | Replace non-top-K logits with -inf |

---

**Next:** Loading pre-trained GPT-2 weights from OpenAI — applying these decoding strategies with real production-scale weights.
