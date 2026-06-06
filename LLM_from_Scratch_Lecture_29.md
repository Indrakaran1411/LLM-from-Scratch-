# Temperature Scaling in LLMs — Controlling Output Randomness
### Lecture Notes | Build LLMs from Scratch Series — Lecture 29

> **Context:** In the previous lecture, we trained a GPT model completely from scratch for 10 epochs. The output was largely incoherent — words were generated but didn't form meaningful sentences. The reason: the model was using **greedy decoding**, which always picks the single most probable next token. This lecture introduces **temperature scaling** and **probabilistic sampling** — techniques that make the output more varied, creative, and eventually more meaningful.

---

## 1. The Problem with Greedy Decoding — Why It Fails

### What greedy decoding does

In `generate_text_simple`, the next token is selected like this:

```python
idx_next = torch.argmax(probas, dim=-1, keepdim=True)
```

`argmax` scans all 50,257 probabilities across the vocabulary and picks the index with the single highest value. Every time, without exception.

### Why this is a problem

**It is 100% deterministic.** Given the exact same input, the model will always produce the exact same output — every single run. There is:
- No exploration of alternative plausible tokens
- No variety between runs
- No creativity — the model is locked into one fixed path

But here's the key insight: the model doesn't just output one number — it outputs a **full probability distribution** over the entire vocabulary. Greedy decoding throws away almost all of that information by only looking at the maximum. That's wasteful.

For example, if the model predicts:
```
forward:  0.57   ← greedy always picks this
towards:  0.34   ← also very plausible, completely ignored
closer:   0.06   ← occasionally valid, completely ignored
```

"towards" has 34% probability — it's a very reasonable next word — but greedy decoding never generates it. This is why the output feels robotic and repetitive.

---

## 2. The Fix — Probabilistic Sampling with Multinomial Distribution

### Core idea

Instead of always picking the maximum, we **sample** the next token from the probability distribution itself. A token with 57% probability gets picked 57% of the time. A token with 34% probability gets picked 34% of the time. The selection is now **random but informed by the model's learned probabilities**.

### The multinomial distribution — what it actually is

The **multinomial distribution** is a probability distribution over `K` mutually exclusive outcomes, each with its own probability. Think of it like a weighted die:
- A regular die has 6 faces, each with equal probability (1/6)
- A multinomial "die" has 50,257 faces (one per vocab token), each with a different probability based on what the model learned

When you "roll" this die, you get a token — and the chance of getting any particular token is exactly its probability score.

Formally: if you have K possible outcomes with probabilities p₁, p₂, ..., pₖ where Σpᵢ = 1, and you run n independent trials, the count of each outcome follows a multinomial distribution.

In PyTorch this is implemented as:

```python
# Before (greedy — deterministic)
idx_next = torch.argmax(probas, dim=-1, keepdim=True)

# After (probabilistic — samples from distribution)
idx_next = torch.multinomial(probas, num_samples=1)
```

`torch.multinomial(probas, num_samples=1)` takes the probability tensor and draws one sample from it proportional to the probabilities.

### Concrete demonstration — 1,000 trials

To see the difference clearly, imagine running the sampling 1,000 times with the same input "every effort moves you" and a 9-token vocabulary:

| Token | Probability | Times Sampled (1000 trials) |
|-------|-------------|------------------------------|
| forward | 0.582 | 582 |
| towards | 0.343 | 343 |
| closer | 0.073 | 73 |
| inches | 0.002 | 2 |
| pizza | ~0.000 | 0 |

**Key observation:** The model now sometimes generates:
- "every effort moves you **forward**" (most of the time, 58%)
- "every effort moves you **towards**" (quite often, 34%)
- "every effort moves you **closer**" (occasionally, 7%)
- "every effort moves you **inches**" (rarely, 0.2%)

This is exactly the kind of variety and creativity we want. The most likely token is still most likely — we haven't broken anything. But we've opened the door to other valid completions.

### Important: multinomial still respects the model's knowledge

A common misconception: "sampling means random, so the model could say anything." That's wrong. The sampling is **proportional to learned probabilities**. If the model has learned that "forward" is a natural word after "every effort moves you", it will still be the most common output. Sampling just allows other plausible alternatives to occasionally appear.

---

## 3. What is Temperature Scaling — The Mathematical Mechanism

### Definition

**Temperature scaling** is the operation of dividing the raw logits (the model's output before softmax) by a positive scalar value called the **temperature**, before computing probabilities.

```python
scaled_logits = logits / temperature        # Step 1: scale
probas = torch.softmax(scaled_logits, dim=-1)  # Step 2: convert to probabilities
idx_next = torch.multinomial(probas, num_samples=1)  # Step 3: sample
```

That's it. One division. But the effect on the probability distribution is significant.

### Why dividing logits changes the distribution

Softmax is defined as:

```
softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)
```

When you divide all logits by a temperature T:

```
softmax(xᵢ / T) = exp(xᵢ/T) / Σ exp(xⱼ/T)
```

- When **T is small (< 1):** dividing by a small number makes the logits **larger in magnitude**. The differences between logits get amplified. The exponential function then makes the largest value dominate even more → **spiky, peaked distribution**
- When **T is large (> 1):** dividing by a large number makes the logits **smaller in magnitude**. The differences between logits shrink. The exponential function sees nearly equal values → **flat, uniform distribution**
- When **T = 1:** no change, standard softmax behaviour

### The complete generation workflow with temperature

```
Raw logits from model output
        ↓
Divide by temperature  →  scaled_logits = logits / T
        ↓
Apply softmax  →  probas = softmax(scaled_logits)
        ↓
Sample using multinomial  →  idx_next = multinomial(probas)
        ↓
Append to input sequence and repeat
```

---

## 4. Effect of Temperature — Deep Dive with Numbers

### Setup

Let's say the raw logits for our 9-token vocabulary after processing "every effort moves you" are:

```
forward:  2.1
towards:  1.8
closer:   0.9
inches:  -0.5
pizza:   -2.3
...
```

### Temperature = 1.0 (Baseline)

Logits unchanged. Standard softmax applied:

```
forward:  0.57
towards:  0.34
closer:   0.06
inches:   0.02
pizza:    ~0.00
```

This is your reference point. Multinomial sampling from this gives natural variety.

### Temperature = 0.1 (Very Low)

Logits divided by 0.1 (multiplied by 10 effectively):

```
scaled logits:
forward:  21.0
towards:  18.0
closer:    9.0
inches:   -5.0
pizza:   -23.0
```

After softmax, `exp(21)` is astronomically larger than `exp(18)`. Result:

```
forward:  ~0.999
towards:  ~0.001
closer:   ~0.000
inches:   ~0.000
pizza:    ~0.000
```

The multinomial now almost always picks `forward`. This is **nearly identical to greedy argmax**. The model becomes very consistent but loses all variety.

### Temperature = 5.0 (Very High)

Logits divided by 5:

```
scaled logits:
forward:  0.42
towards:  0.36
closer:   0.18
inches:  -0.10
pizza:   -0.46
```

All values are now very close together. After softmax:

```
forward:  0.20
towards:  0.18
closer:   0.16
inches:   0.13
pizza:    0.09
...
```

Now even "pizza" has 9% probability! The multinomial will occasionally generate "every effort moves you pizza". This is **too much randomness** — the output becomes incoherent.

### Visual Summary

```
Low Temperature (0.1)        Temperature = 1         High Temperature (5.0)
                                                      
forward ████████████████     forward ████████         forward ████
towards                      towards █████            towards ███
closer                       closer █                 closer ███
inches                       inches                   inches ██
pizza                        pizza                    pizza █
                                                      
(all mass on one token)    (natural distribution)   (nearly uniform)
```

### Summary Table

| Temperature | Mathematical Effect | Distribution Shape | Model Behaviour |
|-------------|--------------------|--------------------|-----------------|
| → 0 | Logits → ±∞ | Delta function (one spike) | Always same token, identical to argmax |
| 0.1 – 0.5 | Large logit gaps amplified | Very peaked | Consistent, low creativity |
| 1.0 | No change | Natural softmax | Balanced sampling |
| 1.5 – 3.0 | Logit gaps compressed | Moderately flat | High variety, some nonsense |
| > 5.0 | Logits nearly equal | Nearly uniform | Mostly incoherent |

---

## 5. Why is it Called "Temperature"? — The Physics Analogy

The term comes from **statistical mechanics** and **thermodynamics** in physics. The connection is through the concept of **entropy**.

### In physics

In a physical system (like a gas):
- **Low temperature:** Particles have low energy. The system settles into its lowest-energy state. Behaviour is ordered and predictable. **Low entropy.**
- **High temperature:** Particles have high energy. The system explores many states randomly. Behaviour is chaotic and unpredictable. **High entropy.**

### In LLMs

The exact same logic applies:
- **Low temperature:** The probability distribution has low entropy — one token dominates. The system (model) settles into its most likely state. Predictable output.
- **High temperature:** The probability distribution has high entropy — all tokens have roughly equal probability. The system explores many possible outputs. Chaotic, creative output.

### Entropy formula connection

Information entropy is defined as:
```
H = -Σ pᵢ × log(pᵢ)
```
- When one token has p ≈ 1 and all others p ≈ 0: H ≈ 0 (low entropy) → low temperature behaviour
- When all tokens have equal p = 1/K: H = log(K) (maximum entropy) → high temperature behaviour

This is exactly why the metaphor works — temperature directly controls the entropy of the token distribution.

---

## 6. Practical Guidance — Choosing the Right Temperature

### Use cases and recommended values

| Use Case | Recommended Temperature | Reason |
|----------|------------------------|--------|
| Code generation | 0.1 – 0.4 | Need syntactically correct, deterministic output |
| Factual Q&A | 0.3 – 0.6 | Accuracy matters more than variety |
| General conversation | 0.7 – 1.0 | Natural balance of coherence and variety |
| Creative writing | 0.8 – 1.2 | Want varied, interesting output |
| Brainstorming | 1.0 – 1.5 | Maximise diversity of ideas |
| Avoid (too high) | > 2.0 | Output becomes largely incoherent |

### The key tradeoff

```
Low Temperature ←————————————————→ High Temperature
More consistent                     More creative
More repetitive                     More varied  
More accurate (for factual tasks)   More surprising
Less interesting                    More risk of nonsense
```

There is no universally correct temperature. It depends entirely on what you want the model to do.

---

## 7. Updated `generate_text` Function — Full Implementation

Here is the updated generation function with temperature baked in:

```python
def generate_text(model, idx, max_new_tokens, context_size, temperature=1.0):
    """
    Generate tokens autoregressively with temperature-scaled probabilistic sampling.
    
    Args:
        model:          GPTModel instance
        idx:            Input token IDs, shape [batch, n_tokens]
        max_new_tokens: Number of new tokens to generate
        context_size:   Max tokens model can attend to (crop window)
        temperature:    Controls randomness. 1.0 = standard. < 1 = sharp. > 1 = flat.
    
    Returns:
        idx: Extended token ID sequence, shape [batch, n_tokens + max_new_tokens]
    """
    for _ in range(max_new_tokens):
        
        # Crop to context window if input grows too long
        idx_cond = idx[:, -context_size:]
        
        # Forward pass — no gradient needed during inference
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on last token's prediction: [batch, vocab_size]
        logits = logits[:, -1, :]
        
        # Apply temperature scaling BEFORE softmax
        # temperature=1.0 → no change (standard softmax)
        # temperature<1.0 → sharper distribution
        # temperature>1.0 → flatter distribution
        probas = torch.softmax(logits / temperature, dim=-1)
        
        # Sample from the distribution (NOT argmax)
        idx_next = torch.multinomial(probas, num_samples=1)  # [batch, 1]
        
        # Append newly generated token to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # [batch, n_tokens+1]
    
    return idx
```

### Key differences from `generate_text_simple`

| Aspect | `generate_text_simple` | `generate_text` |
|--------|------------------------|-----------------|
| Token selection | `torch.argmax` — always max | `torch.multinomial` — sampled |
| Deterministic? | Yes — same output every run | No — different each run |
| Temperature support | No | Yes — `/ temperature` before softmax |
| Creativity | None | Tunable |

### Usage example

```python
# Encode input
start_context = "Every effort moves you"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

model.eval()

# Low temperature — consistent output
out_low = generate_text(model, encoded_tensor, max_new_tokens=25,
                        context_size=GPT_CONFIG_124M["context_length"],
                        temperature=0.1)

# Balanced — natural sampling
out_balanced = generate_text(model, encoded_tensor, max_new_tokens=25,
                             context_size=GPT_CONFIG_124M["context_length"],
                             temperature=1.0)

# High temperature — creative/varied output
out_high = generate_text(model, encoded_tensor, max_new_tokens=25,
                         context_size=GPT_CONFIG_124M["context_length"],
                         temperature=5.0)

print(tokenizer.decode(out_low.squeeze(0).tolist()))
print(tokenizer.decode(out_balanced.squeeze(0).tolist()))
print(tokenizer.decode(out_high.squeeze(0).tolist()))
```

---

## 8. Temperature Scaling vs Greedy Decoding — Side by Side

```
Input: "Every effort moves you"

Greedy decoding (argmax):
  Run 1: "Every effort moves you forward forward forward forward..."
  Run 2: "Every effort moves you forward forward forward forward..."  ← identical
  Run 3: "Every effort moves you forward forward forward forward..."  ← identical

Temperature = 0.1 (near-greedy):
  Run 1: "Every effort moves you forward forward forward forward..."
  Run 2: "Every effort moves you forward forward towards forward..."  ← slight variation
  Run 3: "Every effort moves you forward forward forward towards..."

Temperature = 1.0 (balanced):
  Run 1: "Every effort moves you forward towards closer forward..."
  Run 2: "Every effort moves you towards forward inches forward..."   ← clear variety
  Run 3: "Every effort moves you closer towards forward forward..."

Temperature = 5.0 (high):
  Run 1: "Every effort moves you pizza inches closer forward towards..."
  Run 2: "Every effort moves you towards pizza forward inches pizza..."  ← too random
  Run 3: "Every effort moves you inches pizza towards closer pizza..."
```

---

## 9. What's Coming Next — Top-k Sampling

Temperature scaling alone has one remaining problem: **even at moderate temperatures, very unlikely tokens can still get sampled**.

If "pizza" has 0.01% probability and you're generating 1,000 tokens, statistically it will appear about 10 times. For a model generating coherent English text, generating "pizza" in the middle of a sentence about effort and movement is clearly wrong.

**Top-k sampling** solves this by:
1. Looking at all 50,257 probabilities
2. Keeping only the top `k` highest-probability tokens (e.g., top 50)
3. Setting all other probabilities to zero
4. Renormalising the remaining k probabilities to sum to 1
5. Sampling only from those top k tokens

This ensures the model never picks from the "long tail" of very unlikely tokens, regardless of temperature.

**The combination of temperature + top-k** is the standard decoding strategy used in production LLMs including GPT-2, GPT-3, and ChatGPT.

---

## Key Concepts to Remember

- **Greedy decoding** — `argmax`, always picks the single highest probability token; deterministic, zero variety
- **Multinomial sampling** — draws the next token proportionally to its probability; enables variety while respecting learned distributions
- **Temperature** — a positive scalar you divide logits by before softmax; controls the shape of the probability distribution
- **Low temperature (< 1)** — amplifies differences between logits → sharper distribution → more consistent, less creative output; approaches argmax as T → 0
- **High temperature (> 1)** — compresses differences between logits → flatter distribution → more creative, more varied, more risk of incoherence
- **Temperature = 1** — standard softmax, no change to the raw distribution
- **The physics analogy** — temperature in thermodynamics controls entropy of a physical system; same concept here — higher temperature = higher entropy in token distribution
- **Entropy connection** — low temperature = low entropy (one outcome dominates); high temperature = high entropy (all outcomes roughly equal)
- **Workflow:** logits → divide by T → softmax → multinomial sample → append → repeat

---

*Lecture 29 | LLM from Scratch Series | Next: Top-k Sampling*
