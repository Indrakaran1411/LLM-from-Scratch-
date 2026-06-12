# Loading Pre-trained OpenAI GPT-2 Weights into Custom GPT

> **Series:** Build LLMs from Scratch — Pre-training  
> **Topic:** Integrating OpenAI GPT-2 Pre-trained Weights  
> **Milestone:** This lecture is the culmination of the pre-training series — our custom GPT architecture now produces coherent text using real OpenAI weights.

---

## Table of Contents
1. [Why Load Pre-trained Weights?](#1-why-load-pre-trained-weights)
2. [GPT-2 Released Model Sizes](#2-gpt-2-released-model-sizes)
3. [The 7 Downloaded Files — What Each One Is](#3-the-7-downloaded-files--what-each-one-is)
4. [The params Dictionary — 5 Keys You Must Understand](#4-the-params-dictionary--5-keys-you-must-understand)
5. [How the params Dictionary Maps to the GPT Architecture](#5-how-the-params-dictionary-maps-to-the-gpt-architecture)
6. [Code — Downloading and Loading GPT-2 Weights](#6-code--downloading-and-loading-gpt-2-weights)
7. [Config Changes Required Before Loading](#7-config-changes-required-before-loading)
8. [Code — Assigning Weights to the Custom GPT Model](#8-code--assigning-weights-to-the-custom-gpt-model)
9. [Weight Tying — Why Parameters Drop from 164M to 124M](#9-weight-tying--why-parameters-drop-from-164m-to-124m)
10. [Testing the Model — Before vs After Pre-trained Weights](#10-testing-the-model--before-vs-after-pre-trained-weights)
11. [Summary](#11-summary)

---

## 1. Why Load Pre-trained Weights?

In this series we trained a GPT model from scratch on a single small book. The output was incoherent because:
- Training data was tiny (one book)
- Only 10 epochs of training
- The model never saw enough language patterns to learn meaningful representations

**The problem with training from scratch:**

| Approach | Data | Cost | Output Quality |
|---|---|---|---|
| Our training loop (1 book) | ~1MB | Free | Incoherent, overfitting |
| OpenAI GPT-2 pre-training | Huge internet dataset | ~$1M+ | Coherent, meaningful |

**Solution:** OpenAI has publicly released the GPT-2 weights. We can load these into our own GPT architecture and immediately get coherent output — without spending a million dollars on training.

> This is the standard workflow in industry: use a pre-trained foundation model, then fine-tune it for your specific task. Pre-training from scratch is rare unless you are a large research lab.

---

## 2. GPT-2 Released Model Sizes

OpenAI released weights for four GPT-2 model sizes:

| Model | Parameters | Transformer Blocks | Embedding Dim |
|---|---|---|---|
| GPT-2 Small | **124M** | 12 | 768 |
| GPT-2 Medium | 355M | 24 | 1024 |
| GPT-2 Large | 774M | 36 | 1280 |
| GPT-2 XL | 1558M | 48 | 1600 |

We are loading the **124M model** in this lecture. The same code works for larger models — just change the `model_size` argument.

---

## 3. The 7 Downloaded Files — What Each One Is

When you download GPT-2 weights (from Kaggle or OpenAI), you get 7 files. Each has a specific purpose:

| File | Purpose |
|---|---|
| `checkpoint` | Points to the path where model weights are stored (`model.ckpt`) |
| `model.ckpt.data` | **The actual weight values** — all 124M parameter values live here |
| `model.ckpt.index` | Index file to look up where each tensor is in the `.data` file |
| `model.ckpt.meta` | TensorFlow graph definition — structure of the model |
| `encoder.json` | Vocabulary — maps every token string to its token ID (50,257 entries) |
| `vocab.bpe` | BPE merge rules — which token pairs were merged and in what order |
| `hparams.json` | Model configuration — vocab size, context length, embedding dim, heads, layers |

### What hparams.json contains (GPT-2 Small)

```json
{
  "n_vocab": 50257,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12
}
```

### Note on vocab.bpe

The `Ġ` symbol in BPE merge rules is a special convention OpenAI uses to mark the **start of a new word** (equivalent to a space). For example `Ġhello` means `" hello"` with a leading space.

---

## 4. The params Dictionary — 5 Keys You Must Understand

The download utility loads all GPT-2 weights into a Python dictionary called `params`. This dictionary has exactly **5 keys**:

| Key | What it holds | Shape (GPT-2 Small) |
|---|---|---|
| `wte` | Token embedding matrix | `[50257, 768]` |
| `wpe` | Positional embedding matrix | `[1024, 768]` |
| `blocks` | All weights inside each Transformer block (list of 12) | nested dict |
| `g` | Final LayerNorm scale parameter | `[768]` |
| `b` | Final LayerNorm shift parameter | `[768]` |

**Why these 5 exactly?**  
They map directly to where trainable parameters exist in the GPT architecture:

```
Input Token IDs
      │
  [wte] Token Embeddings          ← params["wte"]
      +
  [wpe] Positional Embeddings     ← params["wpe"]
      │
  Transformer Block × 12          ← params["blocks"][0..11]
      │
  Final LayerNorm                 ← params["g"], params["b"]
      │
  Output Head (reuses wte)        ← weight tying, no new key needed
```

---

## 5. How the params Dictionary Maps to the GPT Architecture

### Inside each Transformer block (`params["blocks"][i]`)

Each of the 12 blocks is a nested dictionary. Here is the full key path to every trainable component:

#### Attention Layer (Query, Key, Value — fused)

```python
# GPT-2 fuses Q, K, V into one large matrix — we split it after loading
params["blocks"][i]["attn"]["c_attn"]["w"]  # shape: [768, 2304]  (768 * 3)
params["blocks"][i]["attn"]["c_attn"]["b"]  # bias: [2304]

# Output projection
params["blocks"][i]["attn"]["c_proj"]["w"]  # shape: [768, 768]
params["blocks"][i]["attn"]["c_proj"]["b"]  # bias: [768]
```

#### Feed Forward Neural Network (2-layer MLP)

```python
# First layer: fully connected (expand 768 → 3072, i.e. 4× expansion)
params["blocks"][i]["mlp"]["c_fc"]["w"]     # shape: [768, 3072]
params["blocks"][i]["mlp"]["c_fc"]["b"]     # bias: [3072]

# Second layer: projection (contract 3072 → 768)
params["blocks"][i]["mlp"]["c_proj"]["w"]   # shape: [3072, 768]
params["blocks"][i]["mlp"]["c_proj"]["b"]   # bias: [768]
```

#### Layer Normalization (LayerNorm 1 and 2)

LayerNorm normally has no weights — but GPT-2 adds trainable **scale (g)** and **shift (b)** after normalization. Both have size `[768]`.

```python
# LayerNorm before attention (LN1)
params["blocks"][i]["ln_1"]["g"]   # scale
params["blocks"][i]["ln_1"]["b"]   # shift

# LayerNorm before FFN (LN2)
params["blocks"][i]["ln_2"]["g"]   # scale
params["blocks"][i]["ln_2"]["b"]   # shift
```

---

## 6. Code — Downloading and Loading GPT-2 Weights

### Install dependencies

```bash
pip install tensorflow>=2.15 tqdm>=4.66
```

> GPT-2 weights were originally saved in TensorFlow format. We install TF only to read the checkpoint — our model stays in PyTorch.

### Download and load weights

```python
from gpt_download import download_and_load_gpt2

# Downloads 7 files (~500MB) and returns settings + params dicts
settings, params = download_and_load_gpt2(
    model_size="124M",          # options: "124M", "355M", "774M", "1558M"
    models_dir="gpt2"           # local directory to save files
)

# Verify settings
print(settings)
# {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}

# Verify params keys
print(params.keys())
# dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])

# Verify token embedding shape
print(params["wte"].shape)
# (50257, 768) — 50257 tokens × 768 embedding dimensions ✓
```

> Download takes 5–10 minutes on a good connection (~500MB). Do not close your session during download.

---

## 7. Config Changes Required Before Loading

Our training configuration used two settings that differ from GPT-2's actual configuration. We must align them before loading weights — otherwise shape mismatches occur.

| Setting | Our Training Config | GPT-2 Actual | Why it matters |
|---|---|---|---|
| `context_length` | 256 | **1024** | Positional embedding matrix shape must match |
| `qkv_bias` | False | **True** | GPT-2 attention layers have bias vectors; we must enable them |

```python
# Updated config to match GPT-2 Small exactly
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,      # changed from 256
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True             # changed from False
}

# Create model instance with updated config
model = GPTModel(GPT_CONFIG_124M)
```

> **Note:** Bias vectors in attention layers are not commonly used in newer LLMs (they don't improve performance). We enable them here only for compatibility with the released GPT-2 weights.

---

## 8. Code — Assigning Weights to the Custom GPT Model

### The assign() helper function

```python
import torch
import numpy as np

def assign(left, right):
    """
    Assigns 'right' tensor values to 'left' parameter.
    Checks shape compatibility first — mismatch means wrong weight mapping.
    """
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch: left {left.shape} vs right {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))
```

### Full weight loading function

```python
def load_weights_into_gpt(gpt, params):
    """
    Loads all GPT-2 pre-trained weights from params dict
    into our custom GPTModel instance.
    """

    # 1. Token and Positional Embeddings
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])

    # 2. Load weights for each Transformer block
    for b in range(len(params["blocks"])):
        block = params["blocks"][b]

        # --- Attention: Q, K, V (fused in GPT-2, split into 3 for our model) ---
        q_w, k_w, v_w = np.split(block["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight   = assign(gpt.trf_blocks[b].att.W_key.weight,   k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # --- Attention: Q, K, V biases ---
        q_b, k_b, v_b = np.split(block["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias   = assign(gpt.trf_blocks[b].att.W_key.bias,   k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        # --- Attention: Output projection ---
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, block["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,   block["attn"]["c_proj"]["b"])

        # --- Feed Forward: Layer 1 (fully connected) ---
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, block["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,   block["mlp"]["c_fc"]["b"])

        # --- Feed Forward: Layer 2 (projection) ---
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, block["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,   block["mlp"]["c_proj"]["b"])

        # --- LayerNorm 1 (before attention) ---
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, block["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, block["ln_1"]["b"])

        # --- LayerNorm 2 (before FFN) ---
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, block["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, block["ln_2"]["b"])

    # 3. Final LayerNorm (outside Transformer blocks)
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    # 4. Output head — NOT loaded separately (see weight tying below)
```

### Running the loader

```python
load_weights_into_gpt(model, params)
model.eval()   # set to inference mode

print("Weights loaded successfully.")
```

---

## 9. Weight Tying — Why Parameters Drop from 164M to 124M

GPT-2 uses a technique called **weight tying**: the token embedding matrix (`wte`) is **reused** as the output projection head.

```
Without weight tying:
  Token Embedding Matrix:  50257 × 768  = ~38.6M params
  Output Head Matrix:      768 × 50257  = ~38.6M params
  Total (just these two):  ~77.2M params

With weight tying:
  Token Embedding Matrix:  50257 × 768  = ~38.6M params
  Output Head:             reuses same matrix (no new params)
  Total (just these two):  ~38.6M params

  Savings: ~38.6M params → total drops from ~164M to ~124M
```

```python
# Weight tying in code — output head shares weights with token embeddings
gpt.out_head.weight = gpt.tok_emb.weight   # same object, no new parameters
```

**Why it works conceptually:**  
The token embedding matrix maps token IDs → vector space. The output head maps vector space → token probabilities. Using the same matrix for both forces the model to learn consistent token representations — a token's "meaning" going in matches its "meaning" coming out.

> Weight tying is less common in modern LLMs (GPT-3, LLaMA, etc.) because larger models can afford separate matrices and the performance difference justifies the cost.

---

## 10. Testing the Model — Before vs After Pre-trained Weights

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
input_text = "Every effort moves you"

token_ids = tokenizer.encode(input_text)
input_tensor = torch.tensor(token_ids).unsqueeze(0)

output_ids = generate_with_topk(
    model=model,
    idx=input_tensor,
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    temperature=1.5,
    top_k=50
)

output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
print(output_text)
```

### Comparison

| Condition | Output |
|---|---|
| **Without pre-trained weights** (random init) | `"one of the xmc laid down across the seers and silver of an exquisitely appointed"` — memorized, incoherent |
| **With GPT-2 pre-trained weights** | `"Every effort moves you toward finding an ideal new way to practice something what makes us want to be on top of that"` — coherent ✓ |

### Effect of temperature on pre-trained model

```python
# Temperature = 0.1 → Conservative, focused, low diversity
generate_with_topk(..., temperature=0.1, top_k=50)
# "Every effort moves you forward in a meaningful direction..."

# Temperature = 1.4 → Balanced (recommended starting point)
generate_with_topk(..., temperature=1.4, top_k=50)
# "Every effort moves you toward finding an ideal new way to practice..."

# Temperature = 10 → Very random
generate_with_topk(..., temperature=10, top_k=50)
# "Every effort moves you towards finding an ideal new set piece but only at times..."
```

> If the model produces coherent output after loading, the weights were assigned correctly. Any mistake in the key paths or shape mappings would cause the model to output garbage — coherent text is your confirmation test.

---

## 11. Summary

```
Goal:
  Load OpenAI GPT-2 pre-trained weights (124M) into our custom GPT architecture

Steps:
  1. Install TensorFlow (to read .ckpt format) + tqdm
  2. Download 7 files via download_and_load_gpt2() → returns settings + params
  3. Update config: context_length=1024, qkv_bias=True
  4. Create GPTModel(updated_config)
  5. Call load_weights_into_gpt(model, params)
  6. Verify: model.eval() → generate text → output should be coherent

params dict structure:
  params["wte"]              → Token embedding weights     [50257, 768]
  params["wpe"]              → Positional embedding weights [1024, 768]
  params["blocks"][i]        → Transformer block i weights  (nested dict)
    ├── ["attn"]["c_attn"]   → Fused Q, K, V weights + biases
    ├── ["attn"]["c_proj"]   → Attention output projection
    ├── ["mlp"]["c_fc"]      → FFN first layer (expand)
    ├── ["mlp"]["c_proj"]    → FFN second layer (contract)
    ├── ["ln_1"]             → LayerNorm 1 scale + shift
    └── ["ln_2"]             → LayerNorm 2 scale + shift
  params["g"]                → Final LayerNorm scale        [768]
  params["b"]                → Final LayerNorm shift        [768]

Key concepts:
  Weight tying     → Output head reuses token embedding matrix (saves ~38M params)
  Shape checking   → assign() validates shapes before every assignment
  TF → PyTorch     → Loaded via TensorFlow, converted to torch.nn.Parameter
```

---

**Next:** Fine-tuning — taking this pre-trained GPT-2 model and adapting it for specific downstream tasks (text classification, question answering, custom applications).
