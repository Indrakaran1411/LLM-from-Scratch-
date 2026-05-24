# 📘 Build a Large Language Model from Scratch
## Lecture 19 — Bird's Eye View of the LLM Architecture
> **Video:** https://youtu.be/4i23dYoXp-A

---

## 🗺️ What This Lecture Covers
1. Where we are in the series — what's done, what's next
2. The Transformer Block — what's inside it
3. GPT-2 Configuration — key numbers explained
4. The Full Forward Pass — step by step
5. Input → Output shape walkthrough
6. What are Logits and why the output is [tokens × 50257]

---

## 1. Series Progress So Far

```
STAGE 1: Building the LLM
  ✅ Data Preparation  → Tokenization, Token Embeddings, Positional Embeddings
  ✅ Attention         → Simplified, Self-Attention, Causal, Multi-Head
  🔥 LLM Architecture  ← WE ARE HERE (next 4-5 lectures)

STAGE 2: Pre-training
STAGE 3: Fine-tuning
```

---

## 2. The Transformer Block — What's Inside

The **Transformer Block** is the core repeating unit of the GPT model. Everything we've learned so far fits inside it.

```
Input Embeddings (token + positional)
        ↓
┌─────────────────────────────┐
│     TRANSFORMER BLOCK       │
│                             │
│  Layer Normalization        │
│        ↓                    │
│  Masked Multi-Head Attention│  ← (5 lectures we just finished)
│        ↓                    │
│  Dropout                    │
│        ↓ + shortcut         │
│  Layer Normalization        │
│        ↓                    │
│  Feed Forward Network       │  ← (GELU activation inside)
│        ↓                    │
│  Dropout                    │
│        ↓ + shortcut         │
└─────────────────────────────┘
        ↓
(Repeat × 12 blocks for GPT-2 small)
        ↓
Final Layer Normalization
        ↓
Output Head → Logits [tokens × vocab_size]
```

**Components we'll learn in upcoming lectures:**
- Layer Normalization
- Feed Forward Network with GELU activation
- Shortcut (Residual) Connections
- How all blocks stack together
- How logits become the predicted next word

---

## 3. GPT-2 Configuration — Key Numbers

We'll use **GPT-2 Small** (124M parameters) for this entire series. Why?
- Small enough to run locally
- OpenAI released the weights publicly (GPT-3/4 weights are NOT public)

```python
GPT2_CONFIG = {
    "vocab_size":    50257,   # number of unique tokens (BPE tokenizer)
    "context_length": 1024,   # max tokens to look at when predicting next word
    "emb_dim":         768,   # each token → 768-dimensional vector
    "n_heads":          12,   # number of attention heads per Transformer block
    "n_layers":         12,   # number of Transformer blocks stacked
    "drop_rate":       0.1,   # dropout probability
    "qkv_bias":       False   # no bias in Q, K, V matrices
}
```

### What each number means

**vocab_size = 50257**
The BPE tokenizer knows 50,257 unique tokens (subwords, words, characters). Every token in any input text gets converted to one of these 50,257 IDs.

**context_length = 1024**
The model looks at a maximum of 1,024 tokens at once to predict the next one. This is the "memory window" of the model.

**emb_dim = 768**
Every single token ID is converted into a 768-dimensional vector. This is what gets passed through all the layers.

**n_heads = 12**
There are 12 attention heads inside each Transformer block. Each head creates its own Q, K, V matrices and focuses on different patterns.

**n_layers = 12**
There are 12 Transformer blocks stacked on top of each other. Each block refines the representations from the previous one.

**GPT-2 model sizes for reference:**

| Model | n_layers | emb_dim | Parameters |
|-------|---------|---------|-----------|
| Small (ours) | 12 | 768 | 124M |
| Medium | 24 | 1024 | 345M |
| Large | 36 | 1280 | 762M |
| XL | 48 | 1600 | 1542M |

---

## 4. The Full Forward Pass — Step by Step

**Input sentence:** *"every effort moves you"*

### Step 1 — Tokenize → Token IDs
```
"every effort moves you"
    ↓  (BPE tokenizer)
[ID_1, ID_2, ID_3, ID_4]   ← 4 token IDs
```

### Step 2 — Token IDs → Token Embeddings (768D)
Each token ID is looked up in the **Token Embedding Matrix** (shape: [50257, 768]).

```
Token Embedding Matrix: 50257 rows × 768 columns
  Row 0     → 768-dim vector for token 0
  Row 1     → 768-dim vector for token 1
  ...
  Row 50256 → 768-dim vector for token 50256

ID_1 → look up row ID_1 → get 768-dim vector
ID_2 → look up row ID_2 → get 768-dim vector
ID_3 → look up row ID_3 → get 768-dim vector
ID_4 → look up row ID_4 → get 768-dim vector
```

Result: 4 vectors, each 768-dimensional.

### Step 3 — Add Positional Embeddings
Each token also gets a **positional vector** (position 0, 1, 2, 3) added to it.

```
Positional Embedding Matrix: 1024 rows × 768 columns
  Row 0 → vector for position 0
  Row 1 → vector for position 1
  ...

Position 0 vector + Token 1 embedding → final embedding for token 1
Position 1 vector + Token 2 embedding → final embedding for token 2
...
```

Result: 4 vectors of 768D, each now encoding both **meaning + position**.

### Step 4 — Dropout
A small number of embedding values are randomly zeroed out during training. Prevents overfitting.

### Step 5 — 12× Transformer Blocks
The 4×768 tensor passes through 12 Transformer blocks in sequence. Each block:
- Refines the 768-dim representation using multi-head attention
- Updates it with a feed-forward network
- Adds shortcut connections to prevent vanishing gradients

Throughout all 12 blocks, the shape stays **[4, 768]** — it doesn't change size.

### Step 6 — Final Layer Normalization
One more normalization pass after all 12 blocks.

### Step 7 — Output Head (Linear projection)
The 768-dim vectors are projected to **50257 dimensions** — one value per vocabulary token.

```
[4, 768]  ×  [768, 50257]  =  [4, 50257]
```

This gives us **logits** — a score for every possible next token, for every position.

---

## 5. What are Logits and Why [tokens × 50257]?

### What are Logits?
**Logits** = raw scores before softmax. Each of the 50,257 values represents how likely that token is to come next. The highest score wins.

### Why 4 rows?
Because input "every effort moves you" has 4 tokens, and the model makes **4 simultaneous predictions**:

```
Input token 1: "every"              → predict next word → "effort"
Input token 2: "every effort"       → predict next word → "moves"
Input token 3: "every effort moves" → predict next word → "you"
Input token 4: "every effort moves you" → predict next word → "forward"
```

### Why 50257 columns?
Because there are 50,257 possible next tokens. The model outputs a score for each one. We pick the highest.

```
Output shape for 2-sentence batch, 4 tokens each:
[2, 4, 50257]
 ↑  ↑  ↑
 │  │  └── score for each of 50257 vocabulary tokens
 │  └───── 4 tokens per sentence
 └──────── 2 sentences in the batch
```

---

## 6. Dummy GPT Model — Skeleton Code

This is a placeholder to show the overall structure. Each component (Transformer block, layer norm) will be filled in during later lectures.

```python
import torch
import torch.nn as nn

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Placeholder — full implementation in later lectures

    def forward(self, x):
        return x   # Just passes through for now

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # Placeholder — full implementation in later lectures

    def forward(self, x):
        return x   # Just passes through for now

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Token embedding: vocab_size rows × emb_dim columns
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Positional embedding: context_length rows × emb_dim columns
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 12 Transformer blocks stacked
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final layer normalization
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # Output head: projects 768-dim vectors to vocab_size (50257) scores
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx shape: [batch_size, seq_len]
        batch_size, seq_len = in_idx.shape

        # Step 1: Token embeddings  →  [batch, seq_len, emb_dim]
        tok_embeds = self.tok_emb(in_idx)

        # Step 2: Positional embeddings  →  [seq_len, emb_dim]
        pos_embeds = self.pos_emb(torch.arange(seq_len))

        # Step 3: Add them together  →  [batch, seq_len, emb_dim]
        x = tok_embeds + pos_embeds

        # Step 4: Dropout
        x = self.drop_emb(x)

        # Step 5: Pass through all 12 Transformer blocks
        x = self.trf_blocks(x)

        # Step 6: Final layer norm
        x = self.final_norm(x)

        # Step 7: Project to vocabulary size → Logits
        logits = self.out_head(x)
        return logits   # shape: [batch, seq_len, vocab_size]
```

### Running It

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# Two input sentences
batch = torch.stack([
    torch.tensor(tokenizer.encode("every effort moves you")),
    torch.tensor(tokenizer.encode("every day holds a"  ))
])
# batch shape: [2, 4]

model = DummyGPTModel(GPT2_CONFIG)
logits = model(batch)

print(logits.shape)   # torch.Size([2, 4, 50257])
```

**Output shape explained:**
```
[2, 4, 50257]
 ↑  ↑  ↑
 │  │  └── probability score for each of 50,257 vocabulary tokens
 │  └───── 4 tokens per sentence (4 prediction tasks per sentence)
 └──────── 2 sentences in the batch
```

Right now the logits are random (nothing is trained). After pre-training, these scores will actually tell us the correct next word.

---

## 🔑 Key Terms

| Term | Meaning |
|------|---------|
| **Transformer Block** | The repeating core unit — contains attention + feed-forward + layer norm + dropout |
| **n_layers = 12** | 12 Transformer blocks stacked on top of each other |
| **n_heads = 12** | 12 attention heads inside each Transformer block |
| **emb_dim = 768** | Every token is a 768-dimensional vector throughout the model |
| **Token Embedding Matrix** | [50257 × 768] lookup table — one 768D vector per vocabulary token |
| **Positional Embedding Matrix** | [1024 × 768] — one 768D vector per position in the sequence |
| **Logits** | Raw scores output by the model — one score per vocabulary token per position |
| **Output Head** | Final linear layer projecting 768D → 50257D |
| **Shortcut Connection** | Skip connection that adds the input directly to the output of a layer |
| **Layer Normalization** | Normalizes values within a layer to stabilize training |
| **GELU Activation** | Smooth activation function used inside the feed-forward network |

---

## ⚡ Quick Summary

1. The **Transformer Block** = Multi-Head Attention + Layer Norm + Feed Forward + Dropout + Shortcut Connections
2. GPT-2 Small has **12 Transformer blocks**, each using 768D embeddings and 12 attention heads
3. **Forward pass:** Token IDs → Token Embeddings → +Positional Embeddings → 12× Transformer Blocks → Layer Norm → Logits
4. **Logits shape = [batch, tokens, 50257]** — one score per vocabulary token per position
5. With 4 input tokens, there are **4 prediction tasks** happening in parallel
6. All weights (embedding matrices, attention weights) start **random** and get trained via backpropagation
7. At inference time, we pick the token with the **highest logit score** as the predicted next word

---

## 📌 Upcoming Lectures in This Module

| Lecture | Topic |
|---------|-------|
| Next | Layer Normalization — why and how |
| +1 | Feed-Forward Network + GELU activation |
| +2 | Shortcut (Residual) Connections |
| +3 | Putting it all together — full Transformer Block |
| +4 | Full GPT Model + Text Generation |

---

*Notes — Vizuara: "Build a Large Language Model from Scratch"*
