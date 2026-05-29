# 📘 Build a Large Language Model from Scratch
## Lecture 24 — Coding the Full GPT Model
> **Video:** Lecture 24 in the series

---

## 🗺️ What This Lecture Covers
1. The complete GPT forward pass — all steps with dimensions
2. Step-by-step dimension tracking (every effort moves you)
3. The full GPTModel class in PyTorch
4. Parameter count — why 163M and not 124M?
5. Weight Tying — what it is and why GPT-2 used it

---

## 1. The Complete GPT Forward Pass

Here is the full journey of input text through GPT-2:

```
Input text: "every effort moves you"
        ↓
Tokenize → Token IDs: [E_id, F_id, M_id, Y_id]
        ↓
STEP 1: Token Embeddings       [4, 768]
        ↓
STEP 2: Positional Embeddings  [4, 768]
        ↓
STEP 3: Input Embeddings = Token + Positional  [4, 768]
        ↓
STEP 4: Dropout                [4, 768]
        ↓
STEP 5: × 12 Transformer Blocks  [4, 768]
        ↓
STEP 6: Final Layer Normalization  [4, 768]
        ↓
STEP 7: Output Head (Linear 768 → 50257)  [4, 50257]
        ↓
Logits [4, 50257]   ← next word predictions
```

The shape changes **only once** — at the very last step when we go from 768 to 50257.

---

## 2. Step-by-Step Dimension Tracking

Working through a single batch with 4 tokens: `[every, effort, moves, you]`

### Step 1 — Token Embeddings

Each token ID looks up the **Token Embedding Matrix** ([50257, 768]):
```
"every"  → row E_id → 768-dim vector
"effort" → row F_id → 768-dim vector
"moves"  → row M_id → 768-dim vector
"you"    → row Y_id → 768-dim vector

Result: [4, 768]
```

Values are random at init, trained during pretraining.

### Step 2 & 3 — Positional Embeddings + Addition

The **Positional Embedding Matrix** ([1024, 768]) has one vector per position:
```
Position 0 → 768-dim vector (for "every")
Position 1 → 768-dim vector (for "effort")
Position 2 → 768-dim vector (for "moves")
Position 3 → 768-dim vector (for "you")

Input Embedding = Token Embedding + Positional Embedding
Result: [4, 768]
```

### Step 4 — Dropout

~10% of values across all embeddings are randomly zeroed out. Shape stays [4, 768]. Purpose: prevent overfitting during training.

### Step 5 — 12 Transformer Blocks

Each block takes [4, 768] and outputs [4, 768]. Internally it runs:
```
LayerNorm → Multi-Head Attention → Dropout → Shortcut
LayerNorm → Feed-Forward (768→3072→768) → Dropout → Shortcut
```
After all 12 blocks, vectors are now **context vectors** — not just semantic meaning but how each token relates to every other token.

### Step 6 — Final Layer Normalization

One more normalization pass. Shape stays [4, 768].

### Step 7 — Output Head

A single Linear layer maps from 768 to 50257:
```
[4, 768] × [768, 50257] = [4, 50257]
```

Each row is now a **logit vector** — 50,257 scores, one per vocabulary token. The token with the highest score is the predicted next word.

### The 4 Prediction Tasks

With 4 input tokens there are 4 simultaneous prediction tasks:
```
Row 0:  "every"             → predict "effort"
Row 1:  "every effort"      → predict "moves"
Row 2:  "every effort moves"→ predict "you"
Row 3:  "every effort moves you" → predict "forward"
```

That's why the output has 4 rows — one prediction per input position.

### With Batches (the full shape)

```
Input:  [batch=2, tokens=4]
Output: [batch=2, tokens=4, vocab=50257]

= [2, 4, 50257]
```

---

## 3. The Full GPTModel Class

```python
import torch
import torch.nn as nn

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Token Embedding Matrix: [vocab_size, emb_dim]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Positional Embedding Matrix: [context_length, emb_dim]
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 12 Transformer Blocks chained with nn.Sequential
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final Layer Normalization
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Output Head: 768 → 50257
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx shape: [batch, seq_len]
        batch_size, seq_len = in_idx.shape

        # Step 1 & 2: Token + Positional Embeddings
        tok_embeds = self.tok_emb(in_idx)                        # [batch, seq, 768]
        pos_embeds = self.pos_emb(torch.arange(seq_len))         # [seq, 768]

        # Step 3: Combine them
        x = tok_embeds + pos_embeds                              # [batch, seq, 768]

        # Step 4: Dropout
        x = self.drop_emb(x)                                     # [batch, seq, 768]

        # Step 5: 12 Transformer Blocks
        x = self.trf_blocks(x)                                   # [batch, seq, 768]

        # Step 6: Final Layer Norm
        x = self.final_norm(x)                                   # [batch, seq, 768]

        # Step 7: Output Head → Logits
        logits = self.out_head(x)                                # [batch, seq, 50257]
        return logits
```

### GPT-2 Small Configuration

```python
GPT2_CONFIG = {
    "vocab_size":     50257,
    "context_length": 1024,
    "emb_dim":        768,
    "n_heads":        12,
    "n_layers":       12,
    "drop_rate":      0.1,
    "qkv_bias":       False
}
```

### Running the Model

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# Two sentences, 4 tokens each
batch = torch.stack([
    torch.tensor(tokenizer.encode("every effort moves you")),
    torch.tensor(tokenizer.encode("I like learning deep"))
])
# batch shape: [2, 4]

model = GPTModel(GPT2_CONFIG)
logits = model(batch)

print(logits.shape)   # torch.Size([2, 4, 50257])
```

---

## 4. Parameter Count — Why 163M not 124M?

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# 163,009,536  ← ~163 million
```

But GPT-2 Small is advertised as **124M parameters**. Where's the difference?

The answer is **Weight Tying**.

---

## 5. Weight Tying

### What is it?

Weight Tying = the **Token Embedding Matrix** and the **Output Head Matrix** are the **same weights** — not separate copies.

```
Token Embedding: [50257, 768]   ← maps token IDs to 768-dim vectors
Output Head:     [768, 50257]   ← maps 768-dim vectors to 50257 scores
```

Notice: these have the exact same shape (just transposed). The intuition is:
- If the embedding knows "cat" is related to "kitten", the output layer should understand the same relationship in reverse.

### How it affects parameter count

```
Without weight tying (our model):
  Token embedding: 50257 × 768 = 38.6M params
  Output head:     768 × 50257 = 38.6M params
  Subtotal:        77.2M
  Rest of model:   ~85.8M
  TOTAL:           163M

With weight tying (original GPT-2):
  Token embedding: 50257 × 768 = 38.6M params  ← shared!
  Output head:     same matrix, no extra params
  Rest of model:   ~85.8M
  TOTAL:           124M
```

### Implementing Weight Tying

```python
# After creating the model, tie the weights
model.out_head.weight = model.tok_emb.weight
```

One line. After this, both layers point to the same weight matrix.

### Trade-off

| | Weight Tying | Separate Weights |
|--|-------------|-----------------|
| Parameters | 124M | 163M |
| Memory | Less | More |
| Training speed | Faster | Slower |
| Model quality | Slightly lower | Slightly better |

GPT-2 used weight tying to save memory. Modern LLMs tend to use separate weights for better performance.

### Memory Footprint

```python
total_size_bytes = total_params * 4   # 4 bytes per float32
total_size_mb = total_size_bytes / (1024 ** 2)
print(f"Model size: {total_size_mb:.1f} MB")
# ~620 MB for 163M parameters
```

620 MB for the smallest GPT-2. GPT-3 (175B params) would need ~700 GB — impossible on a laptop, which is why cloud GPUs are needed.

---

## 🔑 Key Terms

| Term | Meaning |
|------|---------|
| **Logits** | Raw output scores before softmax — one per vocabulary token, per position |
| **Token Embedding Matrix** | [vocab_size × emb_dim] — maps each token ID to a vector |
| **Positional Embedding Matrix** | [context_length × emb_dim] — one vector per sequence position |
| **Output Head** | Final Linear layer: 768 → 50257 to produce vocabulary scores |
| **Weight Tying** | Sharing the token embedding and output head weights to reduce parameters |
| **`nn.Sequential`** | PyTorch module for chaining layers — used to chain 12 Transformer Blocks |
| **`numel()`** | PyTorch method returning number of elements in a tensor (used to count params) |
| **163M vs 124M** | Difference due to whether weight tying is used or not |

---

## ⚡ Quick Summary

1. **Full GPT forward pass has 7 steps:** Token Emb → Pos Emb → Add → Dropout → 12× Transformer → LayerNorm → Output Head
2. **Shape changes only once:** stays [batch, tokens, 768] until the final step → [batch, tokens, 50257]
3. **Output = Logits [batch, 4, 50257]** — 50,257 scores per token position
4. **4 input tokens = 4 prediction tasks** — each row predicts the next token for that context
5. **Our model has 163M params** — GPT-2's original 124M used weight tying to share token embedding and output head
6. **Weight tying** = same matrix for input embedding and output projection; saves ~38.6M params
7. **620 MB** on disk for 163M params (4 bytes per float32)
8. The entire model is **~8 lines of code** once you have the building blocks

---

## 📌 What's Coming Next

- **Text Generation from Logits** — converting the 50,257-dimensional output into actual words
- Sampling strategies: greedy decoding, temperature scaling, top-k sampling
- Running a full forward pass and printing the predicted next word

---

*Notes — Vizuara: "Build a Large Language Model from Scratch"*
