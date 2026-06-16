# Classification Fine-Tuning — Model Initialization, Pre-trained Weights & Architecture Modification
### Lecture Notes | Build LLMs from Scratch Series — Lecture 35

> **Context:** Lectures 31–34 covered fine-tuning concepts, dataset preparation, and DataLoader creation for spam classification. This lecture begins Stage 2: loading GPT-2 pre-trained weights into our architecture, modifying the output head for binary classification, and selectively freezing/unfreezing layers for efficient fine-tuning.

---

## 1. Where We Are — Full Pipeline Recap

### Completed so far (Stage 1)

```
✅ Downloaded SMS Spam Collection dataset (UCI repository)
✅ Balanced: 747 spam + 747 ham = 1,494 total
✅ Encoded labels: ham=0, spam=1
✅ Split: 70% train (1,045) / 10% val (149) / 20% test (300)
✅ Created SpamDataset class with tokenization + padding to 120 tokens
✅ Created DataLoaders: train (130 batches), val (19), test (38)
   Each batch: input_tensor[8, 120] + label_tensor[8]
```

### Stage 2 goals (this lecture)

```
Step 4: Initialize GPT model with GPT-2 configuration
Step 5: Load pre-trained GPT-2 weights from OpenAI
Step 6: Modify architecture for classification (replace output head)
        + Freeze/unfreeze selective layers
```

---

## 2. Why Load Pre-trained Weights Instead of Training From Scratch

Training GPT-2 from scratch requires:
- ~40GB of internet text data
- Millions of dollars in compute
- Weeks of training time

OpenAI has already done this and released the GPT-2 weights publicly. By loading these weights:
- Our model immediately understands English grammar and semantics
- It has learned relationships between words from massive text
- The lower layers already encode useful linguistic features
- We only need to teach it the narrow task of spam/ham classification

**The transfer learning principle:** General knowledge (pre-training on all of English) transfers to specific tasks (spam classification) — the model doesn't need to re-learn what words mean, just how to classify them.

---

## 3. GPT-2 Configuration for Fine-Tuning

When loading pre-trained weights, you must use **exactly the same configuration** that was used during pre-training. Any mismatch causes a shape error when loading the weight tensors.

```python
# Base configuration matching GPT-2 Small (124M parameters)
BASE_CONFIG = {
    "vocab_size": 50257,      # BPE vocabulary size — must match exactly
    "context_length": 1024,   # max tokens per forward pass
    "drop_rate": 0.0,         # dropout disabled for fine-tuning inference
    "qkv_bias": True          # GPT-2 used bias in attention projections
}

# Model size options from OpenAI
MODEL_CONFIGS = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# We use the smallest model for efficiency
choose_model = "gpt2-small (124M)"
BASE_CONFIG.update(MODEL_CONFIGS[choose_model])

# Final config:
# vocab_size=50257, context_length=1024, drop_rate=0.0, qkv_bias=True,
# emb_dim=768, n_layers=12, n_heads=12
```

### Context length check

```python
# If training emails are longer than GPT-2's context window:
if train_dataset.max_length > BASE_CONFIG["context_length"]:
    BASE_CONFIG["context_length"] = train_dataset.max_length
# In our case: max_length=120 << 1024, so no change needed
# But this guard prevents errors if a dataset had very long documents
```

---

## 4. Downloading and Loading GPT-2 Weights

### What gets downloaded

OpenAI provides 7 files for GPT-2 Small (~500MB total):
- `checkpoint` — training metadata
- `encoder.json` — BPE vocabulary
- `hparams.json` — hyperparameter configuration
- `model.ckpt.data-00000-of-00001` — the actual weight values
- `model.ckpt.index` — weight index file
- `model.ckpt.meta` — TensorFlow graph metadata
- `vocab.bpe` — BPE merge rules

```python
from gpt_download import download_and_load_gpt2

# Download weights and return settings + params dictionaries
settings, params = download_and_load_gpt2(
    model_size="124M",
    models_dir="gpt2"       # local directory to store downloaded files
)
```

### What `settings` and `params` contain

**`settings`** — the configuration dictionary:
```python
{
    "n_vocab": 50257,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12
}
```

**`params`** — a nested dictionary of all weight tensors:
```python
{
    "wte": array(shape=[50257, 768]),    # token embeddings
    "wpe": array(shape=[1024, 768]),     # positional embeddings
    "blocks": [                           # list of 12 transformer blocks
        {
            "attn": {
                "c_attn": {"w": ..., "b": ...},    # Q, K, V projections
                "c_proj": {"w": ..., "b": ...},    # output projection
            },
            "mlp": {
                "c_fc": {"w": ..., "b": ...},      # feed-forward expand
                "c_proj": {"w": ..., "b": ...},    # feed-forward contract
            },
            "ln_1": {"g": ..., "b": ...},          # layer norm 1
            "ln_2": {"g": ..., "b": ...},          # layer norm 2
        },
        # ... 11 more blocks
    ],
    "ln_f": {"g": ..., "b": ...}         # final layer norm
}
```

### Loading weights into our GPTModel

```python
# Create our GPTModel instance with random weights
model = GPTModel(BASE_CONFIG)

# Load the GPT-2 weights into our architecture
# This function maps OpenAI's key names to our key names and copies values
load_weights_into_gpt(model, params)

model.eval()
```

The `load_weights_into_gpt` function handles the key name mismatch between OpenAI's format (`wte`, `wpe`, `blocks[i].attn.c_attn`) and our model's format (`tok_emb`, `pos_emb`, `trf_blocks[i].att.W_query`). It copies each tensor value into the corresponding layer of our model.

### Verifying the weights loaded correctly

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# Test with a known input
text = "Every effort moves you"
token_ids = tokenizer.encode(text)
input_tensor = torch.tensor(token_ids).unsqueeze(0)

output = generate_text_simple(model, input_tensor, max_new_tokens=15,
                               context_size=BASE_CONFIG["context_length"])
print(tokenizer.decode(output.squeeze(0).tolist()))
# → "Every effort moves you forward. The first step is to understand the importance of your work"
# ✓ Coherent English = weights loaded correctly
```

**Why does coherent output confirm correct loading?** With random weights, the output would be gibberish (random token IDs). Coherent English text proves the GPT-2 language model weights are functioning correctly in our architecture.

---

## 5. Testing the Model Before Fine-Tuning — An Important Observation

Before modifying the architecture, let's test whether GPT-2 can already classify spam with just instruction prompting (no fine-tuning):

```python
# Can GPT-2 classify spam without any fine-tuning?
prompt = (
    "Is the following text spam? Answer with yes or no.\n\n"
    "You are a winner! You have been specifically selected to receive "
    "$1,000 cash or $2,000 award."
)

token_ids = tokenizer.encode(prompt)
output = generate_text_simple(model, torch.tensor(token_ids).unsqueeze(0),
                               max_new_tokens=23,
                               context_size=BASE_CONFIG["context_length"])
print(tokenizer.decode(output.squeeze(0).tolist()))
```

**Result:** The model produces incoherent or irrelevant output — it fails to answer "yes" or "no" correctly.

**Why does it fail?** GPT-2 was only pre-trained to predict the next token. It was never taught to follow instructions or answer questions in a structured format. This demonstrates exactly why fine-tuning is necessary — pre-training alone does not make an instruction-following classifier.

This is a key motivating example: even with excellent language understanding, the model needs task-specific fine-tuning to perform classification reliably.

---

## 6. Modifying the Architecture — Adding the Classification Head

### The original output head (text generation)

```
Original GPTModel output head:
  Linear(in_features=768, out_features=50257, bias=False)
  
For input "every effort moves you" (4 tokens):
  Output shape: [1, 4, 50257]
  
  Row 0 (every):  [logit_0, logit_1, ..., logit_50256]  ← predicts next after "every"
  Row 1 (effort): [logit_0, logit_1, ..., logit_50256]  ← predicts next after "every effort"
  Row 2 (moves):  [logit_0, logit_1, ..., logit_50256]  ← predicts next after "every effort moves"
  Row 3 (you):    [logit_0, logit_1, ..., logit_50256]  ← predicts next after full sentence
```

### The new classification head

```
New classification head:
  Linear(in_features=768, out_features=2, bias=False)
  
For input "do you have time" (4 tokens):
  Output shape: [1, 4, 2]
  
  Row 0 (do):   [score_ham, score_spam]
  Row 1 (you):  [score_ham, score_spam]
  Row 2 (have): [score_ham, score_spam]
  Row 3 (time): [score_ham, score_spam]  ← WE USE ONLY THIS ROW
```

The key insight: **we only use the last row** because:
- Causal (masked) self-attention means token at position i can only attend to positions 0 through i
- Position 0 (`do`) only knows about itself — minimal information
- Position 3 (`time`) has attended to ALL 4 tokens — contains the most complete representation of the full input
- The last token's representation encodes the entire sequence context

### Why 2 output nodes instead of 1?

For binary classification, technically 1 output node with sigmoid would work. But 2 output nodes with softmax is more general:

```python
# 1 output node (binary-specific, not generalizable):
output = sigmoid(linear(x))   # outputs probability of spam

# 2 output nodes (generalizable to N classes):
output = softmax(linear(x))   # outputs [p_ham, p_spam]
# For 3-class problem: just change num_classes=3 → output [p_class1, p_class2, p_class3]
```

Using `num_classes` as a variable means the same code works for spam detection (2 classes), sentiment analysis (3+ classes), topic classification (10+ classes) without any modification.

---

## 7. Freezing and Unfreezing Layers — Selective Fine-Tuning

### The full freeze → selective unfreeze strategy

```python
# Step 1: Freeze ALL parameters in the model
for param in model.parameters():
    param.requires_grad = False
# Now: 0 trainable parameters (nothing will be updated during training)

# Step 2: Replace output head with new classification head
# (new layers have requires_grad=True by default)
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],   # 768
    out_features=num_classes,              # 2
    bias=False
)
# Now: ~1,536 trainable parameters (768 × 2)

# Step 3: Unfreeze the final transformer block
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
# Now: ~1,536 + ~7M trainable parameters (last transformer block)

# Step 4: Unfreeze the final layer normalization
for param in model.final_norm.parameters():
    param.requires_grad = True
# Now: ~1,536 + ~7M + 1,536 trainable parameters (final LayerNorm has scale + shift, each 768-dim)
```

### Why freeze the lower layers?

The 12-layer GPT-2 transformer has a hierarchical structure where different layers capture different types of information:

```
Layer 1-4  (lower):  Basic linguistic features
  ├── Token embeddings capture word semantics
  ├── Early attention heads detect syntax (subject/verb agreement)
  ├── Position-sensitive features
  └── Sub-word and morphological patterns
  → Applicable to ANY NLP task — don't retrain these

Layer 5-8  (middle): Compositional features
  ├── Phrase-level representations
  ├── Coreference resolution
  └── Semantic role understanding
  → Still quite general — probably don't need retraining

Layer 9-11 (upper):  Task-specific features
  ├── High-level text representations
  ├── Task-relevant patterns
  └── Pre-training specific behaviours
  → Partially applicable — we unfreeze layer 12 only

Layer 12   (last):   Most task-specific
  └── Directly feeds the output head
  → We definitely retrain this

Final LayerNorm:     Normalises last transformer output
  └── Directly connected to new classification head
  → We definitely retrain this

Classification Head: Brand new, never trained
  └── Maps 768-dim → 2-dim for spam/ham
  → We definitely train this
```

### Trainable parameter count comparison

```python
# Count trainable parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"Total:     {total:,}")
    print(f"Trainable: {trainable:,}  ({100*trainable/total:.1f}%)")
    print(f"Frozen:    {frozen:,}  ({100*frozen/total:.1f}%)")

count_parameters(model)
# Total:     124,441,344
# Trainable:   7,080,192  (5.7%)
# Frozen:    117,361,152  (94.3%)
```

We're training only ~5.7% of the model's parameters. The other 94.3% remain exactly as OpenAI trained them.

### The benefit of this approach

```
Full fine-tuning (all 124M params):
  + Maximum accuracy potential
  - Risk of catastrophic forgetting (model forgets pre-trained knowledge)
  - 124M gradient computations per step
  - Requires much more data to avoid overfitting
  - With only 1,045 training samples, almost certain to overfit

Selective fine-tuning (7M params, ~5.7%):
  + Preserves pre-trained language understanding
  + Much faster training
  + Much lower overfitting risk with small dataset
  + Still adapts the most task-relevant layers
  - Slight accuracy trade-off (acceptable for most tasks)
```

---

## 8. Verifying the Architecture Modification

### Before modification

```python
print(model)
# GPTModel(
#   (tok_emb): Embedding(50257, 768)
#   (pos_emb): Embedding(1024, 768)
#   (drop_emb): Dropout(p=0.0)
#   (trf_blocks): Sequential(
#     (0): TransformerBlock(...)
#     ...
#     (11): TransformerBlock(...)     ← 12 blocks (0 to 11)
#   )
#   (final_norm): LayerNorm()
#   (out_head): Linear(in_features=768, out_features=50257, bias=False)  ← THIS CHANGES
# )
```

### After modification

```python
print(model.out_head)
# Linear(in_features=768, out_features=2, bias=False)   ← 50257 → 2
```

### Testing the modified model

```python
# Test with sample input
inputs = tokenizer.encode("Do you have time")
input_tensor = torch.tensor(inputs).unsqueeze(0)   # [1, 4]

with torch.no_grad():
    outputs = model(input_tensor)

print(outputs.shape)   # torch.Size([1, 4, 2])
# [batch=1, tokens=4, classes=2]

# Extract last token's output (the one with full context)
last_token_output = outputs[:, -1, :]
print(last_token_output.shape)   # torch.Size([1, 2])
print(last_token_output)
# tensor([[-3.5898,  3.9902]])   ← random values (not yet fine-tuned)
# index 0 = ham score, index 1 = spam score
# argmax → index 1 → predicted spam (but randomly, not yet trained)
```

The output dimensions are correct — 2 values per email (one per class). The values are random because the new classification head and last transformer block haven't been trained on spam data yet. But the **shape is right**, which confirms the architecture modification worked.

### Extracting last token — the correct indexing

```python
# Full output: [batch, seq_len, num_classes]
outputs = model(input_tensor)   # [1, 4, 2]

# Wrong — takes all tokens:
all_outputs = outputs          # [1, 4, 2] — don't use this for classification

# Correct — takes only the last token:
last_output = outputs[:, -1, :]   # [1, 2]

# The indexing [:, -1, :] means:
# ":"    → all batches (keep batch dimension)
# "-1"   → last position in the sequence dimension
# ":"    → all class scores (keep class dimension)
```

---

## 9. What's Coming Next — Loss Function and Training

The model architecture is now ready:
- ✅ GPT-2 pre-trained weights loaded (language understanding)
- ✅ Output head replaced: `Linear(768 → 2)` (classification output)
- ✅ Lower layers frozen (preserve pre-trained knowledge)
- ✅ Last transformer block + LayerNorm + new head are trainable

The next lecture implements:

1. **Loss function** — cross-entropy between the model's 2-class output and the true label:
   ```python
   loss = F.cross_entropy(last_token_output, label)
   ```

2. **Accuracy metric** — what fraction of predictions match the true label:
   ```python
   predicted = torch.argmax(last_token_output, dim=-1)
   accuracy = (predicted == labels).float().mean()
   ```

3. **Training loop** — iterate over DataLoader batches, compute loss, backpropagate, update the 7M trainable parameters via AdamW

4. **Evaluation** — measure accuracy on validation and test sets after fine-tuning

---

## 10. Complete Stage 2 Pipeline Summary

```
Stage 2: Model Setup

Step 4 — Initialize GPT model
  GPTModel(BASE_CONFIG)
  → 124M random parameters
  → Architecture: tok_emb → pos_emb → 12×TransformerBlock → LayerNorm → Linear(768→50257)

Step 5 — Load pre-trained GPT-2 weights
  download_and_load_gpt2("124M", "gpt2/")
  load_weights_into_gpt(model, params)
  → 124M parameters now = OpenAI's trained values
  → Model generates coherent English text ✓
  → But fails to classify spam (not fine-tuned yet) ✓ (expected)

Step 6 — Modify architecture for classification
  a) Freeze all parameters (requires_grad=False for all)
  b) Replace out_head: Linear(768→50257) → Linear(768→2)
     (new layer has requires_grad=True by default)
  c) Unfreeze trf_blocks[-1] (last transformer block)
  d) Unfreeze final_norm (final layer normalization)
  
  Result:
  → 7,080,192 trainable parameters (5.7%)
  → 117,361,152 frozen parameters (94.3%)
  → Output shape: [batch, seq_len, 2]
  → Classification uses last token: output[:, -1, :]

Next: Step 7 — Implement loss + accuracy + training loop
```

---

## Key Concepts to Remember

- **Transfer learning** — load pre-trained GPT-2 weights so model starts with language understanding; avoids training from scratch at massive cost
- **Configuration must match** — when loading pre-trained weights, architecture config (vocab_size, emb_dim, n_layers, n_heads) must be identical to what was used during pre-training
- **GPT-2 weight files** — ~500MB, 7 files downloaded from OpenAI; contains all 124M parameters; `params` dict maps component names to weight tensors
- **Coherent output = correct loading** — gibberish output means weight loading failed; coherent English confirms success
- **Pre-training ≠ instruction following** — GPT-2 base model cannot follow "answer yes or no" instructions; fine-tuning is required
- **Classification head** — replace `Linear(768→50257)` with `Linear(768→num_classes)`; only 2 output nodes for spam/ham; generalises to any number of classes
- **Last token only** — always extract `output[:, -1, :]` for classification because causal attention means the last token has attended to ALL previous tokens — maximum information
- **Freeze → selective unfreeze** — freeze all 124M params first, then unfreeze: new classification head + last transformer block + final layer norm; trains only ~7M params (5.7%)
- **Why freeze lower layers** — lower layers encode universal linguistic features (syntax, semantics) applicable to any NLP task; no need to retrain them on 1,045 spam examples
- **`requires_grad=False`** — frozen parameter; gradients not computed; not updated during backprop
- **`requires_grad=True`** — trainable parameter; gradients computed; updated by optimizer
- **New layers default to trainable** — when you create `nn.Linear(...)`, its `requires_grad` is `True` by default; no need to explicitly set it

---

*Lecture 35 | LLM from Scratch Series | Next: Classification Loss, Accuracy Metrics & Fine-Tuning Training Loop*
