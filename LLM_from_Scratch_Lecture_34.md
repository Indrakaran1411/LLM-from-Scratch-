# Fine-Tuning for Classification — Creating DataLoaders for Spam Detection
### Lecture Notes | Build LLMs from Scratch Series — Lecture 32

> **Context:** In Lecture 31 we downloaded the SMS spam dataset, balanced it (747 spam + 747 ham), encoded labels (ham=0, spam=1), and split into train/val/test CSVs. This lecture takes those CSV files and builds PyTorch `Dataset` and `DataLoader` objects — converting raw text emails into padded, tokenized, batched tensors ready for model input.

---

## 1. Where We Are and What We Need

### Current state

After Lecture 31 we have three CSV files:
```
train.csv       — 1,045 rows (70% of 1,494)
validation.csv  —   149 rows (10%)
test.csv        —   300 rows (20%)

Each row: | Label (0 or 1) | Text (raw email string) |
```

### What the model needs

The GPT model cannot accept raw strings. It needs:
```
Input tensor:  [batch_size, sequence_length]  — integer token IDs
Label tensor:  [batch_size]                   — 0 or 1
```

Every email in a batch must have the **same number of tokens** because PyTorch tensors require fixed dimensions. But real emails have wildly different lengths — some might be 10 tokens, others 120.

**The gap this lecture bridges:**
```
Raw emails (variable length strings)
         ↓
Tokenize → token ID sequences (still variable length)
         ↓
Pad all sequences to same length
         ↓
Batch into tensors [8, 120]
         ↓
DataLoader wraps batches for easy iteration
```

---

## 2. The Variable Length Problem — Two Options

### The problem

Emails in the dataset have different lengths. When you tokenize them, you get sequences of different sizes:

```
Email 1: "Win a free iPhone!"       → [5716, 257, 1479, 3633, 0]          (5 tokens)
Email 2: "Can we meet tomorrow?"    → [6090, 356, 1826, 9281, 30]          (5 tokens)
Email 3: "Hello, I am writing to inform you about your account status..." → [... 120 tokens ...]
```

PyTorch cannot stack these into a single tensor because the dimensions don't match.

### Option 1 — Truncate to shortest (NOT recommended)

Find the shortest email and cut all others to that length:

```
Shortest email: 5 tokens
All others: truncated to 5 tokens

Email 1: [5716, 257, 1479, 3633, 0]          ← unchanged
Email 3: [15496, 11, 314, 716, 716, ...]     → [15496, 11, 314, 716, 716]  ← 115 tokens lost!
```

**Why this is bad:** You throw away the majority of the information in longer emails. A long spam email likely contains the most telling spam signals in its later content — truncating removes exactly that evidence.

### Option 2 — Pad to longest (RECOMMENDED)

Find the longest email and extend all shorter ones with a special padding token:

```
Longest email: 120 tokens
Shorter emails: padded with token ID 50256 until length = 120

Email 1: [5716, 257, 1479, 3633, 0, 50256, 50256, ..., 50256]  ← 115 padding tokens added
Email 3: [15496, 11, 314, 716, ...]                             ← unchanged, already 120
```

No information is lost. All sequences reach the same length. The model learns to ignore padding tokens (they carry no meaning).

---

## 3. The Padding Token — Why 50256 (`<|endoftext|>`)

The token ID used for padding is **50256**, which corresponds to the `<|endoftext|>` special token in GPT-2's vocabulary.

### GPT-2 vocabulary structure

GPT-2 uses a vocabulary of 50,257 tokens (token IDs 0 to 50,256):
- IDs 0–50,255: regular subword tokens (words, punctuation, subwords)
- ID 50,256: `<|endoftext|>` — the very last entry in the vocabulary

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# Verify
print(tokenizer.n_vocab)           # 50257
print(tokenizer.decode([50256]))   # <|endoftext|>
print(tokenizer.encode("<|endoftext|>",
      allowed_special={"<|endoftext|>"}))  # [50256]
```

### Why use `<|endoftext|>` specifically as padding?

Any token ID could technically work as padding, but `<|endoftext|>` is the best choice because:

1. **Semantic clarity:** It signals "nothing meaningful here" — exactly what padding means
2. **Pre-training alignment:** GPT-2 was pre-trained with `<|endoftext|>` between documents; the model has already learned that this token means "end of content, ignore what follows"
3. **No confusion:** Using a real word token (like token ID 0) would confuse the model into thinking there's actual content at the padded positions

When the model encounters a sequence of `<|endoftext|>` tokens at the end of an email, it has learned from pre-training to treat these as non-content, which is exactly the behaviour we want.

---

## 4. The `SpamDataset` Class — Implementation and Explanation

PyTorch requires you to define a `Dataset` subclass before creating a `DataLoader`. The dataset class handles the per-sample logic; the dataloader handles batching.

```python
import torch
from torch.utils.data import Dataset
import tiktoken
import pandas as pd

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        Args:
            csv_file:     Path to train.csv, validation.csv, or test.csv
            tokenizer:    tiktoken GPT-2 tokenizer instance
            max_length:   If None, computed from longest sequence in this dataset.
                          If int, sequences are padded/truncated to this length.
                          Use training set's max_length for val and test sets.
            pad_token_id: Token ID used for padding (default: 50256 = <|endoftext|>)
        """
        self.data = pd.read_csv(csv_file)
        self.pad_token_id = pad_token_id
        
        # ── STEP 1: Tokenize all text messages ──────────────────────────────
        # Convert every raw email string into a list of integer token IDs
        # tokenizer.encode() uses BPE: "hello world" → [31373, 995] (not word-by-word)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        # self.encoded_texts is now a list of lists (variable length)
        # e.g. [
        #   [5716, 257, 1479, 3633, 0],       ← 5 tokens
        #   [15496, 11, 314, 716, 716, ...],   ← 120 tokens
        #   ...
        # ]
        
        # ── STEP 2: Determine maximum sequence length ────────────────────────
        if max_length is None:
            # Find the longest tokenized sequence in THIS dataset
            self.max_length = self._longest_encoded_length()
        else:
            # Use externally provided max_length (e.g., training set's max)
            self.max_length = max_length
        
        # ── STEP 3: Truncate sequences exceeding max_length ──────────────────
        # Only relevant if user-specified max_length is shorter than some sequences
        # If max_length came from the dataset itself, no truncation happens
        self.encoded_texts = [
            encoded[:self.max_length]          # keep only first max_length tokens
            for encoded in self.encoded_texts
        ]
        
        # ── STEP 4: Pad sequences shorter than max_length ────────────────────
        # Append pad_token_id (50256) until every sequence has length = max_length
        self.encoded_texts = [
            # Current tokens + padding tokens to fill up to max_length
            encoded + [pad_token_id] * (self.max_length - len(encoded))
            for encoded in self.encoded_texts
        ]
        # Now every encoded text has exactly max_length tokens
        # e.g. [5716, 257, 1479, 3633, 0, 50256, 50256, ..., 50256]  ← 120 tokens
    
    def _longest_encoded_length(self):
        """Find the maximum token sequence length across all emails."""
        return max(len(encoded) for encoded in self.encoded_texts)
    
    def __len__(self):
        """Required by PyTorch: returns total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Required by PyTorch: returns one (input, label) pair.
        Called by DataLoader for each sample in a batch.
        
        Returns:
            encoded: LongTensor of shape [max_length] — token IDs for email idx
            label:   LongTensor scalar — 0 (ham) or 1 (spam)
        """
        encoded = self.encoded_texts[idx]
        label = self.data.iloc[idx]["Label"]
        
        return (
            torch.tensor(encoded, dtype=torch.long),   # [max_length]
            torch.tensor(label, dtype=torch.long)       # scalar: 0 or 1
        )
```

### Why inherit from `torch.utils.data.Dataset`?

PyTorch's `Dataset` base class requires exactly two methods to be implemented:
- `__len__()` — so the DataLoader knows how many samples exist
- `__getitem__(idx)` — so the DataLoader can fetch any sample by index

Once these are defined, PyTorch handles all batching, shuffling, and parallel loading automatically through the `DataLoader`.

### The three required methods in detail

**`__len__`:**
```python
dataset = SpamDataset("train.csv", tokenizer)
len(dataset)   # → 1045  (calls __len__)
```

**`__getitem__`:**
```python
encoded, label = dataset[0]   # calls __getitem__(0)
print(encoded.shape)          # torch.Size([120])
print(label)                  # tensor(1) or tensor(0)
print(encoded[:5])            # tensor([5716, 257, 1479, 3633, 0])
print(encoded[-3:])           # tensor([50256, 50256, 50256])  ← padding
```

---

## 5. Creating Dataset Instances — Training, Validation, and Test

### Training dataset — max_length computed from data

```python
tokenizer = tiktoken.get_encoding("gpt2")

# Create training dataset — max_length computed internally
train_dataset = SpamDataset(
    csv_file="train.csv",
    tokenizer=tokenizer,
    max_length=None       # automatically finds longest sequence
)

print(train_dataset.max_length)  # → 120
# The longest email in the training set has 120 tokens
```

GPT-2 supports up to 1,024 tokens (its context length), so 120 is well within range.

### Why 120 tokens for SMS messages?

SMS messages are short. An average sentence is ~15-20 tokens with BPE encoding. The longest message in the dataset spans about 6-8 sentences — hence 120 tokens maximum.

### Validation and test datasets — use training max_length

```python
# IMPORTANT: use the SAME max_length as training set
# This ensures consistent tensor dimensions across all splits

val_dataset = SpamDataset(
    csv_file="validation.csv",
    tokenizer=tokenizer,
    max_length=train_dataset.max_length   # = 120
)

test_dataset = SpamDataset(
    csv_file="test.csv",
    tokenizer=tokenizer,
    max_length=train_dataset.max_length   # = 120
)
```

**Why use training max_length for val/test?**

If a validation or test email is longer than 120 tokens, it gets **truncated** to 120. This is intentional:
- The model was trained on sequences of max 120 tokens
- It should be evaluated on the same length sequences
- Using a different length at evaluation time would be inconsistent

If we set `max_length=None` for val/test, those datasets would find their own longest sequence, potentially giving different sequence lengths — meaning validation and test tensors would have different shapes than training tensors. This breaks batching when mixing data.

---

## 6. Creating DataLoaders

The `Dataset` defines what each sample looks like. The `DataLoader` wraps the dataset and handles:
- **Batching:** grouping N samples into one tensor
- **Shuffling:** randomising sample order each epoch (training only)
- **Parallel loading:** using multiple CPU workers to load data while GPU computes

```python
from torch.utils.data import DataLoader

# Training DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,       # 8 emails per batch
    shuffle=True,       # randomise order each epoch (important for training)
    drop_last=True,     # discard final incomplete batch
    num_workers=0       # no parallel loading (simplicity; increase for speed)
)

# Validation DataLoader
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=8,
    shuffle=False,      # no shuffling for validation (order doesn't matter)
    drop_last=False,    # keep all samples for accurate evaluation
    num_workers=0
)

# Test DataLoader
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=8,
    shuffle=False,
    drop_last=False,
    num_workers=0
)
```

### DataLoader parameters explained

| Parameter | Training | Val/Test | Reason |
|-----------|----------|----------|--------|
| `batch_size` | 8 | 8 | Number of samples per forward pass |
| `shuffle` | True | False | Training benefits from random order to prevent learning order patterns; val/test order doesn't matter |
| `drop_last` | True | False | Training: drop incomplete last batch to keep consistent batch size (avoids batch norm issues); Val/Test: keep all samples for accurate metrics |
| `num_workers` | 0 | 0 | 0 = single process loading; increase to 2/4 for faster loading on multi-core machines |

### `drop_last=True` explained in detail

With 1,045 training samples and batch_size=8:
```
1045 ÷ 8 = 130 full batches + 1 partial batch of 5 samples

drop_last=True:  use 130 batches (1040 samples), discard last 5
drop_last=False: use 130 full batches + 1 batch of 5 (131 total)
```

The partial batch is dropped during training because inconsistent batch sizes can cause issues with batch normalisation layers and make loss curves slightly noisy. For validation/test, we want every single sample evaluated, so `drop_last=False`.

---

## 7. Verifying the DataLoaders — Batch Dimensions

### Inspecting a single batch

```python
# Extract one batch from training loader
input_batch, label_batch = next(iter(train_loader))

print(input_batch.shape)   # torch.Size([8, 120])
print(label_batch.shape)   # torch.Size([8])
print(label_batch)         # tensor([1, 0, 1, 0, 0, 1, 1, 0])  ← 0=ham, 1=spam
```

**`input_batch` shape `[8, 120]`:**
- 8 rows = batch_size (8 emails)
- 120 columns = max_length (120 token IDs per email)
- dtype: `torch.long` (integer, required for embedding layers)

**`label_batch` shape `[8]`:**
- 8 values, one per email in the batch
- Each value is 0 (ham) or 1 (spam)
- dtype: `torch.long` (integer class labels)

### Verifying total batch counts

```python
print(f"Training batches:   {len(train_loader)}")    # 130
print(f"Validation batches: {len(val_loader)}")      # 19
print(f"Test batches:       {len(test_loader)}")     # 38
```

**Sanity check — do the numbers add up?**

```
Training:
  1,045 samples ÷ 8 per batch = 130.6 → 130 batches (drop_last removes the 0.6)
  130 batches × 8 samples = 1,040 samples used ✓ (5 discarded by drop_last)

Validation:
  149 samples ÷ 8 = 18.6 → 19 batches (no drop_last: last batch has 5 samples)
  18 × 8 + 5 = 149 ✓

Test:
  300 samples ÷ 8 = 37.5 → 38 batches (last batch has 4 samples)
  37 × 8 + 4 = 300 ✓

Ratio check:
  Test batches (38) ≈ 2× Validation batches (19) ✓
  Because test set (20%) is 2× the size of validation set (10%)
```

### Full 3D shape of each loader

```
train_loader:   130 batches × [8, 120] = effectively [130, 8, 120]
val_loader:      19 batches × [8, 120] = effectively [19, 8, 120]
test_loader:     38 batches × [8, 120] = effectively [38, 8, 120]
```

---

## 8. Key Differences from Pre-training DataLoaders

This is classification fine-tuning, so the data pipeline differs from the pre-training pipeline in an important way:

| Aspect | Pre-training DataLoader | Classification DataLoader |
|--------|------------------------|--------------------------|
| Input | Token IDs of text chunks | Token IDs of emails |
| Target | Token IDs shifted right by 1 | Binary label (0 or 1) |
| Target shape | `[batch, seq_len]` | `[batch]` |
| Task | Predict next token | Predict spam/ham |
| Loss function | Cross-entropy over 50,257 classes | Binary cross-entropy over 2 classes |
| Padding | No padding needed (fixed chunk size) | Padding needed (variable email lengths) |

In pre-training:
```
Input:  [every, effort, moves, you]    → [6109, 3626, 6100, 345]
Target: [effort, moves, you, forward]  → [3626, 6100, 345, 2651]
```

In classification fine-tuning:
```
Input:  [Win, a, free, iPhone, !, <pad>, ..., <pad>]  → [5716, 257, 1479, ..., 50256]
Target: 1  (spam)
```

The target is no longer a token ID sequence — it's a single integer class label.

---

## 9. Complete Pipeline — Data to Model-Ready Batches

```
train.csv  /  validation.csv  /  test.csv
              │
              ▼
    SpamDataset.__init__()
    ├── pd.read_csv() → load all rows
    ├── tokenizer.encode(text) for each email → list of token IDs
    ├── find max_length (120 from training set)
    ├── truncate sequences > max_length
    └── pad sequences < max_length with 50256
              │
              ▼
    SpamDataset.__getitem__(idx)
    ├── encoded: torch.tensor([...], dtype=long)  shape [120]
    └── label:   torch.tensor(0 or 1, dtype=long) scalar
              │
              ▼
    DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    ├── Randomly samples 8 indices
    ├── Calls __getitem__ for each
    ├── Stacks 8 encoded tensors → [8, 120]
    └── Stacks 8 label tensors   → [8]
              │
              ▼
    One batch: (input_tensor[8,120], label_tensor[8])
              │
              ▼
    130 training batches total, yielded one at a time during training
```

---

## 10. What's Coming Next — Initializing the GPT Model for Classification

Now that the data pipeline is complete, the next lectures move to Stage 2:

1. **Initialize the GPT model** with the same architecture we built from scratch
2. **Load pre-trained OpenAI GPT-2 weights** — so the model starts with real language understanding instead of random weights
3. **Modify the output head** — replace `Linear(768 → 50,257)` with `Linear(768 → 2)` for binary classification
4. **Freeze/unfreeze layers** — decide which layers to fine-tune vs keep frozen
5. **Fine-tune and evaluate** — train the modified model on spam data, measure accuracy

The DataLoaders we built today are exactly what gets passed to the training loop in those upcoming lectures. Each iteration of the training loop will call:
```python
for input_batch, label_batch in train_loader:
    # input_batch: [8, 120] token IDs
    # label_batch: [8] spam/ham labels
    output = model(input_batch)   # → [8, 2] logits
    loss = criterion(output, label_batch)
    # ... backprop, optimizer step
```

---

## Key Concepts to Remember

- **Variable length problem** — emails have different token counts; PyTorch tensors require fixed dimensions; must standardise all sequences to the same length
- **Truncation vs padding** — truncation loses information; padding preserves all data by appending special tokens; padding is the correct approach
- **Padding token 50256 (`<|endoftext|>`)** — chosen because GPT-2 pre-training already taught the model this token means "no content here"; avoids confusing real content with padding
- **`SpamDataset` class** — inherits from `torch.utils.data.Dataset`; must implement `__len__` and `__getitem__`; handles tokenization, padding, and truncation
- **`max_length` from training set** — always compute max_length from training data, then pass that same value to val/test datasets to ensure consistent tensor shapes
- **`DataLoader`** — wraps Dataset; handles batching, shuffling, and parallel loading automatically
- **`shuffle=True` for training only** — prevents the model from learning spurious patterns from data order; val/test don't need shuffling
- **`drop_last=True` for training** — discards the final incomplete batch to keep batch sizes consistent; val/test use `drop_last=False` to evaluate all samples
- **Output tensor shapes** — `input_batch: [8, 120]` (batch × sequence), `label_batch: [8]` (batch only, not a sequence)
- **Key difference from pre-training** — pre-training targets are token IDs shifted right by 1; classification targets are single integer class labels (0 or 1)
- **Batch count verification** — 1045 ÷ 8 = 130 training batches; test (38) ≈ 2× validation (19) because test set is 2× the size

---

*Lecture 32 | LLM from Scratch Series | Next: Initializing GPT for Classification & Loading Pre-trained Weights*
