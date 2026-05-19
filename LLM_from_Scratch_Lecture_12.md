# 📘 Build a Large Language Model from Scratch
## Lecture 12 — The Complete Data Preprocessing Pipeline
> **Source:** Vizuara YouTube Channel | **Instructor:** Dr. Raj Dandekar
> **Video:** https://youtu.be/mk-6cFebjis
> **Series:** Stage 1 — Data Preparation & Sampling

---

## 🗺️ What This Lecture Covers

This is a **mega lecture** — the complete end-to-end data preprocessing pipeline in one place:

1. The 4-Step Pipeline Overview
2. Step 1 — Tokenization (Word-Based from scratch)
3. Vocabulary Construction & Token IDs
4. The Tokenizer Class (encode + decode)
5. Special Context Tokens (`<|endoftext|>` and `<|unk|>`)
6. Problems with Word-Based Tokenization
7. Character-Based Tokenization
8. Subword-Based Tokenization & BPE
9. BPE with `tiktoken` (GPT-2 tokenizer in code)
10. Input-Target Pairs & Context Size
11. DataLoader & Sliding Window
12. Step 2 — Token Embeddings
13. Step 3 — Positional Embeddings
14. Step 4 — Input Embeddings (Token + Positional)

---

## The Complete Pipeline at a Glance

```
Raw Text (book, PDF, articles...)
        ↓
STEP 1: TOKENIZATION
    Raw Text → Tokens → Token IDs
        ↓
STEP 2: TOKEN EMBEDDINGS
    Token IDs → Semantic Vectors [vocab_size × vector_dim]
        ↓
STEP 3: POSITIONAL EMBEDDINGS
    Position → Position Vectors [context_size × vector_dim]
        ↓
STEP 4: INPUT EMBEDDINGS
    Token Embeddings + Positional Embeddings
        ↓
    INPUT TO LLM TRAINING ✅
```

> 💡 **Input Embedding = Token Embedding + Positional Embedding**

---

## Step 1 — Tokenization

### What is Tokenization?
> Breaking raw text into smaller units called **tokens**, then converting each token to a numerical **token ID**.

### Dataset Used: "The Verdict" (1908 book)
```python
with open("the-verdict.txt", "r") as f:
    raw_text = f.read()

print(len(raw_text))         # 20479 characters
print(raw_text[:100])        # "I had always thought Jack Gisburn rather a cheap genius..."
```

---

### Word-Based Tokenization from Scratch

#### Naive Split (Problem)
```python
import re
tokens = re.split(r'\s+', text)
# Result: ["hello,", "world.", "this"] ← comma attached to "hello"!
```
Problem: punctuation is stuck to words — we want them as **separate tokens**.

#### Better Split (Fix)
```python
# Split on whitespace AND punctuation
tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)

# Remove whitespace tokens
tokens = [item.strip() for item in tokens if item.strip()]
```

Result:
```
["hello", ",", "world", ".", "this", "is", "a", "test"]
✅ Punctuation = separate tokens
✅ No whitespace tokens
```

#### Applying to Full Dataset
```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(len(preprocessed))   # 4690 tokens
print(preprocessed[:30])   # ['I', 'had', 'always', 'thought', 'Jack', ...]
```

---

### Vocabulary Construction

> **Vocabulary** = A dictionary mapping every unique token → a unique integer (Token ID)

```python
# Step 1: Get unique tokens, sorted alphabetically
all_tokens = sorted(set(preprocessed))

# Step 2: Assign integer IDs via enumerate
vocab = {token: integer for integer, token in enumerate(all_tokens)}

print(len(vocab))   # 1130 unique tokens

# Example entries:
# {'!': 0, '"': 1, ..., 'for': 35, 'has': 47, ...}
```

- Tokens sorted alphabetically → special characters get lowest IDs (0, 1, 2...)
- Each unique token gets exactly one unique integer

---

### The Tokenizer Class (encode + decode)

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab                          # token → ID
        self.int_to_str = {i: s for s, i in vocab.items()}  # ID → token (reverse)

    def encode(self, text):
        # Split text into tokens
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Convert tokens to IDs
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        # Convert IDs back to tokens
        text = " ".join([self.int_to_str[i] for i in ids])
        # Fix spaces before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

#### Testing
```python
tokenizer = SimpleTokenizerV1(vocab)

text = "It was the last he painted, you know, Mrs. Gisburn said."
ids = tokenizer.encode(text)
print(ids)        # [58, 290, 3, 702, 56, ...]

decoded = tokenizer.decode(ids)
print(decoded)    # "It was the last he painted, you know, Mrs. Gisburn said."
```

---

### Special Context Tokens

#### The Problem — Out-of-Vocabulary (OOV) Words
```python
text = "Hello, do you like tea?"
ids = tokenizer.encode(text)   # ❌ KeyError! "Hello" not in vocabulary
```

"Hello" never appeared in "The Verdict" — so it's not in our vocabulary → crash!

#### The Solution — Add Special Tokens

```python
# Augment vocabulary with 2 special tokens
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab))   # 1132 (1130 + 2)

# Last two entries:
# "<|endoftext|>": 1130
# "<|unk|>": 1131
```

| Special Token | Purpose |
|--------------|---------|
| `<|unk|>` | Replaces any word not in the vocabulary |
| `<|endoftext|>` | Marks the boundary between two separate documents |

#### Why `<|endoftext|>` is Important
When training GPT on multiple sources (news articles + Reddit posts + books):
```
[Article 1 text...] <|endoftext|> [Reddit post...] <|endoftext|> [Book chapter...]
```
This tells the model: "This document ended, a new one begins."

#### Updated Tokenizer V2
```python
class SimpleTokenizerV2:
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Replace unknown words with <|unk|>
        preprocessed = [item if item in self.str_to_int else "<|unk|>"
                       for item in preprocessed]
        return [self.str_to_int[s] for s in preprocessed]
```

#### Testing V2
```python
tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

# Combine with end-of-text separator
combined = " <|endoftext|> ".join([text1, text2])

ids = tokenizer.encode(combined)
# "Hello" → <|unk|> (ID 1131), "palace" → <|unk|> (ID 1131)
# ✅ No crash!

decoded = tokenizer.decode(ids)
# "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."
```

> 💡 GPT models do **not** use `<|unk|>` — BPE handles unknown words by breaking them into subwords. But GPT **does** use `<|endoftext|>`.

---

## Three Types of Tokenization — Comparison

### Type 1 — Word-Based ❌
- Tokens = individual words
- **Problems:**
  - OOV errors (unknown words crash the tokenizer)
  - "boy" and "boys" get completely different tokens — root word meaning lost
  - Huge vocabulary needed (~600,000–1M words in English)

### Type 2 — Character-Based ❌
- Tokens = individual characters (a, b, c, ...)
- **Advantages:** tiny vocabulary (~256 chars), no OOV errors
- **Problems:**
  - All word meaning is destroyed (h, o, b, b, y means nothing)
  - Much longer sequences (5 tokens per 5-letter word)

### Type 3 — Subword-Based / BPE ✅ (Used by GPT)
**Rules:**
1. Frequently used words → kept as single tokens
2. Rare words → split into smaller meaningful subwords

**Example:**
```
"boy"   (frequent) → ["boy"]       ← kept whole
"boys"  (rare)     → ["boy", "s"]  ← split, but root "boy" preserved
"finest"           → ["fin", "est"] ← root meaning preserved
```

---

## BPE Algorithm — How It Works

### Original Purpose (1994)
BPE was a **data compression algorithm** — replace frequent byte pairs with a shorter symbol.

### Applied to Text Tokenization

```
Starting vocabulary (character level):
old → o l d </w>
older → o l d e r </w>
finest → f i n e s t </w>
lowest → l o w e s t </w>

Iteration 1: Most frequent pair = e + s (appears 13 times)
  → Merge: "es" becomes a new token
  finest → f i n es t </w>
  lowest → l o w es t </w>

Iteration 2: Most frequent pair = es + t
  → Merge: "est" becomes a new token
  finest → f i n est </w>
  lowest → l o w est </w>

Iteration 3: o + l appears 10 times
  → Merge: "ol" becomes a token

Iteration 4: ol + d appears 10 times
  → Merge: "old" becomes a token
  old   → old </w>
  older → old e r </w>
```

**Final tokens learned:**
```
["est", "old", "er", "ol", "f", "i", "n", "w", ...]
```

BPE learned that **"est"** (superlative suffix) and **"old"** (root word) are meaningful units — automatically, just from frequency!

---

## BPE with tiktoken (GPT-2 Tokenizer)

```python
import tiktoken

# GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Test with unknown/rare words
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of some unknown place."
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# Decode back
decoded = tokenizer.decode(ids)
print(decoded)   # ✅ Perfect round-trip, no errors!
```

### GPT-2 Vocabulary Sizes by Model Version

| GPT Version | Vocabulary Size |
|------------|----------------|
| GPT-2 | ~50,257 |
| GPT-3 | ~50,257 |
| GPT-4 | Larger (grows with each model) |

### 3 Advantages of BPE over Word-Based Tokenization

| Advantage | Explanation |
|-----------|------------|
| ✅ Efficient vocabulary | ~50K tokens vs 600K+ for word-based |
| ✅ Preserves root meaning | "boy", "boys", "boyfriend" share "boy" token |
| ✅ No OOV errors | Unknown words decomposed into subword units automatically |

---

## Input-Target Pairs & Context Size

### What is Context Size?
> **Context size** = The maximum number of tokens the LLM sees at one time to predict the next token.

### Creating Input-Target Pairs

```
Context size = 4
Token sequence: ["one", "word", "at", "a", "time", ...]
Token IDs:      [1,     2,      3,    4,   5,     ...]

Input-Target pairs:
Input: [1]           → Target: 2
Input: [1, 2]        → Target: 3
Input: [1, 2, 3]     → Target: 4
Input: [1, 2, 3, 4]  → Target: 5
```

One input sequence of 4 tokens = **4 prediction tasks**!

### As Tensors
```
Input tensor X:   [[1, 2, 3, 4],   ← row 1
                   [2, 3, 4, 5],   ← row 2 (shifted by stride)
                   ...]

Target tensor Y:  [[2, 3, 4, 5],   ← same as X but shifted right by 1
                   [3, 4, 5, 6],
                   ...]
```

---

## DataLoader & Sliding Window

### What is Stride?
> **Stride** = How many positions the window slides to create the next input.

```
Token sequence: [in, the, heart, of, the, city, stood, the, old, library]

Stride = 1 (overlapping):
  Input 1: [in, the, heart, of]
  Input 2: [the, heart, of, the]   ← overlaps heavily with Input 1
  Input 3: [heart, of, the, city]

Stride = 4 (non-overlapping):
  Input 1: [in, the, heart, of]
  Input 2: [the, city, stood, the]   ← no overlap
  Input 3: [old, library, ...]
```

### DataLoader Code

```python
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i : i + max_length]
            target_chunk = token_ids[i+1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return self.input_ids[idx], self.target_ids[idx]

# Create DataLoader
dataloader = DataLoader(GPTDatasetV1(raw_text, tokenizer, max_length=4, stride=4),
                        batch_size=8, shuffle=False)

inputs, targets = next(iter(dataloader))
print(inputs.shape)    # torch.Size([8, 4])
```

---

## Step 2 — Token Embeddings

### Why Not Just Use Token IDs?
- "cat" = ID 34, "kitten" = ID -13 → no relationship encoded
- One-hot encoding also fails — no semantic meaning captured

### The Embedding Weight Matrix

```python
vocab_size  = 50257   # GPT-2 BPE vocabulary
output_dim  = 256     # vector dimension per token

token_embedding_layer = nn.Embedding(vocab_size, output_dim)
# Creates matrix: [50257, 256] — randomly initialized
```

### Getting Token Embeddings for a Batch

```python
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)   # torch.Size([8, 4, 256])
```

**Dimensions:**
```
8   = batch size
4   = context size (tokens per sequence)
256 = vector dimension per token
```

---

## Step 3 — Positional Embeddings

### Why Are They Needed?
Without positional embeddings, "cat" at position 2 = "cat" at position 5 → same vector!

### The Positional Embedding Layer

```python
context_size = 4   # only 4 positions needed

positional_embedding_layer = nn.Embedding(context_size, output_dim)
# Creates matrix: [4, 256]

positions = torch.arange(context_size)   # [0, 1, 2, 3]
positional_embeddings = positional_embedding_layer(positions)
print(positional_embeddings.shape)   # torch.Size([4, 256])
```

**Why only 4 rows?** We only need one vector per position (1st, 2nd, 3rd, 4th token). The same 4 positional vectors apply to every sequence in the batch.

### Absolute vs Relative Positional Encoding

| | Absolute | Relative |
|--|---------|---------|
| What's encoded | Exact position number | Distance between tokens |
| Used by GPT | ✅ Yes | ❌ No |
| Original Transformer | ✅ Yes (sinusoidal formula) | ❌ No |
| Best for | Sequence generation | Long sequences |
| Values | Learned during training (GPT) | Learned during training |

---

## Step 4 — Input Embeddings (Final Step)

```python
input_embeddings = token_embeddings + positional_embeddings
print(input_embeddings.shape)   # torch.Size([8, 4, 256])
```

### Broadcasting in Action
```
token_embeddings:      [8, 4, 256]
positional_embeddings: [4, 256]   → broadcast → [8, 4, 256]

Python duplicates the [4, 256] positional matrix 8 times,
then adds it element-wise to each of the 8 sequences.
```

### What Gets Optimized During Training?
Both embedding matrices are randomly initialized and **optimized via backpropagation**:

| Matrix | Shape | Learns |
|--------|-------|--------|
| Token Embedding Matrix | [50257, 256] | Semantic meaning between tokens |
| Positional Embedding Matrix | [4, 256] | Meaning of each position |

---

## 🔑 Key Terms to Know

| Term | Simple Meaning |
|------|---------------|
| **Tokenization** | Splitting raw text into tokens (word/char/subword) |
| **Token ID** | Unique integer assigned to each token |
| **Vocabulary** | Dictionary mapping every unique token → integer |
| **OOV (Out-of-Vocabulary)** | Word not in the vocabulary → causes crash in word-based tokenizer |
| **`<\|unk\|>`** | Special token replacing unknown words |
| **`<\|endoftext\|>`** | Marks boundary between two separate documents |
| **BPE** | Byte Pair Encoding — subword tokenizer used by GPT |
| **tiktoken** | OpenAI's Python library for BPE tokenization |
| **Context Size** | Max number of tokens the LLM sees at once |
| **Stride** | How many positions the sliding window shifts per step |
| **DataLoader** | PyTorch tool for batching and feeding data efficiently |
| **Token Embedding** | Converting token IDs to semantic vectors |
| **Embedding Weight Matrix** | Lookup table of [vocab_size × vector_dim] learned vectors |
| **Positional Embedding** | Vector encoding the position of each token in the sequence |
| **Input Embedding** | Token Embedding + Positional Embedding = final LLM input |
| **Broadcasting** | Python automatically expands smaller tensors to match larger ones for addition |
| **Absolute Positional Encoding** | One unique vector per exact position number |
| **Relative Positional Encoding** | Encodes distance/gap between tokens |

---

## ⚡ Quick Summary

1. **4-step pipeline:** Tokenization → Token Embeddings → Positional Embeddings → Input Embeddings
2. **Word-based tokenizer** built from scratch: split on whitespace + punctuation, sort, assign IDs
3. **Vocabulary** = dictionary of {token: integer} — size 1130 for "The Verdict"
4. **Tokenizer class** has `encode()` (text→IDs) and `decode()` (IDs→text) methods
5. **`<|unk|>`** handles unknown words; **`<|endoftext|>`** separates documents
6. **Word-based problems:** OOV errors, no root word meaning, huge vocabulary
7. **Character-based problems:** meaning destroyed, very long sequences
8. **BPE (subword)** = best of both worlds — used by GPT-2/3/4 via `tiktoken`
9. **Context size** = how many tokens LLM sees at once; more = better but costlier
10. **Sliding window + stride** creates input-target pairs from raw token sequence
11. **Token embeddings:** [8, 4, 256] — each token ID → 256-dim semantic vector
12. **Positional embeddings:** [4, 256] — one vector per position
13. **Input embeddings:** token + positional = [8, 4, 256] — final LLM input
14. Both embedding matrices **start random** and are **optimized during training**

---

## 🧠 Key Takeaways to Remember

- 🔹 **Full pipeline:** Raw Text → Tokens → IDs → Token Embeddings + Positional Embeddings → Input Embeddings
- 🔹 **BPE = best tokenizer** for LLMs — efficient vocab, retains root meaning, handles OOV
- 🔹 **GPT-2 vocab = 50,257** tokens via BPE; `tiktoken` library implements this
- 🔹 **Context size** = window of tokens the LLM sees; **stride** = how far window moves
- 🔹 **Target tensor = input tensor shifted right by 1** — one window → multiple prediction tasks
- 🔹 **Token embedding [8,4,256]** = semantic meaning of words
- 🔹 **Positional embedding [4,256]** = position of words (same 4 vectors applied to all 8 sequences via broadcasting)
- 🔹 **Input embedding = token + positional** — captures BOTH meaning AND position
- 🔹 **Everything is randomly initialized** and learned through backpropagation during training

---

## 📌 What's Coming Next

- **Attention Mechanism** — the heart of the Transformer
- **Self-Attention from scratch** — Keys, Queries, Values
- **Multi-Head Attention** — why multiple attention heads are needed
- Coding the full attention block in Python

---

*Notes prepared from Vizuara YouTube Series: "Build a Large Language Model from Scratch"*
*GitHub-ready markdown — push directly to your notes repo* 🚀
