# 📘 Build a Large Language Model from Scratch
## Lecture 9 — Data Pre-processing: Input-Target Pairs & DataLoaders
> **Source:** Vizuara YouTube Channel | **Instructor:** Dr. Raj Dandekar
> **Video:** https://youtu.be/iQZFH8dr2yI
> **Series:** Stage 1 — Data Preparation & Sampling

---

## 🗺️ What This Lecture Covers

1. Input-Target Pairs — how LLMs are trained
2. Context Size — how many tokens the model sees
3. Implementing DataLoaders in PyTorch
4. Sliding Window Approach
5. Key Hyperparameters — Stride, Batch Size, Number of Workers

---

## 1. Input-Target Pairs

### How LLMs Are Trained

LLMs use **auto-regressive, self-supervised learning** — the model learns to predict the **next token** in a sequence, using the sentence itself as both input and label (no manual labeling needed).

### The Core Idea

> Given a sequence of tokens, the **input** is everything up to the last word, and the **target** is the next word.

### Example

```
Full sentence: "LLMs learn to predict"

Input:  "LLMs learn to"      ← fed into the model
Target: "predict"            ← what the model must predict
```

### Multiple Training Pairs from One Sentence

A single sentence generates **multiple input-target pairs** — one for each position:

```
Sentence: "LLMs learn to predict next tokens"

Pair 1:  Input = ["LLMs"]                        → Target = "learn"
Pair 2:  Input = ["LLMs", "learn"]               → Target = "to"
Pair 3:  Input = ["LLMs", "learn", "to"]         → Target = "predict"
Pair 4:  Input = ["LLMs", "learn", "to", "predict"] → Target = "next"
Pair 5:  Input = ["LLMs", "learn", "to", "predict", "next"] → Target = "tokens"
```

> 💡 This is why pre-training is called **self-supervised** — the sentence structure itself creates the labels. No human labeling needed!

### Masking
- The model is **masked** — it **cannot see** any tokens that come after the target
- When predicting word at position N, words at positions N+1, N+2, ... are hidden
- This forces the model to genuinely predict rather than just copy future tokens
- This masking is implemented via the **Masked Multi-Head Attention** in the Transformer

### Why It's Called Auto-Regressive
- The predicted output from one step becomes the **input for the next step**
- The model builds its answer one token at a time, always feeding back its previous output

```
Step 1: Input = "LLMs learn to"  → Predict: "predict"
Step 2: Input = "LLMs learn to predict"  → Predict: "next"
Step 3: Input = "LLMs learn to predict next"  → Predict: "tokens"
```

---

## 2. Context Size

### What Is Context Size?

> **Context size** (also called **context length** or **context window**) = The maximum number of tokens the model can receive as input at one time.

- Defines how far back the model can "look" when predicting the next token
- A larger context = model can use more past information = better predictions
- But larger context = more memory and compute needed

### Toy Example (Context Size = 4)

```
Token sequence: [101, 204, 39, 567, 88, 92, 403]

Window 1: Input = [101, 204, 39, 567]  → Target = 88
Window 2: Input = [204, 39, 567, 88]   → Target = 92
Window 3: Input = [39, 567, 88, 92]    → Target = 403
```

### Context Size in Production Models

| Model | Context Size |
|-------|-------------|
| Toy example (this lecture) | 4 tokens |
| GPT-2 | 1,024 tokens |
| GPT-3 | 2,048 tokens |
| GPT-4 | 8,192 – 32,768 tokens |
| Claude (some versions) | 100,000+ tokens |

> 💡 GPT-2/3 use context lengths of **256+ tokens** at minimum to capture meaningful dependencies across sentences.

### Why Context Size Matters
- Too small → model can't remember earlier parts of the text → poor predictions
- Too large → exponentially more computation (attention is O(n²) with sequence length)
- Choosing the right context size = critical design decision

---

## 3. Implementing DataLoaders in PyTorch

### Why Do We Need a DataLoader?

- Training data = **billions of tokens**
- You can't load all data into memory at once
- **DataLoader** feeds data to the model in manageable **batches**
- PyTorch's `Dataset` + `DataLoader` framework handles this efficiently

### The `GPTDatasetV1` Class

```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Sliding window to create input-target pairs
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i : i + max_length]
            target_chunk = token_ids[i+1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

### Key Methods Explained

| Method | Purpose |
|--------|---------|
| `__init__` | Initializes the dataset — tokenizes text and creates all input-target pairs |
| `__len__` | Returns total number of samples in the dataset |
| `__getitem__` | Fetches a specific input-target pair by index — used by DataLoader |

---

## 4. The Sliding Window Approach

### What Is It?

> A **sliding window** moves across the token sequence step by step, creating input-target pairs at each position.

### How It Works

```
Token sequence: [A, B, C, D, E, F, G]
Context size = 4, Stride = 1

Window 1: Input=[A,B,C,D]  Target=[B,C,D,E]
Window 2: Input=[B,C,D,E]  Target=[C,D,E,F]
Window 3: Input=[C,D,E,F]  Target=[D,E,F,G]
```

### Input and Target Tensors

> The **target tensor** is simply the **input tensor shifted by one position to the right**

```
Input:  [token_0, token_1, token_2, token_3]
Target: [token_1, token_2, token_3, token_4]
                                            ↑
                                    (shifted by 1)
```

This means for every position in the input, the corresponding target is the very next token — creating multiple prediction tasks from a single window.

### Visual Example

```
Input tensor:  [ 101,  204,   39,  567 ]
Target tensor: [ 204,   39,  567,   88 ]

Training tasks from this one window:
  101  →  204
  204  →   39
   39  →  567
  567  →   88
```

---

## 5. Key Hyperparameters

### Hyperparameter 1 — Stride

> **Stride** = How many positions the window moves forward after each sample

#### Stride = 1 (Overlapping Windows)
```
Window 1: [A, B, C, D] → E
Window 2: [B, C, D, E] → F   ← overlaps heavily with Window 1
Window 3: [C, D, E, F] → G
```
- High data overlap between batches
- ⚠️ Risk of **overfitting** — model memorizes repeated sequences
- More training samples generated

#### Stride = Context Size (Non-Overlapping Windows)
```
Context size = 4, Stride = 4
Window 1: [A, B, C, D] → E
Window 2: [E, F, G, H] → I   ← no overlap with Window 1
Window 3: [I, J, K, L] → M
```
- Zero overlap between batches
- ✅ Reduces overfitting
- Fewer training samples generated

#### Rule of Thumb

| Stride Value | Overlap | Overfitting Risk | Samples Generated |
|-------------|---------|-----------------|------------------|
| stride = 1 | Maximum | High | Most |
| stride = context_size | None | Low | Fewest |
| stride in between | Partial | Medium | Medium |

> 💡 **Best practice for pre-training:** Set stride = context size to prevent overlapping data and reduce overfitting.

---

### Hyperparameter 2 — Batch Size

> **Batch size** = How many input-target samples are processed **in parallel** before the model updates its weights

```python
dataloader = DataLoader(
    dataset,
    batch_size=4,    # process 4 samples at once
    shuffle=True
)
```

#### How Batch Size Affects Training

| Batch Size | Gradient Stability | Memory Usage | Training Speed |
|------------|-------------------|-------------|----------------|
| Small (e.g. 1) | Noisy, unstable | Low | Slow |
| Medium (e.g. 32) | Balanced | Medium | Medium |
| Large (e.g. 512) | Stable, smooth | High | Fast (if GPU fits) |

- **Larger batch** → averages gradients over more samples → **more stable weight updates**
- **Larger batch** → needs more GPU/CPU memory
- **Smaller batch** → noisier updates but sometimes helps escape local minima

> 💡 Common batch sizes used in practice: **32, 64, 128, 256** — chosen based on available GPU memory

---

### Hyperparameter 3 — Number of Workers

> **num_workers** = How many CPU threads work in parallel to load and pre-process data

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4    # use 4 CPU threads in parallel
)
```

- While the GPU is training on one batch, workers are **pre-loading the next batch** in the background
- This eliminates the bottleneck of waiting for data to load
- Setting `num_workers=0` → single-threaded, slower
- Setting `num_workers=4` → 4 threads work in parallel → significantly faster data throughput

---

## 🔧 Putting It All Together — Full DataLoader Setup

```python
import tiktoken
from torch.utils.data import DataLoader

def create_dataloader(txt, batch_size=4, max_length=256,
                      stride=128, shuffle=True, num_workers=0):

    # Initialize GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader

# Usage
with open("the-verdict.txt", "r") as f:
    raw_text = f.read()

dataloader = create_dataloader(
    raw_text,
    batch_size=4,
    max_length=256,
    stride=256,      # stride = context size → no overlap
    num_workers=0
)

# Get first batch
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print(inputs.shape)   # torch.Size([4, 256])  → 4 samples, 256 tokens each
print(targets.shape)  # torch.Size([4, 256])  → same shape, shifted by 1
```

---

## 🔑 Key Terms to Know

| Term | Simple Meaning |
|------|---------------|
| **Auto-regressive** | Previous output becomes input for next prediction |
| **Self-supervised** | Labels come from the data itself — no manual labeling |
| **Input-Target Pair** | (input tokens, next token to predict) — the training unit |
| **Masking** | Hiding future tokens so the model can't cheat by looking ahead |
| **Context Size / Context Window** | Max number of tokens the model sees at once |
| **Sliding Window** | Moving a fixed-size window across the token sequence to create pairs |
| **Stride** | How many positions the window shifts between samples |
| **Batch Size** | Number of samples processed in parallel before weight update |
| **num_workers** | Number of CPU threads loading data in parallel |
| **DataLoader** | PyTorch utility that feeds batches of data to the model |
| **Dataset** | PyTorch class that defines how to load and access individual samples |
| **GPTDatasetV1** | Our custom Dataset class for LLM training data |
| **`__getitem__`** | Method that lets DataLoader fetch a sample by index |
| **Overfitting** | Model memorizes training data instead of learning general patterns |
| **Gradient** | The direction and magnitude of weight update during training |

---

## ⚡ Quick Summary

1. LLMs are trained on **input-target pairs** — input = N tokens, target = N+1th token
2. Training is **auto-regressive** (output feeds back as input) and **self-supervised** (no manual labels)
3. The model is **masked** — it cannot see future tokens beyond the target
4. **Context size** = how many tokens the model sees at once (toy=4, GPT-3=2048)
5. **Sliding window** creates input-target pairs — target tensor = input tensor shifted right by 1
6. `GPTDatasetV1` class uses PyTorch `Dataset` with `__init__`, `__len__`, `__getitem__`
7. **Stride** controls window overlap — stride = context size → no overlap → less overfitting
8. **Batch size** controls how many samples are processed in parallel — larger = more stable gradients but more memory
9. **num_workers** enables parallel data loading → faster training throughput

---

## 🧠 Key Takeaways to Remember

- 🔹 **Input = tokens[i : i+context]** | **Target = tokens[i+1 : i+context+1]** (shifted by 1)
- 🔹 **Self-supervised** = sentence labels itself, no human annotation needed
- 🔹 **Context size** = how far back the model looks — bigger = smarter but costlier
- 🔹 **Sliding window** = the mechanism to generate training pairs from raw text
- 🔹 **Stride = context size** → best practice to avoid overfitting
- 🔹 **Batch size** = parallel processing — larger is more stable but needs more memory
- 🔹 **num_workers** = parallel data loading on CPU while GPU trains
- 🔹 `GPTDatasetV1` + `DataLoader` = the complete data pipeline for LLM training

---

## 📌 What's Coming Next

- **Token Embeddings** — converting token IDs into dense vectors
- **Positional Embeddings** — encoding the order/position of tokens
- These two combined = the full input representation before entering the Transformer

---

*Notes prepared from Vizuara YouTube Series: "Build a Large Language Model from Scratch"*
*GitHub-ready markdown — push directly to your notes repo* 🚀
