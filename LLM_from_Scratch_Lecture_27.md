# LLM Training & Validation Loss — Complete Notes
> **Series:** Build LLMs from Scratch | **Stage 2 — Pre-training**  
> **Lecture Goal:** Evaluate LLM performance on a real dataset using Training + Validation Loss

---

## 1. Where We Are in the LLM Building Pipeline

Building an LLM from scratch has two major stages:

**Stage 1 (Completed):**
- Data preparation and tokenization
- Attention mechanism (self-attention, multi-head attention)
- Full GPT architecture built from scratch

**Stage 2 (Now — Pre-training), progressing in this order:**
```
Text Generation
       ↓
Cross-Entropy Loss on 2 toy sentences  ← previous lecture
       ↓
Training + Validation Loss on real dataset  ← THIS LECTURE
       ↓
Backpropagation + Full Training Loop  ← next lecture
```

The previous lecture used just two inputs — *"every effort moves"* and *"I really like"* — to explain how cross-entropy loss works conceptually. This lecture applies that same logic to a real storybook and computes meaningful loss numbers.

---

## 2. The Dataset — "The Verdict" by Edith Wharton (1906)

### Why this specific book?
- Publicly available — no copyright issues, downloadable from a direct link
- Very short — runs in under 30 seconds on a regular laptop CPU
- Demonstrates all the concepts clearly without requiring expensive hardware
- The same code works on any dataset — Harry Potter, Wikipedia, news articles — just swap the file

### Dataset statistics
| Property | Value |
|---|---|
| Total characters | ~20,479 |
| Total tokens after BPE encoding | ~5,145 |

### Real-world scale to understand why we use a small dataset

| Model | Dataset Size | GPU Hours | Estimated Cost |
|---|---|---|---|
| LLaMA 2 (7B parameters) | 2 trillion tokens | 84,320 GPU hours | ~$700,000 |
| This lecture | ~5,145 tokens | < 1 | $0 |

The logic, code structure, and concepts are **completely identical** regardless of dataset size. We use this short story purely so you can learn the ideas and run everything yourself in minutes.

---

## 3. Tokenization — Text to Numbers

### Why tokenize?
Neural networks only understand numbers. Raw text (letters, words, spaces, punctuation) must be converted into integer IDs before the model can process it.

### Byte Pair Encoding (BPE) — what it is
BPE is a **subword-level tokenization** scheme. This means:
- One word ≠ always one token
- Single characters can be tokens (e.g., just "t")
- Common letter pairs can be tokens (e.g., "th")
- Very common words are a single token (e.g., "the")
- Rare or unknown words get split into smaller known pieces

This makes the vocabulary manageable while still handling any text, including made-up words or foreign words.

### tiktoken — the library used
`tiktoken` is OpenAI's tokenizer library, which implements GPT-2's BPE encoder. Using it ensures our tokenization is consistent with how GPT-2 was built.

```
20,479 characters  →  tiktoken BPE  →  5,145 token IDs
```

The compression happens because common words map to single tokens, reducing total count.

### Vocabulary size = 50,257
Every time the model makes a prediction, it produces a score for each of these 50,257 possible next tokens. This number is critical — it appears in every tensor dimension related to model output.

> Tokenization was covered deeply in previous lectures. Here it is used as a tool, not re-explained from scratch.

---

## 4. Train / Validation Split — The Most Important Evaluation Principle

### The core reason for splitting

**Training loss alone is not a reliable measure of how good your model is.**

A model could theoretically memorise every sentence in the training data perfectly. It would score a training loss of near zero. But if you asked it to complete a new sentence it had never seen, it might produce complete nonsense. This is called **overfitting**.

**Validation loss** measures performance on text the model has never seen during training. This is the number that actually matters. If validation loss is much higher than training loss, the model has overfitted — it memorised instead of learning.

### How the split works — 90/10 Sequential Split

```
Full dataset (5,145 tokens):
|←————————————— 90% Training (4,608 tokens) ————————————→|← 10% Validation (512 tokens) →|
```

**Why sequential (not random)?**  
Text is a sequence. Randomly shuffling individual tokens would break sentences and destroy meaning. We preserve the natural flow by taking the first 90% for training and the last 10% for validation.

**Why 90/10?**  
This is a standard choice in language modelling. 90% gives the model enough data to learn from. 10% gives enough validation data for a reliable signal on generalisation.

---

## 5. Constructing Input–Target Pairs (Most Important Concept in the Lecture ⭐⭐⭐)

This is unique to language models and essential to understand deeply.

### Why this is different from all other ML problems

**Standard machine learning (e.g., image classification):**
- Input = image of a dog
- Label = "dog" — manually created by a human
- Inputs and outputs are completely separate, pre-labelled

**Large language models:**
- There are NO pre-labelled outputs
- No human needs to annotate anything
- The text itself generates both input AND target automatically
- This is called **self-supervised learning**

### The fundamental rule — autoregressive prediction

An LLM is trained to **predict the next token** given all previous tokens. Because the next token is always present in the text, we can generate unlimited training pairs with zero manual labelling.

**The rule: Target Y = Input X shifted right by exactly 1 token**

### Two key parameters that control how pairs are built

**Parameter 1: Context Length (max_length)**
How many tokens the model sees at one time before predicting.
- Used in the whiteboard demonstration: **4 tokens** (to make it easy to visualise)
- Used in the actual code: **256 tokens**
- GPT-2 (original): **1024 tokens**
- Modern models (GPT-4, Claude): **tens of thousands of tokens**

Larger context = the model understands more surrounding information before predicting. But it also requires exponentially more compute because attention mechanisms scale as O(n²) with sequence length.

**Parameter 2: Stride**
How many positions to move forward before starting the next input window.
- `stride = 1`: maximum overlap — every possible window is used as an input
- `stride = context_length`: no overlap at all — used in GPT-style training
- GPT standard uses `stride = context_length` so no token is skipped but no redundant overlapping windows are created either

### Building pairs visually (context = 4 for illustration)

Text: `I  had  always  thought  Jack  Gisburn  rather  a  cheap  genius`

**stride = context_length = 4 (GPT style):**
```
X1 = [I, had, always, thought]         →  Y1 = [had, always, thought, Jack]
X2 = [Jack, Gisburn, rather, a]        →  Y2 = [Gisburn, rather, a, cheap]
X3 = [cheap, genius, --, do]           →  Y3 = [genius, --, do, ...]
```
- X2 starts exactly where X1 ended — no gap, no overlap
- Every single token in the dataset is covered exactly once

**stride = 1 (maximum overlap):**
```
X1 = [I, had, always, thought]         →  Y1 = [had, always, thought, Jack]
X2 = [had, always, thought, Jack]      →  Y2 = [always, thought, Jack, Gisburn]
X3 = [always, thought, Jack, Gisburn]  →  Y3 = [thought, Jack, Gisburn, rather]
```
Many more training pairs but very redundant.

### One input contains MULTIPLE prediction tasks

This is a critical but often misunderstood point.

A single input `[I, had, always, thought]` does not result in ONE prediction. It results in **4 predictions happening simultaneously in a single forward pass**:

| Tokens model has seen | Target to predict |
|---|---|
| `I` | `had` |
| `I, had` | `always` |
| `I, had, always` | `thought` |
| `I, had, always, thought` | `Jack` |

This is possible because the attention mechanism uses masking — when predicting the token at position 2, the model cannot see positions 3 and 4. Each position only attends to itself and earlier positions.

This makes every forward pass extremely efficient — 256 predictions happen for every single input of context_length=256.

### The full input and target tensors

After processing the whole training dataset:
- **X tensor**: Each row = one input window of 256 tokens
- **Y tensor**: Each row = same window shifted right by 1

These tensors are then batched and fed to the model.

---

## 6. The DataLoader — Batching and Organising Data

### What a DataLoader does
The DataLoader takes the full set of (X, Y) pairs and:
1. Groups them into fixed-size batches
2. Optionally shuffles the order of batches
3. Streams batches one at a time during training

Without a DataLoader, you would have to manually manage batch creation, shuffling, and iteration — PyTorch's DataLoader handles all of this automatically.

### Batch size = 2 in this lecture
With `batch_size=2` and `context_length=256`, each batch has:
```
X shape = [2, 256]     → 2 input sequences, each 256 tokens
Y shape = [2, 256]     → 2 target sequences, each 256 tokens
```

Visualising one batch with context=4 for clarity:
```
X batch = [ [I, had, always, thought],       ← sample 1
            [Jack, Gisburn, rather, a] ]     ← sample 2

Y batch = [ [had, always, thought, Jack],    ← target 1
            [Gisburn, rather, a, cheap] ]    ← target 2
```

Real industry batch sizes: LLaMA 2 used batch_size=1024. We use 2 because we want fast laptop execution.

### DataLoader settings explained in depth

**`shuffle=True` (used for training, NOT validation)**  
Before each training epoch, the order in which batches are presented is randomised. Why? Because if the model always sees batches in the same order, it can learn the order itself rather than the language. Randomising forces it to generalise. Validation is never shuffled — we want consistent, reproducible evaluation results.

**`drop_last=True`**  
After creating all batches, if there are leftover samples that don't fill a complete batch, they are discarded. For example, with 19 samples and batch_size=2, you get 9 complete batches and 1 leftover sample — that 1 sample gets dropped. An incomplete batch can cause subtle bugs in loss normalisation.

**`num_workers=0`**  
Specifies how many CPU processes load data in parallel. Zero means the main process handles loading — acceptable for a small dataset. For large datasets, increasing this speeds up data loading significantly.

### Resulting DataLoader shapes confirmed in code
```
Training loader:   9 batches,  each [2, 256]  → 9 × 2 × 256 = 4,608 training tokens
Validation loader: 1 batch,    each [2, 256]  → 1 × 2 × 256 = 512 validation tokens
Total tokens: 5,120 ✓
```

---

## 7. The GPT Model Architecture (How Input Becomes Logits)

The full model was built in Stage 1. Here we need to understand what happens inside at a high level to understand what the output means.

### Configuration used (GPT-2, 124M parameters)
| Parameter | Value | Meaning |
|---|---|---|
| vocab_size | 50,257 | Size of BPE vocabulary |
| context_length | 256 | Max input sequence length |
| emb_dim | 768 | Size of embedding vectors |
| n_heads | 12 | Number of attention heads |
| n_layers | 12 | Number of transformer blocks stacked |
| drop_rate | 0.1 | 10% of activations randomly zeroed during training |
| qkv_bias | False | No bias in query/key/value projections |

### Step-by-step flow inside the model

**Step 1 — Token Embedding**  
Each input token ID is looked up in an embedding table to get a vector of size `emb_dim=768`. This vector encodes the *semantic meaning* of that token — similar words have similar vectors.

**Step 2 — Positional Embedding**  
The model has no built-in sense of order (unlike RNNs). So a positional embedding is added to each token embedding. Position 0 gets one vector added, position 1 gets another, etc. This tells the model where each token sits in the sequence.

**Step 3 — Dropout**  
During training, 10% of activations are randomly set to zero. This prevents the model from relying too heavily on any specific neuron and helps generalisation.

**Step 4 — Transformer Block × 12 (The Core Engine)**  
This is where the real intelligence happens. The transformer block contains:
- **Layer Normalisation** — stabilises training by normalising activations
- **Multi-Head Attention** — the most important component (explained below)
- **Dropout + Shortcut (Residual) Connection** — allows gradients to flow easily during training
- **Feed-Forward Neural Network** — two linear layers with a non-linearity between them
- **Another Dropout + Shortcut**

This entire block is stacked 12 times. Each layer refines the representation further.

**Step 5 — Final Layer Normalisation**

**Step 6 — Output Linear Layer**  
Maps from `emb_dim=768` back to `vocab_size=50,257`. This is the final layer that produces one score per vocabulary word for each token position.

**Output: Logits tensor of shape [batch, seq_len, vocab_size] = [2, 256, 50257]**

### Token Embeddings vs Context Vectors — a crucial distinction

**Token embedding** (input to transformer):
- Fixed vector that only encodes what the word means in isolation
- "effort" always has the same embedding regardless of surrounding words
- No information about the context, sentence structure, or relationships

**Context vector** (output of transformer block):
- Much richer representation
- For the word "effort" in the sentence "every effort moves you," the context vector encodes not just what "effort" means but also HOW MUCH attention "effort" should pay to "every," "moves," and "you"
- Different sentences produce different context vectors for the same word
- This is what allows LLMs to understand meaning, ambiguity, and relationships

### Multi-Head Attention — why it is the engine of LLM power

Multi-head attention is the mechanism that transforms token embeddings into context vectors. It answers the question: "For each token, which other tokens in the sequence are most relevant to understanding it?"

This is what allows GPT to understand that in "The bank was steep," "bank" refers to a riverbank, while in "The bank closed early," it refers to a financial institution. The context vectors capture this distinction because attention looks at all surrounding tokens.

---

## 8. Logits — What the Model Actually Outputs

After a forward pass, the model returns a **logits tensor**.

### What logits are
Raw, unnormalised scores — one per vocabulary word, per token position.

```
For input batch of [2, 256]:
Logits shape = [2, 256, 50257]
  ↑batch  ↑sequence  ↑vocabulary
```

For every one of the 512 token positions across the batch, there are 50,257 scores.

### What logits are NOT
- They are NOT probabilities
- They do NOT sum to 1
- They CAN be negative
- They have no direct interpretable meaning until converted by Softmax

An untrained model produces essentially random logits — the scores have no meaningful pattern yet.

---

## 9. Softmax — Converting Logits to Probabilities

### What Softmax does
Takes any vector of real numbers (the logits) and converts them into a valid probability distribution:
- Every value becomes positive and between 0 and 1
- All values in the vector sum to exactly 1
- The relative ordering is preserved — the highest logit becomes the highest probability

### Why this is needed
We need probabilities to:
1. Make a prediction (pick the token with the highest probability)
2. Calculate loss (compare assigned probability to target)

After Softmax, each of the 50,257 values tells you: "the model thinks this is the probability that THIS vocabulary word is the next token."

For an untrained model, the probabilities will be roughly equal across all 50,257 words — around 0.00002 each.

---

## 10. Cross-Entropy Loss — Complete Breakdown

This is the mathematical heart of the lecture.

### What cross-entropy measures
It measures the difference between:
- What the model predicted (the probability distribution over 50,257 words)
- What the correct answer was (the actual next token)

A lower number means the model's predictions are closer to correct. Zero means perfect.

### The three internal steps (all done automatically by PyTorch in one line)

**Step 1 — Softmax**  
Convert the raw logits to a probability distribution (already described above).

**Step 2 — Index by Target Token**  
Look up the probability that was assigned to the *correct* target token at each position.

Example: Target sequence is `[had, always, thought, Jack]` with token IDs `[23, 3881, 1123, 15]`
- Position 0: model assigned probability P₁ to token index 23 ("had")
- Position 1: model assigned probability P₂ to token index 3881 ("always")
- Position 2: model assigned probability P₃ to token index 1123 ("thought")
- Position 3: model assigned probability P₄ to token index 15 ("Jack")

These P values are what we care about. They tell us: how confident was the model in the correct answers?

**Step 3 — Negative Log-Likelihood**  
Compute `-log(P)` for each position, then take the mean.

### Why the negative logarithm?

The `-log(P)` function creates a perfect loss signal:

| Probability assigned to correct token | Loss = -log(P) | Interpretation |
|---|---|---|
| 1.00 (100% confident, correct) | 0.000 | Perfect |
| 0.50 (50% confident) | 0.693 | Some uncertainty |
| 0.10 (10% confident) | 2.303 | Mostly wrong |
| 0.01 (1% confident) | 4.605 | Very wrong |
| 0.001 (0.1% confident) | 6.908 | Extremely wrong |
| → 0 (near-zero confidence) | → ∞ | Catastrophically wrong |

The key property: the model is **penalised extremely heavily** for being confidently wrong. Being uncertain (P=0.1) is bad but recoverable. Being confidently wrong about the wrong token is catastrophically penalised.

### Why use log specifically?

The logarithm turns multiplication into addition, which is computationally stable. Taking logs of small probabilities (like 0.00002) avoids numerical underflow. The negative is there because log of a number between 0 and 1 is always negative, and we want a positive loss.

### Goal of training: minimise this loss

As the model is trained through backpropagation, the weights are updated to assign higher probability to the correct tokens. The loss decreases. When the loss is very low, the model has learned to predict the right tokens reliably.

### The flatten step before cross-entropy

The model outputs `[2, 256, 50257]` (batch × sequence × vocab). PyTorch's cross_entropy expects `[N, vocab_size]`. So we flatten the first two dimensions:

```
[2, 256, 50257]  →  flatten(0,1)  →  [512, 50257]
[2, 256]         →  flatten()     →  [512]
```

Now 512 predictions are treated as 512 independent samples. Cross-entropy computes loss for each and returns the average — one single number representing the batch loss.

### From batch loss to dataset loss

The `calc_loss_loader` function:
1. Iterates through every batch in the DataLoader
2. Computes loss for each batch using `calc_loss_batch`
3. Accumulates all batch losses
4. Divides by number of batches → average loss per batch

This single average number is the **training loss** or **validation loss**.

---

## 11. Interpreting the Actual Loss Values

### What the code produces
```
Training loss:   ~10.98
Validation loss: ~10.98
```

### Why ~10.98 is correct and expected

The model is completely untrained with random weights. A random model has no knowledge of language — it assigns roughly equal probability to all 50,257 vocabulary words.

Probability assigned to the correct token ≈ `1/50257 ≈ 0.0000199`

Expected loss = `-log(1/50257) = log(50257) ≈ 10.82`

Our result of **~10.98 confirms the model is initialised correctly.** It is behaving exactly like a random model, which is exactly what we expect before any training. If this number were wildly off (e.g., 3.0 or 50.0), it would indicate a bug in the architecture or data pipeline.

### Why training loss ≈ validation loss at this point

The model has not been trained on either dataset. On both training and validation data, it is making completely random predictions. There is no performance difference yet.

Once training begins:
- **Training loss falls** — model is learning the patterns in training data
- **Validation loss falls more slowly** — generalisation to unseen data lags behind
- The **gap between them is the overfitting signal**
  - Small gap → model is learning generalisable patterns
  - Large gap → model is memorising training data without generalising

---

## 12. torch.no_grad() — Turning Off Gradient Computation

### What gradients are
During training, after the forward pass and loss computation, PyTorch computes gradients — mathematical quantities that tell each weight: "should you increase or decrease to reduce the loss?" PyTorch automatically tracks all operations in the forward pass to enable this.

This tracking uses memory and adds computation time.

### Why turn it off during evaluation
When we are only *measuring* the loss (not updating weights), we do NOT need gradients. No backpropagation will happen. Wrapping the evaluation in `torch.no_grad()`:
- Saves significant GPU/CPU memory
- Runs faster — no overhead from tracking operations
- Prevents accidental gradient accumulation

### The rule
- **Training:** gradients ON — backprop needs them
- **Evaluating loss / generating text:** `torch.no_grad()` — no weight updates, no need for gradients

In the next lecture, when the actual training loop is written, the training loss calculation will run WITHOUT `torch.no_grad()` so gradients can flow. The validation loss calculation will always remain inside `torch.no_grad()`.

---

## 13. Device Handling — CPU, GPU, Apple Silicon

### Why device management matters
PyTorch operations run on a specific device. If the model is on GPU but the data is on CPU, PyTorch throws an error. Both must be on the same device.

### Supported devices

**CPU** — available on any machine, slowest  
**CUDA (GPU)** — NVIDIA graphics cards, dramatically faster for large models  
**MPS (Apple Silicon)** — M1/M2/M3 Mac chips, approximately 2× faster than Apple CPU  

The code auto-detects the best available device:
- CUDA available → GPU
- Otherwise → CPU
- Optional block: also checks for MPS (Apple Silicon)

`model.to(device)` moves the model to the device. `input_batch.to(device)` moves each batch to the device. Both are required for the forward pass to work.

---

## 14. The Sanity Checks — How to Verify the Pipeline is Working

The lecture includes several important checks to confirm data is loaded and processed correctly. These are good engineering practices.

**Check 1 — First and last 100 characters of raw text**  
Confirms the file was loaded correctly and the content is as expected.

**Check 2 — Total character count and token count**  
Verifies tokenization worked correctly: 20,479 characters → 5,145 tokens.

**Check 3 — DataLoader batch shapes**  
Printing `x.shape, y.shape` for each batch confirms:
- Training: 9 batches of [2, 256]
- Validation: 1 batch of [2, 256]

**Check 4 — Token counts**  
`train_tokens = 4,608`, `val_tokens = 512`, `total = 5,120`  
The sum must roughly match the total encoded token count.

**Check 5 — Context length validation**  
Sanity check: the number of training tokens must be at least equal to context_length. If your dataset is smaller than your context window, nothing can be learned.

**Check 6 — The loss value itself**  
Final output ~10.98 ≈ log(50257) ≈ 10.82 confirms correct implementation end-to-end.

---

## 15. The Big Picture — Full End-to-End Flow

```
Raw text file ("the-verdict.txt")
         ↓
Load into Python string (20,479 characters)
         ↓
Encode with tiktoken BPE tokenizer
  → 5,145 integer token IDs
         ↓
Sequential 90/10 split
  → Training:   first 4,608 tokens
  → Validation: last  512 tokens
         ↓
GPTDatasetV1 class:
  → Slides context window of 256 with stride 256
  → Creates X (input) and Y (target = X shifted right by 1) pairs
         ↓
PyTorch DataLoader (batch_size=2, shuffle=True, drop_last=True)
  → Training:   9 batches of shape [2, 256]
  → Validation: 1 batch  of shape [2, 256]
         ↓
GPT Model (124M parameters, randomly initialised)
  → Token embeddings + positional embeddings
  → 12 Transformer blocks with multi-head attention
  → Output linear layer
  → Logits: [2, 256, 50257]
         ↓
calc_loss_batch:
  → Flatten logits: [512, 50257]
  → Flatten targets: [512]
  → nn.functional.cross_entropy → single loss value
         ↓
calc_loss_loader:
  → Loop all batches, average losses
  → Training Loss ≈ 10.98  (baseline — model is random)
  → Validation Loss ≈ 10.98 (same — both random)
         ↓
NEXT LECTURE: Backpropagation drives these losses down
```

---

## 16. What's Coming Next — LLM Pre-training

This lecture ends at the **starting point** — we have our baseline loss of ~10.98. We have not touched the model weights at all.

The next lecture implements the actual training loop:
1. Forward pass → compute loss (exactly what this lecture built)
2. `loss.backward()` → compute gradients for every weight in the model
3. `optimizer.step()` → update every weight slightly in the direction that reduces loss
4. Repeat for many batches and epochs
5. Watch training and validation loss decrease over time
6. Eventually: generate text that is coherent and reads like the training dataset

The entire evaluation pipeline built in this lecture (DataLoaders, calc_loss_batch, calc_loss_loader) is reused directly inside the training loop.

---

## Summary — Everything at a Glance

| Topic | Key Point |
|---|---|
| Dataset | "The Verdict" — tiny, fast, educational. Same code scales to any dataset |
| Real scale | LLaMA 2 = 2T tokens, $700K. Logic identical, only scale differs |
| Tokenization | BPE via tiktoken, vocab=50,257, subword-level, same as GPT-2/OpenAI |
| Train/Val split | 90/10 sequential. Val loss = true performance. Training loss alone can be misleading |
| Why validation | Catches overfitting — memorisation vs. genuine learning |
| Autoregressive | LLMs predict next token. Y = X shifted right by 1. Zero manual labelling needed |
| Context length | Tokens seen at once. 256 here. GPT-2 = 1024. Bigger = richer but more compute |
| Stride | Step between windows. Stride = context_length → no overlap, no skip |
| One input = N tasks | context_length predictions per forward pass due to attention masking |
| Token embedding | Fixed vector = word meaning in isolation, no context |
| Context vector | Rich vector = meaning + attention to all other tokens. Output of transformer |
| Multi-head attention | Converts embeddings → context vectors. The engine of LLM intelligence |
| Logits | Raw model output [batch, seq, vocab_size]. NOT probabilities. Can be any real number |
| Softmax | Converts logits → probabilities. Positive, sum to 1 |
| Cross-entropy | Softmax + index by target + negative log mean. Punishes confident wrong answers |
| -log(P) loss | P=1→loss=0. P→0→loss→∞. Model must learn to assign high P to correct tokens |
| Flatten before loss | Merge batch+seq dims. [2,256,50257]→[512,50257] so cross_entropy can process it |
| Random model loss | log(50257)≈10.82. Our ~10.98 confirms model is correctly initialised |
| Train ≈ Val now | Both random. Gap grows as training proceeds. Large gap = overfitting |
| torch.no_grad() | No gradients during eval. Faster + less memory. Always use during evaluation |
| Device handling | Model + data must be on same device. CPU/CUDA/MPS auto-detected |
| Next lecture | Backprop + optimiser + training loop → loss decreases → coherent text |
