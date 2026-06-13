# Introduction to LLM Fine-Tuning — Concepts, Categories & Dataset Preparation
### Lecture Notes | Build LLMs from Scratch Series — Lecture 31

> **Context:** Stages 1 (Architecture) and 2 (Pre-training, Evaluation, Loading Pre-trained Weights) are now complete. This lecture begins Stage 3 — **Fine-Tuning**. We cover what fine-tuning is, why it's needed, its two main categories, efficiency methods (LoRA/QLoRA), and then begin the hands-on classification fine-tuning project with dataset download and preprocessing.

---

## 1. Why Fine-Tuning is Necessary — The Gap Between Pre-training and Real Applications

### What pre-training gives you

After pre-training, you have a model that:
- Understands grammar, syntax, and general language patterns
- Can generate coherent English text
- Has absorbed general world knowledge from internet-scale data
- Can complete text in a plausible way

### What pre-training does NOT give you

Pre-training trains on **generic internet data** — Wikipedia, books, web pages, code. The model has no knowledge of:
- Your company's internal documents, tone, or policies
- Domain-specific terminology (medical, legal, financial)
- Your specific task format (classify emails, answer questions about your product)
- Your private data that was never on the internet

**The fundamental problem:** A model pre-trained on generic data produces generic outputs. If you want a chatbot that answers like your company, classifies your specific emails, or follows your specific instructions — generic pre-training is not sufficient.

### Fine-tuning bridges this gap

```
Pre-trained Model                Fine-tuned Model
(generic internet data)    →     (your specific data + task)
────────────────────────────────────────────────────────────
Knows general English            Knows your tone and style
Generic world knowledge          Knows your domain
Generates plausible text         Follows your instructions
One-size-fits-all output         Task-specific output
```

**Formal definition:** Fine-tuning is the process of taking a pre-existing model that has already learned patterns and features from a large dataset, and training it further on a smaller, domain-specific dataset to adapt its weights and biases for a specific task.

### Key terminology in the definition

- **"Pre-existing model"** — you don't train from scratch; you start from already-learned weights
- **"Additional training"** — the weights ARE updated again; fine-tuning is actual gradient descent, not just prompting
- **"Smaller domain-specific dataset"** — much less data than pre-training required; the model already knows language, it just needs to learn the new task
- **"Specific task"** — this is what defines what kind of fine-tuning you're doing

### Practical example — personal chatbot

Imagine you have a personal website with blog posts, publications, and talks written in your specific style. You want a chatbot that:
- Answers questions about your work
- Writes in your voice and tone
- Uses your vocabulary and sentence structure

A pre-trained model was never exposed to your private website. It cannot replicate your tone. Fine-tuning on your website content teaches the model your writing style, and the resulting chatbot speaks like you — not like a generic AI.

This exact use case appeared as a real question in the OpenAI community forum, where a user asked how to fine-tune a model to capture the tone of their blog. The answer: collect your blog data, fine-tune, and the model adapts to your style.

---

## 2. The Two Categories of Fine-Tuning

Fine-tuning is not one monolithic technique. There are two fundamentally different approaches depending on what you want the model to do.

### Category 1 — Instruction Fine-Tuning

**Definition:** Training the model on a dataset of (instruction, input, output) triples so it learns to follow natural language instructions and perform a variety of tasks.

**Structure of training data:**

```
Instruction: "Is the following text spam? Answer with yes or no."
Input:       "Congratulations! You've won a free iPhone. Click here."
Output:      "Yes"

Instruction: "Translate the following sentence into German."
Input:       "The weather is beautiful today."
Output:      "Das Wetter ist heute schön."
```

The instruction is explicitly provided as part of the input. The model learns to read the instruction and act accordingly.

**Characteristics:**
- Can handle a **broad range of tasks** with a single model (translation, summarisation, classification, Q&A, code generation — all with the same model)
- Requires **larger and more diverse datasets** because the model must learn many task types
- Requires **more computational power** — the model searches a much larger space of possible responses
- This is what ChatGPT, Claude, and most modern LLMs use

**Examples of instruction fine-tuning in practice:**
- ChatGPT (fine-tuned from GPT on instruction-following data + RLHF)
- Alpaca (instruction fine-tuned LLaMA)
- Any "assistant" model that responds to natural language commands

### Category 2 — Classification Fine-Tuning

**Definition:** Training the model to output one of a fixed set of class labels given an input, without natural language instructions.

**Structure of training data:**

```
Input:  "Congratulations! You've won a free iPhone. Click here."
Output: 1  (spam)

Input:  "Can we meet tomorrow at 3pm to discuss the project?"
Output: 0  (not spam / ham)
```

No instructions are given — just the raw input. The model learns to map inputs directly to discrete class labels.

**Characteristics:**
- Handles a **narrow, specific task** — only the classification it was trained for
- Requires **less data** than instruction fine-tuning
- Requires **less compute** — the output space is small and fixed
- The architecture is modified — the final output head is replaced with a classification head

**Examples of classification fine-tuning:**
- Spam/not spam email classifier
- Sentiment analysis (positive/negative/neutral)
- Emotion classification (angry/happy/sad/fearful)
- Medical diagnosis from text (disease present/absent)
- Content moderation (safe/unsafe)

### Side-by-side comparison

| Aspect | Instruction Fine-Tuning | Classification Fine-Tuning |
|--------|------------------------|---------------------------|
| Instructions in input? | Yes | No |
| Output format | Natural language text | Fixed class label |
| Task flexibility | Many tasks, one model | One specific task |
| Dataset size needed | Large, diverse | Smaller, focused |
| Compute needed | Higher | Lower |
| Architecture change | Minimal | Output head replaced |
| Common usage | ChatGPT-style assistants | Spam filters, sentiment analysis |

### The same task, two different approaches — spam detection

Both approaches can solve spam detection, but differently:

**Instruction fine-tuning approach:**
```
Input:  "Is the following text spam? Answer yes or no. Text: Win a free iPhone!"
Output: "Yes"   ← text output, parsed by code
```

**Classification fine-tuning approach:**
```
Input:  "Win a free iPhone!"
Output: 1   ← direct integer label, no parsing needed
```

Same result, different mechanisms. Classification fine-tuning is more efficient for this specific task; instruction fine-tuning is more flexible for future tasks.

---

## 3. Parameter-Efficient Fine-Tuning — LoRA and QLoRA

### The problem with full fine-tuning

In standard fine-tuning, **all** model parameters are updated during training. For a 124M parameter GPT-2 model, that means computing and storing gradients for 124 million parameters. For GPT-3 (175B parameters) or larger models, this is computationally infeasible for most users — it requires hundreds of GB of GPU memory.

### Parameter-efficient fine-tuning (PEFT)

PEFT methods update only a **small subset** of parameters while **freezing** the rest. The frozen parameters don't need gradients computed or stored — dramatically reducing memory requirements.

```
Full fine-tuning:
  All 124M params → compute gradients → update all → huge memory

PEFT:
  2M trainable params + 122M frozen params → compute gradients for 2M only → tiny memory
```

### LoRA — Low-Rank Adaptation

**Core idea:** Instead of fine-tuning the full weight matrix W (which might be 768×768 = 589,824 parameters), decompose the weight update ΔW into two much smaller matrices:

```
W_updated = W_original + ΔW
         = W_original + A × B

where:
  W_original: 768 × 768  (frozen, not updated)
  A:           768 × r   (trainable, r << 768)
  B:           r × 768   (trainable)
  r = rank (e.g., 4, 8, 16)
```

If r = 8:
- Original matrix: 768 × 768 = 589,824 parameters
- LoRA matrices: (768 × 8) + (8 × 768) = 12,288 parameters
- **98% reduction in trainable parameters** for this layer

The full weight matrix is never modified — only the small A and B matrices are trained. During inference, you can merge them: `W_final = W_original + A × B`.

**Why does this work?** Research has shown that weight updates during fine-tuning tend to have low "intrinsic rank" — they can be well-approximated by low-rank matrices. LoRA exploits this mathematical property.

### QLoRA — Quantized LoRA

QLoRA takes LoRA further by also **quantizing** the frozen base model weights to lower precision (typically 4-bit integers instead of 32-bit floats):

```
Full fine-tuning:
  Model weights: 32-bit float × 124M params = ~496 MB

LoRA:
  Frozen weights: 32-bit float × 124M = ~496 MB
  Trainable A+B:  32-bit float × ~2M  = ~8 MB
  Total: ~504 MB

QLoRA:
  Frozen weights: 4-bit int × 124M = ~62 MB   ← 8x compression!
  Trainable A+B:  16-bit float × ~2M = ~4 MB
  Total: ~66 MB
```

QLoRA makes it possible to fine-tune very large models (7B, 13B, 70B parameters) on consumer hardware like a single GPU with 24GB VRAM.

### Summary: When to use each method

| Method | Memory | Speed | Quality | Use when |
|--------|--------|-------|---------|----------|
| Full fine-tuning | Highest | Slowest | Best | You have large GPU resources |
| LoRA | Medium | Medium | Near-full | Limited GPU memory |
| QLoRA | Lowest | Fastest | Slightly lower | Consumer hardware, very large models |

> **Note:** LoRA and QLoRA are covered in detail in a subsequent lecture. This lecture introduces the concepts only.

---

## 4. The Hands-On Project — Spam Classification Fine-Tuning

For the practical component of this module, we will fine-tune the GPT model for **classification fine-tuning** on a spam detection task.

### Why classification fine-tuning first?

- Simpler output space (2 classes) makes it easier to understand the fine-tuning mechanics
- Clearly measurable accuracy metric
- Demonstrates how the same GPT architecture can be repurposed for non-generative tasks
- Builds foundation before tackling the more complex instruction fine-tuning

### The overall pipeline (across multiple lectures)

```
Stage 1 — Data Preparation (this lecture)
  ├── Download SMS spam dataset
  ├── Balance the dataset
  ├── Encode labels (ham=0, spam=1)
  └── Split into train/validation/test

Stage 2 — Model Setup (next lecture)
  ├── Create DataLoaders for batch processing
  ├── Initialize GPTModel
  ├── Load pre-trained OpenAI weights
  └── Modify model architecture for classification

Stage 3 — Fine-Tuning & Evaluation (subsequent lectures)
  ├── Fine-tune the model on training data
  ├── Evaluate on validation and test sets
  └── Use model on new, unseen emails
```

---

## 5. The Dataset — UCI SMS Spam Collection

### Source

The dataset comes from the **UCI Machine Learning Repository**, one of the most well-known public machine learning dataset repositories. The specific dataset is the **SMS Spam Collection**.

**URL:** UCI Machine Learning Repository → SMS Spam Collection

### Dataset composition

| Category | Count | Source |
|----------|-------|--------|
| Spam | 425 | Manually extracted from a UK public forum where users reported SMS spam |
| Spam | 322 | Additional publicly available spam messages |
| **Spam total** | **747** | |
| Ham (not spam) | 4,825 | Legitimate messages from NUS (National University of Singapore) student corpus |
| **Total** | **5,572** | |

**Why so many more ham than spam?** Real-world spam is less common than legitimate messages, and the ham messages came from a university email corpus where almost all messages are legitimate.

**Why NUS?** University email systems between students are extremely unlikely to contain spam, making them a reliable source of confirmed ham messages.

### File format

The downloaded file is a `.tsv` (tab-separated values) file named `SMSSpamCollection`. Each row contains:
```
label    message_text
ham      Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam     Free entry in 2 a wkly comp to win FA Cup final tkts...
```

### Loading and inspecting with Pandas

```python
import pandas as pd

# Load the TSV file
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None,
                 names=["Label", "Text"])

print(df.shape)          # (5572, 2)
print(df["Label"].value_counts())
# ham     4825
# spam     747
```

---

## 6. Data Preprocessing — Step by Step

### Step 1 — Balancing the Dataset

**The problem — class imbalance:** With 4,825 ham and 747 spam messages, the dataset is highly imbalanced (6.5:1 ratio). If you train on this as-is, the model will learn to always predict "ham" because that's correct 87% of the time — without actually learning to detect spam.

**The solution — undersampling:** Randomly sample 747 ham messages so both classes have equal representation.

```python
def create_balanced_dataset(df):
    """
    Undersample the majority class (ham) to match minority class (spam).
    Simple approach for educational purposes.
    """
    # Count spam instances (the minority class)
    num_spam = df[df["Label"] == "spam"].shape[0]   # = 747
    
    # Randomly sample that many ham instances
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )
    
    # Combine the balanced subsets
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    
    return balanced_df.reset_index(drop=True)

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
# ham     747
# spam    747
# Total:  1494
```

**Why this approach?** More sophisticated methods exist (oversampling with SMOTE, class weights, etc.) but for this educational context, simple undersampling is sufficient and keeps the focus on LLM fine-tuning rather than class imbalance techniques.

**What is "ham"?** "Ham" is the standard term in spam filtering for non-spam messages — legitimate email. The etymology is unclear but the terminology is standard in the field.

### Step 2 — Label Encoding

Machine learning models cannot use string labels. Convert "ham"/"spam" to integers:

```python
# Map string labels to integers
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# Verify
print(balanced_df["Label"].value_counts())
# 0    747   (ham / not spam)
# 1    747   (spam)
```

**Analogy to tokenization:** Just as pre-training converted words to token IDs (50,257 possible values), here we're converting class labels to class IDs (2 possible values: 0 or 1). The concept is identical — mapping human-readable strings to integers that the model can process mathematically.

### Step 3 — Train/Validation/Test Split

Standard practice in machine learning: never evaluate a model on data it was trained on. Split the data into three non-overlapping sets:

```python
def random_split(df, train_frac, validation_frac):
    """
    Split dataframe into train, validation, and test sets.
    
    Args:
        df:              Balanced dataframe (1494 rows)
        train_frac:      Fraction for training (0.7 = 70%)
        validation_frac: Fraction for validation (0.1 = 10%)
        
    Returns:
        train_df, validation_df, test_df
        (remaining fraction automatically goes to test)
    """
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # Calculate split indices
    train_end = int(len(df) * train_frac)              # 70% mark
    validation_end = train_end + int(len(df) * validation_frac)  # 80% mark
    
    # Slice into three parts
    train_df      = df[:train_end]           # 0% to 70%
    validation_df = df[train_end:validation_end]   # 70% to 80%
    test_df       = df[validation_end:]      # 80% to 100%
    
    return train_df, validation_df, test_df
```

**Applying the split:**

```python
train_df, validation_df, test_df = random_split(
    balanced_df,
    train_frac=0.70,
    validation_frac=0.10
)

# Verify sizes add up to 1494
print(len(train_df))       # 1045  (70% of 1494)
print(len(validation_df))  # 149   (10% of 1494)
print(len(test_df))        # 300   (20% of 1494, the remainder)

# Sanity check
assert len(train_df) + len(validation_df) + len(test_df) == 1494  # ✓
```

**What each split is used for:**

| Split | Size | Purpose |
|-------|------|---------|
| Train (70%) | 1,045 | Model sees this data during fine-tuning; weights are updated based on this |
| Validation (10%) | 149 | Monitor training progress; check for overfitting; tune hyperparameters |
| Test (20%) | 300 | Final evaluation only; model never sees this during training; unbiased accuracy estimate |

**Why 70/10/20 and not 80/10/10?**
With only 1,494 total samples, a 20% test set gives 300 examples for final evaluation — enough for reliable accuracy estimates. Larger test sets give more reliable final metrics.

### Step 4 — Save to CSV Files

The splits are saved as CSV files for reuse in subsequent lectures without rerunning the preprocessing:

```python
train_df.to_csv("train.csv", index=False)
validation_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

# These files can be loaded in later lectures with:
# train_df = pd.read_csv("train.csv")
```

---

## 7. What's Coming Next — Modifying GPT for Classification

At the end of this data pipeline, you'll have three clean CSV files ready. The next steps in subsequent lectures will be:

### How does a text-generation model do classification?

This is the key architectural question. The GPT model was designed to output a probability distribution over 50,257 vocabulary tokens. But classification needs only 2 outputs (0 or 1).

The solution: **replace the final output head**.

```
Standard GPT:
  ... → Transformer blocks → Layer Norm → Linear(768 → 50,257) → 50,257 logits

Classification GPT:
  ... → Transformer blocks → Layer Norm → Linear(768 → 2) → 2 logits → softmax → [p_ham, p_spam]
```

The entire transformer backbone is kept (this is where the language understanding lives). Only the final linear layer is replaced with a new one that outputs 2 values instead of 50,257.

During fine-tuning:
- The new classification head is trained from scratch (random initialisation)
- The transformer backbone weights can be either frozen (faster) or updated (better accuracy)
- Loss function changes from cross-entropy over 50,257 classes to binary cross-entropy over 2 classes

This architectural modification is what makes classification fine-tuning different from the generation we've done so far.

---

## 8. Complete Data Pipeline Summary

```
Raw data (SMSSpamCollection.tsv)
           │
           ▼
    Load with pandas
    5,572 rows: 4,825 ham + 747 spam
           │
           ▼
    Balance dataset (undersample ham)
    1,494 rows: 747 ham + 747 spam
           │
           ▼
    Encode labels: ham→0, spam→1
    1,494 rows with integer labels
           │
           ▼
    Random split (shuffle first)
    ┌──────┬──────────┬──────┐
    │Train │Validation│ Test │
    │ 70%  │   10%    │ 20%  │
    │ 1045 │   149    │ 300  │
    └──────┴──────────┴──────┘
           │
           ▼
    Save to CSV files
    train.csv, validation.csv, test.csv
           │
           ▼
    Ready for DataLoader creation (next lecture)
```

---

## Key Concepts to Remember

- **Fine-tuning** — additional training of a pre-trained model on smaller domain-specific data; model weights ARE updated; not just prompting
- **Why fine-tuning is needed** — pre-trained models know general language but not your specific task, data, or tone
- **Instruction fine-tuning** — model trained on (instruction, input, output) triples; handles many tasks; requires more data and compute; what ChatGPT-style models use
- **Classification fine-tuning** — model trained on (input, class label) pairs; handles one specific classification task; requires less data; output head is replaced
- **LoRA** — fine-tuning method that trains only two small low-rank matrices per layer instead of the full weight matrix; 95%+ reduction in trainable parameters
- **QLoRA** — LoRA + quantization of frozen weights to 4-bit; enables fine-tuning large models on consumer hardware
- **Class imbalance** — 4825 ham vs 747 spam must be balanced; we undersample ham to 747 so both classes are equal
- **"Ham"** — standard spam-filtering term for legitimate (non-spam) messages
- **Label encoding** — ham→0, spam→1; same concept as token IDs but with only 2 values
- **Train/val/test split** — 70/10/20; train for learning, validation for monitoring, test for final unbiased evaluation
- **Architecture modification for classification** — replace Linear(768→50257) output head with Linear(768→2); keep all transformer blocks intact

---

*Lecture 31 | LLM from Scratch Series | Next: DataLoaders, Model Initialization & Loading Pre-trained Weights for Classification*
