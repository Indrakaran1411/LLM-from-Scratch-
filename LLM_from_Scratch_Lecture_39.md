# Lecture 39 — Instruction Fine-Tuning: Creating DataLoaders

> **Where we are:** Lecture 37 = download & format data. Lecture 38 = batch the data (5 steps). This lecture = wrap those batches into DataLoaders so training can access them easily.

---

## What Is a DataLoader? (Simple Idea)

Think of your dataset as a big pile of flash cards.

A **DataLoader** is like a helper that picks up 8 cards at a time, hands them to you in order, and keeps track of where it left off.

That's it. It makes accessing batches easy and efficient during training.

> PyTorch docs say: *"DataLoader wraps an iterable around the dataset"* — meaning you can go through batches one by one, like a loop.

---

## Why Do We Need It?

Without a DataLoader you would have to manually:
- Slice the dataset into batches
- Track which batch you're on
- Move data to GPU yourself inside the training loop

With a DataLoader, all of this is handled automatically. Training code becomes simple:

```python
for inputs, targets in train_loader:
    # train on this batch
```

---

## Quick Recap of What We Built So Far

Before DataLoaders, we already did:

| Step | What was done |
|------|--------------|
| Lecture 37 | Downloaded 1100 instruction-response pairs, formatted into Alpaca prompts |
| Lecture 38 | Batched the data — tokenised, padded with `50256`, created targets (shift by 1), replaced extra padding with `-100` |
| **This lecture** | **Wrap batches into DataLoaders** |

---

## The `InstructionDataset` Class (From Last Lecture)

This class converts raw data into token IDs. The DataLoader uses this as its source.

```python
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            # Format into Alpaca prompt
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            # Convert to token IDs
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encoded_texts[index]
```

Each item returned is a list of token IDs. The DataLoader will call `__getitem__` repeatedly to build batches.

---

## Setting Up the Device

Before creating DataLoaders, we decide where tensors should live (CPU or GPU).

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Mac users with Apple Silicon can use: torch.device("mps")
```

**Why put this inside the collate function instead of the training loop?**

In Lecture 32 (pre-training), we moved data to GPU *inside* the training loop. This blocks the GPU briefly on every batch.

Now we move data to the device *inside the collate function* — this happens in the background, separately from training. The GPU stays free for actual model computation.

> Simple rule: **device transfer in collate function = background task = faster training.**

---

## The `custom_collate_fn` (From Last Lecture, Slightly Extended)

This function is the heart of the DataLoader. For each batch it receives, it:
1. Pads all sequences to the same length using `50256`
2. Creates input and target tensors (target = input shifted right by 1)
3. Replaces extra `50256` padding in targets with `-100`
4. Moves tensors to the device

We also add a `allowed_max_length` limit (set to 1024 — the GPT-2 context length). Any sequence longer than this gets cut off.

```python
from functools import partial

# Pre-fill the device and max_length arguments
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)
```

`partial` creates a new version of `custom_collate_fn` with `device` and `allowed_max_length` already set. Now you don't have to pass them every time — cleaner code.

---

## Creating the Three DataLoaders

```python
from torch.utils.data import DataLoader

# Training DataLoader
train_dataset = InstructionDataset(train_data, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True
)

# Validation DataLoader
val_dataset = InstructionDataset(val_data, tokenizer)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False
)

# Test DataLoader
test_dataset = InstructionDataset(test_data, tokenizer)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False
)
```

**Why `shuffle=True` for training only?**
Shuffling means the model sees prompts in a different order each epoch. This prevents it from memorising the sequence and helps it generalise.

**Why `drop_last=True` for training only?**
If 935 samples ÷ 8 = 116 batches with 7 samples left over, that last incomplete batch gets dropped. Incomplete batches can cause problems with certain loss calculations. Validation and test sets are small, so we keep every sample.

---

## Understanding the DataLoader Output Shape

When you print the first batch from `train_loader`:

```python
for inputs, targets in train_loader:
    print(inputs.shape)   # torch.Size([8, 61])
    print(targets.shape)  # torch.Size([8, 61])
    break
```

The shape `[8, 61]` means:
- **8** = batch size (8 prompts per batch)
- **61** = number of token IDs in each prompt (after padding)

**Why does the second batch have shape `[8, 76]`?**

Each batch finds its *own* longest prompt and pads others to match. Different batches have different max lengths.

```
Batch 1: longest prompt = 61 tokens  → all padded to 61
Batch 2: longest prompt = 76 tokens  → all padded to 76
Batch 3: longest prompt = 58 tokens  → all padded to 58
```

This is more efficient than padding everything to the global maximum (which might be 500+). Most batches use much less memory this way.

---

## How Many Batches Are There?

```
Train:      935 samples ÷ 8 = ~116 batches
Validation:  55 samples ÷ 8 =   ~6 batches
Test:       110 samples ÷ 8 =  ~13 batches
```

Each time you loop through `train_loader`, you go through 116 batches = 1 epoch.

---

## How to Access Data from a DataLoader

```python
# Get the first batch
first_batch = next(iter(train_loader))
inputs, targets = first_batch

print(inputs.shape)    # [8, token_length]
print(targets.shape)   # [8, token_length]

# Access the 3rd prompt in the first batch
print(inputs[2])       # token IDs for prompt 3
print(targets[2])      # shifted token IDs for prompt 3
```

This is the clean access pattern the training loop will use.

---

## What Stage 1 Looked Like (Completed)

We spent three full lectures on data preparation:

```
Lecture 37 → Download + Format (Alpaca style)
Lecture 38 → Batch (tokenise, pad, targets, -100)
Lecture 39 → DataLoaders (wrap batches, device transfer)
```

This is intentional. In machine learning, **data preparation is the hardest part**. If the data going into the model is wrong, nothing else matters. The training loop itself (next lecture) is actually much shorter than all of this.

---

## Summary

| Concept | Simple way to remember it |
|---------|--------------------------|
| DataLoader | Helper that picks up 8 samples at a time and hands them to you |
| `collate_fn` | The function that does all the hard work (padding, targets, -100) inside each batch |
| `partial` | Pre-fills function arguments so you don't repeat them every call |
| Device in collate | Moves tensors to GPU in background → training loop is not blocked |
| Different batch sizes | Each batch pads to its own max length → saves memory |
| `shuffle=True` | Only for training — avoids memorising the order |
| `drop_last=True` | Only for training — drops the last incomplete batch |

---

## What's Next

Stage 2: Load the pre-trained GPT-2 weights and actually fine-tune the model on these DataLoaders.
