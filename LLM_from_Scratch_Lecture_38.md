# Lecture 38 — Instruction Fine-Tuning: Batching the Dataset

> **Where we are:** Lecture 37 covered downloading the dataset and formatting it into Alpaca-style prompts. This lecture covers the hardest part of instruction fine-tuning data prep — turning those formatted prompts into proper training batches. This is a 5-step process, and each step has a specific reason behind it.

---

## Why Batching Is More Complex Here

In classification fine-tuning (spam project), every input was padded to a fixed `max_length`. Simple.

In instruction fine-tuning, each prompt is unique — some are short, some are long. The key constraint is:

> **All samples inside one batch must have the same number of token IDs (same number of columns).**

The GPU cannot process rows of different lengths together. So we need a smart batching strategy.

---

## The 5-Step Batching Process (Roadmap)

Memorise this before reading the code. Everything in the code maps to one of these steps.

| Step | What happens |
|------|-------------|
| 1 | Format data using Alpaca prompt template |
| 2 | Tokenise each formatted prompt into token IDs |
| 3 | Pad all sequences in a batch to the same length using token `50256` |
| 4 | Create Target token IDs (input shifted right by one) |
| 5 | Replace padding tokens in Target with `-100` (except the first `50256`) |

---

## Step 1 & 2 — Format Then Tokenise

This was covered in Lecture 37. The `InstructionDataset` class applies both steps:

```python
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            # Step 1: Convert to Alpaca prompt format
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            # Step 2: Tokenise into token IDs
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encoded_texts[index]
```

After this step, each sample is a list of token IDs — but lists of different lengths.

**The problem:**
```
Sample 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]       ← 10 tokens
Sample 2: [0, 1, 2, 3]                            ← 4 tokens
Sample 3: [0, 1, 2, 3, 4, 5]                      ← 6 tokens
```
Cannot batch these together — different column counts.

---

## Step 3 — Padding to Equal Length

Inside each batch, find the longest sequence and pad all others to match using token ID `50256` (the `<|endoftext|>` token).

**Why token 50256 specifically?**

GPT-2 has a vocabulary of 50,257 tokens. Token IDs go from `0` to `50256`. The last one (`50256`) is `<|endoftext|>` — it marks where one text sample ends and another begins. Using any other token ID for padding would be wrong because that token carries real word meaning and would confuse the model.

**Visual example:**

```
Before padding (batch of 3):
  Sample 1: [0, 1, 2, 3, 4]          ← 5 tokens (longest)
  Sample 2: [5, 6]                    ← 2 tokens
  Sample 3: [7, 8, 9]                 ← 3 tokens

After padding (all equal to 5):
  Sample 1: [0, 1, 2, 3, 4]
  Sample 2: [5, 6, 50256, 50256, 50256]
  Sample 3: [7, 8, 9, 50256, 50256]
```

> **Important:** Padding is done per-batch, not globally. Each batch finds its own longest sequence. This is more efficient than padding everything to the global maximum length.

---

## Step 4 — Creating Target Token IDs

This is the most non-intuitive step. Read carefully.

### The "next token prediction" connection

The LLM is trained using **next token prediction** — the same task as pre-training. The target for any input is just the input shifted right by one position:

```
Input:   [0, 1, 2, 3, 4]
Target:  [1, 2, 3, 4, 50256]
          ↑ drop first, add 50256 at end
```

This means: *given token 0, predict token 1. Given tokens 0,1, predict token 2.* And so on.

### Why include the instruction in the target?

You might think: the model only needs to learn the response, so the target should only contain the response tokens.

But this is not how it works. The full prompt (instruction + input + response) is the input, and the full prompt shifted by one is the target. This is intentional.

Here's why it works: as the LLM trains on next token prediction, it eventually reaches the point in the sequence where it has seen the instruction and the input. At that position, it must predict the first token of the response. Then the next token. Then the next. This is how it learns to generate the response given the instruction.

```
If input so far is:  "Below is an instruction... ### Instruction: Fix grammar ### Input: He go..."
Then predict:         "### Response:"    ← model learns this transition

If input so far is:  "...### Response:"
Then predict:         "He goes"         ← model learns the actual answer
```

The redundant training on the instruction portion still helps — it's not wasted.

### Why add an extra `50256` before padding?

A small trick in the code: before padding, an extra `50256` token is appended to every input. This makes creating the target trivial — you just drop the first token and take the rest. No extra addition needed.

```python
# Add extra 50256 to every input first
padded = item + [pad_token_id]   # always add one 50256

# Then pad to max_length
padded += [pad_token_id] * (max_length - len(padded))

# Truncate to max_length
input_tensor = padded[:-1]       # drop the last element
target_tensor = padded[1:]       # shift right by one — done!
```

This is why the function first finds `max_length + 1` as the target length, then truncates back. It's a clean way to handle target creation without special cases.

---

## Step 5 — Replace Padding with `-100`

After creating the target tensor, most of the `50256` values in it are just padding — meaningless filler. Including them in the loss calculation would penalise the model for failing to predict padding, which is wrong.

**The fix:** Replace all `50256` padding tokens in the target with `-100`.

```
Target before replacement:
  [1, 2, 3, 4, 50256]             ← no padding, keep as is
  [6, 50256, 50256, 50256, 50256] ← lots of padding
  [8, 9, 50256, 50256, 50256]     ← some padding

Target after replacement:
  [1, 2, 3, 4, 50256]             ← unchanged (no padding here)
  [6, 50256, -100, -100, -100]    ← first 50256 kept, rest → -100
  [8, 9, 50256, -100, -100]       ← first 50256 kept, rest → -100
```

**Why `-100` specifically?**

PyTorch's `CrossEntropyLoss` has a built-in parameter:

```python
torch.nn.CrossEntropyLoss(ignore_index=-100)  # default value
```

Any target position with value `-100` is automatically **skipped** when calculating the loss. PyTorch will not compute gradients for those positions. This means the model is not penalised for those tokens — they contribute nothing to training.

**Proof this works:**
```python
logits = torch.tensor([[-1.0, 1.0], [-0.5, 1.5], [2.0, -1.0]])
targets_normal  = torch.tensor([0, 1, 1])    # 3 examples
targets_ignored = torch.tensor([0, 1, -100]) # 3rd example ignored

loss_fn = torch.nn.CrossEntropyLoss()

print(loss_fn(logits[:2], targets_normal[:2]))   # 1.1269
print(loss_fn(logits,     targets_ignored))       # also 1.1269 — 3rd row is ignored!
```

The loss is identical because `-100` makes the third example completely invisible to the loss function.

**Why keep the first `50256`?**

The first `50256` in the target (right after the actual response ends) must be kept. It teaches the model to generate `<|endoftext|>` when the response is complete — this is how the model knows to stop generating.

---

## The Complete `custom_collate_fn`

```python
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, device="cpu"):
    # Find the longest sequence, add 1 for the target-creation trick
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()

        # Add one extra 50256 to every sequence
        new_item += [pad_token_id]

        # Pad to max length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # Input: all tokens except last
        inputs = torch.tensor(padded[:-1])

        # Target: all tokens except first (shifted right by 1)
        targets = torch.tensor(padded[1:])

        # Replace all 50256 in targets except the first one with -100
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index  # keep first, replace rest

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Stack into tensors and send to device
    inputs_tensor  = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
```

Every line maps directly to one of the 5 steps. No magic — just the workflow implemented cleanly.

---

## Optional: Masking Instruction Tokens in the Target

Some researchers go one step further — they replace the instruction portion of the target with `-100` as well, so the loss is only calculated on the response tokens.

```
Prompt:    [... instruction tokens ... | response tokens ...]
Target:    [-100, -100, ..., -100     | response tokens ...]
                   ↑ masked out              ↑ only this contributes to loss
```

**Argument for masking:** The model should focus on learning to generate the response, not memorise the instruction. It reduces overfitting.

**Argument against masking:** A recent paper (*"Instruction Tuning with Loss Over Instructions"*) showed that keeping the instruction in the loss actually improves LLM performance. The model benefits from practicing the full sequence.

**Current status:** Not yet settled. Both approaches are used. The implementation in this series does **not** mask instructions — it's left as an optional experiment for you to try. Changing it is simple: find the token boundary between instruction and response, and replace everything before it with `-100` in the target.

---

## Summary Table

| Step | Operation | Key detail |
|------|-----------|-----------|
| 1 | Alpaca prompt formatting | Same as Lecture 37 |
| 2 | Tokenise with tiktoken | Each sample → list of token IDs, different lengths |
| 3 | Pad within each batch | Use `50256` (`<|endoftext|>`), not a random token |
| 4 | Shift input → target | Drop first token, append `50256`; the trick is adding an extra `50256` upfront |
| 5 | Replace padding with `-100` | PyTorch ignores `-100` in cross-entropy loss by default; keep first `50256` to teach end-of-text |

**The key insight of the whole lecture:**

The model doesn't have a separate "instruction input" and "response output". The full formatted prompt is the input, and the full formatted prompt shifted by one position is the target. The model learns to follow instructions through next-token prediction alone — by the time it has processed all the instruction tokens, it must predict the response tokens, which is exactly what we want.

---

## What's Next

Next lecture: creating DataLoaders from these batches, loading the pre-trained GPT-2 weights, and starting the actual fine-tuning loop.
