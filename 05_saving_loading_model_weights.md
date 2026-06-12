# Saving & Loading Model Weights in PyTorch

> **Series:** Build LLMs from Scratch — Pre-training  
> **Topic:** Weight Persistence (Saving & Loading)  
> **Why it matters:** A 124M parameter model takes significant compute to train. Proper checkpointing lets you pause, resume, share, and deploy without losing progress.

---

## Table of Contents
1. [What is a State Dictionary?](#1-what-is-a-state-dictionary)
2. [Saving Only the Model](#2-saving-only-the-model)
3. [Loading the Model Back](#3-loading-the-model-back)
4. [Why You Must Also Save the Optimizer](#4-why-you-must-also-save-the-optimizer)
5. [Saving a Full Training Checkpoint](#5-saving-a-full-training-checkpoint)
6. [Loading a Checkpoint to Resume Training](#6-loading-a-checkpoint-to-resume-training)
7. [model.eval() vs model.train()](#7-modeleval-vs-modeltrain)
8. [Quick Reference — When to Use What](#8-quick-reference--when-to-use-what)
9. [Common Mistakes](#9-common-mistakes)

---

## 1. What is a State Dictionary?

PyTorch stores all learnable parameters of a model in a **state dictionary**.  
It is a plain Python `dict` that maps each **layer name → parameter tensor**.

Think of it as a snapshot of every weight and bias your model has learned.

```python
sd = model.state_dict()

for key, val in sd.items():
    print(key, val.shape)

# Example output:
# transformer.wte.weight        torch.Size([50257, 768])
# transformer.wpe.weight        torch.Size([1024, 768])
# transformer.h.0.ln_1.weight   torch.Size([768])
# transformer.h.0.attn.c_attn.weight  torch.Size([768, 2304])
```

**What it contains:** Layer names as keys, parameter tensors as values.  
**What it does NOT contain:** Model architecture code, optimizer state, or training config — just raw numbers.

> You must always recreate the model architecture before loading a state dict into it. PyTorch saves numbers, not class definitions.

---

## 2. Saving Only the Model

Use `torch.save()` to serialize the state dictionary to disk.

```python
import torch

# Save model weights to disk
torch.save(model.state_dict(), "model.pth")
```

**When to use this:**
- Sharing a trained model with a collaborator for inference
- Deploying a model where you will not resume training
- Storing a final checkpoint after training is complete

> `.pth` is a convention for PyTorch files. Any file extension technically works, but stick to `.pth` or `.pt` for clarity.

---

## 3. Loading the Model Back

```python
# Step 1 — Recreate the exact same architecture
model = GPTModel(GPT_CONFIG_124M)

# Step 2 — Load the saved weights into it
model.load_state_dict(
    torch.load("model.pth", map_location="cpu")
)

# Step 3 — Set to evaluation mode before inference
model.eval()
```

**`map_location="cpu"`** — use this when the file was saved on a GPU machine but you are loading on CPU. It prevents device mismatch errors.

---

## 4. Why You Must Also Save the Optimizer

Saving only the model is **not enough** when you plan to resume training.

### The problem with adaptive optimizers (AdamW)

AdamW is **stateful**. For every single parameter in the model, it internally tracks:

| What | Variable | Purpose |
|---|---|---|
| First moment | `m` | Running average of past gradients — direction parameters have been moving |
| Second moment | `v` | Running average of squared gradients — variance each parameter has seen |
| Step count | `t` | Used for bias correction in the Adam update rule |

**If you discard this history when resuming:**
- The optimizer behaves as if training just started
- Learning rates get miscalibrated per parameter
- Convergence slows significantly or the model diverges
- All previous training progress is partially wasted

> **Senior engineer rule:** Losing optimizer state mid-training is one of the most common and costly mistakes in production ML. Always save both model and optimizer together.

---

## 5. Saving a Full Training Checkpoint

A checkpoint bundles model state, optimizer state, and metadata into one file. This is the **industry-standard pattern** for resumable training.

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": current_epoch,
        "loss": train_loss,
    },
    "model_and_optimizer.pth"
)
```

You can add anything else useful to this dictionary — current learning rate, config, random seed, etc.

---

## 6. Loading a Checkpoint to Resume Training

```python
# Step 1 — Load the checkpoint file
checkpoint = torch.load("model_and_optimizer.pth", map_location="cpu")

# Step 2 — Recreate model and load weights
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

# Step 3 — Recreate optimizer and load its state
optimizer = torch.optim.AdamW(model.parameters())
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Step 4 — Resume from where you left off
start_epoch = checkpoint["epoch"] + 1

# Step 5 — Set model to training mode
model.train()

# Now run your training loop starting from start_epoch
```

---

## 7. model.eval() vs model.train()

These are **mode switches**, not training commands. They change the behaviour of certain layers.

| Call | Use when | What it does |
|---|---|---|
| `model.eval()` | Inference / text generation / loss evaluation | Disables Dropout (all neurons active). Freezes BatchNorm statistics. Gives deterministic outputs. |
| `model.train()` | Before running training loop | Enables Dropout. Activates running BatchNorm. Required before forward + backward pass. |

```python
# Inference — always call eval() first
model.eval()
with torch.no_grad():
    output = model(input_ids)

# Training — always call train() first
model.train()
optimizer.zero_grad()
loss = criterion(model(input_ids), targets)
loss.backward()
optimizer.step()
```

> Forgetting `model.eval()` during inference is a **silent bug** — Dropout randomly zeros neurons and you get different outputs every run on the same input.

---

## 8. Quick Reference — When to Use What

### Sharing a model for inference only
```python
# Sender
torch.save(model.state_dict(), "model.pth")

# Receiver
model = GPTModel(config)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()
```

### Pausing and resuming training
```python
# Before pausing
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
}, "checkpoint.pth")

# When resuming
checkpoint = torch.load("checkpoint.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()
```

### Loading pre-trained weights (e.g. OpenAI GPT-2)
```python
# You only receive model weights — no optimizer needed
model = GPTModel(config)
model.load_state_dict(torch.load("gpt2_weights.pth", map_location="cpu"))
model.eval()
# Ready for text generation
```

---

## 9. Common Mistakes

| Mistake | Why it's a problem | Fix |
|---|---|---|
| Saving only model, not optimizer | Optimizer loses gradient history. Resuming training is degraded. | Always save `optimizer.state_dict()` in checkpoint |
| Not calling `model.eval()` before inference | Dropout is active — non-deterministic, wrong outputs | Always `model.eval()` + `torch.no_grad()` for inference |
| Not calling `model.train()` before training loop | Some layers behave incorrectly | Always `model.train()` before your training loop |
| Forgetting `map_location="cpu"` | Crashes when loading a GPU-saved model on a CPU machine | Always pass `map_location` explicitly |
| Loading weights into wrong architecture | Size mismatch error | Ensure architecture definition exactly matches saved model |

---

## Summary

```
torch.save(model.state_dict(), "model.pth")           → save weights only
torch.save({model + optimizer + meta}, "ckpt.pth")    → save full checkpoint

model.load_state_dict(torch.load("model.pth"))         → load weights only
checkpoint = torch.load("ckpt.pth")                    → load full checkpoint
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

model.eval()    → inference mode (disable dropout, freeze batchnorm)
model.train()   → training mode (enable dropout, running batchnorm)
```

---

**Next:** Loading pre-trained GPT-2 weights from OpenAI into the custom GPT architecture — applying the inference-only loading pattern above with real-world weights.
