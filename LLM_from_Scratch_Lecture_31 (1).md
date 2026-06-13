# Saving and Loading PyTorch Model Weights
### Lecture Notes | Build LLMs from Scratch Series — Lecture 32

> **Context:** We have pre-trained a GPT model, implemented temperature scaling and top-k sampling for text generation. The next major step is loading OpenAI's pre-trained GPT-2 weights into our architecture. This lecture builds the essential foundation for that — understanding how to save and load model parameters and optimizer states in PyTorch. Think of this as a toolbox lecture.

---

## 1. Why Saving and Loading Weights Matters

### The problem without saving

Training a 124M parameter GPT model takes significant time and compute. Without saving:
- Close your Jupyter notebook → all trained weights are gone
- Crash mid-training → start from epoch 1 again
- Want to share your model with a collaborator → impossible
- Want to fine-tune someone else's model → impossible
- Want to load OpenAI's pre-trained GPT-2 weights → impossible

### What saving solves

```
Without saving:                    With saving:
─────────────────                  ──────────────────────────────
Train for 10 hours                 Train for 10 hours
Close session                      Save weights to model.pth
Open session tomorrow         →    Open session tomorrow
Train from scratch again           Load model.pth → continue training
(10 more hours wasted)             (0 hours wasted)
```

For large language models specifically, this is critical:
- **Memory:** A 124M parameter model at 32-bit precision = ~496MB. Recomputing these from random initialisation every session is wasteful.
- **Time:** Pre-training takes hours/days. Saving checkpoints means you never lose progress.
- **Sharing:** The entire field of transfer learning (including loading OpenAI weights) depends on being able to save and load model parameters.

---

## 2. Understanding the State Dictionary

Before learning to save/load, you need to understand what exactly gets saved.

### What is a state dictionary?

A **state dictionary** (`state_dict`) is a Python dictionary that maps each layer name to its parameter tensor. It captures the complete learned state of a model.

```python
model = GPTModel(GPT_CONFIG_124M)

# Look at the state dictionary
state_dict = model.state_dict()
print(type(state_dict))         # <class 'collections.OrderedDict'>
print(len(state_dict))          # number of parameter tensors

# Print first few entries
for key, value in list(state_dict.items())[:5]:
    print(f"{key}: shape={value.shape}")

# Output (example):
# tok_emb.weight:           shape=torch.Size([50257, 768])
# pos_emb.weight:           shape=torch.Size([1024, 768])
# trf_blocks.0.att.W_query.weight: shape=torch.Size([768, 768])
# trf_blocks.0.att.W_key.weight:   shape=torch.Size([768, 768])
# trf_blocks.0.att.W_value.weight: shape=torch.Size([768, 768])
# ...
```

Every learnable parameter in the model has an entry in this dictionary:
- **Token embedding weights** — the 50,257 × 768 lookup table
- **Positional embedding weights** — the 1024 × 768 position table
- **Attention weights** — W_query, W_key, W_value, output projection for each of 12 layers
- **Feed-forward weights** — the two linear layers in each transformer block
- **Layer normalisation parameters** — scale and shift for each LayerNorm
- **Output head weights** — the final 768 × 50,257 projection

This is everything the model has learned. Save this dictionary, and you've saved the entire model's knowledge.

### What is NOT in the state dictionary

The state dictionary contains only **parameter values** — not:
- The model architecture (class definition, layer structure)
- Hyperparameters (learning rate, batch size)
- Optimizer state (gradients, momentum)
- Training history (epoch number, loss curves)

To fully reconstruct a training session, you need to save these separately (covered in Section 4).

---

## 3. Saving and Loading Model Parameters

### Saving — `torch.save`

```python
import torch

# Assume model is a trained GPTModel instance
torch.save(model.state_dict(), "model.pth")
```

**What this does:**
1. `model.state_dict()` — extracts the parameter dictionary
2. `torch.save(...)` — serialises the dictionary to disk using Python's `pickle` format
3. `"model.pth"` — the filename; `.pth` is PyTorch convention but any extension works

**File size:** For a 124M parameter model at 32-bit float:
```
124,000,000 parameters × 4 bytes/parameter = ~496 MB
```

### Loading — `model.load_state_dict`

```python
import torch

# Step 1: Create a fresh model instance with the SAME architecture
model = GPTModel(GPT_CONFIG_124M)

# Step 2: Load the saved parameters from disk
state_dict = torch.load("model.pth", map_location="cpu")
# map_location="cpu" ensures the model loads to CPU even if originally saved on GPU

# Step 3: Apply the loaded parameters to the model
model.load_state_dict(state_dict)

# Step 4: Set to evaluation mode for inference
model.eval()
```

**Critical requirement — architecture must match:** The model you create in Step 1 must have **exactly the same architecture** as the model that was saved. If the saved model had 12 transformer layers and you try to load into an 8-layer model, PyTorch will raise an error because the state dictionary keys won't match.

```python
# This will FAIL if architectures differ:
wrong_model = GPTModel({"n_layers": 8, ...})  # 8 layers
wrong_model.load_state_dict(state_dict)        # was saved from 12-layer model
# → RuntimeError: Missing keys / Unexpected keys
```

### What load_state_dict actually does internally

```python
# Conceptually, load_state_dict does:
for name, param in state_dict.items():
    # Find the corresponding layer in the model
    # Replace that layer's random initialisation with the saved value
    model_layer = get_layer_by_name(model, name)
    model_layer.data.copy_(param)
```

The model starts with random weights (from `GPTModel(config)`). `load_state_dict` replaces every single one of those random values with the values from the file. The result is a model with exactly the same parameters as when it was saved.

### Practical use case — sharing with a collaborator

```python
# Your end (you trained the model):
torch.save(model.state_dict(), "model.pth")
# Send model.pth to your collaborator (e.g., via email, cloud storage)

# Collaborator's end (they receive the file):
model = GPTModel(GPT_CONFIG_124M)          # they must use the same config!
model.load_state_dict(
    torch.load("model.pth", map_location="cpu")
)
model.eval()
# The collaborator now has your exact trained model
# They can generate text, fine-tune further, evaluate — all without retraining
```

---

## 4. Saving and Loading the Optimizer State

### Why the optimizer state matters

When you save only the model parameters and close your session, then resume training, there is a subtle but important problem: the **optimizer loses its history**.

### The AdamW optimizer and its internal state

AdamW (the optimizer used for GPT training) doesn't just track the current gradient. It maintains:

```
For each parameter w in the model:
  - m_t: First moment (exponential moving average of gradients)
          m_t = β₁ × m_{t-1} + (1 - β₁) × gradient
          
  - v_t: Second moment (exponential moving average of squared gradients)
          v_t = β₂ × v_{t-1} + (1 - β₂) × gradient²
          
  - Bias-corrected versions of both
  
  - Adaptive learning rate per parameter:
          lr_effective = lr × √(1 - β₂ᵗ) / (1 - β₁ᵗ)
          w = w - lr_effective × m_t / (√v_t + ε)
```

These historical values (`m_t`, `v_t`) accumulate over thousands of training steps. They represent the "momentum" the optimizer has built up — some parameters get larger effective learning rates, others get smaller, based on their gradient history.

**If you only save model parameters and reload:**
- Model weights ✓ (correct)
- Optimizer history ✗ (reset to zero)
- Adaptive learning rates ✗ (reset to initial values)
- The optimizer thinks it's back at step 0 — it has to rebuild its momentum from scratch
- Training may behave erratically or converge more slowly for many steps

### Saving both model and optimizer state

```python
# Save both together in a single checkpoint file
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)
```

The checkpoint is a dictionary containing two sub-dictionaries:

**`model_state_dict`** — all 124M parameter values (same as before)

**`optimizer_state_dict`** — contains:
```python
{
    "state": {
        0: {"step": 1000, "exp_avg": tensor(...), "exp_avg_sq": tensor(...)},
        1: {"step": 1000, "exp_avg": tensor(...), "exp_avg_sq": tensor(...)},
        # ... one entry per parameter group
    },
    "param_groups": [
        {
            "lr": 0.0004,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0.01,
            # ... all hyperparameters
        }
    ]
}
```

Every single `exp_avg` (first moment) and `exp_avg_sq` (second moment) value for every parameter is stored. This is why optimizer checkpoints can actually be larger than the model itself.

### Loading both model and optimizer state

```python
# Load the checkpoint file
checkpoint = torch.load("model_and_optimizer.pth", map_location="cpu")

# Reconstruct the model
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

# Reconstruct the optimizer
# Must use the same optimizer type and parameters as before
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.01)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Set to training mode to resume training
model.train()

# Now continue training from exactly where you left off
# The optimizer has full gradient history — behaves as if session never ended
for epoch in range(remaining_epochs):
    # training loop...
```

### `model.train()` vs `model.eval()` — clarification

```python
model.eval()   # Inference mode:
               # - Dropout disabled (all neurons active)
               # - BatchNorm uses running statistics
               # Use this when: generating text, evaluating accuracy

model.train()  # Training mode:
               # - Dropout active (randomly zeros neurons)
               # - BatchNorm uses batch statistics
               # Use this when: running forward + backward pass to update weights
```

`model.train()` does NOT start training. It only sets the model's internal mode flag. Training still requires you to explicitly run the forward pass, compute loss, call `.backward()`, and call `optimizer.step()`.

---

## 5. Complete Save/Load Patterns — Reference

### Pattern 1 — Save for inference (sharing a trained model)

```python
# SAVE (after training is complete)
torch.save(model.state_dict(), "model.pth")

# LOAD (for generating text or evaluation)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()
# Ready to generate text
```

Use this when: you're done training and just want to use or share the model.

### Pattern 2 — Save checkpoint (to resume training later)

```python
# SAVE (mid-training, e.g., after each epoch)
torch.save(
    {
        "epoch": current_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    },
    f"checkpoint_epoch_{current_epoch}.pth"
)

# LOAD (resuming training in a new session)
checkpoint = torch.load("checkpoint_epoch_5.pth", map_location="cpu")

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"] + 1   # resume from next epoch
model.train()
```

Use this when: training across multiple sessions, or when you want safety checkpoints in case of crashes.

### Pattern 3 — Loading pre-trained weights (from OpenAI or HuggingFace)

```python
# Download pre-trained weights (covered in next lecture)
# OpenAI provides GPT-2 weights in a specific format

# Load into our GPTModel architecture
model = GPTModel(GPT_CONFIG_124M)

# The weight loading process maps OpenAI's key names to our key names
# (covered in detail in the next lecture)
load_weights_into_gpt(model, openai_weights)

model.eval()
# Now generating coherent English text using OpenAI's trained weights
```

Use this when: you want to use someone else's trained weights in your own model architecture.

---

## 6. Common Errors and How to Fix Them

### Error 1 — Architecture mismatch

```
RuntimeError: Error(s) in loading state_dict for GPTModel:
    Unexpected key(s) in state_dict: "trf_blocks.12.att.W_query.weight"
    Missing key(s) in state_dict: ...
```

**Cause:** The saved model had a different architecture (e.g., 13 layers) than the model you're loading into (e.g., 12 layers).

**Fix:** Ensure the config used to create the model exactly matches the config used when saving:
```python
# Must use SAME config as when the model was saved
model = GPTModel(GPT_CONFIG_124M)  # not GPT_CONFIG_SMALL or any other variant
```

### Error 2 — Device mismatch

```
RuntimeError: Attempting to deserialize object on a CUDA device but 
torch.cuda.is_available() is False.
```

**Cause:** Model was saved on GPU, but you're loading on a CPU-only machine.

**Fix:** Use `map_location` parameter:
```python
# Always specify map_location to avoid device issues
state_dict = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state_dict)
# Then move to GPU if available:
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Error 3 — Forgetting model.eval() before inference

```python
# Without model.eval():
model.load_state_dict(...)
# model is in training mode by default after load_state_dict
# Dropout is ACTIVE → output changes on every call
out1 = generate_text(model, input)
out2 = generate_text(model, input)
# out1 != out2  ← same input, different outputs due to dropout randomness

# With model.eval():
model.load_state_dict(...)
model.eval()
out1 = generate_text(model, input)
out2 = generate_text(model, input)
# out1 == out2  ← deterministic ✓
```

---

## 7. File Format and Conventions

### The `.pth` extension

`.pth` stands for "PyTorch" and is the conventional extension for PyTorch saved files. However, it's purely convention — PyTorch will save and load any file extension:

```python
torch.save(model.state_dict(), "model.pth")      # convention
torch.save(model.state_dict(), "model.pt")       # also common
torch.save(model.state_dict(), "model.bin")      # HuggingFace uses this
torch.save(model.state_dict(), "weights.pkl")    # works, just unconventional
```

### What format is used internally?

`torch.save` uses Python's `pickle` serialisation format. The `.pth` file is a pickle file containing the OrderedDict of tensors. You could technically open it with:

```python
import pickle
with open("model.pth", "rb") as f:
    data = pickle.load(f)
```

But always use `torch.load` — it handles PyTorch tensor reconstruction correctly.

### Checkpoint naming best practice

```python
# Include epoch number and key metric in filename
torch.save(checkpoint, f"gpt2_epoch{epoch}_valloss{val_loss:.2f}.pth")
# e.g.: "gpt2_epoch5_valloss2.34.pth"

# This way you can identify which checkpoint was best
# and load specific checkpoints for comparison
```

---

## 8. Connection to the Next Lecture — Loading OpenAI's GPT-2 Weights

Everything in this lecture directly enables the next lecture. OpenAI has publicly released the pre-trained weights for GPT-2 (all 124M parameters, trained on 40GB of internet text). Loading these weights is exactly the save/load mechanism we just learned, with one additional complication:

**OpenAI's weight file uses different key names than our GPTModel.**

```
OpenAI's key:           Our GPTModel's key:
────────────────────────────────────────────────────────
"h.0.attn.c_attn.weight"  →  "trf_blocks.0.att.W_query.weight"
"h.0.mlp.c_fc.weight"     →  "trf_blocks.0.ff.layers.0.weight"
"wte.weight"              →  "tok_emb.weight"
...
```

The weights themselves are identical in value — only the dictionary key names differ. The next lecture will write a mapping function that translates OpenAI's key names to our key names, then uses `model.load_state_dict()` (which we just learned) to load the weights.

After loading OpenAI's weights, our GPT model — the exact architecture we built from scratch — will generate coherent, meaningful English text without any training on our part, because we're using weights that OpenAI trained on 40GB of text data.

---

## Key Concepts to Remember

- **`model.state_dict()`** — returns an OrderedDict mapping layer names to parameter tensors; captures the complete learned state of the model
- **`torch.save(state_dict, "file.pth")`** — serialises the state dictionary to disk using pickle format; `.pth` is convention but any extension works
- **`torch.load("file.pth", map_location="cpu")`** — deserialises the file back into a Python dictionary; always use `map_location` to avoid device errors
- **`model.load_state_dict(state_dict)`** — copies saved parameter values into the model, replacing its current (random) values; architecture must match exactly
- **Optimizer state dictionary** — stores not just hyperparameters but gradient history (`exp_avg`, `exp_avg_sq`); must be saved alongside model weights to resume training properly
- **AdamW stores per-parameter history** — first moment (gradient moving average) and second moment (squared gradient moving average); losing this forces the optimizer to rebuild momentum from scratch
- **`model.eval()`** — disables dropout for inference; call after loading weights before generating text
- **`model.train()`** — re-enables dropout; call before resuming training; does NOT start training by itself
- **Checkpoint pattern** — save both `model_state_dict` and `optimizer_state_dict` in one file when training across sessions
- **Architecture must match** — the model class and config used to create the model for loading must be identical to the one used when saving

---

*Lecture 32 | LLM from Scratch Series | Next: Loading Pre-trained GPT-2 Weights from OpenAI*
