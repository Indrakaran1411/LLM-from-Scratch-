# Classification Fine-Tuning — Loss Function, Training Loop & Model Evaluation
### Lecture Notes | Build LLMs from Scratch Series — Lecture 36

> **Context:** Lectures 31–35 covered dataset prep, DataLoaders, model initialization, loading GPT-2 weights, and modifying the architecture with a classification head. This lecture completes the project: implementing cross-entropy loss, accuracy metrics, the full training loop with backpropagation, and finally testing the fine-tuned model on new unseen emails.

---

## 1. Recap — Where We Are

```
✅ Stage 1: Data Preparation
   - Downloaded SMS Spam Collection (UCI)
   - Balanced: 747 spam + 747 ham
   - Split: 70/10/20 → train/val/test
   - DataLoaders: batches of [8, 120] input + [8] labels

✅ Stage 2: Model Setup
   - Loaded GPT-2 small (124M) pre-trained weights
   - Replaced output head: Linear(768→50257) → Linear(768→2)
   - Froze 94.3% of parameters; only ~7M trainable
   - Model outputs shape [batch, seq_len, 2]
   - We use only the last token: output[:, -1, :]

🔲 Stage 3: Fine-Tuning (this lecture)
   - Step 7: Convert logits → class label predictions
   - Step 8: Implement accuracy metric
   - Step 9: Implement cross-entropy loss function
   - Step 10: Training loop with backpropagation
   - Step 11: Evaluate on full train/val/test sets
   - Step 12: Test on new unseen emails
```

---

## 2. Converting Model Output to Class Predictions

### What the model currently outputs

After the modified architecture, for any input email the model outputs:
```
Input: "You won the lottery"
Model output (last token): tensor([-3.5898, 3.9902])
                                      ↑        ↑
                                  ham score  spam score
```

These are raw **logits** — unnormalized scores. We need to convert them to a predicted class label.

### Method 1 — Softmax then argmax

```python
import torch.nn.functional as F

logits = torch.tensor([-3.5898, 3.9902])

# Convert to probabilities
probas = torch.softmax(logits, dim=-1)
print(probas)   # tensor([0.0005, 0.9995])
# 0.05% chance ham, 99.95% chance spam

# Pick the class with highest probability
predicted_class = torch.argmax(probas, dim=-1)
print(predicted_class.item())   # 1 → spam
```

### Method 2 — Argmax directly on logits (equivalent, simpler)

```python
# Softmax is monotonic — it preserves the ordering of values
# So argmax(logits) == argmax(softmax(logits)) always
predicted_class = torch.argmax(logits, dim=-1)
print(predicted_class.item())   # 1 → spam (same result, no softmax needed)
```

**Why are they equivalent?** Softmax applies `exp()` to every value and normalises. Since `exp()` is a monotonically increasing function, the largest logit always maps to the largest probability. The argmax position never changes.

In practice: use argmax directly for prediction. Softmax is only needed when you want the actual probability values for interpretation or when computing cross-entropy loss.

### Two concrete examples

```
Example 1:
  Input:  "You won the lottery"
  Logits: [-3.59, 3.99]
  Probas: [0.0005, 0.9995]
  Argmax: index 1 → SPAM ✓ (correct)

Example 2:
  Input:  "Do you have time?"
  Logits: [3.12, -2.85]
  Probas: [0.9978, 0.0022]
  Argmax: index 0 → HAM ✓ (correct)
```

---

## 3. Implementing the Accuracy Metric

### What accuracy measures

```
Accuracy = (Number of correct predictions) / (Total number of examples)

If model correctly classifies 970 out of 1,000 emails → accuracy = 97%
```

### Why not use accuracy as the loss function?

Accuracy is **not differentiable**. It's a step function — it either jumps from 0 to 1 or stays the same for each prediction. Step functions have gradient = 0 almost everywhere, which means backpropagation cannot compute useful gradient updates.

We need a smooth, differentiable loss function as a proxy for accuracy. Cross-entropy serves this role.

### `calculate_accuracy_loader` function

```python
def calculate_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    Calculate classification accuracy over a DataLoader.
    
    Args:
        data_loader: DataLoader yielding (input_batch, label_batch) pairs
        model:       Fine-tuned GPTModel with classification head
        device:      "cpu" or "cuda"
        num_batches: If specified, evaluate on only this many batches
                     (faster but less representative)
                     If None, evaluate on ALL batches
    
    Returns:
        accuracy: float, fraction of correct predictions [0.0, 1.0]
    """
    model.eval()   # disable dropout for deterministic evaluation
    
    correct_predictions = 0
    num_examples = 0
    
    # Determine how many batches to evaluate
    if num_batches is None:
        num_batches = len(data_loader)   # use all batches
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        
        input_batch = input_batch.to(device)   # [batch_size, seq_len]
        target_batch = target_batch.to(device)  # [batch_size]
        
        with torch.no_grad():   # no gradients needed for evaluation
            logits = model(input_batch)   # [batch_size, seq_len, 2]
        
        # Extract last token's logits → [batch_size, 2]
        last_token_logits = logits[:, -1, :]
        
        # Get predicted class for each email in the batch
        predicted_labels = torch.argmax(last_token_logits, dim=-1)   # [batch_size]
        
        # Compare predictions to ground truth
        correct_predictions += (predicted_labels == target_batch).sum().item()
        num_examples += input_batch.shape[0]   # add batch_size
    
    return correct_predictions / num_examples
```

### Step-by-step walkthrough for one batch

```
Batch with 8 emails:
  input_batch:  [8, 120]  — token IDs
  target_batch: [8]       — true labels [1, 0, 1, 1, 0, 0, 1, 0]

After model forward pass:
  logits: [8, 120, 2]

Extract last token:
  last_token_logits: [8, 2]
  = [[-3.59, 3.99],   ← email 1: spam predicted
     [ 3.12, -2.85],  ← email 2: ham predicted
     [-2.10, 2.87],   ← email 3: spam predicted
     ...
    ]

After argmax:
  predicted_labels: [1, 0, 1, 1, 0, 0, 1, 0]

Compare with target: [1, 0, 1, 1, 0, 0, 1, 0]
  Matches:          [✓, ✓, ✓, ✓, ✓, ✓, ✓, ✓]

correct_predictions += 8
num_examples += 8
```

### Initial accuracy (before training)

```python
# Before any fine-tuning:
train_acc = calculate_accuracy_loader(train_loader, model, device, num_batches=10)
val_acc   = calculate_accuracy_loader(val_loader,   model, device, num_batches=10)
test_acc  = calculate_accuracy_loader(test_loader,  model, device, num_batches=10)

print(f"Training accuracy:   {train_acc*100:.2f}%")   # ~46%
print(f"Validation accuracy: {val_acc*100:.2f}%")     # ~45%
print(f"Test accuracy:       {test_acc*100:.2f}%")    # ~48%
```

~46-48% accuracy — **worse than random guessing** (which would be 50% for balanced classes). This confirms the model's classification head and last transformer block are randomly initialised and need training.

---

## 4. The Cross-Entropy Loss Function

### Why cross-entropy for classification

For a classification task with predicted probabilities `p` and true labels `y`, the **categorical cross-entropy loss** is:

```
L = -Σ yᵢ × log(pᵢ)    (summed over all classes i)
```

For binary classification (spam/ham):
```
L = -(y_ham × log(p_ham) + y_spam × log(p_spam))
```

Since labels are one-hot (true class = 1, all others = 0), only one term survives:
```
L = -log(p_correct_class)
```

### Why this is a good loss function — intuition

```
Case 1: Model is very confident and CORRECT
  True label: spam (index 1)
  Predicted:  p_spam = 0.99
  Loss = -log(0.99) = 0.01   ← very low ✓

Case 2: Model is uncertain
  True label: spam (index 1)
  Predicted:  p_spam = 0.50
  Loss = -log(0.50) = 0.69   ← moderate

Case 3: Model is very confident and WRONG
  True label: spam (index 1)
  Predicted:  p_spam = 0.01
  Loss = -log(0.01) = 4.60   ← very high, strong gradient signal ✓
```

The `-log(x)` curve approaches 0 as x→1 (perfect prediction = 0 loss) and approaches infinity as x→0 (confident wrong prediction = massive loss). This strongly penalises confident wrong predictions.

### Numerical example from the lecture

```
True label: ham (not spam) → one-hot = [1, 0]
Predicted probabilities:   p_ham=0.8, p_spam=0.2

L = -(1 × log(0.8) + 0 × log(0.2))
  = -(log(0.8))
  = -(-0.2231)
  = 0.2231

If prediction were perfect (p_ham=1.0):
L = -(1 × log(1.0)) = -(0) = 0   ← zero loss when perfectly correct ✓
```

### Why not use accuracy directly as loss?

```python
# Accuracy is NOT differentiable:
# Small change in weight → prediction either stays same or flips
# Gradient is 0 almost everywhere → backprop can't improve it

# Cross-entropy IS differentiable:
# Small change in weight → smooth change in loss
# Gradient always exists → backprop can guide improvement
```

### Implementing loss functions

**Loss for a single batch:**
```python
def calculate_loss_batch(input_batch, target_batch, model, device):
    """
    Compute cross-entropy loss for one batch.
    
    Args:
        input_batch:  [batch_size, seq_len] token IDs
        target_batch: [batch_size] true class labels (0 or 1)
        model:        Modified GPTModel
        device:       "cpu" or "cuda"
    
    Returns:
        loss: scalar tensor (differentiable)
    """
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    # Forward pass
    logits = model(input_batch)     # [batch_size, seq_len, 2]
    
    # Extract last token → [batch_size, 2]
    last_token_logits = logits[:, -1, :]
    
    # Cross-entropy loss
    # torch.nn.functional.cross_entropy expects:
    #   input:  [batch_size, num_classes]  — the logits (NOT softmax output)
    #   target: [batch_size]               — integer class indices
    # It internally applies softmax + log + negative → fully handles everything
    loss = torch.nn.functional.cross_entropy(last_token_logits, target_batch)
    
    return loss
```

**Important:** `F.cross_entropy` takes raw logits (not softmax output). It applies log-softmax internally for numerical stability. Never apply softmax before `F.cross_entropy`.

**Loss over multiple batches:**
```python
def calculate_loss_loader(data_loader, model, device, num_batches=None):
    """
    Compute average cross-entropy loss over a DataLoader.
    """
    total_loss = 0.0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calculate_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    
    return total_loss / num_batches   # average loss per batch
```

### Initial loss (before training)

```python
train_loss = calculate_loss_loader(train_loader, model, device, num_batches=5)
val_loss   = calculate_loss_loader(val_loader,   model, device, num_batches=5)
test_loss  = calculate_loss_loader(test_loader,  model, device, num_batches=5)

# Initial values (untrained model):
# train_loss ≈ 0.69   (log(2) ≈ 0.693 — exactly random binary classification)
# val_loss   ≈ 0.69
# test_loss  ≈ 0.69
```

Loss ≈ 0.693 = ln(2) is exactly what you'd expect from a model that assigns 50% probability to each class — confirming the classification layers are randomly initialised.

---

## 5. The Training Loop — Fine-Tuning with Backpropagation

### The gradient descent update rule

At each training step, for each trainable parameter `w`:
```
w_new = w_old - α × (∂L/∂w)

where:
  α    = learning rate (how large a step to take)
  ∂L/∂w = gradient of loss with respect to this parameter
```

With 7M trainable parameters, PyTorch computes all 7M gradients simultaneously via backpropagation.

### The AdamW optimizer

We use **AdamW** instead of vanilla gradient descent:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,          # learning rate (small for fine-tuning)
    weight_decay=0.1  # L2 regularisation to prevent overfitting
)
```

AdamW maintains per-parameter state:
- `m_t` — first moment: exponential moving average of gradients (momentum)
- `v_t` — second moment: exponential moving average of squared gradients (adaptive LR)

The effective update rule per parameter:
```
m_t = β₁ × m_{t-1} + (1 - β₁) × gradient        (β₁ = 0.9)
v_t = β₂ × v_{t-1} + (1 - β₂) × gradient²        (β₂ = 0.999)

w_new = w_old - lr × m_t / (√v_t + ε)  - weight_decay × w_old
```

Benefits over vanilla gradient descent:
- **Adaptive learning rates** — parameters with large historical gradients get smaller effective LR; rare parameters get larger LR
- **Momentum** — smooths gradient updates, reduces oscillation
- **Weight decay** — shrinks weights slightly each step, preventing overfitting
- **Rarely gets stuck** in local minima compared to vanilla gradient descent

### The `train_classifier_simple` function

```python
def train_classifier_simple(model, train_loader, val_loader, optimizer,
                             device, num_epochs, eval_freq, eval_iter):
    """
    Fine-tune the GPT classification model.
    
    Args:
        model:        Modified GPTModel with classification head
        train_loader: DataLoader for training data
        val_loader:   DataLoader for validation data
        optimizer:    AdamW optimizer
        device:       "cpu" or "cuda"
        num_epochs:   Number of complete passes through training data
        eval_freq:    Print train/val loss every eval_freq batches
        eval_iter:    Number of batches to use for loss evaluation during training
                      (smaller = faster but less accurate estimate)
    
    Returns:
        train_losses, val_losses, train_accs, val_accs, examples_seen
    """
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    examples_seen = 0     # total training examples processed
    global_step = -1
    
    for epoch in range(num_epochs):
        
        # ── TRAINING MODE ──────────────────────────────────────────────────
        model.train()   # enable dropout (if any), enable gradient tracking
        
        for input_batch, target_batch in train_loader:
            
            # Step 1: Zero out gradients from previous batch
            # CRITICAL: PyTorch accumulates gradients by default
            # If you forget this, gradients from multiple batches add up → wrong updates
            optimizer.zero_grad()
            
            # Step 2: Forward pass + compute loss
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            
            # Step 3: Backward pass — compute ∂L/∂w for all trainable parameters
            # PyTorch traverses the computation graph in reverse, applying chain rule
            loss.backward()
            
            # Step 4: Update parameters using computed gradients
            # w_new = w_old - lr × gradient  (simplified; AdamW adds momentum etc.)
            optimizer.step()
            
            # Tracking
            examples_seen += input_batch.shape[0]   # add batch_size
            global_step += 1
            
            # ── PERIODIC EVALUATION ────────────────────────────────────────
            # Print train and validation loss every eval_freq batches
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calculate_loss_loader(train_loader, model,
                                                       device, eval_iter)
                    val_loss   = calculate_loss_loader(val_loader,   model,
                                                       device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} | Step {global_step:06d} | "
                      f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f}")
                model.train()   # back to training mode
        
        # ── END OF EPOCH: compute accuracy ─────────────────────────────────
        model.eval()
        with torch.no_grad():
            train_acc = calculate_accuracy_loader(train_loader, model,
                                                   device, eval_iter)
            val_acc   = calculate_accuracy_loader(val_loader,   model,
                                                   device, eval_iter)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}: Train acc={train_acc*100:.2f}%, "
              f"Val acc={val_acc*100:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen
```

### Why `optimizer.zero_grad()` must be called every batch

PyTorch **accumulates gradients by default** — if you don't zero them, each `.backward()` call adds to the existing gradients rather than replacing them:

```python
# WRONG (gradients accumulate across batches):
for batch in train_loader:
    loss = calculate_loss_batch(batch)
    loss.backward()     # gradients ADD to previous
    optimizer.step()    # updates based on summed gradients → wrong

# CORRECT (gradients reset each batch):
for batch in train_loader:
    optimizer.zero_grad()   # clear gradients from last batch
    loss = calculate_loss_batch(batch)
    loss.backward()         # compute fresh gradients for this batch
    optimizer.step()        # update based on this batch's gradients only
```

### Running the training

```python
import time

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

# Train
start = time.time()
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=5,
    eval_freq=50,    # print loss every 50 batches
    eval_iter=5      # use 5 batches for quick loss estimates during training
)
end = time.time()
print(f"Training time: {(end - start)/60:.2f} minutes")
# → Training time: 8.83 minutes (MacBook Air 2020)
```

---

## 6. Training Results and Interpretation

### Loss curves

```
During training, printed every 50 batches:

Ep 1 | Step 000050 | Train loss: 0.531 | Val loss: 0.498
Ep 1 | Step 000100 | Train loss: 0.312 | Val loss: 0.289
Ep 2 | Step 000150 | Train loss: 0.198 | Val loss: 0.187
...
Ep 5 | Step 000650 | Train loss: 0.083 | Val loss: 0.074

Final: train_loss=0.083, val_loss=0.074
```

### Accuracy curves (per epoch)

```
Epoch 1: Train acc=85.4%,  Val acc=83.2%
Epoch 2: Train acc=92.1%,  Val acc=90.8%
Epoch 3: Train acc=96.3%,  Val acc=95.1%
Epoch 4: Train acc=98.7%,  Val acc=96.9%
Epoch 5: Train acc=99.8%,  Val acc=97.5%
```

### Interpreting the results

**Loss curves — what good training looks like:**
```
Good training:                    Overfitting:
─────────────                     ────────────
Train loss ↘                      Train loss ↘
Val loss   ↘  (tracks train)      Val loss   ↗  (diverges)

Both converge to low values       Large gap between train and val
```

Our results show both losses decreasing together — minimal overfitting.

**Why val_loss (0.074) < train_loss (0.083)?**
This can happen when validation loss is measured at a point after some training has occurred within an epoch. It doesn't indicate a problem — the difference is small.

### Full dataset evaluation (after training)

```python
# Evaluate on ALL batches — more accurate than the 5-batch estimates during training
model.eval()
with torch.no_grad():
    train_acc = calculate_accuracy_loader(train_loader, model, device)  # all 130 batches
    val_acc   = calculate_accuracy_loader(val_loader,   model, device)  # all 19 batches
    test_acc  = calculate_accuracy_loader(test_loader,  model, device)  # all 38 batches

print(f"Training accuracy:   {train_acc*100:.2f}%")   # → 97.0%
print(f"Validation accuracy: {val_acc*100:.2f}%")     # → 97.3%
print(f"Test accuracy:       {test_acc*100:.2f}%")    # → 95.0%
```

### Why test accuracy (95%) < training accuracy (97%)

A 2% gap between training and test accuracy is a small but expected sign of slight overfitting. Possible causes:
- Only 1,045 training samples — relatively small for fine-tuning
- The model has seen training data multiple times (5 epochs)
- Test set contains novel phrasing the model didn't encounter during training

**How to reduce this gap (experimentation suggestions):**
```python
# Option 1: Add more dropout
BASE_CONFIG["drop_rate"] = 0.1   # was 0.0

# Option 2: Increase weight decay (stronger regularisation)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.3)

# Option 3: Train fewer epochs (early stopping)
num_epochs = 3   # stop before overfitting begins

# Option 4: Unfreeze more transformer blocks
for param in model.trf_blocks[-3:].parameters():   # unfreeze last 3 blocks
    param.requires_grad = True

# Option 5: Use more training data (data augmentation or larger dataset)
```

---

## 7. Testing on New Unseen Emails

### The `classify_review` function

```python
def classify_review(text, model, tokenizer, device, max_length=None,
                    pad_token_id=50256):
    """
    Classify a single email as spam or ham.
    
    Args:
        text:          Raw email string
        model:         Fine-tuned GPTModel
        tokenizer:     tiktoken GPT-2 tokenizer
        device:        "cpu" or "cuda"
        max_length:    Sequence length to pad/truncate to
                       (should match training max_length = 120)
        pad_token_id:  Padding token (50256 = <|endoftext|>)
    
    Returns:
        "spam" or "not spam"
    """
    model.eval()
    
    # Step 1: Tokenize the input text
    input_ids = tokenizer.encode(text)
    
    # Step 2: Determine max_length (from model's positional embedding)
    # model.pos_emb.weight.shape = [context_length, emb_dim] = [1024, 768]
    supported_context_length = model.pos_emb.weight.shape[0]   # 1024
    
    # Use the smaller of: training max_length or model's context length
    if max_length is None:
        max_length = supported_context_length
    else:
        max_length = min(max_length, supported_context_length)
    
    # Step 3: Truncate if too long
    input_ids = input_ids[:max_length]
    
    # Step 4: Pad if too short
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    # e.g., [5716, 257, 1479, ..., 50256, 50256, ..., 50256] → length 120
    
    # Step 5: Convert to tensor with batch dimension
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    # shape: [1, 120]
    
    # Step 6: Forward pass
    with torch.no_grad():
        logits = model(input_tensor)   # [1, 120, 2]
    
    # Step 7: Extract last token, get predicted class
    last_token_logits = logits[:, -1, :]       # [1, 2]
    predicted_label = torch.argmax(last_token_logits, dim=-1).item()   # 0 or 1
    
    return "spam" if predicted_label == 1 else "not spam"
```

### Testing on real examples

```python
# Example 1: Clear spam
text1 = ("You are a winner! You have been specifically selected "
         "to receive $1,000 cash or $2,000 award.")
result1 = classify_review(text1, model, tokenizer, device, max_length=120)
print(f"'{text1[:50]}...' → {result1}")
# → 'You are a winner! You have been specifically sel...' → spam ✓

# Example 2: Legitimate message
text2 = "Hey, just wanted to check if we are still on for dinner tonight. Let me know!"
result2 = classify_review(text2, model, tokenizer, device, max_length=120)
print(f"'{text2}' → {result2}")
# → 'Hey, just wanted to check if we are still on for dinner tonight...' → not spam ✓
```

**Both correct!** The fine-tuned model successfully classifies:
- Obvious spam (lottery winner, cash prize) → spam
- Normal social message → not spam

This is data the model has never seen during training (these are from the test set).

---

## 8. Saving the Fine-Tuned Model

```python
# Save the fine-tuned model for reuse
torch.save(model.state_dict(), "spam_classifier_finetuned.pth")

# To reload later:
model_loaded = GPTModel(BASE_CONFIG)
model_loaded.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], num_classes, bias=False)
model_loaded.load_state_dict(torch.load("spam_classifier_finetuned.pth",
                                         map_location=device))
model_loaded.eval()
# model_loaded is now the same fine-tuned model — no retraining needed
```

**Why saving is critical:** Without saving, you'd need to re-run 8+ minutes of training every time you want to use the classifier. Saving preserves all 7M fine-tuned parameter values permanently.

---

## 9. Complete Project Summary — All 12 Steps

```
Step 1:  Download SMS Spam Collection dataset (UCI repository)
Step 2:  Balance: 4825→747 ham by undersampling; total 1494
Step 3:  Encode labels (ham=0, spam=1), split 70/10/20
Step 4:  Create SpamDataset: tokenize + pad to 120 tokens
Step 5:  Create DataLoaders: [8, 120] input + [8] labels per batch
Step 6:  Initialize GPTModel with GPT-2 Small (124M) config
Step 7:  Load pre-trained OpenAI GPT-2 weights
Step 8:  Modify architecture: replace Linear(768→50257) with Linear(768→2)
         Freeze 94.3% of params; keep last transformer block + LayerNorm + head
Step 9:  Implement cross-entropy loss (differentiable proxy for accuracy)
Step 10: Implement accuracy metric (argmax of last token logits vs true label)
Step 11: Train with AdamW for 5 epochs; monitor train/val loss + accuracy
         Training time: ~9 minutes on CPU
Step 12: Evaluate on full dataset: train=97%, val=97%, test=95%
Step 13: Test on new unseen emails → correct predictions ✓
Step 14: Save model weights to .pth file for reuse

Final result:
  Pre-training:   GPT-2 knows English but cannot classify spam
  Fine-tuning:    GPT-2 modified + trained → 95% test accuracy spam classifier
```

---

## 10. Key Takeaways and Generalisation

### This exact pipeline works for ANY classification task

Just change:
1. The dataset (spam → medical text, tweets, reviews, etc.)
2. `num_classes` (2 → 3, 5, 10, etc.)
3. The label encoding

Everything else stays the same:
```python
# For sentiment analysis (3 classes: positive/neutral/negative):
num_classes = 3
model.out_head = torch.nn.Linear(768, 3, bias=False)

# For topic classification (10 classes):
num_classes = 10
model.out_head = torch.nn.Linear(768, 10, bias=False)
```

### Pre-training vs fine-tuning — why both are needed

```
Without fine-tuning:
  model.eval()
  prompt = "Is this spam? Answer yes or no. Text: You won $1000!"
  → Model generates random next tokens, cannot follow instructions ✗

With classification fine-tuning:
  classify_review("You won $1000!")
  → Model outputs [0.001, 0.999] → argmax → spam ✓ (95% accuracy)
```

The pre-trained model understands language but lacks task structure. Fine-tuning adds the task structure by training the classification head and last transformer block on labelled data.

---

## Key Concepts to Remember

- **Logits → class prediction** — `argmax(logits)` directly gives predicted class; softmax not needed for prediction (only for probability interpretation or loss computation)
- **Cross-entropy loss** formula: `L = -log(p_correct_class)`; 0 when perfectly correct, → ∞ when confidently wrong; differentiable unlike accuracy
- **`F.cross_entropy` takes raw logits** — never apply softmax before it; it applies log-softmax internally for numerical stability
- **Initial loss ≈ 0.693** — `ln(2)` for balanced binary classification with random weights; confirms untrained classification head
- **`optimizer.zero_grad()`** — must be called before every `.backward()`; PyTorch accumulates gradients by default
- **The 4-step training loop** — `zero_grad()` → forward pass + loss → `loss.backward()` → `optimizer.step()`
- **AdamW over vanilla SGD** — adaptive per-parameter learning rates, momentum, weight decay; rarely gets stuck in local minima
- **`eval_freq` and `eval_iter`** — control how often and how many batches to use for loss estimates during training; full dataset evaluation done separately after training
- **Training result** — train loss 0.69→0.083, val loss 0.69→0.074; train acc→97%, val acc→97%, test acc→95%
- **2% train-test gap** — small overfitting; reduce with dropout, weight decay, fewer epochs, or more data
- **`classify_review`** — tokenize → pad to max_length → forward pass → `argmax(output[:, -1, :])` → class label
- **Save after fine-tuning** — `torch.save(model.state_dict(), "model.pth")`; prevents re-running expensive training
- **Generalises to any classification** — change dataset + `num_classes`; same architecture and training code

---

*Lecture 36 | LLM from Scratch Series | Next: Instruction Fine-Tuning — Building a Chatbot*
