# Lecture 41 — Instruction Fine-Tuning: The Training Loop
### (Deep notes, simple English)

> **Where we are:** Data ready (Lectures 37-39). Pre-trained weights loaded into our model (Lecture 40). We saw the model fail to convert "The chef cooks the meal every day" into passive voice. Now we actually train the model on our 1100 instruction-response pairs. This is the moment everything we built so far gets used.

---

## 1. Why This Step Even Works — Connect the Dots First

Before the code, get this straight in your head, because it explains everything that follows:

- Pre-training gave the model knowledge of language — grammar, word meaning, sentence flow.
- It did **not** teach the model to follow commands.
- Fine-tuning does **not** teach language from zero. It only nudges the already-trained weights so they learn the new skill: "read an instruction, then produce the matching output."

This is why fine-tuning is fast (one epoch, a few hours on a basic laptop) compared to pre-training (which took OpenAI huge compute and huge data). We are not starting over — we are adjusting.

> **Simple rule:** Pre-training = learn the language. Fine-tuning = learn the task. Always do pre-training first, fine-tuning second — never the other way around.

---

## 2. The Training Loop — Same Loop as Pre-training, Just a Different Dataset

This is one of the most important points in the lecture, and it's easy to miss:

**The training loop code and the loss function are exactly the same as what was used in pre-training.** Nothing new is invented here. Only the data changes.

### The loop, step by step

```
For each epoch:
    For each batch in training data:
        1. Reset gradients to zero
        2. Calculate loss on this batch
        3. Backward pass (calculate gradients)
        4. Optimizer step (update the weights)
```

### What "backward pass" really means here

The model has 355 million parameters. The backward pass calculates **how much each one of those 355 million numbers should change** to reduce the loss. That's the partial derivative of the loss with respect to every parameter — done automatically by PyTorch's autograd.

### What the optimizer does with those gradients

The simplest version of weight updating is:

```
new_weight = old_weight - learning_rate × gradient
```

This is called **gradient descent**. Picture the loss as a valley. You start somewhere on the slope, and each step nudges you a little further down toward the lowest point (the minimum loss).

We don't use plain gradient descent though — we use **AdamW**.

### Why AdamW instead of plain gradient descent?

AdamW keeps track of two things as it trains:
- The recent direction of the gradient (momentum) — helps avoid getting stuck or moving too slowly.
- The recent size of the gradient squared — helps it take bigger steps where the slope is gentle, smaller steps where it's steep.

The "W" in AdamW stands for **weight decay** — a small penalty added to keep weights from growing too large, which helps prevent overfitting.

> **Simple takeaway:** AdamW is just a smarter, faster version of basic gradient descent. It's the standard choice in almost all modern deep learning, not something special to this lecture.

---

## 3. Where the Loss Actually Comes From

This connects directly back to the DataLoader work from Lecture 38-39. Here's the chain:

```
Batch → Input tensor + Target tensor
Input tensor  → passed through the GPT model → logits → probabilities (predicted values)
Target tensor → the "true" correct next tokens (already prepared in Lecture 38)
Predicted vs Target → Cross-Entropy Loss
```

**Cross-entropy loss in one sentence:** it measures how far the model's predicted probability for the *correct* token is from 1.0. If the model gives the correct token a probability close to 1, loss is close to 0. If it gives the correct token a low probability, loss is high.

This is the exact same loss function used in pre-training. Nothing new to learn here — just reused.

### Reusing the same functions

Two functions carry over directly from pre-training:

```python
calc_loss_batch(input_batch, target_batch, model, device)
# computes cross-entropy loss for ONE batch

calc_loss_loader(data_loader, model, device, num_batches=None)
# averages calc_loss_batch over many batches in a loader
# (sum of all batch losses) / (number of batches)
```

You pass `train_loader` to get training loss, or `val_loader` / `test_loader` to get validation/test loss. Same function, different input.

---

## 4. Setting Up the Training Run

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 1
eval_freq = 5
eval_iter = 5

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs,
    eval_freq=eval_freq,
    eval_iter=eval_iter,
    start_context=format_input(val_data[0]),
    tokenizer=tokenizer
)
```

### What each setting actually controls (in plain words)

| Setting | What it means | Why this value |
|---|---|---|
| `lr = 0.00005` | How big each weight update step is | Small — protects the pre-trained knowledge from being damaged |
| `weight_decay = 0.1` | Penalty to keep weights from growing too big | Standard AdamW default-ish value, reduces overfitting |
| `num_epochs = 1` | How many full passes through training data | Kept low on purpose — small dataset (935 examples), more epochs risks memorizing instead of learning the general skill |
| `eval_freq = 5` | After every 5 batches, print the loss | Just for monitoring — lets you watch progress without waiting till the end |
| `eval_iter = 5` | When checking loss, only check 5 batches (not the whole loader) | Makes evaluation fast — a quick estimate is enough during training |
| `start_context` | A real example from validation data to generate from during training | Lets you literally watch the model's answer to one question improve epoch by epoch |

**Important mindset:** these are hyperparameters — numbers you're free to change. They aren't fixed laws. The lecture explicitly encourages experimenting with different learning rates, weight decay, and epoch counts to see how results change. That experimentation is how you build real intuition.

---

## 5. Checking the Starting Point (Before Training)

Before running the loop, the loss was checked once with **no training done yet**:

```
Training loss:   3.82
Validation loss: 3.76
```

This matches exactly what we discussed in Lecture 40 — always measure a baseline before training, so you have a number to compare against later.

---

## 6. Why Training Takes So Long on a Basic Laptop (Practical Reality)

This part matters if you're following along without a GPU:

- **MacBook Air 2020, 8GB RAM, CPU only, 1 epoch → about 2 hours.**
- Trying 2 epochs on this same machine actually crashed the system — too much memory pressure.
- This is mentioned deliberately to show: **you don't need a fancy GPU to do real fine-tuning.** It just takes patience.

If you do have GPU access (Google Colab, AWS EC2, etc.), training will be much faster — this is recommended if available, but not required to follow along.

---

## 7. The Result — Before vs After Fine-Tuning

This is the actual proof that fine-tuning works. Same exact test sentence used in Lecture 40:

```
Instruction: Convert the active sentence to passive.
Input:       "The chef cooks the meal every day."
```

| Stage | Model's response |
|---|---|
| **Before fine-tuning** | "The chef cooks the meal every day. ### Instruction: Convert the active sentence to passive..." (just repeated input, ignored the task) |
| **After 1 epoch of fine-tuning** | "The meal is prepared every day by the chef." (almost perfect passive voice!) |
| **After 2 epochs** (tested separately on a better machine) | "The meal is cooked by the chef every day." (exactly correct) |

**What this tells you in simple words:** even just one epoch on a CPU laptop is enough to teach the model the basic skill of following this kind of instruction. More epochs (when compute allows) sharpens the answer further — "prepared" became "cooked," matching the exact expected word.

### Another example shown

```
Instruction: Rewrite the sentence using a simile.
Input:       "The car is very fast."
Expected:    "The car is as fast as lightning."
Model said:  "The car is as fast as a bullet."
```

Not word-for-word identical to the expected answer, but it correctly understood and applied the *task* — that's the real win. The model learned the **skill** of writing similes, not just one memorized sentence.

---

## 8. Reading the Loss Curve Correctly

After training, both training loss and validation loss were plotted over time.

**How to read this plot like an engineer, not just "loss go down good":**

- **Early in training:** loss drops fast. This means the model is quickly picking up the basic shape of the task — recognizing the instruction format, learning where the response should start, etc.
- **Later in training:** loss drops more slowly. The model has already learned the easy, general patterns. Now it's fine-tuning smaller details — exact wording, nuance — and settling into a stable solution.
- **Validation loss still a bit higher than training loss, and both still trending down:** this means the model hasn't reached its full potential yet. With more epochs (if compute allowed), both losses could likely go lower still. This is not a sign of overfitting yet — overfitting would look like validation loss going *up* while training loss keeps going *down*. That's not what's happening here.

> **Simple way to remember this:** Fast drop = learning the big patterns. Slow drop = polishing the details. Validation loss rising while training loss falls = overfitting warning sign (not seen here).

---

## 9. Why "Loss Looks Good" Isn't the Whole Story

Even though both losses dropped nicely, the lecture is careful to point out: **a good loss number does not automatically mean the responses are good.**

Loss is a mathematical, easy-to-measure number. But judging whether a sentence "makes sense" or "correctly answers the instruction" is a **qualitative** judgment — much harder to measure with a single number.

This is exactly why a separate, dedicated topic exists in LLM development: **LLM evaluation**. It's an active area of research because there's no single perfect way to score open-ended text responses the way you can score a simple yes/no classification.

This lecture relies on manually reading a few example outputs to judge quality. The next stage of this project (coming in a later lecture) will look at more structured ways to evaluate LLM responses.

---

## Summary Table

| Topic | Simple takeaway |
|---|---|
| Training loop | Exactly the same loop used in pre-training — only the dataset changed |
| Backward pass | Calculates how each of the 355M weights should change to reduce loss |
| AdamW | Smarter gradient descent — tracks momentum and adapts step size; "W" = weight decay |
| Loss function | Cross-entropy — same one used in pre-training, no new math here |
| `lr = 0.00005` | Small on purpose — protects pre-trained knowledge from damage |
| `num_epochs = 1` | Small dataset → keep epochs low to avoid memorizing instead of generalizing |
| Before training | Baseline loss recorded first (3.82 train / 3.76 val) — always do this |
| After 1 epoch | Model goes from repeating input → producing near-correct passive voice |
| Reading the loss curve | Fast drop = learning big patterns; slow drop = polishing details |
| Loss vs response quality | Good loss numbers ≠ guaranteed good answers — quality needs separate evaluation |

---

## What's Next

Stage 3: Evaluating the fine-tuned LLM's responses properly — moving beyond "this looks good" to a more structured way of scoring instruction-following quality.
