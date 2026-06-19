# Lecture 40 — Loading Pre-trained Weights for Instruction Fine-Tuning
### (Deep notes, simple English)

> **Where we are:** Data is fully ready (Lectures 37-39). Now we load a pre-trained GPT-2 model. This lecture is about picking the right starting point before training — and understanding *why* each choice matters, not just doing it.

---

## 1. Why Use Pre-trained Weights? (The Real Reason)

The lecture says "it saves time." That's true. But here's the deeper reason, in simple words:

When a model is pre-trained, its weights are not random anymore. They already sit in a "good zone" — a place where the model already understands grammar, word meaning, and sentence structure.

If you started from random weights and tried to fine-tune using only 935 examples, the model would have to learn **basic language** and **the task** at the same time. 935 examples is way too few to learn a language from scratch.

> **Simple way to think about it:** Pre-training = the model already knows how to read and write well. Fine-tuning = we are now teaching this person our specific style of writing. You don't teach someone to read using only 935 sentences. You only teach style using 935 sentences — and that works fine.

### A warning to keep in mind for later

Because the model already knows a lot, if we fine-tune too aggressively (too high learning rate, too many epochs), we can damage what it already knows. This is called **catastrophic forgetting**. This is exactly why, in the next lecture, you will see:

- a **small learning rate**
- **very few epochs** (2-3, not 50)

Now you know *why* those choices will be made before you even see them.

---

## 2. Why GPT-2 Medium Instead of GPT-2 Small?

In the spam classifier project, the small model (124M parameters) worked fine. Here, it doesn't work well. Why?

Spam classification is simple: read an email, output one of two labels (spam/not spam). Easy task.

Instruction following is harder. The model must:
1. Understand a new instruction it has never seen exactly worded that way before.
2. Apply the right transformation (active → passive, translate, fix grammar, etc.) to new sentences.
3. Stay focused and not wander off into random text.

This needs more "thinking space" inside the model — more layers, more dimensions. A small model just doesn't have enough room to learn these general patterns well. A bigger model (355M) has more capacity to learn the *general skill* of following instructions, not just memorize the 935 examples it sees.

> **Simple rule:** Bigger models are better at learning a *general skill*. Smaller models are more likely to just memorize, which doesn't generalize to new instructions.

---

## 3. Setting Up the Model Config — Why Each Setting Matters

```python
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}
}
```

### Why `drop_rate = 0.0`?

Dropout randomly turns off some neurons during training. It's used to stop the model from memorizing too much during **pre-training**, where there is huge data.

But now we are fine-tuning on a tiny dataset (935 examples). If we turn dropout on here:
- It adds random noise to an already small and precious amount of training signal.
- The pre-trained weights were not "designed" to handle this kind of fine-tuning noise.

So we simply turn dropout off (`0.0`) during fine-tuning. Less noise, more stable training on small data.

### Why `qkv_bias = True`? (This one is NOT optional)

This is the most important setting to get right. Here's why:

OpenAI's GPT-2 weights were trained **with** bias terms in the query/key/value layers. If our model is built **without** bias terms (`qkv_bias = False`), then:
- The shapes won't match when we try to load the weights → you'll get an error.
- Even if you somehow force it to load, you'd be throwing away part of what makes those weights correct. The bias was trained together with the rest — removing it changes the math.

> **Simple rule:** When loading someone else's pre-trained weights, your model's structure (bias on/off, layer types, etc.) is not your choice anymore — it must match exactly what they trained with. Otherwise loading breaks or quietly gives you a broken model.

---

## 4. How the Weights Actually Get Loaded (Two Simple Steps)

```python
settings, params = download_and_load_gpt2(model_size="355M", models_dir="gpt2")
load_weights_into_gpt(model, params)
model.eval()
```

Even though it's two lines, two very different things are happening:

### Step 1: Download and convert the file format

OpenAI saved these weights using **TensorFlow's old file format**, not PyTorch's format. These two formats store the same numbers, but organize them completely differently — different file structure, different naming.

So the function `download_and_load_gpt2` does two jobs:
1. Downloads the raw files (7 files, ~1.42 GB for the medium model).
2. Reads those TensorFlow files and rearranges the numbers into a clean Python dictionary called `params`, in a structure that matches our own model's layout (token embeddings, position embeddings, each transformer block's weights, etc.)

Think of this as a **translator**. Same information, different language. The function translates TensorFlow's way of storing weights into something we can use in PyTorch.

### Step 2: Copy the numbers into our model

`load_weights_into_gpt` takes that `params` dictionary and copies each set of numbers into the matching part of our own model — our query weights get OpenAI's query weights, our embedding layer gets OpenAI's embedding layer, and so on.

### A tricky detail worth knowing (in simple terms)

OpenAI stores the Query, Key, and Value weights as **one single big matrix**. Our model (built earlier in this series) keeps them as **three separate matrices**. So the loading code has to:
- Take that one big matrix
- Cut it into three equal pieces
- Give each piece to the right place in our model

If this cutting is done wrong, the model won't crash — it will just quietly produce garbage. This is something to remember as a debugging clue:

> If a model "runs" without errors but gives weird/wrong output, suspect this kind of silent mistake before blaming your training code.

### Why `model.eval()`?

This tells the model: "We are testing now, not training." It turns off dropout (already 0 here, but good habit) and other training-only behavior. Always do this before generating text with a model you're not currently training.

---

## 5. Testing the Model BEFORE Fine-Tuning — Why This Step Is Not Optional

The lecture treats this test almost like a side note. As a careful engineer, you should treat this as a **required step**, every single time you fine-tune a model. Here's why.

### Reason 1 — You need a "before" picture

If you don't record what the model does *before* training, you cannot prove fine-tuning actually helped. You need a baseline to compare against.

### Reason 2 — It catches broken pipelines early

If something was wrong in the weight-loading step (like the QKV-splitting mistake mentioned above), this test would show **completely broken, incoherent text** — not just "wrong but readable" text. Since our test shows fluent English (just doing the wrong task), that tells us:
- Weight loading worked correctly.
- The only problem is: the model doesn't yet know how to *follow* instructions.

This is a very useful debugging skill — learn to tell the difference between these two types of failure:

| Type of failure | What it tells you |
|---|---|
| Output is broken, doesn't even look like real sentences | Something is wrong in your pipeline (loading, tokenizing, config) — fix this first |
| Output is fluent, readable, but does the wrong task | Pipeline is fine — you just need to fine-tune for the task |

### What actually happened in the test

```
Instruction: Convert the active sentence to passive: "The chef cooks the meal every day."
Expected:    "The meal is cooked by the chef every day."

Model said:  "The chef cooks the meal every day.
              ### Instruction:
              Convert the active sentence to passive..."
```

In simple words: the model just kept repeating the text it was given, and even started writing a fake new instruction block, instead of actually doing the task.

**Why does this happen?** The model is trained to predict "what comes next" in a believable way. Since instruction-style text (`### Instruction:` etc.) might look similar to patterns it saw in its training data, it just continues in that *style* — without understanding it's supposed to *act* on the instruction. It's copying the **shape** of the text, not doing the actual **job**.

This exact gap — knowing how to continue text vs. knowing how to follow a command — is what fine-tuning closes.

---

## 6. What You Should Now Be Able to Predict (Before the Next Lecture)

Because you understand the reasoning above, you can guess what's coming next, before it's shown:

- The learning rate will be **small** (to protect the pre-trained knowledge).
- Training will run for **only a few epochs** (small dataset, easy to overfit).
- Validation loss will be **watched closely** (catch overfitting early).
- The exact same test sentence ("The chef cooks the meal every day") will be tested **again after training**, to show a clear before/after comparison.

Being able to guess this in advance is a useful skill — it means you understand the reasons behind decisions, not just the steps.

---

## Summary Table

| Topic | Simple takeaway |
|---|---|
| Why use pre-trained weights | Model already knows language; we just teach it the task, not language from zero |
| Catastrophic forgetting | Training too hard can damage what the model already knows — keep LR and epochs low |
| Why Medium not Small | Bigger model = more room to learn the *general skill* of following instructions |
| `drop_rate = 0.0` | Small dataset + already-trained weights → dropout adds noise we don't want |
| `qkv_bias = True` | Must match OpenAI's format exactly, or loading breaks or quietly corrupts the model |
| TF → params dict | A translator step — converts OpenAI's old file format into something our model can use |
| Fused QKV splitting | OpenAI stores Q/K/V as one matrix; our loader must split it correctly — easy place for silent bugs |
| Why test before fine-tuning | Gives you a "before" to compare to, and proves your pipeline isn't broken |
| The actual failure | Model repeats input / writes fake instructions — it copies the text *style*, not the *task* |

---

## What's Next

Fine-tune the model on the training data with a small learning rate and few epochs. Watch the loss. Then test this exact same sentence again and see the difference.
