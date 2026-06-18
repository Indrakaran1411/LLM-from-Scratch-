# Lecture 37 — Instruction Fine-Tuning: Dataset Download & Formatting

> **Series position:** You have already built the full GPT architecture, pre-trained it, loaded OpenAI GPT-2 weights, and completed a classification fine-tuning project (spam detection). This lecture starts the **instruction fine-tuning** chapter — a new type of fine-tuning that turns a text-completion model into a personal assistant.

---

## 1. Why Instruction Fine-Tuning? (The Core Problem)

Your pre-trained model is good at one thing: **predicting the next token**. Give it a sentence fragment and it continues it. That's it.

But a real assistant needs to **follow commands**:

| User says | Pre-trained model does | What we want |
|-----------|------------------------|--------------|
| "Fix the grammar in this sentence" | Continues generating more text | Corrects only the grammar |
| "Convert active voice to passive" | Ignores the instruction | Transforms the sentence |
| "Make this more concise" | Keeps rambling | Shortens it |

The model was never trained to treat text as an **instruction + expected response pair**. It just saw raw text during pre-training. Instruction fine-tuning fixes this by training on thousands of (instruction → correct response) examples.

> **Key insight:** Pre-training teaches the model *language*. Instruction fine-tuning teaches the model *behaviour*.

---

## 2. What Is Instruction Fine-Tuning? (Formal Definition)

Also called **Supervised Instruction Fine-Tuning (SIFT)**. You provide the model with explicit input-output pairs:

```
Input:  [instruction] + [optional context/input]
Output: [desired response]
```

It's "supervised" because every training example has a known correct answer — unlike pre-training which is self-supervised (predict next token, no labels needed).

### Real-world use cases

**E-commerce chatbot (e.g. Walmart):**
- Pre-trained model has no knowledge of Walmart's return policy, shipping rules, or product catalog.
- Fine-tune with instructions like: *"If user asks about return policy, respond with steps to initiate a return."*
- Result: The bot answers company-specific questions accurately.

**Healthcare virtual assistant:**
- Pre-trained model has general medical knowledge but not hospital-specific protocols.
- Fine-tune with instructions tied to treatment plans, appointment scheduling, medication reminders.
- Result: The assistant gives personalised advice following the provider's guidelines.

> **Pattern:** Fine-tuning is needed when (1) you need instruction-following behaviour, or (2) you have domain-specific private data the model never saw during pre-training.

---

## 3. The Dataset: What It Looks Like

The dataset used in this lecture has **1,100 instruction-response pairs** stored as a JSON file. Each entry is a Python dictionary with three keys:

```python
{
  "instruction": "Edit the following sentence for grammar",
  "input":       "He go to the park every day",
  "output":      "He goes to the park every day"
}
```

### The three fields explained

| Field | Purpose | Can it be empty? |
|-------|---------|-----------------|
| `instruction` | The task the model must perform | No — always present |
| `input` | Extra context or the text to act on | Yes — optional |
| `output` | The correct response | No — always present |

### When `input` is empty

Some instructions have a single fixed answer and don't need extra context:

```python
{
  "instruction": "Convert 45 km to meters",
  "input":       "",         # no extra input needed
  "output":      "45 km is 45,000 meters"
}
```

When `input` is empty you skip that section in the prompt (shown in Section 5 below).

### Loading the dataset

```python
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

data = download_and_load_file("instruction-data.json", url)
print(len(data))   # → 1100
```

The function checks if the file already exists before downloading — avoids re-downloading every run.

---

## 4. Why You Can't Feed Raw JSON to the LLM

You might think: just feed the dictionary directly. You can't. The LLM only accepts **text tokens**. The dictionary structure (curly braces, colons, keys) is meaningless to it.

More importantly, researchers found through experimentation that **how you structure the prompt matters a lot** for training quality. A specific format helps the model learn the pattern:
*"When I see this structure → I should produce a response in this style."*

This is why prompt formatting standards exist.

---

## 5. Prompt Formatting: Alpaca Style (What We Use)

The most widely adopted format comes from **Stanford Alpaca** (29k+ GitHub stars, 4k+ forks). It wraps the instruction/input/output into a structured template:

### When `input` IS present:

```
Below is an instruction that describes a task, paired with an input 
that provides further context. Write a response that appropriately 
completes the request.

### Instruction:
Edit the following sentence for grammar

### Input:
He go to the park every day

### Response:
He goes to the park every day
```

### When `input` is NOT present:

```
Below is an instruction that describes a task. Write a response 
that appropriately completes the request.

### Instruction:
Convert 45 km to meters

### Response:
45 km is 45,000 meters
```

The `### Response:` section is what the model must learn to generate. Everything before it is the prompt fed as input.

---

## 6. The `format_input()` Function — Line by Line

```python
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task"
    )

    # Only add "paired with an input" clause if input exists
    if entry["input"]:
        instruction_text += (
            f", paired with an input that provides further context."
            f" Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
            f"\n\n### Input:\n{entry['input']}"
        )
    else:
        instruction_text += (
            f". Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )

    return instruction_text
```

**Usage — with input:**
```python
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)
```

Output:
```
Below is an instruction that describes a task, paired with an input...

### Instruction:
Identify the correct spelling of the following word

### Input:
Occasion

### Response:
The correct spelling is "occasion"
```

**Usage — without input:**
```python
model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)
```

Output:
```
Below is an instruction that describes a task...

### Instruction:
What is the antonym of complicated?

### Response:
The antonym of "complicated" is "simple"
```

> **Critical point:** The entire string (prompt + response) is fed to the LLM during training. The model learns to generate the response portion given the prompt portion. This is exactly how classification fine-tuning worked, just with text responses instead of class labels.

---

## 7. Alternative Format: Phi-3 Style (Microsoft)

For awareness — not used in this project but you should know it exists:

```
<|user|>
Identify the correct spelling of: Occasion

<|assistant|>
The correct spelling is "occasion"
```

**Key difference from Alpaca:**

| | Alpaca | Phi-3 |
|--|--------|-------|
| Instruction & input | Separated into `### Instruction` and `### Input` | Merged into one `<|user|>` block |
| Context header | Verbose ("Below is an instruction…") | None — straight to content |
| Common use | Research, open models | Microsoft Phi-3 models |

Both are valid. Alpaca is used here because it's more common in the open-source community.

---

## 8. Train / Validation / Test Split

The 1,100 entries are split into three sets:

```python
train_portion = int(len(data) * 0.85)   # 935 entries
test_portion  = int(len(data) * 0.10)   # 110 entries
val_portion   = len(data) - train_portion - test_portion  # 55 entries

train_data = data[:train_portion]
test_data  = data[train_portion:train_portion + test_portion]
val_data   = data[train_portion + test_portion:]
```

| Split | Size | Purpose |
|-------|------|---------|
| Train (85%) | 935 | Model learns from these |
| Test (10%) | 110 | Final evaluation after training is done |
| Validation (5%) | 55 | Monitor loss *during* training to detect overfitting |

This is the same split logic as the spam classification project — nothing new here architecturally.

---

## 9. What Comes Next (Preview)

The next lecture covers **batching** — which is more complex than classification batching because:

1. Each formatted prompt has a **different length** (some instructions are short, some long).
2. The LLM requires all sequences in a batch to be the **same length**.
3. Solution: **padding** — add filler tokens to shorter sequences to match the longest one.
4. During training, padded positions must be **ignored** in the loss calculation (you don't want the model to learn to predict padding tokens).

This is why the dataset preparation for instruction fine-tuning needs more steps than classification.

---

## Summary

| Concept | What to remember |
|---------|-----------------|
| Why instruction fine-tuning | Pre-trained models complete text but can't follow commands |
| Supervised fine-tuning | Training on (instruction → response) pairs with known correct answers |
| Dataset structure | Three fields: `instruction`, `input` (optional), `output` |
| Alpaca format | Wraps raw data into a structured prompt template before training |
| `input` can be empty | When absent, skip the `### Input:` section in the prompt |
| Full string to model | Prompt + response fed together; model learns to generate the response part |
| 85 / 10 / 5 split | Train / Test / Validation — same pattern as classification project |
| Next step | Padding variable-length sequences to create uniform batches |
