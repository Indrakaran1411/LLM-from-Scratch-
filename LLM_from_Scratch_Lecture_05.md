# 📘 Build a Large Language Model from Scratch
## Lecture 5 — GPT: History, Zero-Shot vs Few-Shot, Auto-regression & Emergent Behavior
> **Source:** Vizuara YouTube Channel | **Instructor:** Dr. Raj Dandekar
> **Video:** Lecture 5 in the series

---

## 🗺️ What This Lecture Covers

1. History — From Transformers → GPT → GPT-2 → GPT-3 → GPT-4
2. Zero-Shot vs One-Shot vs Few-Shot Learning
3. GPT-3 Dataset — How much data was used
4. Open Source vs Closed Source LLMs
5. GPT Architecture — Decoder only, Auto-regression
6. Unsupervised Learning in GPT
7. Emergent Behavior

---

## 1. History — The GPT Evolution Timeline

### Full Progression

```
2017 → Transformer ("Attention Is All You Need")
          ↓
2018 → GPT-1 (Generative Pre-training)
          ↓
2019 → GPT-2 (Language Models are Unsupervised Multitask Learners)
          ↓
2020 → GPT-3 (Language Models are Few-Shot Learners)
          ↓
2022 → GPT-3.5 (ChatGPT goes viral commercially)
          ↓
2023+ → GPT-4 (Where we are today)
```

---

### Paper 1 — Transformers (2017)
- **Title:** "Attention Is All You Need"
- **Key Innovation:** Self-Attention Mechanism
- **What it did:** Captured long-range dependencies in sentences — much better than RNNs and LSTMs
- **Architecture:** Had both **Encoder + Decoder**
- **Original purpose:** Machine Translation (English → German/French)

---

### Paper 2 — GPT-1 (2018)
- **Title:** "Improving Language Understanding with Unsupervised Learning"
- **Key Innovation:** Generative Pre-training on **unlabeled text**
- **What changed from Transformer:**
  - Removed the **Encoder block** — only Decoder remains
  - Trained on huge diverse dataset to predict the **next word**
  - Used **unsupervised learning** — no need for labeled data
- **OpenAI's claim:** "Large gains on language tasks can be realized by generative pre-training on a diverse corpus of unlabeled text"
- **Two key ideas combined:** Transformer architecture + Unsupervised pre-training

---

### Paper 3 — GPT-2 (2019)
- **Title:** "Language Models are Unsupervised Multitask Learners"
- **What changed:** Took MORE data than GPT-1 + bigger model
- **Four model sizes released:**

| Model | Parameters |
|-------|-----------|
| GPT-2 Small | 117 Million |
| GPT-2 Medium | 345 Million |
| GPT-2 Large | 762 Million |
| GPT-2 Extra Large | ~1 Billion |

- **First time** a language model crossed **1 Billion parameters**
- Citations: 10,000+ on Google Scholar

---

### Paper 4 — GPT-3 (2020)
- **Title:** "Language Models are Few-Shot Learners"
- **Parameters:** **175 Billion** — 10x larger than any previous model
- **Key claim:** GPT-3 is a **few-shot learner** — give it a few examples and it can do amazing things
- **Pre-training cost:** **$4.6 Million**
- **Dataset:** 300 Billion tokens total
- A massive leap — people were shocked at how capable it was

---

### GPT-3.5 and GPT-4
- **GPT-3.5 (2022):** Became **commercially viral** — the ChatGPT everyone started using
- **GPT-4 (2023+):** Current state of the art — both zero-shot and few-shot capable
- In just **7 years** (2017–2024), the field went from the original Transformer paper to GPT-4

---

## 2. Zero-Shot vs One-Shot vs Few-Shot Learning

### Definitions

| Type | Meaning | Examples Provided |
|------|---------|------------------|
| **Zero-Shot** | Model performs task with **no examples** given — just the instruction | 0 |
| **One-Shot** | Model sees **one example** before performing the task | 1 |
| **Few-Shot** | Model sees **a few examples** before performing the task | 2–5+ |

---

### Visual Examples (from GPT-3 Paper)

#### Zero-Shot Learning
```
Prompt: "Translate English to French: cheese → "
Model Output: "fromage"
(No examples given — just the instruction)
```

#### One-Shot Learning
```
Prompt:
"sea otter → loutre de mer  (one example given)
Translate: cheese → "
Model Output: "fromage"
```

#### Few-Shot Learning
```
Prompt:
"sea otter → loutre de mer
peppermint → menthe poivrée
giraffe → girafe  (multiple examples given)
Translate: cheese → "
Model Output: "fromage"
```

---

### GPT-3's Claim
- The paper **"Language Models are Few-Shot Learners"** claims GPT-3 excels at **few-shot learning**
- It CAN do zero-shot learning but is more accurate with examples

### GPT-4's Own Answer (asked directly)
- "I am a **few-shot learner** — I can understand and perform tasks better with a few examples"
- "I also have **zero-shot capabilities** — I can perform tasks without prior examples"
- **Bottom line:** GPT-4 is both, but performs **better with examples** (few-shot)

> 💡 **Practical tip:** When using ChatGPT, give it examples of the output format you want — it will do a much better job!

---

## 3. GPT-3 Dataset

| Data Source | Tokens Used | % of Total |
|-------------|-----------|-----------|
| **Common Crawl** | 410 Billion | ~60% |
| **WebText2** (Reddit posts 2015–2020) | 19 Billion | ~22% |
| **Books** | ~67 Billion | ~10% |
| **Wikipedia** | ~3 Billion | ~3% |
| **Total** | **~300 Billion tokens** | 100% |

> 💡 **Token** = A unit of text the model reads. For now, think: **1 token ≈ 1 word**

> 💡 **Common Crawl** = Free, open repository of web data — 250 billion+ pages, spanning 17 years since 2007

### Why Does Pre-training Cost $4.6 Million?
- 300 Billion sentences need to be broken into training pairs
- Each sentence → split into input (context) and output (next word)
- This has to be done for ALL data
- **175 Billion parameters** need to be optimized via backpropagation
- Requires massive GPU clusters running for weeks/months

---

## 4. Open Source vs Closed Source LLMs

| | **Closed Source** | **Open Source** |
|--|------------------|----------------|
| **Example** | GPT-4 (OpenAI) | Llama 3.1 (Meta) |
| **Access** | Use via API/chat only | Full model weights available |
| **Customization** | Limited | Can fine-tune yourself |
| **Performance (2022)** | Much better | Significantly behind |
| **Performance (2024)** | State of art | Almost equal / sometimes better |

### Key Development: Llama 3.1 (Meta, 2024)
- **Parameters:** 405 Billion
- **Type:** Open Source
- **Performance:** Comparable to or slightly better than GPT-4 in some benchmarks
- **Significance:** The gap between open source and closed source is **closing fast**

> 💡 For students: Using GPT-4 directly is perfectly fine — you don't need to worry about model weights or fine-tuning for general use.

---

## 5. GPT Architecture — Decoder Only

### Key Difference from Original Transformer

| | Original Transformer | GPT |
|--|---------------------|-----|
| Encoder | ✅ Yes | ❌ No |
| Decoder | ✅ Yes | ✅ Yes |
| Layers | 6 encoder + 6 decoder | **96 Transformer layers** |
| Parameters | Millions | **175 Billion** |

> GPT is actually a **simpler** architecture than the original Transformer — no encoder. But it's **scaled massively** in depth and parameters.

### GPT Architecture Flow (Each Iteration)

```
Iteration 1:
Input:  "The lion roars in the"
Output: "jungle"  ← predicted next word

Iteration 2:
Input:  "The lion roars in the jungle"  ← previous output added
Output: "every"

Iteration 3:
Input:  "The lion roars in the jungle every"
Output: "night"

...and so on
```

- One word generated per iteration
- Previous output becomes part of next input
- This is what makes it **Auto-Regressive**

---

## 6. Unsupervised Learning & Auto-Regression in GPT

### Why GPT Pre-training is Called Unsupervised

> In normal supervised learning → you need **human-labeled** data (expensive & scarce)
> In GPT pre-training → the **sentence itself** provides the label!

```
Sentence: "The lion roars in the jungle"

Training input:  "The lion roars in the"
Label (target):  "jungle"   ← taken directly from the sentence!

No human labeling needed → UNSUPERVISED
```

> "We don't collect labels for the training data — we use the **structure of the data itself** to make the labels."

### Why GPT is Called Auto-Regressive

> **Auto-Regressive** = The **previous output is used as input** for the next prediction

```
Step 1: Input = "Second law of Robotics:"  → Output = "A"
Step 2: Input = "Second law of Robotics: A"  → Output = "robot"
Step 3: Input = "Second law of Robotics: A robot"  → Output = "must"
Step 4: Input = "Second law of Robotics: A robot must"  → Output = "obey"
```

Each output feeds back into the next input — hence **auto-regressive**.

### Two Key Properties to Remember

| Property | Meaning |
|----------|---------|
| **Unsupervised** | No external labels needed — the next word in the sentence IS the label |
| **Auto-Regressive** | Previous output becomes input for the next prediction |

---

## 7. Emergent Behavior

### What Is It?

> **Emergent Behavior** = The ability of a model to perform tasks it was **never explicitly trained to perform**.

### The Core Surprise
- GPT is trained **only** to predict the next word
- Yet it can do:
  - ✅ Language Translation
  - ✅ Multiple Choice Question Generation
  - ✅ Text Summarization
  - ✅ Sentiment Analysis
  - ✅ Essay Grading
  - ✅ Lesson Plan Creation
  - ✅ Spelling Correction
  - ✅ Three-digit Arithmetic
  - ✅ Report Card Generation
  - ✅ Proofreading

### OpenAI's Own Statement
> *"We noticed that we can use the underlying language model to begin to perform tasks without ever training on them. Performance on tasks like picking the right answer to a multiple choice question steadily increases as the underlying language model improves."*

### Why Does Emergent Behavior Happen?
- Still an **open research question** — no complete explanation yet
- Active area of research with many papers being published
- Hypothesis: Training on massive diverse data forces the model to learn deep patterns about language, reasoning, and the world

> 💡 If you're looking for a **research topic** → Emergent Behavior in LLMs is a great area to explore!

---

## 🔑 Key Terms to Know

| Term | Simple Meaning |
|------|---------------|
| **GPT** | Generative Pre-trained Transformer |
| **Zero-Shot Learning** | Model does a task with zero examples given |
| **One-Shot Learning** | Model gets one example before doing the task |
| **Few-Shot Learning** | Model gets a few examples before doing the task |
| **Auto-Regressive** | Previous output is used as input for next prediction |
| **Unsupervised Learning** | No human labels needed — data labels itself |
| **Emergent Behavior** | Model performs tasks it was never trained for |
| **Common Crawl** | Open repository of all web data — used to train GPT-3 |
| **Token** | Unit of text the model reads (~1 word) |
| **Closed Source LLM** | Model weights are private (e.g. GPT-4) |
| **Open Source LLM** | Model weights are public (e.g. Llama 3.1) |
| **Llama 3.1** | Meta's 405B parameter open-source LLM — rivals GPT-4 |
| **Self-Supervised Learning** | Another name for GPT's training — labels come from the data itself |

---

## ⚡ Quick Summary

1. **2017 → Transformer** (Attention Is All You Need) — encoder + decoder, self-attention
2. **2018 → GPT-1** — removed encoder, generative pre-training on unlabeled text
3. **2019 → GPT-2** — bigger data, up to 1B parameters (4 model sizes)
4. **2020 → GPT-3** — 175B parameters, few-shot learner, $4.6M training cost
5. **2022 → GPT-3.5** — goes commercially viral (ChatGPT)
6. **2023+ → GPT-4** — current state of art, both zero-shot + few-shot
7. **Zero-shot** = no examples | **One-shot** = 1 example | **Few-shot** = 2+ examples
8. GPT performs better with examples → it's primarily a **few-shot learner**
9. GPT-3 trained on **300 Billion tokens** (Common Crawl + WebText2 + Books + Wikipedia)
10. GPT is **decoder-only** with 96 Transformer layers in GPT-3
11. Pre-training is **unsupervised** — sentence itself provides the label
12. Pre-training is **auto-regressive** — previous output becomes next input
13. **Emergent behavior** = GPT doing tasks it was never trained for (translation, MCQs, etc.)
14. **Llama 3.1** (405B, open-source) is closing the gap with GPT-4

---

## 🧠 Key Takeaways to Remember

- 🔹 GPT evolved over 7 years: **Transformer → GPT-1 → GPT-2 → GPT-3 → GPT-3.5 → GPT-4**
- 🔹 GPT = **decoder only** — no encoder (unlike original Transformer)
- 🔹 **Zero-shot** = no examples | **Few-shot** = give examples → better results
- 🔹 GPT trained only for **next word prediction** but can do everything = **Emergent Behavior**
- 🔹 GPT pre-training is **unsupervised** (labels from data itself) + **auto-regressive** (output → next input)
- 🔹 GPT-3 = **175B parameters**, **300B tokens**, **$4.6M** training cost
- 🔹 **Llama 3.1** = 405B open-source model by Meta — rivals GPT-4
- 🔹 Emergent behavior is still an **open research question** — great area to explore!

---

## 📌 What's Coming Next

- Stages of building an LLM (deeper dive)
- **Start coding!** — Data preprocessing from scratch
- Tokenization in code

---

*Notes prepared from Vizuara YouTube Series: "Build a Large Language Model from Scratch"*
*GitHub-ready markdown — push directly to your notes repo* 🚀
