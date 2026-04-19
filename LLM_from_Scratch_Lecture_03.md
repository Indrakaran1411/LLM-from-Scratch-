# 📘 Build a Large Language Model from Scratch
## Lecture 3 — Stages of Building Large Language Models
> **Source:** Vizuara YouTube Channel | **Instructor:** Dr. Raj Dandekar
> **Video:** https://youtu.be/Xpr8D6LeAtw (Lecture 3)

---

## 🗺️ What This Lecture Covers

- The **two stages** of building an LLM
  1. Pre-training
  2. Fine-tuning
- What data is used in each stage
- Real-world examples of fine-tuning
- Labeled vs. Unlabeled data
- Types of fine-tuning
- The full schematic: Data → Pre-training → Fine-tuning

---

## 🔁 The Two Stages of Building an LLM

```
Stage 1: PRE-TRAINING
        ↓
Stage 2: FINE-TUNING
```

Every LLM you use today — ChatGPT, Llama, Harvey — went through both these stages.

---

## Stage 1 — Pre-Training

### Simple Definition
> **Pre-training** = Training an LLM on a **huge and diverse dataset** so it can perform a wide range of tasks.

### What Problem Does It Solve?
When you chat with ChatGPT and it answers everything perfectly — that's because of pre-training. It has read an enormous amount of text and learned from it.

### Analogy
> 💡 Think of it like raising a child — parents teach them about the world, they absorb everything, and later they can apply that knowledge to many different situations.

---

### GPT-3 Pre-Training Data (Real Numbers)

GPT-3 was trained on **300 Billion tokens** (~300 Billion words).

| Data Source | Words Used | Description |
|-------------|-----------|-------------|
| **Common Crawl** | 410 Billion | Massive open repository of all internet data |
| **WebText2** | 20 Billion | Reddit posts, blog posts, Stack Overflow, code |
| **Books** | 67 Billion | Large number of books (Books1 + Books2) |
| **Wikipedia** | 3 Billion | Wikipedia articles |

> 💡 **Common Crawl** = A huge, publicly available archive of the entire internet.

> 💡 If 1 sentence ≈ 10 words, then 410 Billion words = **41 Billion sentences** just from Common Crawl alone!

---

### What Task Was the LLM Trained On?
LLMs were originally trained on a single, simple task:

> **Word Completion** — Given a sentence with a missing word, predict the next word.

**Example:**
```
The lion is in the ___
→ LLM predicts: "forest"
```

### The Surprising Discovery 🤯
Even though LLMs were only trained for **next word prediction**, they turned out to be capable of **many other tasks** without being trained on them:

- ✅ Translation
- ✅ Answering multiple choice questions
- ✅ Text summarization
- ✅ Sentiment analysis
- ✅ Question answering
- ✅ Linguistic analysis

> OpenAI wrote in their 2018 blog post: *"We noticed we can use the underlying language model to begin to perform tasks without even training on them."*

This is why LLMs are so much more powerful than old NLP models — **one pre-trained model can do everything.**

---

### Cost of Pre-Training (Real Numbers)
Pre-training is extremely expensive:

| Model | Pre-Training Cost |
|-------|-----------------|
| **GPT-3** | **$4.6 Million** |

> ⚠️ Normal students or individuals **cannot afford** to pre-train an LLM. You need powerful GPUs and millions of dollars.

---

### Key Term: Foundational Model
- A **pre-trained LLM** is also called a **Foundational Model** or **Base Model**
- Examples: GPT-4, Llama 3.1
- These are incredibly capable even without any fine-tuning

---

## Stage 2 — Fine-Tuning

### Simple Definition
> **Fine-tuning** = Taking a pre-trained (foundational) model and **training it further on a smaller, specific dataset** to make it an expert at a particular task or domain.

### Why Is Fine-Tuning Needed?

Pre-trained models are **generic** — they know a little about everything. But sometimes you need a model that knows **a lot about one specific thing**.

**Example:**
- You are the CEO of an airline company
- You want a chatbot that answers: *"What is the price for the Lanza flight leaving at 6:00 PM?"*
- GPT-4 (pre-trained only) was **not trained on your company's data**
- Its answer will be **generic and inaccurate**
- Solution → **Fine-tune** it on your airline's specific data

---

### Real-World Fine-Tuning Examples

#### 1. 📱 SK Telecom (Telecom Sector)
- **Goal:** Build a chatbot for Telecom customer service in **Korean**
- **Problem:** GPT-4 was not trained on Korean Telecom conversations
- **Solution:** Fine-tuned GPT-4 on their own Korean Telecom data
- **Result:**
  - 35% increase in conversation summarization quality
  - 33% increase in intent recognition accuracy

#### 2. ⚖️ Harvey (Legal Sector)
- **Goal:** AI tool for **lawyers and attorneys**
- **Problem:** Pre-trained models lacked knowledge of **legal case history**
- **Solution:** Fine-tuned on legal case studies and legal documents
- **Result:** A trusted legal AI platform used by top law firms worldwide
- **Website:** harvey.ai

#### 3. 🏦 JP Morgan Chase (Banking Sector)
- **Goal:** LLM for their own employees and research analysts
- **Problem:** JP Morgan's internal financial data is **not publicly available**
- **Solution:** Fine-tuned their own LLM on private proprietary data
- **Result:** Answers specific to JP Morgan's internal operations

> 💡 **Key Rule:** Big companies and startups going into production **always fine-tune** — they never rely only on the foundational model.

---

### Who Needs Fine-Tuning?

| User Type | Need Fine-Tuning? | Reason |
|-----------|------------------|--------|
| Student / General user | ❌ Usually No | GPT-4 is good enough for general tasks |
| Startup / Company | ✅ Yes | Need domain-specific, accurate answers |
| Large Enterprise | ✅ Absolutely | Have private data, need specific performance |

---

## 📊 The Full Schematic: 3 Steps

```
STEP 1: DATA
    ↓
  Internet + Books + Media + Research Articles
  (Billions or Trillions of words)

STEP 2: PRE-TRAINING → Foundational Model
    ↓
  Train LLM on huge, unlabeled text data
  (Expensive: $4.6M for GPT-3)
  Result: A model capable of many tasks

STEP 3: FINE-TUNING → Specialized Model
    ↓
  Train further on smaller, labeled dataset
  Result: Expert model for your specific use case
```

### After Fine-Tuning, You Can Build:
- 🤖 Personal Assistant
- 🌐 Language Translation Bot
- 📄 Summarization Assistant
- 📧 Email Classification Bot
- 💬 Customer Support Chatbot

---

## 🏷️ Labeled vs. Unlabeled Data

| | Pre-Training | Fine-Tuning |
|--|-------------|-------------|
| **Data Type** | Unlabeled (Raw Text) | Labeled |
| **What it means** | Text with no predefined tags or categories | Text paired with correct answers/categories |
| **Example** | Random internet articles | Emails tagged as "Spam" or "Not Spam" |
| **Learning type** | Unsupervised / Auto-regression | Supervised |

### What is Raw Text?
> **Raw text** = Regular text **without any labeling information** — just plain sentences from the internet, books, etc.

### What is Auto-regression?
- The model predicts the **next word** in a sentence
- The predicted word then becomes part of the training input for the next prediction
- This is how pre-training works — self-supervised on its own predictions

---

## 🔀 Two Types of Fine-Tuning

### Type 1 — Instruction Fine-Tuning
- Used when you want the model to **follow specific instructions**
- Label data = **Instruction + Answer pairs**
- Examples:
  - English text → French translation (input-output pairs)
  - Airline query → Correct customer support response
  - Any instruction → Expected output

### Type 2 — Classification Fine-Tuning
- Used when you want the model to **classify** inputs into categories
- Label data = **Text + Label**
- Example: 10,000 emails → 5,000 labeled "Spam", 5,000 labeled "Not Spam"
- The LLM learns to classify new emails correctly

---

## 🔑 Key Terms to Know

| Term | Simple Meaning |
|------|---------------|
| **Pre-training** | Training LLM on huge, diverse, unlabeled data |
| **Fine-tuning** | Further training on smaller, labeled, specific data |
| **Foundational Model** | Another name for a pre-trained LLM (e.g., GPT-4) |
| **Base Model** | Same as Foundational Model |
| **Raw Text** | Plain text with no labels — used in pre-training |
| **Labeled Data** | Text paired with correct answers/categories — used in fine-tuning |
| **Common Crawl** | Open repository of all internet data |
| **Word Completion** | The original pre-training task — predict the next word |
| **Auto-regression** | Self-supervised learning by predicting the next word |
| **Instruction Fine-Tuning** | Fine-tune using instruction-answer pairs |
| **Classification Fine-Tuning** | Fine-tune to classify inputs into categories |
| **Token** | Approximately = 1 word (for simplicity) |

---

## ⚡ Quick Summary

1. Building an LLM has **2 stages**: Pre-training → Fine-tuning
2. **Pre-training** = Train on massive unlabeled data (e.g., 300B tokens for GPT-3)
3. GPT-3's training data = Common Crawl + WebText2 + Books + Wikipedia
4. Original training task = **word completion** (predict the next word)
5. Surprisingly, this also makes LLMs good at translation, summarization, QA, etc.
6. Pre-training cost = **$4.6 million** for GPT-3
7. Pre-trained model = **Foundational / Base model**
8. **Fine-tuning** = Train further on labeled, domain-specific data
9. Examples: SK Telecom (35% better), Harvey (legal AI), JP Morgan (banking AI)
10. Pre-training = **unlabeled data** | Fine-tuning = **labeled data**
11. Two types of fine-tuning: **Instruction fine-tuning** and **Classification fine-tuning**

---

## 🧠 Key Takeaways to Remember

- 🔹 **Pre-training** = huge data + no labels + teaches general knowledge
- 🔹 **Fine-tuning** = small data + labels + teaches specific skills
- 🔹 Pre-trained model is called **Foundational Model** or **Base Model**
- 🔹 GPT-3 trained on **300 Billion words** at a cost of **$4.6 Million**
- 🔹 LLMs were trained for **next word prediction** but can do **everything**
- 🔹 General users → use foundational model | Companies → must fine-tune
- 🔹 Fine-tuning types: **Instruction** (input-output pairs) vs **Classification** (text + label)
- 🔹 You **cannot** pre-train an LLM without powerful GPUs and millions of dollars

---

## 📌 What's Coming Next

- **Introduction to Transformer Architecture**
- A look at the landmark paper: **"Attention Is All You Need"** (2017)
- 2–3 more theory lectures → then **coding begins!**

---

*Notes prepared from Vizuara YouTube Series: "Build a Large Language Model from Scratch"*
*GitHub-ready markdown — push directly to your notes repo* 🚀
