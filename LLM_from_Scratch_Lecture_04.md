# 📘 Build a Large Language Model from Scratch
## Lecture 4 — Introduction to Transformers
> **Source:** Vizuara YouTube Channel | **Instructor:** Dr. Raj Dandekar
> **Video:** https://youtu.be/3dWzNZXA8DY (Lecture 4)

---

## 🗺️ What This Lecture Covers

1. What is the Transformer architecture?
2. The landmark paper — "Attention Is All You Need" (2017)
3. Simplified Transformer — 8 steps walkthrough
4. Tokenization & Vector Embeddings explained
5. Encoder vs. Decoder
6. Self-Attention Mechanism (intuition)
7. BERT vs. GPT — differences
8. Transformers vs. LLMs — are they the same?

---

## 1. What is the Transformer Architecture?

> **Transformer** = A **deep neural network architecture** that is the secret sauce behind most modern LLMs.

- Introduced in **2017** via the paper **"Attention Is All You Need"**
- Has **100,000+ citations** on Google Scholar — one of the most impactful AI papers ever
- The **GPT architecture** (which powers ChatGPT) is **heavily based** on the Transformer
- Originally designed for **machine translation** (English → German/French)
- Later discovered to be useful for **many other tasks** beyond translation

---

## 2. The Landmark Paper — "Attention Is All You Need" (2017)

| Detail | Info |
|--------|------|
| **Paper title** | Attention Is All You Need |
| **Year** | 2017 |
| **Authors** | Google Research team |
| **Original purpose** | Machine Translation (English → German & French) |
| **Key innovation** | Self-Attention Mechanism |
| **Citations** | 100,000+ (in ~7 years) |
| **Pages** | 15 pages |

> 💡 The word **"Attention"** in the title is a **technical term** — it refers to how the model decides which words to focus on when making predictions. We will go deep into this in later lectures.

---

## 3. Simplified Transformer — 8 Steps Walkthrough

The Transformer takes an English sentence and translates it to German in **8 steps**.

### Full Flow Diagram

```
INPUT TEXT (English)
        ↓
STEP 1: Input text to be translated
        ↓
STEP 2: Pre-processing → Tokenization
        ↓
STEP 3: Encoder → Token IDs passed in
        ↓
STEP 4: Vector Embeddings generated
        ↓ (embeddings sent to decoder)
STEP 5: Partial output text (German so far)
        ↓
STEP 6: Pre-processing → Tokenization of partial output
        ↓
STEP 7: Decoder → predicts next German word
        ↓
STEP 8: Final translated output
```

### Step-by-Step Explanation

#### ✅ Step 1 — Input Text
- The English sentence to be translated
- Example: *"This is an example"*

#### ✅ Step 2 — Pre-processing (Tokenization)
- The sentence is broken into **tokens** (approximately = individual words)
- Each token is assigned a **unique numerical ID**
- Example:
  ```
  "Fine tuning is Fun For All"
  → ["Fine", "tuning", "is", "Fun", "For", "All"]
  → [101, 204, 39, 567, 88, 92]
  ```
- This process = **Tokenization**
- For simplicity: think of **1 token ≈ 1 word** (not exactly true, but close enough for now)

#### ✅ Steps 3 & 4 — Encoder + Vector Embeddings
- Token IDs are passed into the **Encoder**
- The Encoder converts tokens into **Vector Embeddings**
- (Full explanation in next section below)

#### ✅ Step 5 — Partial Output Text
- The model translates **one word at a time**
- By the time it's predicting word 4, it already has words 1–3 translated
- Example: Input = *"This is an example"* → Model already has *"Das ist ein"* and now needs to predict the German word for *"example"*
- This partial text is **available to the decoder**

#### ✅ Steps 6 & 7 — Decoder
- The decoder receives:
  1. **Vector embeddings** from the encoder (left side)
  2. **Partial output text** (already translated words)
- It uses both to **predict the next word**
- Trained like a neural network — makes mistakes early, improves via loss function

#### ✅ Step 8 — Final Output
- Complete translated sentence is produced **one word at a time**
- Example output: *"Das ist ein Beispiel"* (German for "This is an example")

> 💡 **Key insight:** At the simplest level, a Transformer is just a neural network where you optimize parameters. Don't be intimidated by it!

---

## 4. Tokenization & Vector Embeddings

### Tokenization
> **Tokenization** = Breaking a sentence into individual words/subwords and assigning a **unique ID** to each.

- Input: Huge text data from internet, books, etc.
- Process: Split sentences → assign numerical IDs
- Output: A list of token IDs the model can process

### Vector Embeddings
> **Vector Embedding** = Converting words into **numerical vectors** in a high-dimensional space such that **semantically similar words are placed closer together**.

#### The Problem Tokenization Alone Doesn't Solve
- Tokenization assigns **random IDs** to words
- "Dog" might get ID 450 and "Puppy" might get ID 7832
- But the model needs to **know** that dog and puppy are related!
- Vector embeddings solve this

#### How Embeddings Work

```
Words:    King   Man   Woman   Apple  Banana  Orange  Football  Golf  Tennis
                 ↓
After embedding:
- King, Man, Woman → clustered CLOSE together (all human-related)
- Apple, Banana, Orange → clustered CLOSE together (all fruits)
- Football, Golf, Tennis → clustered CLOSE together (all sports)
```

- Words are placed in **high-dimensional space** (500,000+ dimensions — impossible to visualize)
- Similar words end up **near each other** in that space
- Neural networks are trained specifically to learn these embeddings

#### Visual Summary
```
Text Documents
     ↓
Token IDs  [101, 204, 39...]
     ↓
Embedding Model
     ↓
Vector Embeddings  [[0.2, 0.8, 0.1...], [0.9, 0.1, 0.3...], ...]
```

> 💡 The **main purpose of the Encoder** = Convert input tokens → Vector Embeddings that capture semantic meaning

---

## 5. Encoder vs. Decoder

| | **Encoder** | **Decoder** |
|--|------------|------------|
| **Location** | Left side of Transformer | Right side of Transformer |
| **Input** | Tokenized input text | Vector embeddings + partial output text |
| **Output** | Vector Embeddings | Next predicted word |
| **Main job** | Understand the input | Generate the output |
| **Used in** | BERT | GPT |
| **Transformer** | Has both | Has both |

> ⚠️ **Important:** GPT models **do NOT have an encoder** — they only use the **decoder**. BERT only uses the **encoder**.

---

## 6. Self-Attention Mechanism

### What Is It?
> **Self-Attention** = The model's ability to **weigh the importance of different words** relative to each other when predicting the next word.

### Why Is It Needed? — The Long-Range Dependency Problem

Consider this story:
```
Sentence 1: Harry Potter is on Platform 9¾.
Sentence 2: Harry Potter wants to board the train.
Sentence 3: He picks up his luggage.
Sentence 4: Harry Potter waves goodbye and ___
```
To predict the blank in Sentence 4, the model needs to **remember context from Sentences 1, 2, 3** — not just the sentence right before.

This is called a **long-range dependency** — words far apart in text still affect each other's meaning.

### How Self-Attention Solves This
- Maintains an **attention score** for every word relative to every other word
- The score tells the model: *"When predicting the next word, how much attention should I give to each past word?"*
- Example: "Harry" and "train" might get higher attention scores than "the" or "and"

### Why "Attention Is All You Need"?
- The self-attention mechanism is so powerful that the entire Transformer is built around it
- It allows the model to look **across the entire context** (all previous sentences) at once
- This is what makes LLMs so much better than old NLP models that could only look at nearby words

### Formal Definition (from the paper)
> *"Attention mechanisms allow modeling of dependencies without regard to their distance in the input or output sequences."*

In plain English: **It doesn't matter how far apart two words are — attention can connect them.**

---

## 7. BERT vs. GPT

### Full Forms
| Model | Full Form |
|-------|----------|
| **BERT** | **B**idirectional **E**ncoder **R**epresentations from **T**ransformers |
| **GPT** | **G**enerative **P**re-trained **T**ransformers |

### How Each Works

#### BERT — Predicts Hidden/Masked Words
```
Input:  "This is an ___ of how LLM ___ perform"
            (some words are randomly masked)
Output: Fills in → "This is an EXAMPLE of how LLM CAN perform"
```
- Looks at the sentence from **BOTH directions** (left → right AND right → left)
- That's why it's called **Bi**directional
- Has to find masked words that could be **anywhere** in the sentence
- Only uses the **Encoder** (no decoder)

#### GPT — Predicts the Next Word
```
Input:  "This is an example of how LLM can ___"
            (always predicts what comes NEXT)
Output: Predicts → "perform"
```
- Only looks from **left to right**
- Always predicts the **rightmost unknown word**
- Only uses the **Decoder** (no encoder)

### Comparison Table

| Feature | **BERT** | **GPT** |
|---------|---------|---------|
| Direction | Bidirectional (both ways) | Left to right only |
| Task | Predict masked/hidden words | Predict next word |
| Architecture | Encoder only | Decoder only |
| Strength | Sentiment analysis, understanding | Text generation, conversation |
| Why bidirectional helps | Any word can be masked → must look both ways | Only the last word is unknown |
| Example use | Detecting emotion in text | ChatGPT conversations |

> 💡 BERT can differentiate between **"bank"** (financial institution) vs **"bank"** (river bank) by looking at surrounding words from both sides. GPT can't do this as well because it only reads left to right.

> 💡 Both BERT and GPT have the word **"Transformers"** in them — because both are derived from the original Transformer architecture.

---

## 8. Transformers vs. LLMs — Are They the Same?

### ❌ Common Mistake: Using Them Interchangeably

> **Not all Transformers are LLMs.**
> **Not all LLMs are Transformers.**

### Transformers Are Used Beyond Language

Transformers are also used in **Computer Vision**:
- **Vision Transformers (ViT)** — used for image classification, object detection
- Can detect potholes on roads from images
- Can classify tumors as malignant or benign from medical images
- Often outperform CNNs (Convolutional Neural Networks) with fewer resources

### LLMs Can Use Non-Transformer Architectures

Before Transformers (2017), LLMs were built on:

| Architecture | Year Introduced | How They Work |
|-------------|----------------|--------------|
| **RNN** (Recurrent Neural Network) | 1980s | Feedback loop to incorporate memory |
| **LSTM** (Long Short-Term Memory) | 1997 | Two paths: one for long-term memory, one for short-term memory |
| **CNN** (Convolutional) | Various | Can also be used for text in some architectures |

- RNNs and LSTMs **can do text completion** → they qualify as language models
- So LLMs existed **before** Transformers!

### Summary

```
Transformers ≠ LLMs

Transformers can be used for:           LLMs can be based on:
- Text (language tasks)                 - Transformers (modern)
- Images (Vision Transformers)          - RNNs (older)
- Audio                                 - LSTMs (older)
- Other domains                         - CNNs (some cases)
```

---

## 🔑 Key Terms to Know

| Term | Simple Meaning |
|------|---------------|
| **Transformer** | Deep neural network architecture introduced in 2017 |
| **"Attention Is All You Need"** | The 2017 paper that introduced the Transformer |
| **Tokenization** | Breaking sentences into words/subwords and assigning IDs |
| **Token** | Approximately one word (a unit of text) |
| **Token ID** | Unique number assigned to each token |
| **Vector Embedding** | Converting words into numerical vectors that capture meaning |
| **Semantic meaning** | The actual meaning/relationship between words |
| **Encoder** | Converts input tokens → vector embeddings |
| **Decoder** | Takes embeddings + partial output → predicts next word |
| **Self-Attention** | Mechanism to weigh importance of all words relative to each other |
| **Attention Score** | A number that tells how much focus to give each past word |
| **Long-range dependency** | When words far apart in text still affect each other's meaning |
| **BERT** | Bidirectional Encoder Representations from Transformers — predicts masked words |
| **GPT** | Generative Pre-trained Transformers — predicts next word |
| **Vision Transformer (ViT)** | Transformer used for image tasks, not language |
| **RNN** | Recurrent Neural Network — older LLM architecture with memory loop |
| **LSTM** | Long Short-Term Memory — older architecture with short + long term memory paths |
| **Multi-head Attention** | Multiple attention mechanisms running in parallel inside Transformer |

---

## ⚡ Quick Summary

1. **Transformer** = deep neural network architecture from the 2017 paper "Attention Is All You Need"
2. Originally built for **English → German/French translation**
3. Transformer has **8 steps**: Input → Tokenize → Encode → Embed → Partial output → Decode → Predict → Final output
4. **Tokenization** = break sentences into tokens + assign IDs
5. **Vector Embedding** = convert tokens into vectors so similar words cluster together
6. **Encoder** = input tokens → embeddings | **Decoder** = embeddings + partial text → next word
7. **Self-Attention** = lets the model weigh importance of ALL past words when predicting next word
8. **BERT** = bidirectional, encoder-only, predicts masked words, great for sentiment analysis
9. **GPT** = left-to-right, decoder-only, predicts next word, great for text generation
10. **Not all Transformers are LLMs** — Vision Transformers exist for images
11. **Not all LLMs are Transformers** — RNNs and LSTMs can also be LLMs

---

## 🧠 Key Takeaways to Remember

- 🔹 **Transformer = the secret sauce** behind modern LLMs
- 🔹 **"Attention Is All You Need"** (2017) = most impactful AI paper ever — 100K+ citations
- 🔹 **Tokenization** = sentence → tokens → IDs
- 🔹 **Vector Embedding** = words → vectors where similar words are close together
- 🔹 **Encoder** converts input to embeddings | **Decoder** generates output word by word
- 🔹 **Self-Attention** lets model look at ALL past context, not just nearby words
- 🔹 **BERT** = bidirectional + encoder only | **GPT** = left-to-right + decoder only
- 🔹 **Transformers ≠ LLMs** — don't use them interchangeably!
- 🔹 Before Transformers: **RNNs and LSTMs** were used for language tasks

---

## 📌 What's Coming Next

- Deep dive into **Attention Mechanism** — the mathematics and code
- Understanding **keys, queries, and values**
- Starting to **code** the Transformer components from scratch

---

*Notes prepared from Vizuara YouTube Series: "Build a Large Language Model from Scratch"*
*GitHub-ready markdown — push directly to your notes repo* 🚀
