# 📘 Build a Large Language Model from Scratch
## Lecture 8 — Byte Pair Encoding (BPE) Tokenization
> **Source:** Vizuara YouTube Channel | **Instructor:** Dr. Raj Dandekar
> **Video:** https://youtu.be/fKd8s29e-l4

---

## 🗺️ What This Lecture Covers

1. What is Tokenization & why it matters
2. Three Tokenization Approaches — Word, Character, Subword
3. BPE Algorithm — How it works step by step
4. Manual Walkthrough — BPE on a real dataset
5. Implementation using `tiktoken` (OpenAI's library)
6. Why BPE is the best choice for LLMs

---

## 1. What is Tokenization?

> **Tokenization** = The process of breaking raw text into smaller units called **tokens** that the LLM can process.

- Before any LLM can learn, text must be converted to numbers
- The tokenizer decides **how** to split the text
- The choice of tokenization strategy **hugely impacts** model performance
- **BPE (Byte Pair Encoding)** is the tokenization strategy used by:
  - ✅ GPT-2
  - ✅ GPT-3
  - ✅ Most modern LLMs

---

## 2. Three Tokenization Approaches

### Approach 1 — Word-Based Tokenization

> Split text into **individual words** — each word = one token

**Example:**
```
"The cat sat on the mat"
→ ["The", "cat", "sat", "on", "the", "mat"]
→ [101,   202,   303,   404,   101,   505]
```

#### ✅ Pros
- Simple and intuitive
- Preserves word-level meaning

#### ❌ Cons
- **Out-of-Vocabulary (OOV) errors** — if a word was never seen during training, the model crashes or produces `[UNK]`
- Cannot handle **morphological relationships**
  - e.g., *"boy"* and *"boys"* get completely different tokens — model doesn't know they're related
  - e.g., *"run"*, *"running"*, *"runner"* all treated as separate unrelated words
- Vocabulary becomes **extremely large** (millions of words in English)
- Rare words, names, technical terms → always OOV

---

### Approach 2 — Character-Based Tokenization

> Split text into **individual characters** — each character = one token

**Example:**
```
"cat"
→ ["c", "a", "t"]
→ [3,   1,   20]
```

#### ✅ Pros
- **No OOV errors** — every word can always be represented using individual characters
- Very **small vocabulary** (just ~26 letters + punctuation + numbers = ~100 tokens)

#### ❌ Cons
- **Very long sequences** — a 100-word sentence becomes 500+ tokens
- **Loses semantic meaning** — individual characters carry no meaning on their own
  - *"c"*, *"a"*, *"t"* individually mean nothing — but *"cat"* does
- Model has to work much harder to learn patterns from characters
- Computationally expensive due to long sequences

---

### Approach 3 — Subword-Based Tokenization ✅ (Best!)

> Split text into **subword units** — common words stay whole, rare/complex words are broken into meaningful pieces

**Example:**
```
"boys"    → ["boy", "s"]
"lowest"  → ["low", "est"]
"running" → ["run", "ning"]
"ChatGPT" → ["Chat", "G", "PT"]
```

#### ✅ Pros
- **No OOV errors** — any unknown word can be broken into known subwords
- **Captures morphological relationships** — *"boy"* and *"boys"* share the token *"boy"*
- **Balanced vocabulary size** — not too large (word-based) or too small (char-based)
- Retains **semantic meaning** at root level
- Handles rare words, names, technical terms gracefully

#### ❌ Cons
- More complex to implement than word or character tokenization
- Requires training the tokenizer on a large corpus first

### Comparison Table

| Feature | Word-Based | Character-Based | Subword-Based (BPE) |
|---------|-----------|----------------|-------------------|
| OOV Errors | ❌ Yes | ✅ No | ✅ No |
| Vocabulary Size | Very Large | Very Small | Balanced (~50K) |
| Sequence Length | Short | Very Long | Medium |
| Semantic Meaning | ✅ Good | ❌ Poor | ✅ Good |
| Morphological Relationships | ❌ Misses | ❌ Misses | ✅ Captures |
| Used by GPT-2/3 | ❌ No | ❌ No | ✅ **Yes** |

---

## 3. BPE Algorithm — How It Works

### Background
- BPE = **Byte Pair Encoding**
- Originally a **data compression algorithm** introduced in **1994**
- Later adapted for NLP tokenization
- Core idea: **find the most frequent pair of consecutive bytes/characters and merge them into one new token**

### The Algorithm Step by Step

```
STEP 1: Start with character-level tokenization
        (every character is its own token)

STEP 2: Count all consecutive pairs of tokens in the corpus

STEP 3: Find the MOST FREQUENT pair

STEP 4: Merge that pair into a single new token
        Add the new token to the vocabulary

STEP 5: Repeat Steps 2-4 until:
        - No pair occurs more than once, OR
        - Desired vocabulary size is reached
```

### Simple Example

```
Starting text: "a a b c a a b c"

Pairs count: (a,a)=2, (a,b)=2, (b,c)=2, (c,a)=1

Most frequent (tie): merge (a,a) → "aa"
Text becomes: "aa b c aa b c"

Next pairs: (aa,b)=2, (b,c)=2, (c,aa)=1

Merge (aa,b) → "aab"
Text becomes: "aab c aab c"

...continues until vocabulary is built
```

> 💡 Each merge creates a **new token** that gets added to the vocabulary. After thousands of merges, you have a rich vocabulary of subword tokens.

---

## 4. Manual BPE Walkthrough — Real Dataset Example

### The Corpus
Words used: **old, older, finest, lowest**

### Pre-processing Step
- Add a special **end-of-word marker** (`</w>`) to each word
- This helps track word boundaries during merging

```
old   → o l d </w>
older → o l d e r </w>
finest → f i n e s t </w>
lowest → l o w e s t </w>
```

### Starting Vocabulary (all individual characters)
```
{o, l, d, e, r, f, i, n, s, t, w, </w>}
```

### Iteration 1 — Find Most Frequent Pair
- Count all consecutive pairs across all words
- Most frequent pair found: **e + s**
- Merge → new token: **"es"**

```
finest → f i n es t </w>
lowest → l o w es t </w>
```

### Iteration 2 — Next Most Frequent Pair
- New most frequent pair: **es + t**
- Merge → new token: **"est"**

```
finest → f i n est </w>
lowest → l o w est </w>
```

### Iteration 3 — Continue Merging
- Eventually merges: **o + l + d** → **"old"**

```
old   → old </w>
older → old e r </w>
```

### Final Result — What BPE Learned

| Token Created | Meaning Captured |
|--------------|-----------------|
| **"est"** | Superlative suffix (finest, lowest, greatest...) |
| **"old"** | Root word preserved (old, older, oldest...) |
| **"er"** | Comparative suffix |

> 🎯 **Key insight:** BPE automatically discovers that *"est"* is a meaningful suffix — something word-based tokenization would completely miss! The algorithm learns **morphological structure** purely from data frequency.

---

## 5. Implementation with `tiktoken`

### What is tiktoken?
- **tiktoken** = OpenAI's fast BPE tokenizer library
- Used for GPT-2, GPT-3, GPT-4 tokenization
- Available as a Python package

### Code Example

```python
import tiktoken

# Initialize the GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Encode a sentence
text = "in the sunlit terraces of some unknown place"
tokens = tokenizer.encode(text)
print(tokens)
# Output: [259, 262, 4252, 18250, 8812, 2114, 286, 617, 6439, 1295]

# Decode back to text
decoded = tokenizer.decode(tokens)
print(decoded)
# Output: "in the sunlit terraces of some unknown place"

# Check vocabulary size
print(tokenizer.n_vocab)
# Output: 50257
```

### Testing with OOV Words
- Even completely made-up or rare words like *"sunlit"* and *"terraces"* are handled without any error
- BPE simply breaks them into known subword pieces
- **Zero OOV errors** — the tokenizer never crashes on unknown input

### GPT-2 Vocabulary Size
- GPT-2 uses exactly **50,257 tokens** in its vocabulary
- This was carefully chosen to:
  - Cover common English words as single tokens
  - Handle rare/unknown words via subword decomposition
  - Keep vocabulary size manageable for computation

---

## 6. Why BPE is the Best Choice for LLMs

### Summary of Benefits

| Benefit | Why It Matters |
|---------|---------------|
| **No OOV errors** | LLM never fails on new/unseen words |
| **Optimized vocabulary (~50K)** | Not too large, not too small — efficient computation |
| **Captures morphology** | *"est"*, *"ing"*, *"old"* etc. learned automatically |
| **Handles any language** | Works on code, math, foreign languages, emojis |
| **Balances sequence length** | Not as long as char-based, not as short as word-based |
| **Learned from data** | Vocabulary reflects actual patterns in training corpus |

### The Core Trade-off BPE Solves

```
Word-based:      Great meaning, terrible OOV handling
Character-based: No OOV, terrible meaning and long sequences
BPE:             ✅ No OOV + ✅ Good meaning + ✅ Efficient sequences
```

---

## 🔑 Key Terms to Know

| Term | Simple Meaning |
|------|---------------|
| **Tokenization** | Breaking text into smaller units (tokens) for the model |
| **Token** | A unit of text — could be a word, subword, or character |
| **Vocabulary** | The complete set of all tokens the model knows |
| **OOV (Out-of-Vocabulary)** | A word the tokenizer has never seen — causes errors in word-based tokenization |
| **BPE (Byte Pair Encoding)** | Subword tokenization algorithm — merges frequent character pairs iteratively |
| **Subword** | A piece of a word — e.g., *"est"* in *"finest"* |
| **Morphology** | The structure of words — roots, prefixes, suffixes |
| **End-of-word token (`</w>`)** | Special marker added during BPE to track word boundaries |
| **Merge** | Combining two tokens into one new token in BPE |
| **tiktoken** | OpenAI's Python library for BPE tokenization |
| **50,257** | GPT-2's vocabulary size — the number of unique tokens it knows |
| **Data compression** | Original purpose of BPE (1994) — reduce file size by replacing frequent pairs |

---

## ⚡ Quick Summary

1. **Tokenization** = splitting text into tokens before feeding to an LLM
2. **Word-based** = simple but suffers from OOV errors and misses morphology
3. **Character-based** = no OOV but loses meaning and creates very long sequences
4. **Subword-based (BPE)** = best of both worlds — used by GPT-2, GPT-3, and most modern LLMs
5. **BPE algorithm** = iteratively merges the most frequent consecutive byte pairs until vocabulary is complete
6. BPE was originally a **data compression algorithm from 1994**, later adapted for NLP
7. Manual walkthrough: corpus of *old, older, finest, lowest* → BPE learns *"est"*, *"old"*, *"er"* as meaningful tokens
8. **tiktoken** = OpenAI's BPE library — GPT-2 tokenizer has **50,257 tokens**
9. BPE handles **any unseen word** by decomposing it into known subword units — zero OOV errors

---

## 🧠 Key Takeaways to Remember

- 🔹 **BPE = Byte Pair Encoding** — the tokenization method behind GPT-2/3
- 🔹 **Three methods:** Word (OOV problem) → Character (meaning problem) → **Subword/BPE** (best!)
- 🔹 **BPE algorithm:** find most frequent pair → merge → repeat
- 🔹 BPE learns **meaningful subwords** like *"est"*, *"ing"*, *"old"* purely from frequency
- 🔹 **tiktoken** is OpenAI's library to use BPE — GPT-2 vocab = **50,257 tokens**
- 🔹 BPE **never crashes on unknown words** — always decomposes into subword pieces
- 🔹 `</w>` end-of-word marker is added during BPE pre-processing to track boundaries

---

## 📌 What's Coming Next

- Deeper dive into **tokenization implementation in code**
- **Vector Embeddings** — converting token IDs into high-dimensional vectors
- **Positional Encoding** — telling the model where each token appears in the sequence

---

*Notes prepared from Vizuara YouTube Series: "Build a Large Language Model from Scratch"*
*GitHub-ready markdown — push directly to your notes repo* 🚀
