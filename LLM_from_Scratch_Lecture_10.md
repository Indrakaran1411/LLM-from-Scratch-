# 📘 Build a Large Language Model from Scratch
## Lecture 10 — Token Embeddings & The Embedding Layer
> **Source:** Vizuara YouTube Channel | **Instructor:** Dr. Raj Dandekar
> **Video:** https://youtu.be/ghCSGRgVB_o
> **Series:** Stage 1 — Data Preparation & Sampling

---

## 🗺️ What This Lecture Covers

1. What are Token Embeddings and why are they needed?
2. Why Random Token IDs and One-Hot Encoding fail
3. How Vectors capture Semantic Meaning
4. Hands-On Demo — Word2Vec Google News 300
5. How Token Embeddings are created for LLMs
6. The Embedding Layer Weight Matrix
7. Embedding Layer as a Lookup Table
8. Embedding Layer vs. Neural Network Linear Layer

---

## The LLM Input Pipeline — Where Embeddings Fit

```
Raw Text
    ↓
Step 1: Tokenization      → ["This", "is", "an", "example"]
    ↓
Step 2: Token IDs         → [101, 204, 39, 567]
    ↓
Step 3: Token Embeddings  → [[0.2, 0.8, ...], [0.5, 0.1, ...], ...]
    ↓
LLM Training (GPT)
    ↓
Output
```

> Today's lecture focuses on **Step 3** — converting Token IDs into Token Embeddings.

---

## 1. Why Do We Need Token Embeddings?

### The Core Problem
Computers can't understand words — they only understand numbers.
So we need to represent words as numbers. But **how** we do this matters enormously.

### Attempt 1 — Random Number Assignment ❌

```
cat    → 34
book   → 2.9
tablet → -20
kitten → -33
```

**Problem:** cat and kitten are closely related, but 34 and -33 are completely unrelated numbers. The **semantic meaning is lost**.

---

### Attempt 2 — One-Hot Encoding ❌

```
Vocabulary: [dog, puppy, cat, apple, banana]

dog    → [1, 0, 0, 0, 0]
puppy  → [0, 1, 0, 0, 0]
cat    → [0, 0, 1, 0, 0]
apple  → [0, 0, 0, 1, 0]
banana → [0, 0, 0, 0, 1]
```

**Problem:** dog and puppy should be close — they're related. But their one-hot vectors are completely different — **no relationship is encoded at all**.

---

### The Real Insight — Words Carry Meaning

Just like **CNNs** exploit the **spatial relationships between pixels** in images (eyes close together, ears close together), we should exploit the **semantic relationships between words** in text.

> 💡 **Analogy:**
> - CNNs for images → exploit spatial proximity of pixels
> - Token Embeddings for text → exploit semantic proximity of words

We need a representation where:
- *cat* and *kitten* are **close** to each other
- *dog* and *puppy* are **close** to each other
- *dog* and *banana* are **far** from each other

---

## 2. How Vectors Capture Semantic Meaning

### The Key Idea
> Represent each word as a **multi-dimensional vector** where the values reflect meaningful features of that word.

### Feature-Based Example

Words: **dog, cat, apple, banana**
Features: has a tail | is edible | has four legs | makes sound | is a pet

| Word | Has Tail | Is Edible | Has 4 Legs | Makes Sound | Is a Pet |
|------|---------|-----------|-----------|------------|---------|
| **dog** | HIGH | low | HIGH | HIGH | HIGH |
| **cat** | HIGH | low | HIGH | HIGH | HIGH |
| **apple** | low | HIGH | low | low | low |
| **banana** | low | HIGH | low | low | low |

### What This Shows
- **dog** and **cat** → similar feature values → **vectors are close**
- **apple** and **banana** → similar feature values → **vectors are close**
- **dog** and **banana** → very different feature values → **vectors are far apart**

> This is the magic of vector embeddings — **similar words cluster together** in vector space!

---

## 3. Hands-On Demo — Word2Vec (Google News 300)

### What is Word2Vec Google News 300?
- Pre-trained vector embeddings by **Google**
- Trained on **100 billion words** from Google News
- Every word → a **300-dimensional vector**
- Already available as a pre-trained model

### Demo 1 — King + Woman − Man = ?

```python
result = word_vectors['king'] + word_vectors['woman'] - word_vectors['man']
similar = word_vectors.most_similar([result])
# Output: [('queen', 0.71), ...]
```

**Answer: Queen (71% confidence)** ✅

**Why?**
```
King encodes: royalty + masculinity
Woman encodes: femininity
Man encodes: masculinity

King + Woman - Man
= royalty + masculinity + femininity - masculinity
= royalty + femininity
= QUEEN
```

> 🤯 This proves vectors genuinely encode meaning — arithmetic on words works!

### Demo 2 — Similarity Scores

```python
word_vectors.similarity('woman', 'man')    # HIGH similarity ✅
word_vectors.similarity('king', 'queen')   # HIGH similarity ✅
word_vectors.similarity('uncle', 'aunt')   # HIGH similarity ✅
word_vectors.similarity('boy', 'girl')     # HIGH similarity ✅
word_vectors.similarity('paper', 'water')  # LOW similarity ✅
```

Related word pairs → high similarity score
Unrelated words → low similarity score

### Demo 3 — Vector Distance Shows Meaning

```python
np.linalg.norm(word_vectors['man'] - word_vectors['woman'])    # 1.73  (close)
np.linalg.norm(word_vectors['nephew'] - word_vectors['niece']) # 1.96  (close)
np.linalg.norm(word_vectors['semiconductor'] - word_vectors['earthworm']) # 5.67 (far)
```

> 💡 **Key rule:** Smaller vector distance = more similar meaning. Larger distance = less related.

### Demo 4 — Find Similar Words

```python
word_vectors.most_similar('tower')
# Output: ['skyscraper', 'spire', 'turret', ...]  ← all architecture-related!
```

---

## 4. How Token Embeddings Are Created for LLMs

### Two Things You Need

| Parameter | GPT-2 Value | Meaning |
|-----------|------------|---------|
| **Vocabulary Size** | 50,257 | Total number of unique tokens |
| **Vector Dimension** | 768 (small model) | Size of each embedding vector |

### The Embedding Weight Matrix

```
Embedding Weight Matrix shape = (Vocabulary Size) × (Vector Dimension)
For GPT-2 = 50,257 × 768

Row 0    → 768-dimensional vector for token ID 0
Row 1    → 768-dimensional vector for token ID 1
Row 2    → 768-dimensional vector for token ID 2
...
Row 50256 → 768-dimensional vector for token ID 50,256
```

**Total parameters in just the embedding layer:**
```
50,257 × 768 = ~38.6 Million parameters
```

### How Are the Values Determined?

```
Step 1: Initialize all weights RANDOMLY
           ↓
Step 2: Train the LLM (predict next word)
           ↓
Step 3: Backpropagation updates embedding weights
           ↓
Step 4: After training, embedding weights capture semantic meaning
```

> 💡 Embedding weights are **not pre-set** — they start random and are learned during LLM training. Two things are optimized simultaneously:
> 1. The **embedding layer weights** (so similar words cluster together)
> 2. The **prediction weights** (so the model correctly predicts the next word)

---

## 5. PyTorch Implementation

### Creating the Embedding Layer

```python
import torch
import torch.nn as nn

# Small example: vocabulary of 6 words, 3-dimensional embeddings
vocab_size = 6       # quick, fox, is, in, the, house
output_dim = 3       # each word → 3-dimensional vector

# Create embedding layer
embedding_layer = nn.Embedding(vocab_size, output_dim)

# Check initial (random) weights
print(embedding_layer.weight)
# Output: 6×3 matrix of random values
# tensor([[ 0.3374, -0.1778, -0.3035],
#         [-0.5880,  0.3486,  0.6603],
#         [-0.2196, -0.3792,  0.7671],
#         [-1.0935,  0.5284,  0.0516],
#         [-0.0868, -0.6562,  0.6905],
#         [ 0.9029, -0.0912,  0.3563]])
```

### GPT-2 Scale

```python
# GPT-2 actual scale
vocab_size = 50257
output_dim = 768

embedding_layer = nn.Embedding(vocab_size, output_dim)
# Creates a 50,257 × 768 matrix initialized randomly
```

---

## 6. The Embedding Layer as a Lookup Table

> **The embedding layer is simply a lookup table** — give it a token ID, it returns the corresponding vector row.

### How Lookup Works

```
Embedding Matrix (6×3):
Row 0: [0.33, -0.17, -0.30]   ← vector for token ID 0 (fox)
Row 1: [-0.58, 0.34,  0.66]   ← vector for token ID 1 (house)
Row 2: [-0.21, -0.37,  0.76]  ← vector for token ID 2 (in)
Row 3: [-1.09,  0.52,  0.05]  ← vector for token ID 3 (is)
Row 4: [-0.08, -0.65,  0.69]  ← vector for token ID 4 (quick)
Row 5: [ 0.90, -0.09,  0.35]  ← vector for token ID 5 (the)
```

### Single ID Lookup

```python
# Get vector for token ID 3 ("is")
vector = embedding_layer(torch.tensor(3))
# Returns: [-1.09, 0.52, 0.05]  ← Row 3 of the matrix
```

### Multiple ID Lookup

```python
# Get vectors for multiple token IDs at once
input_ids = torch.tensor([2, 3, 5, 1])  # [in, is, the, house]
vectors = embedding_layer(input_ids)
# Returns 4×3 matrix — one row per token ID
```

### Visual Summary

```
Input IDs: [2, 3, 5, 1]
               ↓
       Lookup in Matrix
               ↓
Output: Row 2 → vector for "in"
        Row 3 → vector for "is"
        Row 5 → vector for "the"
        Row 1 → vector for "house"
```

---

## 7. Embedding Layer vs. nn.Linear Layer

### They Produce the Same Output!

An embedding layer is mathematically equivalent to:
1. Convert token IDs → **one-hot vectors**
2. Multiply by a **weight matrix** (linear layer)

```python
# These two are mathematically equivalent:
output_embedding = embedding_layer(input_ids)

# Same as:
one_hot = F.one_hot(input_ids, vocab_size).float()
output_linear = nn.Linear(vocab_size, output_dim, bias=False)(one_hot)
```

### Why Embedding Layer is Preferred

| | Embedding Layer | nn.Linear Layer |
|--|----------------|----------------|
| **Method** | Direct row lookup | One-hot × weight matrix |
| **Computation** | O(1) per lookup | O(vocab_size) multiplications |
| **Zero multiplications** | None | Many (one-hot has mostly zeros) |
| **Efficiency** | ✅ Very efficient | ❌ Wasteful at scale |
| **Preferred for LLMs** | ✅ Yes | ❌ No |

> 💡 With a vocabulary of 50,257, using `nn.Linear` would mean ~50,000 unnecessary multiplications by zero for every single token. The embedding layer skips all that with a simple row lookup.

---

## 🔑 Key Terms to Know

| Term | Simple Meaning |
|------|---------------|
| **Token Embedding** | A vector representation of a token that captures its meaning |
| **Vector Embedding** | Same as token embedding — words represented as vectors |
| **Word Embedding** | Less accurate term (tokens can be subwords, not just words) |
| **Semantic Meaning** | The actual meaning/relationship between words |
| **Embedding Weight Matrix** | The full table of all token vectors — shape: (vocab_size × vector_dim) |
| **Vector Dimension** | How many numbers are in each token's vector (GPT-2 small = 768) |
| **Vocabulary Size** | Total number of unique tokens (GPT-2 = 50,257) |
| **Lookup Table** | What the embedding layer really is — input ID → output vector |
| **Word2Vec** | Google's pre-trained word embedding model |
| **300-dimensional** | Each word in Google News Word2Vec mapped to 300 numbers |
| **nn.Embedding** | PyTorch class for creating embedding layers |
| **nn.Linear** | PyTorch linear layer — same output as embedding but less efficient |
| **Random Initialization** | Starting point for embedding weights before training |
| **Backpropagation** | Algorithm that updates embedding weights during training |
| **One-Hot Encoding** | Binary vector representation — only one 1, rest 0s |

---

## ⚡ Quick Summary

1. **Token embeddings** = converting token IDs into meaningful vectors (Step 3 in the LLM pipeline)
2. **Random IDs** and **one-hot encoding** both fail — they don't capture semantic relationships
3. **Vectors can encode meaning** — similar words cluster together in vector space
4. **King + Woman − Man = Queen** proves vectors do semantic arithmetic
5. **Smaller vector distance** = more similar meaning between words
6. GPT-2 embedding matrix = **50,257 × 768** = ~38.6M parameters
7. Embedding weights are **initialized randomly** then **optimized during LLM training**
8. Embedding layer = a **lookup table** — input token ID → output vector (corresponding row)
9. Pass multiple IDs at once → get multiple vectors in one operation
10. `nn.Embedding` is preferred over `nn.Linear` because it avoids wasteful zero-multiplications

---

## 🧠 Key Takeaways to Remember

- 🔹 **Why embeddings?** Random IDs & one-hot fail to capture word meaning
- 🔹 **Core idea:** Represent words as vectors so similar words are close in vector space
- 🔹 **King + Woman − Man = Queen** — vectors truly encode semantic meaning
- 🔹 **GPT-2:** vocab_size = **50,257**, vector_dim = **768**
- 🔹 **Embedding Matrix** = (50,257 × 768) initialized randomly, optimized during training
- 🔹 **Lookup table** = give token ID → get vector from that row of the matrix
- 🔹 `nn.Embedding` >> `nn.Linear` for efficiency at large vocabulary sizes
- 🔹 During LLM training, **both** embedding weights AND prediction weights are optimized

---

## 📌 What's Coming Next — Lecture 11

- **Positional Embeddings** — encoding the *order/position* of tokens in a sentence
- Token embeddings capture **what** a word means
- Positional embeddings capture **where** a word appears
- Both together = the complete input representation for the Transformer

---

*Notes prepared from Vizuara YouTube Series: "Build a Large Language Model from Scratch"*
*GitHub-ready markdown — push directly to your notes repo* 🚀
