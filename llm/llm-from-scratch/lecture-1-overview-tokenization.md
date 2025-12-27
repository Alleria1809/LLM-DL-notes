# CS336: Language Modeling from Scratch  
## Lecture 1 — Overview & Tokenization

These notes summarize **Lecture 1** of [*Stanford CS336: Language Modeling from Scratch (Spring 2025)*](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_), based solely on the lecture transcript.  
:contentReference[oaicite:0]{index=0}

---

## 1. Motivation: Why Build Language Models from Scratch?

The course is motivated by a growing disconnect between **LLM users/researchers** and the **underlying systems and algorithms**.

Historically:
- Researchers implemented and trained models themselves
- Later, models like BERT were downloaded and fine-tuned
- Today, many workflows rely purely on prompting proprietary APIs

While abstraction enables productivity, **LLM abstractions are leaky**:
- String-in / string-out hides critical modeling, data, and system choices
- Fundamental research often requires *co-designing data, models, and systems*

**Core philosophy of the course:**  
> *To truly understand language models, you must build them.*

---

## 2. Reality Check: Frontier Models vs Academic Constraints

Modern frontier models:
- GPT-4 rumored at ~1.8T parameters
- Training costs on the order of \$100M+
- Built with massive proprietary clusters (e.g. 100k+ H100s)
- Details largely undisclosed due to competition and safety

As a result:
- Training GPT-4–scale models is out of reach
- Small models may not faithfully reflect large-scale behavior

Two key examples:
1. **Compute distribution shifts with scale**  
   - At small scale, attention and MLP FLOPs are comparable  
   - At large scale, MLP dominates → optimizing attention alone can be misleading

2. **Emergent behaviors**  
   - Capabilities like in-context learning appear only beyond certain scale thresholds  
   - Small-scale experiments may falsely suggest models “don’t work”

---

## 3. What This Course Can (and Cannot) Teach

The instructors frame learning into **three types of knowledge**:

### 1. Mechanics (Teach-able)
- Tokenization
- Transformer architectures
- Parallelism and GPU utilization
- Training pipelines

### 2. Mindset (Critical)
- Take **scaling seriously**
- Maximize **hardware efficiency**
- Treat compute as a constrained, valuable resource
- Think in terms of *accuracy per FLOP*

This “scaling mindset” is credited as a major driver of modern LLM breakthroughs.

### 3. Intuition (Partially Teach-able)
- Which data, architectures, and design choices work well
- Difficult to fully transfer, since behaviors change with scale
- Experiments often matter more than explanations

> Some architectural choices work “because they work” — experiments speak louder than theory.

---

## 4. The Bitter Lesson (Reinterpreted)

A common misconception:
> “Scale is all that matters; algorithms don’t.”

The course argues instead:
> **Algorithms × Scale = Performance**

Key point:
- Model quality ≈ efficiency × resources
- At large scale, inefficiency becomes prohibitively expensive

Evidence:
- Algorithmic efficiency improved **~44× (2012–2019)** on ImageNet
- Faster than Moore’s Law
- Without this, costs would be 44× higher

**Correct framing:**  
> *What is the best model you can train given a fixed compute and data budget?*

---

## 5. Historical Context: How We Got Here

Key milestones:
- Shannon: language modeling as entropy estimation
- Large N-gram models at Google (2007, trillions of tokens)
- Neural language models (Bengio et al., 2003)
- Seq2Seq models
- Adam optimizer
- Attention mechanisms
- Transformer (2017)
- Model parallelism and MoE research (late 2010s)

Foundation models:
- ELMo, BERT, T5
- GPT-2 / GPT-3 emphasized scaling + engineering
- Rise of open models (Eleuther, BLOOM, Meta, DeepSeek, etc.)

Degrees of openness:
1. Closed models (API only)
2. Open weights
3. Detailed papers (no data)
4. Fully open source (weights + data)

---

## 6. Course Structure & Philosophy

The course is organized around **efficiency-first design**, assuming a compute-constrained regime.

### Five Core Units
1. **Basics** — tokenizer, transformer, training loop
2. **Systems** — kernels, GPUs, parallelism, inference
3. **Scaling Laws** — compute-optimal training
4. **Data** — filtering, deduplication, evaluation
5. **Alignment** — SFT, preference learning, RL

Assignments:
- No scaffolding code
- Emphasis on correctness *and* performance
- Heavy focus on benchmarking and profiling
- Leaderboards constrained by compute budgets

---

## 7. Tokenization: Why It Matters

Tokenization converts raw text into integer sequences consumable by models.

Requirements:
- Reversible (encode → decode)
- Efficient in sequence length
- Reasonable vocabulary size

### Naive Approaches and Limitations

**Character-based**
- Very large vocab
- Inefficient frequency usage

**Byte-based**
- Small vocab (256)
- Very long sequences
- Poor compression (1 byte per token)
- Inefficient due to quadratic attention cost

**Word-based**
- Adaptive, but:
  - Unbounded vocabulary
  - OOV issues
  - Poor handling of rare words

---

## 8. Byte Pair Encoding (BPE)

BPE:
- Originally a compression algorithm (1994)
- Introduced to NLP for neural machine translation
- Used by GPT-2 and many modern models

Key properties:
- Adaptive token lengths
- Fixed vocabulary size
- Good compression vs sequence length trade-off
- Reversible mapping

Important details:
- Spaces are typically included as part of tokens
- Tokens are not semantic units; they are efficiency-driven

---

## 9. Efficiency as the Unifying Principle

Across all course components:
- Tokenization → shorter sequences
- Architecture → compute-efficient blocks
- Training → often single epoch (see more data, not same data)
- Scaling laws → extrapolate cheaply
- Data → aggressive filtering
- Alignment → reduce base model size needed

The course assumes a **compute-limited regime**, which shapes all design decisions.

---

## 10. Key Takeaway

> Language modeling is not just about architectures —  
> it is about **efficiency-aware co-design of data, models, and systems**.

Understanding LLMs deeply requires building them, measuring them, and reasoning under real constraints.
