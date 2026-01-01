# CS336: Language Modeling from Scratch  
## Lecture 3 — Transformer Architecture & Hyperparameters

These notes summarize **Lecture 3** of *Stanford CS336: Language Modeling from Scratch*, focusing on **modern Transformer architecture consensus** and **empirically validated hyperparameter choices** derived from large-scale LLMs. :contentReference[oaicite:0]{index=0}

---

## 1. Lecture Theme

Since training frontier-scale LLMs is infeasible in class settings, this lecture distills lessons from the **evolution of successful models**, identifying architectural and hyperparameter choices that consistently work in practice.

---

## 2. From Vanilla to Modern Transformers

Compared to the original Transformer (2017), modern LLMs have converged on:

- **Pre-norm** instead of post-norm  
- **RoPE** (Rotary Position Embeddings)  
- **GLU-style MLPs** (e.g., SwiGLU)  
- **RMSNorm** instead of LayerNorm  
- **Bias-free linear layers**  
- Mostly **serial attention → MLP blocks**

These choices define today’s common “LLaMA-like” architecture.

---

## 3. Normalization & Stability

### 3.1 Pre-Norm vs Post-Norm

- **Post-norm (original)**: `LN(x + F(x))`  
- **Pre-norm (modern)**: `x + F(LN(x))`

**Why pre-norm dominates**:
- Preserves identity gradient paths
- Avoids gradient attenuation
- Enables stable training of deep Transformers
- Removes need for careful learning-rate warmup

Nearly all modern LLMs use pre-norm.

---

### 3.2 RMSNorm vs LayerNorm

LayerNorm:
$$
\text{LN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

RMSNorm:
$$
\text{RMSNorm}(x) = \gamma \frac{x}{\sqrt{\mathbb{E}[x^2] + \epsilon}}
$$

Key differences:
- No mean subtraction
- No bias term
- Fewer parameters and memory reads

**Why RMSNorm is preferred**:
- Similar model quality to LayerNorm
- Faster in practice due to reduced memory movement
- Normalization ops are FLOP-light but runtime-heavy

Most modern models (LLaMA, PaLM, T5, Chinchilla) use RMSNorm.

---

### 3.3 Dropping Bias Terms

- Modern Transformers often remove bias terms from linear layers
- Empirically improves training stability
- Reduces parameter count and memory traffic
- Biases are unnecessary when normalization is present

---

## 4. MLP Activations & Gating

### 4.1 From ReLU / GeLU to GLUs

- **ReLU**: hard threshold, non-smooth  
- **GeLU**: smooth, probabilistic gating (used in GPT-2/3)  

Modern models favor **Gated Linear Units (GLUs)**:
- ReGLU, GeGLU, **SwiGLU** (most common)

General form:
$$
\text{MLP}(x) = W_2 \big( \phi(xW_1) \odot (xV) \big)
$$

Where:
- `φ` = activation (ReLU, GeLU, Swish)
- `⊙` = elementwise gating

**Why GLUs work well**:
- Explicit gating improves expressiveness
- Consistent empirical gains across models
- Gating is a recurring successful design pattern

GLUs are helpful but **not strictly necessary** (e.g., GPT-3 works without them).

---

## 5. Serial vs Parallel Transformer Blocks

### Serial (Standard)
- Attention → MLP sequentially
- More expressive composition
- Most common today

### Parallel
- Attention and MLP computed in parallel, then added
- Better system-level parallelism
- Used in GPT-J, PaLM, some Cohere models

Parallel blocks trade expressiveness for potential efficiency.

---

## 6. Position Embeddings: RoPE

### Motivation
Language modeling depends on **relative positions**, not absolute indices.

### Rotary Position Embeddings (RoPE)

- Encodes position via **rotations** of query/key vectors
- Inner products depend only on relative position
- Applied **inside attention**, not added to embeddings

Mechanism:
- Split embedding into 2D pairs
- Rotate each pair by a position-dependent angle
- Use multiple rotation frequencies (like sin/cos)

**Why RoPE won**:
- Strong empirical performance
- Enables context-length extrapolation
- Now used by almost all modern LLMs

---

## 7. MLP Width (d_ff) Selection

Let:
- `d_model` = hidden size
- `d_ff` = MLP expansion size

### Standard Rules
- **ReLU / GeLU MLP**:  
  $$
  d_{ff} \approx 4 \times d_{model}
  $$

- **GLU-based MLPs** (parameter-matched):  
  $$
  d_{ff} \approx \frac{8}{3} \times d_{model} \ (\approx 2.6\times)
  $$

These ratios are widely used and empirically validated.

### Exception: T5
- Used extremely large `d_ff` (≈64×)
- Worked, but later T5v1.1 reverted to standard ratios
- Shows defaults are robust but not mandatory

---

## 8. Attention Head Dimensions

Consensus choice:
$$
d_{model} = n_{heads} \times d_{head}
$$

- Keeps per-head dimension fixed
- Avoids low-rank attention bottlenecks
- Used by GPT-3, PaLM, LLaMA, T5

This hyperparameter is rarely tuned in practice.

---

## 9. Depth–Width Aspect Ratio

Define:

$$
\text{Aspect ratio} = \frac{d_{\text{model}}}{n_{\text{layers}}}
$$

Empirical consensus:
- ~**128 hidden dimensions per layer**
- Stable across multiple model scales

Findings:
- Pretraining loss is mostly parameter-count driven
- Downstream performance may favor deeper models
- Aspect-ratio optima are relatively scale-invariant

---

## 10. Vocabulary Size

Trends:
- Early models: 30k–50k tokens
- Modern production models: **100k–250k tokens**

Reasons:
- Multilingual coverage
- Emojis, symbols, mixed scripts
- Fewer tokens per word for low-resource languages → cheaper inference

Large vocabularies help multilingual and production use cases.

---

## 11. Regularization in Pretraining

### Dropout
- Largely abandoned
- Pretraining rarely overfits (often <1 epoch)

### Weight Decay (Still Used)

Not for regularization, but for **optimization dynamics**:
- Interacts with learning-rate schedules
- Improves final training loss during LR decay
- Affects late-stage convergence behavior

Weight decay is used to **optimize better**, not to prevent overfitting.

---

## 12. Key Takeaways

- Modern Transformer design is highly convergent
- Stability and memory efficiency matter as much as FLOPs
- Most hyperparameters have robust, well-tested defaults
- Architectural “details” (norms, gating, bias) materially affect trainability
- Deviations are possible, but defaults are safe and effective


## 13. Transformer Hyperparameter Cheat Sheet (Modern Defaults)

| Component | Symbol | Common Default | Notes / Rationale |
|---------|--------|----------------|-------------------|
| Model width | `d_model` | Scale-dependent | Primary knob controlling capacity |
| Number of layers | `n_layers` | `d_model / 128` | Matches the aspect-ratio sweet spot |
| Aspect ratio | `d_model / n_layers` | ≈ **128** | Empirical optimum across scales |
| Attention heads | `n_heads` | `d_model / d_head` | Usually chosen to keep ratio consistent |
| Head dimension | `d_head` | 64 or 128 | Fixed per-head dimension is common |
| Attention dim ratio | `d_model : (n_heads × d_head)` | **1 : 1** | Most models follow this |
| MLP expansion (ReLU/GeLU) | `d_ff` | `4 × d_model` | Robust default |
| MLP expansion (GLU) | `d_ff` | `(8/3) × d_model ≈ 2.6×` | Parameter-matched with non-GLU MLP |
| Activation | — | **SwiGLU** | Dominant modern choice |
| Normalization placement | — | **Pre-norm** | Key for stable training |
| Norm type | — | **RMSNorm** | Faster; fewer params; similar quality |
| Bias terms | — | **None** | Often improves stability + reduces memory traffic |
| Position embedding | — | **RoPE** | Relative positions; supports extrapolation |
| Transformer block | — | Serial | Parallel blocks used selectively |
| Vocabulary size | `|V|` | 100k–250k | Typical for multilingual / production |
| Dropout | — | 0 or very small | Often omitted in pretraining |
| Weight decay | — | Non-zero | Used for optimization dynamics (not overfitting) |
| Training epochs | — | ≈ 1 | Data-rich, compute-limited regime |