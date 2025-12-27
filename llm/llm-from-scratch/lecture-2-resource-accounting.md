# CS336: Language Modeling from Scratch  
## Lecture 2 — PyTorch Primitives & Resource Accounting

These notes summarize **Lecture 2** of *Stanford CS336: Language Modeling from Scratch (Spring 2025)*, focusing on PyTorch mechanics and systematic accounting of memory and compute.  
:contentReference[oaicite:1]{index=1}

---

## 1. Lecture Goal: Mechanics + Mindset

This lecture focuses on two types of reminders introduced in Lecture 1:

- **Mechanics:**  
  How PyTorch works at a low level (tensors, operations, optimizers, training loops)

- **Mindset:**  
  Treat **memory and compute as first-class constraints**, not afterthoughts

The emphasis is not on Transformers yet, but on **building blocks** that apply to all deep learning models, including LLMs.

---

## 2. Napkin Math: Why Resource Accounting Matters

The lecture begins with motivating “back-of-the-envelope” questions:

- How long would it take to train a 70B-parameter model on trillions of tokens?
- What is the largest model you can train on 8 H100 GPUs using AdamW?

The goal is to develop the habit of reasoning about:
- **FLOPs**
- **Memory usage**
- **Hardware limits**
- **Dollar cost**

Key takeaway:
> Efficiency is not optional. At scale, inefficiency directly translates to wasted money.

---

## 3. Tensors as the Fundamental Unit

All deep learning components are tensors:
- Parameters
- Gradients
- Optimizer state
- Activations
- Data

### Memory of a Tensor

Memory usage is determined by:
- Number of elements
- Data type (precision)

Example:
- Float32 → 4 bytes per element
- Total memory = number of elements × bytes per element

Large LLM matrices can easily occupy **multiple gigabytes** each.

---

## 4. Floating Point Representations & Trade-offs

### Float32 (FP32)
- 32 bits: sign + exponent + fraction
- Large dynamic range
- Stable but memory- and compute-expensive
- Default and safest choice

### Float16 (FP16)
- 16 bits
- Reduced dynamic range
- Prone to underflow / overflow
- Generally discouraged for modern large-scale training

### BFloat16 (BF16)
- Same memory as FP16
- Dynamic range similar to FP32
- Lower precision but sufficient for deep learning
- Common choice for forward / backward computation

### FP8
- 8-bit representation (H100 and newer)
- Extremely fast and memory-efficient
- Numerically fragile
- Requires careful model and system design

**General rule:**
- Parameters and optimizer states → FP32
- Forward / backward passes → BF16 or lower
- This leads to **mixed-precision training**

---

## 5. Compute Accounting: Where Computation Happens

### CPU vs GPU

- Tensors default to CPU memory
- GPU computation requires explicit tensor transfer
- Data movement between CPU and GPU is expensive

Key habit:
> Always know where your tensors live.

---

## 6. Tensor Internals: Storage, Views, and Contiguity

In PyTorch:
- A tensor is a **pointer to memory + metadata**
- Metadata includes:
  - Shape
  - Strides

Multiple tensors can share the same underlying storage.

### Views
- Operations like slicing, transpose, and reshape often create **views**
- Views are cheap (no memory allocation)
- Mutating one view mutates all views sharing storage

### Contiguity
- Some operations require contiguous memory
- Calling `.contiguous()` may create a copy
- Copies increase memory usage and should be minimized

---

## 7. Matrix Multiplication as the Dominant Cost

Matrix multiplication dominates compute in deep learning.

### FLOP Counting Rule

For matrix multiplication:
FLOPs ≈ 2 × (product of dimensions)

This rule generalizes:
- Linear layers
- Most Transformer components
- Many LLM computations

Other operations are typically negligible compared to matmuls at scale.

---

## 8. FLOPs vs FLOP/s and Model FLOPs Utilization (MFU)

- **FLOPs:** total number of floating-point operations
- **FLOP/s:** hardware throughput (speed)

### Model FLOPs Utilization (MFU)

MFU measures:
MFU = actual useful FLOPs / theoretical peak FLOPs


Rules of thumb:
- MFU > 0.5 → good
- MFU < 0.05 → very inefficient

MFU helps quantify how well you are using your hardware, independent of clever optimizations.

---

## 9. Forward vs Backward Pass Cost

For typical neural networks:
- Forward pass ≈ 2 × (tokens × parameters)
- Backward pass ≈ 4 × (tokens × parameters)

**Total training cost:**
≈ 6 × tokens × parameters


This explains the recurring “×6” factor in large-scale training cost estimates.

---

## 10. Parameter Initialization

Naive initialization (e.g., standard Gaussian) can cause activations to explode with depth.

Solution:
- Scale initialization by inverse square root of input dimension
- Leads to stable activation variance

This idea corresponds to **Xavier (normalized) initialization**, often with truncation for safety.

---

## 11. Building Models Bottom-Up

The lecture demonstrates constructing a simple deep linear network:
- Multiple linear layers
- Explicit parameter counting
- GPU placement
- Forward pass tracing

The goal is not model quality, but **understanding parameter count, memory use, and compute cost**.

---

## 12. Randomness & Reproducibility

Sources of randomness include:
- Initialization
- Dropout
- Data ordering

Best practice:
- Set explicit random seeds
- Use different seeds for different sources

Determinism is critical for debugging and reproducibility.

---

## 13. Data Loading at Scale

LLM data is typically:
- Tokenized sequences of integers
- Stored on disk, not fully loaded into memory

Technique:
- **Memory mapping (mmap)** allows on-demand loading
- Enables working with multi-terabyte datasets

---

## 14. Optimizers & Optimizer State

Optimizers discussed:
- SGD
- Momentum
- AdaGrad
- RMSProp
- Adam

Key insight:
> Optimizers store **state**, not just parameters.

Example:
- AdaGrad stores accumulated squared gradients
- Adam stores first and second moments

Optimizer state often consumes **as much memory as the model parameters themselves**.

---

## 15. Memory Breakdown in Training

Total memory includes:
- Parameters
- Activations
- Gradients
- Optimizer state

For large models, optimizer state and activations can dominate memory usage.

Activation checkpointing can trade compute for memory by recomputing activations during backpropagation.

---

## 16. Training Loop & Checkpointing

A typical training loop:
1. Load batch
2. Forward pass
3. Compute loss
4. Backward pass
5. Optimizer step

Checkpointing should save:
- Model parameters
- Optimizer state
- Training step / iteration

This enables **true resumption** of training, not just restarting from weights.

---

## 17. Mixed Precision as a Systems–Model Co-Design

Precision choice affects:
- Speed
- Memory
- Stability

Recommended strategy:
- Use FP32 where stability matters
- Use BF16 / FP8 where possible
- Leverage PyTorch AMP for automation

Long-term trend:
> Model architectures increasingly adapt to hardware capabilities.

---

## 18. Key Takeaway

> Training language models is fundamentally an exercise in **resource management**.

Understanding tensors, memory, FLOPs, and optimizer state is essential to:
- Scaling models
- Reducing cost
- Designing efficient systems

This lecture establishes the computational foundation for all subsequent work in the course.
