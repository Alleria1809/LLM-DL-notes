# Large Language Model (LLM) Questions & Notes

This document summarizes common **LLM questions and answers**, with clarifications added to strengthen conceptual understanding and connect theory to practice.

---

## Foundations of LLMs

### Q1. What is a Large Language Model?

A Large Language Model is a neural network trained to model the probability distribution of token sequences. Given previous tokens, it predicts the next token using maximum likelihood estimation over large-scale text data.

At its core, an LLM is a probabilistic sequence model that learns statistical regularities of language from massive corpora.

---

### Q2. How is an LLM different from traditional NLP models?

Traditional NLP models rely on task-specific architectures and handcrafted features, often trained separately for each task.

LLMs use a single pretrained model that learns general-purpose language representations and can be adapted to many tasks via prompting or fine-tuning, significantly reducing task-specific engineering.

---

### Q3. What does “autoregressive” mean?

Autoregressive models generate text one token at a time, conditioning each prediction on all previously generated tokens.

Formally, the model factorizes the joint probability of a sequence into a product of conditional probabilities.

---

## Transformer Architecture

### Q4. Why are Transformers used instead of RNNs?

Transformers enable parallel computation across tokens, handle long-range dependencies more effectively, and avoid vanishing gradient issues caused by sequential recurrence in RNNs.

This makes Transformers more scalable and better suited for large datasets and long contexts.

---

### Q5. Explain self-attention.

Self-attention allows each token to attend to all other tokens in the sequence by computing weighted combinations of token representations using query, key, and value projections.

This mechanism enables the model to capture contextual relationships regardless of token distance.

---

### Q6. Why multi-head attention?

Multi-head attention allows the model to capture different types of relationships—such as syntax, semantics, and positional interactions—simultaneously in different representation subspaces.

Each attention head can specialize in a different pattern of dependency.

---

### Q7. What is positional encoding and why is it needed?

Transformers lack inherent awareness of token order. Positional encoding injects sequence order information into token representations so the model can distinguish positions in the input sequence.

---

## Training LLMs

### Q8. What objective is used to train LLMs?

LLMs are trained using Maximum Likelihood Estimation via next-token prediction, typically implemented as cross-entropy loss over the vocabulary.

This objective encourages the model to assign high probability to the correct next token.

---

### Q9. Why is cross-entropy suitable for LLMs?

Cross-entropy directly optimizes the probability assigned to the correct next token and provides stable gradients when combined with softmax.

It aligns naturally with probabilistic language modeling and scales well to large vocabularies.

---

### Q10. What is pretraining vs fine-tuning?

Pretraining learns general language representations from large, diverse corpora.

Fine-tuning adapts the pretrained model to specific tasks, domains, or behaviors by further training on targeted datasets.

---

### Q11. What is instruction tuning?

Instruction tuning fine-tunes an LLM on instruction–response pairs, improving its ability to follow natural language instructions and respond helpfully to user prompts.

This stage bridges raw language modeling and interactive usage.

---

## Optimization & Stability

### Q12. Why are Adam / AdamW commonly used for LLMs?

Adam and AdamW provide adaptive learning rates, handle noisy gradients well, and converge faster in large, high-dimensional parameter spaces.

AdamW is generally preferred because it decouples weight decay from gradient updates, improving regularization and generalization.

---

### Q13. Why is Layer Normalization preferred over Batch Normalization?

Layer Normalization does not depend on batch statistics and works reliably with variable sequence lengths and small batch sizes.

This makes it better suited for Transformers and autoregressive generation than Batch Normalization.

---

### Q14. Why are residual connections critical?

Residual connections enable stable gradient flow across many layers, making very deep Transformer models trainable.

They mitigate vanishing gradients and allow layers to learn incremental refinements rather than entirely new representations.

---

### Q15. What causes training instability in LLMs?

Training instability can arise from:
- Large learning rates
- Poor initialization
- Exploding gradients
- Long sequence lengths
- Numerical issues in attention or softmax

Stability techniques such as normalization, gradient clipping, and careful learning-rate schedules are essential at scale.

---

## Scaling & Efficiency

### Q16. What happens when you scale model size?

Scaling model size increases representational capacity and often improves performance, but also significantly increases training cost, memory usage, and inference latency.

Practical scaling requires careful balancing of compute, data, and optimization.

---

### Q17. What are scaling laws?

Scaling laws are empirical relationships that describe how model performance improves predictably with increased model size, dataset size, and compute.

They guide resource allocation when training large models.

---

### Q18. What is gradient checkpointing?

Gradient checkpointing is a memory-saving technique that recomputes activations during backpropagation instead of storing them, trading extra computation for reduced memory usage.

---

### Q19. What is mixed-precision training?

Mixed-precision training uses lower-precision arithmetic (such as FP16 or BF16) to reduce memory consumption and increase throughput while maintaining numerical stability.

---

## Inference & Decoding

### Q20. Greedy decoding vs sampling?

Greedy decoding selects the highest-probability token at each step, producing deterministic but potentially repetitive outputs.

Sampling introduces randomness, enabling more diverse and creative generations.

---

### Q21. What is temperature?

Temperature scales logits before softmax to control randomness. Lower temperatures make outputs more deterministic, while higher temperatures increase diversity.

---

### Q22. Top-k vs top-p (nucleus sampling)?

Top-k sampling restricts choices to the k most probable tokens.

Top-p (nucleus sampling) selects the smallest set of tokens whose cumulative probability exceeds a threshold p, adapting dynamically to the distribution shape.

---

## Alignment & Safety

### Q23. What is RLHF?

Reinforcement Learning from Human Feedback aligns model outputs with human preferences using a reward model and policy optimization.

It is commonly used to improve helpfulness, safety, and alignment beyond supervised learning.

---

### Q24. Why is supervised fine-tuning not enough?

Supervised fine-tuning teaches correctness but does not adequately capture human preferences, safety constraints, or behavior under ambiguous prompts.

RLHF addresses these gaps.

---

### Q25. What are common alignment challenges?

Common challenges include hallucination, harmful content generation, reward hacking, and distribution shift between training and deployment.

---

## Evaluation & Failure Modes

### Q26. What is hallucination?

Hallucination occurs when an LLM generates fluent but factually incorrect or unsupported content.

---

### Q27. Why do LLMs hallucinate?

LLMs optimize likelihood rather than truth and lack direct grounding or external verification mechanisms.

---

### Q28. How do you evaluate LLMs?

Evaluation methods include perplexity, task-specific benchmarks, human evaluation, and application-specific metrics.

No single metric fully captures real-world performance.

---

## Practical / Applied Questions

### Q29. Difference between prompt engineering and fine-tuning?

Prompt engineering modifies inputs at inference time, while fine-tuning updates model parameters.

Prompting is flexible and cheap; fine-tuning offers stronger, more consistent performance.

---

### Q30. When would you fine-tune instead of prompt?

Fine-tuning is preferred when task patterns are consistent, performance requirements are strict, or prompts become brittle and hard to maintain.

---

## System-Level Considerations

### Q31. Why are LLMs expensive to train?

LLMs are expensive due to quadratic attention costs, large parameter counts, massive datasets, and long training schedules.

---

### Q32. What limits context length?

Context length is limited by memory and compute costs of attention, which scale quadratically with sequence length.

---

### Q33. How do modern models extend context length?

Modern approaches include sparse attention, linear attention variants, key–value caching, and sliding-window mechanisms.
