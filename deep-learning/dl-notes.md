# Deep Learning Interview Questions & Notes

This document summarizes common **Deep Learning questions and answers**, with clarifications added for better understanding.

**Source:**  
[Deep Learning Interview Prep Course](https://www.youtube.com/watch?v=BAregq0sdyY)

---

## Loss Functions

### Q1. What is a Loss Function and what are various Loss Functions used in Deep Learning?

A loss function measures how much error a neural network makes when producing predictions. It compares the predicted output with the true labels and produces a scalar value representing the model’s error.

Loss functions guide the training process because the gradients of the loss function are used to update the model’s parameters through backpropagation.

Common loss functions include:
- **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** for regression
- **Cross-Entropy loss** for classification problems

The loss function defines the optimization objective and is the sole driver of learning during training.

---

### Q2. What is Cross-Entropy loss function and how is it called in industry?

Cross-entropy loss measures the difference between the true probability distribution and the predicted probability distribution produced by the model.

In industry, cross-entropy loss is often referred to as **log loss**. It is widely used in classification tasks because it works well with probabilistic outputs such as those produced by sigmoid or softmax functions.

For multi-class classification with one-hot labels, cross-entropy reduces to the negative log-probability of the true class, which aligns with maximum likelihood estimation.

---

### Q3. Why is Cross-Entropy preferred for multi-class classification problems?

Cross-entropy is preferred because it strongly penalizes incorrect predictions when the model assigns low probability to the true class. It works naturally with softmax outputs and provides stable gradients during training.

The loss depends only on the predicted probability of the true class. In practice, assigning high probability to an incorrect class implies low probability for the true class due to softmax normalization.

---

## Gradient Descent Variants & Batch Size

### Q4. What is SGD and why is it used in training Neural Networks?

Stochastic Gradient Descent (SGD) is an optimization algorithm that updates model parameters using a single training sample or a small subset of samples at each step.

It is used because computing gradients on the entire dataset for every update is computationally expensive. SGD makes training feasible for large datasets by reducing computation and memory requirements.

In practice, SGD usually refers to **mini-batch SGD**, which balances noisy updates with efficient hardware utilization.

---

### Q5. Why does Stochastic Gradient Descent oscillate towards local minima?

SGD oscillates because it uses only a small subset of data to compute gradients. These gradients are noisy estimates of the true gradient, which causes the updates to fluctuate instead of moving smoothly toward the minimum.

This noise can slow convergence near minima but may help escape saddle points and improve generalization.

---

### Q6. How is Gradient Descent different from SGD?

Gradient Descent uses the entire training dataset to compute gradients and update parameters, resulting in accurate but computationally expensive updates.

SGD updates parameters more frequently using fewer samples, making it faster and more scalable, but also noisier.

---

### Q7. How can optimization methods like Gradient Descent be improved? What is the role of Momentum?

Gradient Descent can be improved by adding a momentum term, which accumulates gradients from previous steps.

Momentum helps accelerate convergence in consistent directions and reduces oscillations, especially in narrow or steep regions of the loss surface. It introduces inertia into optimization, allowing updates to maintain direction across noisy gradients.

---

### Q8. Compare Batch Gradient Descent, Mini-batch Gradient Descent, and SGD.

- **Batch Gradient Descent:** Uses all data per update; stable but slow and memory-intensive  
- **SGD:** Uses one sample per update; very fast but noisy  
- **Mini-batch Gradient Descent:** Uses a small batch of samples, balancing stability and efficiency  

Mini-batch Gradient Descent is most commonly used in practice.

---

### Q9. How to decide batch size in deep learning?

If the batch size is too small, training becomes noisy and unstable.  
If the batch size is too large, training becomes slow, memory-intensive, and may generalize poorly.

Choosing batch size is a trade-off between stability, speed, and hardware constraints.

---

### Q10. How does batch size impact model performance?

Batch size affects convergence speed, training stability, and generalization. Smaller batches introduce noise that can help generalization, while larger batches provide smoother updates but may lead to poorer generalization.

---

## Advanced Optimization

### Q11. What is Hessian, and how can it be used for faster training? What are its disadvantages?

The Hessian matrix contains second-order derivatives of the loss function and captures curvature information. It can be used to accelerate optimization by adjusting step sizes based on curvature.

However, computing the Hessian is computationally expensive and impractical for large neural networks.

---

### Q12. What is RMSProp and how does it work?

RMSProp is an adaptive learning rate method that adjusts the learning rate for each parameter based on a moving average of squared gradients.

It helps prevent very large updates and improves training stability.

---

### Q13. What is an adaptive learning rate? Describe adaptive learning methods.

An adaptive learning rate changes automatically during training instead of remaining fixed. Methods like RMSProp and Adam adjust learning rates per parameter to improve convergence and stability.

---

### Q14. What is Adam and why is it used most of the time in Neural Networks?

Adam is an optimization algorithm that combines momentum and adaptive learning rates. It maintains running averages of both gradients and squared gradients.

Adam is widely used because it converges faster, is robust to noisy gradients, and requires minimal hyperparameter tuning. It also maintains optimizer state, which is important when resuming training.

---

### Q15. What is AdamW and why is it preferred over Adam?

AdamW decouples weight decay from the gradient update step. This improves regularization and generalization compared to Adam, which mixes weight decay with gradient updates.

---

## Normalization, Architecture & Initialization

### Q16. What is Batch Normalization and why is it used?

Batch Normalization normalizes activations across a batch, reducing internal covariate shift. It stabilizes training, accelerates convergence, and allows higher learning rates.

By stabilizing activation distributions, BatchNorm makes training less sensitive to learning rate choice.

---

### Q17. What is Layer Normalization and why is it used?

Layer Normalization normalizes across feature dimensions instead of batch dimension. It works well when batch sizes are small and is commonly used in Transformer architectures.

---

### Q18. What are Residual Connections and their function in Neural Networks?

Residual connections add the input of a layer directly to its output. They improve gradient flow and enable training of very deep neural networks by mitigating vanishing gradients.

---

### Q19. What is Gradient Clipping and its impact on Neural Networks?

Gradient clipping limits the magnitude of gradients to prevent exploding gradients. It stabilizes training, especially in deep or recurrent networks.

---

### Q20. What is Xavier Initialization and why is it used?

Xavier Initialization sets initial weights so that variance is preserved across layers. This helps prevent vanishing and exploding gradients early in training.

---

### Q21. What are different ways to solve Vanishing Gradients?

- Use ReLU or Leaky ReLU  
- Use Batch or Layer Normalization  
- Use Residual Connections  
- Use proper weight initialization  

---

### Q22. What are ways to solve Exploding Gradients?

- Gradient clipping  
- Smaller learning rates  
- Proper initialization  
- Normalization techniques  
