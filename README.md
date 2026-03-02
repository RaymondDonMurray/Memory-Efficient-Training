# Memory Optimization Techniques for Deep Neural Networks

## Overview

This repository documents my exploration of four memory optimization techniques for deep learning models, implemented as part of a graduate-level deep learning course. The project demonstrates practical approaches to reduce memory consumption while maintaining model performance.

**Note**: This repository contains results and analysis only. Full implementations are not included to maintain academic integrity, as this was a course assignment.

---

## Problem Statement

Modern deep learning models often require substantial memory resources, limiting deployment on resource-constrained devices. This project explores techniques to reduce memory footprint from a baseline 72MB model while preserving accuracy and, in some cases, maintaining trainability for fine-tuning.

---

## Baseline Model

The baseline network (BigNet) is a 6-block residual architecture with the following characteristics:

| Metric | Value |
|--------|-------|
| Total parameters | 18.90 M |
| Memory footprint | 72.11 MB |
| Parameter precision | float32 (32 bits) |

All optimizations are compared against this baseline.

---

## Technique 1: Half Precision (float16)

### Theory

Half-precision training reduces memory by storing weights in 16-bit floating point format instead of 32-bit. This simple dtype conversion cuts memory usage in half with minimal accuracy loss for inference tasks.

**Key Concept**: While weights are stored in float16, careful dtype management is required during computation to prevent PyTorch type mismatch errors.

### Results

| Metric | Baseline | Half Precision | Improvement |
|--------|----------|----------------|-------------|
| Theoretical memory | 72.11 MB | 36.07 MB | **-50.0%** |
| Actual memory | 72.11 MB | 36.07 MB | **-50.0%** |
| Max difference | - | 0.0014 | - |
| Mean difference | - | 0.0002 | - |
| Trainable | Yes | No | - |

### Analysis

- Exact 50% memory reduction (32 bits → 16 bits)
- Negligible accuracy loss (mean difference < 0.001)
- Not suitable for training due to gradient numerical instability
- Ideal for inference-only deployment scenarios

---

## Technique 2: LoRA (Low-Rank Adaptation)

### Theory

LoRA enables parameter-efficient fine-tuning by adding small trainable low-rank matrices to frozen pretrained weights:

```
output = frozen_weights(x) + B × A(x)
```

Where:
- Frozen weights: Stored in float16 (memory efficient)
- Matrix A: Projects input to low-rank space (e.g., 1024 → 32 dimensions)
- Matrix B: Projects back to output space (e.g., 32 → 1024 dimensions)

For a 1024×1024 weight matrix:
- Full fine-tuning: 1,048,576 trainable parameters
- LoRA (rank=32): 65,536 trainable parameters (**94% reduction**)

### Results

| Metric | Baseline | LoRA | Improvement |
|--------|----------|------|-------------|
| Theoretical memory | 72.11 MB | 40.57 MB | **-43.7%** |
| Actual memory | 72.11 MB | 40.57 MB | **-43.7%** |
| Trainable params | 18.90 M | 1.19 M | **-93.7%** |
| Backward memory | 72.11 MB | 4.66 MB | **-93.5%** |
| Max difference | - | 0.0015 | - |
| Mean difference | - | 0.0002 | - |
| Trainable | Yes | Yes | ✓ |
| Training accuracy | - | 100% | - |

### Analysis

- Significant memory savings from half-precision base + small adapters
- Backward memory reduced 93.5% (gradients only for adapters)
- Maintains full model accuracy while enabling fine-tuning
- Successfully overfits to training data, proving trainability
- Practical for transfer learning and model adaptation

---

## Technique 3: 4-Bit Block Quantization

### Theory

Block quantization achieves extreme compression by storing weights in 4-bit format with group-wise normalization:

1. **Grouping**: Divide weights into blocks (e.g., 16 values per group)
2. **Normalization**: Find max absolute value per group: `v̂ = max|vᵢ|`
3. **Quantization**: Map normalized values to 4-bit integers: `qᵢ ∈ {0, ..., 15}`
4. **Packing**: Store two 4-bit values per byte
5. **Overhead**: Store one float16 normalization factor per group

**Effective bits per parameter**: 4 + (16 bits / 16 values) = 5 bits

This explains why memory reduction is ~7x instead of 8x.

### Results

| Metric | Baseline | 4-Bit Quantized | Improvement |
|--------|----------|-----------------|-------------|
| Theoretical memory | 72.11 MB | 11.36 MB | **-84.2%** |
| Actual memory | 72.11 MB | 11.36 MB | **-84.2%** |
| Max difference | - | 0.1931 | - |
| Mean difference | - | 0.0308 | - |
| Trainable | Yes | No | - |

### Analysis

- Nearly 7x memory reduction (32 bits → 5 effective bits)
- Moderate accuracy degradation (mean diff ~3%)
- Not trainable (weights are frozen)
- Group size trades off compression vs accuracy
- Suitable for inference on edge devices with strict memory limits

---

## Technique 4: QLoRA (Quantized LoRA)

### Theory

QLoRA combines the strengths of quantization and low-rank adaptation:

- **Base model**: 4-bit quantized and frozen (extreme compression)
- **LoRA adapters**: float32 and trainable (learning capacity)

This hybrid approach enables fine-tuning of highly compressed models, making it practical to adapt billion-parameter models on consumer GPUs.

### Results

| Metric | Baseline | QLoRA | Improvement |
|--------|----------|-------|-------------|
| Theoretical memory | 72.11 MB | 15.86 MB | **-78.0%** |
| Actual memory | 72.11 MB | 15.86 MB | **-78.0%** |
| Trainable params | 18.90 M | 1.19 M | **-93.7%** |
| Backward memory | 72.11 MB | 14.65 MB | **-79.7%** |
| Max difference | - | 0.1975 | - |
| Mean difference | - | 0.0308 | - |
| Trainable | Yes | Yes | ✓ |
| Training accuracy | - | 100% | - |

### Analysis

- 78% memory reduction while maintaining trainability
- Accuracy matches 4-bit baseline (quantization dominates error)
- Successfully fine-tunes on training data
- Higher backward memory than LoRA (due to quantized base)
- Enables fine-tuning large models on limited hardware

---

## Comparative Analysis

### Memory Efficiency

| Technique | Memory (MB) | Reduction | Effective Bits/Param |
|-----------|-------------|-----------|---------------------|
| Baseline | 72.11 | - | 32 |
| Half Precision | 36.07 | 50.0% | 16 |
| LoRA | 40.57 | 43.7% | ~17 |
| 4-Bit Quantized | 11.36 | 84.2% | 5 |
| QLoRA | 15.86 | 78.0% | ~7 |

### Accuracy vs Memory Tradeoff

| Technique | Mean Difference | Memory Reduction | Trainable |
|-----------|-----------------|------------------|-----------|
| Half Precision | 0.0002 | 50.0% | ✗ |
| LoRA | 0.0002 | 43.7% | ✓ |
| 4-Bit Quantized | 0.0308 | 84.2% | ✗ |
| QLoRA | 0.0308 | 78.0% | ✓ |

### Use Case Recommendations

**Half Precision**
- ✓ Inference-only workloads
- ✓ Minimal accuracy loss acceptable
- ✗ Training required

**LoRA**
- ✓ Fine-tuning pretrained models
- ✓ Parameter-efficient transfer learning
- ✓ Limited GPU memory for training
- ✗ Need for maximum compression

**4-Bit Quantization**
- ✓ Edge device deployment
- ✓ Extreme memory constraints
- ✓ Moderate accuracy loss acceptable
- ✗ Fine-tuning required

**QLoRA**
- ✓ Fine-tuning compressed models
- ✓ Consumer GPU training
- ✓ Balance between memory and trainability
- ✓ Production fine-tuning pipelines

---

## Experimental Setup

All experiments used:
- **Model**: 6-block residual network with 18.90M parameters
- **Baseline**: float32 weights (72.11 MB)
- **Testing**: Forward pass accuracy and training capability
- **Training**: Overfitting to 1000 random samples (proof of trainability)

---

## Conclusion

Memory optimization for deep learning is essential for practical deployment. This project demonstrates that significant memory reductions (43-84%) are achievable with minimal accuracy loss. The choice of technique depends on deployment requirements:

- **Maximum accuracy**: Half precision or LoRA
- **Maximum compression**: 4-bit quantization
- **Balance**: QLoRA

These techniques are not academic exercises—they enable real-world applications like running 7B parameter models on consumer GPUs and deploying vision models on mobile devices.

---

## About

This project was completed as part of a graduate-level course in advanced deep learning. The implementations explored fundamental techniques used in production systems for memory-efficient deep learning.

**Academic Integrity Note**: Implementation details are not included in this repository to maintain academic integrity. This document presents only the conceptual approaches and experimental results.
