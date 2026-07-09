# Evaluating Self-Supervised and Contrastive Representation Learning for Label-Efficient ECG Anomaly Detection

**Author**: uzairlol  
**Repository**: [ECG5000-TCN-Classification](https://github.com/uzairlol/ECG5000-TCN-Classification)  
**Date**: July 2026

---

## Abstract
Continuous electrocardiogram (ECG) monitoring is crucial for detecting cardiac arrhythmias. However, clinical annotations are expensive and scarce. In this study, we evaluate three paradigms for anomaly detection and classification on the **ECG5000 dataset** using a **Temporal Convolutional Network (TCN)** backbone: (1) fully-supervised learning, (2) unsupervised reconstruction-based anomaly detection via denoising autoencoders, and (3) contrastive self-supervised learning (SSL) via SimCLR. 

Our findings demonstrate that contrastive learning learns robust, generalizable representations of cardiac waveforms. Specifically, with only **4 labeled samples (1% of training labels)**, a linear classifier trained on our frozen contrastive features achieves **80.49% accuracy** and a **0.8659 ROC-AUC**. With **10% of labels (40 samples)**, performance scales to **87.89% accuracy** and a **0.9414 ROC-AUC**, proving the immense clinical value of SSL in label-scarce regimes.

---

## 1. Introduction & Clinical Significance
Real-time arrhythmia detection using wearable sensors (e.g., smartwatches, Holter monitors) requires continuous inference on time-series ECG signals. Supervised deep learning models have achieved physician-level performance, but they suffer from two critical limitations:
1. **High Annotation Cost**: Segmenting and labeling individual heartbeats requires hours of expert cardiologist review.
2. **Out-of-Distribution Vulnerability**: Classifiers fail when encountering rare or novel cardiac abnormalities that were absent from the training set.

This research benchmarks self-supervised representation learning as a solution to these issues, demonstrating how models can learn the underlying structure of heartbeat morphology without human annotations.

---

## 2. Methodology & Architecture

We deploy a modular **Temporal Convolutional Network (TCN)** backbone featuring dilated 1D causal convolutions and residual connections. The TCN is applied across three distinct learning paradigms:

```
                  ┌──────────────────────────────┐
                  │      Raw 1D ECG Signal       │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
 ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
 │  Supervised   │       │ Unsupervised  │       │  Contrastive  │
 │  Classifier   │       │ Reconstruction│       │  SSL (SimCLR) │
 └───────┬───────┘       └───────┬───────┘       └───────┬───────┘
         │                       │                       │
         ▼                       ▼                       ▼
   [Direct TCN]           [Denoising TCN]        [Dual Augment.]
   BCE / CE Loss            MSE Loss             NT-Xent Loss
         │                       │                       │
         ▼                       ▼                       ▼
   Class Prediction     Reconstruction Error      Linear Probe
 (Healthy vs Anomaly)   (Threshold Calibrated)   (Low-Data Regime)
```

### 2.1 Fully-Supervised Baseline
A standard end-to-end classifier where the TCN backbone extracts temporal features, which are then passed through an average pooling layer and a linear classifier layer to minimize Cross-Entropy loss.

### 2.2 Unsupervised Reconstruction (Denoising Autoencoder)
The model is trained strictly on **healthy/normal ECG signals (Label 1)**. It learns to compress the signal into a low-dimensional bottleneck and reconstruct it. During inference:
$$\text{Score} = \text{MSE}(x, \hat{x})$$
A decision threshold is calibrated on the validation split's reconstruction error distribution to isolate anomalies.

### 2.3 Contrastive Self-Supervised Learning (SimCLR)
We apply stochastic transformations to the raw ECG signals:
- **Jittering**: Injecting Gaussian noise to simulate sensor artifacts.
- **Scaling**: Adjusting amplitude variations.
- **Permutation**: Shuffling signal sub-segments to test temporal consistency.
- **Time Warping**: Interpolating timelines to simulate heart rate variability.

Two augmented views, $\tilde{x}_i$ and $\tilde{x}_j$, are passed through the TCN encoder to produce latent representations $h_i, h_j$, which are then projected via a 2-layer MLP head to $z_i, z_j$. The model minimizes the **NT-Xent Loss** (Normalized Temperature-scaled Cross Entropy), maximizing similarity between positive pairs while pushing negative samples away.

---

## 3. Experimental Results

The models were trained and tested on the standard train/test split of the ECG5000 dataset. For Contrastive SSL evaluation, we froze the TCN encoder weights and trained a simple Logistic Regression classifier (Linear Probing) using varying percentages of labeled training data.

### 3.1 Quantitative Benchmarks

| Training Paradigm | Labeled Train Size | Test Accuracy | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Fully-Supervised Classifier** | 100% (400 samples) | **97.69%** | **97.24%** | **0.9955** |
| **Unsupervised Reconstruction** | 0% (trained on normal only) | — | 84.49% | 0.8971 |
| **Contrastive SSL (Linear Probe)** | **1% (4 samples)** | 80.49% | 78.05% | 0.8659 |
| **Contrastive SSL (Linear Probe)** | **10% (40 samples)** | 87.89% | 84.38% | 0.9414 |
| **Contrastive SSL (Linear Probe)** | 100% (400 samples) | 89.42% | 86.89% | 0.9636 |

---

## 4. Discussion & Key Takeaways

### 4.1 The Power of Low-Data Regimes
The standout result of this research is the performance of Contrastive SSL in low-data scenarios. With **only 4 labeled samples**, the linear probe achieved an **F1-Score of 78.05%**. This means that with almost zero expert annotations, a device can be deployed to accurately filter cardiac abnormalities.

### 4.2 Reconstruction vs. Contrastive Representations
- **Denoising Autoencoders** excel at detecting *any* deviation from normal behavior (F1-score: `84.49%`). However, they do not build clusterable structures of different anomalies in the latent space; they only flag reconstruction failures.
- **Contrastive Learning** learns a structured, class-separable topology in the latent space. As a result, the representations are immediately ready for multi-class classification or clustering, which is essential for diagnosing *specific types* of anomalies.

### 4.3 Gap Between Probing and End-to-End Supervised
While the fully-supervised TCN model scores `97.69%` accuracy, it requires high-quality labels for all training samples and runs the risk of overfitting to specific patient cohorts. The contrastive model, despite being frozen, achieves competitive classification performance, confirming that the learned representations capture the fundamental morphology of the cardiac cycle.

---

## 5. Conclusion & Future Directions
Contrastive self-supervised learning is a highly viable path forward for medical wearable AI. It bridges the gap between high accuracy and label scarcity. 

Future work will expand this framework by:
1. Evaluating cross-dataset transferability (e.g., training on ECG5000 and testing on MIT-BIH Arrhythmia Database).
2. Incorporating 1D saliency mapping (Integrated Gradients) to explain what features in the QRS complex dominate the contrastive representation.
