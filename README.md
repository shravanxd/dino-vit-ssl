# Implementing and Evaluating DINO Self-Supervised Learning with ViT-Base Backbone

**Authors:**
- Shravan Khunti (ssk10036@nyu.edu)
- Harshit Bhargava (hb2976@nyu.edu)
- Mihir Upadhyay (mu2253@nyu.edu)

## Abstract
Self-supervised learning (SSL) has revolutionized computer vision by enabling the learning of rich visual representations without the need for extensive human annotations. In this work, we implement and evaluate DINO (Self-Distillation with No Labels), a simpler yet effective SSL method, using a Vision Transformer (ViT-Base) backbone. We train the model for 144 epochs on the provided training dataset using a multi-crop strategy with high-resolution global and low-resolution local crops. Our experiments demonstrate the effectiveness of the DINO framework, with the model learning robust features as evidenced by kNN and Linear Probing evaluation on standard benchmarks including CUB-200, MiniImageNet, and SUN397. We analyze the training stability and the impact of hyperparameter choices, specifically focusing on a customized configuration optimized for high-performance GPUs.

## 1. Introduction
The field of computer vision has traditionally relied on large-scale supervised learning, where models are trained on massive datasets of labeled images. While effective, this approach is limited by the cost and scalability of data annotation. Self-Supervised Learning (SSL) addresses this bottleneck by leveraging the intrinsic structure of data to learn representations. Recent methods like Contrastive Learning (e.g., SimCLR [2], MoCo [3, 4]) have shown great promise, but they often require large batch sizes or complex negative sample mining.

DINO [1] (Self-Distillation with No Labels) offers a substantial simplification by replacing the contrastive objective with a self-distillation framework. It trains a student network to match the output of a teacher network, which is constructed as an exponential moving average (EMA) of the student. This approach notably eliminates the need for negative pairs.

In this project, we explore the implementation of DINO using a Vision Transformer (ViT) architecture. Specifically, we utilize a ViT-Base [5] backbone trained with a highly optimized configuration. Our contribution is twofold: first, we provide a detailed implementation of the DINO training loop with a specific focus on multi-crop augmentation and centering/sharpening mechanisms. Second, we evaluate the learned representations on CUB-200, MiniImageNet, and SUN397, analyzing the impact of our specific hyperparameter choices, such as a patch size of 12 and an aggressive batch size strategy.

## 2. Methodology

### 2.1. Architecture
We employ a Vision Transformer (ViT-Base) [5] backbone for feature extraction. Unlike standard convolutional networks that process pixels locally, the ViT treats the image as a sequence of patches, enabling the modeling of long-range dependencies via self-attention.

1. **Patch Embedding:** The input image `x ∈ R^(96×96×3)` is divided into non-overlapping patches of size `12 × 12`. These patches are linearly projected into D-dimensional embedding vectors (where `D = 768` for ViT-Base) and augmented with learnable positional embeddings to retain spatial information.
2. **Transformer Encoder:** The sequence of patch embeddings, prepended with a learnable `[CLS]` token, is processed by a stack of Transformer blocks. Each block consists of Multi-Head Self-Attention (MSA) and a Feed-Forward Network (FFN), with LayerNorm (LN) and residual connections applied before and after each sub-layer respectively.
3. **Projection Head:** The final representation of the `[CLS]` token serves as the global image feature. This feature is fed into a non-linear projection head `h`, constructed as a 3-layer Multi-Layer Perceptron (MLP). It comprises hidden layers of dimension 2048, a bottleneck layer of dimension 256, and a final weight-normalized output layer of dimension 4096. No ReLU activation is applied to the last bottleneck layer, ensuring the full expressivity of the features before the final softmax.

### 2.2. DINO Framework
The DINO [1] framework simplifies self-supervised learning by formulating it as a knowledge distillation problem without the need for negative pairs (contrastive learning).

**Teacher-Student Distillation**
The framework consists of two networks with identical architectures: a student `g_θs` and a teacher `g_θt`. The student is trained to match the probability distribution of the teacher. Crucially, the teacher is not trained via gradient descent. Instead, its parameters `θt` are updated as an Exponential Moving Average (EMA) of the student parameters `θs`:

`θt <- λθt + (1 - λ)θs` (1)

where `λ` follows a cosine schedule from 0.996 to 1.0. This slowly evolving teacher serves as a stable "mean teacher," providing consistent targets for the student and enabling the ensemble effect that boosts performance.

**Collapse Avoidance Mechanisms**
A common failure mode in non-contrastive SSL is "representation collapse," where the model outputs a uniform or static distribution for all inputs. DINO prevents this through a synergy of two operations applied to the teacher's output:
*   **Centering:** We maintain a running average `c` of the teacher's output mean. This center `c` is subtracted from the teacher's logits: `Pt(x) ∝ exp((g_θt(x) - c)/τt)`. Centering prevents the model from predicting a single dominant class (mode collapse).
*   **Sharpening:** The teacher uses a low temperature `τt` (warmed up to 0.07) compared to the student `τs` (0.1). This artificial sharpening forces the student to produce confident (low-entropy) predictions, preventing the trivial solution of a uniform distribution.

### 2.3. Data Curation and Preprocessing
To improve the robustness and generalization of our model, we significantly expanded the training dataset beyond the initial 500k subset. We incorporated additional data from:
*   **CC12M:** A large-scale conceptual captioning dataset. We utilized a subset of 600,000 images to augment our training distribution.
*   **iNat:** The iNaturalist dataset, providing fine-grained species classification images.

**Deduplication Strategy**
With the aggregation of multiple datasets, duplicates became a concern. We implemented a perceptual hashing (pHash) based deduplication pipeline to curate the final training set.
1. **Hashing:** We computed a 64-bit perceptual hash for every image in the expanded dataset. pHash is robust to minor modifications like resizing or compression artifacts.
2. **Indexing:** These hashes were stored in a searchable index.
3. **Filtering:** For every new candidate image, we queried the index. If the Hamming distance between the candidate's hash and any existing hash was below a strict threshold (indicating a near-duplicate), the image was discarded. This ensured that our effective training set consisted of unique, high-value samples, preventing the model from memorizing redundant data.

### 2.4. Data Augmentation
Key to DINO's success is its aggressive data augmentation pipeline, which forces the model to learn invariant representations. The specific parameters for our implementation, aligned with standard SSL practices, are rigorously defined as follows:
*   **Multi-Crop Strategy:** We generate two large global crops (96 × 96) with a scale range of (0.4, 1.0) and six small local crops (48 × 48) with a scale range of (0.05, 0.4). The local crops encourage the model to attend to fine-grained details while maintaining semantic consistency with the global context.
*   **Color Jitter:** A random probabilistic transformation applied with `p = 0.8`. The jitter intensity is set to: brightness 0.4, contrast 0.4, saturation 0.2, and hue 0.1. This variation prevents the model from relying on trivial color statistics (histograms) for discrimination.
*   **Gaussian Blur:** To learn texture invariance, we apply a Gaussian blur using a random kernel size (specifically 23 pixels for our 96 × 96 input) with `σ ∈ [0.1, 2.0]`. Crucially, this is applied asymmetrically: the first global view is always blurred (`p = 1.0`), while the second global view (`p = 0.1`) and local views (`p = 0.5`) are blurred stochastically.
*   **Solarization:** As a strong non-linear augmentation, we apply solarization (pixel inversion above threshold 128) exclusively to the second global view with `p = 0.2`. This transformation has been shown to be particularly effective in decoupling shape features from texture and intensity.

### 2.5. Loss Function
Our training objective is a composite of three distinct loss terms. While the core DINO loss drives self-distillation, we explicitly incorporate variance and covariance regularization terms (inspired by VICReg) to actively prevent representation collapse - a constant risk in non-contrastive learning.

#### 2.5.1. Global-to-Global Distillation (Ldino)
The primary DINO loss aligns the student's global view representation with the teacher's. It minimizes the cross-entropy between the teacher's centered, sharpened output distribution and the student's output:

`Ldino = - Σ pt(g1) log ps(g2)` (2)

where `pt` differs from `ps` by using a moving average center (to prevent dominant mode collapse) and a sharper temperature `τt`.

#### 2.5.2. Local-to-Global Alignment (LL2G)
To ensure fine-grained features capture global semantics, we enforce an alignment between local crops (seen only by the student) and global crops (seen by the teacher). The student's prediction from a small 48 × 48 local crop must match the teacher's prediction from the full 96 × 96 global view:

`LL2G = - Σ pt(g) log ps(l)` (3)

#### 2.5.3. Feature Regularization (Lreg)
We augment the distillation objective with explicit regularization on the batch-wise feature statistics. Let `Z` be the batch of feature representations.
*   **Variance Loss:** Maintains activity in all feature dimensions. We penalize the model if the standard deviation of valid features drops below a threshold `γ = 1`:

    `Lvar = (1/d) Σ max(0, γ - S(Z^j))` (4)

    where `S(Z^j)` is the standard deviation of the j-th dimension.
*   **Covariance Loss:** Decorrelates feature dimensions to maximize information content. We penalize the off-diagonal terms of the covariance matrix `C(Z)`:

    `Lcov = 1/(d(d - 1)) Σ C(Z)^2_{i,j}` (5)

The total loss is a weighted sum:

`L = Ldino + λ1*LL2G + λ2*Lreg` (6)

## 3. Experiments

### 3.1. Implementation Details
We utilized a ViT-Base backbone (patch size 12) pretrained on our curated dataset. The implementation leveraged 8 × NVIDIA H200 GPUs. To ensure reproducibility, we explicitly list the hyperparameters used for the pretraining stage in Table 1.

| Parameter | Value |
| --- | --- |
| **Architecture** | |
| Backbone | ViT-Base (patch 12 × 12) |
| Projection Head | 3-layer MLP (2048/2048/256) -> 4096 |
| Student Temp `τs` | 0.1 |
| Teacher Temp `τt` | 0.04 -> 0.07 (warmup 30 ep) |
| **Optimization** | |
| Optimizer | AdamW |
| Base Learning Rate | 0.001 (scaled linearly) |
| Weight Decay | 0.04 -> 0.4 (cosine schedule) |
| Betas `(β1, β2)` | (0.9, 0.999) |
| Batch Size | 2040 (effective) |
| Epochs | 144 (linear warmup 25 ep) |
| **Augmentation & Regularization** | |
| Global Crops | 2 × 96 × 96 (scale 0.4-1.0) |
| Local Crops | 6 × 48 × 48 (scale 0.05-0.4) |
| Center Momentum | 0.9 |
| EMA Momentum | 0.996 -> 1.0 |

*Table 1. Detailed DINO Pre-training Hyperparameters.*

The model was trained using the AdamW optimizer with a base learning rate scaled linearly with the batch size (`lr = 0.001 × BatchSize/256`). The weight decay followed a cosine schedule from 0.04 to 0.4. We used automatic mixed precision (AMP) to maximize throughput.

### 3.2. Evaluation Protocols
To thoroughly assess the quality of the learned representations, we employ two distinct evaluation protocols: non-parametric k-NN classification and parametric linear probing. We meticulously tune hyperparameters for both methods to ensure a fair evaluation.

#### 3.2.1. k-Nearest Neighbors (k-NN)
We implement a weighted k-NN classifier (`k = 20`) that predicts labels based on feature similarity. We used the default configuration found to be robust: utilizing `[CLS]` token features and `τ = 0.07`.

#### 3.2.2. Linear Probing
To evaluate the quality of the frozen representations, we train a linear classifier (single fully-connected layer) on top of the fixed ViT features. We rigorously follow the standard evaluation protocol, training the classifier for 30 epochs on the training split and selecting the best checkpoint based on validation accuracy.

To ensure optimal performance, we implemented an automated hyperparameter sweep for each dataset independently. The grid search covered:
*   **Optimizers:** SGD (with momentum 0.9), RMSprop, and Adam.
*   **Learning Rates:** A broad range of values including `{0.1, 0.05, 0.01}` for SGD/RMSprop and `{1e-3, 5e-4}` for Adam.
*   **Regularization:** Weight decay of `0.0` and `1e-4`.

Features were cached prior to training to accelerate this sweep. The best configuration was typically SGD with a high learning rate (0.1) for CUB-200, while Adam (1e-3) performed better on the larger-scale MiniImageNet and SUN397 datasets. This underscores the necessity of dataset-specific tuning when evaluating transfer performance.

### 3.3. Datasets
We utilize custom splits generated specifically for this evaluation. While Training and Validation sets generally followed a 70/15 split, the Test sets (used for the Kaggle leaderboard) had specific sizes:
*   **CUB-200-2011:** Contains 11,788 images of 200 bird species. Images are resized to 96 × 96 pixels. The final evaluation was performed on a Test set of 1,829 images.
*   **MiniImageNet:** Consists of 60,000 images across 100 classes. Originally 84 × 84, upscaled to 96 × 96 pixels. The Test set for evaluation consisted of 5,760 images.
*   **SUN397:** A scene recognition dataset resized to 96 × 96 pixels. The evaluation was conducted on a curated Test set of 2,978 images.

### 3.4. Quantitative Results
Table 2 summarizes the performance on these three benchmark datasets. We report the Top-1 classification accuracy for both k-NN (k = 20) and Linear Probing on the Validation Set, as well as the final Test Set (Kaggle) accuracy obtained from the linear classifier. As observed, training a parametric linear head significantly outperforms non-parametric k-NN. The test set performance tracks the validation performance closely, confirming that the model has specialized effectively without overfitting.

| Dataset | k-NN (Val) | Linear (Val) | Test (Kaggle) | Gain (Val) |
| --- | --- | --- | --- | --- |
| CUB-200 | 19.56% | 33.12% | 30.60% | +13.56% |
| MiniImageNet | 69.08% | 76.48% | 75.10% | +7.40% |
| SUN397 | 39.87% | 47.01% | 45.53% | +7.14% |

*Table 2. Evaluation Results: Comparison of Validation (Val) performance between k-NN and Linear Probing, alongside final Test Set accuracy.*

### 3.5. Training Stability
Training Vision Transformers with self-supervision in mixed precision (FP16) commonly leads to exploding gradients and numerical instability (NaN losses). We mitigated this by rigorously tuning the gradient scaler:
*   **Conservative Initialization:** We initialized the GradScaler with a scale of `2^12 (4096)`, significantly lower than the default `2^16`, preventing early overflow in the attention mechanism.
*   **Slow Growth:** The scale growth factor was reduced to 1.5 (default 2.0) with a longer interval of 2000 steps, ensuring stable convergence before increasing precision.
*   **Hard Clipping:** We clipped gradient norms to 3.0 prior to optimizer steps. If NaNs were detected, the scaler actively discarded the batch and reduced the scale factor by 4x, allowing the model to recover automatically.

## 4. Conclusion
We successfully implemented and trained a DINO model with a ViT-Base backbone, demonstrating the efficacy of self-supervised learning with transformers on a curated dataset of over 1.2 million images. Our optimized configuration, leveraging multi-crop augmentation and variance regularization, achieved robust transfer performance across fine-grained (CUB-200), few-shot (MiniImageNet), and scene (SUN397) understanding tasks.

Future work often focuses on scaling; however, having already scaled the data, the next logical step is Model Scaling: training larger backbones (e.g., ViT-Large or ViT-Huge), as demonstrated by DINOv2 [6], to fully capitalize on the curated data. Additionally, evaluating the learned representations on Dense Prediction Tasks such as semantic segmentation and object detection would further validate the emergent properties of the self-attention mechanism observed in DINO.

## References
[1] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers, 2021.
[2] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597-1607. PMLR, 2020.
[3] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020.
[4] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9640-9649, 2021.
[5] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
[6] Maxime Oquab, Timothee Darcet, Theo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Haziza Daniel, Babiloni Adina, Cideron Manning, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
