# Face Recognition with Knowledge Distillation

![Face Recognition System](https://via.placeholder.com/800x400.png?text=Face+Recognition+System)

## Overview
This project implements a **Face Recognition System** using **Knowledge Distillation**, where a **Teacher model** transfers knowledge to a **Student model** to improve efficiency while maintaining accuracy. The system includes **training, evaluation, and an API for image-based registration and verification**.

## Dataset
The project uses the **Labeled Faces in the Wild (LFW)** dataset, which contains over **13,000 labeled images** from **5,749 individuals** collected from online sources. For training, **triplet samples (Anchor, Positive, Negative)** were formed, ensuring that Anchor and Positive images belong to the same person, while the Negative image is from a different individual.

To prevent data leakage:
- **90%** of the folders were used for training.
- **10%** were used for validation.
- A total of **27,000 triplets** were created for training and **3,000 triplets** for validation.

## Preprocessing & Data Augmentation
To improve model generalization, various **preprocessing** and **augmentation** techniques were applied:
- **Resizing** to a fixed dimension.
- **Cropping** to introduce pose variation.
- **Random horizontal flipping** for better invariance.
- **Random rotation** to handle angle variations.
- **Affine transformations** with slight distortions.
- **Brightness & contrast adjustments**.
- **Grayscale conversion** (randomly applied).
- **Normalization** to standardize pixel values.

For validation, only **center cropping, resizing, and normalization** were applied to maintain evaluation consistency.

## Methodology & Models

### Knowledge Distillation
A **Teacher-Student training framework** was used to compress a large, high-performing model into a smaller, efficient model. The goal was to **transfer knowledge** from a powerful **Teacher model** to a compact **Student model** while preserving accuracy.

### Teacher Model
- Based on **ResNet-101** with pre-trained **ImageNet weights**.
- The **first layers were frozen**, and only the **last 10 layers were fine-tuned**.
- The **Global Average Pooling** layer was removed.
- The model outputs a **128-dimensional embedding**.
- **Batch Normalization, Dropout, and L2 normalization** were applied for stability.

### Student Model
- **EfficientNet-B0** was used as the backbone.
- Only the **last two layers were trainable** (unfrozen) to improve learning.
- **Batch Normalization & Dropout** were applied for generalization.
- **L2 normalization** was used for feature stability.
- The final model is **lightweight and efficient**.

### Model Training & Optimization

#### **Teacher Model Training**
- **Optimizer:** Adam
- **Learning Rate:** 0.0003
- **Loss Function:** Triplet Loss (Margin = 1)
- The model was trained to ensure that the **distance between positive pairs is minimized**, while **negative pairs are separated**.

#### **Student Model Training**
- **Optimizer:** Adam
- **Learning Rate:** 0.0001
- **Loss Function:** Distillation Loss (Combination of Triplet Loss & MSE)
- The **Student model** learns from both:
  - The **ground truth triplet loss**.
  - The **MSE loss between the embeddings of the Teacher & Student models**.
- A **weighting parameter (Alpha)** controls the influence of the distillation loss.

## Model Evaluation
The **Student model** was evaluated using the validation set, with the following metrics:

| Metric      | Value  |
|------------|--------|
| **Accuracy** | 77.82% |
| **Precision** | 84.84% |
| **Recall** | 67.73% |
| **F1-Score** | 75.33% |

Despite being **significantly smaller**, the **Student model's accuracy was very close to the Teacher model's performance**.

## Additional Training on VGGFace2
The same methodology was applied to the **VGGFace2** dataset:
- **30,000 triplets** were used for training.
- **3,000 triplets** were used for validation.
- The **Teacher model achieved 77% accuracy**.
- The **Student model achieved 75% accuracy**, with similar performance to the LFW dataset.

## API for Face Registration & Verification
A **FastAPI-based API** is provided for **real-time face recognition**:
- **Register a new person** by uploading an image.
- **Verify an identity** by comparing a new
