# AI Storytelling: Oxford Flowers 102 Image Classification with Transfer Learning

## 📌 Project Overview
This project applies **Artificial Intelligence (AI)** techniques to the **Oxford Flowers 102** dataset, which contains 8,189 high-resolution images across 102 flower categories.  

We leverage **transfer learning** using pre-trained convolutional neural networks (CNNs) — **ResNet50, VGG16, and MobileNetV2** — to classify flower images efficiently. This approach reduces training time and improves accuracy by using features learned from large-scale datasets like ImageNet.

---

## 📊 Key Insights from Analysis

**Dataset:**
- 8,189 images, 102 classes
- Train: ~1,020 | Validation: ~1,020 | Test: ~6,149
- Images resized to 224×224 for CNN input

**Preprocessing:**
- Model-specific preprocessing applied (e.g., `tf.keras.applications.resnet50.preprocess_input`)
- Labels one-hot encoded
- Dataset batched, shuffled, and prefetched for efficient training

**Model Comparison:**
| Model         | Accuracy (Test Set) |
|---------------|------------------|
| ResNet50      | Highest (~85–86%) |
| MobileNetV2   | Slightly lower, lightweight |
| VGG16         | Lowest (~75–78%) |

- **ResNet50:** Deep residual architecture effectively distinguishes subtle differences among 102 flower classes.  
- **MobileNetV2:** Lightweight, faster inference, suitable for deployment on mobile devices.  
- **VGG16:** Older architecture, fewer feature extraction capabilities, performs worst.

**Visualization & Evaluation:**
- Training vs validation accuracy and loss curves
- Confusion matrix to identify misclassifications
- Sample predictions show correct vs incorrect classifications

**Takeaways:**
- Transfer learning significantly reduces training time and improves performance
- Preprocessing (resizing, normalization, one-hot encoding) is critical for stable training
- Visual evaluation highlights strengths and weaknesses beyond accuracy metrics
- Misclassifications mostly occur between visually similar flowers (e.g., daisy 🌼 vs sunflower 🌻)

---

## 📂 Repository Contents
- `Aadya_Patel_OxfordFlowers_TransferLearning.ipynb` → Full workflow:
  - Dataset loading and exploration
  - Data preprocessing
  - Model building (ResNet50, VGG16, MobileNetV2)
  - Training with EarlyStopping and ModelCheckpoint
  - Evaluation and visualization
- Training history plots (accuracy/loss)
- Confusion matrices
- Sample prediction visualizations
- Model comparison summary

---

## 🔮 Next Steps / Future Improvements
- **Data Augmentation:** Apply flips, rotations, and shifts to improve generalization  
- **Fine-tuning:** Unfreeze top layers of pre-trained models for higher accuracy  
- **Hyperparameter Tuning:** Adjust learning rate, batch size, number of epochs  
- **Explainable AI (XAI):** Use Grad-CAM or other techniques to visualize what models focus on  

---

## ⚡ References
- [Oxford Flowers 102 Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [TensorFlow Datasets – Oxford Flowers 102](https://www.tensorflow.org/datasets/catalog/oxford_flowers102)
- [Keras Applications – Pre-trained Models](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
