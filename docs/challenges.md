# Challenges — why deep bearing-fault diagnosis with limited data is hard

This document expands on the real-world problems that motivate DTCNN. Each section explains why the problem arises, why it breaks conventional models, and how the proposed method addresses it.

---

## 1. Labeled fault data is scarce and expensive

Changing a bearing from healthy to faulty and gathering faulty data is time-intensive and costly, so it is often not viable to collect substantial labeled fault data.

**Why it breaks models.** Deep CNNs with many layers need large datasets to avoid overfitting. With little labeled fault data, fault-diagnosis CNNs have historically been kept shallow (≈ five hidden layers), which lowers generalization and accuracy compared with the deep architectures used in computer vision.

**How DTCNN responds.** It transfers ImageNet-pretrained networks (DenseNet201, ResNet152V2, MobileNetV2). Their parameters already encode powerful features, so deep feature extraction works without massive labeled fault data.

---

## 2. Training deep CNNs from scratch is expensive

Advanced CNNs with hundreds of layers are designed for large-scale tasks like ImageNet and take a long time to train.

**Why it matters.** Beyond accuracy, fault diagnosis must be fast and early; long training is impractical in industry.

**How DTCNN responds.** The pretrained layers are **frozen**, so only a light classifier is trained. This dramatically reduces training time, which the results demonstrate directly.

---

## 3. Hand-crafted features need expertise

Classical ML (SVM, KNN, DT, RF, LR) requires extracting appropriate features from the input, which demands human proficiency and limits generalization. ML algorithms also struggle to model complex nonlinear relationships.

**How DTCNN responds.** The pretrained CNN provides abstract, high-level features automatically; these are far more informative than raw input for the downstream classifiers and remove the manual feature-engineering bottleneck.

---

## 4. A 1-D signal does not fit a 2-D image CNN

Famous CNN architectures expect 2-D image input, but bearing vibration is a 1-D time series that, in raw form, lacks joint time–frequency information.

**How DTCNN responds.** **CWT** converts the 1-D signal into a 2-D time–frequency RGB image that carries both time and frequency information at once — the correct input shape and a richer representation than raw time data.

---

## 5. Picking the right wavelet basis

The quality of the time–frequency features depends on the wavelet basis function (WBF); a poor choice yields weak features.

**How DTCNN responds.** The WBF is selected quantitatively via the **Energy-to-Shannon entropy ratio** (a higher ratio is better). On CWRU, the Morlet wavelet has the highest ratio and is used. See [`datasets.md`](datasets.md).

---

*This document is part of the documentation-only release; the source code is not yet public. See the [Roadmap](../README.md#roadmap).*
