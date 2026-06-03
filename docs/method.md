# Method — Deep Transfer CNN (DTCNN)

This document describes DTCNN in full: transfer learning, the CWT time–frequency input, the pipeline, the classifiers, and the training configuration reported in the paper.

---

## 1. Overview

DTCNN diagnoses rolling-bearing faults from vibration signals using **transfer learning**. The pipeline:

1. Convert each 1-D vibration signal to a 2-D time–frequency RGB image via **CWT**.
2. Extract high-level features with a **frozen ImageNet-pretrained CNN** (DenseNet201, ResNet152V2, or MobileNetV2).
3. Reduce dimensions with **global average pooling (GAP)**.
4. Classify with an ML or DL classifier.

---

## 2. Transfer learning

Transfer learning reuses knowledge from a model trained on a source domain for a new but related target task. A domain is $D = \{\chi, P(x)\}$ (feature space + marginal distribution) and a task $T = \{Y, F(x)\}$ where $F(x) = P(Y\mid X)$.

**Parameter transfer** splits model parameters into a common part and a specific part:

$$\omega_s = \omega_{com} + \acute{\omega}_s, \qquad \omega_t = \omega_{com} + \acute{\omega}_t$$

where $\omega_{com}$ is shared between source and target and $\acute{\omega}$ are specific. Here, the source model (trained on ImageNet with abundant data) provides parameters that transfer to the data-scarce bearing target. The last layer of the pretrained network is removed and the remaining layers are **frozen**, so the backbone is a fixed feature extractor.

---

## 3. Continuous Wavelet Transform (CWT)

CWT is a time–frequency analysis that (a) extracts non-stationary, nonlinear characteristics from the vibration signal and (b) provides time and frequency information simultaneously, converting a 1-D signal into a 2-D representation suitable for image CNNs:

$$C_\psi = \frac{1}{\sqrt{a}} \int x(t)\,\psi^*\!\left(\frac{t-b}{a}\right) dt, \qquad a, b > 0$$

where $x(t)$ is the input signal, $\psi$ the wavelet function, $\psi^*$ its complex conjugate, and $a, b$ the scale and shift parameters. The **wavelet basis function (WBF)** is chosen using the **Energy-to-Shannon entropy ratio** — a higher ratio is a better WBF. On CWRU, the Morlet wavelet has the highest ratio (see [`datasets.md`](datasets.md)).

---

## 4. Pipeline details

1. CWT image is **resized to (224, 224)**, normalized to **[0, 1]**, converted to **RGB**.
2. Fed to a pretrained backbone — **DenseNet201** (201 layers), **ResNet152V2** (152), **MobileNetV2** (53) — with the last layer removed and other layers frozen.
3. Extracted features pass to a **GAP** layer, which cuts the parameter count and spatial dimensions and limits overfitting.
4. Features go to classifiers:
   - **DL head:** three FC layers (1024, 512, 10 neurons) + SoftMax — these layers are trained.
   - **ML classifiers:** SVM, LR, RF, KNN, DT — the pretrained model is a feature extractor whose output is the classifier input.

---

## 5. Training setup (implementation details)

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Categorical cross-entropy |
| Hardware | Tesla T4 GPU |
| Framework | TensorFlow v2.3 |
| Repetitions | 10 runs |
| ML hyperparameters | Selected by cross-validation |

The proposed bearing-fault-diagnosis models (highest accuracy) are **DenseNet201 + RF**, **ResNet152V2 + FC**, and **MobileNetV2 + LR**. Feature separability across layers is illustrated with **t-SNE**, which reduces high-dimensional features to 2-D by minimizing the Kullback–Leibler divergence between high- and low-dimensional joint probabilities; features become increasingly separable from the input layer through to the FC layers.

---

*Equations and tables are transcribed from the published paper for documentation purposes. Figures remain © 2021 IEEE — see [LICENSE](../LICENSE).*
