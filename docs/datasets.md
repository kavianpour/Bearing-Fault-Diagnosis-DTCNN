# Dataset — CWRU bearing dataset & task design

This document describes the data used in the paper, the fault classes, the wavelet selection, and how to obtain the data.

---

## 1. The CWRU bearing dataset

| Property | Detail |
|---|---|
| Name | Case Western Reserve University (CWRU) bearing dataset |
| Role | Standard benchmark for bearing fault diagnosis |
| Sensor | Drive-end accelerometer |
| Sample rate | 12 kHz |
| Load conditions | 0, 1, 2, 3 hp |
| Fault locations | Inner race, ball, outer race + normal |
| Fault diameters | 0.18 mm, 0.36 mm, 0.53 mm |
| Conditions | 10 total (nine faulty + one normal) |

---

## 2. Datasets and classes (Table I in the paper)

Each load condition forms a dataset (A, B, C, D), each with the same ten labels:

| Fault location | Inner race | Ball | Outer race | Normal |
|---|---|---|---|---|
| Labels | 0, 1, 2 | 3, 4, 5 | 6, 7, 8 | 9 |

| Dataset | Load (hp) | Train/class | Test/class |
|---|---|---|---|
| A | 0 | 250 | 50 |
| B | 1 | 250 | 50 |
| C | 2 | 250 | 50 |
| D | 3 | 250 | 50 |

Samples are increased with an **overlap technique** (window length 1024, step 350). The frequency–time characteristics of the data are then used as input to the DTCNN models.

---

## 3. Input representation

Each 1-D vibration segment is converted to a 2-D **time–frequency image** by **CWT**, resized to **224×224**, normalized to **[0, 1]**, and converted to **RGB** before entering the pretrained backbone.

---

## 4. Wavelet basis selection (Table II in the paper)

The wavelet basis function is chosen by the **Energy-to-Shannon entropy ratio** (higher = better):

| Wavelet | Ratio |
|---|---|
| **Morlet** | **13.143** |
| Gaussian | 8.085 |
| B-Spline | 7.772 |
| Harmonic | 2.125 |
| Shannon | 1.923 |

The Morlet wavelet has the highest ratio and is used for CWT.

---

## 5. Obtaining the data

The CWRU bearing dataset is publicly distributed by Case Western Reserve University and is widely mirrored. It is **not redistributed in this repository.** To reproduce:

1. Obtain the CWRU drive-end data at 12 kHz for loads 0–3 hp.
2. Build the ten classes (inner/ball/outer at three diameters + normal).
3. Segment with overlap (length 1024, step 350); 250 train / 50 test per class per load.
4. Apply CWT (Morlet) → resize 224×224 → normalize [0,1] → RGB.

> Always check and comply with the dataset's usage terms.

---

*This document is part of the documentation-only release; preprocessing scripts will accompany the code when released. See the [Roadmap](../README.md#roadmap).*
