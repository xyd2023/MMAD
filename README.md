# A Unified Model for Multi-class Unsupervised Anomaly Detection with State Space Model

## Table of Contents
- [Abstract](#abstract)
- [Results](#results)
- [Authors](#authors)

## Introduction
Unsupervised anomaly detection aims to identify data instances that significantly deviate from the norm without the need for labeled training data. Traditional approaches typically utilize a one-class-one-model strategy for each category, which becomes memory-intensive and impractical as the number of categories increases. To address this limitation, multi-class anomaly detection has emerged as a more feasible alternative, focusing on the development of a unified model capable of detecting anomalies across various categories. Most existing methods rely on Convolutional Neural Networks (CNNs) and transformer architectures, both demonstrating robust performance. However, CNNs often struggle to capture long-range dependencies, while transformers are hindered by quadratic computational complexity. Recently, state space models have garnered significant attention due to their superior long-range modeling capabilities coupled with linear computational complexity. In this paper, we explore the application of state space models in anomaly detection and propose a unified model for multi-class anomaly detection. Our model comprises a pre-trained encoder, a multi-scale feature fusion module, and a decoder. The proposed Dilated Mamba Fuser (DMF) uses dilated convolutions with different dilation rates to prevent the long-standing ``identical shortcut'' problem in reconstruction-based methods and combines the plain Mamba block to increase the receptive field. The proposed Hybrid Mamba Decoder (HMD) features multiple Hybrid State Space (HSS) modules stacked at different scales. These HSS modules integrate state space models and convolutional neural networks to efficiently process regions of varying complexity. Additionally, the Grouped State Space (GSS) blocks within the HSS modules employ a group scanning strategy to enhance the quality-complexity trade-off. Comprehensive experiments on diverse benchmark datasets demonstrate the superior performance of our method. For implementation details and access to the code, please contact the authors via email.

## Results

### Comparison with State-of-the-Art Methods on Different Anomaly Detection Datasets

| Dataset              | Method                                                        |               Image-level                      | Pixel-level |                  |                  |                  |                  |      |
|----------------------|---------------------------------------------------------------|----------------|------------|------------------|-------------|------------------|------------------|------------------|------------------|------|
|                      |                                                               | AU-ROC         | AP         | F1-max           | AU-ROC      | AP               | F1-max           | AU-PRO           | mAD              |      |
| **MVTec-AD**         | RD4AD                                                         | 94.6           | 96.5       | 95.2             | 96.1        | 48.6             | 53.8             | 91.1             | 82.3             |      |
|                      | Simplenet                                                     | 95.3           | 98.4       | 95.8             | 96.9        | 45.9             | 49.7             | 86.5             | 81.2             |      |
|                      | DeSTSeg                                                       | 89.2           | 95.5       | 91.6             | 93.1        | 54.3             | 50.9             | 64.8             | 77.1             |      |
|                      | UniAD                                                         | 96.5           | 98.8       | 96.2             | 96.8        | 43.4             | 49.5             | 90.7             | 81.7             |      |
|                      | DiAD                                                          | 97.2           | 99.0       | 96.5             | 96.8        | 52.6             | 55.5             | 90.7             | 84.0             |      |
|                      | **MMAD (Ours)**                                               | **97.4**       | **99.1**   | **97.0**         | **97.2**    | **54.4**         | **57.5**         | **93.2**         | **85.1**         |      |
| **VisA**             | RD4AD                                                         | 92.4           | 92.4       | 89.6             | 98.1        | 38.0             | 42.6             | 91.8             | 77.8             |      |
|                      | Simplenet                                                     | 87.2           | 87.0       | 81.8             | 96.8        | 34.7             | 37.8             | 81.4             | 72.4             |      |
|                      | DeSTSeg                                                       | 88.9           | 89.0       | 85.2             | 96.1        | 39.6             | 43.4             | 67.4             | 72.8             |      |
|                      | UniAD                                                         | 88.8           | 90.8       | 85.8             | 98.3        | 33.7             | 39.0             | 85.5             | 74.6             |      |
|                      | DiAD                                                          | 86.8           | 88.3       | 85.1             | 96.0        | 26.1             | 33.0             | 76.2             | 70.1             |      |
|                      | **MMAD (Ours)**                                               | **93.1**       | **92.9**   | **89.1**         | **98.4**    | **38.0**         | **42.4**         | **91.0**         | **77.8**         |      |


## Authors
This project was developed by:

- **Xuesen Ma**
- **Yudong Xu**
- **Dacheng Li**
- **Wei Jia**
- **Yang Wang**
- **Meng Wang**




