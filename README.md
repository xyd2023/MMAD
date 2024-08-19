# A Unified Model for Multi-class Unsupervised Anomaly Detection with State Space Model

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Results](#results)
- [Authors](#authors)

## Introduction
This repository contains source code for MMAD implemented with PyTorch. MMAD is a unified model for multi-class anomaly detection tasks based on state-space models. With the proposed Dilated Mamba Fuser, MMAD stands out as a robust and identical-shortcut-resistant reconstruction model. Through the proposed Hybrid State Space (HSS) module, MMAD improves computational efficiency. In addition, it further reduces the computational workload and improves the spatial perception capability of the HSS module through group scanning and multiple scanning strategies. Extensive experiments on seven evaluation metrics demonstrate the effectiveness of our approach in achieving SoTA performance.

## Installation

## Dataset
#### MVTec-AD and VisA 

> **1、Download and prepare the original [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) datasets to any desired path. The original dataset format is as follows:**

```
path1
├── mvtec
    ├── bottle
        ├── train
            ├── good
                ├── 000.png
        ├── test
            ├── good
                ├── 000.png
            ├── anomaly1
                ├── 000.png
        ├── ground_truth
            ├── anomaly1
                ├── 000.png
```

```
path2
├── visa
    ├── candle
        ├── Data
            ├── Images
                ├── Anomaly
                    ├── 000.JPG
                ├── Normal
                    ├── 0000.JPG
            ├── Masks
                ├── Anomaly
                    ├── 000.png
    ├── split_csv
        ├── 1cls.csv
        ├── 1cls.xlsx
```

> **2、Standardize the MVTec-AD and VisA datasets to the same format and generate the corresponding .json files.**

- run **./dataset/make_dataset_new.py** to generate standardized datasets **./dataset/mvisa/data/visa** and **./dataset/mvisa/data/mvtec**
- run **./dataset/make_meta.py** to generate **./dataset/mvisa/data/meta_visa.json** and **./dataset/mvisa/data/meta_mvtec.json** (This step can be skipped since we have already generated them.)

The format of the standardized datasets is as follows:

```
./datasets/mvisa/data
├── visa
    ├── candle
        ├── train
            ├── good
                ├── visa_0000_000502.bmp
        ├── test
            ├── good
                ├── visa_0011_000934.bmp
            ├── anomaly
                ├── visa_000_001000.bmp
        ├── ground_truth
            ├── anomaly1
                ├── visa_000_001000.png
├── mvtec
    ├── bottle
        ├── train
            ├── good
                ├── mvtec_000000.bmp
        ├── test
            ├── good
                ├── mvtec_good_000272.bmp
            ├── anomaly
                ├── mvtec_broken_large_000209.bmp
        ├── ground_truth
            ├── anomaly
                ├── mvtec_broken_large_000209.png

```

## Results

## Authors
This project was developed by:

- **Xuesen Ma**
- **Yudong Xu**
- **Dacheng Li**
- **Wei Jia**
- **Yang Wang**
- **Meng Wang**




