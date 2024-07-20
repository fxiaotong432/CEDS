# CEDS
This repository contains the code for **Confidence-Enhanced Semi-supervised Learning for Mediastinal Neoplasm Segmentation**.

## Introduction

Mediastinal tumors are a common type of thoracic disease, characterized by tumors located in the central region of the chest, specifically in the mediastinum. Due to their specific location, even non-cancerous mediastinal tumors can cause severe health issues if left untreated.

As an efficient, non-invasive imaging technique, CT imaging can provide clear images of the mediastinal region. Though convolutional neural networks (CNNs) have proven effective in medical imaging analysis, the segmentation of mediastinal neoplasms, which vary greatly in shape, size, and texture, presents a unique challenge due to the inherent local focus of convolution operations.

To address this limitation, we propose a confidence-enhanced semi-supervised learning framework for mediastinal neoplasm segmentation. Quantitative and qualitative analysis on a real-world dataset demonstrates that our model significantly improves the performance by leveraging unlabeled data.

## Model

Figure 1 shows the pipeline of the proposed model. 

![Model.pdf](https://github.com/fxiaotong432/CEDS/blob/main/Model.pdf)
