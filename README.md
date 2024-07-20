# CEDS
This repository contains the code for **Confidence-Enhanced Semi-supervised Learning for Mediastinal Neoplasm Segmentation**.

## Introduction

Mediastinal tumors are a common type of thoracic disease, characterized by tumors located in the central region of the chest, specifically in the mediastinum. Due to their specific location, even non-cancerous mediastinal tumors can cause severe health issues if left untreated.

As an efficient, non-invasive imaging technique, CT imaging can provide clear images of the mediastinal region. Though convolutional neural networks (CNNs) have proven effective in medical imaging analysis, the segmentation of mediastinal neoplasms, which vary greatly in shape, size, and texture, presents a unique challenge due to the inherent local focus of convolution operations.

To address this limitation, we propose a confidence-enhanced semi-supervised learning framework for mediastinal neoplasm segmentation. Quantitative and qualitative analysis on a real-world dataset demonstrates that our model significantly improves the performance by leveraging unlabeled data.

## Model

Figure 1 shows the pipeline of the proposed model. 

![Model.png](https://github.com/fxiaotong432/CEDS/blob/main/Model.png)

First, a 3D cropped block from a CT scan is fed into both the teacher and student models. Then, the teacher model, utilizing the exponential moving average (EMA) of the student model’s weights, generates segmentation predictions and a corresponding confidence map through two distinct decoders. Finally, this confidence map guides the calculation of consistency loss, which is determined by the differences in high-confidence predictions between the teacher and student models.

## Requirements
The code is written in Python and requires the following packages: 

* Python                       3.8.15
* numpy                        1.23.5
* torch                        1.12.1+cu116
* tensorflow                   2.12.0
* keras                        2.12.0
* monai-weekly                 1.2.dev2323
* cachetools                   5.2.0
* matplotlib                   3.6.2
* opencv-python                4.6.0.66
* scikit-learn                 1.2.0
* pandas                       1.5.2
