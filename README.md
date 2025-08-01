# Precise ablation zone segmentation on CT images after liver cancer ablation using semi-automatic CNN-based segmentation
By Le Quoc Anh, Xuan Loc Pham, Theo van Walsum, Viet Hang Dao, Tuan Linh Le, Daniel Franklin, Adriaan Moelker, Vu Ha Le, Nguyen Linh Trung, and Manh Ha Luu.

Paper link: 
[[MP]](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.17373) 

Demo link: 
[[Video]](https://drive.google.com/file/d/12IVCkXf9-RO78YgvkLSQ0Bz3iR8ry7za/view)

## Introduction
Ablation zone segmentation in contrast-enhanced computed tomography (CECT) images enables the quantitative assessment of treatment success in the ablation of liver lesions. However, fully automatic liver ablation zonesegmentation in CTimages still remains challenging, such as low accuracy and time-consuming manual refinement of the incorrect regions. In this study, we developed a semi-automatic technique to address the remaining drawbacksandimprovetheaccuracyof theliverablation zone segmentation in the CT images.

## Method
Our approach uses a combination of a CNN-based automatic seg mentation method and an interactive CNN-based segmentation method. First, automatic segmentation is applied for coarse ablation zone segmentation in the whole CT image. Human experts then visually validate the segmentation results. If there are errors in the coarse segmentation,local corrections can be performed on each slice via an interactive CNN-based segmentation method.

## How to run

This repository includes the necessary code and configuration files to replicate our work on ablation zone segmentation.

### System requirements
The source code is intended to run on a Linux workstation equipped with a GPU card.

### Dataset Preparation
In order to evaluate the performance of our framework, we utilize the [Liver Radiofrequency Ablation (RFA) Zones Benchmark Dataset](https://graz.elsevierpure.com/en/publications/liver-radiofrequency-ablation-rfa-zones-benchmark-dataset). Please download the dataset and convert it to NIfTI format to conduct tests with our framework.

### Acknowledgements
This repository utilizes code from [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) and [nnUNet](https://github.com/MIC-DKFZ/nnUNet)


