# Prostate-Cancer-Image-Classification

> Authors: [Apavou Clément](https://github.com/clementapa) & [Belkada Younes](https://github.com/younesbelkada)

![Python](https://img.shields.io/badge/Python-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-orange.svg?style=plastic)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-blueviolet.svg?style=plastic)

With more than 1 million new diagnoses reported every year, prostate cancer (PCa) is the second most common cancer among males worldwide that results in more than 350,000 deaths annually. The key to decreasing mortality is developing more precise diagnostics. Diagnosis of PCa is based on the grading of prostate tissue biopsies. These tissue samples are examined by a pathologist and scored according to the Gleason grading system. In this challenge, you will develop models for detecting PCa on images of prostate tissue samples, and estimate severity of the disease using the most extensive multi-center dataset on Gleason grading yet available.

The grading process consists of finding and classifying cancer tissue into so-called Gleason patterns (3, 4, or 5) based on the architectural growth patterns of the tumor (Fig. 1). After the biopsy is assigned a Gleason score, it is converted into an ISUP grade on a 1-5 scale. The Gleason grading system is the most important prognostic marker for PCa, and the ISUP grade has a crucial role when deciding how a patient should be treated. There is both a risk of missing cancers and a large risk of overgrading resulting in unnecessary treatment. However, the system suffers from significant inter-observer variability between pathologists, limiting its usefulness for individual patients. This variability in ratings could lead to unnecessary treatment, or worse, missing a severe diagnosis.

- [x] Set-up template code
- [x] Understand how to load the dataset
- [x] create dataset
- [ ] model baseline ? ViT ? timm library https://github.com/rwightman/pytorch-image-models
- [x] metrics for classification + metrics for the competition (Area Under ROC metric) (all with torchmetrics)
- [ ] implement the prediction function for submissions
- [ ] analyse the dataset for split (individual etc...)
- [ ] data augmentations for medical imaging ?
- [ ] set-up clean conda environment
- [ ] create wandb team 

- [ ] let's experiments :
    - different split (may be the best thing to achieve good results on the test)
    - various architectures 
    - tune hyper-parameters

## Interesting articles:
- https://openreview.net/pdf?id=Hk2YYqssf
- https://www.nature.com/articles/s41586-021-03922-4

## Data statistics:

```
labels  counts
0      85
1      85
2      37
3      45
4      42
5      46
```
Need to apply data augmentation