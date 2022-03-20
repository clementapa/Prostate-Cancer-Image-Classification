# Classification of ISUP grades from Whole Slide Images - DLMI Kaggle Challenge

> Authors: [Apavou Clément](https://github.com/clementapa) & [Belkada Younes](https://github.com/younesbelkada)

![Python](https://img.shields.io/badge/Python-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-orange.svg?style=plastic)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-blueviolet.svg?style=plastic)

The kaggle challenge is the following : https://www.kaggle.com/c/mvadlmi/leaderboard

## :mag_right: Introduction

With more than 1 million new diagnoses reported every year, prostate cancer (PCa) is the second most common cancer among males worldwide that results in more than 350,000 deaths annually. The key to decreasing mortality is developing more precise diagnostics. Diagnosis of PCa is based on the grading of prostate tissue biopsies. These tissue samples are examined by a pathologist and scored according to the Gleason grading system. In this challenge, you will develop models for detecting PCa on images of prostate tissue samples, and estimate severity of the disease using the most extensive multi-center dataset on Gleason grading yet available.

The grading process consists of finding and classifying cancer tissue into so-called Gleason patterns (3, 4, or 5) based on the architectural growth patterns of the tumor (Fig. 1). After the biopsy is assigned a Gleason score, it is converted into an ISUP grade on a 1-5 scale. The Gleason grading system is the most important prognostic marker for PCa, and the ISUP grade has a crucial role when deciding how a patient should be treated. There is both a risk of missing cancers and a large risk of overgrading resulting in unnecessary treatment. However, the system suffers from significant inter-observer variability between pathologists, limiting its usefulness for individual patients. This variability in ratings could lead to unnecessary treatment, or worse, missing a severe diagnosis.

 <p align="center">
  <img src="https://github.com/clementapa/Prostate-Cancer-Image-Classification/blob/main/assets/readme_img/isup_grade_explain.png" width="100%" height="100%" alt="Isup Grade explication"/>
</p>

The goal of this challenge is to predict the ISUP Grade using only Histopathology images. For that, you will need to deal with the process of Whole Slide Images as huge gigapixel images and deal with the limited number of patients provided in the train set.

## :hammer: Getting started

Dowload the dataset and put it in the [```assets```](https://github.com/clementapa/Prostate-Cancer-Image-Classification/tree/main/assets/) folder. 

Chose the mode that you want:
- Classification: Classify isup grade of images
- Segmentation: Semantic segmentation on images 
- Classif_WITH_Seg: Classification using a semantic segmentation models trained with Segmentation

Chose a dataset and a model adapted to the mode.\
Models for: 
- [Classification](https://github.com/clementapa/Prostate-Cancer-Image-Classification/tree/main/models/Classification.py)
- [Segmentation](https://github.com/clementapa/Prostate-Cancer-Image-Classification/tree/main/models/Segmentation.py)
- [Classif_WITH_Seg](https://github.com/clementapa/Prostate-Cancer-Image-Classification/tree/main/models/ClassifWithSeg.py) 

Check dataset in [```datasets.py```](https://github.com/clementapa/Prostate-Cancer-Image-Classification/tree/main/datasets/datasets.py) 

```
python3 main.py
```
## :star: :diamonds: Best model (Submission)

```
python main.py --MODE Classification --feature_extractor_name tresnet_xl_448 --network_name SimpleModel --dataset_name ConcatPatchDataset --patch_size 256 --nb_samples 36 --max_epochs 150 --batch_size 2 --accumulate_grad_batches 16 --discounted_draw True 
```
<p align="center">

| Model| Backbone |Area Under ROC (weighted) validation | Area Under ROC (macro) test (private leaderboard) | Run  |
|---|---|---|---|---|
| SimpleModel | tresnet_xl_448 | 0.80 | 0.92647 | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/test-dlmi/runs/2cbesog0?workspace=user-clementapa) |
</p>


## :art: Semantic Segmentation 

<p align="center">

| Model| Backbone | Data provider| Patch Size | Level | IoU (average over classes) validation | Run  |
|---|---|---|---|---|---|---|
| DeepLabV3Plus | resnet152 | All | 384 | 1 | 0.79 | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/test-dlmi/runs/w1qry9c1?workspace=user-clementapa) |
| DeepLabV3Plus | resnet34 | Radboud | 512 | 0 | 0.7029 | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/dlmi/runs/3mmxo0az?workspace=user-clementapa) |
| DeepLabV3Plus | resnet34 | Karolinska | 512 | 0 | 0.5958 | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/dlmi/runs/egp1owm6?workspace=user-clementapa) |
</p>

Karolinska is composed of 3 classes:
- 0: background (non tissue) or unknown 
- 1: benign tissue (stroma and epithelium combined) 
- 2: cancerous tissue (stroma and epithelium combined)

Radboud is composed of 6 classes:  
- 0: background (non tissue) or unknown 
- 1: stroma (connective tissue, non-epithelium tissue) 
- 2: healthy (benign) epithelium 
- 3: cancerous epithelium (Gleason 3) 
- 4: cancerous epithelium (Gleason 4) 
- 5: cancerous epithelium (Gleason 5)

We merged in 3 classes to have the same number as karolinska: 
- 0: background (non tissue) or unknown {0} 
- 1: benign tissue (stroma and epithelium combined) {1,2} 
- 2: cancerous tissue (stroma and epithelium combined) {3,4,5}

 <p align="center">
     <img src="https://github.com/clementapa/Prostate-Cancer-Image-Classification/blob/main/assets/readme_img/media_images_val_predictions_508_0.png" width="25%" height="25%" alt="segmentation prediction"/>
    <img src="https://github.com/clementapa/Prostate-Cancer-Image-Classification/blob/main/assets/readme_img/media_images_val_predictions_508_0_pred.png" width="25%" height="25%" alt="segmentation prediction"/>
    <img src="https://github.com/clementapa/Prostate-Cancer-Image-Classification/blob/main/assets/readme_img/media_images_val_predictions_508_0_gt.png" width="25%" height="25%" alt="segmentation ground truth"/>
</p>
<p align="center">
<em> Segmentation of a Patch 384x384 from WSI: Patch, Prediction, Ground Truth </em>
</p>

 <p align="center">
     <img src="https://github.com/clementapa/Prostate-Cancer-Image-Classification/blob/main/assets/readme_img/media_images_val_predictions_508_2.png" width="25%" height="25%" alt="segmentation prediction"/>
    <img src="https://github.com/clementapa/Prostate-Cancer-Image-Classification/blob/main/assets/readme_img/media_images_val_predictions_508_2_pred.png" width="25%" height="25%" alt="segmentation prediction"/>
    <img src="https://github.com/clementapa/Prostate-Cancer-Image-Classification/blob/main/assets/readme_img/media_images_val_predictions_508_2_gt.png" width="25%" height="25%" alt="segmentation ground truth"/>
</p>
<p align="center">
<em> Segmentation of a Patch 384x384 from WSI of the Radboud data provider: Patch, Prediction, Ground Truth </em>
</p>

- Blue: background or unknown 
- Red: benign tissue 
- Green: Cancerous tissue


#########################################################################################
- [x] Set-up template code
- [x] Understand how to load the dataset
- [x] create dataset
- [x] model baseline ? ViT ? timm library https://github.com/rwightman/pytorch-image-models
- [x] metrics for classification + metrics for the competition (Area Under ROC metric) (all with torchmetrics)
- [x] implement the prediction function for submissions
- [x] analyse the dataset for split (individual etc...)
- [x] data augmentations for medical imaging ?
- [ ] set-up clean conda environment
- [x] create wandb team 

- [x] let's experiments :
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

## TODO:
- [x] merge code usefull of all branches
- [x] two mode to pick patches:
    - [x] before training (create if not exists)
    - [x] pick random patch with openslide open
- [x]  Deal with the dataset (split train/val)
    - equal repartition of data provider in the validation (train_test_split sklearn attribut stratify)
    - equal reparition of classes in the validation (train_test_split sklearn attribut stratify)
- [x] Segmentation
    - [x] possibility to chose data provider (radboud(6), karolinska (3) and all(3, merge radboud))
- [ ] Ressayer les techniques de ML en utilisant le modèle de segmentation (https://www.nature.com/articles/s41746-019-0112-2.pdf)
- [x] combiner concat patches (donne une mosaique de patch) -> segmentation -> segmentation + features pour classifier

Bien faire attention à la taille des patchs qu'on prends et le level count quand on ouvre le tiff