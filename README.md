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

The goal of this challenge is to predict the ISUP Grade using only Histopathology images. For that, we dealt with the process of Whole Slide Images as huge gigapixel images and deal with the limited number of patients provided in the train set.

Classes: [0, 1, 2, 3, 4, 5]

## :hammer: Getting started

Download the dataset and extract it in the [```assets```](https://github.com/clementapa/Prostate-Cancer-Image-Classification/tree/main/assets/) folder. 

Chose the mode that you want:
- ```Classification```: Classify isup grade of images
- ```Segmentation```: Semantic segmentation on images 
- ```Classif_WITH_Seg```: Classification using a semantic segmentation models trained with Segmentation

Chose a dataset and a model adapted to the mode.\
Models for: 
- [Classification](https://github.com/clementapa/Prostate-Cancer-Image-Classification/tree/main/models/Classification.py)
- [Segmentation](https://github.com/clementapa/Prostate-Cancer-Image-Classification/tree/main/models/Segmentation.py)
- [Classif_WITH_Seg](https://github.com/clementapa/Prostate-Cancer-Image-Classification/tree/main/models/ClassifWithSeg.py) 

Check dataset in [```datasets.py```](https://github.com/clementapa/Prostate-Cancer-Image-Classification/tree/main/datasets/datasets.py) 

Feature extractor from [timm library](https://rwightman.github.io/pytorch-image-models/). 

## :star: Best model with segmentation (Final Submission)

Name method: ```Concatenate top patches``` 

MODE: ```Classif_WITH_Seg```\
dataset_name: ```ConcatTopPatchDataset```\
feature_extractor_name: ```tresnet_xl_448```\
network_name: ```SimpleModel```

Command line to train the model:
```
python main.py --train True --MODE Classif_WITH_Seg --dataset_name ConcatTopPatchDataset --patch_size 256 --nb_samples 16 --max_epochs 150 --batch_size 2 --accumulate_grad_batches 8 --discounted_draw False --percentage_blank 0.5 --resized_img 512 --seed_everything 6836
```
```drawn-dream-632``` is the name of the wandb run of the segmentation model trained with our framework (mode ```Segmentation```)

Command line to create submission csv file:
```
python main.py --train False --MODE Classif_WITH_Seg --dataset_name ConcatTopPatchDataset --patch_size 256 --nb_samples 16 --discounted_draw False --percentage_blank 0.5 --resized_img 512 --best_model rich-jazz-915
```
```rich-jazz-915``` is the name of the wandb run with weights of the model. (Name change if you train your model yourself)

<p align="center">

| Model| Backbone |Area Under ROC (weighted) validation | Area Under ROC (macro) test (private leaderboard) | Run  |
|---|---|---|---|---|
| SimpleModel | [tresnet_xl_448](https://rwightman.github.io/pytorch-image-models/models/tresnet/) | 0.8126 | 0.8833 | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/test-dlmi/runs/nm7y1p0l?workspace=user-clementapa) |
</p>


 <p align="center">
     <img src="https://github.com/clementapa/Prostate-Cancer-Image-Classification/blob/main/assets/readme_img/media_images_train_predictions_seg_1756_0.png" width="50%" height="50%" alt="wsi"/>

</p>
<p align="center">
<em> Top patches concatenated from a wsi images. Prediction: 4, Label: 4.  </em>
</p>


## :star: Best model without segmentation (Submission)

Name method: ```Concatenate random patches``` 

MODE: ```Classification```\
dataset_name: ```ConcatPatchDataset```\
feature_extractor_name: ```tresnet_xl_448```\
network_name: ```SimpleModel```


Command line to train the model:
```
python main.py --train True --MODE Classification --dataset_name ConcatPatchDataset --patch_size 256 --nb_samples 36 --max_epochs 150 --batch_size 2 --accumulate_grad_batches 16 --discounted_draw True --seed_everything 6130
```

Command line to create submission csv file:
```
python main.py --train False --MODE Classification --dataset_name ConcatPatchDataset --patch_size 256 --nb_samples 36 --discounted_draw True --best_model denim-terrain-844
```
```denim-terrain-844``` is the name of the wandb run with weights of the model. (Name change if you train your model yourself)

<p align="center">

| Model| Backbone |Area Under ROC (weighted) validation | Area Under ROC (macro) test (private leaderboard) without voting| with voting | Run  |
|---|---|---|---|---|---|
| SimpleModel | [tresnet_xl_448](https://rwightman.github.io/pytorch-image-models/models/tresnet/) | 0.8034 | [0.8774, 0.92647] | 0.8641 | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/test-dlmi/runs/2cbesog0?workspace=user-clementapa) |
</p>


 <p align="center">
     <img src="https://github.com/clementapa/Prostate-Cancer-Image-Classification/blob/main/assets/readme_img/media_images_val_predictions_902_0.png" width="25%" height="25%" alt="wsi"/>
    <img src="https://github.com/clementapa/Prostate-Cancer-Image-Classification/blob/main/assets/readme_img/media_images_val_predictions_902_1.png" width="25%" height="25%" alt="wsi"/>
</p>
<p align="center">
<em> Random patches concatenated from a wsi images. Left: label 1, Radboud provider, Right: label 1, Karolinska provider </em>
</p>

## :art: Semantic Segmentation 

MODE: ```Segmentation```\
dataset_name: ```PatchSegDataset```

<p align="center">

| Model| Backbone | Data provider| Patch Size | Level | IoU (average over classes) validation | Run  |
|---|---|---|---|---|---|---|
| [DeepLabV3Plus](https://smp.readthedocs.io/en/latest/models.html#id9) | resnet152 | All | 384 | 1 | 0.7858 | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/test-dlmi/runs/w1qry9c1?workspace=user-clementapa) |
| [DeepLabV3Plus](https://smp.readthedocs.io/en/latest/models.html#id9) | resnet34 | Radboud | 512 | 0 | 0.7029 | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/dlmi/runs/3mmxo0az?workspace=user-clementapa) |
| [DeepLabV3Plus](https://smp.readthedocs.io/en/latest/models.html#id9) | resnet34 | Karolinska | 512 | 0 | 0.5958 | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/dlmi/runs/egp1owm6?workspace=user-clementapa) |
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

### :star: Best Segmentation Model

MODE: ```Segmentation```\
dataset_name: ```PatchSegDataset```\
network_name: ```DeepLabV3Plus```\
feature_extractor_name: ```resnet152```

```
python main.py --train True --MODE Segmentation --dataset_name PatchSegDataset --dataset_static False --max_epochs 150 --batch_size 4 --accumulate_grad_batches 16 --nb_samples 4 --patch_size 384 --percentage_blank 0.5 --level 1 --seed_everything 4882
```
<p align="center">

| Model| Backbone | Data Provider | mIoU validation | Run  |
|---|---|---|---|---|
| [DeepLabV3Plus](https://smp.readthedocs.io/en/latest/models.html#id9) | resnet152 | Both | 0.7858 | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/test-dlmi/runs/w1qry9c1?workspace=user-clementapa) |
</p>

 <p align="center">
    <a href="https://smp.readthedocs.io/en/latest/">
    <img src="https://i.ibb.co/dc1XdhT/Segmentation-Models-V2-Side-1-1.png" width="50%" height="50%" alt="logo"/>
    </a>
</p>
