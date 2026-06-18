# Car Detection Using Facebook Detectron2

---

## Overview

A practical implementation of transfer learning for object detection, fine-tuning Facebook AI's Detectron2 framework on a custom car detection dataset. The project explores how a large pretrained model — Faster R-CNN with a ResNeXt-101 backbone pretrained on COCO — can be adapted to a single-class detection task with a relatively small dataset.

The pipeline covers the full workflow: dataset preparation and registration, model configuration and fine-tuning, evaluation, and exporting bounding box predictions to CSV.

---

## Approach

**Base model:** Faster R-CNN with ResNeXt-101-32x8d-FPN backbone, pretrained on MS-COCO (`COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x`)

**Task:** Single-class object detection — identifying cars in images

**Transfer learning strategy:** The pretrained weights are loaded from the Detectron2 model zoo and fine-tuned on the custom dataset, with the detection head reconfigured for a single class.

---

## Pipeline

**1. Dataset preparation**
- Images loaded from a custom annotated car dataset with COCO-format JSON annotations
- 90/10 random train/validation split
- Bounding boxes parsed from annotations filtered by category ID (cars only)
- Dataset registered with Detectron2's `DatasetCatalog` and `MetadataCatalog`

**2. Model configuration**
- Base config loaded from Detectron2 model zoo
- Key hyperparameters: lr=0.001, warmup over 1000 iterations, max 200 iterations, batch size 4 images, 64 ROIs per image
- Detection head set to 1 class, confidence threshold 0.8 at inference

**3. Training**
- Fine-tuned using Detectron2's `DefaultTrainer`
- Model checkpoint saved to output directory

**4. Evaluation**
- COCO-style evaluation on the validation set using `COCOEvaluator`
- Qualitative visualisation of predictions on validation images

**5. Output generation**
- Custom `csv_maker` function runs inference on the full validation set and exports predicted bounding box coordinates to CSV — one row per detected car per image

---

## Key skills demonstrated

| Area | Details |
|---|---|
| Transfer learning | Fine-tuning a large pretrained detection model on a custom dataset |
| Object detection | Faster R-CNN · Feature Pyramid Network · Region of Interest heads |
| Framework usage | Facebook Detectron2 · model zoo · DefaultTrainer · COCOEvaluator |
| Data engineering | COCO-format annotation parsing · dataset registration · train/val splitting |
| Scientific Python | PyTorch · OpenCV · NumPy · pandas · Matplotlib |
| Deployment | Inference pipeline · bounding box export to CSV |

---

## Requirements

```
detectron2
torch
torchvision
opencv-python
numpy
pandas
matplotlib
```

Install Detectron2:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

---

## Repository note

This project was developed and run on Google Colab. The car dataset and annotations are not included in this repository.

---


