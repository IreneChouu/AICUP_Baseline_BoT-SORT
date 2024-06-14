# AICUP Baseline: BoT-SORT With YOLOv5-SimAM

> [**BoT-SORT: Robust Associations Multi-Pedestrian Tracking**](https://arxiv.org/abs/2206.14651)
> 
> Nir Aharon, Roy Orfaig, Ben-Zion Bobrovsky

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bot-sort-robust-associations-multi-pedestrian/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=bot-sort-robust-associations-multi-pedestrian)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bot-sort-robust-associations-multi-pedestrian/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=bot-sort-robust-associations-multi-pedestrian)

> [!IMPORTANT]  
> **This baseline is based on the code released by the original author of [BoT-SORT](https://github.com/NirAharon/BoT-SORT). Special thanks for their release.**


> [!WARNING]
>  - **This baseline only provides single-camera object tracking and does not include cross-camera association.**
>  - **Due to our dataset's low frame rate (fps: 1), we have disabled the Kalman filter in BoT-SORT. Low frame rates can cause the Kalman filter to deviate, hence we only used appearance features for tracking in this baseline.**
## YOLOv5-SimAM Integration

### Introduction

YOLOv5-SimAM is an enhanced version of the original YOLOv5 object detection model that incorporates the SimAM (Simple Attention Mechanism) module. This combination leverages the strengths of YOLOv5's efficient object detection capabilities with SimAM's attention mechanism to improve performance, especially in scenarios with cluttered backgrounds or varying lighting conditions.

### Motivation

In recent years, the applications of camera systems in home security and crime prevention have increased significantly. However, existing surveillance systems are typically based on recordings from individual cameras, each operating independently. This setup leads to several issues:
- When a moving object leaves the field of view of one camera, the system cannot continue tracking it.
- During incidents such as traffic accidents or crimes, the lack of coordination between cameras forces law enforcement to spend considerable human resources manually searching through recordings to trace the movements of suspicious vehicles or pedestrians.

To address these problems, we need a multi-camera multi-object tracking technology that can provide continuous tracking across different cameras.

### Purpose

The purpose of integrating YOLOv5 with SimAM in this project is to develop an AI model capable of tracking the same vehicle across multiple cameras. This competition provides cross-camera road vehicle videos and aims to encourage teams to develop models that can seamlessly track the same vehicle across different cameras. This effort will deepen Taiwan's AI technology in the smart transportation field and promote its diverse development.


## ToDo
- [x] Complete evaluation guide
- [x] Visualize results on AICUP train_set
- [ ] Release test set

### Visualization results on AICUP train_set

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/Ofc7FS5D8uY/0.jpg)](https://youtu.be/Ofc7FS5D8uY)


# YOLOv5 with SimAM Attention Mechanism: Setup and Training Guide

This guide details the steps taken to set up and train a YOLOv5 model with the SimAM attention mechanism for the AICUP Baseline BoT-SORT project.

## Environment Setup

1. **Activate YOLOv7 environment:**

    ```bash
    conda activate yolov7
    ```

2. **Install Git:**

    ```bash
    conda install git
    ```

3. **Navigate to project directory:**

    ```bash
    cd D:\meeting\AICUP_Baseline_BoT-SORT
    ```

4. **Clone the project repository:**

    ```bash
    git clone https://github.com/IreneChouu/AICUP_Baseline_BoT-SORT.git
    ```

5. **Install required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

6. **Reinstall pycocotools:**

    ```bash
    pip install --no-cache-dir --force-reinstall pycocotools
    ```

7. **Install CMake:**

    ```bash
    pip install Cmake
    ```

8. **Install cython_bbox:**

    ```bash
    pip install cython_bbox
    ```

9. **Add conda-forge channel:**

    ```bash
    conda config --add channels conda-forge
    ```

10. **Install FAISS for GPU:**

    ```bash
    conda install faiss-gpu
    ```

## Prepare ReID Dataset

Generate ReID patches from the AICUP dataset:

```bash
python fast_reid/datasets/generate_AICUP_patches.py --data_path /datasets/train
```
## Prepare YOLOv7 Dataset

Convert the AICUP dataset to YOLOv7 format:

```bash
python yolov7/tools/AICUP_to_YOLOv7.py --AICUP_dir D:\meeting\AICUP_Baseline_BoT-SORT\datasets\train --YOLOv7_dir D:\meeting\AICUP_Baseline_BoT-SORT\datasets\yolo
```
## Training (Fine-tuning)
Fine-tune the model using the provided configuration file:
```bash
python3 fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"

```
## SimAM Attention Mechanism
1.Start training with SimAM attention mechanism:
```bash
python yolov7/train.py --device 0 --batch-size 2 --epochs 50 --data yolov7/data/AICUP.yaml --img 1280 1280 --cfg yolov7/cfg/training/SimAm.yaml --weights 'pretrained/yolov7-e6e.pt' --name yolov7-AICUP-SimAM --hyp data/hyp.scratch.custom.yaml

```
2.Address NameError: name 'C3' is not defined error:
Follow the guide to add the C3 module:
YOLOv7 添加 C3 模块 https://blog.csdn.net/qq_38668236/article/details/127106905

## 模型訓練結果

在經過45個epochs的訓練後，模型達到了以下性能指標：

| Epoch | GPU Memory | Box Loss | Obj Loss | Cls Loss | Total Loss | Labels | Img Size | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|------------|----------|----------|----------|------------|--------|----------|-----------|--------|---------|--------------|
| 41/44 | 1.74G      | 0.01638  | 0.004218 | 0        | 0.0206     | 2      | 1280     | 0.822     | 0.907  | 0.925   | 0.71         |
| 42/44 | 1.74G      | 0.01632  | 0.004193 | 0        | 0.02052    | 2      | 1280     | 0.821     | 0.907  | 0.925   | 0.71         |
| 43/44 | 1.74G      | 0.01623  | 0.004181 | 0        | 0.02041    | 5      | 1280     | 0.822     | 0.905  | 0.925   | 0.71         |
| 44/44 | 1.74G      | 0.01626  | 0.004197 | 0        | 0.02045    | 0      | 1280     | 0.821     | 0.907  | 0.924   | 0.709        |


