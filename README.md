# Shape Detection with Jetson Nano

This project demonstrates how to build a shape detection system using the **NVIDIA Jetson Nano**. It covers training a custom shape detection model, applying **transfer learning** techniques, and deploying the model for inference using the `jetson_inference` library.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Prepare the Setup](#prepare-the-setup)
- [Training the Model](#training-the-model)
- [Inference with `jetson_inference`](#inference-with-jetson_inference)
- [Results](#results)

## Introduction

The goal of this project is to detect various shapes (e.g., circles, squares, triangles) in images using deep learning on the Jetson Nano. By leveraging **transfer learning**, we fine-tune a pre-trained model for our specific shape detection task, and then use the Jetson Nano for real-time inference.

## Requirements

- **NVIDIA Jetson Nano** with JetPack installed
- **Jetson Inference Library** (`jetson_inference`) for model deployment and inference
- **Python 3.6+**
- **CUDA** and **cuDNN** (installed with JetPack on Jetson Nano)
- **PyTorch** for model training
- **OpenCV** for image processing

## Prepare the Setup

1. [Cloning the Repo](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md#cloning-the-repo) jetson-inference
2. Cloning the Repo shape-detection-with-jetson
``` bash
$ cd $HOME
$ git clone https://github.com/giangdo11/shape-detection-with-jetson
```
3. [Launching the Container](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md#launching-the-container) and [Mounted Data Volumes](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md#mounted-data-volumes)
``` bash
$ cd jetson-inference
$ docker/run.sh --volume ~/shape-detection-with-jetson:/shape-detection
```

## Training the Model

### 1. Transfer Learning

Transfer learning allows us to leverage a model pre-trained on a large dataset (like `SSD-Mobilenet-v2`) and fine-tune it to recognize specific shapes with a smaller dataset. This reduces the training time and resources needed, making it efficient for Jetson Nano's capabilities.

### 2. Prepare the Dataset

- [Colleting dataset](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect-detection.md#collecting-your-own-detection-datasets) by using [camera-capture](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect-detection.md#launching-the-tool) tool

![alt text](https://github.com/giangdo11/shape-detection-with-jetson/blob/main/images/image.png)

### 3. [Re-training SSD-Mobilenet](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md#re-training-ssd-mobilenet)
- Using train_ssd.py to [Re-training SSD-Mobilenet](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md#training-the-ssd-mobilenet-model)
``` bash
$ cd jetson-inference/python/training/detection/ssd
$ python3 train_ssd.py --dataset-type=voc --data=data/<YOUR-DATASET> --model-dir=models/<YOUR-MODEL> --epochs=100
```

### 4. [Converting the Model to ONNX](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md#converting-the-model-to-onnx)
``` bash
$ python3 onnx_export.py --model-dir=models/<YOUR-MODEL>
```
After converting the model, we will have `ssd-mobilenet.onnx` and `labels.txt` under path /jetson-inference/python/training/detection/ssd/models/YOUR-MODEL/

## Inference with `jetson_inference`

Use the model that is trained in the above step to detect shapes with jetson_inference. In this project, I've already trained the model and put it in /shape-detection-with-jetson/models.zip. You have to unzip this file.

``` bash
$ cd /shape-detection
$ python3 shape-detection.py <path-to-the-model> <input> <output>
$ python3 shape-detection.py /shape-detection/models /dev/video0
```

## Results

After running inference, the model should be able to accurately detect and classify shapes within the images. Performance may vary based on shape complexity, image quality, and lighting conditions.

![alt text](https://github.com/giangdo11/shape-detection-with-jetson/blob/main/images/image-1.png)
