# stanford-car-grab-challenge
This repository is an attempt for the Computer Vision Challenge by Grab. 

Table of Contents:
* [Overview](#overview)
* [Results](#results)
* [Discussion](#discussion)
* [Evaluation with Custom Dataset](#evaluation-with-custom-dataset)

## Overview
Model is built with fast.ai v1 and PyTorch v1, trained on Google Cloud Platform's Deep Learning VM with 16GB NVIDIA Tesla T4.

Data consist of 8144 Training Images (80:20 Train:Validation Split) and 8041 Test Images. Architecture used is ResNet-152 with squared image (299x299), pretrained with ImageNet. Data is augmented with several affine and perspective transformation. Mixup technique is used. Final Top-1 Accuracy is **92.53%** on Test Images.

## Results
All models are evaluated with **Top-1 Accuracy** based on the test set provided [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

Stopping Criteria for all models is **no improvement on validation loss** across 2 Cycles of training. One cycle of training refers to training with any number of epochs with the [One Cycle Policy](https://arxiv.org/abs/1803.09820).

1. Comparing different image dimension (Squared Image)

| Training Technique  | Resnet 50 |  Resnet 101 | Resnet 152 |
| ------------- | ------------- |  ------------- | ------------- |
| Baseline - Image Size (224x224) | 87.3  |  88.9  | 89.9  |
| **Baseline - Image Size (299x299)** | **88.0**  |  **90.3**  | **90.7**  |

299x299 image size yield better results. This criteria is applied to all further models.

2. Comparing Resizing Methods

| Training Technique  | Resnet 50 |  Resnet 101 | Resnet 152 |
| ------------- | ------------- |  ------------- | ------------- |
| Resizing Method - Zero Padding | 86.0  |  -  | -  |
| Resizing Method - Crop | 86.6  |  -  | -  |
| **Resizing Method - Squishing** | **88.0**  |  -  | -  |

Squishing image yield better results. This criteria is applied to all further models.

3. Using training set with cropped Bounding Box provided

| Training Technique  | Resnet 50 |  Resnet 101 | Resnet 152 |
| ------------- | ------------- |  ------------- | ------------- |
| **Without Bounding Box** | **88.0**  |  **90.3**  | **90.7**  |
| With Bounding Box | 70.3  |  71.7  | 71.9  |

Training Set without bounding box yield better results. This criteria is applied to all further models.

4. Using Mix Up on training data

| Training Technique  | Resnet 50 |  Resnet 101 | Resnet 152 |
| ------------- | ------------- |  ------------- | ------------- |
| Without Mix Up | 88.0  |  90.3  | 90.7  |
| **With Mix Up** | **89.3**  |  **90.9**  | **92.53**  |


### Other Performance Metrics

Training done on Google Cloud Platform Deep Learning VM with GPU 16GB NVIDIA Tesla T4, with batch size of 16.

|  | Resnet 50 |  Resnet 101 | Resnet 152 |
| ------------- | ------------- |  ------------- | ------------- |
| Training Time per epoch | 3:30 minutes |  4:10 minutes  | 5:40 minutes  |

## Discussion
1. I chose **ResNet** as the model architecture because it has achieved State-of-the-Art results for many fine-grained image classification problem since 2015. Recent breakthrough in fine-grained image classification aren't  architectural breakthrough. [arXiv:1901.09891v2](https://arxiv.org/abs/1901.09891v2) and [arXiv:1712.01034v2](https://arxiv.org/abs/1712.01034v2) suggests improvements in data augmentation and normalization layers yield better results.

2. **ResNet-152** provides the best accuracy (2-3% increase) over **ResNet-50** in the expense of increased training time ( 2 minutes/epoch increase).

3. Several Transfer Learning steps are used to achieve the best performing model (in order) : 
* Transfer Learning from model trained with **ImageNet images** to **Mixed-Up Stanford Car's dataset**.
* Transfer Learning from model trained with **Mixed-Up Stanford Car's dataset** to **vanilla Stanford Car's dataset**.

4. Training data are **augmented** with several permutation to improve variety of the dataset. This helps model to generalize better. Details of data augmentation are explained in the *model_training.ipynb* notebook.

5. Image with **higher resolution** trains better model. However that comes at a cost of training time. Due to time constraint I am not able to train images with higher resolution than 299x299.

6. Training with image **squished** to target resolution trains better model. Automatic cropping risks deleting important features that are out of the cropping boundary. Padding introduce artefacts that lowers the training accuracy. Squished Image preserve most features, except in the scenario where the model/make of a car is mostly determined by the width:height ratio (aspect ratio) of a car.

7. Instead of using squared Image, I have experimented on resizing the dataset to **rectangular image** with 16:9 and 4:3 aspect ratios. The aim is to preserve features that is determined by the asepct ratio of a car. It shows a slight increase in accuracy (0.3%). However, this is only achievable because of the dataset provided are mostly in landscape. 

8. Considering most **Grab** users are **mobile**, image taken are usually in portrait. Resizing a portrait image to landscape will severely distort the features of a car. Therefore, I have decided not to select rectangular model as our final model.
  
9. Training with images cropped with **bounding box** works significantly worse. The model trained is not able to distinguish the noise in the background and feature in the foreground in the test dataset.

10. Augmenting data with **[mixup](https://arxiv.org/abs/1710.09412)** yields over 2-3% increase of accuracy. 

## Evaluation with Custom Dataset
### Prerequisites
* Linux Based Operating System (fast.ai does not support MacOS in their current build)
* Use of Virtual Environment such as `conda` or `virtualenv`
* 10 GB of free disk space (To be safe). Pytorch, Fast.ai, and their dependencies takes up good amount of disk space.
* (Optional) [Git Large File Storage](https://git-lfs.github.com/). Used for hosting model files (They are huge).

### Downloading Model File
#### With Git LFS
Before cloning the repository, run:
```
git lfs install
```
in the repository directory to initialize Git LFS. Then, clone repository as usual.

**OR**

If you cloned the repository before initializing, run:
```
git lfs install
git lfs pull
```
in the repository directory to download the model file.
#### Manual download
Download the `best-performing-model.pkl` manually from github and replace the file in your local repository.

### Setting up virtual environment
Setup a `python >= 3.6.0` virtual environement with `conda` or `virtualenv`
#### Installing dependencies
with `pip`
```
pip install -r requirements.txt
```
with `conda`
```
conda install --yes --file requirements.txt
```
### Test if everything works.
Run the following in terminal:
```
python predict.py test_images
```
It should generate a `test.csv` with predictions based on images in `test_images` directory.


### Running test script
#### Generate a .csv with predictions based on images in a folder
1. Activate virtual environment.
2. Create a fresh directory and placed all the test image in the folder. (Make sure there is nothing else other than images in the folder)
3. Run `python predict.py <your_test_folder_path>` in terminal.
