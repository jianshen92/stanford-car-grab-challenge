# stanford-car-grab-challenge
Computer Vision Challenge by Grab

## Results
All models are evaluated with **Top-1 Accuracy** based on the test set provided [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

Stopping Criteria for all models is **no improvement on validation loss** across 2 Cycles of training. One cycle of training refers to training with any number of epochs with the [One Cycle Policy](https://arxiv.org/abs/1803.09820).

1. Comparing different image dimension (Squared Image)

| Training Technique  | Resnet 50 |  Resnet 101 | Resnet 152 |
| ------------- | ------------- |  ------------- | ------------- |
| Baseline - Image Size (224x224) | 87.3  |  88.9  | 89.9  |
| **Baseline - Image Size (299x299)** | **88.0**  |  **90.3**  | **90.7**  |

299x299 image size yield better results. This criteria is applied to all further models.

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

Training done on 16GB NVIDIA Tesla T4, with batch size of 16.

|  | Resnet 50 |  Resnet 101 | Resnet 152 |
| ------------- | ------------- |  ------------- | ------------- |
| Training Time per epoch | 3:30 minutes |  4:10 minutes  | 5:40 minutes  |

## Discussion

## Evaluation with Custom Dataset
### Prerequisites
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
python predict.py
```
It should generate a `test.csv` with predictions based on images in `test_images` directory.


### Running test script
#### Generate a .csv with predictions based on images in a folder
1. Activate virtual environment.
2. Create a fresh directory and placed all the test image in the folder. (Make sure there is nothing else other than images in the folder)
3. In `predict.py`, replace:
* `FILE_PATH`'s value with the absolute path string or object pointing to the folder created above.
* `OUTPUT_PATH`'s value with the desired filename of the output csv, e.g `test.csv`
4. Run `python predict.py` in terminal.
