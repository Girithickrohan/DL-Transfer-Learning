# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## THEORY


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

 Import Libraries and Load the Dataset

### STEP 2: 

Load Pretrained VGG19 Model

### STEP 3: 

Modify Classifier

### STEP 4: 

Define Loss Function and Optimizer

### STEP 5: 

Train the Model

### STEP 6: 

 Evaluate the Model

## PROGRAM

### Name: GIRITHICK ROHAN

### Register Number: 212223230063

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models 
from torchvision.utils import make_grid
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

Include your plot here

## Confusion Matrix

Include confusion matrix here

## Classification Report
Include classification report here

### New Sample Data Prediction
Include your sample input and output here

## RESULT
Thus,the program excuted sucessfuly.
