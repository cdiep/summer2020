#!/usr/bin/env python
# coding: utf-8

# # Use PyTorch to predict handwritten digits
# <table style="border: none" align="left">
#    <tr style="border: none">
#        <td style="border: none"><img src="https://github.com/IBM/pytorch-on-watson-studio/raw/master/doc/source/images/pytorch-pattern-header.jpg" width="600" alt="Icon"></td>
#    </tr>
# </table>

# The sample model used here is the <a href="https://github.com/pytorch/examples/tree/master/mnist">MNIST model</a> from the official PyTorch examples repository. 
# 
# ## Training MNIST Model

# ** PyTorch Tools & Libraries **
# 
# An active community of researchers and developers have built a rich
# ecosystem of tools and libraries for extending PyTorch and supporting
# development in areas from computer vision to reinforcement learning.
# 
# PyTorch's <a href="https://github.com/pytorch/vision" target="_blank"
# rel="noopener no referrer">torchvision</a> is one of those packages.
# `torchvision` consists of popular datasets, model architectures, and common
# image transformations for computer vision.
# 
# This tutorial will use `torchvision's MNIST dataset` package to download
# and process the training data.

# The following code will download and process the MNIST training and test data. 


import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import numpy as np
#from IPython.display import display
import os
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def parse_args():
    import argparse
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('image', type=str,
                    help='The image file of a hand-written-digit used for prediction')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--saved-model-file', type=str, default='mnist_cnn.pt',
                    help='The saved model used for weights initialization')
   
    args = parser.parse_args()
    return(args)


def main():
    args=parse_args()
  
    if not os.path.isfile(args.saved_model_file):
        print(args.saved_model_file, "file does not exist. Please use --saved-model-file option to pass in a model file")
        exit()
    if not os.path.isfile(args.image):
        print(args.image, "file does not exist to perform prediction.")
        exit() 

    saved_model_file = args.saved_model_file    
    mnist_model = Net()
    mnist_model.load_state_dict(torch.load(saved_model_file, map_location='cpu'))

    filename = args.image

    img = Image.open(filename).resize((28, 28)).convert('L')
    img.show()
    data = torch.from_numpy(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
    output = mnist_model(data)

    digits = [i for i in range(10)]
    # get the index of the max log-probability
    prediction = output.max(1, keepdim=True)[1]
    print("Prediction for the number in the image file is:",  digits[prediction[0,0]])

if __name__ == "__main__":
    main()