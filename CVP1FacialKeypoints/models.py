## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)

        num_features1 = 32
        self.conv1 = nn.Conv2d(1, num_features1, 3,padding =1)
        self.bn1 = nn.BatchNorm2d(num_features = num_features1, eps=1e-05, momentum=0.1, affine=True)
        self.maxPool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.2)

        #here input size(32 features, 112x112)
        num_features2 = 64
        self.conv2 = nn.Conv2d(num_features1,num_features2, 3, padding =1)
        self.bn2 =nn.BatchNorm2d(num_features= num_features2, eps=1e-05, momentum=0.1, affine=True)
        self.maxPool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(p=0.2)

        #here input size (64 features, 56x56)
        num_features3 = 128
        self.conv3 = nn.Conv2d(num_features2,num_features3, 3, padding =1)
        self.bn3 =nn.BatchNorm2d(num_features= num_features3, eps=1e-05, momentum=0.1, affine=True)
        self.maxPool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(p=0.2)

        #here input size (128 features, 28x28)
        num_features4 = 128
        self.conv4 = nn.Conv2d(num_features3, num_features4, 3,padding =1)
        self.bn4 = nn.BatchNorm2d(num_features = num_features4, eps=1e-05, momentum=0.1, affine=True)
        self.maxPool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(p=0.2)

        #here input size(128 features, 14x14)
        num_features5 = 128
        self.conv5 = nn.Conv2d(num_features4,num_features5, 3, padding =1)
        self.bn5 =nn.BatchNorm2d(num_features= num_features5, eps=1e-05, momentum=0.1, affine=True)
        self.maxPool5 = nn.MaxPool2d(2, 2)
        self.drop5 = nn.Dropout2d(p=0.2)

        #here input size (128 features, 7x7)
        num_features6 = 256
        self.conv6 = nn.Conv2d(num_features5,num_features6, 3, padding =1)
        self.bn6 =nn.BatchNorm2d(num_features= num_features6, eps=1e-05, momentum=0.1, affine=True)
        self.maxPool6 = nn.MaxPool2d(2, 2, padding =1)
        self.drop6 = nn.Dropout2d(p=0.2)

        #here input size (128 features, 4x4)

        self.avg = nn.AvgPool2d(4, stride=None, padding=0, ceil_mode=False, count_include_pad=False)
        #print(self.avg)
        self.fc1 = nn.Linear(num_features6, 200)
        self.dropfc = nn.Dropout(p=.3)
        self.fc2 = nn.Linear(200, 68*2)



    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        #print (x)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxPool1(x)
        x = self.drop1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxPool2(x)
        x = self.drop2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.maxPool3(x)
        x = self.drop3(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.maxPool4(x)
        x = self.drop4(x)

        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = self.maxPool5(x)
        x = self.drop5(x)

        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = self.maxPool6(x)
        x = self.drop6(x)

        x = self.avg(x)
        x = x.view(-1,256)
        x = F.relu(self.fc1(x))
        x = self.dropfc(x)

        x = self.fc2(x)
        #x = x.view(-1, 128*4*4)
        #print (x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
