# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision.datasets as dataset
import torch.nn as nn
import cv2
import numpy as np
import os
from PIL import Image


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
 print("hello")

 class basicBlock(nn.Module):
  def __init__(self,input_channel,num_channel,use_1x1conv=False,strides=1):#the default para(strides=1) keep same size, if strind =2 , size reduce to half, 如果下采样或者通道数改变（strind!=1） 都需要启动1x1
      super().__init__()
      self.conv1=nn.Conv2d(in_channels=input_channel,out_channels=num_channel,stride=strides,padding=1,kernel_size=3,bias=None)
      self.conv2=nn.Conv2d(in_channels=num_channel,out_channels=num_channel,stride=1,padding=1,kernel_size=3,bias=None)
      if use_1x1conv:
       self.conv1x1=nn.Conv2d(in_channels=input_channel,out_channels=num_channel,kernel_size=1,stride=strides,bias=None)#padding=0 这一层bias也可以关闭，因为输出后的结果会在下一个残差块进行bn，又会bias重复
      else:
       self.conv1x1=None
      self.bn1=nn.BatchNorm2d(num_channel)
      self.bn2=nn.BatchNorm2d(num_channel)
      self.relu=nn.ReLU()#可以复用因为没有参数在上面
  def forward(self,x):#pre-ativation
      output=self.conv1(self.relu(self.bn1(x)))
      output = self.conv2(self.relu(self.bn2(output)))
      if self.conv1x1:
          x=self.conv1x1(x)
      return output+x

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
class resNet(nn.Module):
    def __init__(self,input_channel,classes_num):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(input_channel,64,7,2,3),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.r1=self.block(input_channel,64,2,True)
        self.r2=self.block(64,128,2)
        self.r3=self.block(128,256,2)
        self.r4=self.block(256,512,2)
        self.avgPool=nn.daptiveAvgPool2d((1,1))
        self.flateen=nn.Flatten(start_dim=1)
        #flatten
        self.fc=nn.Linear(512,classes_num)

    def block(self,inputchannel,outputchannel,num_radius,is_first_block=False):
        blk=[]
        for i in range(num_radius):
            if(i==0 and  not is_first_block):#先变通道再缩小
                blk.append(basicBlock(inputchannel,outputchannel,True,2))
            else:
                blk.append(basicBlock(outputchannel,outputchannel))# 其他的都不需要变通道也不需要缩小，所以1x1conv不用，默认stride=1

        return nn.Sequential(*blk)
    def forward(self,x):
        x=self.stem(x)
        x=self.r1(x)
        x=self.r2(x)
        x=self.r3(x)
        x=self.r4(x)
        x=self.avgPool(x)
        x=self.flateen(x)
        x=self.fc(x)
        return x


