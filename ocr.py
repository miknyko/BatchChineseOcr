#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Rockstar He
#Date: 2020-06-30
#Description:

import os
import numpy as np
import torch.nn as nn
import torch
import math
import time

from tqdm import tqdm
from crnn.keys import alphabetChinese, alphabetEnglish
from collections import OrderedDict
from crnn.util import resizeNormalize, strLabelConverter
from PIL import Image

class BidirectionalLSTM(nn.Module):
    
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output
    


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nh, leakyRelu=False,lstmFlag=True,GPU=True,alphabet=alphabetChinese):
        """CRNN网络"""
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        self.lstmFlag = lstmFlag
        self.GPU = GPU
        self.alphabet = alphabet
        self.nclass = len(self.alphabet) + 1
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        
        self.cnn = cnn
        if self.lstmFlag:
            self.rnn = nn.Sequential(
                BidirectionalLSTM(512, nh, nh),
                BidirectionalLSTM(nh, nh, self.nclass))
        else:
            self.linear = nn.Linear(nh*2, self.nclass)
            

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        if self.lstmFlag:
           # rnn features
           output = self.rnn(conv)
           T, b, h = output.size()
           output = output.view(T, b, -1)
           
        else:
             T, b, h = conv.size()
             t_rec = conv.contiguous().view(T * b, h)
             output = self.linear(t_rec)  # [T * b, nOut]
             output = output.view(T, b, -1)
             
             
        return output
    
    def load_weights(self,path):
        
        trainWeights = torch.load(path,map_location=lambda storage, loc: storage)
        modelWeights = OrderedDict()
        for k, v in trainWeights.items():
            name = k.replace('module.','') # remove `module.`
            modelWeights[name] = v      
        self.load_state_dict(modelWeights)
        if torch.cuda.is_available() and self.GPU:
            self.cuda()
        self.eval()
        
    
class OcrDataGenerator(torch.utils.data.Dataset):
    def __init__(self,files,batch_size=32,GPU=False):
        self.files = files
        self.batch_size = batch_size
        self.imgH = 32
        self.GPU = GPU

    def __len__(self):
        return math.ceil(len(self.files) / len(self.batch_size))
    
    def __getitem__(self,idx):
        batch_files = self.files[idx * self.batch_size : (idx + 1) *  self.batch_size]
        try:
            batch_image = [Image.open(pth) for pth in batch_files]
        except:
            batch_image = batch_files
        
        batch_max_width = 0
        batch_array = []
        for img in batch_image:
            # 采集batch内每张图片的宽度，找出最长宽度，并按比例初步resize至高为指定高度
            img = img.convert('L')
            img = resizeNormalize(img,self.imgH)
            h, w = img.shape
            batch_max_width = max(w,batch_max_width)
            batch_array.append(np.array(img,dtype=np.float32))
        
        batch_array_final = np.zeros((self.batch_size,1,self.imgH,batch_max_width),dtype=np.float32)

        for i in range(self.batch_size):
            # 将每个batch内的所有图片pad至该batch最大的长度
            h,w = batch_array[i].shape
            batch_array_final[i][:,:,:w] = batch_array[i]

        batch_array_final = torch.from_numpy(batch_array_final)

        if torch.cuda.is_available() and self.GPU:
            batch_array_final = batch_array_final.cuda()
        else:
            batch_array_final = batch_array_final.cpu()

        return batch_array_final


def main():
    pwd = os.getcwd()
    test_path = os.path.join(pwd,"result")
    # OCR模型文件
    ocr_model_path = os.path.join(pwd,"models","ocr-lstm.pth")
    filenames = [os.path.join(test_path, pth) for pth in os.listdir(test_path)]
    # 初始化一个dataloader
    dataloader = OcrDataGenerator(files=filenames, batch_size=16, GPU=True)
    # 读取字母表
    alphabet = alphabetChinese
    # 读取字母表长度，加1位blank位
    nclass = len(alphabet)+1 
    # 初始化一个CRNN模型,注意参数
    crnn = CRNN(32, 1, 256, leakyRelu=False,GPU=True,alphabet=alphabet)
    # 读取参数
    crnn.load_weights(ocr_model_path)
    # 遍历dataloader，开始batch prediction
    result = []
    start = time.time()
    for batch_image in tqdm(dataloader):
        preds = crnn(batch_image) # size of [seq_len,batchsize,nclass]
        preds = preds.argmax(axis=2)
        preds = preds.permute(1, 0)
    # 逐句解码
        for line in preds:
            result.append(strLabelConverter(line,alphabet))
    print(result)
    end = time.time()
    print("\n" + f"[INFO]预测一共使用{end - start}秒" + "\n" )


if __name__ == "__main__":
    main()
    # print(torch.cuda.is_available())


        

            
