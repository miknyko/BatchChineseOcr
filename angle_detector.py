
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Rockstar He
#Date: 2020-06-30
#Description:

import numpy as np
import cv2
import os
from tensorflow.keras.utils import Sequence
import time
import math
from PIL import Image

class AngleDetector():
    """
    使用opencv判别图片文字方向，只能判别4个角度
    """
    def __init__(self):
        pwd = os.getcwd()
        AngleModelPb    = os.path.join(pwd,"models","saved_model.pb")
        AngleModelPbtxt = os.path.join(pwd,"models","saved_model.pbtxt")
        self.angleNet = cv2.dnn.readNetFromTensorflow(AngleModelPb,AngleModelPbtxt)

    def angle_detect_dnn(self,img,adjust=True):
        """
        角度预测
        @param img(array):size of [batchsize,w,h,c]
        @return result(array):size of [batchsize],每张图片需要旋转的角度，顺时针方向
        """
        ROTATE = [0,90,180,270]
        inputBlob = cv2.dnn.blobFromImages(img, 
                                        scalefactor=1.0,
                                        swapRB=True,
                                        mean=[103.939,116.779,123.68],
                                        crop=False)
        self.angleNet.setInput(inputBlob)
        pred = self.angleNet.forward()
        index = np.argmax(pred,axis=1)

        return np.array(ROTATE)[index]
    

class AngleImageGenerator(Sequence):
    """
    图片侦测器的图片读取器
    具体请参照tf文档
    """
    def __init__(self,filenames,batch_size):
        self.batch_size = batch_size
        self.filenames = filenames # 所有待检测图片完整路径的列表
        self.adjust_thresh = 0.001
        
    def __len__(self):
        return math.ceil(len(self.filenames) / self.batch_size)
    
    def __getitem__(self,idx):
        """
        @return batchimage(array):size of [batchsize,w,h,c]
        @return batch_files(list):该batch中的对应文件名
        """
        batch_files = self.filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_imgs = []
        for filename in batch_files:
            try:
                img = Image.open(filename).convert('RGB')
            except:
                print("\n" + f"[WARNING!]{filename} 无法读取，请检查图片" + "\n" )
                continue
            w, h = img.size
            # 需要裁剪图片边缘以便提升效果
            xmin, ymin, xmax, ymax = int(self.adjust_thresh * w), int(self.adjust_thresh * h), w-int(self.adjust_thresh * w), h-int(self.adjust_thresh * h)
            img = img.crop((xmin, ymin, xmax, ymax))
            img = img.resize((224, 224))
            batch_imgs.append(np.array(img))

        return np.stack(batch_imgs),batch_files

def main():
    # 使用示例
    test_pth = r'C:\Users\Thinkpad\chineseocr-app\test'
    filenames = [os.path.join(test_pth,pth) for pth in os.listdir(test_pth)]
    total = len(filenames)
    generator = AngleImageGenerator(filenames, batch_size=20)
    detector = AngleDetector()
    start = time.time()
    for batch_image,batch_filenames in generator:
       res = detector.angle_detect_dnn(batch_image)
       print(res)
    end = time.time()
    print("\n" + f"[INFO]一共{total}张图片，共使用{end - start}秒, 每张图片使用{(end - start)/total}秒" + "\n" )


if __name__ == "__main__":
    main()
