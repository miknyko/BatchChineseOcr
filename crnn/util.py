#!/usr/bin/python
# encoding: utf-8
import numpy as np
from PIL import Image

def resizeNormalize(img,imgH=32):
        scale = img.size[1]*1.0 / imgH
        w     = img.size[0] / scale
        w     = int(w)
        img   = img.resize((w,imgH),Image.BILINEAR)
        w,h   = img.size
        img = (np.array(img)/255.0-0.5)/0.5
        return img
    
    
def strLabelConverter(res,alphabet):
        """
        原始解码，将序列中的连续字母塌缩为一个字母，
        @param res(tensor):模型预测的结果，size of (seqlen)
        @param alphabet(string):所使用的字母表
        @return string:识别的文字
        """
        N = len(res)
        raw = []
        for i in range(N):
            if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
                raw.append(alphabet[res[i] - 1])
        return ''.join(raw)
    
