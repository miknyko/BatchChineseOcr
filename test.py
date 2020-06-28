import os
import json
import time
import numpy as np
from PIL import Image
from config import *

from apphelper.image import union_rbox,adjust_box_to_origin,base64_to_PIL

#CPU 启动
os.environ["CUDA_VISIBLE_DEVICES"] = ''

scale,maxScale = IMGSIZE[0],2048
from text.keras_detect import  text_detect
from text.opencv_dnn_detect import angle_detect

from crnn.keys import alphabetChinese,alphabetEnglish
from crnn.network_torch import CRNN

alphabet = alphabetChinese
ocrModel = ocrModelTorchLstm

nclass = len(alphabet)+1 
crnn = CRNN( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=GPU,alphabet=alphabet)
crnn.load_weights(ocrModel)
ocr = crnn.predict_job

from main import TextOcrModel

model = TextOcrModel(ocr,text_detect,angle_detect)



def ocr():
    img = Image.open(image_path)
    img = np.array(img)
    result,angle= model.model(img,
                                scale=scale,
                                maxScale=maxScale,
                                detectAngle=False,##是否进行文字方向检测，通过web传参控制
                                MAX_HORIZONTAL_GAP=100,##字符之间的最大间隔，用于文本行的合并
                                MIN_V_OVERLAPS=0.6,
                                MIN_SIZE_SIM=0.6,
                                TEXT_PROPOSALS_MIN_SCORE=0.1,
                                TEXT_PROPOSALS_NMS_THRESH=0.3,
                                TEXT_LINE_NMS_THRESH = 0.99,##文本行之间测iou值
                                LINE_MIN_SCORE=0.1,
                                leftAdjustAlph=0.01,##对检测的文本行进行向左延伸
                                rightAdjustAlph=0.01,##对检测的文本行进行向右延伸
                                )
    result = union_rbox(result,0.2)
    res = [{'text':x['text'],
            'name':str(i),
            'box':{'cx':x['cx'],
                    'cy':x['cy'],
                    'w':x['w'],
                    'h':x['h'],
                    'angle':x['degree']
        
                }
            } for i,x in enumerate(result)]
    res = adjust_box_to_origin(img,angle, res)
    print(res)

if __name__ == '__main__':
    image_name = r'test\img.jpeg'
    cwd = os.getcwd()
    image_path = os.path.join(cwd,image_name)
    ocr()