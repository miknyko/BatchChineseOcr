import tensorflow as tf
import os
import numpy as np
import math
import time
from PIL import Image
from text.keras_yolo3 import yolo_text,box_layer,K
from text.detector.detectors import TextDetector
from apphelper.image import sort_box

#规定anchor box大小，具体请见yolo原理
keras_anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
class_names = ['none','text',]
pwd = os.getcwd()
kerasTextModel=os.path.join(pwd,"models","text.h5")

#预处理anchors
anchors = [float(x) for x in keras_anchors.split(',')]
anchors = np.array(anchors).reshape(-1, 2)

num_anchors = len(anchors)
num_classes = len(class_names)

#读取模型，加载参数
textModel = yolo_text(num_classes,anchors)
textModel.load_weights(kerasTextModel)

def pad_image(img,reshape_size):
    """
    将图片pad至指定大小，归一，方便模型批处理
    @params img(array):PIL读取的图像
    @params reshape_size(tuple):短边固定大小，长边固定大小
    @return img(array):shape of (padded_h,padded_w,3)
    @return original_size(tuple):图片原始宽和高
    """
    w,h = img.size
    ratio = max(h,w) / min(h,w)

    # 横向图片和竖向图片需要不同的pad策略
    if w > h:
        new_w = math.ceil(reshape_size[0] * ratio)
        new_h = reshape_size[0]
        img = img.resize((new_w,new_h))
        img = np.array(img,dtype='float32')
        img /= 255.
        pad_size = reshape_size[1] - new_w
        # 横向pad至指定size
        img = np.pad(img,((0,0),(0,pad_size),(0,0)))
    
    else:
        new_w = reshape_size[0]
        new_h = math.ceil(reshape_size[0] * ratio)
        img = img.resize((new_w,new_h))
        img = np.array(img,dtype='float32')
        img /= 255.
        pad_size = reshape_size[1] - new_h
        # 竖向pad至指定size
        img = np.pad(img,((0,pad_size),(0,0),(0,0)))

    return img,(w,h)

def net_output_process(batch_preds,batch_shape,batch_shape_padded,prob=0.05):
    """
    将主干网络的批输出转换为boxes,scores,这里方便兼容之前代码，
    暂且使用for循环,以后可替换为vectorize处理
    @params batch_preds(list of arrays):list长度代表n个采样尺度，其中每个\
        array形状为(batch_size,grid_size_w,grid_size_h,3*(4+1+num_classes))
    @params batch_shape(list of tuples):图片原始长宽
    @params prob(float):置信度小于prob的box将被忽略
    @returns batch_boxes
    @returns batch_scores
    """
    batch_boxes = []
    batch_scores = []

    MAX_HORIZONTAL_GAP = 100
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6
    textdetector = TextDetector(MAX_HORIZONTAL_GAP,MIN_V_OVERLAPS,MIN_SIZE_SIM)

    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    TEXT_LINE_NMS_THRESH = 0.3
    LINE_MIN_SCORE = 0.8

    # 首先初步对模型主干输出进行预处理
    for y1,y2,y3,image_shape,input_shape in zip(batch_preds[0],batch_preds[1],batch_preds[2],batch_shape,batch_shape_padded):
        outputs = [y1,y2,y3,image_shape,input_shape]
        box,scores = box_layer(outputs,anchors,num_classes)
        w,h = image_shape
        keep = np.where(scores>prob)
        # box[:, 0:4][box[:, 0:4]<0] = 0
        box = np.array(box)
        scores = np.array(scores)
        box[box < 0] = 0
        box[:, 0][box[:, 0]>=w] = w-1
        box[:, 1][box[:, 1]>=h] = h-1
        box[:, 2][box[:, 2]>=w] = w-1
        box[:, 3][box[:, 3]>=h] = h-1
        boxes = box[keep[0]]
        scores = scores[keep[0]]

         # 筛选出需要的box，并且进行nms，字符行组合
        boxes,scores = textdetector.detect(boxes,
                                scores[:, np.newaxis],
                                (image_shape[1],image_shape[0]),
                                TEXT_PROPOSALS_MIN_SCORE,
                                TEXT_PROPOSALS_NMS_THRESH,
                                TEXT_LINE_NMS_THRESH,
                                LINE_MIN_SCORE
                                )
        boxes = sort_box(boxes)
        batch_boxes.append(boxes)
        batch_scores.append(scores)
        print('done')

    return batch_boxes,batch_scores



class YoloImageGenerator(tf.keras.utils.Sequence):
    """
    读取图片，预处理，送入YOLO网络寻找文字区域   
    """

    def __init__(self,filenames,batch_size=32,reshape_size=(608,1280)):
        """
        @params filenames(list):每一个元素为图片绝对路径
        @batch_size(int):批处理大小
        @reshape_size(tuple):短边固定大小，长边固定大小
        """
        self.batch_size = batch_size
        self.filenames = filenames
        # 需要根据横向或竖向图像分别制定不同的reshape size
        self.channels = 3
        self.reshape_size = reshape_size
        self.reshape_horizontal_size = reshape_size
        self.reshape_vertical_size  = (reshape_size[1],reshape_size[0])
        self.horizontal_image = []
        self.vertical_image = []
        
        # 横向和竖向图像不能放在一个batch里面读取，需要分开
        for pth in self.filenames:
            img = Image.open(pth).convert('RGB')
            w,h = img.size
            if w >= h:
                self.horizontal_image.append(pth)
            else:
                self.vertical_image.append(pth)

        self.horizontal_batch = math.ceil(len(self.horizontal_image) / self.batch_size)
        self.vertical_batch = math.ceil(len(self.vertical_image) / self.batch_size)
     
    def __len__(self):
        return self.horizontal_batch + self.vertical_batch

    def __getitem__(self,idx):
        """
        tensorflow sequence内置函数，每一次调用返回一个批次的
        图像array,原始尺寸，pad后尺寸，及图片完整路径
        """
        # 首先预测横向图片
        if idx < self.horizontal_batch:
            batch_image = self.horizontal_image[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_array = []
            batch_shape = []
            batch_shape_padded = []

            for i,pth in enumerate(batch_image):
                img = Image.open(pth).convert('RGB')
                img,(w,h)= pad_image(img,self.reshape_size)
                batch_array.append(img)
                batch_shape.append((w,h))
                batch_shape_padded.append(self.reshape_horizontal_size)
            

            return np.stack(batch_array),batch_shape,batch_shape_padded,batch_image
        
        # 横向batch结束后预测竖向batch
        else:
            new_idx = idx - self.horizontal_batch
            batch_image = self.vertical_image[new_idx * self.batch_size:(new_idx + 1) * self.batch_size]
            batch_array = []
            batch_shape = []
            batch_shape_padded = []
            for i,pth in enumerate(batch_image):
                img = Image.open(pth).convert('RGB')
                img,(w,h) = pad_image(img,self.reshape_size)
                batch_array.append(img)
                batch_shape.append((w,h))
                batch_shape_padded.append(self.reshape_vertical_size)

            return np.stack(batch_array),batch_shape,batch_shape_padded,batch_image

def sort_box(box):
    """
    对box排序,及页面进行排版
        box[index, 0] = x1
        box[index, 1] = y1
        box[index, 2] = x2
        box[index, 3] = y2
        box[index, 4] = x3
        box[index, 5] = y3
        box[index, 6] = x4
        box[index, 7] = y4
    """
    
    box = sorted(box,key=lambda x:sum([x[1],x[3],x[5],x[7]]))
    return list(box)

def rotate_cut_img(im,box,leftAdjustAlph=0.0,rightAdjustAlph=0.0):
    angle,w,h,cx,cy = solve(box)
    degree_ = angle*180.0/np.pi
    
    box = (max(1,cx-w/2-leftAdjustAlph*(w/2))##xmin
           ,cy-h/2,##ymin
           min(cx+w/2+rightAdjustAlph*(w/2),im.size[0]-1)##xmax
           ,cy+h/2)##ymax
    newW = box[2]-box[0]
    newH = box[3]-box[1]
    tmpImg = im.rotate(degree_,center=(cx,cy)).crop(box)
    box = {'cx':cx,'cy':cy,'w':newW,'h':newH,'degree':degree_,}
    return tmpImg,box

def cut_batch(filename,boxes,leftAdjustAlph=0.0,rightAdjustAlph=0.0):
    """
    将批图像剪切测试结果
    """
    im = Image.open(filename).convert('RGB')
    cut_areas = []
    for index,box in enumerate(boxes):
        partImg,box = rotate_cut_img(im,box,leftAdjustAlph,rightAdjustAlph)
        partImg.save(f'result/{os.path.split(filename)[1]}_{index}.jpg')
    print(f'[INFO]图片{os.path.split(filename)[1]}已经处理完毕')


def main():
    start = time.time()
    # img = Image.open(r'C:\Users\Thinkpad\LabelOCR\test_iamge\pic00067747cx_guowejie1578902351244.jpeg').convert('RGB')
    batchtest = np.random.rand(16,608,900,3)
    batch_shape_test = [(406,406)] * 16
    batch_shape_padded_test = [(608,900)] * 16
    pred = textModel.predict_on_batch(batchtest)
    boxes,scores = net_output_process(pred,batch_shape_test,batch_shape_padded_test)
    end = time.time()
    print('test passed!')
    print(f'{end - start}s')

def test():
    start = time.time()
    test_pth = r'E:\dataset\yolotest'
    filenames = [os.path.join(test_pth,pth) for pth in os.listdir(test_pth) ]
    generator = YoloImageGenerator(filenames,batch_size=4)
    for batch_img,batch_shape,batch_shape_padded,batch_filenames in generator:
        pred = textModel(batch_img,training=False)
        boxes,scores = net_output_process(pred,batch_shape,batch_shape_padded)
        for i,box in enumerate(boxes):
            # 分别处理每张图片
            box = sort_box(box)
            filename = batch_filenames[i]
            cut_batch(filename,box)


if __name__ == "__main__":
    test()
    # main()

