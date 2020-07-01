'''
@Descripttion: 
@version: 
@Author: Rockstar He
@Date: 2020-07-01 10:41:36
@LastEditors: Rockstar He
@LastEditTime: 2020-07-01 12:58:10
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Rockstar He
#Date: 2020-07-01
#Description:对已经放正角度的自然场景图片进行文字检测并识别

import os
import textdetect
import ocr
import time
from tqdm import tqdm

yolo_batchsize = 2 # 根据配置调整


class BatchDetectOcr():
    """批量发现自然场景图片中的文字并进行识别"""
    def __init__(self):
        self.text_detector = textdetect.yolo_text(num_classes, anchors)
        self.ocr = ocr.CRNN(32, 1, nclass, 256, leakyRelu=False, GPU=GPU,alphabet=alphabet)

    def job(self,filenames):
        self.filenames = filenames
        self.yolo_dataloader = textdetect.YoloImageGenerator(self.filenames, batch_size=yolo_batchsize)
        result = {}
        start = time.time()
        print("\n" + "#" * 30 + f"开始检测，时间{start}" + "#" * 30 + "\n")
        print("\n" + f"[INFO] 一共{len(self.filenames)}张图片" + "\n" )
        for batch_img,batch_shape,batch_shape_padded,batch_filenames in tqdm(self.yolo_dataloader):
            # 从所有待测图片中批读取图片进行文字检测
            batch_preds = self.text_detector(batch_img)
            batch_boxes,batch_scores = net_output_process(batch_preds,batch_shape_padded,batch_shape_padded)
            for img,filename,boxes in zip(batch_img,batch_filenames,batch_boxes):
                # 遍历批图片逐图进行ocr
                result[filename] = []
                partImgs = cut_batch(img,filename,boxes,save=False)
                temp_partImg_loader = ocr.OcrDataGenerator(partImgs,batch_size=16,GPU=GPU)
                for batch_partImg in temp_partImg_loader:
                    # 从批截取图片中进行ocr
                    preds = crnn(batch_image) # size of [seq_len,batchsize,nclass]
                    preds = preds.argmax(axis=2)
                    preds = preds.permute(1, 0)
                    for line in preds:
                        # 逐句解码
                        result[filename].append(strLabelConverter(line,alphabet))
        end = time.time()
        print("\n" + "#" * 30 + f"结束检测，时间{end}" + "#" * 30 + "\n")
        print("\n" + f"[INFO]一共用时{end - start}秒,每张图片平均用时{(end - start)/len(self.filenames)}秒。" + "\n" )
        print(result)
        return result


def main():
    ocr = BatchDetectOcr()


if __name__ == "__main__":
    main()
