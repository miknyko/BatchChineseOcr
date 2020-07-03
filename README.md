# 批量自然场景中文OCR工具

能批量侦测图片中的中文并识别

## 介绍

* 文本检测使用YOLO3网络，tf2实现

* OCR使用CRNN，Pytorch实现
* 附带文本方向检测工具，tf2实现



## 使用

1. 安装环境

   由于此项目两个框架共存，安装时需注意tf和pytorch版本！

   推荐

   `pip install tensorflow-gpu==2.0.0`

   `conda install pytorch==1.2.0 torchvision==0.4.0`

   `cuda == 10.0`

   `cudnn == 7.4.2`

   其他依赖安装

   `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

     

2. 下载模型

   模型下载地址:  [百度云](https://pan.baidu.com/s/1PonzgGvHO5JeMMGCfNFrog)

   提取码: ebzj

     

3. 使用说明（默认使用GPU）

   角度检测，文本检测，OCR三个模块推荐分别运行，以最大化批运算速度，脚本内有详细示例说明用法。