# LightRay:Lightweight Network for Prohibited Items Detection in X-ray Images during Security Inspection
Link to the original paper:[LightRay:Lightweight Network for Prohibited Items Detection in X-ray Images during Security Inspection]
## Requirements
python 3.9 <br>
pytorch 1.11.10 <br>
cuda 11.2 
## Dataset Preparation
1.本文使用.xml格式进行训练，训练前需要下载好SIXray数据集，解压后放在Xraydevkit目录下的子文件(Annotations:标签文件，JPEGImages：SIXray图片)<br>
2.修改Xray_annotation.py里面的# annotation_mode=0 #，运行Xray_annotation.py生成根目录下的2007_train.txt和2007_val.txt。
## Train

## Evaluate (mAP)
## Prediction
## Acknowledgement
