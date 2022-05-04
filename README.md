# LightRay:Lightweight Network for Prohibited Items Detection in X-ray Images during Security Inspection
The code for the paper "LightRay:Lightweight Network for Prohibited Items Detection in X-ray Images during Security Inspection".
## Requirements
python 3.9 <br>
pytorch 1.11.10 <br>
cuda 11.2 
## Dataset Preparation
1.This article uses the .xml format for training, and the SIXray <https://github.com/MeioJane/SIXray> dataset needs to be downloaded before training.<br>
  Subfiles (Annotations: label files, JPEGImages: SIXray images) after decompression are placed in the Xraydevkit/Xray2021 directory).<br>
2.When training your own data set, you can create a cls_classes.txt by yourself, and write the classes you need to distinguish in it.<br>
  The content of the model_data/xray_classes.txt file is:
  ```python
  Gun
  Knife
  Wrench
  Pliers
  Scissors
  ...
  (Note: Here you can change to the classes you need)
  ```
3.Modify the parameters in Xray_annotation.py # annotation_mode=0 # , # classes_path = 'model_data/xray_classes.txt' #<br>
  ```python
  annotation_mode     = 0

  classes_path        = 'model_data/xray_classes.txt'

  trainval_percent    = 0.9
  train_percent       = 0.9
  (Note: The proportion of dataset divided according to needs.)
  ```
4.Run Xray_annotation.py to generate 2021_train.txt and 2021_val.txt in the root directory.
## Train
1.The Pretrained weights required for training can be downloaded from Google Cloud Drive.<br>
  After downloading the library, unzip it, download LightRay.pth or yolov4_mobilenet_v3_voc.pth in Google Cloud Drive, and put it into model_data.<br>
  Link: https://pan.baidu.com/s/1oXz13QwLx1lnXct538qL2Q <br>
  Extraction code: 16qc <br>
2.The default parameters of train.py are used to train the VOC dataset, and the training can be started directly by running train.py.<br>
  Before training classes_path, make it correspond to its own dataset path as the target class.<br>
3.Modify parameters of train.py:
  ```python
  classes_path = 'model_data/xray_classes.txt'
  anchors_path = 'model_data/yolo_anchors.txt'
  ......
  model_path = 'model_data/LightRay.pth' 
  (Note: LightRay.pth is the Pretrained weights of the SIXray dataset. yolov4_mobilenet_v3_voc.pth is the Pretrained weights of the voc dataset.)
  ```
4.train.py. After modifying the classes_path, you can run train.py to start training. After training multiple epochs, the weights will be generated in the logs folder.
## Evaluate (mAP)
1.Modify model_path and classes_path in yolo.py:<br>
  ```python
  "model_path"        : 'model_data/ ',  

  "classes_path"      : 'model_data/xray_classes.txt',
  (Note: This can be modified as needed.)
  ```
2.model_path points to the trained weights file in the logs folder(Select a trained weight you want under the logs file and put it in model_data/).<br>
3.Run mAP.py
## Prediction
1.Training result prediction requires two files, yolo.py and predict.py. Modify model_path and classes_path in yolo.py.<br>
2.model_path points to the trained weights file in the logs folder(Select a trained weight you want under the logs file and put it in model_data/).
  classes_path points to the txt corresponding to the detection classes in yolo.py:
  ```python
  "model_path"        : 'model_data/ ',  

  "classes_path"      : 'model_data/xray_classes.txt',
  (Note: Other parameters can be modified as required.)
  ```
3.Run predict.py. After completing the modification, you can run predict.py for detection. After running, enter the image path to detect.
## Citation
## Acknowledgement
