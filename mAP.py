# ---------------   mAP ---------------#

import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from yolo import YOLO
from util.utils import get_classes
from util.utils_map import get_coco_map, get_map

if __name__ == "__main__":

    # ------------------------------------------------------------------------------------------------------------------#
    #   map_mode is used to specify what is calculated when the file is run
    #   map_mode = 0 represents the entire map calculation process,
    #   including obtaining the prediction result, obtaining the real frame, and calculating the VOC_map
    #   map_mode = 1 means that only prediction results are obtained.
    #   map_mode = 2 means only the ground truth box is obtained.
    #   map_mode = 3 means only VOC_map is calculated.
    # -------------------------------------------------------------------------------------------------------------------#
    map_mode = 0

    classes_path = 'model_data/xray_classes.txt'

    #   MINOVERLAP is used to specify the mAP0.x you want to get

    MINOVERLAP = 0.5
    # -------------------------------------------------------------------------------#
    #  map_vis is used to specify whether to enable visualization of map calculations
    # -------------------------------------------------------------------------------#
    map_vis = True
    Xraydevkit_path = 'Xraydevkit'
    # -------------------------------------------------------#
    #   The folder of the result output,
    #   the default address folder in the same level directory is map_out
    # -------------------------------------------------------#
    map_out_path = 'map_out'

    image_ids = open(os.path.join(Xraydevkit_path, "Xray2021/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence=0.001, nms_iou=0.5)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(Xraydevkit_path, "Xray2021/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(Xraydevkit_path, "Xray2021/Annotations/" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') != None:
                        difficult = obj.find('difficult').text
                        if int(float(difficult)) == 1:
                            difficult_flag = True
                    # obj_name = obj.find('name').text
                    obj_name = getattr(obj.find('name'), 'text', None)
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path=map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")
