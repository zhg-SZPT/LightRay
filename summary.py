#--------------------------------------------#
#  This part of the code is used to see the network structure and parameters
#--------------------------------------------#
import torch
from torchsummary import summary
from nets.yolov4 import YoloBody

if __name__ == "__main__":
    # Need to use device to specify whether the network runs on GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = YoloBody(anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], num_classes=5).to(device)
    summary(m, input_size=(3, 416, 416))

