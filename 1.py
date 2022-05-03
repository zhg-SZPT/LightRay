#计算运算量FLOPs 参数Params

import torch
from nets.yolov4 import YoloBody
from thop import profile

if __name__ == "__main__":
    model = YoloBody(anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],num_classes=5)
    input = torch.randn(1, 3, 416, 416) #模型输入的形状,batch_size=1
    flops, params = profile(model, inputs=(input, ))
    print("flops=",flops/1e9,"params=",params/1e6) #flops单位G，par