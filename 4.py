import time
import torch
import numpy as np

from nets.yolov4 import YoloBody

net = YoloBody(anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],num_classes=5)
net.eval().cuda()

# x是输入图片的大小
x = torch.zeros((1,3,416,416)).cuda()
t_all = []

for i in range(1000):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))
