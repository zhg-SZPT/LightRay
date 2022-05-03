#计算吞吐量

import torch
from nets.yolov4 import YoloBody

model = YoloBody(anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],num_classes=5)
device = torch.device("cuda")
model.to(device)
dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(device)
repetitions=100
total_time = 0
optimal_batch_size=16
with torch.no_grad():
  for rep in range(repetitions):
     starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     starter.record()
     _ = model(dummy_input)
     ender.record()
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)/1000
     total_time += curr_time
Throughput = (repetitions*optimal_batch_size)/total_time
print("Final Throughput:",Throughput)