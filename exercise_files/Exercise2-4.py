### Torch version
#import torch
#print(torch.__version__)

### Exercise 2
print("Running exercise 2")

## 2a
print("2a")
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
   model(inputs)

## 2b
print("2b")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# aten::conv2d takes most of the CPU

## 2c
print("2c")
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
# Cannot see a correlation between shape of input an cost of operation...


### Exercise 3
# See change from line 17 to 18
# Added argument: profile_memory=True
print("Exercsie 3")
with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
   model(inputs)
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


### Exercise 4
print("Exercise 4")
prof.export_chrome_trace("ex4_trace.json")