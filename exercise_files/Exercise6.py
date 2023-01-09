### Exercise 6
print("Exercise 6")

## a
print("a")
import torch
import torchvision.models as models
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)
with profile(activities=[ProfilerActivity.CPU], on_trace_ready=tensorboard_trace_handler("log/resnet18")) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

## b
# To open TensorFlow use either:
# 1. "Launch TensorBoard Session" extension in VS
# 2. type "tensorboard --logdir=./log" in the terminal and go to "http://localhost:6006/#pytorch_profiler"
# in a browser

## c
print("c")
model = models.resnet34()
inputs = torch.randn(5, 3, 224, 224)
with profile(activities=[ProfilerActivity.CPU], on_trace_ready=tensorboard_trace_handler("log/resnet34")) as prof:
    for i in range(10):
        model(inputs)
        prof.step()