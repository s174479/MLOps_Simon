### Exercise 7
print("Exercise 7")
import torch
import torchvision.models as models
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU], on_trace_ready=tensorboard_trace_handler("log/vae_mnist_working")) as prof:
    import vae_mnist_working

# To open TensorFlow use either:
# 1. "Launch TensorBoard Session" extension in VS
# 2. type "tensorboard --logdir=./log" in the terminal and go to "http://localhost:6006/#pytorch_profiler"
# in a browser