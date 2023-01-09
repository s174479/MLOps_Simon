### Exercise 5
print("Exercise 5")
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
   for i in range(10):
      model(inputs)
      prof.step()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# Same operations takes most time as in exercise 4
# But percentages have changed a bit
# In the for loop the percentages are higher for the top 4 (conv2d+convolution), and lower for the remaining, 
# compared to the non for loop (exercise 4)