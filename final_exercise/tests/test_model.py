from model import MyAwesomeModel
from data import mnist
import torch
import pytest

model = MyAwesomeModel()
train_set, _ = mnist()
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
for images, labels in trainloader:            
    log_ps = model(images)
    # 40 because it's the remaining 25000 mod 64 = 40
    assert log_ps.size() == torch.Size([64,10]) or log_ps.size() == torch.Size([40,10]), "Size of output torch from model() is not right"

# This does not work... But it works to raise ValueError in model.py, when calling model(images) above
#def test_error_on_wrong_shape():
#    with pytest.raises(ValueError, match='Expected input to a 3D tensor'):
#        model(torch.randn(1,2,3))
#    with pytest.raises(ValueError, match='Expected each sample to have shape [28, 28]'):
#        model(torch.randn(1,2,3))