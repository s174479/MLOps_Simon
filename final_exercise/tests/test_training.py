from main import train
import torch

#train()
# Train the model manually before
# This test then tests whether the saved weights and biases has the right shape
def test_fc1_weight_size():
    state_dict = torch.load("checkpoint.pth")
    assert state_dict['fc1.weight'].size() == torch.Size([256, 784]), "Size of fc1 weights are not [256, 784]"
def test_fc1_bias_size():
    state_dict = torch.load("checkpoint.pth")
    assert state_dict['fc1.bias'].size() == torch.Size([256]), "Size of fc1 bias are not [256]"
def test_fc2_weight_size():
    state_dict = torch.load("checkpoint.pth")
    assert state_dict['fc2.weight'].size() == torch.Size([128,256]), "Size of fc2 weights are not [128, 256]"
def test_fc2_bias_size():
    state_dict = torch.load("checkpoint.pth")
    assert state_dict['fc2.bias'].size() == torch.Size([128]), "Size of fc2 bias are not [128]"
def test_fc3_weight_size():
    state_dict = torch.load("checkpoint.pth")
    assert state_dict['fc3.weight'].size() == torch.Size([64,128]), "Size of fc3 weights are not [64, 128]"
def test_fc3_bias_size():
    state_dict = torch.load("checkpoint.pth")
    assert state_dict['fc3.bias'].size() == torch.Size([64]), "Size of fc3 bias are not [64]"
def test_fc4_weight_size():
    state_dict = torch.load("checkpoint.pth")
    assert state_dict['fc4.weight'].size() == torch.Size([10,64]), "Size of fc4 weights are not [10, 64]"
def test_fc4_bias_size():
    state_dict = torch.load("checkpoint.pth")
    assert state_dict['fc4.bias'].size() == torch.Size([10]), "Size of fc4 bias are not [10]"