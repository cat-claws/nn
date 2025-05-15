"""Resnet implementation is based on the implementation found in:
https://github.com/YisenWang/MART/blob/master/resnet.py
https://github.com/yaodongyu/TRADES/blob/master/models/resnet.py
https://github.com/WenRuiUSTC/EntF/blob/main/models/resnet.py
"""

import torch
import torchvision

def get_resnet_cifar(block, layers, **kwargs):
	block = torchvision.models.resnet.Bottleneck if ('bottle' in block.lower() or 'neck' in block.lower()) else torchvision.models.resnet.BasicBlock
	model = torchvision.models.ResNet(block, layers, **kwargs)
	
	model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
	model.maxpool = torch.nn.Identity()  # Remove maxpool
	return model
