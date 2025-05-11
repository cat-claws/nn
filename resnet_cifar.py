'''
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

"""Resnet implementation is based on the implementation found in:
https://github.com/YisenWang/MART/blob/master/resnet.py
https://github.com/yaodongyu/TRADES/blob/master/models/resnet.py
https://github.com/WenRuiUSTC/EntF/blob/main/models/resnet.py
"""

import torch
import torchvision

# .models import resnet, resnet18, resnet34, resnet50, resnet101, resnet152

# def _resnet(block, layers, **kwargs):
    
#     _model = torchvision.models.ResNet(block, layers, **kwargs)

   

# def get_resnet18_cifar():
# 	_model = ResNet(block, layers, **kwargs)
# 	_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# 	_model.maxpool = torch.nn.Identity()
# 	_model.fc = torch.nn.Linear(512, 10)	
# 	return _model
# _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

def get_resnet_cifar(block, layers, **kwargs):
	if ('bottle' in block.lower() or 'neck' in block.lower()):
		block = torchvision.models.resnet.Bottleneck
	else:
		block = torchvision.models.resnet.BasicBlock
	print(block)
	model = torchvision.models.ResNet(block, layers, **kwargs)
	
	model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
	model.maxpool = torch.nn.Identity()  # Remove maxpool
	
	# model.fc = torch.nn.Linear(512, kwargs["num_classes"])

	# Optional: modify BasicBlock if you don't want to remove ReLU after bn1
	if not bool(kwargs.get("relu", False)):
		for m in model.modules():
			if isinstance(m, torchvision.models.resnet.BasicBlock):
				m.relu = torch.nn.Identity()
	
	return model
