import torch
from torchvision.models import resnet18

def get_resnet18_cifar():
	_model = resnet18()
	_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
	_model.maxpool = torch.nn.Identity()
	_model.fc = torch.nn.Linear(512, 10)	
	return _model
