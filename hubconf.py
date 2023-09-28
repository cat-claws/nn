# Optional list of dependencies required by the package
dependencies = ["torch"]

import torch

def loader(constructor, pretrained, **kwargs):
	model = constructor(**kwargs)
	if pretrained:
		checkpoint = f'https://github.com/cat-claws/nn/releases/download/parameters/{pretrained}.tar.gz'
		model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
	return model



from simple import Net, Net_, MultiLayerPerceptron
from convduonet import ConvDuoNet

def mlp(pretrained=False, **kwargs):
	return loader(MultiLayerPerceptron, pretrained=pretrained, **kwargs)
	
def convduonet(pretrained=False, **kwargs):
	return loader(ConvDuoNet, pretrained=pretrained, **kwargs)

def tutorialconvnet(pretrained=False, **kwargs):
	return loader(Net, pretrained=pretrained, **kwargs)

def exampleconvnet(pretrained=False, **kwargs):
	return loader(Net_, pretrained=pretrained, **kwargs)



from pytorchcv.models.resnet_cifar import get_resnet_cifar
from pytorchcv.models.wrn_cifar import get_wrn_cifar

def resnet_cifar(pretrained=False, **kwargs):
	return loader(get_resnet_cifar, pretrained=pretrained, **kwargs)
	
def wrn_cifar(pretrained=False, **kwargs):
	return loader(get_wrn_cifar, pretrained=pretrained, **kwargs)
