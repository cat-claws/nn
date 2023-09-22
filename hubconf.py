# Optional list of dependencies required by the package
dependencies = ["torch"]

import torch
from simpleconv import Net, Net_
from convduonet import ConvDuoNet

def loader(constructor, pretrained, **kwargs):
	model = constructor(**kwargs)
	if pretrained:
		checkpoint = f'https://github.com/cat-claws/nn/releases/download/parameters/{pretrained}.tar.gz'
		model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
	return model


def convduonet(pretrained=False, **kwargs):
	return loader(ConvDuoNet, pretrained=pretrained, **kwargs)

def tutorialconvnet(pretrained=False, **kwargs):
	return loader(Net, pretrained=pretrained, **kwargs)

def exampleconvnet(pretrained=False, **kwargs):
	return loader(Net_, pretrained=pretrained, **kwargs)
