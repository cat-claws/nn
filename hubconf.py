# Optional list of dependencies required by the package
dependencies = ["torch"]

import torch
from simpleconv import Net, Net_
from convduonet import ConvDuoNet

def loader(constructor, pretrained, **kwargs):
	model = constructor(**kwargs)
	if pretrained:
		checkpoint = f'https://github.com/cat-claws/nn/releases/{pretrained}.tar.gz'
		model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
	return model


def convduonet(pretrained=False):
	return loader(ConvDuoNet, pretrained=pretrained)

def tutorialconvnet(pretrained=False):
	return loader(Net, pretrained=pretrained)

def exampleconvnet(pretrained=False):
	return loader(Net_, pretrained=pretrained)
