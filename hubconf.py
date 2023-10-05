# Optional list of dependencies required by the package
dependencies = ["torch"]

import torch

def loader(constructor, pretrained, **kwargs):
	model = constructor(**kwargs)
	if pretrained:
		checkpoint = f'https://github.com/cat-claws/nn/releases/download/parameters/{pretrained}.tar.gz'
		model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
	return model



from simple import Net, Net_, MultiLayerPerceptron, SimpleCNN
from convduonet import ConvDuoNet

def mlp(pretrained=False, **kwargs):
	return loader(MultiLayerPerceptron, pretrained=pretrained, **kwargs)
	
def convduonet(pretrained=False, **kwargs):
	return loader(ConvDuoNet, pretrained=pretrained, **kwargs)

def tutorialconvnet(pretrained=False, **kwargs):
	return loader(Net, pretrained=pretrained, **kwargs)

def exampleconvnet(pretrained=False, **kwargs):
	return loader(Net_, pretrained=pretrained, **kwargs)

def simplecnn(pretrained=False, **kwargs):
	return loader(SimpleCNN, pretrained=pretrained, **kwargs)

from pytorchcv.models.resnet_cifar import get_resnet_cifar
from pytorchcv.models.wrn_cifar import get_wrn_cifar
from pytorchcv.models.densenet_cifar import get_densenet_cifar

def resnet_cifar(pretrained=False, **kwargs):
	return loader(get_resnet_cifar, pretrained=pretrained, **kwargs)
	
def wrn_cifar(pretrained=False, **kwargs):
	return loader(get_wrn_cifar, pretrained=pretrained, **kwargs)

def densenet_cifar(pretrained=False, **kwargs):
	return loader(get_densenet_cifar, pretrained=pretrained, **kwargs)

from inception_resnet_v1 import InceptionResnetV1

def inception_resnet_v1(pretrained=False, **kwargs):
	return loader(InceptionResnetV1, pretrained=pretrained, **kwargs)

from iresnet import IResNet

def iresnet(pretrained=False, **kwargs):
	return loader(IResNet, pretrained=pretrained, **kwargs)

from lightcnn import LightCNN

def lightcnn(pretrained=False, **kwargs):
	return loader(LightCNN, pretrained=pretrained, **kwargs)


from MobileFaceNets import MobileFaceNet
from EfficientNets import EfficientNet
from GhostNet import GhostNet
from AttentionNets import ResidualAttentionNet
from TF_NAS import TF_NAS_A
from ReXNets import ReXNetV1
from LightCNN import LightCNN
from RepVGG import RepVGG

def mobilefacenet_facexzoo(pretrained=False, **kwargs):
	return loader(MobileFaceNet, pretrained=pretrained, **kwargs)

def efficientnet_facexzoo(pretrained=False, **kwargs):
	return loader(EfficientNet, pretrained=pretrained, **kwargs)

def ghostnet_facexzoo(pretrained=False, **kwargs):
	return loader(GhostNet, pretrained=pretrained, **kwargs)

def attentionnet_facexzoo(pretrained=False, **kwargs):
	return loader(ResidualAttentionNet, pretrained=pretrained, **kwargs)

def tf_nas_facexzoo(pretrained=False, **kwargs):
	return loader(TF_NAS_A, pretrained=pretrained, **kwargs)

def rexnet_facexzoo(pretrained=False, **kwargs):
	return loader(ReXNetV1, pretrained=pretrained, **kwargs)

def lightcnn_facexzoo(pretrained=False, **kwargs):
	return loader(LightCNN, pretrained=pretrained, **kwargs)

def repvgg_facexzoo(pretrained=False, **kwargs):
	return loader(RepVGG, pretrained=pretrained, **kwargs)
