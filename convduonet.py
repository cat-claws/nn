import torch
import torch.nn as nn

class ConvDuoNet(nn.Module):
	""" adapted from 7-layer CNN with batch normalisation in:
	https://github.com/Verified-Intelligence/auto_LiRPA/blob/master/examples/vision/models/feedforward.py
	and
	https://github.com/shizhouxing/Fast-Certified-Robust-Training/blob/main/models/feedforward.py
	"""
	def __init__(self, in_channels = 3, num_classes = 10, planes = 64, features = 16, adaptive = False):
		super(ConvDuoNet, self).__init__()
		self.layers = nn.ModuleList([
			nn.Conv2d(in_channels, planes, 3, stride=1, padding=1),
			nn.BatchNorm2d(planes),
			nn.ReLU(),
			nn.Conv2d(planes, planes, 3, stride=1, padding=1),
			nn.BatchNorm2d(planes),
			nn.ReLU(),
			nn.Conv2d(planes, 2 * planes, 3, stride=2, padding=1),
			nn.BatchNorm2d(2 * planes),
			nn.ReLU(),
			nn.Conv2d(2 * planes, 2 * planes, 3, stride=1, padding=1),
			nn.BatchNorm2d(2 * planes),
			nn.ReLU(),
			nn.Conv2d(2 * planes, 2 * planes, 3, stride=1, padding=1),
			nn.BatchNorm2d(2 * planes),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(features ** 2 * planes * 2, features ** 2 * 2),
			nn.BatchNorm1d(features ** 2 * 2),
			nn.ReLU(),
			nn.Linear(features ** 2 * 2, num_classes)
		])
		if adaptive:
			self.layers.insert(15, nn.AdaptiveAvgPool2d((features, features)))
	
	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
