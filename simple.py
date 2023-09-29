import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):	
	def __init__(self, in_features=784, out_features=10, hidden = [120, 84]):
		super().__init__()
		layers = []
		for i, j in zip([in_features] + hidden, hidden + [out_features]):
			layers.extend([nn.Linear(i, j), nn.ReLU()])
		self.dense = nn.ModuleList(layers[:-1])
		
	def forward(self, x):
		x = x.flatten(1)
		for d in self.dense:
			x = d(x)
		return x

class Net(nn.Module):
	
	# https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/neural_networks_tutorial.py
	
	def __init__(self, in_channels = 3):
		super(Net, self).__init__()
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		self.conv1 = nn.Conv2d(in_channels, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
	
	def forward(self, x):
		# Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# If the size is a square, you can specify with a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class Net_(nn.Module):
	
	# https://github.com/pytorch/examples/blob/main/mnist/main.py
	
	def __init__(self, in_channels = 3):
		super(Net_, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)
	
	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		# output = F.log_softmax(x, dim=1)
		return x # output

class SimpleCNN(nn.Module):
	def __init__(self, convs, linears, num_classes=10, dropout=None):
		super(SimpleCNN, self).__init__()
	
		layers = []
	
		for l in convs:
			layers.extend([nn.Conv2d(*l), nn.ReLU(), nn.MaxPool2d(2)])
			if dropout:
				layers.append(nn.Dropout2d(dropout))
		
		layers.append(nn.Flatten(start_dim=1))
	
		for i, j in zip(linears[:-1], linears[1:]):
			layers.extend([nn.Linear(i, j), nn.ReLU()])
			if dropout:
				layers.append(nn.Dropout(dropout))
				
		layers.append(nn.Linear(linears[-1], num_classes))
	
		self.layers = nn.Sequential(*layers)   # Changed to nn.Sequential
	
	def forward(self, x):
		x = self.layers(x)
		return x
