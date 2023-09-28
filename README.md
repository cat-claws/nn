# Neural network models in PyTorch
These might not be the most popular models, but are the ones we find useful in our research.

## Load a neural network model with pure ```torch```
```python
import torch
model = torch.hub.load('cat-claws/nn', 'convduonet', pretrained='convduonet_JQyC', in_channels = 3)
x = torch.rand(16, 3, 32, 32)
print(model(x))
```

The ```pytorchcv``` library is not necessary unless we use models from this library, _e.g._, load a 14-layer ResNet for CIFAR-size images
```python
model = torch.hub.load('cat-claws/nn', 'resnet_cifar', pretrained= False, num_classes=10, blocks=14, bottleneck=False, in_channels = 1)
```
In this case, ```pip install pytorchcv``` is required.

To apply models designed for CIFAR-size images on other sizes, _e.g._, MNIST, padding might be a good solution
```python
x = torch.rand(16, 1, 28, 28)

model = torch.hub.load('cat-claws/nn', 'densenet_cifar', num_classes=10, blocks=10, growth_rate=12, bottleneck=False, in_channels = 1)
# or
model = torch.hub.load('cat-claws/nn', 'wrn_cifar', num_classes=10, blocks=10, width_factor=4, in_channels = 1)

model(torch.nn.functional.pad(x, 4 * [2]))
```

Others
```
model = torch.hub.load('cat-claws/nn', 'mlp', hidden = [120, 84])
```
