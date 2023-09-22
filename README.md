# Neural network models in PyTorch
These might not be the most popular models, but are the ones we find useful in our research.

## Load a neural network model with pure ```torch```
```python
import torch
model = torch.hub.load('cat-claws/nn', 'convduonet', pretrained='convduonet_JQyC', in_channels = 3)
x = torch.rand(16, 3, 32, 32)
print(model(x))
```
