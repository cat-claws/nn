# Neural network models in PyTorch (Don't fear making copies!)
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

A few models that work on MNIST
```python
model = torch.hub.load('cat-claws/nn', 'mlp', hidden = [120, 84])
model = torch.hub.load('cat-claws/nn', 'simplecnn', convs = [], linears = [784, 120, 84], pretrained = 'mlp_784_120_84_GdyC')
# note that the two above are actually the same networks in different coding style

model = torch.hub.load('cat-claws/nn', 'simplecnn', convs = [ (1, 16, 5), (16, 24, 5) ], linears = [24*4*4, 100], pretrained = 'simplecnn_5_16_24_100_ebyC')
model = torch.hub.load('cat-claws/nn', 'simplecnn', convs = [ (1, 10, 5), (10, 20, 5) ], linears = [320, 50], pretrained = 'simplecnn_5_10_20_50_ibyC')
model = torch.hub.load('cat-claws/nn', 'simplecnn', convs = [ (1, 32, 5, 1, 2),  (32, 64, 5, 1, 2)], linears = [64*7*7, 1024], pretrained = 'simplecnn_5_32_64_1024_dbyC')
model = torch.hub.load('cat-claws/nn', 'exampleconvnet', in_channels = 1, pretrained = 'exampleconvnet_cbyC')
```

A few face feature extraction models
```python
model = torch.hub.load('cat-claws/nn', 'lightcnn', layers = [1, 2, 3, 4], pretrained = 'lightcnn29')
model = torch.hub.load('cat-claws/nn', 'lightcnn_facexzoo', depth = 29, drop_ratio = 0.2, out_h = 7, out_w = 7, feat_dim = 512, pretrained = 'lightcnn_facexzoo')
# the two models above are supposed to be the same model

model = torch.hub.load('cat-claws/nn', 'inception_resnet_v1', num_classes = 8631, pretrained = 'inceptionresnetv1_vggface2')
model = torch.hub.load('cat-claws/nn', 'iresnet', layers = [3, 4, 14, 3], pretrained = 'inceptionresnetv1_vggface2')
model = torch.hub.load('cat-claws/nn', 'efficientnet_facexzoo', out_h = 7, out_w = 7, feat_dim = 512, pretrained = 'efficientnet_facexzoo')
model = torch.hub.load('cat-claws/nn', 'ghostnet_facexzoo', out_h = 7, out_w = 7, feat_dim = 512, pretrained = 'ghostnet_facexzoo')
model = torch.hub.load('cat-claws/nn', 'tf_nas_facexzoo',  out_h = 7, out_w = 7, feat_dim = 512, pretrained = 'tfnas_facexzoo')
model = torch.hub.load('cat-claws/nn', 'attentionnet_facexzoo', stage1_modules = 1, stage2_modules = 2, stage3_modules = 3,  out_h = 7, out_w = 7, feat_dim = 512, pretrained = 'attentionnet_facexzoo')
model = torch.hub.load('cat-claws/nn', 'rexnet_facexzoo', use_se=False, pretrained = 'rexnet_facexzoo')
model = torch.hub.load('cat-claws/nn', 'repvgg_facexzoo', num_blocks = [2, 4, 14, 1], width_multiplier = [0.75, 0.75, 0.75, 2.5], pretrained = 'repvgg_facexzoo')
model = torch.hub.load('cat-claws/nn', 'mobilefacenet_facexzoo', embedding_size = 512, out_h = 7, out_w = 7) # unfortunately, I did not get pretrained weights
```
