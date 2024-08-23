# cv-papers
This is my feeble attempt at reading and implementing various computer vision papers. Mostly for educational purposes. 

## Table of Contents
- [Resnet](#resnet)

## Resnet
The ResNet <a href="https://arxiv.org/abs/1512.03385"> paper </a> introduces the concept of residual learning, where instead of directly learning the desired mapping, the network learns the residual (difference) between the input and the output. This is formalized as $ F(x) = H(x) - x$, $H(x)$ is the desired function and x is the input.

A residual block consists of a series of convolutional layers with a skip connection (or shortcut) that bypasses these layers and adds the input directly to the output. This helps in addressing the vanishing gradient problem and allows for the training of much deeper networks
You can use it by importing the `resnet` as shown below

```python
import torch
from cv_imp import resnet

model = resnet.ResNet152(input_channels=3, num_classes=10)
img = torch.randn(1, 3, 256, 256)
preds = model(img)
print(preds)
print(preds.shape) # torch.Size([1, 10])
```
