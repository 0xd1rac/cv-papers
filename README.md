# cv-papers
This is my feeble attempt at reading and implementing various computer vision papers. Mostly for educational purposes. 

## Table of Contents
- [Resnet](#resnet)
- [ViT](#vit)

## Resnet
<img src="images/resnet.svg" alt="Resnet Blocks" width="500" height="300">

The ResNet <a href="https://arxiv.org/abs/1512.03385"> paper </a> introduces the concept of residual learning, where instead of directly learning the desired mapping, the network learns the residual (difference) between the input and the output. This is formalized as $F(x) = H(x) - x$, $H(x)$ is the desired function and $x$ is the input.

A residual block consists of a series of convolutional layers with a skip connection (or shortcut) that bypasses these layers and adds the input directly to the output. This helps in addressing the vanishing gradient problem and allows for the training of much deeper networks.

You can use it by importing the `resnet` model as shown below:

```python
import torch
from cv_imp import resnet

model = resnet.ResNet152(input_channels=3, num_classes=10)
img = torch.randn(1, 3, 256, 256)
preds = model(img)
print(preds)
print(preds.shape) # torch.Size([1, 10])
```

## ViT 
<img src="images/vit.png" alt="ViT Model" width="700" height="300">

Vision Transformer is an encoder only transformer model adapted for computer vision task. 

Before reading the paper, I went through a few youtube videos and found these to be of a lot of help: 
 1. <a href="https://www.youtube.com/watch?v=TrdevFK_am4"> An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Paper Explained) by Yannic Kilcher 
 </a>

2. <a href="https://www.youtube.com/watch?v=vsqKGZT8Qn8"> Visual Transformer Basics by Samuel Albanie </a>

You can use it by importing the `ViT` model as shown below:

```python
import torch
from cv_imp.vit import ViT
model = ViT(image_size=(512,512),
            patch_size=(32,32),
            num_classes=1000,
            dim = 1024,
            num_transformer_layers = 7,
            num_heads = 16,
            mlp_hidden_dim = 2048,
            pool="cls",
            num_channels=3,
            dropout_proba=0.5,
            emb_dropout_proba=0.5
            )
img = torch.randn(1, 3, 256, 256)

preds = model(img)
print(preds)
print(preds.shape) # torch.Size([1, 1000])


```
