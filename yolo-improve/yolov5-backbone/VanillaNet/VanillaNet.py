#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import weight_init, DropPath
import numpy as np

__all__ = ['vanillanet_5', 'vanillanet_6', 'vanillanet_7', 'vanillanet_8', 'vanillanet_9', 'vanillanet_10', 'vanillanet_11', 'vanillanet_12', 'vanillanet_13', 'vanillanet_13_x1_5', 'vanillanet_13_x1_5_ada_pool']

class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num
        weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x), 
                self.weight, self.bias, padding=(self.act_num*2 + 1)//2, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=self.act_num, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        if not self.deploy:
            kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
            self.weight.data = kernel
            self.bias = torch.nn.Parameter(torch.zeros(self.dim))
            self.bias.data = bias
            self.__delattr__('bn')
            self.deploy = True


class Block(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False, ada_pool=None):
        super().__init__()
        self.act_learn = 1
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        if not ada_pool:
            self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2d((ada_pool, ada_pool))

        self.act = activation(dim_out, act_num)
 
    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            x = self.conv2(x)

        x = self.pool(x)
        x = self.act(x)
        return x

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        if not self.deploy:
            kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
            self.conv1[0].weight.data = kernel
            self.conv1[0].bias.data = bias
            # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
            kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
            self.conv = self.conv2[0]
            self.conv.weight.data = torch.matmul(kernel.transpose(1,3), self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
            self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
            self.__delattr__('conv1')
            self.__delattr__('conv2')
            self.act.switch_to_deploy()
            self.deploy = True
    

class VanillaNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, dims=[96, 192, 384, 768], 
                 drop_rate=0, act_num=3, strides=[2,2,2,1], deploy=False, ada_pool=None, **kwargs):
        super().__init__()
        self.deploy = deploy
        if self.deploy:
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                activation(dims[0], act_num)
            )
        else:
            self.stem1 = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                nn.BatchNorm2d(dims[0], eps=1e-6),
            )
            self.stem2 = nn.Sequential(
                nn.Conv2d(dims[0], dims[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(dims[0], eps=1e-6),
                activation(dims[0], act_num)
            )

        self.act_learn = 1

        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            if not ada_pool:
                stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], deploy=deploy)
            else:
                stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], deploy=deploy, ada_pool=ada_pool[i])
            self.stages.append(stage)
        self.depth = len(strides)

        self.apply(self._init_weights)
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight_init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def change_act(self, m):
        for i in range(self.depth):
            self.stages[i].act_learn = m
        self.act_learn = m

    def forward(self, x):
        input_size = x.size(2)
        scale = [4, 8, 16, 32]
        features = [None, None, None, None]
        if self.deploy:
            x = self.stem(x)
        else:
            x = self.stem1(x)
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            x = self.stem2(x)
        if input_size // x.size(2) in scale:
            features[scale.index(input_size // x.size(2))] = x
        for i in range(self.depth):
            x = self.stages[i](x)
            if input_size // x.size(2) in scale:
                features[scale.index(input_size // x.size(2))] = x
        return features

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        if not self.deploy:
            self.stem2[2].switch_to_deploy()
            kernel, bias = self._fuse_bn_tensor(self.stem1[0], self.stem1[1])
            self.stem1[0].weight.data = kernel
            self.stem1[0].bias.data = bias
            kernel, bias = self._fuse_bn_tensor(self.stem2[0], self.stem2[1])
            self.stem1[0].weight.data = torch.einsum('oi,icjk->ocjk', kernel.squeeze(3).squeeze(2), self.stem1[0].weight.data)
            self.stem1[0].bias.data = bias + (self.stem1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
            self.stem = torch.nn.Sequential(*[self.stem1[0], self.stem2[2]])
            self.__delattr__('stem1')
            self.__delattr__('stem2')

            for i in range(self.depth):
                self.stages[i].switch_to_deploy()

            self.deploy = True

def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict

def vanillanet_5(pretrained='',in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 256*4, 512*4, 1024*4], strides=[2,2,2], **kwargs)
    if pretrained:
        weights = torch.load(pretrained)['model_ema']
        model.load_state_dict(update_weight(model.state_dict(), weights))
    return model

def vanillanet_6(pretrained='',in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[2,2,2,1], **kwargs)
    if pretrained:
        weights = torch.load(pretrained)['model_ema']
        model.load_state_dict(update_weight(model.state_dict(), weights))
    return model

def vanillanet_7(pretrained='',in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,2,1], **kwargs)
    if pretrained:
        weights = torch.load(pretrained)['model_ema']
        model.load_state_dict(update_weight(model.state_dict(), weights))
    return model

def vanillanet_8(pretrained='', in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,1,2,1], **kwargs)
    if pretrained:
        weights = torch.load(pretrained)['model_ema']
        model.load_state_dict(update_weight(model.state_dict(), weights))
    return model

def vanillanet_9(pretrained='', in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,1,1,2,1], **kwargs)
    if pretrained:
        weights = torch.load(pretrained)['model_ema']
        model.load_state_dict(update_weight(model.state_dict(), weights))
    return model

def vanillanet_10(pretrained='', in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,2,1],
        **kwargs)
    if pretrained:
        weights = torch.load(pretrained)['model_ema']
        model.load_state_dict(update_weight(model.state_dict(), weights))
    return model

def vanillanet_11(pretrained='', in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,1,2,1],
        **kwargs)
    if pretrained:
        weights = torch.load(pretrained)['model_ema']
        model.load_state_dict(update_weight(model.state_dict(), weights))
    return model

def vanillanet_12(pretrained='', in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,1,1,2,1],
        **kwargs)
    if pretrained:
        weights = torch.load(pretrained)['model_ema']
        model.load_state_dict(update_weight(model.state_dict(), weights))
    return model

def vanillanet_13(pretrained='', in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,1,1,1,2,1],
        **kwargs)
    if pretrained:
        weights = torch.load(pretrained)['model_ema']
        model.load_state_dict(update_weight(model.state_dict(), weights))
    return model

def vanillanet_13_x1_5(pretrained='', in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*6, 128*6, 256*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 1024*6, 1024*6],
        strides=[1,2,2,1,1,1,1,1,1,2,1],
        **kwargs)
    if pretrained:
        weights = torch.load(pretrained)['model_ema']
        model.load_state_dict(update_weight(model.state_dict(), weights))
    return model

def vanillanet_13_x1_5_ada_pool(pretrained='', in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*6, 128*6, 256*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 1024*6, 1024*6],
        strides=[1,2,2,1,1,1,1,1,1,2,1],
        ada_pool=[0,40,20,0,0,0,0,0,0,10,0],
        **kwargs)
    if pretrained:
        weights = torch.load(pretrained)['model_ema']
        model.load_state_dict(update_weight(model.state_dict(), weights))
    return model

if __name__ == '__main__':
    inputs = torch.randn((1, 3, 640, 640))
    model = vanillanet_10()
    # weights = torch.load('vanillanet_5.pth')['model_ema']
    # model.load_state_dict(update_weight(model.state_dict(), weights))
    pred = model(inputs)
    for i in pred:
        print(i.size())