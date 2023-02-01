import torch
from torch import nn as nn
from timm.models.layers.create_act import create_act_layer


class EffectiveSEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False, gate_layer='hard_sigmoid'):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    Ese = EffectiveSEModule(512)
    output=Ese(input)
    print(output.shape)