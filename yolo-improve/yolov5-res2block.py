class Bottle2neck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, shortcut, baseWidth=26, scale = 4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = Conv(inplanes, width*scale, k=1)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        for i in range(self.nums):
          convs.append(Conv(width, width, k=3))
        self.convs = nn.ModuleList(convs)

        self.conv3 = Conv(width*scale, planes * self.expansion, k=1, act=False)

        self.silu = nn.SiLU(inplace=True)
        self.scale = scale
        self.width  = width
        self.shortcut = shortcut

    def forward(self, x):
        print(1)
        if self.shortcut:
            residual = x
        out = self.conv1(x)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1:
          out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        if self.shortcut:
            out += residual
        out = self.silu(out)
        return out

class C3_Res2Block(C3):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottle2neck(c_, c_, shortcut) for _ in range(n)))