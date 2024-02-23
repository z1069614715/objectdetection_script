class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act=True):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1, act=act)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=act)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1, act=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1, act=act)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5, act=act), Conv(c4, c4, 3, 1, act=act))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5, act=act), Conv(c4, c4, 3, 1, act=act))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1, act=act)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

# ------------------------------yolo----------------------------
if hasattr(m, 'fuse_convs'):
    m.fuse_convs()
    m.forward = m.forward_fuse

# ------------------------------yolov7-tiny----------------------------------------
# parameters
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# yolov7-tiny backbone
backbone:
  # [from, number, module, args] c2, k=1, s=1, p=None, g=1, act=True
  [[-1, 1, Conv, [32, 3, 2, None, 1, nn.LeakyReLU(0.1)]],  # 0-P1/2  
  
   [-1, 1, Conv, [64, 3, 2, None, 1, nn.LeakyReLU(0.1)]],  # 1-P2/4    

   [-1, 1, RepNCSPELAN4, [64, 32, 32, 1, nn.LeakyReLU(0.1)]], # 2

   [-1, 1, MP, []],  # 3-P3/8
   [-1, 1, RepNCSPELAN4, [128, 64, 32, 1, nn.LeakyReLU(0.1)]], # 4

   [-1, 1, MP, []],  # 5-P4/16
   [-1, 1, RepNCSPELAN4, [256, 128, 64, 1, nn.LeakyReLU(0.1)]], # 6

   [-1, 1, MP, []],  # 7-P5/32
   [-1, 1, RepNCSPELAN4, [512, 256, 128, 1, nn.LeakyReLU(0.1)]], # 8
  ]

# yolov7-tiny head
head:
  [[-1, 1, Yolov7_Tiny_SPP, [256, nn.LeakyReLU(0.1)]], # 9-Yolov7-tiny-spp
   
   [-1, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]], 
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [6, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],
   [-1, 1, RepNCSPELAN4, [128, 64, 32, 1, nn.LeakyReLU(0.1)]], # 14

   [-1, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [4, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],
   [-1, 1, RepNCSPELAN4,[64, 32, 32, 1, nn.LeakyReLU(0.1)]], # 19
   
   [-1, 1, Conv, [128, 3, 2, None, 1, nn.LeakyReLU(0.1)]],
   [[-1, 14], 1, Concat, [1]],
   [-1, 1, RepNCSPELAN4, [128, 64, 32, 1, nn.LeakyReLU(0.1)]], # 22

   [-1, 1, Conv, [256, 3, 2, None, 1, nn.LeakyReLU(0.1)]],
   [[-1, 9], 1, Concat, [1]],
   [-1, 1, RepNCSPELAN4, [256, 128, 64, 1, nn.LeakyReLU(0.1)]], # 25

   [19, 1, Conv, [128, 3, 1, None, 1, nn.LeakyReLU(0.1)]], # 26-P3
   [22, 1, Conv, [256, 3, 1, None, 1, nn.LeakyReLU(0.1)]], # 27-P4
   [25, 1, Conv, [512, 3, 1, None, 1, nn.LeakyReLU(0.1)]], # 28-P5

   [[26, 27, 28], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]


# -----------------------------yolov7--------------------------------
# parameters
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple

# anchors
anchors:
  - [12,16, 19,36, 40,28]  # P3/8
  - [36,75, 76,55, 72,146]  # P4/16
  - [142,110, 192,243, 459,401]  # P5/32

# yolov7 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [32, 3, 1]],  # 0
  
   [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2      
   [-1, 1, Conv, [64, 3, 1]],
   
   [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4  
   [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]], # 4
         
   [-1, 1, V7DownSampling, [128]],  # 5-P3/8  
   [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]], # 6
         
   [-1, 1, V7DownSampling, [256]],  # 7-P4/16  
   [-1, 1, RepNCSPELAN4, [1024, 512, 256, 1]], # 8
         
   [-1, 1, V7DownSampling, [512]],  # 9-P5/32  
   [-1, 1, RepNCSPELAN4, [1024, 512, 256, 1]],  # 10
  ]

# yolov7 head
head:
  [[-1, 1, SPPCSPC, [512]], # 11

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [8, 1, Conv, [256, 1, 1]], # 14 route backbone P4
   [[-1, -2], 1, Concat, [1]], # 15
   
   [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]], # 16
   
   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [6, 1, Conv, [128, 1, 1]], # 19 route backbone P3
   [[-1, -2], 1, Concat, [1]], # 20
   
   [-1, 1, RepNCSPELAN4, [128, 64, 32, 1]], # 21
      
   [[-1, 16], 1, V7DownSampling_Neck, [128]], # 22
   
   [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]], # 23
      
   [[-1, 11], 1, V7DownSampling_Neck, [256]], # 24
   
   [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]], # 25
   
   [21, 1, RepConv, [256, 3, 1]], # 26-P3
   [23, 1, RepConv, [512, 3, 1]], # 27-P4
   [25, 1, RepConv, [1024, 3, 1]], # 28-P5

   [[26, 27, 28], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]
