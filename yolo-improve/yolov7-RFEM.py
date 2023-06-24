class TridentBlock(nn.Module):
    def __init__(self, c1, c2, stride=1, c=False, e=0.5, padding=[1, 2, 3], dilate=[1, 2, 3], bias=False):
        super(TridentBlock, self).__init__()
        self.stride = stride
        self.c = c
        c_ = int(c2 * e)
        self.padding = padding
        self.dilate = dilate
        self.share_weightconv1 = nn.Parameter(torch.Tensor(c_, c1, 1, 1))
        self.share_weightconv2 = nn.Parameter(torch.Tensor(c2, c_, 3, 3))

        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)

        self.act = nn.SiLU()

        nn.init.kaiming_uniform_(self.share_weightconv1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.share_weightconv2, nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.Tensor(c2))
        else:
            self.bias = None

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward_for_small(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[0],
                                   dilation=self.dilate[0])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_middle(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[1],
                                   dilation=self.dilate[1])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_big(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[2],
                                   dilation=self.dilate[2])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward(self, x):
        xm = x
        base_feat = []
        if self.c is not False:
            x1 = self.forward_for_small(x)
            x2 = self.forward_for_middle(x)
            x3 = self.forward_for_big(x)
        else:
            x1 = self.forward_for_small(xm[0])
            x2 = self.forward_for_middle(xm[1])
            x3 = self.forward_for_big(xm[2])

        base_feat.append(x1)
        base_feat.append(x2)
        base_feat.append(x3)

        return base_feat

class RFEM(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, stride=1):
        super(RFEM, self).__init__()
        c = True
        layers = []
        layers.append(TridentBlock(c1, c2, stride=stride, c=c, e=e))
        c1 = c2
        for i in range(1, n):
            layers.append(TridentBlock(c1, c2))
        self.layer = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.layer(x)
        out = out[0] + out[1] + out[2] + x
        out = self.act(self.bn(out))
        return out

# Yolov7-REFM
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
   [-1, 1, Yolov7_E_ELAN, [256, 64]], # 4
         
   [-1, 1, V7DownSampling, [128]],  # 5-P3/8  
   [-1, 1, Yolov7_E_ELAN, [512, 128]], # 6
         
   [-1, 1, V7DownSampling, [256]],  # 7-P4/16  
   [-1, 1, Yolov7_E_ELAN, [1024, 256]], # 8
         
   [-1, 1, V7DownSampling, [512]],  # 9-P5/32  
   [-1, 1, Yolov7_E_ELAN, [1024, 256]],  # 10
  ]

# yolov7 head
head:
  [[-1, 1, SPPCSPC, [512]], # 11
   [-1, 1, RFEM, [512]], # 12

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [8, 1, Conv, [256, 1, 1]], # 15 route backbone P4
   [[-1, -2], 1, Concat, [1]], # 16
   
   [-1, 1, Yolov7_E_ELAN_NECK, [256, 128]], # 17
   
   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [6, 1, Conv, [128, 1, 1]], # 20 route backbone P3
   [[-1, -2], 1, Concat, [1]], # 21
   
   [-1, 1, Yolov7_E_ELAN_NECK, [128, 64]], # 22
      
   [[-1, 17], 1, V7DownSampling_Neck, [128]], # 23
   
   [-1, 1, Yolov7_E_ELAN_NECK, [256, 128]], # 24
      
   [[-1, 12], 1, V7DownSampling_Neck, [256]], # 25
   
   [-1, 1, Yolov7_E_ELAN_NECK, [512, 256]], # 26
   
   [22, 1, RepConv, [256, 3, 1]], # 27-P3
   [24, 1, RepConv, [512, 3, 1]], # 28-P4
   [26, 1, RepConv, [1024, 3, 1]], # 29-P5

   [[27, 28, 29], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
]
