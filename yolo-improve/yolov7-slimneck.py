class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    # act参数在yolov7-tiny上记得修改为nn.LeakyReLU(0.1)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2*e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)

class GSBottleneckC(GSBottleneck):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, k, s, act=False)

class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)  #


    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))

class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2)
        c_ = int(c2 * 0.5)  # hidden channels
        self.gsb = GSBottleneckC(c_, c_, 1, 1)


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

   [-1, 1, GSConv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [8, 1, GSConv, [256, 1, 1]], # 14 route backbone P4
   [[-1, -2], 1, Concat, [1]], # 15
   
   [-1, 1, VoVGSCSP, [256]], # 16
   
   [-1, 1, GSConv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [6, 1, GSConv, [128, 1, 1]], # 19 route backbone P3
   [[-1, -2], 1, Concat, [1]], # 20
   
   [-1, 1, VoVGSCSP, [128]], # 21
      
   [[-1, 16], 1, V7DownSampling_Neck, [128]], # 22
   
   [-1, 1, VoVGSCSP, [256]], # 23
      
   [[-1, 11], 1, V7DownSampling_Neck, [256]], # 24
   
   [-1, 1, VoVGSCSP, [512]], # 25
   
   [21, 1, RepConv, [256, 3, 1]], # 26-P3
   [23, 1, RepConv, [512, 3, 1]], # 27-P4
   [25, 1, RepConv, [1024, 3, 1]], # 28-P5

   [[26, 27, 28], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]