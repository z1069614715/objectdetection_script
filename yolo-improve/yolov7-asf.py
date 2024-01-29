import torch.nn.functional as F
class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms

class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(ScalSeq, self).__init__()
        self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 =  Conv(inc[1], channel,1)
        self.conv2 =  Conv(inc[2], channel,1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d,p4_3d,p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x
    
class Add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self):
        super().__init__()

    def forward(self, x):
        input1,input2 = x[0],x[1]
        x = input1 + input2
        return x

class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()
        
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu   = nn.ReLU()
        self.bn     = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
        _, _, h, w = x.size()
        
        x_h = torch.mean(x, dim = 3, keepdim = True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim = 2, keepdim = True)
 
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
    
class attention_model(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch = 256):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)
    def forward(self, x):
        input1,input2 = x[0],x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x

elif m is Zoom_cat:
    c2 = sum(ch[x] for x in f)
elif m is Add:
    c2 = ch[f[-1]]
elif m is attention_model:
    c2 = ch[f[-1]]
    args = [c2]
elif m is ScalSeq:
    c1 = [ch[x] for x in f]
    c2 = make_divisible(args[0] * gw, 8)
    args = [c1, c2]

##################################################### YOLOV7-TINY #####################################################
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

   [-1, 1, Yolov7_Tiny_E_ELAN, [64, 32, nn.LeakyReLU(0.1)]], # 2

   [-1, 1, MP, []],  # 3-P3/8
   [-1, 1, Yolov7_Tiny_E_ELAN, [128, 64, nn.LeakyReLU(0.1)]], # 4

   [-1, 1, MP, []],  # 5-P4/16
   [-1, 1, Yolov7_Tiny_E_ELAN, [256, 128, nn.LeakyReLU(0.1)]], # 6

   [-1, 1, MP, []],  # 7-P5/32
   [-1, 1, Yolov7_Tiny_E_ELAN, [512, 256, nn.LeakyReLU(0.1)]], # 8
  ]

# yolov7-tiny head
head:
  [[-1, 1, Yolov7_Tiny_SPP, [256, nn.LeakyReLU(0.1)]], # 9-Yolov7-tiny-spp
   
   [-1, 1, Conv, [256, 1, 1, None, 1, nn.LeakyReLU(0.1)]], 
   [4, 1, Conv, [256, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
   [[-1, 6, -2], 1, Zoom_cat, []], # route backbone P4
   [-1, 1, Yolov7_Tiny_E_ELAN, [128, 64, nn.LeakyReLU(0.1)]], # 13

   [-1, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
   [2, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]], # 15
   [[-1, 4, -2], 1, Zoom_cat, []],
   [-1, 1, Yolov7_Tiny_E_ELAN, [64, 32, nn.LeakyReLU(0.1)]], # 17
   
   [-1, 1, Conv, [128, 3, 2, None, 1, nn.LeakyReLU(0.1)]], # 18
   [[-1, 13], 1, Concat, [1]],
   [-1, 1, Yolov7_Tiny_E_ELAN, [128, 64, nn.LeakyReLU(0.1)]], # 20

   [-1, 1, Conv, [256, 3, 2, None, 1, nn.LeakyReLU(0.1)]],
   [[-1, 9], 1, Concat, [1]],
   [-1, 1, Yolov7_Tiny_E_ELAN, [256, 128, nn.LeakyReLU(0.1)]], # 23

   [[4, 6, 8], 1, ScalSeq, [64]], #24 args[inchane]
   [[17, -1], 1, attention_model, []], #25

   [25, 1, Conv, [128, 3, 1, None, 1, nn.LeakyReLU(0.1)]], # 26-P3
   [23, 1, Conv, [256, 3, 1, None, 1, nn.LeakyReLU(0.1)]], # 27-P4
   [20, 1, Conv, [512, 3, 1, None, 1, nn.LeakyReLU(0.1)]], # 28-P5

   [[26,27,28], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]


##################################################### YOLOV7 #####################################################
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

   [-1, 1, Conv, [1024, 1, 1, None, 1, nn.LeakyReLU(0.1)]], 
   [6, 1, Conv, [1024, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
   [[-1, 8, -2], 1, Zoom_cat, []], # route backbone P4
   [-1, 1, Yolov7_E_ELAN_NECK, [256, 128]], # 15
   
   [-1, 1, Conv, [512, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
   [4, 1, Conv, [512, 1, 1, None, 1, nn.LeakyReLU(0.1)]], # 17
   [[-1, 6, -2], 1, Zoom_cat, []], # 18
   [-1, 1, Yolov7_E_ELAN_NECK, [128, 64]], # 19
      
   [[-1, 15], 1, V7DownSampling_Neck, [128]], # 20
   
   [-1, 1, Yolov7_E_ELAN_NECK, [256, 128]], # 21
      
   [[-1, 11], 1, V7DownSampling_Neck, [256]], # 22
   
   [-1, 1, Yolov7_E_ELAN_NECK, [512, 256]], # 23
   
   [[6, 8, 10], 1, ScalSeq, [128]], #24 args[inchane]
   [[19, -1], 1, attention_model, []], #25

   [25, 1, RepConv, [256, 3, 1]], # 26-P3
   [21, 1, RepConv, [512, 3, 1]], # 27-P4
   [23, 1, RepConv, [1024, 3, 1]], # 28-P5

   [[26, 27, 28], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]
