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
   
   [-1, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]], 
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [6, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]], # route backbone P4
   [-1, 1, EVCBlock, []],
   [[-1, -2], 1, Concat, [1]],
   [-1, 1, Yolov7_Tiny_E_ELAN, [128, 64, nn.LeakyReLU(0.1)]], # 15

   [-1, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [4, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],
   [-1, 1, Yolov7_Tiny_E_ELAN, [64, 32, nn.LeakyReLU(0.1)]], # 20
   
   [-1, 1, Conv, [128, 3, 2, None, 1, nn.LeakyReLU(0.1)]],
   [[-1, 15], 1, Concat, [1]],
   [-1, 1, Yolov7_Tiny_E_ELAN, [128, 64, nn.LeakyReLU(0.1)]], # 23
   
   [-1, 1, Conv, [256, 3, 2, None, 1, nn.LeakyReLU(0.1)]],
   [[-1, 9], 1, Concat, [1]],
   
   [-1, 1, Yolov7_Tiny_E_ELAN, [256, 128, nn.LeakyReLU(0.1)]], # 26

   [20, 1, Conv, [128, 3, 1, None, 1, nn.LeakyReLU(0.1)]], # 27-P3
   [23, 1, Conv, [256, 3, 1, None, 1, nn.LeakyReLU(0.1)]], # 28-P4
   [26, 1, Conv, [512, 3, 1, None, 1, nn.LeakyReLU(0.1)]], # 29-P5

   [[27, 28, 29], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]