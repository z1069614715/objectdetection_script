from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init
class ContextAggregation(nn.Module):
    """
    Context Aggregation Block.

    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """

    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        conv_params = dict(kernel_size=1, act_cfg=None)

        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)

    def forward(self, x):
        n, c = x.size(0), self.inter_channels

        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a

        return x + y


# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.25  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [17, 1, ContextAggregation, []], # 24
   [20, 1, ContextAggregation, []], # 25
   [23, 1, ContextAggregation, []], # 26

   [[24, 25, 26], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
