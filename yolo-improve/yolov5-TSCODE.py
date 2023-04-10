
from einops import rearrange
class TSCODE_Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m_sce = nn.ModuleList(SCE(ch[id:id+2]) for id in range(1, len(ch) - 1))
        self.m_dpe = nn.ModuleList(DPE(ch[id-1:id+2], ch[id]) for id in range(1, len(ch) - 1))
        
        self.m_cls = nn.ModuleList(nn.Sequential(Conv(sum(ch[id:id+2]), ch[id], 1), Conv(ch[id], ch[id], 3), nn.Conv2d(ch[id], self.na * self.nc * 4, 1)) for id in range(1, len(ch) - 1))  # cls conv
        self.m_reg_conf = nn.ModuleList(nn.Sequential(*[Conv(ch[id], ch[id], 3) for i in range(2)]) for id in range(1, len(ch) - 1))  # reg_conf stem conv
        self.m_reg = nn.ModuleList(nn.Conv2d(ch[id], self.na * 4, 1) for id in range(1, len(ch) - 1))  # reg conv
        self.m_conf = nn.ModuleList(nn.Conv2d(ch[id], self.na * 1, 1) for id in range(1, len(ch) - 1))  # conf conv
        self.ph, self.pw = 2, 2
        
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x_):
        x, z = [], []  # inference output
        for i, idx in enumerate(range(1, self.nl + 1)):
            bs, _, ny, nx = x_[idx].shape
            
            x_sce, x_dpe = self.m_sce[i](x_[idx:idx+2]), self.m_dpe[i](x_[idx-1:idx+2])
            x_cls = rearrange(self.m_cls[i](x_sce), 'bs (nl ph pw nc) h w -> bs nl nc (h ph) (w pw)', nl=self.nl, ph=self.ph, pw=self.pw, nc=self.nc)
            x_cls = x_cls.permute(0, 1, 3, 4, 2).contiguous()
            
            x_reg_conf = self.m_reg_conf[i](x_dpe)
            x_reg = self.m_reg[i](x_reg_conf).view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_conf = self.m_conf[i](x_reg_conf).view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x.append(torch.cat([x_reg, x_conf, x_cls], dim=4))
        
            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    
class Decoupled_Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        
        self.m_stem = nn.ModuleList(Conv(x, x, 1) for x in ch)  # stem conv
        self.m_cls = nn.ModuleList(nn.Sequential(Conv(x, x, 3), nn.Conv2d(x, self.na * self.nc, 1)) for x in ch)  # cls conv
        self.m_reg_conf = nn.ModuleList(Conv(x, x, 3) for x in ch)  # reg_conf stem conv
        self.m_reg = nn.ModuleList(nn.Conv2d(x, self.na * 4, 1) for x in ch)  # reg conv
        self.m_conf = nn.ModuleList(nn.Conv2d(x, self.na * 1, 1) for x in ch)  # conf conv
        
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m_stem[i](x[i])  # conv
            
            bs, _, ny, nx = x[i].shape
            x_cls = self.m_cls[i](x[i]).view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_reg_conf = self.m_reg_conf[i](x[i])
            x_reg = self.m_reg[i](x_reg_conf).view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_conf = self.m_conf[i](x_reg_conf).view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x[i] = torch.cat([x_reg, x_conf, x_cls], dim=4)

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
    # https://arxiv.org/abs/1708.02002 section 3.3
    # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    m = self.model[-1]  # Detect() module
    
    if isinstance(m, Detect):
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    elif isinstance(m, Decoupled_Detect) or isinstance(m, TSCODE_Detect):
        for mi, s in zip(m.m_conf, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for mi, s in zip(m.m_cls, m.stride):  # from
            b = mi[-1].bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

### Task-Specific Context Decoupling for Object Detection

class SCE(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.down = Conv(c1[0], c1[0], k=3, s=2)
        
    def forward(self, x):
        x_p1, x_p2 = x
        x = torch.concat([self.down(x_p1), x_p2], dim=1)
        return x

class DPE(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.adjust_channel_forp1 = Conv(c1[0], c2, k=1)
        self.adjust_channel_forp2 = Conv(c1[1], c2, k=1)
        
        self.up_forp2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(c2, c2, k=1)
        )
        self.up_forp3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(c1[2], c2, k=1)
        )
        self.down = Conv(c2, c2, k=3, s=2)
        self.middle = Conv(c2, c2, k=1)
        
    def forward(self, x):
        x_p2 = self.adjust_channel_forp2(x[1])
        x_p1 = self.adjust_channel_forp1(x[0]) + self.up_forp2(x_p2)
        x_p1 = self.down(x_p1)
        
        x_p3 = self.up_forp3(x[2])
        
        return x_p1 + x_p2 + x_p3

#### yolov5-FPN-TSCODE
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
  [[-1, 1, Conv, [512, 1, 1]], # 10
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 11
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4 12
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]], # 14
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 15
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3 16
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [9, 1, Conv, [1024, 3, 2]], # 18-P6/64
   [-1, 3, C3, [1024]], # 19

   [[2, 17, 13, 10, 19], 1, TSCODE_Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]


#### yolov5-PFPN-TSCODE
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
  [[-1, 1, Conv, [512, 1, 1]], # 10
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 11
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4 12
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]], # 14
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 15
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3 16
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]], # 18
   [[-1, 14], 1, Concat, [1]],  # cat head P4 19
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]], # 21
   [[-1, 10], 1, Concat, [1]],  # cat head P5 # 22
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [9, 1, Conv, [1024, 3, 2]], # 24-P6/64
   [-1, 3, C3, [1024]], # 25

   [[2, 17, 20, 23, 25], 1, TSCODE_Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
