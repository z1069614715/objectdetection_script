from models.ops_dcnv3.modules import DCNv3
class DCNV3_YoLo(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        
        self.conv = Conv(inc, ouc, k=1)
        self.dcnv3 = DCNv3(ouc, kernel_size=k, stride=s, group=g, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.dcnv3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.bn(x))
        return x

if isinstance(m, Detect):
    s = 256  # 2x min stride
    self.model.to(torch.device('cuda'))
    m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s).to(torch.device('cuda')))]).cpu()  # forward
    self.model.cpu()
    check_anchor_order(m)
    m.anchors /= m.stride.view(-1, 1, 1)
    self.stride = m.stride
    self._initialize_biases()  # only run once
    # print('Strides: %s' % m.stride.tolist())
if isinstance(m, IDetect):
    s = 256  # 2x min stride
    self.model.to(torch.device('cuda'))
    m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s).to(torch.device('cuda')))]).cpu()  # forward
    self.model.cpu()
    check_anchor_order(m)
    m.anchors /= m.stride.view(-1, 1, 1)
    self.stride = m.stride
    self._initialize_biases()  # only run once
    # print('Strides: %s' % m.stride.tolist())
if isinstance(m, IAuxDetect):
    s = 256  # 2x min stride
    self.model.to(torch.device('cuda'))
    m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s).to(torch.device('cuda')))[:4]]).cpu()  # forward
    self.model.cpu()
    #print(m.stride)
    check_anchor_order(m)
    m.anchors /= m.stride.view(-1, 1, 1)
    self.stride = m.stride
    self._initialize_aux_biases()  # only run once
    # print('Strides: %s' % m.stride.tolist())