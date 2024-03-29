import torch, time, math, thop, tqdm, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from prettytable import PrettyTable

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv2D(nn.Module):
    def __init__(self, inc, ouc, kernel_size, g=1):
        super().__init__()
        
        self.conv = nn.Conv2d(inc, ouc, kernel_size, padding=autopad(kernel_size), groups=g)
        self.bn = nn.BatchNorm2d(num_features=ouc)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def __str__(self):
        return 'Conv2D'

class DConv2D(nn.Module):
    def __init__(self, inc, ouc, kernel_size):
        super().__init__()
        
        self.pw = Conv2D(inc, ouc, 1)
        self.dw = Conv2D(ouc, ouc, kernel_size, g=ouc)
    
    def forward(self, x):
        return self.dw(self.pw(x))

    def __str__(self):
        return 'Depth-Conv2D'

class GhostConv2D(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = Conv2D(inp, init_channels, kernel_size)
        self.cheap_operation = Conv2D(init_channels, new_channels, dw_size, g=init_channels)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

    def __str__(self):
        return 'Ghost-Conv2D'

class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv2D(c1, c_, k, g)
        self.cv2 = Conv2D(c_, c_, 5, c_)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)
    
    def __str__(self):
        return 'GSConv2D'

class DSConv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, block_size=32, stride=1,
                 padding=None, dilation=1, groups=1, padding_mode='zeros', bias=False, KDSBias=False, CDS=False):
        padding = _pair(autopad(kernel_size, padding, dilation))
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)

        blck_numb = math.ceil(((in_channels)/(block_size*groups)))
        super(DSConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # KDS weight From Paper
        self.intweight = torch.Tensor(out_channels, in_channels, *kernel_size)
        self.alpha = torch.Tensor(out_channels, blck_numb, *kernel_size)

        # KDS bias From Paper
        self.KDSBias = KDSBias
        self.CDS = CDS

        if KDSBias:
            self.KDSb = torch.Tensor(out_channels, blck_numb, *kernel_size)
        if CDS:
            self.CDSw = torch.Tensor(out_channels)
            self.CDSb = torch.Tensor(out_channels)

        self.reset_parameters()

    def get_weight_res(self):
        # Include expansion of alpha and multiplication with weights to include in the convolution layer here
        alpha_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        # Include KDSBias
        if self.KDSBias:
            KDSBias_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        # Handy definitions:
        nmb_blocks = self.alpha.shape[1]
        total_depth = self.weight.shape[1]
        bs = total_depth//nmb_blocks

        llb = total_depth-(nmb_blocks-1)*bs

        # Casting the Alpha values as same tensor shape as weight
        for i in range(nmb_blocks):
            length_blk = llb if i==nmb_blocks-1 else bs

            shp = self.alpha.shape # Notice this is the same shape for the bias as well
            to_repeat=self.alpha[:, i, ...].view(shp[0],1,shp[2],shp[3]).clone()
            repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
            alpha_res[:, i*bs:(i*bs+length_blk), ...] = repeated.clone()

            if self.KDSBias:
                to_repeat = self.KDSb[:, i, ...].view(shp[0], 1, shp[2], shp[3]).clone()
                repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
                KDSBias_res[:, i*bs:(i*bs+length_blk), ...] = repeated.clone()

        if self.CDS:
            to_repeat = self.CDSw.view(-1, 1, 1, 1)
            repeated = to_repeat.expand_as(self.weight)
            print(repeated.shape)

        # Element-wise multiplication of alpha and weight
        weight_res = torch.mul(alpha_res, self.weight)
        if self.KDSBias:
            weight_res = torch.add(weight_res, KDSBias_res)
        return weight_res

    def forward(self, input):
        # Get resulting weight
        #weight_res = self.get_weight_res()

        # Returning convolution
        return F.conv2d(input, self.weight, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)

class DSConv2D(Conv2D):
    def __init__(self, inc, ouc, kernel_size, g=1):
        super().__init__(inc, ouc, kernel_size, g)
        self.conv = DSConv(inc, ouc, kernel_size)
    
    def __str__(self):
        return 'DSConv2D'

class Partial_conv3(nn.Module):
    def __init__(self, dim, kernel_size, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, 1, autopad(kernel_size), bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

class PConv(Conv2D):
    def __init__(self, inc, ouc, kernel_size, g=1):
        super().__init__(inc, ouc, kernel_size, g)
        self.conv = Partial_conv3(inc, kernel_size)
    
    def __str__(self):
        return 'PConv2D-FasterNet'

class DCNV2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, groups=1, act=True, dilation=1, deformable_groups=1):
        super(DCNV2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (autopad(kernel_size, padding), autopad(kernel_size, padding))
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def __str__(self):
        return 'DCNV2'

from ops_dcnv3.modules import DCNv3
class DCNV3(Conv2D):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(inc, ouc, k, g)
        self.conv = DCNv3(inc, kernel_size=k, stride=s, group=g, dilation=d)
    
    def __str__(self):
        return 'DCNV3'

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2)
        return self.act(self.bn(x))
    
if __name__ == '__main__':
    warmup, test_times = 1000, 3000
    bs, h, w = 8, 256, 256
    inc, ouc, kernel_size = 128, 128, 3
    cuda, half = True, True
    module_list = [
                   Conv2D(inc, ouc, kernel_size), 
                   DConv2D(inc, ouc, kernel_size), 
                   GhostConv2D(inc, ouc, kernel_size=1, ratio=2, dw_size=kernel_size), 
                   GSConv(inc, ouc, kernel_size),
                   DSConv2D(inc, ouc, kernel_size),
                   PConv(inc, ouc, kernel_size),
                   DCNV2(inc, ouc, kernel_size),
                   DCNV3(inc, ouc, kernel_size)
                   ]
    
    device = torch.device("cuda:0") if cuda else torch.device("cpu")
    inputs = torch.randn((bs, inc, h, w)).to(device)
    if half:
        inputs = inputs.half()
    table = PrettyTable()
    table.title = 'Conv Family Speed'
    table.field_names = ['Name', 'All_Time', 'Mean_Time', 'FPS', "FLOPs", "Params"]
    for module in module_list:
        module = module.to(device)
        if half:
            module = module.half()
        for i in tqdm.tqdm(range(warmup), desc=f'{str(module)} Warmup....'):
            module(inputs)
        all_time = 0
        for i in tqdm.tqdm(range(test_times), desc=f'{str(module)} Calculate Speed....'):
            begin = time_synchronized()
            module(inputs)
            all_time += time_synchronized() - begin
        FLOPs, Params = thop.profile(module, inputs=(inputs, ), verbose=False)
        FLOPs, Params = thop.clever_format([FLOPs, Params], "%.3f")
        # print(f'{str(module)} all_time:{all_time:.5f} mean_time:{all_time / test_times:.5f} fps:{1 / (all_time / test_times)} FLOPs:{FLOPs} Params:{Params}')
        table.add_row([str(module), f'{all_time:.5f}', f'{all_time / test_times:.5f}', f'{1 / (all_time / test_times)}', f'{FLOPs}', f'{Params}'])
    print(table)