import torch, time, math, thop, tqdm
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

if __name__ == '__main__':
    warmup, test_times = 1000, 3000
    bs, h, w = 8, 512, 512
    inc, ouc, kernel_size = 128, 256, 3
    cuda, half = True, True
    module_list = [
                   Conv2D(inc, ouc, kernel_size), 
                   DConv2D(inc, ouc, kernel_size), 
                   GhostConv2D(inc, ouc, kernel_size=1, ratio=2, dw_size=kernel_size), 
                   GSConv(inc, ouc, kernel_size),
                   DSConv2D(inc, ouc, kernel_size)
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