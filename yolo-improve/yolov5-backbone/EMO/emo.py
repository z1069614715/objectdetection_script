import math
import numpy as np
import torch.nn as nn
from einops import rearrange, reduce
from timm.models.layers.activations import *
from timm.models.layers import DropPath, trunc_normal_, create_attn
from timm.models.efficientnet_blocks import num_groups, SqueezeExcite as SE
from functools import partial

__all__ = ['EMO_1M', 'EMO_2M', 'EMO_5M', 'EMO_6M']

inplace = True

def get_act(act_layer='relu'):
	act_dict = {
		'none': nn.Identity,
		'sigmoid': Sigmoid,
		'swish': Swish,
		'mish': Mish,
		'hsigmoid': HardSigmoid,
		'hswish': HardSwish,
		'hmish': HardMish,
		'tanh': Tanh,
		'relu': nn.ReLU,
		'relu6': nn.ReLU6,
		'prelu': PReLU,
		'gelu': GELU,
		'silu': nn.SiLU
	}
	return act_dict[act_layer]

class LayerNorm2d(nn.Module):
	
	def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
		super().__init__()
		self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
	
	def forward(self, x):
		x = rearrange(x, 'b c h w -> b h w c').contiguous()
		x = self.norm(x)
		x = rearrange(x, 'b h w c -> b c h w').contiguous()
		return x

def get_norm(norm_layer='in_1d'):
	eps = 1e-6
	norm_dict = {
		'none': nn.Identity,
		'in_1d': partial(nn.InstanceNorm1d, eps=eps),
		'in_2d': partial(nn.InstanceNorm2d, eps=eps),
		'in_3d': partial(nn.InstanceNorm3d, eps=eps),
		'bn_1d': partial(nn.BatchNorm1d, eps=eps),
		'bn_2d': partial(nn.BatchNorm2d, eps=eps),
		'bn_3d': partial(nn.BatchNorm3d, eps=eps),
		'gn': partial(nn.GroupNorm, eps=eps),
		'ln_1d': partial(nn.LayerNorm, eps=eps),
		'ln_2d': partial(LayerNorm2d, eps=eps),
	}
	return norm_dict[norm_layer]

class ConvNormAct(nn.Module):
	
	def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
				 skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
		super(ConvNormAct, self).__init__()
		self.has_skip = skip and dim_in == dim_out
		padding = math.ceil((kernel_size - stride) / 2)
		self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
		self.norm = get_norm(norm_layer)(dim_out)
		self.act = get_act(act_layer)(inplace=inplace)
		self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
	
	def forward(self, x):
		shortcut = x
		x = self.conv(x)
		x = self.norm(x)
		x = self.act(x)
		if self.has_skip:
			x = self.drop_path(x) + shortcut
		return x

inplace = True

# ========== Multi-Scale Populations, for down-sampling and inductive bias ==========
class MSPatchEmb(nn.Module):
	
	def __init__(self, dim_in, emb_dim, kernel_size=2, c_group=-1, stride=1, dilations=[1, 2, 3],
				 norm_layer='bn_2d', act_layer='silu'):
		super().__init__()
		self.dilation_num = len(dilations)
		assert dim_in % c_group == 0
		c_group = math.gcd(dim_in, emb_dim) if c_group == -1 else c_group
		self.convs = nn.ModuleList()
		for i in range(len(dilations)):
			padding = math.ceil(((kernel_size - 1) * dilations[i] + 1 - stride) / 2)
			self.convs.append(nn.Sequential(nn.Conv2d(dim_in, emb_dim, kernel_size, stride, padding, dilations[i], groups=c_group),
				get_norm(norm_layer)(emb_dim),
				get_act(act_layer)(emb_dim)))
	
	def forward(self, x):
		if self.dilation_num == 1:
			x = self.convs[0](x)
		else:
			x = torch.cat([self.convs[i](x).unsqueeze(dim=-1) for i in range(self.dilation_num)], dim=-1)
			x = reduce(x, 'b c h w n -> b c h w', 'mean').contiguous()
		return x


class iRMB(nn.Module):
	def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
				 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=64, window_size=7,
				 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
		super().__init__()
		self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
		dim_mid = int(dim_in * exp_ratio)
		self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
		self.attn_s = attn_s
		if self.attn_s:
			assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
			self.dim_head = dim_head
			self.window_size = window_size
			self.num_head = dim_in // dim_head
			self.scale = self.dim_head ** -0.5
			self.attn_pre = attn_pre
			self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer='none')
			self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
			self.attn_drop = nn.Dropout(attn_drop)
		else:
			if v_proj:
				self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
			else:
				self.v = nn.Identity()
		self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation, groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
		self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()
		
		self.proj_drop = nn.Dropout(drop)
		self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
		self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
	
	def forward(self, x):
		shortcut = x
		x = self.norm(x)
		B, C, H, W = x.shape
		if self.attn_s:
			# padding
			if self.window_size <= 0:
				window_size_W, window_size_H = W, H
			else:
				window_size_W, window_size_H = self.window_size, self.window_size
			pad_l, pad_t = 0, 0
			pad_r = (window_size_W - W % window_size_W) % window_size_W
			pad_b = (window_size_H - H % window_size_H) % window_size_H
			x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
			n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
			x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
			# attention
			b, c, h, w = x.shape
			qk = self.qk(x)
			qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
			q, k = qk[0], qk[1]
			attn_spa = (q @ k.transpose(-2, -1)) * self.scale
			attn_spa = attn_spa.softmax(dim=-1)
			attn_spa = self.attn_drop(attn_spa)
			if self.attn_pre:
				x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ x
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
				x_spa = self.v(x_spa)
			else:
				v = self.v(x)
				v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ v
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
			# unpadding
			x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
			if pad_r > 0 or pad_b > 0:
				x = x[:, :, :H, :W].contiguous()
		else:
			x = self.v(x)

		x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
		
		x = self.proj_drop(x)
		x = self.proj(x)
		
		x = (shortcut + self.drop_path(x)) if self.has_skip else x
		return x


class EMO(nn.Module):
	def __init__(self, dim_in=3, num_classes=1000, img_size=224,
				 depths=[1, 2, 4, 2], stem_dim=16, embed_dims=[64, 128, 256, 512], exp_ratios=[4., 4., 4., 4.],
				 norm_layers=['bn_2d', 'bn_2d', 'bn_2d', 'bn_2d'], act_layers=['relu', 'relu', 'relu', 'relu'],
				 dw_kss=[3, 3, 5, 5], se_ratios=[0.0, 0.0, 0.0, 0.0], dim_heads=[32, 32, 32, 32],
				 window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True], qkv_bias=True,
				 attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False, pre_dim=0):
		super().__init__()
		self.num_classes = num_classes
		assert num_classes > 0
		dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
		self.stage0 = nn.ModuleList([
			MSPatchEmb(  # down to 112
				dim_in, stem_dim, kernel_size=dw_kss[0], c_group=1, stride=2, dilations=[1],
				norm_layer=norm_layers[0], act_layer='none'),
			iRMB(  # ds
				stem_dim, stem_dim, norm_in=False, has_skip=False, exp_ratio=1,
				norm_layer=norm_layers[0], act_layer=act_layers[0], v_proj=False, dw_ks=dw_kss[0],
				stride=1, dilation=1, se_ratio=1,
				dim_head=dim_heads[0], window_size=window_sizes[0], attn_s=False,
				qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=0.,
				attn_pre=attn_pre
			)
		])
		emb_dim_pre = stem_dim
		for i in range(len(depths)):
			layers = []
			dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
			for j in range(depths[i]):
				if j == 0:
					stride, has_skip, attn_s, exp_ratio = 2, False, False, exp_ratios[i] * 2
				else:
					stride, has_skip, attn_s, exp_ratio = 1, True, attn_ss[i], exp_ratios[i]
				layers.append(iRMB(
					emb_dim_pre, embed_dims[i], norm_in=True, has_skip=has_skip, exp_ratio=exp_ratio,
					norm_layer=norm_layers[i], act_layer=act_layers[i], v_proj=True, dw_ks=dw_kss[i],
					stride=stride, dilation=1, se_ratio=se_ratios[i],
					dim_head=dim_heads[i], window_size=window_sizes[i], attn_s=attn_s,
					qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=dpr[j], v_group=v_group,
					attn_pre=attn_pre
				))
				emb_dim_pre = embed_dims[i]
			self.__setattr__(f'stage{i + 1}', nn.ModuleList(layers))
		
		self.norm = get_norm(norm_layers[-1])(embed_dims[-1])
		self.apply(self._init_weights)
		self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
	
	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if m.bias is not None:
				nn.init.zeros_(m.bias)
		elif isinstance(m, (nn.LayerNorm, nn.GroupNorm,
							nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
							nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
			nn.init.zeros_(m.bias)
			nn.init.ones_(m.weight)
	
	@torch.jit.ignore
	def no_weight_decay(self):
		return {'token'}
	
	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'alpha', 'gamma', 'beta'}
	
	@torch.jit.ignore
	def no_ft_keywords(self):
		# return {'head.weight', 'head.bias'}
		return {}
	
	@torch.jit.ignore
	def ft_head_keywords(self):
		return {'head.weight', 'head.bias'}, self.num_classes
	
	def get_classifier(self):
		return self.head
	
	def reset_classifier(self, num_classes):
		self.num_classes = num_classes
		self.head = nn.Linear(self.pre_dim, num_classes) if num_classes > 0 else nn.Identity()
	
	def check_bn(self):
		for name, m in self.named_modules():
			if isinstance(m, nn.modules.batchnorm._NormBase):
				m.running_mean = torch.nan_to_num(m.running_mean, nan=0, posinf=1, neginf=-1)
				m.running_var = torch.nan_to_num(m.running_var, nan=0, posinf=1, neginf=-1)
	
	def forward_features(self, x):
		for blk in self.stage0:
			x = blk(x)
		x1 = x
		for blk in self.stage1:
			x = blk(x)
		x2 = x
		for blk in self.stage2:
			x = blk(x)
		x3 = x
		for blk in self.stage3:
			x = blk(x)
		x4 = x
		for blk in self.stage4:
			x = blk(x)
		x5 = x
		return [x1, x2, x3, x4, x5]
	
	def forward(self, x):
		x = self.forward_features(x)
		x[-1] = self.norm(x[-1])
		return x

def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict

def EMO_1M(weights='', **kwargs):
	model = EMO(
		# dim_in=3, num_classes=1000, img_size=224,
		depths=[2, 2, 8, 3], stem_dim=24, embed_dims=[32, 48, 80, 168], exp_ratios=[2., 2.5, 3.0, 3.5],
		norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
		dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 20, 21], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
		qkv_bias=True, attn_drop=0., drop=0., drop_path=0.04036, v_group=False, attn_pre=True, pre_dim=0,
		**kwargs)
	if weights:
		pretrained_weight = torch.load(weights)
		model.load_state_dict(update_weight(model.state_dict(), pretrained_weight))
	return model

def EMO_2M(weights='', **kwargs):
	model = EMO(
		# dim_in=3, num_classes=1000, img_size=224,
		depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[32, 48, 120, 200], exp_ratios=[2., 2.5, 3.0, 3.5],
		norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
		dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 20, 20], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
		qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
		**kwargs)
	if weights:
		pretrained_weight = torch.load(weights)
		model.load_state_dict(update_weight(model.state_dict(), pretrained_weight))
	return model

def EMO_5M(weights='', **kwargs):
	model = EMO(
		# dim_in=3, num_classes=1000, img_size=224,
		depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 288], exp_ratios=[2., 3., 4., 4.],
		norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
		dw_kss=[3, 3, 5, 5], dim_heads=[24, 24, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
		qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
		**kwargs)
	if weights:
		pretrained_weight = torch.load(weights)
		model.load_state_dict(update_weight(model.state_dict(), pretrained_weight))
	return model

def EMO_6M(weights='', **kwargs):
	model = EMO(
		# dim_in=3, num_classes=1000, img_size=224,
		depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
		norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
		dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
		qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
		**kwargs)
	if weights:
		pretrained_weight = torch.load(weights)
		model.load_state_dict(update_weight(model.state_dict(), pretrained_weight))
	return model

if __name__ == '__main__':
    model = EMO_1M('EMO_1M/net.pth')
    model = EMO_2M('EMO_2M/net.pth')
    model = EMO_5M('EMO_5M/net.pth')
    model = EMO_6M('EMO_6M/net.pth')