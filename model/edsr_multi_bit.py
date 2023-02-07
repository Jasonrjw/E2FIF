from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import pdb
from utils.quant_ops_new import QuantizeConv


def make_model(args, parent=False):
    return EDSR(args)


class BasicBlock(nn.Module):  #用于head
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, bn=True):

        super(BasicBlock, self).__init__()
        self.residual = in_channels == out_channels
        m = [conv(in_channels, out_channels, kernel_size, padding=(kernel_size//2,kernel_size//2), bias=False if bn else True)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        self.conv1 = nn.Sequential(*m)

    def forward(self, x):
        if self.residual:
            return self.conv1(x) + x
        else:
            return self.conv1(x)

class ResBlock(nn.Module): #用于body
    def __init__(
        self, conv, n_feats, kernel_size, weight_bits, input_bits, learnable,
        symmetric1, symmetric2, weight_layerwise, input_layerwise,
        weight_quant_method, input_quant_method1, input_quant_method2,
        bias=True, act=functools.partial(nn.LeakyReLU, negative_slope=0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        #pdb.set_trace()
        self.conv1 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, padding=(kernel_size//2,kernel_size//2), bias=bias,
                weight_bits=weight_bits, input_bits=input_bits, 
                learnable=learnable, symmetric=symmetric1, 
                weight_layerwise=weight_layerwise, input_layerwise=input_layerwise,
                weight_quant_method=weight_quant_method, input_quant_method=input_quant_method1),
            nn.BatchNorm2d(n_feats)
        )
        self.act = act() #used for LeakyReLU
        #self.act = act  #used for PReLU
        self.conv2 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, padding=(kernel_size//2,kernel_size//2), bias=bias,
                weight_bits=weight_bits, input_bits=input_bits, 
                learnable=learnable, symmetric=symmetric2, 
                weight_layerwise=weight_layerwise, input_layerwise=input_layerwise,
                weight_quant_method=weight_quant_method, input_quant_method=input_quant_method2),
            nn.BatchNorm2d(n_feats)
        )
        self.res_scale = res_scale

    def forward(self, x):
        # res = self.body(x).mul(self.res_scale)
        # res += x
        out = self.conv1(x).mul(self.res_scale) + x
        out = self.act(out)
        out = self.conv2(out).mul(self.res_scale) + out
        return out


class EDSR(nn.Module):
    def __init__(self, args, conv=QuantizeConv): #这里定义了conv的类型
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        #print(args.n_feats)
        kernel_size = 3 
        scale = args.scale[0]
        self.need_mid_feas = args.need_mid_feas

        weight_bits = args.weight_bits
        input_bits = args.input_bits
        learnable = args.learnable
        symmetric1 = args.symmetric1
        symmetric2 = args.symmetric2
        weight_layerwise = args.weight_layerwise
        input_layerwise = args.input_layerwise
        weight_quant_method = args.weight_quant_method
        input_quant_method1 = args.input_quant_method1
        input_quant_method2 = args.input_quant_method2

        act = functools.partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)
        #act = nn.PReLU()

        # url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        # if url_name in url:
        #     self.url = url[url_name]
        # else:
        #     self.url = None
        if args.n_colors == 3:
            self.sub_mean = common.MeanShift(args.rgb_range)
            self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        else:
            self.sub_mean = common.MeanShift(args.rgb_range, n_colors=1, rgb_mean=[0.5], rgb_std=[1])
            self.add_mean = common.MeanShift(args.rgb_range, n_colors=1, rgb_mean=[0.5], rgb_std=[1], sign=1)

        # define head module
        m_head = [BasicBlock(nn.Conv2d, args.n_colors, n_feats, kernel_size, bn=False)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale,
                weight_bits=weight_bits, input_bits=input_bits, learnable=learnable,
                symmetric1 = symmetric1, symmetric2=symmetric2, 
                weight_layerwise = weight_layerwise, input_layerwise=input_layerwise,
                weight_quant_method = weight_quant_method, input_quant_method1=input_quant_method1,
                input_quant_method2=input_quant_method2
                ) for _ in range(n_resblocks)
        ]
        # m_body.append(nn.Sequential(
        #     conv(n_feats, n_feats, kernel_size, padding=kernel_size//2),
        #     nn.BatchNorm2d(n_feats)
        # ))

        # define tail module
        m_tail = [
            nn.Conv2d(n_feats, args.n_colors * scale**2, 3, padding=1),
            nn.PixelShuffle(scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.need_mid_feas:
            mid_feas = res
        x = self.tail(res)
        x = self.add_mean(x)
        if self.need_mid_feas:
            return (x,mid_feas)
        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

