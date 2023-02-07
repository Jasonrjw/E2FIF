from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import pdb

def make_model(args, parent=False):
    return EDSR(args)


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class BinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(BinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.binary_activation = BinaryActivation()


    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        x = self.binary_activation(x)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

class BasicBlock(nn.Module): # head module
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

class ResBlock(nn.Module): # body module 
    def __init__(
        self, n_feats, kernel_size,
        bias=True, res_scale=1):

        super(ResBlock, self).__init__()
        self.move1 = LearnableBias(n_feats)

        self.conv = BinaryConv(n_feats, n_feats, kernel_size, padding=(kernel_size//2,kernel_size//2))

        self.move21 = LearnableBias(n_feats)
        self.prelu = nn.PReLU(n_feats)
        self.move22 = LearnableBias(n_feats)

        self.res_scale = res_scale

    def forward(self, x):
        # res = self.body(x).mul(self.res_scale)
        # res += x
        out = self.move1(x)

        out = self.conv(out)

        out = self.move21(out)
        out = self.prelu(out)
        out = self.move22(out)

        out = out + x
        return out


class EDSR(nn.Module):
    def __init__(self, args): 
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        #print(args.n_feats)
        kernel_size = 3 
        scale = args.scale[0]
        self.need_mid_feas = args.need_mid_feas

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
                n_feats, kernel_size, res_scale=args.res_scale
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
        self.k = 130

    def forward(self, x):
        x = self.sub_mean(x)

        #residual alignment
        x = x * self.k

        x = self.head(x)

        res = self.body(x)
        res += x

        if self.need_mid_feas:
            mid_feas = res
        x = self.tail(res)

        #residual alignment
        x = torch.div(x, self.k)

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

