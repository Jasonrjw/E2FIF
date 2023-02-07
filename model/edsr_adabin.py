from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import pdb

def make_model(args, parent=False):
    return EDSR(args)


class Maxout(nn.Module):
    '''
        activation function
    '''
    def __init__(self, channel, neg_init=0.25, pos_init=1.0):
        super(Maxout, self).__init__()
        self.neg_scale = nn.Parameter(neg_init*torch.ones(channel))
        self.pos_scale = nn.Parameter(pos_init*torch.ones(channel))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Maxout
        x = self.pos_scale.view(1,-1,1,1)*self.relu(x) - self.neg_scale.view(1,-1,1,1)*self.relu(-x)
        return x


class BinaryActivation(nn.Module):
    '''
        learnable distance and center for activation
    '''
    def __init__(self):
        super(BinaryActivation, self).__init__() 
        self.alpha_a = nn.Parameter(torch.tensor(1.0))
        self.beta_a = nn.Parameter(torch.tensor(0.0))
    
    def gradient_approx(self, x):
        '''
            from Bi-Real Net
            (https://github.com/liuzechun/Bi-Real-net/blob/master/pytorch_implementation/BiReal18_34/birealnet.py)
        '''
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out
        
    def forward(self, x): 
        x = (x-self.beta_a)/self.alpha_a
        x = self.gradient_approx(x)
        return self.alpha_a*x + self.beta_a


class Q_W(torch.autograd.Function):  
    '''
        binary quantize function
        (https://github.com/htqin/IR-Net/blob/master/CIFAR-10/ResNet20/1w1a/modules/binaryfunction.py)
    ''' 

    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k, t = k.cuda(), t.cuda() 
        grad_input = k * t * (1-torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class BinaryConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, bitW=1, stride=1, padding=0, bias=True, groups=1, mode='binary'):
        super(BinaryConv, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.groups = groups
        self.bitW = bitW
        self.padding = padding
        self.stride = stride
        self.change_nums = 0
        self.mode = mode
        assert self.mode in ['pretrain', 'binary', 'binaryactonly']
        print('conv mode : {}'.format(self.mode))
        self.k = torch.tensor([10]).float().cpu()
        self.t = torch.tensor([0.1]).float().cpu() 
        self.act_quant = BinaryActivation()

        self.filter_size = self.kernel_size[0]*self.kernel_size[1]*self.in_channels

    def forward(self, input):

        if self.mode == 'binaryactonly' or self.mode == 'binary':
            input = self.act_quant(input)
        elif self.mode == 'pretrain':
            pass
        else:
            assert False
            
        if self.mode == 'binaryactonly' or self.mode == 'pretrain':
            weight = self.weight
        elif self.mode == 'binary':
            w = self.weight 
            beta_w = w.mean((1,2,3)).view(-1,1,1,1)
            alpha_w = torch.sqrt(((w-beta_w)**2).sum((1,2,3))/self.filter_size).view(-1,1,1,1)
            w = (w - beta_w)/alpha_w 
            wb = Q_W.apply(w, self.k, self.t)
            weight = wb * alpha_w + beta_w
        else:
            assert False
        output = F.conv2d(input, weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return output


class BasicBlock(nn.Module):
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

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, act=None, res_scale=1):

        super(ResBlock, self).__init__()
        #pdb.set_trace()
        self.conv1 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, padding=(kernel_size//2,kernel_size//2), bias=bias),
            nn.BatchNorm2d(n_feats)
        )
        #self.act = act()
        self.act = act
        self.conv2 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, padding=(kernel_size//2,kernel_size//2), bias=bias),
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
    def __init__(self, args, conv=BinaryConv): #这里定义了conv的类型
        super(EDSR, self).__init__()

        conv = functools.partial(conv, mode=args.binary_mode)

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        #print(args.n_feats)
        kernel_size = 3 
        scale = args.scale[0]
        self.need_mid_feas = args.need_mid_feas

        #act = functools.partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)
        act = Maxout(n_feats)

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
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
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

