import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=32,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--gpus', type=str, default="0",
                    help='gpus ids')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/home/rjwei/Data_raid/dataset',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-900',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--need_rgb', action='store_true',
                    help='id need the rgb values')
parser.add_argument('--need_mid_feas', action='store_true',
                    help='id need the rgb values')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default=None,
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')


#quantization specifications
parser.add_argument('--weight_bits', type=int, default=32,
                    help='The k_bits of the quantzie')

parser.add_argument('--input_bits', type=int, default=32,
                    help='The k_bits of the quantzie')

parser.add_argument('--weight_quant_method', type=str, default='twn',
                    help='pre-trained model directory')

parser.add_argument('--input_quant_method1', type=str, default='elastic_signed',
                    help='pre-trained model directory')

parser.add_argument('--input_quant_method2', type=str, default='elastic_unsigned',
                    help='pre-trained model directory')

parser.add_argument('--symmetric1', action='store_true', default=False,
                    help='不加默认是False, 即第一个activation quantizer是asymmetric')

parser.add_argument('--symmetric2', action='store_true', default=False,
                    help='不加默认是False, 即第二个activation quantizer是asymmetric')

parser.add_argument('--weight_layerwise', action='store_true', default=False,
                    help='不加默认是False,即per channel quantize')

parser.add_argument('--input_layerwise', action='store_true', default=False,
                    help='不加默认是False,即per channel quantize') 


# Option for Residual dense network (RDN)
parser.add_argument('--n_convs', type=int, default=8,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
# parser.add_argument('--RDNconfig', type=str, default='B',
#                     help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--initial', action='store_true',
                    help='用train好的model initialize')

# Optimization specifications
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200-400-600-800',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# PAMS setting
parser.add_argument('--k_bits', type=int, default=32,
                    help='The k_bits of the quantzie')
parser.add_argument('--ema_epoch', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--w_l1', type=float, default=1.0,
                    help='loss coefficient')
parser.add_argument('--w_at', type=float, default=1e+3,
                    help='loss coefficient') 
parser.add_argument('--refine', type=str, default=None,
                    help='refine model directory')
                    
# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--postfix', type=str, default='',
                    help='postfix added to the model as save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint, -2: 从model_best.pt导入继续训练\
                                                            -1: 从最新(最后一个)模型model_latest.pt导入，用于继续训练； \
                                                            0：从args.pre_train直接导入； \
                                                            其它数字：指定从第几个epoch继续训练')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

# Binary Model setting
parser.add_argument('--binary_mode', type=str, default='binary',
                    help='binary model mode : binary/binaryactonly/pretrain')

# Stereo images SR
parser.add_argument('--tb_dir', type=str, default='',
                    help='Tensorboard log file root path')

args = parser.parse_args()
template.set_template(args)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# if args.load:
#     args.save = args.load

if not args.save:
    args.save = args.model.lower() + args.postfix

args.save += '_{}_x{}'.format(args.binary_mode, args.scale)
if 'edsr' in args.model.lower():
    args.save += '_n{}_c{}'.format(args.n_resblocks, args.n_feats)
elif 'rcan' in args.model.lower():
    args.save += '_g{}_n{}_c{}'.format(args.n_resgroups, args.n_resblocks, args.n_feats)
elif 'rdn' in args.model.lower():
    args.save += '_nr{}_nc{}_c{}'.format(args.n_resblocks, args.n_convs, args.n_feats)
else:
    assert False
args.save += '_e{}_lr{}_b{}_p{}'.format(args.epochs, args.lr, args.batch_size, args.patch_size)

# if args.load:
#     args.load = args.save

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')
# print(args.data_train)

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

