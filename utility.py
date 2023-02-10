import copy
import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from evaluate import new_psnr, new_ssim

from tensorboardX import SummaryWriter
import pdb
from skimage.metrics import structural_similarity


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        self.log_ssim = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        # args.load和args.save都是experiment中的文件夹名字
        #pdb.set_trace()
        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('.', 'experiment', args.save)
            assert not os.path.exists(self.dir) or args.reset, 'the save dir {} has exist'.format(args.save)
        else:
            self.dir = os.path.join('.', 'experiment', args.save)
            assert not os.path.exists(self.dir) or args.reset, 'the save dir {} has exist'.format(args.save)
            if os.path.exists(self.dir) and not args.test_only:
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''
        # 设置Tensorboard文件的root目录
        self.tb_dir = self.dir.replace('experiment', 'tensorboard_files')

        if args.reset:
            assert not args.load and args.resume != -1
            os.system('rm -rf ' + self.dir)
            os.system('rm -rf ' + self.tb_dir)
            args.load = ''

        os.makedirs(os.path.join('..', 'tensorboard_files'), exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        if args.test_only:
            open_type = 'a' if os.path.exists(self.get_path('log_test.txt'))else 'w'
            self.log_file = open(self.get_path('log_test.txt'), open_type)
        else:
            open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
            self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def add_log_ssim(self, log):
        self.log_ssim = torch.cat([self.log_ssim, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def tb_write_log(self, name_value, step, prefix=''):
        if prefix != '': prefix += '_'
        for name in name_value:
            value = name_value[name]
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    value = value.mean()
                value = value.item()
            self.tb_writer.add_scalar(prefix + name, value, step)

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
                    #cv2.imwrite(filename, tensor)
                    # cv2.imwrite(filename, tensor.numpy())
                    # imagesc可视化图
                    # fig = plt.figure()
                    # fig.set_size_inches(1. * tensor.shape[0] / tensor.shape[1], 1, forward = False)
                    # ax = plt.Axes(fig, [0., 0., 1., 1.])
                    # ax.set_axis_off()
                    # fig.add_axes(ax)
                    # plt.imshow(tensor.numpy())
                    # # plt.axis('off')
                    # # plt.xticks([])
                    # # plt.yticks([])
                    # plt.savefig(filename, dpi = tensor.shape[0])
                    # plt.close(fig)
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            if len(save_list)==3:
                postfix = ('SR', 'LR', 'HR')
            elif len(save_list)==5:
                postfix = ('SR', 'LR', 'HR', 'LR_RGB', 'HR_RGB')
            elif len(save_list)==6:
                postfix = ('SR', 'LR', 'HR', 'LR_RGB', 'HR_RGB', 'MID_FEAS')
            else:
                assert False
                postfix = ('SR_l', 'SR_r', 'LR_l', 'LR_r', 'HR_l', 'HR_r')
            print(postfix)
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))
            # if not self.args.need_mid_feas:
            #     return
            # mid_feas = save_list[-1]
            # mid_feas_sign = mid_feas.sign()
            # print(mid_feas.shape)
            # # mid_feas = (mid_feas - mid_feas.min(dim=[0,2,3])) / (mid_feas.max(dim=[0,2,3]) - mid_feas.min(dim=[0,2,3]))
            # mid_feas = mid_feas.squeeze(0).mul(255 / self.args.rgb_range).cpu().clamp(0,1) * 255
            # mid_feas_sign = mid_feas_sign.squeeze(0).mul(255 / self.args.rgb_range).cpu().clamp(0,1) * 255
            # print(mid_feas_sign)
            # for i in range(mid_feas.shape[0]):
            #     show = mid_feas[i]
            #     show = (show - show.min()) / (show.max()- show.min()) * 255
            #     self.queue.put(('{}{}_original.png'.format(filename, i), show))
            #     self.queue.put(('{}{}_sign.png'.format(filename, i), mid_feas_sign[i]))

    def save_results_my(self, dataset, filename, save_list, scale, args):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            if len(save_list)==3:
                postfix = ('SR', 'LR', 'HR')
            elif len(save_list)==5:
                postfix = ('SR', 'LR', 'HR', 'LR_RGB', 'HR_RGB')
            elif len(save_list)==6:
                postfix = ('SR', 'LR', 'HR', 'LR_RGB', 'HR_RGB', 'MID_FEAS')
            else:
                assert False
            print(postfix)
            imgname = filename.split('/')[-1]
            filename = os.path.join('./imgs', dataset.dataset.name + '_' + imgname + self.args.model.split('_')[1] + '_')
            hr_filename = os.path.join('./imgs', dataset.dataset.name + '_' + imgname)
            print('here', filename)
            # return
            # for i,(v, p) in enumerate(zip(save_list, postfix)):
            #     print(p, v.shape)
            #     normalized = v[0].mul(255 / self.args.rgb_range)
            #     tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            #     self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))
            save_list = [i[0].mul(255 / self.args.rgb_range).byte().permute(1, 2, 0).cpu().numpy() for i in save_list]
            sr, lr, hr, lr_ycbcr, hr_ycbcr = save_list
            lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
            sr_bicubic = cv2.resize(lr_ycbcr[:,:,0], (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
            print(sr_bicubic.shape, hr_ycbcr[0].shape)
            psnr_bicubic, ssim_bicubic = new_psnr(hr_ycbcr[:,:,0], sr_bicubic, scale=scale, data_range=self.args.rgb_range, chop=False), new_ssim(hr_ycbcr[:,:,0], sr_bicubic, scale=scale, data_range=self.args.rgb_range, chop=False)
            lr = np.expand_dims(lr, axis=2)
            print(sr.shape)
            sr_rgb = get_rgb(sr, hr_ycbcr)
            hr_rgb = get_rgb(None, hr_ycbcr)
            lr_rgb = get_rgb(lr, hr_ycbcr)
            self.queue.put(('{}{}.png'.format(filename, 'SR'), sr_rgb))
            self.queue.put(('{}{}.png'.format(hr_filename, 'HR'), hr_rgb))
            self.queue.put(('{}{}.png'.format(hr_filename, 'LR'), lr_rgb))
            return psnr_bicubic, ssim_bicubic


def get_rgb(y, ycbcr):
    rgb_show = copy.deepcopy(ycbcr)
    if y is not None:
        # y = y.squeeze().cpu().numpy()
        rgb_show[:, :, 0] = y.squeeze(-1)
    # rgb_show = rgb_show*255
    # rgb_show = rgb_show.astype(np.uint8)
    rgb_show = cv2.cvtColor(rgb_show, cv2.COLOR_YCrCb2RGB)
    # return rgb_show
    
    # # Set14, baboon
    # left_up = (80, 400)
    # width_height = (80, 40)

    # Urban100, img_004
    left_up = (700, 400)
    # left_up = (700, 460)
    width_height = (80, 40)

    right_bottom = (left_up[0]+width_height[0], left_up[1]+width_height[1])
    print(left_up, right_bottom)
    rgb_show_tangle = copy.deepcopy(rgb_show)
    rgb_show_tangle = cv2.rectangle(rgb_show_tangle, left_up, right_bottom, (0,255,0), 4)
    
    toup_img = rgb_show_tangle[left_up[1]:right_bottom[1], left_up[0]:right_bottom[0], :]
    toup_width = int(toup_img.shape[0]/toup_img.shape[1]*rgb_show.shape[1])
    up_img = cv2.resize(toup_img, (rgb_show.shape[1], toup_width))
    print(rgb_show.shape, up_img.shape)
    print()
    final_show = np.concatenate((rgb_show_tangle, up_img), 0)
    return final_show


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave] #算psnr的时候忽略边上的几个pixel, 边界上信息不足 可能会出问题
    # pdb.set_trace()
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

# 以下函数参考utility_pams, 为了计算SSIM on Y channel
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype 
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
        #rlt = np.dot(img, [65.738, 129.057, 25.064]) / 256.0 + 16.0

    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
        
    #pdb.set_trace()

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    #pdb.set_trace()
    return rlt.astype(in_img_type)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    #pdb.set_trace()
    #tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp 这里有问题，tensor原本是0-255的，这样会全被clamp到1
    #tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    tensor = tensor.squeeze().float().cpu()
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        #img_np = np.transpose(img_np, (1, 2, 0))  # HWC

    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        #img_np = np.transpose(img_np, (1, 2, 0))  # HWC

    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    
    # if out_type == np.uint8:
    #     img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    
    #pdb.set_trace()
    return img_np.astype(out_type)


def calc_ssim(img1, img2, scale):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    shave = scale

    img1 = tensor2img(img1)
    img2 = tensor2img(img2)
    
    # img1 = tensor2img(img1) / 255.
    # img2 = tensor2img(img2) / 255.

    #print('\n',img1.shape, img2.shape,'\n')
    # img1 = img1[shave:-shave, shave:-shave, :]
    # img2 = img2[shave:-shave, shave:-shave, :]
    
    img1 = bgr2ycbcr(img1) * 255
    img2 = bgr2ycbcr(img2) * 255

    #print(img1.shape, img2.shape)  # ndim=2
    #pdb.set_trace()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        #return ssim(img1, img2)
        return structural_similarity(img1, img2, gaussian_weights=True)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(structural_similarity(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return structural_similarity(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    print(milestones)
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

