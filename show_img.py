import torchvision.utils as utils
import torch

import utility_pams
import data
import model
import loss
from option import args
from trainer import Trainer
import pdb

from model.edsr_pams import PAMS_EDSR
from model.edsr_org import EDSR
from utils import common as util
from utils.common import AverageMeter
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from decimal import Decimal
import torch.nn.functional as F
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


torch.manual_seed(args.seed)
checkpoint = utility_pams.checkpoint(args)

device = torch.device('cpu' if args.cpu else f'cuda:{args.gpus}')

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

class Trainer():
    def __init__(self, args, loader, t_model, s_model, ckp):
        self.args = args
        self.scale = args.scale

        self.epoch = 0
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.t_model = t_model
        self.s_model = s_model
        arch_param = [v for k, v in self.s_model.named_parameters() if 'alpha' not in k]
        alpha_param = [v for k, v in self.s_model.named_parameters() if 'alpha' in k]

        params = [{'params': arch_param}, {'params': alpha_param, 'lr': 1e-2}]

        self.optimizer = torch.optim.Adam(params, lr=args.lr, betas = args.betas, eps=args.epsilon)
        self.sheduler = StepLR(self.optimizer, step_size=int(args.decay), gamma=args.gamma)
        self.writer_train = SummaryWriter(ckp.dir + '/run/train')

        self.losses = AverageMeter()
        self.att_losses = AverageMeter()
        self.nor_losses = AverageMeter()

    
    def test(self, is_teacher=False):
        torch.set_grad_enabled(False)
        epoch = self.epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )

        self.ckp.add_log_ssim(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        if is_teacher:
            model = self.t_model
        else:
            model = self.s_model
        model.eval()
        timer_test = utility_pams.timer()
        
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                i = 0
                val_images = []
                for lr, hr, filename in tqdm(d, ncols=80):
                    i += 1
                    lr, hr = self.prepare(lr, hr)
                    sr, s_res = model(lr)
                    sr = utility_pams.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    #calculate psnr
                    cur_psnr = utility_pams.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    self.ckp.log[-1, idx_data, idx_scale] += cur_psnr

                    #calculate ssim
                    ssim = utility_pams.calc_ssim(
                        sr, hr, scale
                    )
                    self.ckp.log_ssim[-1, idx_data, idx_scale] += ssim

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        save_name = f'{args.k_bits}bit_{filename[0]}'
                        self.ckp.save_results(d, save_name, save_list, scale)
                    
                    val_images.extend(
                    [display_transform()(lr.data.cpu().squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 15)
                val_save_bar = tqdm(val_images, desc='[saving training results]')
                index = 1
                for image in val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image, 'img_results/pams/set5/' + 'epoch_%d.png' % epoch, padding=5)
                    index += 1


                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                self.ckp.log_ssim[-1, idx_data, idx_scale] /= len(d)
                
                ''' 记录best ssim '''
                best_ssim = self.ckp.log_ssim.max(0)
                best_psnr = self.ckp.log.max(0)

                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best_PSNR: {:.3f}, Best_SSIM: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best_psnr[0][idx_data, idx_scale],
                        best_ssim[0][idx_data, idx_scale],
                        best_psnr[1][idx_data, idx_scale] + 1
                    )
                )

                self.writer_train.add_scalar(f'psnr', self.ckp.log[-1, idx_data, idx_scale], self.epoch)

        if self.args.save_results:
            self.ckp.end_background()
            
        if not self.args.test_only:
            is_best = (best_psnr[1][0, 0] + 1 == epoch)

            state = {
            'epoch': epoch,
            'state_dict': self.s_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.sheduler.state_dict()
        }
            util.save_checkpoint(state, is_best, checkpoint =self.ckp.dir + '/model')
        
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.cuda()

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            return self.epoch >= self.args.epochs

def main():
    if checkpoint.ok:
        loader = data.Data(args)
        if args.model.lower() == 'edsr_pams':
            t_model = EDSR(args, is_teacher=True).to(device)
            s_model = PAMS_EDSR(args, bias=True).to(device)
        else:
            raise ValueError('not expected model = {}'.format(args.model))
        
        if args.test_only:
            if args.refine is None:
                ckpt = torch.load(f'{args.save}/model/model_best.pth.tar')
                refine_path = f'{args.save}/model/model_best.pth.tar'
            else:
                ckpt = torch.load(f'{args.refine}')
                refine_path = args.refine

            #s_checkpoint = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            s_checkpoint = ckpt
            s_model.load_state_dict(s_checkpoint, strict=False)
            print(f"Load model from {refine_path}")

        t = Trainer(args, loader, t_model, s_model, checkpoint)
        
        print(f'{args.save} start!')
        t.test()
        checkpoint.done()
        print(f'{args.save} done!')


if __name__ == '__main__':
    main()
