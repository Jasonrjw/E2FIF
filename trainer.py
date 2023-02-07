import os
import math
from decimal import Decimal

import utility
from evaluate import batch_PSNR, new_psnr, new_ssim

import torch
import torch.nn.utils as utils
from tqdm import tqdm

import pdb
import matplotlib.pyplot as plt
from ridgeplot import ridgeplot

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        # 如果是从之前训练一半的模型继续的话，也导入优化器的参数
        if self.args.load != '' and args.initial == False:
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        self.ckp.tb_write_log({
            'lr' : lr,
        }, step=epoch)

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            #pdb.set_trace()
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                psnr = batch_PSNR(hr, sr, data_range=self.args.rgb_range)
                self.ckp.write_log('[{}/{}]\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    '[PSNR: {:.4f}]'.format(psnr),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
                self.ckp.tb_write_log({
                    'loss' : loss,
                    'psnr' : psnr,
                }, step=len(self.loader_train)*epoch+batch, prefix='train')

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )

        self.ckp.add_log_ssim(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )

        self.model.eval()
        
        # '''plot distribution'''
        # global activation 
        # activation = []

        # global var
        # var = 0

        # def hook(module, input):
        #     input = input[0]
        #     print(input.shape)
        #     num_chn = input.shape[1]
        #     for i in range(num_chn):
        #         act = input[:,i,:,:]
        #         '''plot feature map distribution'''
        #         act = act.reshape(-1).cpu().detach().numpy()
        #         global activation
        #         activation.append(act)
        #         # plt.clf()
        #         # plt.hist(act, bins=100)
        #         # plt.savefig('./plot/1/' + str(i) + '.png')

        #     #     '''calculate variance'''
        #     #     global var
        #     #     var += torch.var(act)
        #     # print("*******这一层所有channel feature map variance的平均:", var/num_chn)


        # for name,module in self.model.named_modules():
        #     if 'model.body.15' == name :
        #         print(name)
        #         module.register_forward_pre_hook(hook)

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test): #可以指定很多个dataset
            dataset_name = self.args.data_test[idx_data]
            for idx_scale, scale in enumerate(self.scale): #可以指定很多个scale
                d.dataset.set_scale(idx_scale)
                psnr_sum, ssim_sum = 0, 0
                psnr_bicubic_sum, ssim_bicubic_sum = 0, 0
                # print(self.ckp.log)
                for lr, hr, filename in tqdm(d, ncols=80):
                    if self.args.need_rgb:
                        lr_rgb, hr_rgb, filename = filename
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    if self.args.need_mid_feas:
                        sr, mid_feas = sr
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]

                    #pdb.set_trace()
                    psnr = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )

                    ssim = utility.calc_ssim(
                        sr, hr, scale
                    )

                    #psnr = new_psnr(hr, sr, scale=scale, data_range=self.args.rgb_range, chop=True)
                    #ssim = new_ssim(hr, sr, scale=scale, data_range=self.args.rgb_range, chop=True)
                    # psnr_sum += psnr
                    # ssim_sum += ssim

                    self.ckp.log[-1, idx_data, idx_scale] += psnr
                    self.ckp.log_ssim[-1, idx_data, idx_scale] += ssim

                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                        if self.args.need_rgb:
                            save_list.extend([lr_rgb, hr_rgb])
                            if self.args.need_mid_feas:
                                save_list.extend([mid_feas])
               
                    if self.args.save_results:
                    # print(dataset_name, filename)
                    # if self.args.save_results and dataset_name == 'Set14' and filename[0] == 'baboon':
                    #if self.args.save_results and dataset_name == 'test' and filename[0] == 'img_004':
                    # if self.args.save_results and dataset_name == 'test' and filename[0] == 'baboon':
                    # if self.args.save_results:
                        print(dataset_name, filename)
                        self.ckp.save_results(d, filename[0], save_list, scale)
                        #psnr_bicubic, ssim_bicubic = self.ckp.save_results_my(d, filename[0], save_list, scale, self.args)
                        #print('bicubic : ', psnr_bicubic, ssim_bicubic)

                    # fig = ridgeplot(samples = (activation[i] for i in range(len(activation))),linewidth=1.0)
                    # fig.show()

                    # pdb.set_trace()  #plot distribution只需要test 1张图片

                self.ckp.log[-1, idx_data, idx_scale] /= len(d) #d是数据集长度, 是一个数据集中所有图片的平均PSNR
                self.ckp.log_ssim[-1, idx_data, idx_scale] /= len(d)

                print(len(d), self.ckp.log.shape)
                # print(self.ckp.log)
                # psnr_sum /= len(d)
                # ssim_sum /= len(d)
                # print(self.ckp.log[-1, idx_data, idx_scale])
                #pdb.set_trace()
                ''' 记录best ssim '''
                best_ssim = self.ckp.log_ssim.max(0)
                best_psnr = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f}, SSIM: {:.3f} (Best_PSNR: {:.3f}, Best_SSIM: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        self.ckp.log_ssim[-1, idx_data, idx_scale],
                        best_psnr[0][idx_data, idx_scale],
                        best_ssim[0][idx_data, idx_scale],
                        best_psnr[1][idx_data, idx_scale] + 1
                    )
                )
                self.ckp.tb_write_log({
                    'psnr' : psnr_sum,
                    'psnr_origin' : self.ckp.log[-1, idx_data, idx_scale],
                    'ssim' : self.ckp.log_ssim[-1, idx_data, idx_scale],
                }, step=epoch, prefix='test_{}'.format(dataset_name))

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best_psnr[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

