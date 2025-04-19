import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel



@MODEL_REGISTRY.register()
class SRModel_new(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network STUDENT
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        if opt.get('train'):
            if opt['train'].get('finetune'):
                # Define teacher network for knowledge distillation (frozen during training)
                self.teacher = build_network(opt['network_teacher']).to(self.device)

                teacher_path = self.opt['path'].get('pretrain_network_g', None)
                if teacher_path is not None:
                    self.load_network(self.teacher, teacher_path, strict=True)

                for param in self.teacher.parameters():
                    param.requires_grad = False

                self.teacher.eval()
                print("Teacher network loaded for knowledge distillation.")
                

        # load pretrained models
        print("LOADING PRETRAINED MODEL")
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

            if opt.get('train'):
                if opt['train'].get('finetune'):
                    print("Freezing the model layers  ..............")
                    for param in self.net_g.to_feat.parameters():
                        param.requires_grad = False

                    print("Freezing  the Attention Blocks ..........")
                    for i in range(4):  # Adjust this number based on how many blocks you want to freeze
                        for param in self.net_g.feats[i].parameters():
                            param.requires_grad = False
                    for i in range(4, len(self.net_g.feats)):  # Unfreeze the last 2 blocks
                        for param in self.net_g.feats[i].parameters():
                            param.requires_grad = True
                    for param in self.net_g.to_img.parameters():
                        param.requires_grad = True
                

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()


        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        if train_opt.get('wave_opt'):
            self.cri_wave = build_loss(train_opt['wave_opt']).to(self.device)
        else:
            self.cri_wave = None

        if train_opt.get('distillation'):
            self.cri_distillation = torch.nn.L1Loss()
        else:
            self.cri_distillation = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_distillation is None and self.cri_wave is None:
            raise ValueError('Pixel, perceptual and frequency losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def adjust_distill_weight(self ,epoch):
        # Linearly adjust the distillation weight based on epoch count
        train_opt = self.opt['train'].get('distillation')
        initial_distill_weight = train_opt.get('initial_distill_weight')
        min_distill_weight = train_opt.get('min_distill_weight')
        weight_decay_epochs = train_opt.get("weight_decay_epochs")
        decay_rate = (initial_distill_weight - min_distill_weight) / weight_decay_epochs
        new_weight = max(min_distill_weight, initial_distill_weight - epoch * decay_rate)
        return new_weight

    def optimize_parameters(self, current_iter , epoch = None):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        
        train_opt = self.opt['train'].get('distillation')

        if self.opt['train'].get('finetune'):
        distillation_loss_weight = self.adjust_distill_weight(epoch)

        # Distillation: Get teacher's output
        with torch.no_grad():
            teacher_output = self.teacher(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # Distillation loss (comparing student and teacher outputs)
        l_distill = self.cri_distillation(self.output, teacher_output)
        loss_dict['l_distill'] = l_distill


        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix + (distillation_loss_weight * l_distill)
            loss_dict['l_pix'] = l_pix

        # frequency loss
        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            loss_dict['l_freq'] = l_fft

        # wavelet-based frequency loss
        if self.cri_wave:
            l_wave = self.cri_wave(self.output, self.gt)
            l_total += l_wave
            loss_dict['l_wave'] = l_wave

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def bicubic_upscaling(self):
        with torch.no_grad():
            self.bicubic_output = F.interpolate(self.lq , scale_factor = self.opt['network_g'].get('upscaling_factor') , mode='bicubic', align_corners=False)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # To store images from prev epoch
        lq_img, sr_img, gt_img = None, None, None

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            if self.opt['val'].get('bic'):
                self.bicubic_upscaling()

            visuals = self.get_current_visuals()

            lq_img = tensor2img([visuals['lq']])
            sr_img = tensor2img([visuals['result']])
            if self.opt['val'].get('bic'):
                bic_img = tensor2img([visuals['bic']])
            metric_data['img'] = sr_img
            
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if self.opt['val'].get('bic'):
                del self.bicubic_output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        if self.opt['val'].get('bic'):
                            save_bic_img_path = osp.join(self.opt['path']['bic_visualization'], dataset_name,
                                                    f'{img_name}_bic.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                        if self.opt['val'].get('bic'):
                            save_bic_img_path = osp.join(self.opt['path']['bic_visualization'], dataset_name,
                                                    f'{img_name}_bic.png')
                imwrite(sr_img, save_img_path)
                if self.opt['val'].get('bic'):
                    imwrite(bic_img , save_bic_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        if self.opt.get('train') :
            if self.opt['train'].get('finetune'):
                if lq_img is not None and sr_img is not None:
                # Display LQ and SR images side by side
                    lq_t , sr_t = lq_img.astype('uint8') , sr_img.astype('uint8')
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(lq_t)
                    axes[0].set_title('Low Resolution')
                    axes[1].imshow(sr_t)
                    axes[1].set_title('Super Resolution')
                    
                    for ax in axes:
                        ax.axis('off')
                    plt.savefig('/kaggle/working/side_by_side.png')
                    plt.tight_layout()
                    plt.show()  # Ensure the images are displayed side by side
                else:
                    print("LQ or SR image is None; skipping plot display.")

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if self.opt['val'].get('bic'):
            out_dict['bic'] = self.bicubic_output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, epoch, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter , epoch)
        self.save_training_state(epoch, current_iter)
