"""Training Script"""
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import Compose

from models.m_fewshot_resnet import FewShotSegResNet
from dataloaders.customized import voc_fewshot, ultrasound_fewshot
from dataloaders.transforms import (
    RandomMirror, Resize, ToTensorNormalize, UltrasoundToTensorNormalize,
    ColorJitter, RandomRotation, RandomScale, GaussianNoise, RandomGaussianBlur
)
from util.utils import set_seed
from util.m_loss import DiceLoss, FocalLoss
from config import ex


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = False
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    init_path = _config['path'].get('init_path')
    init_path = str(init_path).strip() if init_path else ''
    # `model` carries the alignment and ablation settings.
    model_cfg = dict(_config.get('model', {}))
    model = FewShotSegResNet(
        pretrained_path=init_path if init_path else None,
        cfg=model_cfg,
        load_encoder_pretrained=True,
    )
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    model.train()

    use_amp = _config.get('use_amp', True)
    scaler = GradScaler() if use_amp else None


    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'ultrasound':
        make_data = ultrasound_fewshot
    else:
        raise ValueError('Wrong config for dataset! Supported: VOC, ultrasound.')
    labels = [1, 2, 3, 4, 5]
    if data_name == 'ultrasound':
        transforms = Compose([
            Resize(size=_config['input_size']),
            RandomScale(min_scale=0.85, max_scale=1.15),
            RandomRotation(max_angle=10),
            ColorJitter(brightness=0.12, contrast=0.12),
            GaussianNoise(mean=0.0, std=0.015, prob=0.25),
            RandomGaussianBlur(prob=0.2, radius_range=(0.5, 1.2)),
            RandomMirror(),
        ])
    else:
        transforms = Compose([Resize(size=_config['input_size']), RandomMirror()])
    to_tensor = (
        UltrasoundToTensorNormalize()
        if data_name == 'ultrasound'
        else ToTensorNormalize()
    )
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=to_tensor,
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries'],
        input_size=_config['input_size'],
    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    if _config['mode'] == 'train':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
        scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
        criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])
        dice_loss_fn = DiceLoss(ignore_index=_config['ignore_label'])
    else:
        optimizer = None
        scheduler = None
        criterion = None

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0, 'dice_loss': 0}
    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
         
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)
        # # 强制修正维度
        # if query_labels.dim() == 4 and query_labels.size(1) == 1:
        #     query_labels = query_labels.squeeze(1)

        # Forward and Backward
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask,
                                               query_images)
                if len(query_labels.shape) == 4 and query_labels.shape[-1] == 4:
                    query_labels = query_labels[:, :, :, 0]
                elif len(query_labels.shape) == 4:
                    query_labels = query_labels.squeeze(-1)
                query_loss = criterion(query_pred, query_labels)
                d_loss = dice_loss_fn(query_pred, query_labels)
                combined_loss = 0.5 * query_loss + 0.5 * d_loss
                loss = combined_loss + align_loss * _config['align_loss_scaler']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask,
                                           query_images)
            if len(query_labels.shape) == 4 and query_labels.shape[-1] == 4:
                query_labels = query_labels[:, :, :, 0]
            elif len(query_labels.shape) == 4:
                query_labels = query_labels.squeeze(-1)
            query_loss = criterion(query_pred, query_labels)
            d_loss = dice_loss_fn(query_pred, query_labels)
            combined_loss = 0.5 * query_loss + 0.5 * d_loss
            loss = combined_loss + align_loss * _config['align_loss_scaler']
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Log loss
        d_loss_val = d_loss.detach().data.cpu().numpy()
        combined_loss_val = combined_loss.detach().data.cpu().numpy()
        align_loss_val = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', combined_loss_val)
        _run.log_scalar('ce_loss', query_loss.detach().data.cpu().numpy())
        _run.log_scalar('dice_loss', d_loss_val)
        _run.log_scalar('align_loss', align_loss_val)
        log_loss['loss'] += combined_loss_val
        log_loss['dice_loss'] += d_loss_val
        log_loss['align_loss'] += align_loss_val


        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss_avg = log_loss['loss'] / (i_iter + 1)
            dice_loss_avg = log_loss['dice_loss'] / (i_iter + 1)
            align_loss_avg = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss_avg:.6f}, dice_loss: {dice_loss_avg:.6f}, align_loss: {align_loss_avg:.6f}')

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
