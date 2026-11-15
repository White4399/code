"""Evaluation Script"""
import os
import shutil
import csv
import time  # smh0521
import logging  # smh0521

import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
from PIL import Image  # 添加 PIL 用于保存图像

from models.m_fewshot_resnet import FewShotSegResNet
from dataloaders.customized import voc_fewshot,ultrasound_fewshot
# , mydataset_fewshot
from dataloaders.transforms import ToTensorNormalize,UltrasoundToTensorNormalize
from dataloaders.transforms import Resize, DilateScribble
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox
from config import ex


@ex.automain
def main(_run, _config, _log):
    # 将日志写入 Sacred 输出目录 (smh0521)
    log_dir = _run.observers[0].dir
    log_file_path = os.path.join(log_dir, 'test_output.log')
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(file_handler)

    # 清理显存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 保存源文件
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    # 移除 set_seed，允许每次测试随机采样
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    # `model` carries the alignment and ablation settings.
    model_cfg = dict(_config.get('model', {}))
    init_path = _config['path'].get('init_path')
    init_path = str(init_path).strip() if init_path else ''
    model = FewShotSegResNet(
        pretrained_path=init_path if init_path else None,
        cfg=model_cfg,
        load_encoder_pretrained=_config['notrain'],
    )
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    if not _config['notrain']:
        state_dict = torch.load(_config['snapshot'], map_location='cpu')
        # Drop weights for the removed module in older checkpoints.
        for key in tuple(state_dict):
            if key.startswith('module.fam.'):
                del state_dict[key]
        model.load_state_dict(state_dict)
    model.eval()
    
    # 再次清理显存
    torch.cuda.empty_cache()

    _log.info('###### Prepare data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
        to_tensor = ToTensorNormalize()
        max_label = 5
    elif data_name == 'ultrasound' or data_name == 'mydata':  # 添加超声数据集
        make_data = ultrasound_fewshot
        to_tensor = UltrasoundToTensorNormalize()  # 使用超声专用转换
        max_label = 5  # 5个前景类别
    else:
        raise ValueError('Wrong config for dataset!')


    # 设置类别标签
    if data_name == 'ultrasound':
        labels = [1, 2, 3, 4, 5]
    else:
        labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][_config['label_sets']]
    
    transforms = [Resize(size=_config['input_size'])]
    if _config['scribble_dilation'] > 0:
        transforms.append(DilateScribble(size=_config['scribble_dilation']))
    transforms = Compose(transforms)


    _log.info('###### Testing begins ######')
    metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    
    # 创建保存预测结果的目录
    save_dir = os.path.join(_run.observers[0].dir, 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    _log.info(f'Predictions will be saved to: {save_dir}')
    
    with torch.no_grad():
        for run in range(_config['n_runs']):
            _log.info(f'### Run {run + 1} ###')
            # 移除 set_seed 调用，允许每次测试随机采样不同图像

            _log.info(f'### Load data ###')
    
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
                input_size=_config['input_size']
            )

            
            testloader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=True,
                                    num_workers=1, pin_memory=True, drop_last=False)
            _log.info(f"Total # of Data: {len(dataset)}")

            # 为每个 run 创建子目录
            run_save_dir = os.path.join(save_dir, f'run_{run + 1}')
            os.makedirs(run_save_dir, exist_ok=True)
            os.makedirs(os.path.join(run_save_dir, 'predictions'), exist_ok=True)
            os.makedirs(os.path.join(run_save_dir, 'ground_truth'), exist_ok=True)
            os.makedirs(os.path.join(run_save_dir, 'overlay'), exist_ok=True)
            manifest_path = os.path.join(run_save_dir, 'tested_images.csv')#smh0521
            manifest_file = open(manifest_path, 'w', newline='', encoding='utf-8')
            manifest_writer = csv.writer(manifest_file)
            manifest_writer.writerow(['idx', 'query_id', 'class_ids', 'prediction_file', 'ground_truth_file', 'overlay_file'])#smh0521

            for idx, sample_batched in enumerate(tqdm.tqdm(testloader)):
                if _config['dataset'] == 'COCO':
                    label_ids = 0
                elif data_name == 'mydata':
                    label_ids = list(sample_batched['class_ids'])
                else:
                    label_ids = list(sample_batched['class_ids'])

                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]
                suffix = 'scribble' if _config['scribble'] else 'mask'

                if _config['bbox']:
                    support_fg_mask = []
                    support_bg_mask = []
                    for i, way in enumerate(sample_batched['support_mask']):
                        fg_masks = []
                        bg_masks = []
                        for j, shot in enumerate(way):
                            fg_mask, bg_mask = get_bbox(shot['fg_mask'],
                                                        sample_batched['support_inst'][i][j])
                            fg_masks.append(fg_mask.float().cuda())
                            bg_masks.append(bg_mask.float().cuda())
                        support_fg_mask.append(fg_masks)
                        support_bg_mask.append(bg_masks)
                else:
                    support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]
                    support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]

                query_images = [query_image.cuda()
                                for query_image in sample_batched['query_images']]
                query_ids = sample_batched.get('query_ids', [f'unknown_{idx:05d}'])#smh0521
                if isinstance(query_ids, (list, tuple)):
                    query_id = query_ids[0]
                else:
                    query_id = str(query_ids)#smh0521
                query_labels = torch.cat(
                    [query_label.cuda() for query_label in sample_batched['query_labels']], dim=0)

                # Warmup: 跳过第一次推理以排除 cudnn.benchmark 调优和 CUDA 缓存预热
                if run == 0 and idx == 0:
                    _log.info('Warmup: first inference excluded from timing (cudnn benchmark + CUDA cache warmup)')
                    with torch.no_grad():
                        _, _ = model(support_images, support_fg_mask, support_bg_mask, query_images)
                    torch.cuda.synchronize()
                    continue

                # 推理计时 (smh0521)
                torch.cuda.synchronize()
                t_start = time.time()
                query_pred, _ = model(support_images, support_fg_mask, support_bg_mask,
                                      query_images)
                torch.cuda.synchronize()
                inference_time = time.time() - t_start

                # 获取预测结果（取 argmax 得到类别标签）
                pred_mask = query_pred.argmax(dim=1)[0].cpu().numpy()  # [H, W]
                gt_mask = query_labels[0].cpu().numpy()  # [H, W]

                # 保存预测结果和真实标注
                # 使用与原始标注相同的调色板像素值 (0, 51, 102, 153, 204, 255)
                pred_file = f'pred_{idx:05d}_{query_id}.png'
                gt_file = f'gt_{idx:05d}_{query_id}.png'
                overlay_file = f'pred_viz_{idx:05d}_{query_id}.png'

                # 调色板: palette[0]=0, palette[1]=51, palette[2]=102, palette[3]=153, palette[4]=204, palette[5]=255
                palette = np.array([0, 51, 102, 153, 204, 255], dtype='uint8')

                # fewShot 的映射: class_ids[j] (原始类别) -> j+1 (fewShot索引)
                # 需要反向映射: fewShot索引 (0,1,2,3,4,5) -> palette索引 (0,1,2,3,4,5)
                # 由于 class_ids 包含原始类别 ID，直接用 palette[class_ids[j]] 获取调色板值
                class_ids = sample_batched['class_ids']

                # 创建 fewShot 索引到调色板值的映射
                # fewShot 索引 0 是背景 -> palette[0] = 0
                # fewShot 索引 j+1 对应 class_ids[j] -> palette[class_ids[j]]
                fewshot_to_palette = {0: 0}  # 背景
                for j, orig_class in enumerate(class_ids):
                    fewshot_to_palette[j + 1] = palette[orig_class]

                # 将 fewShot 索引映射为调色板像素值
                pred_viz = np.zeros_like(pred_mask, dtype='uint8')
                gt_viz = np.zeros_like(gt_mask, dtype='uint8')
                for fewshot_idx, palette_val in fewshot_to_palette.items():
                    pred_viz[pred_mask == fewshot_idx] = palette_val
                    gt_viz[gt_mask == fewshot_idx] = palette_val

                # 保存预测 mask（使用调色板）
                pred_img = Image.fromarray(pred_viz)
                pred_img.save(os.path.join(run_save_dir, 'predictions', pred_file))

                # 保存真实标注（使用调色板）
                gt_img = Image.fromarray(gt_viz)
                gt_img.save(os.path.join(run_save_dir, 'ground_truth', gt_file))

                # 保存叠加图（同样使用调色板）
                pred_overlay_img = Image.fromarray(pred_viz)
                pred_overlay_img.save(os.path.join(run_save_dir, 'overlay', overlay_file))
                manifest_writer.writerow([idx, query_id, label_ids, pred_file, gt_file, overlay_file])

                metric.record(pred_mask, gt_mask, labels=label_ids, n_run=run, inference_time=inference_time)

            manifest_file.close()
            _log.info(f'Tested image list saved to: {manifest_path}')

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            _run.log_scalar('classIoU', classIoU.tolist())
            dice_scores = (2 * classIoU) / (1 + classIoU)
            _run.log_scalar('dice', dice_scores.tolist())
            _run.log_scalar('meanIoU', meanIoU.tolist())
            # _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            # _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            _log.info(f'classIoU: {classIoU}')
            _log.info(f'      股骨       大收肌      目标神经     半健肌    股动脉')
            _log.info(f'dice: {dice_scores}')
            _log.info(f'meanIoU: {meanIoU}')
            
            # _log.info(f'classIoU_binary: {classIoU_binary}')
            # _log.info(f'meanIoU_binary: {meanIoU_binary}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    _log.info('----- Final Result -----')
    _run.log_scalar('final_classIoU', classIoU.tolist())
    _run.log_scalar('final_classIoU_std', classIoU_std.tolist())
    _run.log_scalar('final_meanIoU', meanIoU.tolist())
    _run.log_scalar('final_meanIoU_std', meanIoU_std.tolist())
    # _run.log_scalar('final_classIoU_binary', classIoU_binary.tolist())
    # _run.log_scalar('final_classIoU_std_binary', classIoU_std_binary.tolist())
    # _run.log_scalar('final_meanIoU_binary', meanIoU_binary.tolist())
    # _run.log_scalar('final_meanIoU_std_binary', meanIoU_std_binary.tolist())
    _log.info(f'classIoU mean: {classIoU}')
    _log.info(f'classIoU std: {classIoU_std}')
    _log.info(f'meanIoU mean: {meanIoU}')
    _log.info(f'meanIoU std: {meanIoU_std}')
    # _log.info(f'classIoU_binary mean: {classIoU_binary}')
    # _log.info(f'classIoU_binary std: {classIoU_std_binary}')
    # _log.info(f'meanIoU_binary mean: {meanIoU_binary}')
    # _log.info(f'meanIoU_binary std: {meanIoU_std_binary}')


    # smh0521: 推理时间统计
    all_times = []
    for n_run in range(len(metric.time_lst)):
        all_times.extend(metric.time_lst[n_run])
    if all_times:
        mean_time = np.mean(all_times)
        std_time = np.std(all_times)
        _log.info(f'Inference time per image - mean: {mean_time:.4f}s, std: {std_time:.4f}s, min: {np.min(all_times):.4f}s, max: {np.max(all_times):.4f}s')
        _run.log_scalar('inference_time_mean', mean_time)
        _run.log_scalar('inference_time_std', std_time)

    # 保存每张测试图的详细 TP, TN, FP, FN 结果（按类别分开，每个类别一个文件）
    excel_save_dir = os.path.join(_run.observers[0].dir, 'confusion_matrix_results')
    # 根据你的数据集设置类别名称（股骨、大收肌、目标神经、半健肌、股动脉）
    class_names = ['股骨', '大收肌', '目标神经', '半健肌', '股动脉']  # 5个类别的中文名称
    metric.save_per_sample_results(save_dir=excel_save_dir, labels=sorted(labels), class_names=class_names[:len(labels)])

    _log.info(f'Predictions saved to: {save_dir}')
