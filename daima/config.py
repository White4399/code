"""Experiment Configuration"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'


ex = Experiment('PANet')
# NOTE: keep ex.path at the default 'PANet' — setting it to '' triggers Sacred
# issue #709 (commands get truncated, e.g. 'main' -> 'ain').
# FileStorageObserver does NOT prepend ex.path on its own, so the run
# subdirectory is exactly what we pass in add_observer() below — and
# run_ablation.py hands it the full "PANet_resnet50_..." string.
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations"""
    input_size = (480, 640)
    seed = 1234
    cuda_visable = '0, 1, 2, 3'
    gpu_id = 2
    backbone = 'resnet50'
    mode = 'test' # 'train' or 'test'
    # mode = 'train' # 'train' or 'test'

    # --- Ablation variant tag -------------------------------------------
    # Purely a NAMING tag: it decides the run directory, so that every
    # ablation experiment gets its own folder. The actual model is chosen
    # by editing the import in train.py / test.py:
    #     from models.m_abl_no_fam import FewShotSegResNet
    # Valid tags: full, no_fpn, no_attn, no_fam, no_ppm, no_pretrain, plain
    #   cum0_base, cum1_fpn, cum2_fam, cum3_ppm, cum4_attn
    variant = 'F1_A1_P1_X1'


    if mode == 'train':
        dataset = 'ultrasound'  # 'VOC' or 'ultrasound'
        n_steps = 30000
        label_sets = 0
        batch_size = 1
        lr_milestones = [10000, 20000, 30000]
        align_loss_scaler = 1
        ignore_label = 255
        print_interval = 100
        save_pred_every = 10000
        use_amp = True

        model = {
            'align': True,
        }

        task = {
            'n_ways': 5,
            'n_shots': 1,
            'n_queries': 1,
        }

        optim = {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }

    elif mode == 'test':
        notrain = False
        # --- snapshot -----------------------------------------------------
        # Manually point to the checkpoint you want to evaluate.
        # The run folder of each ablation variant looks like:
        #   ./runs/PANet_resnet50_ultrasound_align_sets_0_5way_1shot_[train]_Abl_<variant>/<id>/snapshots/30000.pth
        snapshot = './runs/resnet50_ultrasound_align_sets_0_5way_1shot_[train]_Abl_F1_A1_P1_X1/2/snapshots/30000.pth'
        n_runs = 3
        n_steps = 1000
        batch_size = 1
        scribble_dilation = 0
        bbox = False
        scribble = False

        # 添加 optim（测试模式也需要）
        optim = {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }

        # Set dataset config from the snapshot string
        if 'ultrasound' in snapshot:
            dataset = 'ultrasound'
        elif 'VOC' in snapshot:
            dataset = 'VOC'
        else:
            dataset = 'ultrasound'

        # Set model config from the snapshot string
        model = {}
        for key in ['align',]:
            model[key] = key in snapshot

        # Set label_sets from the snapshot string
        label_sets = int(snapshot.split('_sets_')[1][0])

        # Set task config from the snapshot string
        task = {
            'n_ways': int(re.search("[0-9]+way", snapshot).group(0)[:-3]),
            'n_shots': int(re.search("[0-9]+shot", snapshot).group(0)[:-4]),
            'n_queries': 1,
        }
       
    else:
        raise ValueError('Wrong configuration for "mode" !')


    # --- Compose experiment name ---
    # `model` contains boolean flags (align, ...). We include their ON-state names.
    # The ablation tag is appended so every variant is saved under a
    # distinct subdirectory (no snapshot overwrites).
    #
    # `variant` is a free-text tag identifying WHICH ablation model file is
    # imported in train.py / test.py (e.g. 'full', 'no_fam', 'no_ppm', ...).
    # Set it via `with variant=no_fam` on the command line so that each
    # experiment lands in its own run directory.
    ablation_tag = 'Abl_{}'.format(str(variant).strip() or 'full')
    exp_str = '_'.join(
       [backbone, dataset,]
        + [key for key, value in model.items() if value]
        + [f'sets_{label_sets}', f'{task["n_ways"]}way_{task["n_shots"]}shot_[{mode}]', ablation_tag])


    data_root = '.VOCdevkit/VOC2012/'
    if mode == 'train':
        voc_data_split = 'trainaug'
        ultrasound_split = 'train'
    else:
        voc_data_split = 'val'
        ultrasound_split = 'val'

    path = {
        'log_dir': './runs',
        # 'init_path': './pretrained_model/vgg16-397923af.pth',
        'init_path': './pretrained_model/resnet50-19c8e357.pth',
        'VOC': {
            'data_dir': data_root,
            'data_split': voc_data_split,
        },
        'ultrasound': {
            'data_dir': data_root,
            'data_split': ultrasound_split,
        },
    }

@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    # NOTE: FileStorageObserver does NOT prepend ex.path — only `with
    # exp_str=...` controls the subdirectory name. So we hand it the full
    # directory name from run_ablation.py (e.g. "PANet_resnet50_...").
    exp_name = config["exp_str"]
    if config['mode'] == 'test':
        if config['notrain']:
            exp_name += '_notrain'
        if config['scribble']:
            exp_name += '_scribble'
        if config['bbox']:
            exp_name += '_bbox'
    observer = FileStorageObserver.create(
        os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
