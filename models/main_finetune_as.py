# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import models.models_vit
import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import to_2tuple

import util.lr_sched as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed, interpolate_pos_embed_audio, interpolate_patch_embed_audio, interpolate_pos_embed_img2audio
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import math
import torch.nn as nn
from models.engine_finetune_as import train_one_epoch, evaluate #, train_one_epoch_av, evaluate_av
from models.dataset import AudiosetDataset, DistributedWeightedSampler, DistributedSamplerWrapper
from timm.models.vision_transformer import PatchEmbed
from tools.utils import align_word_embedding
from torch.utils.data import WeightedRandomSampler, DataLoader, DistributedSampler
import torch.nn.functional as F
def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for acc', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    #累积梯度迭代（用于在内存约束下增加有效批大小）
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    ## 不一样
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--config')
    parser.add_argument('--ntaken')
    # Dataset parameters
    parser.add_argument('--data_path', default='/public/home/02566/CYX/ast-master/egs/audioset/data/datafiles/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=52268, type=int,
                        help='number of the classification types')#？？？？？？？？
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # For audioset
    parser.add_argument('--audio_exp', action='store_true', help='audio exp')
    parser.add_argument("--data_train", type=str, default='/public/home/02566/CYX/ast-master/egs/audioset/data/datafiles/audiocaps.json', help="training data json")
    parser.add_argument("--data_eval", type=str, default='/public/home/02566/CYX/ast-master/egs/audioset/data/datafiles/eval_audiocaps.json', help="validation data json")
    parser.add_argument("--label_csv", type=str, default='/public/home/02566/CYX/ast-master/egs/audioset/data/train_val.csv', help="csv with class labels")
    parser.add_argument("--weight_csv", type=str, default='/public/home/02566/CYX/ast-master/egs/audioset/data/weight_train_all.csv', help="weight file")

    
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=192)
    parser.add_argument('--timem', help='time mask max length', type=int, default=48)
    #parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "k400"])
    parser.add_argument("--use_fbank", type=bool, default=False)
    parser.add_argument("--use_soft", type=bool, default=False)
    parser.add_argument("--fbank_dir", type=str, default="/public/home/02566/CYX/ast-master/egs/esc50/data/ESC-50-master/fbank", help="fbank dir") 
    parser.set_defaults(audio_exp=True)
    #parser.add_argument("--distributed", type=bool, default=True)
    parser.add_argument('--first_eval_ep', default=0, type=int, help='do eval after first_eval_ep')
    parser.add_argument('--use_custom_patch', type=bool, default=False, help='use custom patch with overlapping and override timm PatchEmbed')
    parser.add_argument('--source_custom_patch', type=bool, default=False, help='the pre-trained model already use custom patch')
    parser.add_argument('--roll_mag_aug', type=bool, default=False, help='use roll_mag_aug')
    parser.add_argument('--mask_t_prob', default=0.0, type=float, help='T masking ratio (percentage of removed patches).') #  
    parser.add_argument('--mask_f_prob', default=0.0, type=float, help='F masking ratio (percentage of removed patches).') #  
    #parser.add_argument('--split_pos', type=bool, default=False, help='use splitted pos emb')
    parser.add_argument('--weight_sampler', type=bool, default=False, help='use weight_sampler')
    parser.add_argument('--epoch_len', default=200000, type=int, help='num of samples/epoch with weight_sampler')
    parser.add_argument('--distributed_wrapper', type=bool, default=False, help='use distributedwrapper for weighted sampler')
    parser.add_argument('--replacement', type=bool, default=False, help='use weight_sampler')
    parser.add_argument('--mask_2d', type=bool, default=True, help='use 2d masking')
    parser.add_argument('--load_video', type=bool, default=False, help='load video')
    parser.add_argument('--av_fusion', type=bool, default=False, help='load video')
    parser.add_argument('--n_frm', default=6, type=int, help='num of frames for video')
    parser.add_argument('--replace_with_mae', type=bool, default=False, help='replace_with_mae')
    parser.add_argument('--load_imgnet_pt', type=bool, default=False, help='when img_pt_ckpt, if load_imgnet_pt, use img_pt_ckpt to initialize audio branch, if not, keep audio branch random')
    return parser

"""这是一个PyTorch模型中的一个模块，用于将图像转换为补丁嵌入。具体来说，
这个模块会将输入的图像分成多个重叠的补丁，并将每个补丁转换为一个嵌入向量。这个模块的输入是一个图像张量，输出是一个嵌入向量张量。
"""
class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        """
        to_2tuple()是一个PyTorch中的辅助函数，用于将输入转换为长度为2的元组（tuple）。具体来说，如果输入是一个整数或浮点数，
        则将其转换为一个长度为2的元组，其中第一个元素为输入本身，第二个元素为None。如果输入已经是一个长度为2的元组，则直接返回该元组。
        """
        self.img_size = img_size
        self.patch_size = patch_size
        
# 使用卷积层将图像分成多个重叠的补丁，并将每个补丁转换为一个嵌入向量
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w  _表示n,emb_dim，但是用不到
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # 使用卷积层将图像分成多个重叠的补丁，并将每个补丁转换为一个嵌入向量
        x = x.flatten(2).transpose(1, 2)  # 将嵌入向量展平，并转置维度
        return x  # 返回嵌入向量张量

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if not args.audio_exp:
        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)
    else:
        norm_stats = {'audioset':[-4.2677393, 4.5689974], 'k400':[-4.2677393, 4.5689974], 
                      'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
        target_length = {'audioset':1024, 'k400':1024, 'esc50':512, 'speechcommands':128}

        """ &需要改吗？
        这两行代码定义了一个字典norm_stats和一个字典target_length，用于存储不同数据集的归一化统计信息和目标长度。

norm_stats字典的键是数据集名称，值是一个包含两个元素的列表，分别表示该数据集的均值和标准差。

target_length字典的键是数据集名称，值是一个整数，表示该数据集中音频的目标长度。"""
        multilabel_dataset = {'audioset': True, 'esc50': False, 'k400': False, 'speechcommands': True}
        audio_conf_train = {'num_mel_bins': 128, 
                      'target_length': target_length[args.dataset], 
                      'freqm': 48,
                      'timem': 192,
                      'mixup': args.mixup,
                      'dataset': args.dataset,
                      'mode':'train',
                      'mean':norm_stats[args.dataset][0],
                      'std':norm_stats[args.dataset][1],
                      'noise':False,
                      'multilabel':multilabel_dataset[args.dataset],
                      }
        '''audio_conf_val = {'num_mel_bins': 128, 
                      'target_length': target_length[args.dataset], 
                      'freqm': 0,
                      'timem': 0,
                      'mixup': 0,
                      'dataset': args.dataset,
                      'mode':'val',
                      'mean':norm_stats[args.dataset][0],
                      'std':norm_stats[args.dataset][1],
                      'noise':False,
                      'multilabel':multilabel_dataset[args.dataset],
                      }'''  
        dataset_train = AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf_train, 
                                        use_fbank=args.use_fbank, fbank_dir=args.fbank_dir, 
                                        roll_mag_aug=args.roll_mag_aug, load_video=args.load_video, mode='train')
        '''dataset_val = AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=audio_conf_val, 
                                        use_fbank=args.use_fbank, fbank_dir=args.fbank_dir, 
                                        roll_mag_aug=False, load_video=args.load_video, mode='eval')'''
        """
        这段代码创建了一个用于训练数据的数据加载器。

        dataset_train 是训练数据集。
        sampler_train 是用于从数据集中采样数据的采样器。它用于在训练过程中对数据进行随机采样和洗牌。
        batch_size 是每个批次中的样本数。
        num_workers 是用于数据加载的子进程数。
        pin_memory 指定是否使用固定内存来加速数据传输到 GPU。
        drop_last 指定是否丢弃最后一个批次，如果它的大小小于批次大小。
        生成的 data_loader_train 对象可以用于在训练过程中迭代训练数据的批次。
        """
    if True: #args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        num_nodes = int(os.environ.get('num_nodes', 1))  #os.environ.get('num_nodes', 1)是Python中获取环境变量num_nodes的值，如果该环境变量不存在，则返回默认值1的操作。
        ddp = int(os.environ.get('DDP', 1))
        num_nodes = max(ddp, num_nodes)
        rank = int(os.environ.get('NODE_RANK', 0))
        print(f"num_nodes:{num_nodes}, rank:{rank}, ddp:{ddp}, num_tasks:{num_tasks}, global_rank:{global_rank}")
        # num_nodes:1, rank:0, ddp:1, num_tasks:8, global_rank:0 (sbatch)
        if args.weight_sampler:
            samples_weight = np.loadtxt(args.weight_csv, delimiter=',') #是NumPy中的一个函数，用于从文本文件中加载数据。具体来说，该函数会从指定的文本文件中读取数据，并将其存储为NumPy数组。
            if args.distributed_wrapper:
                print('use distributed_wrapper sampler')
                epoch_len=args.epoch_len #200000 #=> 250000
                #epoch_len=21000 # AS-20K
                # replacement should be False
                sampler_train = DistributedSamplerWrapper(
                                    sampler=WeightedRandomSampler(samples_weight, num_samples=epoch_len, replacement=args.replacement),
                                    dataset=range(epoch_len),
                                    num_replicas=num_tasks, #num_nodes, #num_tasks?
                                    rank=global_rank, #rank, # global_rank?
                                    )
            else:
                #sampler_train = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
                sampler_train = DistributedWeightedSampler(dataset_train, samples_weight,  num_replicas=num_tasks, rank=global_rank, replacement=args.replacement)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )

        print("Sampler_train = %s" % str(sampler_train))
        '''if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
'''
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    '''data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )'''

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = models.models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        mask_2d=args.mask_2d,
        use_custom_patch=args.use_custom_patch,
        ## remove video part for A-MAE
        #load_video=args.load_video,
        # n_frm=args.n_frm,
        #split_pos=args.split_pos,
        #av_fusion=args.av_fusion,
    )# 加载vit模型
    if args.audio_exp:
        img_size=(target_length[args.dataset],128) # 1024, 128
        in_chans=1
        emb_dim = 768
        if args.model == "vit_small_patch16":
            emb_dim = 384
        if args.use_custom_patch:
            model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=16, in_chans=1, embed_dim=emb_dim, stride=10) #音频通道数1
            """
            模型中用于执行"patch embedding"的模块或层。通常，这个模块会将输入的图像分成许多小块，并将每个小块嵌入到一个低维向量空间中，以便后续的处理。
            """
            model.pos_embed = nn.Parameter(torch.zeros(1, 1212 + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding
        else:
            model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=emb_dim, stride=16) # no overlap. stride=img_size=16
            num_patches = model.patch_embed.num_patches
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

    #if args.finetune and not args.eval:
    if args.finetune:#&&&最好能把args取消掉
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        # 保存模型参数
        torch.save(model.state_dict(), '/public/home/02566/CYX/AudioMAE-main/ckpt/model.pth')
        
        if not args.eval:
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]#将不匹配的删除
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False) #不符合条件的
        print(msg)

        # manually initialize fc layer
        if not args.eval:
            trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    loss_scaler = NativeScaler()
    """
    其中，args.finetune表示预训练模型的路径，如果指定了预训练模型，
    则会加载预训练模型的参数，并将模型移动到指定的设备上。args.lr表示学习率，如果没有指定学习率，
    则根据批量大小和基础学习率计算得到学习率。args.accum_iter表示梯度累积迭代次数，
    用于控制模型的内存占用。args.distributed表示是否使用分布式训练，如果使用，
    则使用分布式数据并行。lrd.param_groups_lrd函数用于构建带有层级学习率衰减的Adam优化器，
    其中model_without_ddp.no_weight_decay()函数返回不需要进行权重衰减的参数列表。"""
    if args.use_soft:
        criterion = SoftTargetCrossEntropy() 
    else:
        criterion = nn.BCEWithLogitsLoss() # works better
    

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    '''if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args.dist_eval)
        with open('aps.txt', 'w') as fp:
            aps=test_stats['AP']
            aps=[str(ap) for ap in aps]
            fp.write('\n'.join(aps))
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['mAP']:.4f}")
        exit(0)
'''
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_mAP = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # if args.load_video:
        #     train_stats = train_one_epoch_av(
        #         model, criterion, data_loader_train,
        #         optimizer, device, epoch, loss_scaler,
        #         args.clip_grad, mixup_fn,
        #         log_writer=log_writer,
        #         args=args
        #     )            
        # else:
        train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )
        
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        if epoch >= args.first_eval_ep:#训练部分
            #test_stats = evaluate(data_loader_val, model, device, args.dist_eval)
            print(f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']:.4f}")
            max_mAP = max(max_mAP, test_stats["mAP"])
            print(f'Max mAP: {max_mAP:.4f}')
        else:
            test_stats ={'mAP': 0.0}
            print(f'too new to evaluate!')

        if log_writer is not None:
            log_writer.add_scalar('perf/mAP', test_stats['mAP'], epoch)#log_writer是一个TensorBoard的日志写入器，用于将模型的性能指标写入到TensorBoard的日志文件中

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}#不一样

        if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush() 
                        """具体来说，log_writer.flush()会将缓存中的日志数据写入到TensorBoard的日志文件中，
                        并清空缓存。这样，即使程序意外终止或出现错误，之前的日志数据也可以被保存下来，以便后续的分析和可视化。"""
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
#json.dumps(log_stats)是Python中的一个函数，用于将Python对象转换为JSON格式的字符串。
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))  #datetime.timedelta()是Python中的一个类，用于表示时间间隔。
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
#