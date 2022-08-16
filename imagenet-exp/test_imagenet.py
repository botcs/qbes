import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from torchvision import models
import gated_resnet
import gated_swin
import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import time
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())), default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0),
    weights=Param(str, default="")
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1),
    top_k_pred=Param(int, "how many values of the prediction should be stored", default=5)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=256),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1),
    proportion=Param(float, 'how much of the validation data to process', default=.1),
    seed=Param(int, "random seed for batch ordering", default=42)
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0)
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

Section('qbes', 'project specific config').params(
    config_file=Param(str, "which blocks to skip", required=True),
    id_from=Param(int, "index in qbes.config_file to read from", required=True),
    id_to=Param(int, "index in qbes.config_file to read until", required=True),
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

class ImageNetTrainer:
    @param('training.distributed')
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.initialize_logger()
        

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing):
        assert optimizer == 'sgd'

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]

        self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('validation.seed')
    @param('validation.proportion')
    @param('training.distributed')
    @param('data.in_memory')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, seed, proportion, distributed, in_memory):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        subset_indices = None
        if proportion < 1:
            dummy_loader = Loader(val_dataset,
                batch_size=1,
                num_workers=num_workers,
                order=OrderOption.SEQUENTIAL,
                drop_last=False,
                pipelines={
                    'image': image_pipeline,
                    'label': label_pipeline
                },
                distributed=distributed,
                indices=subset_indices,
                seed=seed,
                os_cache=in_memory
                )
    
            dataset_size = len(dummy_loader)
            print(f"Original dataset size: {dataset_size}")
            subset_size = int(dataset_size * proportion)
            np.random.default_rng(seed)
            np.random.seed(seed)
            subset_indices = np.random.choice(
                dataset_size, 
                subset_size, 
                replace=False)
            print(f"Subset size: {subset_size}")

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed,
                        indices=subset_indices,
                        seed=seed,
                        os_cache=in_memory
                        )
        return loader

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'val_time': val_time
            }, **extra_dict))

        return stats

    @param('model.arch')
    @param('model.pretrained')
    @param('model.weights')
    @param('training.distributed')
    @param('training.use_blurpool')
    def create_model_and_scaler(self, arch, pretrained, distributed, use_blurpool, weights):
        scaler = GradScaler()
        # model = getattr(models, arch)(pretrained=pretrained)

        #Override for QBES experiments
        if "resnet" in arch:
            model = getattr(gated_resnet, arch)()
        elif "swin" in arch:
            model = getattr(gated_swin, arch)

            model_loader = getattr(gated_swin, arch)
            weights = getattr(gated_swin, "Swin_B_Weights")
            model = model_loader(weights=weights.DEFAULT)
        else:
            raise NotImplementedError("lel")

        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if "swin" not in arch and weights != "":
            state_dict = ch.load(weights)
            # if trained with distributed=True then each
            # weight entry will have "module." beginninng
            standard_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                standard_state_dict[k] = v
            model.load_state_dict(standard_state_dict, strict=False)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler

    @param('qbes.config_file')
    @param('qbes.id_from')
    @param('qbes.id_to')
    @param('validation.lr_tta')
    @param('logging.folder')
    @param('logging.top_k_pred')
    def val_loop(self, config_file, id_from, id_to, lr_tta, folder, top_k_pred):
        model = self.model
        model.eval()
        all_configs = json.load(open(config_file))
        configs = all_configs[id_from:id_to]
        cache_dir = f"{folder}/{os.path.basename(config_file)[:-5]}"
        self.log({"cache dir": cache_dir})
        os.makedirs(cache_dir, exist_ok=True)
        stats = {}
        with ch.no_grad(): 
            with autocast():
                for config_id, skip_block_ids in enumerate(configs, start=id_from):
                    start_time = time.time()
                    for images, target in tqdm(self.val_loader):
                        output = self.model(images, skip_block_ids)
                        if lr_tta:
                            output += self.model(ch.flip(images, dims=[3]), skip_block_ids)

                        for k in ['top_1', 'top_5']:
                            self.val_meters[k](output, target)

                        self.qbes_outputs["preds"](output.argmax(dim=1))
                        self.qbes_outputs["targets"](target)

                        loss_val = self.loss(output, target)
                        self.val_meters['loss'](loss_val)

                    end_time = time.time()
                    fname = f"{cache_dir}/{config_id:05}.pth"
                    save_dict = {
                        "preds": self.qbes_outputs["preds"].compute().cpu(),
                        "targets": self.qbes_outputs["targets"].compute().cpu(),
                        "inference_time": end_time - start_time
                    }

                    ch.save(save_dict, fname)
                    

                    stats_entry = {k: m.compute().item() for k, m in self.val_meters.items()}
                    [meter.reset() for meter in self.val_meters.values()]
                    [meter.reset() for meter in self.qbes_outputs.values()]
                    self.log({"config_id": config_id, "eval_stats": stats_entry})
                    stats[config_id] = stats_entry
        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(compute_on_step=False).to(self.gpu),
            'top_5': torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric(compute_on_step=False).to(self.gpu),
        }
        self.qbes_outputs = {
            "preds": torchmetrics.CatMetric().to(self.gpu),
            "targets": torchmetrics.CatMetric().to(self.gpu),
        }

        if self.gpu == 0:
            folder = (Path(folder) / str(self.uid)).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)

    def log(self, content):
        if self.gpu != 0: return
        print(f'=> Log: {content}')
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log()
        else:
            raise RuntimeError("this script is only for evaluation")

        if distributed:
            trainer.cleanup_distributed()

# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":
    make_config()
    ImageNetTrainer.launch_from_args()
