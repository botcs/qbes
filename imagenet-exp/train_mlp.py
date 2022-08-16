import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.ops.misc import MLP, Permute
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from torchvision import models
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
    pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),
    train_proportion=Param(float, 'how much of the validation data to process', default=1.),
    val_proportion=Param(float, 'how much of the validation data to process', default=1.),
    seed=Param(int, "random seed for batch ordering", default=42)
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
    log_level=Param(int, '0 if only at end 1 otherwise', default=1)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1),
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd', "adam"])), 'The optimizer', default='sgd'),
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

Section('mlp', 'K dropping module specific config').params(
    target_file=Param(str, 'path to target file that gives max K per class', required=True),
    prediction_type=Param(str, "Classify K or regress", required=True),
    trunk_layer=Param(str, OneOf(["first", "last"]), "Whether to use first or last layer's output as feature", required=True),
    balance_weight=Param(int, "Whether to reweight samples or not", required=True)
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


class LambdaLayer(ch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class ImageNetTrainer:
    @param('training.distributed')
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
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
    @param("mlp.prediction_type")
    @param("mlp.target_file")
    @param("mlp.balance_weight")
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, prediction_type, target_file, balance_weight):
        # assert optimizer == 'sgd'

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


        _, counts = ch.load(target_file).unique(return_counts=True)
        if balance_weight:
            weight = 1 / counts.float()
        else:
            weight = ch.ones_like(counts.float())
        this_device = f'cuda:{self.gpu}'
        weight = weight.to(device=this_device)

        if optimizer == "sgd":
            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        elif optimizer == "adam":
            self.optimizer = ch.optim.Adam(param_groups)
        if prediction_type == "classification":
            # self.loss = ch.nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
            self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif prediction_type == "regression":
            # self.loss = ch.nn.MSELoss()
            self.loss = ch.nn.L1Loss()

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    @param('mlp.target_file')
    @param('data.train_proportion')
    @param('data.seed')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory, target_file, train_proportion, seed):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()
        self.max_block_drop_per_class = ch.load(target_file)
        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        subset_indices = None
        if train_proportion < 1:
            dummy_loader = Loader(train_dataset,
                batch_size=1,
                num_workers=num_workers,
                order=OrderOption.RANDOM,
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
            subset_size = int(dataset_size * train_proportion)
            np.random.default_rng(seed)
            np.random.seed(seed)
            subset_indices = np.random.choice(
                dataset_size, 
                subset_size, 
                replace=False)
            print(f"Subset size: {subset_size} - indices: {subset_indices[:5]}")

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        indices=subset_indices,
                        seed=seed,
                        distributed=distributed)

        return loader

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    @param('data.in_memory')
    @param('data.val_proportion')
    @param('data.seed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed, in_memory, val_proportion, seed):
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
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        subset_indices = None
        if val_proportion < 1:
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
            subset_size = int(dataset_size * val_proportion)
            np.random.default_rng(seed)
            np.random.seed(seed)
            subset_indices = np.random.choice(
                dataset_size, 
                subset_size, 
                replace=False)
            print(f"Subset size: {subset_size} - indices: {subset_indices[:5]}")

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        indices=subset_indices,
                        seed=seed,
                        distributed=distributed)
        return loader

    @param('training.epochs')
    @param('logging.log_level')
    def train(self, epochs, log_level):
        for epoch in range(epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            train_loss = self.train_loop(epoch)

            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
                    'epoch': epoch
                }

                val_stats = self.eval_and_log(extra_dict)

            ch.save({
                "model": self.model.state_dict,
                "score": val_stats,
                "epoch": epoch,
            }, self.log_folder / "current_weights.pt")
        self.eval_and_log({'epoch':epoch})
        if self.gpu == 0:
            ch.save({
                "model": self.model.state_dict,
                "score": val_stats,
                "epoch": epoch,
            }, self.log_folder / "final_weights.pt")

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats['top_5'],
                'val_time': val_time
            }, **extra_dict))

        return stats

    @param('model.arch')
    @param('training.distributed')
    @param('training.use_blurpool')
    @param('mlp.prediction_type')
    @param('mlp.trunk_layer')
    @param("mlp.target_file")
    def create_model_and_scaler(self, arch, distributed, use_blurpool, prediction_type, trunk_layer,
        target_file
    ):
        scaler = GradScaler()
        if "swin" in arch:
            import gated_swin
            feature_extractor_loader = getattr(gated_swin, arch)
            weights = getattr(gated_swin, "Swin_B_Weights")
            feature_extractor = feature_extractor_loader(weights=weights.DEFAULT)
            feature_extractor.infer_trunk_only(trunk_layer)
        else:
            raise NotImplementedError()

        if trunk_layer == "first":
            mlp_input_dim = 15, 15
        else:
            mlp_input_dim = 5, 5

        trunk_dim = feature_extractor.trunk_dim
        feature_extractor.require_gradients = False
        num_classes = ch.load(target_file).unique().max() + 1
        output_dim = num_classes if prediction_type == "classification" else 2
        layers = [
            feature_extractor,
            Permute([0, 3, 1, 2]), # N H W C -> N C H W
            ch.nn.AdaptiveAvgPool2d(mlp_input_dim),
            ch.nn.Flatten(),
            MLP(
                mlp_input_dim[0]*mlp_input_dim[1]*trunk_dim, 
                [1024, 1024, 1024, output_dim], 
                activation_layer=ch.nn.GELU, 
                inplace=None, 
                norm_layer=ch.nn.LayerNorm, 
                bias=False
            )
            # MLP(mlp_input_dim[0]*mlp_input_dim[1]*trunk_dim, [1024, 1024, 1024, output_dim], activation_layer=ch.nn.ReLU, inplace=True, bias=True),
        ]
        if prediction_type == "regression":
            sp = F.softplus
            layers.append(LambdaLayer(
                lambda x: sp(x[:, 0]) / (sp(x[:, 0]) + sp(x[:, 1]))
            ))

        model = ch.nn.Sequential(*layers)


        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler


    def get_target_k(self, target):
        return self.max_block_drop_per_class[target].to(device=target.device)
        

    @param('logging.log_level')
    @param("mlp.prediction_type")
    def train_loop(self, epoch, log_level, prediction_type):
        model = self.model
        model.train()
        losses = []

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]


            target_k = self.get_target_k(target)
            self.optimizer.zero_grad(set_to_none=True)
            # with autocast():
            output = self.model(images)
            if prediction_type == "regression":
                target_k = target_k.float() / 24

            loss_train = self.loss(output, target_k)

            # self.scaler.scale(loss_train).backward()
            loss_train.backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            self.optimizer.step()
            ### Training end

            ### Logging start
            if log_level > 0:
                losses.append(loss_train.detach())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss_train.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)
            ### Logging end
        return loss_train.item()

    def scalar_to_onehot(self, scalar_prediction):
        scalar_prediction = ch.clamp(scalar_prediction*24, 0, 24).long()
        onehot = ch.nn.functional.one_hot(scalar_prediction, num_classes=25).float()
        return onehot

    @param('validation.lr_tta')
    @param("mlp.prediction_type")
    def val_loop(self, lr_tta, prediction_type):
        model = self.model
        model.eval()

        
        with ch.no_grad():
            with autocast():
                conf_mat = ch.zeros(25, 25, dtype=int)
                for images, target in tqdm(self.val_loader):
                    target_k = self.get_target_k(target)
                    output = self.model(images)
                    loss_val = self.loss(output, target_k)

                    if prediction_type == "regression":
                        output = self.scalar_to_onehot(output)

                    for p, t in zip(output.argmax(dim=1), target_k):
                        conf_mat[p, t] += 1


                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target_k)

                    self.val_meters['loss'](loss_val)
            
        print(conf_mat.numpy()[:24, :24])
        print("targ", conf_mat.sum(0).numpy())
        print("pred", conf_mat.sum(1).numpy())


        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(compute_on_step=False).to(self.gpu),
            'top_5': torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric(compute_on_step=False).to(self.gpu)
        }

        self.mlp_outputs = {
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
        print(f'=> Log: {content}')
        if self.gpu != 0: return
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
            trainer.train()

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
