import numpy as np
import torch

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from torchvision.ops.misc import MLP, Permute

from pathlib import Path
from typing import List

import gated_swin

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

def create_train_loader(train_dataset, num_workers, batch_size,
                        distributed, in_memory, target_file, train_proportion, seed):
    this_device = f'cuda'
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
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
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
        print(f"Subset size: {subset_size} - indices: {subset_indices}")

    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    loader = Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    indices=subset_indices,
                    seed=seed,
                    distributed=distributed)

    return loader

def create_val_loader(val_dataset, num_workers, batch_size,
                        resolution, distributed, in_memory, val_proportion, seed):
    this_device = f'cuda'
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
    if val_proportion < 1:
        dummy_loader = Loader(val_dataset,
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
        subset_size = int(dataset_size * val_proportion)
        np.random.default_rng(seed)
        np.random.seed(seed)
        subset_indices = np.random.choice(
            dataset_size, 
            subset_size, 
            replace=False)
        print(f"Subset size: {subset_size} - indices: {subset_indices}")

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


feature_extractor_loader = getattr(gated_swin, "swin_b")
weights = getattr(gated_swin, "Swin_B_Weights")
feature_extractor = feature_extractor_loader(weights=weights.DEFAULT)

mlp_input_dim = 20, 20
trunk_dim = feature_extractor.trunk_dim
feature_extractor.require_gradients = False
model = torch.nn.Sequential(
    feature_extractor,
    Permute([0, 3, 1, 2]), # N H W C -> N C H W
    torch.nn.AdaptiveAvgPool2d(mlp_input_dim),
    torch.nn.Flatten(),
    MLP(mlp_input_dim[0]*mlp_input_dim[1]*trunk_dim, [1024, 24], activation_layer=torch.nn.GELU, inplace=None, norm_layer=torch.nn.LayerNorm)
)


