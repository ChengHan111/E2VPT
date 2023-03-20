#!/usr/bin/env python3

"""Data loader."""
import torch
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torch.utils.data.sampler import RandomSampler

from ..utils import logging
from .datasets.json_dataset import (
    CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset
)

logger = logging.get_logger("visual_prompt")
_DATASET_CATALOG = {
    "CUB": CUB200Dataset,
    'OxfordFlowers': FlowersDataset,
    'StanfordCars': CarsDataset,
    'StanfordDogs': DogsDataset,
    "nabirds": NabirdsDataset,
}


def _construct_loader(cfg, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = cfg.DATA.NAME

    # Construct the dataset
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        from .datasets.tf_dataset import TFDataset
        dataset = TFDataset(cfg, split)
        # print('dataset check', dataset)
    elif "imagenet1k" in dataset_name:
        if cfg.DATA.CROPSIZE == 224:
            resize_dim = 256
            crop_dim = 224
        elif cfg.DATA.CROPSIZE == 448:
            resize_dim = 512
            crop_dim = 448
        else: # cfg.DATA.CROPSIZE == 384
            resize_dim = 438
            crop_dim = 384
            
        if split == "train":
            imagenet_path = "/shared/kgcoe-research/spl/imagenet1k/train"
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_dim),
                torchvision.transforms.RandomCrop(crop_dim),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            imagenet_data_train = torchvision.datasets.ImageFolder(imagenet_path, 
                    transform=train_transform)

            sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

            # create a loader
            train_dataloader = torch.utils.data.DataLoader(imagenet_data_train,
                                            batch_size=batch_size,
                                            shuffle=(False if sampler else shuffle),
                                            sampler=sampler,
                                            num_workers=cfg.DATA.NUM_WORKERS,
                                            pin_memory=cfg.DATA.PIN_MEMORY,
                                            drop_last=drop_last,)
            return train_dataloader
    
        elif split == "val":
            imagenet_path = "/shared/kgcoe-research/spl/imagenet1k/val"
            val_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_dim),
                torchvision.transforms.CenterCrop(crop_dim),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            imagenet_data_val = torchvision.datasets.ImageFolder(imagenet_path, 
                    transform=val_transform)
            
            sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

            # create a loader
            val_dataloader = torch.utils.data.DataLoader(imagenet_data_val,
                                            batch_size=batch_size,
                                            shuffle=(False if sampler else shuffle),
                                            sampler=sampler,
                                            num_workers=cfg.DATA.NUM_WORKERS,
                                            pin_memory=cfg.DATA.PIN_MEMORY,
                                            drop_last=drop_last,)
            return val_dataloader
        
        elif split == "test":
            imagenet_path = "/shared/kgcoe-research/spl/imagenet1k/val"
            test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_dim),
                torchvision.transforms.CenterCrop(crop_dim),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            imagenet_data_test = torchvision.datasets.ImageFolder(imagenet_path, 
                    transform=test_transform)

            sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

            # create a loader
            test_dataloader = torch.utils.data.DataLoader(imagenet_data_test,
                                            batch_size=batch_size,
                                            shuffle=(False if sampler else shuffle),
                                            sampler=sampler,
                                            num_workers=cfg.DATA.NUM_WORKERS,
                                            pin_memory=cfg.DATA.PIN_MEMORY,
                                            drop_last=drop_last,)
            return test_dataloader
        
    else:
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)
        dataset = _DATASET_CATALOG[dataset_name](cfg, split)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader(cfg):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_trainval_loader(cfg):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_test_loader(cfg):
    """Test loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="test",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def construct_val_loader(cfg, batch_size=None):
    if batch_size is None:
        bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        bs = batch_size
    """Validation loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
