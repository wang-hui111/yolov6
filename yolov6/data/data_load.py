#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

import os
from torch.utils.data import dataloader, distributed

from .datasets import TrainValDataset
from .dataset_v5 import LoadImagesAndLabels
from yolov6.utils.events import LOGGER
from yolov6.utils.torch_utils import torch_distributed_zero_first


def create_dataloader(
    path,#'../custom_dataset/images/train'
    img_size,#640
    batch_size,#32
    stride,#32
    hyp=None,
    augment=False,
    check_images=False,
    check_labels=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=4,#8
    shuffle=False,
    #{'train': '../custom_dataset/images/train', 'val': '../custom_dataset/images/train', 'test': '../custom_dataset/images/test', 'is_coco': False, 'nc': 80, 'names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']}
    data_dict=None,
    task="Train",
):
    """Create general dataloader.

    Returns dataloader and dataset
    """
    if rect and shuffle:
        LOGGER.warning(
            "WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False"
        )
        shuffle = False
    # class LoadImagesAndLabels(Dataset):  # for training/testing
    #     def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False,
    #                  image_weights=False,
    #                  cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1):

    # with torch_distributed_zero_first(rank):                                             yolov5
    #     dataset = LoadImagesAndLabels(path, imgsz, batch_size,
    #                                   augment=augment,  # augment images
    #                                   hyp=hyp,  # augmentation hyperparameters
    #                                   rect=rect,  # rectangular training
    #                                   cache_images=cache,
    #                                   single_cls=opt.single_cls,
    #                                   stride=int(stride),
    #                                   pad=pad,
    #                                   rank=rank)

    with torch_distributed_zero_first(rank):
        # img_dir,#'../custom_dataset/images/train'
        # img_size=640,#640
        # batch_size=16,#1
        # augment=False,#True
        # hyp=None,#{'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0}
        # rect=False,#False
        # check_images=False,#False
        # check_labels=False,##False
        # stride=32,#32
        # pad=0.0,#0.0
        # rank=-1,#-1
        # data_dict=None,#{'train': '../custom_dataset/images/train', 'val': '../custom_dataset/images/train', 'test': '../custom_dataset/images/test', 'is_coco': False, 'nc': 80, 'names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']}
        # task="train",
        dataset = LoadImagesAndLabels(
            path,
            img_size,
            batch_size,
            augment=augment,#True
            hyp=hyp,#{'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0}
            rect=rect,#False
            stride=int(stride),#32
            pad=pad,#0
            rank=rank,#-1
        )


        # dataset = TrainValDataset(
        #     path,
        #     img_size,
        #     batch_size,
        #     augment=augment,
        #     hyp=hyp,
        #     rect=rect,
        #     check_images=check_images,
        #     check_labels=check_labels,
        #     stride=int(stride),
        #     pad=pad,
        #     rank=rank,
        #     data_dict=data_dict,
        #     task=task,)
        # print("dataset:",dataset)#<yolov6.data.datasets.TrainValDataset object at 0x7fe13fe7d6d0>

    batch_size = min(batch_size, len(dataset))
    workers = min(
        [
            os.cpu_count() // int(os.getenv("WORLD_SIZE", 1)),
            batch_size if batch_size > 1 else 0,
            workers,
        ]
    )  # number of workers
    sampler = (
        None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    )
    return (
        TrainValDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=workers,
            sampler=sampler,
            pin_memory=True,
            collate_fn=TrainValDataset.collate_fn,
        ),
        dataset,
    )


class TrainValDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
