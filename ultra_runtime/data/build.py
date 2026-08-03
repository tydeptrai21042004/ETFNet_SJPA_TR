# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.data.loaders import (LOADERS, LoadImages, LoadPilAndNumpy, LoadRGBIRStreams, LoadScreenshots,
                                      LoadStreams, LoadTensor, SourceTypes, autocast_list)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file

from .dataset import YOLODataset
from .rgb_ir import is_rgb_ir_channels
from .utils import PIN_MEMORY


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


class EpochRandomSampler(torch.utils.data.Sampler):
    """Deterministic random sampler whose permutation is a pure function of seed and epoch."""

    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self):
        return len(self.data_source)


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """Build YOLO Dataset."""
    return YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0)


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1, seed=0):
    """Return a reproducibly seeded infinite dataloader.

    The sample order is determined by ``seed + epoch`` rather than by hidden
    iterator state, so a run resumed at an epoch boundary receives the same
    ordering as an uninterrupted run.
    """
    if len(dataset) == 0:
        raise ValueError('Cannot build a dataloader from an empty dataset')
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()
    cpu_count = os.cpu_count() or 1
    nw = min(cpu_count // max(nd, 1), batch, max(int(workers), 0))
    if rank == -1:
        sampler = EpochRandomSampler(dataset, seed=seed) if shuffle else None
    else:
        sampler = distributed.DistributedSampler(dataset, shuffle=shuffle, seed=int(seed), drop_last=False)
    generator = torch.Generator()
    generator.manual_seed(int(seed) + max(int(rank), 0) * 100003)
    return InfiniteDataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=False if sampler is not None else bool(shuffle),
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY and torch.cuda.is_available(),
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator,
                              persistent_workers=bool(nw > 0))


def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://', 'tcp://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower() == 'screen'
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError('Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict')

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, imgsz=640, vid_stride=1, buffer=False, ir_source=None,
                          data=None, ch=3, resize_ir=False):
    """Load an inference source with optional synchronized RGB--IR pairing."""
    source, webcam, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(webcam, screenshot, from_img, tensor)

    if tensor:
        dataset = LoadTensor(source)
        if is_rgb_ir_channels(ch) and dataset.im0.shape[1] != int(ch):
            raise ValueError(f'Expected tensor with {ch} channels, got {tuple(dataset.im0.shape)}')
    elif in_memory:
        dataset = source
    elif webcam:
        dataset = (LoadRGBIRStreams(source, ir_source, imgsz=imgsz, vid_stride=vid_stride, buffer=buffer,
                                    resize_ir=resize_ir)
                   if is_rgb_ir_channels(ch) else LoadStreams(source, imgsz=imgsz, vid_stride=vid_stride, buffer=buffer))
    elif screenshot:
        if is_rgb_ir_channels(ch):
            raise ValueError('Paired screenshot inference is not supported. Pass a six-channel NumPy array instead.')
        dataset = LoadScreenshots(source, imgsz=imgsz)
    elif from_img:
        dataset = LoadPilAndNumpy(source, imgsz=imgsz, ir_source=ir_source, ch=ch, resize_ir=resize_ir)
    else:
        dataset = LoadImages(source, imgsz=imgsz, vid_stride=vid_stride, ir_source=ir_source,
                             data=data, ch=ch, resize_ir=resize_ir)

    setattr(dataset, 'source_type', source_type)
    return dataset
