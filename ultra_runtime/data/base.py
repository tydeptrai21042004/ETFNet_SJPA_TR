# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM

from .rgb_ir import (build_ir_file_list, is_rgb_ir_channels, paired_cache_path, pairing_options,
                     read_rgb_ir_pair, validate_hwc_modalities)
from .utils import HELP_URL, IMG_FORMATS


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(self,
                 img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=DEFAULT_CFG,
                 prefix='',
                 rect=False,
                 batch_size=16,
                 stride=32,
                 pad=0.5,
                 single_cls=False,
                 classes=None,
                 fraction=1.0):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.hyp = hyp
        self.im_files, self.la_files = self.get_img_files(self.img_path)
        self._ir_by_rgb = dict(zip(self.im_files, self.la_files)) if self.la_files else {}
        self.labels = self.get_labels()
        if is_rgb_ir_channels(getattr(self.hyp, 'ch', 3)):
            # Label verification may remove corrupt RGB files. Rebuild the IR list
            # from the original mapping so both arrays remain index-aligned.
            self.la_files = [self._ir_by_rgb[x] for x in self.im_files]
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.k = 4
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images
        if cache == 'ram' and not self.check_cache_ram():
            cache = False
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        pair_opts = pairing_options(getattr(self, 'data', None))
        cache_suffix = pair_opts.get('cache_suffix', '.rgbir.npy')
        self.npy_files = ([paired_cache_path(rgb, cache_suffix, ir_path=ir, resize_ir=pair_opts['resize_ir'])
                           for rgb, ir in zip(self.im_files, self.la_files)]
                          if is_rgb_ir_channels(getattr(self.hyp, 'ch', 3))
                          else [Path(f).with_suffix('.npy') for f in self.im_files])
        if cache:
            self.cache_images(cache)

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read RGB files and resolve index-aligned IR partners for six-channel input."""
        try:
            f = []
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():
                    with open(p, encoding='utf-8') as t:
                        lines = t.read().strip().splitlines()
                    for line in lines:
                        text = line.strip()
                        if not text:
                            continue
                        candidate = Path(text).expanduser()
                        if not candidate.is_absolute():
                            candidate = p.parent / candidate
                        f.append(str(candidate.resolve()))
                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            assert im_files, f'{self.prefix}No images found in {img_path}'
            if self.fraction < 1:
                im_files = im_files[:round(len(im_files) * self.fraction)]
            la_files = (build_ir_file_list(im_files, img_path, getattr(self, 'data', None))
                        if is_rgb_ir_channels(getattr(self.hyp, 'ch', 3)) else [])
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        return im_files, la_files

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]['cls']
                bboxes = self.labels[i]['bboxes']
                segments = self.labels[i]['segments']
                keypoints = self.labels[i]['keypoints']
                j = (cls == include_class_array).any(1)
                self.labels[i]['cls'] = cls[j]
                self.labels[i]['bboxes'] = bboxes[j]
                if segments:
                    self.labels[i]['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]['keypoints'] = keypoints[j]
            if self.single_cls:
                self.labels[i]['cls'][:, 0] = 0

    def _read_source_image(self, i):
        """Read one raw 3- or 6-channel source image in canonical HWC/BGR order."""
        if is_rgb_ir_channels(getattr(self.hyp, 'ch', 3)):
            opts = pairing_options(getattr(self, 'data', None))
            image = read_rgb_ir_pair(self.im_files[i], self.la_files[i], resize_ir=opts['resize_ir'])
        else:
            image = cv2.imread(self.im_files[i])
            if image is None:
                raise FileNotFoundError(f'Image Not Found {self.im_files[i]}')
        validate_hwc_modalities(image, int(getattr(self.hyp, 'ch', image.shape[2])))
        return image

    def load_image_1(self, i, rect_mode=True):
        """Backward-compatible alias for :meth:`load_image`."""
        return self.load_image(i, rect_mode=rect_mode)

    def load_image_0(self, i, rect_mode=True):
        """Backward-compatible alias for :meth:`load_image`."""
        return self.load_image(i, rect_mode=rect_mode)

    def load_image(self, i, rect_mode=True):
        """Load one RGB or paired RGB--IR image and return resized geometry."""
        im, fn = self.ims[i], self.npy_files[i]
        if im is None:
            if fn.exists():
                try:
                    source_paths = [Path(self.im_files[i])]
                    if is_rgb_ir_channels(getattr(self.hyp, 'ch', 3)):
                        source_paths.append(Path(self.la_files[i]))
                    if any(p.stat().st_mtime_ns > fn.stat().st_mtime_ns for p in source_paths):
                        raise RuntimeError('cache is older than its RGB/IR source')
                    im = np.load(fn, allow_pickle=False)
                    validate_hwc_modalities(im, int(getattr(self.hyp, 'ch', im.shape[2])))
                except Exception as e:
                    LOGGER.warning(f'{self.prefix}WARNING ⚠️ Removing stale/corrupt/incompatible cache {fn}: {e}')
                    fn.unlink(missing_ok=True)
                    im = self._read_source_image(i)
            else:
                im = self._read_source_image(i)

            h0, w0 = im.shape[:2]
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk if cache == 'disk' else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{self.prefix}Caching images ({b / gb:.1f}GB {cache})'
            pbar.close()

    def cache_images_to_disk(self, i):
        """Atomically cache the complete paired array, never only the RGB half."""
        f = self.npy_files[i]
        source_paths = [Path(self.im_files[i])]
        if is_rgb_ir_channels(getattr(self.hyp, 'ch', 3)):
            source_paths.append(Path(self.la_files[i]))
        fresh = f.exists() and all(p.stat().st_mtime_ns <= f.stat().st_mtime_ns for p in source_paths)
        if not fresh:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f'{f.name}.{os.getpid()}.tmp.npy')
            np.save(tmp.as_posix(), self._read_source_image(i), allow_pickle=False)
            os.replace(tmp, f)

    def check_cache_ram(self, safety_margin=0.5):
        """Estimate RAM using complete 3- or 6-channel samples."""
        b, gb = 0, 1 << 30
        n = min(self.ni, 30)
        for i in random.sample(range(self.ni), n):
            im = self._read_source_image(i)
            ratio = self.imgsz / max(im.shape[0], im.shape[1])
            b += im.nbytes * ratio ** 2
        mem_required = b * self.ni / max(n, 1) * (1 + safety_margin)
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available
        if not cache:
            LOGGER.info(f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                        f'with {int(safety_margin * 100)}% safety margin but only '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️')
        return cache

    def set_rectangle(self):
        """Set rectangular batch shapes while preserving RGB/IR index alignment."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)
        nb = bi[-1] + 1
        shapes_array = np.array([x.pop('shape') for x in self.labels])
        aspect_ratio = shapes_array[:, 0] / shapes_array[:, 1]
        order = aspect_ratio.argsort()
        self.im_files = [self.im_files[i] for i in order]
        if self.la_files:
            self.la_files = [self.la_files[i] for i in order]
        self.labels = [self.labels[i] for i in order]
        aspect_ratio = aspect_ratio[order]

        shapes = [[1, 1]] * nb
        for i in range(nb):
            batch_ar = aspect_ratio[bi == i]
            mini, maxi = batch_ar.min(), batch_ar.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]
        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi

    def __getitem__(self, index):
        """Return transformed label information for one sample."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Load an image pair and its labels."""
        label = deepcopy(self.labels[index])
        label.pop('shape', None)
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError
