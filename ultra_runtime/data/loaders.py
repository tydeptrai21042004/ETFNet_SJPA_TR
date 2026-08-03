# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from ultralytics.data.rgb_ir import (is_rgb_ir_channels, load_data_yaml, pairing_options, read_rgb_ir_pair,
                                      resolve_ir_path)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOGGER, is_colab, is_kaggle, ops
from ultralytics.utils.checks import check_requirements


@dataclass
class SourceTypes:
    """Class to represent various types of input sources for predictions."""
    webcam: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False


class LoadStreams:
    """
    Stream Loader for various types of video streams.

    Suitable for use with `yolo predict source='rtsp://example.com/media.mp4'`, supports RTSP, RTMP, HTTP, and TCP streams.

    Attributes:
        sources (str): The source input paths or URLs for the video streams.
        imgsz (int): The image size for processing, defaults to 640.
        vid_stride (int): Video frame-rate stride, defaults to 1.
        buffer (bool): Whether to buffer input streams, defaults to False.
        running (bool): Flag to indicate if the streaming thread is running.
        mode (str): Set to 'stream' indicating real-time capture.
        imgs (list): List of image frames for each stream.
        fps (list): List of FPS for each stream.
        frames (list): List of total frames for each stream.
        threads (list): List of threads for each stream.
        shape (list): List of shapes for each stream.
        caps (list): List of cv2.VideoCapture objects for each stream.
        bs (int): Batch size for processing.

    Methods:
        __init__: Initialize the stream loader.
        update: Read stream frames in daemon thread.
        close: Close stream loader and release resources.
        __iter__: Returns an iterator object for the class.
        __next__: Returns source paths, transformed, and original images for processing.
        __len__: Return the length of the sources object.
    """

    def __init__(self, sources='file.streams', imgsz=640, vid_stride=1, buffer=False):
        """Initialize instance variables and check for consistent input stream shapes."""
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.buffer = buffer  # buffer input streams
        self.running = True  # running flag for Thread
        self.mode = 'stream'
        self.imgsz = imgsz
        self.vid_stride = vid_stride  # video frame-rate stride

        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.fps = [0] * n  # frames per second
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n  # video capture objects
        self.imgs = [[] for _ in range(n)]  # images
        self.shape = [[] for _ in range(n)]  # image shapes
        self.sources = [ops.clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/LNwODJXcvt4'
                s = get_best_youtube_url(s)
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0 and (is_colab() or is_kaggle()):
                raise NotImplementedError("'source=0' webcam not supported in Colab and Kaggle notebooks. "
                                          "Try running 'source=0' in a local environment.")
            self.caps[i] = cv2.VideoCapture(s)  # store video capture object
            if not self.caps[i].isOpened():
                raise ConnectionError(f'{st}Failed to open {s}')
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                'inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            success, im = self.caps[i].read()  # guarantee first frame
            if not success or im is None:
                raise ConnectionError(f'{st}Failed to read images from {s}')
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            LOGGER.info(f'{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)')
            self.threads[i].start()
        LOGGER.info('')  # newline

        # Check for common shapes
        self.bs = self.__len__()

    def update(self, i, cap, stream):
        """Read stream `i` frames in daemon thread."""
        n, f = 0, self.frames[i]  # frame number, frame array
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:  # keep a <=30-image buffer
                n += 1
                cap.grab()  # .read() = .grab() followed by .retrieve()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                        cap.open(stream)  # re-open stream if signal was lost
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)  # wait until the buffer is empty

    def close(self):
        """Close stream loader and release resources."""
        self.running = False  # stop flag for Thread
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # Add timeout
        for cap in self.caps:  # Iterate through the stored VideoCapture objects
            try:
                cap.release()  # release video capture
            except Exception as e:
                LOGGER.warning(f'WARNING ⚠️ Could not release VideoCapture object: {e}')
        cv2.destroyAllWindows()

    def __iter__(self):
        """Iterates through YOLO image feed and re-opens unresponsive streams."""
        self.count = -1
        return self

    def __next__(self):
        """Returns source paths, transformed and original images for processing."""
        self.count += 1

        images = []
        for i, x in enumerate(self.imgs):

            # Wait until a frame is available in each buffer
            while not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord('q'):  # q to quit
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f'WARNING ⚠️ Waiting for stream {i}')

            # Get and remove the first frame from imgs buffer
            if self.buffer:
                images.append(x.pop(0))

            # Get the last frame, and clear the rest from the imgs buffer
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, None, ''

    def __len__(self):
        """Return the length of the sources object."""
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadRGBIRStreams:
    """Synchronize two :class:`LoadStreams` instances and concatenate their frames."""

    def __init__(self, rgb_sources, ir_sources, imgsz=640, vid_stride=1, buffer=False, resize_ir=False):
        if ir_sources in (None, ''):
            raise ValueError('Paired live inference requires ir_source=<IR camera/stream>.')
        self.rgb = LoadStreams(rgb_sources, imgsz=imgsz, vid_stride=vid_stride, buffer=buffer)
        self.ir = LoadStreams(ir_sources, imgsz=imgsz, vid_stride=vid_stride, buffer=buffer)
        if len(self.rgb) != len(self.ir):
            self.close()
            raise ValueError(f'RGB stream count {len(self.rgb)} != IR stream count {len(self.ir)}')
        self.sources = self.rgb.sources
        self.bs = self.rgb.bs
        self.mode = 'stream'
        self.resize_ir = bool(resize_ir)
        self.count = -1

    def __iter__(self):
        self.rgb_iter, self.ir_iter = iter(self.rgb), iter(self.ir)
        self.count = -1
        return self

    def __next__(self):
        rgb_paths, rgb_images, _, rgb_log = next(self.rgb_iter)
        _, ir_images, _, _ = next(self.ir_iter)
        paired = []
        for index, (rgb, ir) in enumerate(zip(rgb_images, ir_images)):
            if rgb.shape[:2] != ir.shape[:2]:
                if not self.resize_ir:
                    raise ValueError(f'Live RGB/IR size mismatch at stream {index}: {rgb.shape[:2]} vs {ir.shape[:2]}')
                ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            paired.append(np.concatenate((rgb, ir), axis=2))
        self.count += 1
        return rgb_paths, paired, None, rgb_log

    def close(self):
        self.rgb.close()
        self.ir.close()

    def __len__(self):
        return self.bs


class LoadScreenshots:
    """
    YOLOv8 screenshot dataloader.

    This class manages the loading of screenshot images for processing with YOLOv8.
    Suitable for use with `yolo predict source=screen`.

    Attributes:
        source (str): The source input indicating which screen to capture.
        imgsz (int): The image size for processing, defaults to 640.
        screen (int): The screen number to capture.
        left (int): The left coordinate for screen capture area.
        top (int): The top coordinate for screen capture area.
        width (int): The width of the screen capture area.
        height (int): The height of the screen capture area.
        mode (str): Set to 'stream' indicating real-time capture.
        frame (int): Counter for captured frames.
        sct (mss.mss): Screen capture object from `mss` library.
        bs (int): Batch size, set to 1.
        monitor (dict): Monitor configuration details.

    Methods:
        __iter__: Returns an iterator object.
        __next__: Captures the next screenshot and returns it.
    """

    def __init__(self, source, imgsz=640):
        """Source = [screen_number left top width height] (pixels)."""
        check_requirements('mss')
        import mss  # noqa

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.imgsz = imgsz
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor['top'] if top is None else (monitor['top'] + top)
        self.left = monitor['left'] if left is None else (monitor['left'] + left)
        self.width = width or monitor['width']
        self.height = height or monitor['height']
        self.monitor = {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}

    def __iter__(self):
        """Returns an iterator of the object."""
        return self

    def __next__(self):
        """mss screen capture: get raw pixels from the screen as np array."""
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]  # BGRA to BGR
        s = f'screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: '

        self.frame += 1
        return [str(self.screen)], [im0], None, s  # screen, img, vid_cap, string


class LoadImages:
    """Load RGB images/videos, optionally paired with an IR source."""

    def __init__(self, path, imgsz=640, vid_stride=1, ir_source=None, data=None, ch=3, resize_ir=False):
        rgb_values, rgb_parent, rgb_manifest = self._expand_manifest(path)
        self.rgb_source = path
        self.ir_source = ir_source
        self.data = load_data_yaml(data) if data not in (None, '', False) else {}
        self.ch = int(ch)
        self.resize_ir = bool(resize_ir or pairing_options(self.data)['resize_ir'])
        self.imgsz = imgsz
        self.vid_stride = int(vid_stride)

        rgb_files = self._supported_files(self._collect(rgb_values, parent=rgb_parent))
        if not rgb_files:
            raise FileNotFoundError(
                f'No images or videos found in {path}. Supported image formats: {IMG_FORMATS}; videos: {VID_FORMATS}')

        pairs = [(rgb, None) for rgb in rgb_files]
        if is_rgb_ir_channels(self.ch):
            if ir_source not in (None, ''):
                ir_values, ir_parent, ir_manifest = self._expand_manifest(ir_source)
                ir_files = self._supported_files(self._collect(ir_values, parent=ir_parent))
                direct_index_pairing = rgb_manifest or ir_manifest or (
                    self._is_file_sequence(rgb_values) and self._is_file_sequence(ir_values))
                if direct_index_pairing:
                    if len(rgb_files) != len(ir_files):
                        raise ValueError(f'RGB source count {len(rgb_files)} != IR source count {len(ir_files)}')
                    pairs = list(zip(rgb_files, ir_files))
                elif len(rgb_files) == 1 and len(ir_files) == 1:
                    pairs = [(rgb_files[0], ir_files[0])]
                else:
                    rgb_roots = self._source_roots(rgb_values, rgb_parent)
                    ir_roots = self._source_roots(ir_values, ir_parent)
                    pairs = []
                    for rgb in rgb_files:
                        ir = resolve_ir_path(rgb, data=self.data, rgb_roots=rgb_roots, ir_roots=ir_roots)
                        if not ir.is_file():
                            raise FileNotFoundError(f'IR partner does not exist for {rgb}: {ir}')
                        pairs.append((rgb, str(ir)))
            else:
                rgb_roots = self._source_roots(rgb_values, rgb_parent)
                pairs = []
                for rgb in rgb_files:
                    ir = resolve_ir_path(rgb, data=self.data, rgb_roots=rgb_roots)
                    if not ir.is_file():
                        raise FileNotFoundError(f'IR partner does not exist for {rgb}: {ir}')
                    pairs.append((rgb, str(ir)))

        image_pairs = [pair for pair in pairs if Path(pair[0]).suffix[1:].lower() in IMG_FORMATS]
        video_pairs = [pair for pair in pairs if Path(pair[0]).suffix[1:].lower() in VID_FORMATS]
        pairs = image_pairs + video_pairs
        self.files = [pair[0] for pair in pairs]
        self.ir_files = [str(pair[1]) for pair in pairs] if is_rgb_ir_channels(self.ch) else []
        ni, nv = len(image_pairs), len(video_pairs)
        self.nf = ni + nv
        self.video_flag = [False] * ni + [True] * nv
        if self.ir_files:
            ir_flags = [Path(x).suffix[1:].lower() in VID_FORMATS for x in self.ir_files]
            if ir_flags != self.video_flag:
                raise ValueError('Each RGB image/video must be paired with the same source type in IR.')
        self.mode = 'image'
        self.bs = 1
        self.cap = self.ir_cap = None
        self.frame = 0
        if video_pairs:
            first_video = self.video_flag.index(True)
            self._new_video(self.files[first_video], self.ir_files[first_video] if self.ir_files else None)

    @staticmethod
    def _expand_manifest(source):
        if isinstance(source, (str, Path)) and Path(str(source)).suffix.lower() == '.txt' and Path(str(source)).is_file():
            manifest = Path(str(source)).expanduser().resolve()
            return manifest.read_text(encoding='utf-8').splitlines(), manifest.parent, True
        return source, None, False

    @staticmethod
    def _is_file_sequence(source):
        values = source if isinstance(source, (list, tuple)) else [source]
        return bool(values) and all(Path(str(x)).expanduser().is_file() for x in values if str(x).strip())

    @staticmethod
    def _supported_files(files):
        supported = set(IMG_FORMATS + VID_FORMATS)
        return [x for x in files if Path(x).suffix[1:].lower() in supported]

    @staticmethod
    def _collect(path, parent=None):
        files = []
        values = path if isinstance(path, (list, tuple)) else [path]
        for item in values:
            p = str(item).strip()
            if not p:
                continue
            candidate = str(Path(p).absolute())
            if '*' in candidate:
                files.extend(sorted(glob.glob(candidate, recursive=True)))
            elif os.path.isdir(candidate):
                files.extend(sorted(glob.glob(os.path.join(candidate, '**', '*.*'), recursive=True)))
            elif os.path.isfile(candidate):
                files.append(candidate)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))
            else:
                raise FileNotFoundError(f'{p} does not exist')
        return files

    @staticmethod
    def _source_roots(source, parent=None):
        roots = []
        for item in source if isinstance(source, (list, tuple)) else [source]:
            p = Path(str(item)).expanduser()
            if not p.is_absolute() and parent is not None:
                p = parent / p
            p = p.absolute()
            roots.append(p if p.is_dir() else p.parent)
        return roots

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        ir_path = self.ir_files[self.count] if self.ir_files else None

        if self.video_flag[self.count]:
            self.mode = 'video'
            for _ in range(self.vid_stride):
                rgb_ok = self.cap.grab()
                ir_ok = self.ir_cap.grab() if self.ir_cap is not None else True
            success, rgb = self.cap.retrieve() if rgb_ok else (False, None)
            ir_success, ir = self.ir_cap.retrieve() if self.ir_cap is not None and ir_ok else (True, None)
            while not success or not ir_success:
                self.count += 1
                self._release_video()
                if self.count == self.nf:
                    raise StopIteration
                path = self.files[self.count]
                ir_path = self.ir_files[self.count] if self.ir_files else None
                self._new_video(path, ir_path)
                success, rgb = self.cap.read()
                ir_success, ir = self.ir_cap.read() if self.ir_cap is not None else (True, None)
            self.frame += 1
            image = self._combine(rgb, ir)
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '
        else:
            self.mode = 'image'
            self.count += 1
            if ir_path is not None:
                image = read_rgb_ir_pair(path, ir_path, resize_ir=self.resize_ir)
            else:
                image = cv2.imread(path)
                if image is None:
                    raise FileNotFoundError(f'Image Not Found {path}')
            s = f'image {self.count}/{self.nf} {path}: '
        return [path], [image], self.cap, s

    def _combine(self, rgb, ir):
        if rgb is None:
            raise FileNotFoundError('Could not read RGB video frame')
        if ir is None:
            return rgb
        if rgb.shape[:2] != ir.shape[:2]:
            if not self.resize_ir:
                raise ValueError(f'RGB/IR video frame mismatch: {rgb.shape[:2]} vs {ir.shape[:2]}')
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return np.concatenate((rgb, ir), axis=2)

    def _new_video(self, path, ir_path=None):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f'Could not open video: {path}')
        self.ir_cap = cv2.VideoCapture(ir_path) if ir_path else None
        if self.ir_cap is not None and not self.ir_cap.isOpened():
            self.cap.release()
            raise FileNotFoundError(f'Could not open IR video: {ir_path}')
        rgb_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        ir_frames = int(self.ir_cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride) if self.ir_cap is not None else rgb_frames
        self.frames = min(rgb_frames, ir_frames)

    def _release_video(self):
        if self.cap is not None:
            self.cap.release()
        if self.ir_cap is not None:
            self.ir_cap.release()

    def __len__(self):
        return self.nf


class LoadPilAndNumpy:
    """
    Load images from PIL and Numpy arrays for batch processing.

    This class is designed to manage loading and pre-processing of image data from both PIL and Numpy formats.
    It performs basic validation and format conversion to ensure that the images are in the required format for
    downstream processing.

    Attributes:
        paths (list): List of image paths or autogenerated filenames.
        im0 (list): List of images stored as Numpy arrays.
        imgsz (int): Image size, defaults to 640.
        mode (str): Type of data being processed, defaults to 'image'.
        bs (int): Batch size, equivalent to the length of `im0`.
        count (int): Counter for iteration, initialized at 0 during `__iter__()`.

    Methods:
        _single_check(im): Validate and format a single image to a Numpy array.
    """

    def __init__(self, im0, imgsz=640, ir_source=None, ch=3, resize_ir=False):
        """Initialize PIL/Numpy input, optionally with an index-aligned IR batch."""
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [getattr(im, 'filename', f'image{i}.jpg') for i, im in enumerate(im0)]
        rgb_images = [self._single_check(im) for im in im0]
        if is_rgb_ir_channels(ch):
            if ir_source is None:
                if not all(x.ndim == 3 and x.shape[2] == 6 for x in rgb_images):
                    raise ValueError('Six-channel array input or ir_source is required for an RGB--IR model.')
                self.im0 = rgb_images
            else:
                ir_values = ir_source if isinstance(ir_source, list) else [ir_source]
                if len(ir_values) != len(rgb_images):
                    raise ValueError(f'RGB input count {len(rgb_images)} != IR input count {len(ir_values)}')
                ir_images = [self._single_check(im) for im in ir_values]
                self.im0 = []
                for rgb, ir in zip(rgb_images, ir_images):
                    if rgb.shape[:2] != ir.shape[:2]:
                        if not resize_ir:
                            raise ValueError(f'RGB/IR array size mismatch: {rgb.shape[:2]} vs {ir.shape[:2]}')
                        ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                    self.im0.append(np.concatenate((rgb, ir), axis=2))
        else:
            self.im0 = rgb_images
        self.imgsz = imgsz
        self.mode = 'image'
        # Generate fake paths
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array."""
        assert isinstance(im, (Image.Image, np.ndarray)), f'Expected PIL/np.ndarray image type, but got {type(im)}'
        if isinstance(im, Image.Image):
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def __len__(self):
        """Returns the length of the 'im0' attribute."""
        return len(self.im0)

    def __next__(self):
        """Returns batch paths, images, processed images, None, ''."""
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, None, ''

    def __iter__(self):
        """Enables iteration for class LoadPilAndNumpy."""
        self.count = 0
        return self


class LoadTensor:
    """
    Load images from torch.Tensor data.

    This class manages the loading and pre-processing of image data from PyTorch tensors for further processing.

    Attributes:
        im0 (torch.Tensor): The input tensor containing the image(s).
        bs (int): Batch size, inferred from the shape of `im0`.
        mode (str): Current mode, set to 'image'.
        paths (list): List of image paths or filenames.
        count (int): Counter for iteration, initialized at 0 during `__iter__()`.

    Methods:
        _single_check(im, stride): Validate and possibly modify the input tensor.
    """

    def __init__(self, im0) -> None:
        """Initialize Tensor Dataloader."""
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = 'image'
        self.paths = [getattr(im, 'filename', f'image{i}.jpg') for i, im in enumerate(im0)]

    @staticmethod
    def _single_check(im, stride=32):
        """Validate and format an image to torch.Tensor."""
        s = f'WARNING ⚠️ torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) ' \
            f'divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible.'
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:  # torch.float32 eps is 1.2e-07
            LOGGER.warning(f'WARNING ⚠️ torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. '
                           f'Dividing input by 255.')
            im = im.float() / 255.0

        return im

    def __iter__(self):
        """Returns an iterator object."""
        self.count = 0
        return self

    def __next__(self):
        """Return next item in the iterator."""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, None, ''

    def __len__(self):
        """Returns the batch size."""
        return self.bs


def autocast_list(source):
    """Merges a list of source of different types into a list of numpy arrays or PIL images."""
    files = []
    for im in source:
        if isinstance(im, (str, Path)):  # filename or uri
            files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im))
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL or np Image
            files.append(im)
        else:
            raise TypeError(f'type {type(im).__name__} is not a supported Ultralytics prediction source type. \n'
                            f'See https://docs.ultralytics.com/modes/predict for supported source types.')

    return files


LOADERS = LoadStreams, LoadPilAndNumpy, LoadImages, LoadScreenshots  # tuple


def get_best_youtube_url(url, use_pafy=True):
    """
    Retrieves the URL of the best quality MP4 video stream from a given YouTube video.

    This function uses the pafy or yt_dlp library to extract the video info from YouTube. It then finds the highest
    quality MP4 format that has video codec but no audio codec, and returns the URL of this video stream.

    Args:
        url (str): The URL of the YouTube video.
        use_pafy (bool): Use the pafy package, default=True, otherwise use yt_dlp package.

    Returns:
        (str): The URL of the best quality MP4 video stream, or None if no suitable stream is found.
    """
    if use_pafy:
        check_requirements(('pafy', 'youtube_dl==2020.12.2'))
        import pafy  # noqa
        return pafy.new(url).getbestvideo(preftype='mp4').url
    else:
        check_requirements('yt-dlp')
        import yt_dlp
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)  # extract info
        for f in reversed(info_dict.get('formats', [])):  # reversed because best is usually last
            # Find a format with video codec, no audio, *.mp4 extension at least 1920x1080 size
            good_size = (f.get('width') or 0) >= 1920 or (f.get('height') or 0) >= 1080
            if good_size and f['vcodec'] != 'none' and f['acodec'] == 'none' and f['ext'] == 'mp4':
                return f.get('url')
