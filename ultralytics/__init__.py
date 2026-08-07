# ETFNet-SJPA-TR organized source layout, AGPL-3.0 license

from pathlib import Path

__version__ = "8.0.238+etfnetsjpa.10.8"

# The source repository keeps implementation groups in sibling component roots so no
# committed subfolder contains more than 100 files.  Extending package __path__ keeps
# every public import unchanged (for example, ultralytics.models and ultralytics.data).
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPOSITORY_DIR = _PACKAGE_DIR.parent
_COMPONENT_ROOT_NAMES = ("ultra_modeling", "ultra_runtime", "ultra_services")
for _name in _COMPONENT_ROOT_NAMES:
    _component = _REPOSITORY_DIR / _name
    if _component.is_dir():
        __path__.append(str(_component))

from ultralytics.data.explorer.explorer import Explorer
from ultralytics.models import RTDETR, SAM, YOLO
from ultralytics.models.fastsam import FastSAM
from ultralytics.models.nas import NAS
from ultralytics.utils import SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

__all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'settings', 'Explorer'
