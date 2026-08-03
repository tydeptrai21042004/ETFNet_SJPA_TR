from pathlib import Path

import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.data.public_multidataset import ADDITIONAL_DATASETS
from ultralytics.nn.modules.block import SJPA
from ultralytics.utils import DEFAULT_CFG_PATH, ROOT


def test_split_source_tree_preserves_public_imports_and_resources():
    expected = {"m3fd", "vedai", "flir-aligned", "rgbtdroneperson", "cvc-14"}
    assert expected.issubset(ADDITIONAL_DATASETS)
    assert SJPA.__name__ == "SJPA"
    assert ROOT == Path(ultralytics.__file__).resolve().parent
    assert DEFAULT_CFG_PATH.is_file()
    assert (ROOT / "cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml").is_file()


def test_split_source_tree_builds_and_runs_sjpa_model():
    model = YOLO(str(ROOT / "cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml"), task="obb")
    model.model.eval()
    with torch.no_grad():
        output = model.model(torch.zeros(1, 6, 64, 64))
    assert output is not None
