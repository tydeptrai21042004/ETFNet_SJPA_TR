"""Repository layout metadata shared by packaging, docs, and audits."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_FILES_PER_SUBFOLDER = 100
ULTRALYTICS_COMPONENT_ROOTS = (
    ROOT / "ultralytics",
    ROOT / "ultra_modeling",
    ROOT / "ultra_runtime",
    ROOT / "ultra_services",
)
DOC_SOURCE_PACKS = {
    "en": (
        (ROOT / "docs_en_main", Path(".")),
        (ROOT / "docs_en_yolov5", Path(".")),
        (ROOT / "docs_en_reference_core", Path(".")),
        (ROOT / "docs_en_reference_models", Path(".")),
        (ROOT / "docs_en_reference_utils", Path(".")),
    ),
    "ar": ((ROOT / "docs_i18n_a" / "ar", Path(".")),),
    "de": ((ROOT / "docs_i18n_a" / "de", Path(".")),),
    "es": ((ROOT / "docs_i18n_a" / "es", Path(".")),),
    "fr": ((ROOT / "docs_i18n_b" / "fr", Path(".")),),
    "hi": ((ROOT / "docs_i18n_b" / "hi", Path(".")),),
    "ja": ((ROOT / "docs_i18n_b" / "ja", Path(".")),),
    "ko": ((ROOT / "docs_i18n_c" / "ko", Path(".")),),
    "pt": ((ROOT / "docs_i18n_c" / "pt", Path(".")),),
    "ru": ((ROOT / "docs_i18n_c" / "ru", Path(".")),),
    "zh": ((ROOT / "docs_i18n_d" / "zh", Path(".")),),
}
