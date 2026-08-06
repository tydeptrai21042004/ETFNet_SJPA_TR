from __future__ import annotations

import pytest

from tools.certify_candidate import certify


def test_certificate_is_finite_and_family_wise():
    report = certify({"a": [0.2] * 1000, "b": [0.1] * 1000}, delta=0.05)
    assert report["family_size"] == 2
    assert report["results"]["a"]["lower_confidence_bound"] > 0
    assert report["results"]["b"]["lower_confidence_bound"] < report["results"]["a"]["lower_confidence_bound"]


def test_certificate_rejects_invalid_delta():
    with pytest.raises(ValueError):
        certify({"a": [0.1]}, delta=1.0)
