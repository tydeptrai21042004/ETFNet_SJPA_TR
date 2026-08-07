from __future__ import annotations

import copy
import io

import pytest
import torch
import torch.nn.functional as F

from ultra_modeling.nn.modules.generic_ccprf import (
    MultiInputAdapter,
    UniversalCCPRFBackbone,
    UniversalCCPRFModel,
    UniversalCCPRFSetFusion,
    UniversalInputAdapter,
    ViewSetAggregator,
)


def _small_kwargs(**overrides):
    settings = {
        "input_mode": "single",
        "input_channels": 3,
        "single_strategy": "direct",
        "widths": (16, 32, 48, 64),
        "fusion_stages": ("P2", "P3"),
        "group_width": 8,
        "stat_grid": 6,
        "trust_radius": {"P2": 0.03, "P3": 0.05},
        "local_windows": {"P2": 4, "P3": 2},
    }
    settings.update(overrides)
    return settings


def _tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_tensors(item))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_tensors(item))
        return result
    return []


def test_universal_input_adapter_supports_direct_decomposed_and_named_subsets():
    image = torch.randn(2, 3, 40, 44)
    depth = torch.randn(2, 1, 40, 44)
    thermal = torch.randn(2, 1, 40, 44)

    direct = UniversalInputAdapter(
        mode="single", input_channels=3, out_channels=8
    ).eval()
    decomposed = UniversalInputAdapter(
        mode="decomposed", input_channels=3, out_channels=8
    ).eval()
    named = UniversalInputAdapter(
        mode="multi",
        view_channels={"image": 3, "depth": 1, "thermal": 1},
        out_channels=8,
    ).eval()

    with torch.no_grad():
        one = direct(image)
        two = decomposed(image)
        three = named({"image": image, "depth": depth, "thermal": thermal})
        optional = named({"image": image, "thermal": thermal})

    assert len(one) == 1 and one[0].shape == (2, 8, 20, 22)
    assert len(two) == 2 and all(value.shape == (2, 8, 20, 22) for value in two)
    assert len(three) == 3 and all(value.shape == (2, 8, 20, 22) for value in three)
    assert len(optional) == 2


def test_auto_adapter_accepts_one_tensor_and_physical_view_mapping():
    adapter = UniversalInputAdapter(
        mode="auto",
        input_channels=3,
        out_channels=8,
        single_strategy="direct",
        view_channels={"image": 3, "depth": 1},
    ).eval()
    image = torch.randn(2, 3, 32, 32)
    depth = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        single = adapter(image)
        paired = adapter({"image": image, "depth": depth})
        mapped_single = adapter({"image": image})
    assert len(single) == 1
    assert len(paired) == 2
    assert len(mapped_single) == 1


def test_ordered_multi_input_adapter_validates_count_and_geometry():
    adapter = MultiInputAdapter((3, 1, 2), out_channels=8).eval()
    values = (
        torch.randn(2, 3, 32, 36),
        torch.randn(2, 1, 32, 36),
        torch.randn(2, 2, 32, 36),
    )
    with torch.no_grad():
        output = adapter(values)
    assert len(output) == 3
    with pytest.raises(ValueError):
        adapter(values[:2])
    with pytest.raises(ValueError):
        adapter((values[0], values[1], torch.randn(2, 2, 31, 36)))


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_set_fusion_is_identity_at_initialization_for_any_view_count(count: int):
    torch.manual_seed(100 + count)
    fusion = UniversalCCPRFSetFusion(
        channels=16,
        groups=2,
        stat_grid=6,
        trust_radius=0.04,
        window_size=3,
    ).eval()
    views = tuple(torch.randn(2, 16, 13, 15) for _ in range(count))
    with torch.no_grad():
        output, diagnostics = fusion.forward_with_diagnostics(views)
    assert diagnostics["view_count"] == count
    assert len(output) == count
    for source, result in zip(views, output):
        assert torch.equal(source, result)


def test_set_fusion_many_view_path_is_permutation_equivariant_and_bounded():
    torch.manual_seed(7)
    fusion = UniversalCCPRFSetFusion(
        channels=16,
        groups=2,
        stat_grid=6,
        trust_radius=0.04,
        window_size=0,
    ).eval()
    with torch.no_grad():
        fusion.pair_fusion.fusion.innovation_weight.normal_(0, 0.2)
    views = tuple(torch.randn(2, 16, 12, 10) for _ in range(3))
    permutation = (2, 0, 1)
    with torch.no_grad():
        original = fusion(views)
        shuffled = fusion(tuple(views[index] for index in permutation))
    for shuffled_index, original_index in enumerate(permutation):
        assert torch.allclose(
            shuffled[shuffled_index], original[original_index], atol=3e-5, rtol=3e-5
        )

    for source, result in zip(views, original):
        consensus = sum(other for other in views if other is not source) / 2.0
        pair_norm = torch.cat((source, consensus), dim=1).flatten(1).float().norm(dim=1)
        delta = (result - source).flatten(1).float().norm(dim=1)
        assert torch.all(delta <= 0.04 * pair_norm + 2e-4 * (1 + pair_norm))


def test_view_set_aggregator_is_identity_for_one_and_permutation_invariant_for_many():
    aggregator = ViewSetAggregator(8).eval()
    one = torch.randn(2, 8, 11, 9)
    views = (torch.randn_like(one), torch.randn_like(one), torch.randn_like(one))
    with torch.no_grad():
        assert torch.equal(aggregator((one,)), one)
        first = aggregator(views)
        second = aggregator((views[2], views[0], views[1]))
    assert torch.allclose(first, second, atol=1e-6, rtol=0)
    assert torch.allclose(first, torch.stack(views).mean(0), atol=1e-6, rtol=0)


def test_universal_backbone_direct_single_path_bypasses_cross_view_fusion():
    model = UniversalCCPRFBackbone(**_small_kwargs()).eval()
    image = torch.randn(2, 3, 64, 80)
    with torch.no_grad():
        pyramid, diagnostics = model.forward_with_diagnostics(image)
    expected = {
        "P2": (2, 16, 16, 20),
        "P3": (2, 32, 8, 10),
        "P4": (2, 48, 4, 5),
        "P5": (2, 64, 2, 3),
    }
    assert {key: tuple(value.shape) for key, value in pyramid.items()} == expected
    assert diagnostics["input_view_count"] == 1
    assert len(model.fusions) == 0
    for stage in ("P2", "P3"):
        assert diagnostics["stages"][stage]["fusion"] is None


def test_same_auto_backbone_runs_single_two_and_three_physical_view_cases():
    model = UniversalCCPRFBackbone(
        **_small_kwargs(
            input_mode="auto",
            view_channels={"image": 3, "depth": 1, "thermal": 1},
        )
    ).eval()
    image = torch.randn(2, 3, 64, 80)
    depth = torch.randn(2, 1, 64, 80)
    thermal = torch.randn(2, 1, 64, 80)
    with torch.no_grad():
        single, single_diag = model.forward_with_diagnostics(image)
        paired, pair_diag = model.forward_with_diagnostics({"image": image, "depth": depth})
        triple, triple_diag = model.forward_with_diagnostics(
            {"image": image, "depth": depth, "thermal": thermal}
        )
    assert single.keys() == paired.keys() == triple.keys()
    assert single_diag["input_view_count"] == 1
    assert pair_diag["input_view_count"] == 2
    assert triple_diag["input_view_count"] == 3
    assert all(torch.isfinite(value).all() for value in (*single.values(), *paired.values(), *triple.values()))


@pytest.mark.parametrize("task", ["classify", "segment", "detect", "anomaly"])
@pytest.mark.parametrize("case", ["single", "decomposed", "multi"])
def test_universal_model_all_tasks_and_input_cases(task: str, case: str):
    torch.manual_seed(10)
    if case == "single":
        kwargs = _small_kwargs(input_mode="single")
        inputs = torch.randn(2, 3, 64, 80)
    elif case == "decomposed":
        kwargs = _small_kwargs(input_mode="decomposed")
        inputs = torch.randn(2, 3, 64, 80)
    else:
        kwargs = _small_kwargs(
            input_mode="multi", view_channels={"image": 3, "auxiliary": 1}
        )
        inputs = {
            "image": torch.randn(2, 3, 64, 80),
            "auxiliary": torch.randn(2, 1, 64, 80),
        }
    model = UniversalCCPRFModel(
        task=task,
        num_classes=None if task == "anomaly" else 3,
        backbone_kwargs=kwargs,
        head_channels=16,
        anomaly_embedding_channels=8,
    )
    output = model(inputs)
    tensors = _tensors(output)
    assert tensors and all(torch.isfinite(value).all() for value in tensors)
    loss = sum(value.float().square().mean() for value in tensors)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_universal_model_checkpoint_roundtrip_and_extreme_inputs():
    torch.manual_seed(11)
    model = UniversalCCPRFModel(
        task="segment",
        num_classes=1,
        backbone_kwargs=_small_kwargs(input_mode="decomposed"),
        head_channels=8,
    ).eval()
    inputs = [
        torch.zeros(2, 3, 64, 64),
        torch.ones(2, 3, 64, 64),
        1e4 * torch.randn(2, 3, 64, 64),
    ]
    with torch.no_grad():
        references = [model(value) for value in inputs]
    assert all(torch.isfinite(value).all() for value in references)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    clone = copy.deepcopy(model)
    clone.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    clone.eval()
    with torch.no_grad():
        restored = clone(inputs[0])
    assert torch.equal(references[0], restored)


def _surface_fixture(batch: int = 6, size: int = 48):
    y = torch.linspace(0, 2 * torch.pi, size).view(1, 1, size, 1)
    x = torch.linspace(0, 2 * torch.pi, size).view(1, 1, 1, size)
    base = 0.45 + 0.10 * torch.sin(10 * x) + 0.08 * torch.sin(12 * y)
    images = base.repeat(batch, 3, 1, 1)
    masks = torch.zeros(batch, 1, size, size)
    for index in range(batch):
        top = 6 + (index * 5) % 25
        left = 7 + (index * 7) % 24
        images[index, :, top : top + 6, left : left + 8] += 0.35
        masks[index, :, top : top + 6, left : left + 8] = 1
    return images.clamp(0, 1), masks


@pytest.mark.parametrize("single_strategy", ["direct", "appearance_texture"])
def test_single_image_surface_segmentation_is_learnable(single_strategy: str):
    torch.manual_seed(12)
    model = UniversalCCPRFModel(
        task="segment",
        num_classes=1,
        backbone_kwargs=_small_kwargs(
            input_mode="auto",
            single_strategy=single_strategy,
            widths=(8, 16, 24, 32),
            group_width=4,
            local_windows={"P2": 3, "P3": 2},
        ),
        head_channels=8,
    )
    images, masks = _surface_fixture()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    with torch.no_grad():
        initial = F.binary_cross_entropy_with_logits(model(images), masks).item()
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(images), masks)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = F.binary_cross_entropy_with_logits(model(images), masks).item()
    assert final < initial, (single_strategy, initial, final)
