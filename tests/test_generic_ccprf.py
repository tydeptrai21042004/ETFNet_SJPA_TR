from __future__ import annotations

import copy
import inspect
import io

import pytest
import torch
import torch.nn.functional as F

from ultra_modeling.nn.modules.generic_ccprf import (
    AppearanceTextureAdapter,
    GenericCCPRF,
    LocalGlobalCCPRF,
    MultiViewCCPRFBackbone,
    PairedViewAdapter,
    TaskAgnosticCCPRFModel,
)


def _small_backbone_kwargs(input_mode: str = "appearance_texture"):
    kwargs = {
        "input_mode": input_mode,
        "widths": (16, 32, 64, 96),
        "fusion_stages": ("P2", "P3"),
        "group_width": 8,
        "stat_grid": 8,
        "trust_radius": {"P2": 0.03, "P3": 0.05},
        "local_windows": {"P2": 4, "P3": 2},
    }
    if input_mode == "paired":
        kwargs.update(view_a_channels=3, view_b_channels=1)
    return kwargs


def _recursive_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_recursive_tensors(item))
        return tensors
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(_recursive_tensors(item))
        return tensors
    return []


def test_appearance_texture_decomposition_is_exact_and_kernel_is_constrained():
    torch.manual_seed(1)
    adapter = AppearanceTextureAdapter(input_channels=3, out_channels=8, kernel_size=5)
    image = torch.randn(2, 3, 31, 29, requires_grad=True)
    low, high = adapter.decompose(image)
    kernel = adapter.normalized_kernel()

    assert torch.all(kernel >= 0)
    assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-7, rtol=0)
    assert torch.allclose(low + high, image, atol=1e-6, rtol=1e-6)

    view_a, view_b = adapter(image)
    loss = view_a.square().mean() + view_b.square().mean()
    loss.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert adapter.raw_kernel.grad is not None
    assert torch.isfinite(adapter.raw_kernel.grad).all()


def test_paired_view_adapter_accepts_different_physical_channel_counts():
    adapter = PairedViewAdapter(view_a_channels=3, view_b_channels=1, out_channels=8).eval()
    a = torch.randn(2, 3, 32, 40)
    b = torch.randn(2, 1, 32, 40)
    with torch.no_grad():
        fa, fb = adapter((a, b))
    assert fa.shape == fb.shape == (2, 8, 16, 20)


def test_generic_ccprf_is_exact_concat_at_initialization_and_batch_independent():
    torch.manual_seed(2)
    fusion = GenericCCPRF(channels=16, groups=2, stat_grid=8, trust_radius=0.05).eval()
    a = torch.randn(1, 16, 13, 11)
    b = torch.randn_like(a)
    with torch.no_grad():
        output = fusion((a, b))
    assert torch.equal(output, torch.cat((a, b), dim=1))

    with torch.no_grad():
        fusion.innovation_weight.normal_(0, 0.03)
    companion1 = (torch.randn_like(a), torch.randn_like(b))
    companion2 = (torch.randn_like(a), torch.randn_like(b))
    batch1 = (torch.cat((a, companion1[0])), torch.cat((b, companion1[1])))
    batch2 = (torch.cat((a, companion2[0])), torch.cat((b, companion2[1])))
    with torch.no_grad():
        y1 = fusion(batch1)[0]
        y2 = fusion(batch2)[0]
    assert torch.allclose(y1, y2, atol=3e-5, rtol=3e-5)


def test_generic_ccprf_trust_bound_and_degenerate_inputs():
    torch.manual_seed(3)
    fusion = GenericCCPRF(channels=16, groups=2, stat_grid=8, trust_radius=0.05).eval()
    with torch.no_grad():
        fusion.innovation_weight.normal_(0, 0.5)

    cases = [
        torch.zeros(2, 16, 12, 10),
        torch.ones(2, 16, 12, 10),
        torch.randn(2, 16, 12, 10),
        1e5 * torch.randn(2, 16, 12, 10),
    ]
    for a in cases:
        b = a.clone() if torch.count_nonzero(a) == 0 else torch.flip(a, dims=(-1,))
        with torch.no_grad():
            output = fusion((a, b))
        raw = torch.cat((a, b), dim=1)
        delta = (output - raw).flatten(1).float().norm(dim=1)
        radius = fusion.trust_radius * raw.flatten(1).float().norm(dim=1)
        assert torch.isfinite(output).all()
        assert torch.all(delta <= radius + 2e-4 * (1 + radius))


def test_local_global_fusion_handles_nondivisible_maps_and_preserves_bound():
    torch.manual_seed(4)
    fusion = LocalGlobalCCPRF(
        channels=16,
        groups=2,
        stat_grid=8,
        trust_radius=0.04,
        window_size=7,
    ).eval()
    a = torch.randn(2, 16, 19, 23)
    b = torch.randn_like(a)
    with torch.no_grad():
        initial = fusion((a, b))
    raw = torch.cat((a, b), dim=1)
    assert torch.equal(initial, raw)

    with torch.no_grad():
        fusion.fusion.innovation_weight.normal_(0, 0.4)
        output, diagnostics = fusion.forward_with_diagnostics((a, b))
    delta = (output - raw).flatten(1).float().norm(dim=1)
    radius = fusion.trust_radius * raw.flatten(1).float().norm(dim=1)
    assert output.shape == raw.shape
    assert torch.isfinite(output).all()
    assert torch.all(delta <= radius + 1e-4 * (1 + radius))
    assert 0 < float(diagnostics["local_weight"]) < 1


def test_backbone_returns_task_neutral_pyramid_for_single_and_paired_inputs():
    single = MultiViewCCPRFBackbone(**_small_backbone_kwargs()).eval()
    paired = MultiViewCCPRFBackbone(**_small_backbone_kwargs("paired")).eval()
    image = torch.randn(2, 3, 64, 80)
    thermal = torch.randn(2, 1, 64, 80)

    with torch.no_grad():
        p_single = single(image)
        p_paired = paired((image, thermal))

    expected = {
        "P2": (2, 16, 16, 20),
        "P3": (2, 32, 8, 10),
        "P4": (2, 64, 4, 5),
        "P5": (2, 96, 2, 3),
    }
    assert {name: tuple(value.shape) for name, value in p_single.items()} == expected
    assert {name: tuple(value.shape) for name, value in p_paired.items()} == expected


@pytest.mark.parametrize("task", ["classify", "segment", "detect", "anomaly"])
def test_all_generic_tasks_forward_backward_and_checkpoint_roundtrip(task: str):
    torch.manual_seed(5)
    model = TaskAgnosticCCPRFModel(
        task=task,
        num_classes=None if task == "anomaly" else 3,
        backbone_kwargs=_small_backbone_kwargs(),
        head_channels=16,
        anomaly_embedding_channels=8,
    )
    image = torch.randn(2, 3, 64, 80)
    output = model(image)
    tensors = _recursive_tensors(output)
    assert tensors and all(torch.isfinite(tensor).all() for tensor in tensors)

    if task == "classify":
        assert output.shape == (2, 3)
    elif task == "segment":
        assert output.shape == (2, 3, 64, 80)
    elif task == "detect":
        assert set(output) == {"P2", "P3", "P4", "P5"}
        assert output["P2"]["class_logits"].shape == (2, 3, 16, 20)
        assert output["P2"]["box_distances"].shape == (2, 4, 16, 20)
    else:
        assert output.shape == (2, 32, 16, 20)

    loss = sum(tensor.float().square().mean() for tensor in tensors)
    loss.backward()
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_gradients
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)

    model.eval()
    with torch.no_grad():
        reference = model(image)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    clone = copy.deepcopy(model)
    clone.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    clone.eval()
    with torch.no_grad():
        restored = clone(image)
    for left, right in zip(_recursive_tensors(reference), _recursive_tensors(restored)):
        assert torch.equal(left, right)


def _fabric_batch(batch: int = 6, size: int = 48):
    y = torch.linspace(0, 2 * torch.pi, size).view(1, 1, size, 1)
    x = torch.linspace(0, 2 * torch.pi, size).view(1, 1, 1, size)
    base = 0.45 + 0.12 * torch.sin(10 * x) + 0.10 * torch.sin(12 * y)
    images = base.repeat(batch, 3, 1, 1)
    masks = torch.zeros(batch, 1, size, size)
    for index in range(batch):
        top = 8 + (index * 5) % 22
        left = 7 + (index * 7) % 24
        height = 5 + index % 3
        width = 7 + (index + 1) % 4
        images[index, :, top : top + height, left : left + width] += 0.35
        masks[index, :, top : top + height, left : left + width] = 1.0
    return images.clamp(0, 1), masks


def test_synthetic_fabric_segmentation_loss_decreases():
    """Small end-to-end learnability check, not a benchmark claim."""
    torch.manual_seed(6)
    model = TaskAgnosticCCPRFModel(
        task="segment",
        num_classes=1,
        backbone_kwargs={
            "input_mode": "appearance_texture",
            "widths": (8, 16, 24, 32),
            "fusion_stages": ("P2", "P3"),
            "group_width": 4,
            "stat_grid": 6,
            "trust_radius": {"P2": 0.03, "P3": 0.05},
            "local_windows": {"P2": 3, "P3": 2},
        },
        head_channels=8,
    )
    images, masks = _fabric_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()

    with torch.no_grad():
        initial = F.binary_cross_entropy_with_logits(model(images), masks).item()
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, masks)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = F.binary_cross_entropy_with_logits(model(images), masks).item()
    assert final < 0.90 * initial, (initial, final)


def test_generic_module_contains_no_dataset_or_oriented_head_assumptions():
    import ultra_modeling.nn.modules.generic_ccprf as module

    source = inspect.getsource(module).lower()
    for forbidden in ("vedai", "uav", "oriented bounding", "obbhead"):
        assert forbidden not in source
