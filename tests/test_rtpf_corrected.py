import torch

from ultralytics.nn.modules.block import GOCI, RTPF


def test_reliability_uses_anchor_domain_not_full_resolution_energy():
    torch.manual_seed(7)
    rgb = torch.randn(2, 32, 64, 64)
    ir = 0.75 * rgb + 0.25 * torch.randn_like(rgb)
    module = GOCI(32, groups=8, anchors=8, momentum=1.0, eps=1e-3).train()
    module([rgb, ir])
    aligned_rgb, aligned_ir = module._align(rgb, ir)
    _, trigger, pooled_rgb, pooled_ir = module._reliability_terms(aligned_rgb, aligned_ir)
    full_rgb = aligned_rgb.float().square().mean((1, 2, 3), keepdim=True)
    full_ir = aligned_ir.float().square().mean((1, 2, 3), keepdim=True)
    old_trigger = torch.sigmoid(
        module.trigger_k
        * (
            torch.maximum(
                torch.log(full_rgb + 1e-4).abs(),
                torch.log(full_ir + 1e-4).abs(),
            )
            - module.trigger_tau
        )
    )
    assert pooled_rgb.mean() < 2.0 and pooled_ir.mean() < 2.0
    assert full_rgb.mean() > 10.0 and full_ir.mean() > 10.0
    assert trigger.mean() < 0.5
    assert old_trigger.mean() > 0.99


def test_rtpf_is_exact_concat_at_initialization():
    module = RTPF(16, groups=4, anchors=4).eval()
    rgb = torch.randn(2, 16, 20, 20)
    ir = torch.randn_like(rgb)
    actual = module([rgb, ir])
    expected = torch.cat((rgb, ir), dim=1)
    assert torch.equal(actual, expected)


def test_rtpf_correction_obeys_frobenius_trust_bound():
    radius = 0.35
    module = RTPF(
        16,
        groups=4,
        anchors=4,
        trust_radius=radius,
        fallback_tau=100.0,
    ).eval()
    module.step_parameter.data.fill_(4.0)
    rgb = torch.randn(3, 16, 20, 20)
    ir = torch.randn_like(rgb)
    raw = torch.cat((rgb, ir), dim=1)
    output = module([rgb, ir])
    relative = (output - raw).flatten(1).norm(dim=1) / raw.flatten(1).norm(dim=1)
    assert torch.all(relative <= radius + 1e-5)


def test_rtpf_step_receives_gradient_at_zero():
    module = RTPF(
        8,
        groups=2,
        anchors=4,
        fallback_tau=100.0,
    ).train()
    rgb = torch.randn(2, 8, 16, 16, requires_grad=True)
    ir = torch.randn_like(rgb, requires_grad=True)
    output = module([rgb, ir])
    # A non-symmetric downstream objective makes the zero-initialized residual
    # step observable while preserving exact concatenation in the forward pass.
    weights = torch.linspace(0.5, 1.5, output.numel(), device=output.device).reshape_as(output)
    (output * weights).mean().backward()
    assert module.step_parameter.grad is not None
    assert torch.isfinite(module.step_parameter.grad)
    assert module.step_parameter.grad.abs() > 0
