"""Five-seed controlled fusion proxy with physically valid zero-padded shifts.

This benchmark is intentionally separate from public-dataset claims.  It tests
whether the fusion mechanism remains useful under a fixed synthetic detection
problem and controlled RGB/IR corruptions.  Spatial shifts never use circular
wraparound, so pixels leaving the field do not reappear on the opposite edge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel

from test_scid_fast import Concat, Data, Model, PBTR, TGF, iou, map50, ortho, seed_all
from test_global_pci import GlobalStats

HERE = Path(__file__).resolve().parent


def shift_no_wrap(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    if dy == 0 and dx == 0:
        return x
    height, width = x.shape[-2:]
    top, bottom = max(dy, 0), max(-dy, 0)
    left, right = max(dx, 0), max(-dx, 0)
    padded = F.pad(x, (left, right, top, bottom))
    return padded[..., bottom:bottom + height, right:right + width]


def corrupt(rgb: torch.Tensor, ir: torch.Tensor, condition: str, seed: int = 77):
    generator = torch.Generator().manual_seed(seed)
    if condition == "clean":
        return rgb, ir
    if condition == "basis_rotate":
        q_rgb = ortho(rgb.shape[1], 3123).to(rgb)
        q_ir = ortho(ir.shape[1], 9321).to(ir)
        return torch.einsum("bchw,cd->bdhw", rgb, q_rgb), torch.einsum("bchw,cd->bdhw", ir, q_ir)
    if condition == "rgb_noise":
        return rgb + 0.5 * torch.randn(rgb.shape, generator=generator), ir
    if condition == "ir_noise":
        return rgb, ir + 0.5 * torch.randn(ir.shape, generator=generator)
    if condition == "missing_rgb":
        return torch.zeros_like(rgb), ir
    if condition == "missing_ir":
        return rgb, torch.zeros_like(ir)
    if condition == "ir_shift":
        return rgb, shift_no_wrap(ir, 1, -1)
    if condition == "mixed":
        rgb_noisy = rgb + 0.35 * torch.randn(rgb.shape, generator=generator)
        ir_noisy = ir + 0.35 * torch.randn(ir.shape, generator=generator)
        return rgb_noisy, shift_no_wrap(ir_noisy, 1, 0)
    raise KeyError(condition)


class FastSJPAProxy(GlobalStats):
    """Small proxy mirroring the production module's sequential formulation."""

    def __init__(self, channels: int, tau=0.6, k=12.0, gamma=1.5, max_shift=1,
                 penalty=0.1, score_threshold=0.3):
        super().__init__(channels, kind="concat", reliable=False)
        self.tau = tau
        self.k = k
        self.gamma = gamma
        self.max_shift = max_shift
        self.penalty = penalty
        self.score_threshold = score_threshold

    def _search(self, aligned_rgb, whitened_ir, eligible):
        batch, channels, height, width = aligned_rgb.shape
        n = height * width
        rgb_groups = aligned_rgb.reshape(batch, self.g, self.d, n).transpose(-1, -2)
        rgb_centered = rgb_groups - rgb_groups.mean(-2, keepdim=True)
        ir_groups = whitened_ir.reshape(batch, self.g, self.d, n).transpose(-1, -2)
        ir_centered = ir_groups - ir_groups.mean(-2, keepdim=True)
        cross_zero = rgb_centered.transpose(-1, -2) @ ir_centered / (n - 1)
        zero_score = torch.linalg.svdvals(cross_zero).sum((-1, -2)) / (self.g * self.d)
        eligible = eligible & (zero_score < self.score_threshold)

        candidates, scores, shifts = [], [], []
        for dy in range(-self.max_shift, self.max_shift + 1):
            for dx in range(-self.max_shift, self.max_shift + 1):
                candidate = shift_no_wrap(whitened_ir, dy, dx)
                tokens = candidate.reshape(batch, self.g, self.d, n).transpose(-1, -2)
                tokens = tokens - tokens.mean(-2, keepdim=True)
                cross = rgb_centered.transpose(-1, -2) @ tokens / (n - 1)
                score = torch.linalg.svdvals(cross).sum((-1, -2)) - self.penalty * (dy * dy + dx * dx)
                candidates.append(candidate)
                scores.append(score)
                shifts.append((dy, dx))
        selected = torch.stack(scores, 1).argmax(1)
        zero_index = shifts.index((0, 0))
        selected = torch.where(eligible, selected, torch.full_like(selected, zero_index))
        output = torch.stack(candidates, 1)[torch.arange(batch), selected]
        return output, eligible

    def forward(self, rgb, ir, aux=False):
        aligned_rgb = self.transform(rgb, self.mr, self.wr)
        whitened_ir = self.transform(ir, self.mi, self.wi)
        batch, channels, height, width = aligned_rgb.shape
        n = height * width
        aligned_rgb = (
            aligned_rgb.reshape(batch, self.g, self.d, n).transpose(-1, -2) @ self.q
        ).transpose(-1, -2).reshape(batch, channels, height, width)
        energy_rgb = aligned_rgb.square().mean((1, 2, 3), keepdim=True)
        energy_ir = whitened_ir.square().mean((1, 2, 3), keepdim=True)
        dev_rgb = torch.log(energy_rgb + 1e-4).abs()
        dev_ir = torch.log(energy_ir + 1e-4).abs()
        anomaly = torch.maximum(dev_rgb, dev_ir)
        whitened_ir, searched = self._search(aligned_rgb, whitened_ir, anomaly.flatten() < self.tau)
        probability = torch.cat((-self.gamma * dev_rgb, -self.gamma * dev_ir), 1).softmax(1)
        trigger = torch.sigmoid(self.k * (anomaly - self.tau))
        rgb_scale = (2 * probability[:, :1]).sqrt()
        ir_scale = (2 * probability[:, 1:]).sqrt()
        fused_rgb = (1 - trigger) * aligned_rgb + trigger * rgb_scale * aligned_rgb
        fused_ir = (1 - trigger) * whitened_ir + trigger * ir_scale * whitened_ir
        output = self.p(torch.cat((fused_rgb, fused_ir), 1))
        return (output, {"searched": searched}) if aux else output


def build_method(name: str, training_subset):
    if name == "concat":
        return Concat(8)
    if name == "tgf":
        return TGF(8)
    if name == "pbtr":
        return PBTR(8)
    if name == "goci":
        module = GlobalStats(8, kind="concat", reliable=True)
        module.fit(training_subset)
        return module
    if name == "sjpa":
        module = FastSJPAProxy(8)
        module.fit(training_subset)
        return module
    raise KeyError(name)


def run_one(name: str, seed: int, epochs: int = 8):
    seed_all(seed)
    dataset = Data(1400, seed=1800 + seed)
    training = torch.utils.data.Subset(dataset, range(1050))
    testing = torch.utils.data.Subset(dataset, range(1050, 1400))
    model = Model(build_method(name, training), 8, 5)
    optimizer = torch.optim.AdamW(model.parameters(), 2.5e-3, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(
        training, 64, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    schedule = ["clean", "clean", "clean", "rgb_noise", "ir_noise", "missing_rgb",
                "missing_ir", "ir_shift", "mixed"]
    step = 0
    for _ in range(epochs):
        model.train()
        for rgb, ir, labels, boxes in loader:
            condition = schedule[(step + seed * 7) % len(schedule)]
            rgb, ir = corrupt(rgb, ir, condition, seed * 10000 + step)
            step += 1
            optimizer.zero_grad()
            logits, predicted, _ = model(rgb, ir, True)
            loss = F.cross_entropy(logits, labels) + 3 * F.smooth_l1_loss(predicted, boxes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

    model.eval()
    test_loader = torch.utils.data.DataLoader(testing, 100)
    metrics = {}
    with torch.no_grad():
        for condition in ["clean", "basis_rotate", "rgb_noise", "ir_noise", "missing_rgb",
                          "missing_ir", "ir_shift", "mixed"]:
            logits_all, boxes_all, labels_all, truth_all = [], [], [], []
            for rgb, ir, labels, boxes in test_loader:
                rgb, ir = corrupt(rgb, ir, condition, seed + 666)
                logits, predicted = model(rgb, ir)
                logits_all.append(logits)
                boxes_all.append(predicted)
                labels_all.append(labels)
                truth_all.append(boxes)
            logits = torch.cat(logits_all)
            predicted = torch.cat(boxes_all)
            labels = torch.cat(labels_all)
            truth = torch.cat(truth_all)
            metrics[condition] = {
                "accuracy": float((logits.argmax(1) == labels).float().mean()),
                "mean_iou": float(iou(predicted, truth).mean()),
                "proxy_map50": map50(logits, predicted, labels, truth, 5),
            }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU intra-op threads; one is fastest and most reproducible for this small proxy")
    parser.add_argument("--output", type=Path, default=HERE / "realistic_proxy_5seed.json")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    methods = ["concat", "tgf", "pbtr", "goci", "sjpa"]
    raw = {name: [run_one(name, seed, args.epochs) for seed in args.seeds] for name in methods}
    summary = {}
    robust_conditions = ["rgb_noise", "ir_noise", "missing_rgb", "missing_ir", "ir_shift", "mixed"]
    for name, runs in raw.items():
        summary[name] = {}
        for condition in runs[0]:
            values = [run[condition]["proxy_map50"] for run in runs]
            summary[name][condition] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
            }
        robust = [np.mean([run[c]["proxy_map50"] for c in robust_conditions]) for run in runs]
        summary[name]["robust_mean"] = {
            "mean": float(np.mean(robust)),
            "sample_std": float(np.std(robust, ddof=1)),
        }

    paired_tests = {}
    for condition in ["clean", "rgb_noise", "ir_noise", "missing_rgb", "missing_ir", "ir_shift", "mixed"]:
        proposal = np.asarray([run[condition]["proxy_map50"] for run in raw["sjpa"]])
        competitors = {
            name: np.asarray([run[condition]["proxy_map50"] for run in raw[name]])
            for name in methods if name != "sjpa"
        }
        strongest = max(competitors, key=lambda name: competitors[name].mean())
        test = ttest_rel(proposal, competitors[strongest], alternative="greater")
        paired_tests[condition] = {
            "strongest_baseline": strongest,
            "mean_difference": float((proposal - competitors[strongest]).mean()),
            "t_statistic": float(test.statistic),
            "one_sided_p": float(test.pvalue),
        }

    result = {
        "scope": "synthetic controlled proxy; not a public-dataset accuracy claim",
        "shift_model": "zero-padded translation; no circular wraparound",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "threads": args.threads,
        "summary": summary,
        "paired_tests": paired_tests,
        "raw": raw,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_tests": paired_tests}, indent=2))


if __name__ == "__main__":
    main()
