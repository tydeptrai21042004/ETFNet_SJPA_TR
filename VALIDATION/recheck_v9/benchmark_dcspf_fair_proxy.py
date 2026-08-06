from __future__ import annotations

"""Ten-seed component-wise controlled proxy for concat, corrected SJPA and DCSPF.

This is a synthetic mechanism test, not a substitute for a full VEDAI run.
All random streams are reset independently for data, fusion construction, head
construction, loader order and corruptions.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "VALIDATION"))

from benchmark_realistic_proxy import FastSJPAProxy, corrupt  # noqa: E402
from test_scid_fast import Concat, Data, Model, Proj, map50  # noqa: E402


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class DCSPFProxy(FastSJPAProxy):
    def __init__(self, channels: int, coherence_tau=0.35, typicality_tau=0.6, dominance_tau=0.8):
        super().__init__(channels)
        self.canonical_projection = self.p
        self.raw_projection = Proj(2 * channels, channels)
        self.coherence_tau = coherence_tau
        self.typicality_tau = typicality_tau
        self.dominance_tau = dominance_tau

    def coherence(self, a, b):
        batch, channels, height, width = a.shape
        tokens = height * width
        x = a.reshape(batch, self.g, self.d, tokens).transpose(-1, -2)
        y = b.reshape(batch, self.g, self.d, tokens).transpose(-1, -2)
        x = x - x.mean(-2, keepdim=True)
        y = y - y.mean(-2, keepdim=True)
        cxy = x.transpose(-1, -2) @ y / (tokens - 1)
        cxx = x.transpose(-1, -2) @ x / (tokens - 1)
        cyy = y.transpose(-1, -2) @ y / (tokens - 1)
        numerator = torch.linalg.matrix_norm(cxy, ord="fro").sum(-1)
        denominator = (
            torch.linalg.matrix_norm(cxx, ord="fro").sum(-1)
            * torch.linalg.matrix_norm(cyy, ord="fro").sum(-1)
            + 1e-8
        ).sqrt()
        return (numerator / denominator).clamp(0, 1).view(-1, 1, 1, 1)

    def forward(self, rgb, ir, aux=False):
        a = self.transform(rgb, self.mr, self.wr)
        b = self.transform(ir, self.mi, self.wi)
        batch, channels, height, width = a.shape
        tokens = height * width
        a = (
            (a.reshape(batch, self.g, self.d, tokens).transpose(-1, -2) @ self.q)
            .transpose(-1, -2)
            .reshape(batch, channels, height, width)
        )
        energy_r = a.square().mean((1, 2, 3), keepdim=True)
        energy_i = b.square().mean((1, 2, 3), keepdim=True)
        dev_r = torch.log(energy_r + 1e-4).abs()
        dev_i = torch.log(energy_i + 1e-4).abs()
        anomaly = torch.maximum(dev_r, dev_i)
        b, searched = self._search(a, b, anomaly.flatten() < self.tau)
        probability = torch.cat((-self.gamma * dev_r, -self.gamma * dev_i), 1).softmax(1)
        trigger = torch.sigmoid(self.k * (anomaly - self.tau))
        canonical = torch.cat(
            (
                (1 - trigger) * a + trigger * (2 * probability[:, :1]).sqrt() * a,
                (1 - trigger) * b + trigger * (2 * probability[:, 1:]).sqrt() * b,
            ),
            1,
        )
        coherence = self.coherence(a, b)
        typicality = torch.minimum(dev_r, dev_i)
        dominance = (probability[:, :1] - probability[:, 1:]).abs()
        gate = (
            ((coherence >= self.coherence_tau) & (typicality <= self.typicality_tau))
            | (dominance >= self.dominance_tau)
        ).to(a.dtype)
        output = gate * self.canonical_projection(canonical) + (1 - gate) * self.raw_projection(torch.cat((rgb, ir), 1))
        diagnostics = {"gate": gate, "searched": searched}
        return (output, diagnostics) if aux else output


def build(name, train_subset, run_seed):
    set_seed(10_000 + run_seed)
    if name == "concat":
        return Concat(8)
    if name == "sjpa":
        model = FastSJPAProxy(8)
        model.fit(train_subset)
        return model
    if name == "dcspf":
        model = DCSPFProxy(8)
        model.fit(train_subset)
        return model
    raise KeyError(name)


def run(name, run_seed, epochs=10):
    set_seed(20_000 + run_seed)
    dataset = Data(1400, seed=1800 + run_seed)
    train_subset = torch.utils.data.Subset(dataset, range(1050))
    test_subset = torch.utils.data.Subset(dataset, range(1050, 1400))
    fusion = build(name, train_subset, run_seed)
    set_seed(30_000 + run_seed)
    model = Model(fusion, 8, 5)
    optimizer = torch.optim.AdamW(model.parameters(), 2.5e-3, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(
        train_subset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(40_000 + run_seed),
    )
    schedule = ["clean", "clean", "clean", "rgb_noise", "ir_noise", "missing_rgb", "missing_ir", "ir_shift", "mixed"]
    step = 0
    for _ in range(epochs):
        model.train()
        for rgb, ir, label, box in loader:
            rgb, ir = corrupt(rgb, ir, schedule[(step + run_seed * 7) % len(schedule)], 50_000 + run_seed * 10_000 + step)
            step += 1
            optimizer.zero_grad()
            logits, pred_box, _ = model(rgb, ir, True)
            loss = F.cross_entropy(logits, label) + 3 * F.smooth_l1_loss(pred_box, box)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

    model.eval()
    metrics, gate_rates = {}, {}
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=100)
    conditions = ["clean", "basis_rotate", "rgb_noise", "ir_noise", "missing_rgb", "missing_ir", "ir_shift", "mixed"]
    with torch.no_grad():
        for condition in conditions:
            all_logits, all_pred, all_label, all_box, all_gate = [], [], [], [], []
            for rgb, ir, label, box in test_loader:
                rgb, ir = corrupt(rgb, ir, condition, 60_000 + run_seed)
                logits, pred_box, diagnostics = model(rgb, ir, True)
                all_logits.append(logits)
                all_pred.append(pred_box)
                all_label.append(label)
                all_box.append(box)
                all_gate.append(diagnostics.get("gate", torch.zeros(rgb.shape[0], 1, 1, 1)).flatten())
            metrics[condition] = map50(torch.cat(all_logits), torch.cat(all_pred), torch.cat(all_label), torch.cat(all_box), 5)
            gate_rates[condition] = float(torch.cat(all_gate).mean())
    return metrics, gate_rates, sum(parameter.numel() for parameter in model.parameters())


def summarize(raw, gates, params):
    robust_conditions = ["rgb_noise", "ir_noise", "missing_rgb", "missing_ir", "ir_shift", "mixed"]
    return {
        "params": params,
        "clean_mean": float(np.mean([item["clean"] for item in raw])),
        "clean_std": float(np.std([item["clean"] for item in raw], ddof=1)),
        "robust_mean": float(np.mean([np.mean([item[c] for c in robust_conditions]) for item in raw])),
        "robust_std": float(np.std([np.mean([item[c] for c in robust_conditions]) for item in raw], ddof=1)),
        "conditions": {c: float(np.mean([item[c] for item in raw])) for c in raw[0]},
        "gate": {c: float(np.mean([item[c] for item in gates])) for c in gates[0]},
    }


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    names = ["concat", "sjpa", "dcspf"]
    seeds = list(range(10))
    raw, gates, output = {}, {}, {}
    params = {}
    for name in names:
        raw[name], gates[name] = [], []
        for run_seed in seeds:
            metrics, gate, count = run(name, run_seed)
            raw[name].append(metrics)
            gates[name].append(gate)
            params[name] = count
        output[name] = summarize(raw[name], gates[name], params[name])

    robust_conditions = ["rgb_noise", "ir_noise", "missing_rgb", "missing_ir", "ir_shift", "mixed"]
    output["paired"] = {}
    utility = {
        "clean": lambda item: item["clean"],
        "robust": lambda item: np.mean([item[c] for c in robust_conditions]),
        "mixed": lambda item: item["mixed"],
    }
    for metric_name, function in utility.items():
        candidate = np.array([function(item) for item in raw["dcspf"]])
        for baseline in ["concat", "sjpa"]:
            reference = np.array([function(item) for item in raw[baseline]])
            _, p_value = ttest_rel(candidate, reference)
            output["paired"][f"{metric_name}_vs_{baseline}"] = {
                "mean_difference": float((candidate - reference).mean()),
                "two_sided_p": float(p_value),
                "wins": int((candidate > reference).sum()),
                "ties": int((candidate == reference).sum()),
            }
    output["raw"] = raw
    destination = Path(__file__).with_name("dcspf_fair_proxy_10seed.json")
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
