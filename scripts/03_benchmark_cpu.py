#!/usr/bin/env python3
"""CPU-only synthetic training benchmark for planned MAVEN MAG/SWEA CNNs.

No BigQuery access, no MAVEN measurements, and no files written to disk.
Use identical options on both computers to compare forward/backward throughput.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F


class MagAutoencoder(nn.Module):
    """Example 10-minute MAG CNN encoder, decoder, and crossing-score head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 24, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(24, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(24, 4, kernel_size=3, padding=1),
        )
        self.detector = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logits = self.detector(z.mean(dim=-1))
        reconstruction = self.decoder(
            F.interpolate(z, size=x.shape[-1], mode="linear", align_corners=False)
        )
        return reconstruction, logits


class SweaAutoencoder(nn.Module):
    """Example SWEA CNN encoder, decoder, and crossing-score head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        self.detector = nn.Linear(24, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logits = self.detector(z.mean(dim=(-2, -1)))
        reconstruction = self.decoder(
            F.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False)
        )
        return reconstruction, logits


def benchmark(model: nn.Module, x: torch.Tensor, *, warmup: int,
              steps: int, repeats: int) -> tuple[float, float, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    targets = torch.zeros(x.shape[0], 1, dtype=x.dtype)
    targets[x.shape[0] // 2:] = 1.0

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        reconstruction, logits = model(x)
        loss = F.mse_loss(reconstruction, x) + F.binary_cross_entropy_with_logits(
            logits, targets
        )
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        train_step()

    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(steps):
            train_step()
        durations.append((time.perf_counter() - start) / steps)

    median_step_s = statistics.median(durations)
    return median_step_s, x.shape[0] / median_step_s, max(durations) / min(durations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=None,
                        help="PyTorch CPU threads; default: current torch setting")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.batch_size < 1 or args.warmup < 0 or args.steps < 1 or args.repeats < 1:
        parser.error("batch-size, steps, repeats must be positive; warmup >= 0")
    if args.threads is not None:
        if args.threads < 1:
            parser.error("threads must be positive")
        torch.set_num_threads(args.threads)

    torch.manual_seed(2026)
    print(f"System: {platform.system()} {platform.release()}")
    print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    print(f"Device used: CPU; PyTorch threads: {torch.get_num_threads()}")
    print(f"Batch size: {args.batch_size}; warmup: {args.warmup}; "
          f"steps/repeat: {args.steps}; repeats: {args.repeats}")
    print("Inputs: MAG [batch,4,601]; SWEA [batch,1,64,38] (synthetic)")

    cases = [
        ("MAG Conv1D", MagAutoencoder(), torch.randn(args.batch_size, 4, 601)),
        ("SWEA Conv2D", SweaAutoencoder(), torch.randn(args.batch_size, 1, 64, 38)),
    ]
    for name, model, x in cases:
        step_s, throughput, variability = benchmark(
            model, x, warmup=args.warmup, steps=args.steps, repeats=args.repeats
        )
        print(f"{name}: {step_s * 1000:.1f} ms/step, "
              f"{throughput:.1f} windows/s, repeat max/min={variability:.2f}x")
    print("Interpretation: lower ms/step and higher windows/s are faster.")
    print("Synthetic relative benchmark only, NOT a real model-training result.")


if __name__ == "__main__":
    main()
