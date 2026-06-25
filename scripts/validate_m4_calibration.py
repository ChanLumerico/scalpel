#!/usr/bin/env python3
"""M4 pre-validation: temperature scaling reduces ECE, abstention helps accuracy.

Milestone M4's acceptance (handout §6, §2.7): after `TemperatureScaler.fit` the
Expected Calibration Error drops, and the abstention threshold trades coverage
for higher selective accuracy. This is data-agnostic, so it's fully testable now
on synthetic *miscalibrated* logits using the real `scalpel.heads` code.

We simulate an overconfident classifier: a true-class margin sets the accuracy,
and an extra logit gain inflates confidence (positive ECE). Temperature scaling
should recover roughly that gain and flatten the calibration gap.

Run (any env with torch)::

    python scripts/validate_m4_calibration.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalpel.config import HeadCfg  # noqa: E402
from scalpel.heads import (  # noqa: E402
    ProductOfExperts,
    TemperatureScaler,
    expected_calibration_error,
)

N, C = 6000, 12
MARGIN = 1.1        # controls base accuracy (~60-70%)
GAIN = 2.6          # logit inflation -> overconfidence (the miscalibration to fix)


def make_overconfident_logits():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, C, (N,), generator=g)
    logits = torch.randn(N, C, generator=g)
    logits[torch.arange(N), y] += MARGIN          # signal -> sets accuracy
    logits = logits * GAIN                          # inflate -> overconfident
    return logits, y


def selective_curve(probs, labels):
    """Sweep a top-1 confidence threshold: (coverage, selective accuracy)."""
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()
    out = []
    for t in np.linspace(0.0, 0.99, 20):
        keep = conf >= t
        cov = float(keep.float().mean())
        acc = float(correct[keep].mean()) if keep.any() else float("nan")
        out.append((t, cov, acc))
    return out


def main() -> int:
    logits, y = make_overconfident_logits()
    val_logits, val_y = logits[:2000], y[:2000]
    test_logits, test_y = logits[2000:], y[2000:]

    acc = float(test_logits.argmax(1).eq(test_y).float().mean())
    ece_before = expected_calibration_error(test_logits.softmax(1), test_y)

    ts = TemperatureScaler(HeadCfg(n_classes=C))
    info = ts.fit(val_logits, val_y)               # fit T on the val split (NLL/LBFGS)
    cal_test = ts(test_logits)
    ece_after = expected_calibration_error(cal_test.softmax(1), test_y)

    print("=== M4 temperature calibration ===")
    print(f"  classifier accuracy : {acc:.3f}  (C={C})")
    print(f"  learned temperature : T = {ts.T:.2f}   (injected overconfidence gain = {GAIN})")
    print(f"  ECE before          : {ece_before:.4f}")
    print(f"  ECE after           : {ece_after:.4f}")

    # abstention: selective accuracy should exceed full-coverage accuracy
    curve = selective_curve(cal_test.softmax(1), test_y)
    full_acc = curve[0][2]
    hi = [a for (_t, cov, a) in curve if cov >= 0.2 and not np.isnan(a)]
    best_sel = max(hi) if hi else full_acc
    print("\n=== abstention (selective accuracy vs coverage) ===")
    for t, cov, a in curve[::4]:
        print(f"  thr {t:.2f}: coverage {cov:5.1%}  selective-acc {a:.3f}")
    print(f"  full-coverage acc {full_acc:.3f}  ->  best selective-acc (cov>=20%) {best_sel:.3f}")

    # decide() abstains on a deliberately flat distribution, answers a peaked one
    poe = ProductOfExperts(HeadCfg(n_classes=C, min_top1_prob=0.40, max_entropy_bits=2.0))
    peaked = torch.log_softmax(torch.tensor([6.0] + [0.0] * (C - 1)), dim=-1)
    flat = torch.log_softmax(torch.zeros(C), dim=-1)
    d_peak, d_flat = poe.decide(peaked), poe.decide(flat)

    ece_ok = ece_after < ece_before
    sel_ok = best_sel > full_acc + 0.05
    abst_ok = (d_flat["abstain"] is True) and (d_peak["abstain"] is False)
    print("\n=== checks ===")
    print(f"  ECE dropped                 : {'YES' if ece_ok else 'NO'} "
          f"({ece_before:.3f} -> {ece_after:.3f})")
    print(f"  abstention raises selective  : {'YES' if sel_ok else 'NO'} "
          f"({full_acc:.3f} -> {best_sel:.3f})")
    print(f"  decide(): flat->abstain, peak->answer : {'YES' if abst_ok else 'NO'}")
    ok = ece_ok and sel_ok and abst_ok
    print(f"\nRESULT: {'M4 calibration + abstention VALIDATED' if ok else 'NOT validated'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
