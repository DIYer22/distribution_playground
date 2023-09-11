#!/usr/bin/env python3
import boxx

if __name__ == "__main__":
    from boxx import *

    eps = 0.05

    p1 = np.linspace(0 + eps, 1 - eps)
    q1 = np.linspace(0 + eps, 1 - eps)

    p2 = 1 - p1
    q2 = 1 - q1

    kl1 = p1[None] * np.log2(p1[None] / q1[:, None])
    kl2 = p2[None] * np.log2(p2[None] / q2[:, None])
    kl = kl1 + kl2

    show(kl, histEqualize(kl))
