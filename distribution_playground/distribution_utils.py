#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 01:35:34 2025

@author: DIYer22
"""
import cv2
import numpy as np

eps = 1e-20


def density_to_rgb(density, max_probability=None):
    if max_probability is None:
        max_probability = density.max()

    img = density.clip(0, max_probability) / max_probability
    img = (img * 255.999).astype(np.uint8)
    vis_rgb = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)[..., ::-1]
    return vis_rgb


if __name__ == "__main__":
    pass
