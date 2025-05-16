#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 01:35:34 2025

@author: DIYer22
"""
import cv2
import boxx

eps = 1e-20


def density_to_rgb(density, max_probability=None):
    if max_probability is None:
        max_probability = density.max()
    return cv2.applyColorMap(
        boxx.uint8(boxx.norma(density.clip(0, max_probability))), cv2.COLORMAP_VIRIDIS
    )[..., ::-1]


if __name__ == "__main__":
    pass
