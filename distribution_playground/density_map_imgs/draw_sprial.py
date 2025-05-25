#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:28:54 2023

@author: yanglei

把这个 xy1, xy2 转换为 转换为线条, 用 cv2 在 512x512 的位图来上画出来. 
坐标转换关系 -10=>0, 0=>256, 10=>512

要求: 每个线条的粗细为20像素

"""
from boxx import *

import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the spiral
theta = np.linspace(0, 3 * np.pi, 100)

r = theta

# Convert polar coordinates to Cartesian
x1 = r * np.cos(theta)
y1 = r * np.sin(theta)


x2 = r * np.cos(theta + np.pi)  # offset 'np.pi' creates the second spiral
y2 = r * np.sin(theta + np.pi)

x1 += 0.5
x2 -= 0.5

# Combine the x and y values of the two spirals
X = np.concatenate((x1, x2))
Y = np.concatenate((y1, y2))

# Create the 2D histogram
plt.hist2d(X, Y, bins=[100, 100], cmap=None)
# plt.colorbar(label='Density')

plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

N = 100

theta = np.linspace(1, 3.5 * np.pi, N)

r = theta

# Convert polar coordinates to Cartesian
x1 = r * np.cos(theta)
y1 = r * np.sin(theta)

x2 = r * np.cos(theta + np.pi)  # offset 'np.pi' creates the second spiral
y2 = r * np.sin(theta + np.pi)

# x1 += .5
# x2 -= .5


# Function to scale points according to provided relation.
def scale_point(p):
    return int((p + 11) / 22 * 512)


# Create a blank 512x512 image
img = np.zeros(
    (
        512,
        512,
    ),
    dtype=np.uint8,
)

# Draw lines
for X, Y in [[x1, y1], [x2, y2]]:
    for i in range(len(X) - 1):
        pt1 = (scale_point(X[i]), scale_point(Y[i]))
        pt2 = (scale_point(X[i + 1]), scale_point(Y[i + 1]))
        # img = cv2.line(img, pt1, pt2, 255, 20)
        color = (
            255 if i < N / 4 else 255 * (N - i) / (N / 2)
        )  # Color decreases when i > N/2
        thickness = (
            25 if i < N / 2 else 10 + 15 * (N - i) / (N / 2)
        )  # Thickness decreases to 5 when i > N/2
        img = cv2.line(img, pt1, pt2, int(color), int(thickness))

blured = cv2.blur(img, (15,) * 2)
show(img, blured)
colored = show / cv2.applyColorMap(blured, cv2.COLORMAP_VIRIDIS)[..., ::-1]
# imsave("distribution_playground/density_map_imgs/sprial.png", blured)
