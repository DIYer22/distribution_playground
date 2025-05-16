import cv2
import boxx
import numpy as np

with boxx.inpkg():
    from .distribution_utils import eps, density_to_rgb


def build_uniform_density_map(shape=100):
    if isinstance(shape, int):
        shape = (shape, shape)
    img = np.ones(shape, dtype=np.uint8)
    density = img / img.sum()
    return dict(density=density, name="uniform")


def build_gaussian_density_map(shape=100):
    if isinstance(shape, int):
        shape = (shape, shape)
    h, w = shape
    img = np.zeros(shape, dtype=np.uint8)
    ys, xs = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    img = np.exp(-(xs**2 + ys**2) / 0.3)
    density = img / img.sum()
    return dict(density=density, name="gaussian")


def build_blurs_density_map(shape=100):
    if isinstance(shape, int):
        shape = (shape, shape)
    h, w = shape
    s = max(h, w)
    img = np.zeros(shape, dtype=np.uint8)

    # 圆心坐标
    circles_centers = [
        (h // 2, w // 2),
        (h // 4, w // 4),
        (h * 3 // 4, w // 4),
        (h // 4, w * 3 // 4),
        (h * 3 // 4, w * 3 // 4),
    ]

    # 画 5 个圆
    radius = s // 10
    for center in circles_centers:
        cv2.circle(img, center, radius, color=128, thickness=-1)
        cv2.circle(img, center, radius // 2, color=0, thickness=-1)

    # 使用高斯模糊核半径为 s//50
    ksize = (s // 40) * 2 + 1
    blurred_img1 = cv2.GaussianBlur(img, (ksize, ksize), 0)

    ksize = (s // 10) * 2 + 1
    blurred_img2 = cv2.GaussianBlur(img, (ksize, ksize), 0)

    ksize = int((s / 2.5) * 2 + 1)
    blurred_img3 = cv2.GaussianBlur(img, (ksize, ksize), 0)
    s = s // 2
    img[:s, s:] = blurred_img1[:s, s:]
    img[s:, :s] = blurred_img2[s:, :s]
    img[s:, s:] = blurred_img3[s:, s:]
    img = img + eps / img.size
    density = img / img.sum()
    return dict(density=density, name="blurs")


def convert_density_map_img(density_map_img_path, shape=None):
    density = cv2.imread(density_map_img_path)
    if shape is None:
        shape = density.shape[:2]
    elif isinstance(shape, int):
        shape = (shape, shape)
    if shape != density.shape[:2]:
        density = cv2.resize(
            density,
            shape[::-1],
            interpolation=cv2.INTER_AREA,
        )
    if density.ndim == 3:
        density = density[..., :3].mean(axis=-1)
    eps = 1e-20
    density = density / density.sum()
    density += eps / density.size
    density = density / density.sum()
    return dict(density=density, name=boxx.filename(density_map_img_path))


density_map_builders = {
    "uniform": build_uniform_density_map,
    "gaussian": build_gaussian_density_map,
    "blurs": build_blurs_density_map,
}

density_map_img_paths = sorted(boxx.glob(boxx.relfile("density_map_imgs/*.png")))

for _density_map_img_path in density_map_img_paths:
    density_map_builders[
        boxx.filename(_density_map_img_path)
    ] = lambda shape=None, path=_density_map_img_path: convert_density_map_img(
        path, shape
    )

if __name__ == "__main__":
    density_maps = [f(100) for f in density_map_builders.values()]
    boxx.tree(density_maps)
    boxx.show(density_maps, density_to_rgb)
