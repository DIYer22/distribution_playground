# !/bash/envs python
import boxx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

with boxx.inpkg():
    from .density_maps import build_blur_circles_density_map, eps


def sample_probability_density(arr, n=1000, domain=None):
    s = arr.ndim
    domain = np.array([(-1, 1)] * s) if domain is None else domain
    assert domain.shape == (s, 2)

    # 计算累积分布函数
    cdf = np.cumsum(arr)

    # 使用均匀随机样本，在[0, 1]范围内生成n个点
    random_samples = np.random.rand(n)

    # 计算每个随机样本对应的索引值
    indices = np.searchsorted(cdf, random_samples)
    indices = np.unravel_index(indices, arr.shape)

    # 将索引值映射到定义域中的相应位置
    samples = np.empty((n, s))
    for i in range(s):
        left, right = domain[i]
        size = arr.shape[i]

        # 计算每个维度上的步长
        step = (right - left) / size

        # 将索引值映射到定义域
        idxs = indices[i]
        if "uniform in one pixel":
            idxs = idxs + np.random.uniform(-0.5 + eps, 0.5 - eps, idxs.shape)
        samples[:, i] = left + idxs * step + step / 2

    xys = samples[:, ::-1] * [[1, -1]]
    return xys


def samples_to_indices(samples, domain, density):
    arr_shape = density.shape
    n = samples.shape[1]
    assert domain.shape == (n, 2)

    indices = np.empty((len(arr_shape), len(samples)), dtype=np.int32)

    for i in range(n):
        left, right = domain[i]
        size = arr_shape[i]
        # 计算每个维度上的步长
        step = (right - left) / size
        # 计算索引值
        indices[i] = (
            np.round((samples[:, i] - left - step / 2.0) / step)
            .astype(int)
            .clip(0, size - 1)
        )
    return indices.T


def xys_to_vus(xys, shape, domain):
    """
    和 samples_to_indices 验证一致， 并只保留一个
    """
    idxs_2d = xys[:, ::-1].copy()  # yxs
    for idx, (l, r) in enumerate(domain):
        idxs_2d[:, idx] = (idxs_2d[:, idx] - l) / (r - l) * shape[idx] - 0.5 + eps
        idxs_2d[:, idx] = idxs_2d[:, idx].clip(0 + eps, shape[idx] - 1 - eps)
    idxs_2d[:, -2] = shape[idx] - 1 - idxs_2d[:, -2]
    return idxs_2d


def jensen_shannon_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence between two probability distributions.
    :param p: First probability distribution
    :param q: Second probability distribution
    :return: JS divergence value
    """
    # p = np.asarray(p, dtype=np.float)
    # q = np.asarray(q, dtype=np.float)

    m = 0.5 * (p + q)
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))
    return jsd


class DistributionByDensityArray:
    def __init__(self, density, domain=None):
        self.density = density
        n = density.ndim
        self.domain = np.array([(-1, 1)] * n) if domain is None else domain

    def sample(self, n=1000):
        return sample_probability_density(self.density, n, self.domain)

    def __str__(self):
        return f"{self.__class__.__name__}(density={self.density.shape}, domain={list(map(tuple,self.domain))})"

    def kl_divergence(self, xys):
        samples = (xys * [[1, -1]])[:, ::-1]
        sample_idxs = samples_to_indices(samples, self.domain, self.density)
        unique, count = np.unique(sample_idxs, axis=0, return_counts=True)
        Q = self.density.flatten()
        estimated_count = np.zeros_like(self.density)
        estimated_count[
            unique[:, 0],
            unique[:, 1],
        ] = count
        estimated = estimated_count / estimated_count.sum()
        estimated = estimated / estimated.sum()
        P = estimated.flatten()
        # gt 分布是预构建的可以直接加 eps 而不影响精度, 所以 gt 分布作为分母
        kl_loss = entropy(P, Q)
        js_loss = jensen_shannon_divergence(P, Q)
        tv_loss = 0.5 * np.abs(P - Q).sum()
        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2))
        boxx.mg()
        return dict(
            kl=kl_loss,
            js=js_loss / np.log(2),
            js_raw=js_loss,
            tv=tv_loss,
            hellinger=hellinger,
            estimated=estimated,
        )

    divergence = kl_divergence

    def str_divergence(self, divergence):
        if not isinstance(divergence, dict):
            divergence = self.divergence(divergence)

        return "js=%.2f%%, " % (divergence["js"] * 100) + ", ".join(
            [f"{k}={divergence[k]}" for k in ["kl", "tv", "hellinger"]]
        )

    __repr__ = __str__


def get_test_dist(size=100):
    density = build_blur_circles_density_map(size)["density"]
    dist = DistributionByDensityArray(
        density,
    )
    return dist


if __name__ == "__main__":
    from boxx import *

    size = 100
    density = build_blur_circles_density_map(size)["density"]
    dist = DistributionByDensityArray(
        density,
    )
    xys = dist.sample(10000)
    divergence = dist.kl_divergence(xys)
    [print(k, "=", divergence[k]) for k in ["kl", "js", "tv", "hellinger"]]
    show(divergence, density, histEqualize)
    plt.scatter(xys[:, 0], xys[:, 1], alpha=0.15)
    plt.axis("equal")
