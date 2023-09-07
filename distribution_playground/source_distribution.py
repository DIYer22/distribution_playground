# !/bash/envs python
import boxx
import numpy as np
import matplotlib.pyplot as plt
import cv2
eps = 1e-20

def draw_circles_and_blur(s):
    # 创建一个 s x s 的空白图像
    img = np.zeros((s, s), dtype=np.uint8)

    # 圆心坐标
    circles_centers = [
        (s // 2, s // 2),
        (s // 4, s // 4),
        (s * 3 // 4, s // 4),
        (s // 4, s * 3 // 4),
        (s * 3 // 4, s * 3 // 4)
    ]

    # 画 5 个圆
    radius = s // 10
    for center in circles_centers:
        cv2.circle(img, center, radius, color=128, thickness=-1)
        cv2.circle(img, center, radius//2, color=0, thickness=-1)

    # 使用高斯模糊核半径为 s//50
    ksize = (s // 40) * 2 + 1
    blurred_img1 = cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    ksize = (s // 10) * 2 + 1
    blurred_img2 = cv2.GaussianBlur(img, (ksize, ksize), 0)

    ksize = int((s / 2.5) * 2 + 1)
    blurred_img3 = cv2.GaussianBlur(img, (ksize, ksize), 0)
    s = s//2
    img[:s,s:] = blurred_img1[:s,s:]
    img[s:,:s] = blurred_img2[s:,:s]
    img[s:,s:] = blurred_img3[s:,s:]
    img = img + eps/img.size
    return img/img.sum()

def sample_probability_density(arr, K=1000, domain=None):
    n = arr.ndim
    domain = np.array([(-1,1)]*n) if domain is None else domain
    assert domain.shape == (n, 2)

    # 计算累积分布函数
    cdf = np.cumsum(arr)
    
    # 使用均匀随机样本，在[0, 1]范围内生成K个点
    random_samples = np.random.rand(K)
    
    # 计算每个随机样本对应的索引值
    indices = np.searchsorted(cdf, random_samples)
    indices = np.unravel_index(indices, arr.shape)
    
    # 将索引值映射到定义域中的相应位置
    samples = np.empty((K, n))
    for i in range(n):
        left, right = domain[i]
        size = arr.shape[i]
        
        # 计算每个维度上的步长
        step = (right - left) / size
        
        # 将索引值映射到定义域
        idxs = indices[i] 
        if "uniform in one pixel":
            idxs = idxs + np.random.uniform(-.5+eps, .5-eps, idxs.shape)
        samples[:, i] = left + idxs* step + step / 2
    
    xys = samples[:, ::-1]*[[1,-1]]
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
        indices[i] = np.round((samples[:, i].clip(left+eps, right-eps) - left - step / 2.) / step).astype(int)
    return indices.T

class DistributionByDensityArray():
    def __init__(self, density, domain=None):
        self.density = density
        n = density.ndim
        self.domain = np.array([(-1,1)]*n) if domain is None else domain
    def sample(self, K=1000):
        return sample_probability_density(self.density, K, self.domain)
    
    def __str__(self):
        return f"{self.__class__.__name__}(density={self.density.shape}, domain={list(map(tuple,self.domain))})"
    
    def kl_divergence(self, xys):
        samples = (xys*[[1,-1]])[:, ::-1]
        sample_idxs = samples_to_indices(samples, self.domain, self.density)
        unique, count = np.unique(sample_idxs, axis=0, return_counts=True)
        original = self.density.flatten()
        estimated_count = np.zeros_like(self.density)
        estimated_count[unique[:, 0],unique[:, 1],] = count 
        estimated = estimated_count/estimated_count.sum()
        estimated = estimated/estimated.sum()
        from scipy.stats import entropy
        kl_loss = entropy(estimated.flatten(), original)
        boxx.mg()
        return dict(kl=kl_loss,estimated=estimated)
    
    __repr__ = __str__
if __name__ == "__main__" :
    from boxx import *
    
    # 设置边长s
    s = 100
    density = draw_circles_and_blur(s)
    xys = sample_probability_density(density, 10000)
    dist = DistributionByDensityArray(density, )
    xys = dist.sample(1000000)
    kl = dist.kl_divergence(xys)
    print(kl['kl'])
    show(kl, density, histEqualize)
    # plt.scatter(xys[:, 0], xys[:, 1], alpha=0.15)
    

