# distribution_playground
### **面向生成模型的二维概率分布“游乐场”**


<br>
<div align="center">
  <a target="_blank" href="https://discrete-distribution-networks.github.io/2d-density-estimation-gif-with-10000-nodes-ddn.html">
    <img src="https://discrete-distribution-networks.github.io/img/frames_bin100_k2000_itern1800_batch40_framen96_2d-density-estimation-DDN.gif" style="height:">
  </a>
  <small><br>概率密度估计优化过程 <a target="_blank" href="https://discrete-distribution-networks.github.io/2d-density-estimation-gif-with-10000-nodes-ddn.html"><small>[详情]</small></a><br>左：生成样本；右：概率密度GT</small>
</div>
<br>

## ▮ 特点
- 各种各样的预设二维分布，从简单分布到复杂分布
- 准确高效地从概率密度图中采样数据
- 计算采样数据和概率密度图的各种散度指标
- 良好的可视化
- 支持任意图像来自定义概率密度图
- 提供生成模型实验完整的样例代码，包含制作炫酷的“优化过程 GIF”

*使用 `distribution_playground` 完成的[实验](https://discrete-distribution-networks.github.io/)：*  
<img src="https://discrete-distribution-networks.github.io/img/2d-density.png" style="width:348px">
  
## ▮ 教程
**Demo**
```bash
# 安装
pip install distribution_playground

# 查看所有 density_maps
python -m distribution_playground.density_maps

# 从分布中采样数据并和 density map 计算散度指标
python -m distribution_playground.source_distribution
```

**样例代码**  
见 [toy_exp.py](https://github.com/DIYer22/sddn/blob/master/toy_exp.py)，包含：
- 训练生成模型拟合概率密度
- 记录采样结果和 GT density map 的散度指标
- 保存最终采样结果的可视化图像
- 制作炫酷的“优化过程 GIF”

# Reference
- [Probability Playground - Buffalo](https://www.acsu.buffalo.edu/~adamcunn/probability/normal.html)
- [DDPM - dataflowr](https://github.com/dataflowr/notebooks/tree/master/Module18)
