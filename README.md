# distribution_playground
### **2D Probability Distribution "Playground" for Generative Models**

<br>
<div align="center">
  <a target="_blank" href="https://discrete-distribution-networks.github.io/2d-density-estimation-gif-with-10000-nodes-ddn.html">
    <img src="https://discrete-distribution-networks.github.io/img/frames_bin100_k2000_itern1800_batch40_framen96_2d-density-estimation-DDN.gif" style="height:">
  </a>
  <small><br>Density estimation optimization process <a target="_blank" href="https://discrete-distribution-networks.github.io/2d-density-estimation-gif-with-10000-nodes-ddn.html"><small>[details]</small></a><br>Left: Generated samples; Right: GT density</small>
</div>
<br>


## ▮ Features
- Various preset 2D distributions, from simple to complex
- Accurate and efficient sampling from probability density maps
- Calculate various divergence metrics between sampled data and probability density maps
- Good visualization
- Support for custom probability density maps from arbitrary images
- Complete sample code for generative model experiments, including creation of cool "optimization process GIFs"

*[Experiments](https://discrete-distribution-networks.github.io/) completed using `distribution_playground`:*  
<img src="https://discrete-distribution-networks.github.io/img/2d-density.png" style="width:348px">
  
## ▮ Tutorials
**Demo**
```bash
# Installation
pip install distribution_playground

# View all density_maps
python -m distribution_playground.density_maps

# Sample data from distributions and calculate divergence metrics with density maps
python -m distribution_playground.source_distribution
```

**Sample Code**  
See [toy_exp.py](https://github.com/DIYer22/sddn/blob/master/toy_exp.py), which includes:
- Training generative models to fit probability densities
- Recording divergence metrics between sampling results and GT density maps
- Saving visualization images of final sampling results
- Creating cool "optimization process GIFs"

# Reference
- [Probability Playground - Buffalo](https://www.acsu.buffalo.edu/~adamcunn/probability/normal.html)
- [DDPM - dataflowr](https://github.com/dataflowr/notebooks/tree/master/Module18)