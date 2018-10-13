# [Deep Learning of Vortex Induced Vibrations](https://maziarraissi.github.io/DeepVIV/)

Vortex induced vibrations of bluff bodies occur when the vortex shedding frequency is close to the natural frequency of the structure. Of interest is the prediction of the lift and drag forces on the structure given some limited and scattered information on the velocity field. This is an inverse problem that is not straightforward to solve using standard computational fluid dynamics (CFD) methods, especially since no information is provided for the pressure. An even greater challenge is to infer the lift and drag forces given some dye or smoke visualizations of the flow field. Here we employ deep neural networks that are extended to encode the incompressible Navier-Stokes equations coupled with the structure's dynamic motion equation. In the first case, given scattered data in space-time on the velocity field and the structure's motion, we use four coupled deep neural networks to infer very accurately the structural parameters, the entire time-dependent pressure field (with no prior training data), and reconstruct the velocity vector field and the structure's dynamic motion. In the second case, given scattered data in space-time on a concentration field only, we use five coupled deep neural networks to infer very accurately the vector velocity field and all other quantities of interest as before. This new paradigm of inference in fluid mechanics for coupled multi-physics problems enables velocity and pressure quantification from flow snapshots in small subdomains and can be exploited for flow control applications and also for system identification.

For more information, please refer to the following: (https://maziarraissi.github.io/DeepVIV/)

  - Raissi, Maziar, Zhicheng Wang, Michael S. Triantafyllou, and George Em Karniadakis. "[Deep Learning of Vortex Induced Vibrations](https://arxiv.org/abs/1808.08952)." arXiv preprint arXiv:1808.08952 (2018).

## Dependencies

  1. Download https://www.dropbox.com/s/gxeyasawxlj7nyu/Data.tar.gz?dl=0
  2. Download https://www.dropbox.com/s/whegu3wugorumnj/Figures.tar.gz?dl=0
  3. Download https://www.dropbox.com/s/wru84hu4s1g6d3c/Results.tar.gz?dl=0

In the DeepVIV directory, execute

  1. tar -xzf Data.tar.gz
  2. tar -xzf Figures.tar.gz
  3. tar -xzf Results.tar.gz

## Citation

    @article{raissi2018deepVIV,
      title={Deep Learning of Vortex Induced Vibrations},
      author={Raissi, Maziar and Wang, Zhicheng and Triantafyllou, Michael S and Karniadakis, George Em},
      journal={arXiv preprint arXiv:1808.08952},
      year={2018}
    }
