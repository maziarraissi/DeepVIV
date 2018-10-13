---
layout: default
---
### Authors
[Maziar Raissi](http://www.dam.brown.edu/people/mraissi/), [Zhicheng Wang](http://meche.mit.edu/people/staff/zhicheng@mit.edu), [Michael Triantafyllou](http://meche.mit.edu/people/faculty/MISTETRI@MIT.EDU), and [George Karniadakis](https://www.brown.edu/research/projects/crunch/george-karniadakis)

### Abstract

[Vortex induced vibrations](https://en.wikipedia.org/wiki/Vortex-induced_vibration) of bluff bodies occur when the [vortex shedding](https://en.wikipedia.org/wiki/Vortex_shedding) frequency is close to the natural frequency of the structure. Of interest is the prediction of the [lift and drag forces](https://en.wikipedia.org/wiki/Lift_(force)) on the structure given some limited and scattered information on the velocity field. This is an [inverse problem](https://www.springer.com/us/book/9780387253640) that is not straightforward to solve using standard [computational fluid dynamics (CFD)](https://en.wikipedia.org/wiki/Computational_fluid_dynamics) methods, especially since no information is provided for the pressure. An even greater challenge is to infer the lift and drag forces given some [dye or smoke visualizations](https://en.wikipedia.org/wiki/Flow_visualization) of the flow field. Here we employ [deep neural networks](https://en.wikipedia.org/wiki/Deep_learning) that are extended to encode the incompressible [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier–Stokes_existence_and_smoothness) coupled with the structure's dynamic motion equation. In the first case, given scattered data in space-time on the velocity field and the structure's motion, we use four coupled deep neural networks to infer very accurately the structural parameters, the entire time-dependent pressure field (with no prior training data), and reconstruct the velocity vector field and the structure's dynamic motion. In the second case, given scattered data in space-time on a concentration field only, we use five coupled deep neural networks to infer very accurately the vector velocity field and all other quantities of interest as before. This new paradigm of inference in fluid mechanics for coupled multi-physics problems enables velocity and pressure quantification from flow snapshots in small subdomains and can be exploited for [flow control](https://en.wikipedia.org/wiki/Flow_control_(fluid)) applications and also for system identification.

* * * * * *
#### Problem setup and solution methodology

We begin by considering the prototype [Vortex induced vibrations](https://en.wikipedia.org/wiki/Vortex-induced_vibration) VIV problem of flow past a circular cylinder. The fluid motion is governed by the incompressible Navier-Stokes equations while the dynamics of the structure is described in a general form involving displacement, velocity, and acceleration terms. In particular, let us consider the two-dimensional version of flow over a flexible cable, i.e., an elastically mounted cylinder. The two-dimensional problem contains most of the salient features of the three-dimensional case and consequently it is relatively straightforward to generalize the proposed framework to the flexible cylinder/cable problem. In two dimensions, the physical model of the cable reduces to a mass-spring-damper system. There are two directions of motion for the cylinder: the streamwise (i.e., $$x$$) direction and the crossflow (i.e., $$y$$) direction. In [this work](https://arxiv.org/abs/1808.08952), we assume that the cylinder can only move in the crossflow (i.e., $$y$$) direction; we concentrate on crossflow vibrations since this is the primary VIV direction. However, it is a simple extension to study cases where the cylinder is free to move in both streamwise and crossflow directions.

**A Pedagogical Example**

The cylinder displacement is defined by the variable $$\eta$$ corresponding to the crossflow motion. The equation of motion for the cylinder is then given by

$$
\rho \eta_{tt} + b \eta_t + k \eta = f_L,
$$

where $$\rho$$, $$b$$, and $$k$$ are the mass, damping, and stiffness parameters, respectively. The fluid lift force on the structure is denoted by $$f_L$$. The mass $$\rho$$ of the cylinder is usually a known quantity; however, the damping $$b$$ and the stiffness $$k$$ parameters are often unknown in practice. In the current work, we put forth a deep learning approach for estimating these parameters from measurements. We start by assuming that we have access to the input-output data $$\{t^n, \eta^n\}_{n=1}^N$$ and $$\{t^n, f_L^n\}_{n=1}^N$$ on the displacement $$\eta(t)$$ and the lift force $$f_L(t)$$ functions, respectively. Having access to direct measurements of the forces exerted by the fluid on the structure is obviously a strong assumption. However, we start with this simpler but pedagogical case and we will relax this assumption later in this section.


Inspired by recent developments in [physics informed deep learning](https://maziarraissi.github.io/PINNs/) and [deep hidden physics models](https://maziarraissi.github.io/DeepHPMs/), we propose to approximate the unknown function $$\eta$$ by a deep neural network. This choice is motivated by modern techniques for solving forward and inverse problems involving partial differential equations, where the unknown solution is approximated either by a neural network or a [Gaussian process](https://maziarraissi.github.io/HPM/). Moreover, placing a prior on the solution is fully justified by similar approaches pursued in the past centuries by classical methods of solving partial differential equations such as finite elements, finite differences, or spectral methods, where one would expand the unknown solution in terms of an appropriate set of basis functions. Approximating the unknown function $$\eta$$ by a deep neural network and using the above equation allow us to obtain the following [physics-informed neural network](https://maziarraissi.github.io/PINNs/) (see the following figure)


$$
\begin{array}{l}
f_L := \rho \eta_{tt} + b \eta_t + k \eta.
\end{array}
$$

![](http://www.dam.brown.edu/people/mraissi/assets/img/DeepVIV_1.png)
> _Pedagogical physics-informed neural network:_ A plain vanilla densely connected (physics uninformed) neural network, with 10 hidden layers and 32 neurons per hidden layer per output variable (i.e., 1 x 32 = 32 neurons per hidden layer), takes the input variable t and outputs the displacement. As for the activation functions, we use sin(x). For illustration purposes only, the network depicted in this figure comprises of 2 hidden layers and 4 neurons per hidden layers. We employ automatic differentiation to obtain the required derivatives to compute the residual (physics informed) networks. The total loss function is composed of the regression loss of the displacement on the training data, and the loss imposed by the differential equation. Here, the differential operator is computed using automatic differentiation and can be thought of as an "activation operator". Moreover, the gradients of the loss function are back-propagated through the entire network to train the parameters of the neural network as well as the damping and the stiffness parameters using the Adam optimizer.

It is worth noting that the damping $$b$$ and the stiffness $$k$$ parameters turn into parameters of the resulting physics informed neural network $$f_L$$. We obtain the required derivatives to compute the residual network $$f_L$$ by applying the chain rule for differentiating compositions of functions using [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation). Automatic differentiation is different from, and in several respects superior to, numerical or symbolic differentiation -- two commonly encountered techniques of computing derivatives. In its most basic description, automatic differentiation relies on the fact that all numerical computations are ultimately compositions of a finite set of elementary operations for which derivatives are known. Combining the derivatives of the constituent operations through the chain rule gives the derivative of the overall composition. This allows accurate evaluation of derivatives at machine precision with ideal asymptotic efficiency and only a small constant factor of overhead. In particular, to compute the required derivatives we rely on [Tensorflow](https://www.tensorflow.org), which is a popular and relatively well documented open source software library for automatic differentiation and deep learning computations.

The shared parameters of the neural networks $$\eta$$ and $$f_L$$, in addition to the damping $$b$$ and the stiffness $$k$$ parameters, can be learned by minimizing the following sum of squared errors loss function

$$
\sum_{n=1}^N |\eta(t^n) - \eta^n|^2 + \sum_{n=1}^N |f_L(t^n) - f_L^n|^2.
$$

The first summation in this loss function corresponds to the training data on the displacement $$\eta(t)$$ while the second summation enforces the dynamics imposed by equation of motion for the cylinder.

**Inferring Lift and Drag Forces from Scattered Velocity Measurements**

So far, we have been operating under the assumption that we have access to direct measurements of the lift force $$f_L$$. In the following, we are going to relax this assumption by recalling that the fluid motion is governed by the incompressible [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier–Stokes_existence_and_smoothness) given explicitly by

$$
\begin{array}{l}
u_t + u u_x + v u_y = - p_x + Re^{-1}(u_{xx} + u_{yy}),\\
v_t + u v_x + v v_y = - p_y + Re^{-1}(v_{xx} + v_{yy}) - \eta_{tt},\\
u_x + v_y = 0.
\end{array}
$$

Here, $$u(t,x,y)$$ and $$v(t,x,y)$$ are the streamwise and crossflow components of the velocity field, respectively, while $$p(t,x,y)$$ denotes the pressure, and $$Re$$ is the [Reynolds number](https://en.wikipedia.org/wiki/Reynolds_number) based on the cylinder diameter and the free stream velocity. We consider the incompressible Navier-Stokes equations in the coordinate system attached to the cylinder, so that the cylinder appears stationary in time. This explains the appearance of the extra term $$\eta_{tt}$$ in the second momentum equation.

**Problem 1 (VIV-I):** Given scattered and potentially noisy measurements $$\{t^n, x^n, y^n, u^n, v^n\}_{n=1}^N$$ of the velocity field -- Take for example the case of reconstructing a flow field from scattered measurements obtained from [Particle Image Velocimetry](https://en.wikipedia.org/wiki/Particle_image_velocimetry) (PIV) or [Particle Tracking Velocimetry](https://en.wikipedia.org/wiki/Particle_tracking_velocimetry) (PTV) -- in addition to the data $$\{t^n,\eta^n\}_{n=1}^{N}$$ on the displacement and knowing the governing equations of the flow, we are interested in reconstructing the entire velocity field as well as the pressure field in space-time. Such measurements are usually collected only in a small sub-domain, which may not be appropriate for classical CFD computations due to the presence of numerical artifacts. Typically, the data points are scattered in both space and time and are usually of the order of a few thousands or less in space. For a visual representation of the distribution of observation points $$\{t^n, x^n, y^n\}_{n=1}^N$$ scattered in space and time please refer to the following figure.

![](http://www.dam.brown.edu/people/mraissi/assets/img/PointCloud.png)
> _Point-cloud:_ Depicted on the left panel is a visual representation of the distribution of observation points scattered in space and time. A randomly selected snapshot of the distribution of data points is also shown in the right panel. To capture the boundary layers a finer resolution of points are sampled closer to the cylinder.


* * * * * *
#### Results

The proposed framework provides a universal treatment of coupled forward-backward stochastic differential equations of  fundamentally  different  nature and their corresponding high-dimensional partial differential equations. This generality  will be demonstrated by applying the algorithm to a wide range of canonical problems spanning a number of scientific domains  including a 100-dimensional [Black-Scholes-Barenblatt](https://en.wikipedia.org/wiki/Black–Scholes_model) equation and a 100-dimensional [Hamilton-Jacobi-Bellman](https://en.wikipedia.org/wiki/Hamilton–Jacobi–Bellman_equation) equation. These  examples are motivated by the pioneering work of [Beck et. al.](https://arxiv.org/abs/1709.05963). All data and codes used in this manuscript will be publicly available on [GitHub](https://github.com/maziarraissi/FBSNNs).

**Black-Scholes-Barenblatt Equation in 100D**

Let us start with the following forward-backward stochastic differential equations

$$
\begin{array}{l}
dX_t = \sigma\text{diag}(X_t)dW_t, ~~~ t \in [0,T],\\
X_0 = \xi,\\
dY_t = r(Y_t - Z_t' X_t)dt + \sigma Z_t'\text{diag}(X_t)dW_t, ~~~ t \in [0,T),\\
Y_T = g(X_T),
\end{array}
$$

where $$T=1$$, $$\sigma = 0.4$$, $$r=0.05$$, $$\xi = (1,0.5,1,0.5,\ldots,1,0.5) \in \mathbb{R}^{100}$$, and $$g(x) = \Vert x \Vert ^2$$. The above equations are related to the Black-Scholes-Barenblatt equation

$$
u_t = -\frac{1}{2} \text{Tr}[\sigma^2 \text{diag}(X_t^2) D^2u] + r(u - (Du)' x),
$$

with terminal condition $$u(T,x) = g(x)$$. This equation admits the explicit solution

$$
u(t,x) = \exp \left( (r + \sigma^2) (T-t) \right)g(x),
$$

which can be used to test the accuracy of the proposed algorithm. We approximate the unknown solution $$u(t,x)$$ by a 5-layer deep neural network with $$256$$ neurons per hidden layer. Furthermore, we partition the time domain $$[0,T]$$ into $$N=50$$ equally spaced intervals. Upon minimizing the loss function, using the [Adam optimizer](https://arxiv.org/abs/1412.6980) with mini-batches of size $$100$$ (i.e., $$100$$ realizations of the underlying Brownian motion), we obtain the results reported in the following figure. In this figure, we are evaluating the learned solution $$Y_t = u(t,X_t)$$ at representative realizations (not seen during training) of the underlying high-dimensional process $$X_t$$. Unlike the state of the art [algorithms](https://arxiv.org/abs/1709.05963), which can only approximate $$Y_0 = u(0,X_0)$$ at time $$0$$ and at the initial spatial point $$X_0=\xi$$, our algorithm is capable of approximating the entire solution function $$u(t,x)$$ in a single round of training as demonstrated in the following figure.

![](http://www.dam.brown.edu/people/mraissi/assets/img/BSB_Apr18_50.png)
> _Black-Scholes-Barenblatt Equation in 100D:_ Evaluations of the learned solution at representative realizations of the underlying high-dimensional process. It should be highlighted that the state of the art algorithms can only approximate the solution at time 0 and at the initial spatial point.

To further scrutinize the performance of our algorithm, in the following figure we report the mean and mean plus two standard deviations of the relative errors between model predictions and the exact solution computed based on $$100$$ independent realizations of the underlying Brownian motion. It is worth noting that in the previous figure we were plotting $$5$$ representative examples of the $$100$$ realizations used to generate the following figure. The results reported in these two figures are obtained after $$2 \times 10^4$$, $$3 \times 10^4$$, $$3 \times 10^4$$, and $$2 \times 10^4$$ consecutive iterations of the Adam optimizer with learning rates of $$10^{-3}$$, $$10^{-4}$$, $$10^{-5}$$, and $$10^{-6}$$, respectively. The total number of iterations is therefore given by $$10^5$$. Every $$10$$ iterations of the optimizer takes about $$0.88$$ seconds on a single NVIDIA Titan X GPU card. In each iteration of the Adam optimizer we are using $$100$$ different realizations of the underlying Brownian motion. Consequently, the total number of Brownian motion trajectories observed by the algorithm is given by $$10^7$$. It is worth highlighting that the algorithm converges to the exact value $$Y_0 = u(0,X_0)$$ in the first few hundred iterations of the Adam optimizer. For instance after only $$500$$ steps of training, the algorithms achieves an accuracy of around $$2.3 \times 10^{-3}$$ in terms of relative error. This is comparable to the results reported [here](https://arxiv.org/abs/1709.05963), both in terms of accuracy and the speed of the algorithm. However, to obtain more accurate estimates for $$Y_t = u(t,X_t)$$ at later times $$t>0$$ we need to train the algorithm using more iterations of the Adam optimizer.

![](http://www.dam.brown.edu/people/mraissi/assets/img/BSB_Apr18_50_errors.png)
> _Black-Scholes-Barenblatt Equation in 100D:_ Mean and mean plus two standard deviations of the relative errors between model predictions and the exact solution computed based on 100 realizations of the underlying Brownian motion.

**Hamilton-Jacobi-Bellman Equation in 100D**

Let us now consider the following forward-backward stochastic differential equations

$$
\begin{array}{l}
dX_t = \sigma dW_t, ~~~ t \in [0,T],\\
X_0 = \xi,\\
dY_t = \Vert Z_t\Vert^2 dt + \sigma Z_t'dW_t, ~~~ t \in [0,T),\\
Y_T = g(X_T),
\end{array}
$$

where $$T=1$$, $$\sigma = \sqrt{2}$$, $$\xi = (0,0,\ldots,0)\in \mathbb{R}^{100}$$, and $$g(x) = \ln\left(0.5\left(1+\Vert x\Vert^2\right)\right)$$. The above equations are related to the Hamilton-Jacobi-Bellman equation

$$
u_t = -\text{Tr}[D^2u] + \Vert Du\Vert^2,
$$

with terminal condition $$u(T,x) = g(x)$$. This equation admits the explicit solution

$$
u(t,x) = -\ln\left(\mathbb{E}\left[\exp\left( -g(x + \sqrt{2} W_{T-t}) \right) \right] \right),
$$

which can be used to test the accuracy of the proposed algorithm. In fact, due to the presence of the expectation operator $$\mathbb{E}$$ in the above equation, we can only approximately compute the exact solution. To be precise, we use $$10^5$$ Monte-Carlo samples to approximate the exact solution and use the result as ground truth. We represent the unknown solution $$u(t,x)$$ by a $$5$$-layer deep neural network with $$256$$ neurons per hidden layer. Furthermore, we partition the time domain $$[0,T]$$ into $$N=50$$ equally spaced intervals. Upon minimizing the loss function, using the Adam optimizer with mini-batches of size $$100$$ (i.e., $$100$$ realizations of the underlying Brownian motion), we obtain the results reported in the following figure. In this figure, we are evaluating the learned solution $$Y_t = u(t,X_t)$$ at a representative realization (not seen during training) of the underlying high-dimensional process $$X_t$$. It is worth noting that computing the exact solution to this problem is prohibitively costly due to the need for the aforementioned Monte-Carlo sampling strategy. That is why we are depicting only a single realization of the solution trajectories in the following figure. Unlike the state of the art [algorithms](https://arxiv.org/abs/1709.05963), which can only approximate $$Y_0 = u(0,X_0)$$ at time $$0$$ and at the initial spatial point $$X_0=\xi$$, our algorithm is capable of approximating the entire solution function $$u(t,x)$$ in a single round of training as demonstrated in the following figure.

![](http://www.dam.brown.edu/people/mraissi/assets/img/HJB_Apr18_50.png)
> _Hamilton-Jacobi-Bellman Equation in 100D:_ Evaluation of the learned solution at a representative realization of the underlying high-dimensional process. It should be highlighted that the state of the art algorithms can only approximate the solution at time 0 and at the initial spatial point.

To further investigate the performance of our algorithm, in the following figure we report the relative error between model prediction and the exact solution computed for the same realization of the underlying Brownian motion as the one used in the previous figure. The results reported in these two figures are obtained after $$2 \times 10^4$$, $$3 \times 10^4$$, $$3 \times 10^4$$, and $$2 \times 10^4$$ consecutive iterations of the Adam optimizer with learning rates of $$10^{-3}$$, $$10^{-4}$$, $$10^{-5}$$, and $$10^{-6}$$, respectively. The total number of iterations is therefore given by $$10^5$$. Every $$10$$ iterations of the optimizer takes about $$0.79$$ seconds on a single NVIDIA Titan X GPU card. In each iteration of the Adam optimizer we are using $$100$$ different realizations of the underlying Brownian motion. Consequently, the total number of Brownian motion trajectories observed by the algorithm is given by $$10^7$$. It is worth highlighting that the algorithm converges to the exact value $$Y_0 = u(0,X_0)$$ in the first few hundred iterations of the Adam optimizer. For instance after only $$100$$ steps of training, the algorithms achieves an accuracy of around $$7.3 \times 10^{-3}$$ in terms of relative error. This is comparable to the results reported [here](https://arxiv.org/abs/1709.05963), both in terms of accuracy and the speed of the algorithm. However, to obtain more accurate estimates for $$Y_t = u(t,X_t)$$ at later times $$t>0$$ we need to train the algorithm using more iterations of the Adam optimizer.

![](http://www.dam.brown.edu/people/mraissi/assets/img/HJB_Apr18_50_errors.png)
> _Hamilton-Jacobi-Bellman Equation in 100D:_ The relative error between model prediction and the exact solution computed based on a single realization of the underlying Brownian motion.

* * * * *

**Summary and Discussion**

In this work, we put forth a deep learning approach for solving coupled forward-backward stochastic differential equations and their corresponding high-dimensional partial differential equations. The resulting methodology showcases a series of promising results for a diverse collection of benchmark problems. As deep learning technology is continuing to grow rapidly both in terms of methodological, algorithmic, and infrastructural developments, we believe that this is a timely contribution that can benefit practitioners across a wide range of scientific domains. Specific applications that can readily enjoy these benefits include, but are not limited to, stochastic control, theoretical economics, and mathematical finance.

In terms of future work, one could straightforwardly extend the proposed framework in the current work to solve second-order backward stochastic differential equations. The key is to leverage the fundamental relationships between second-order backward stochastic differential equations and fully-nonlinear second-order partial differential equations. Moreover, our method can be used to solve [stochastic control](https://en.wikipedia.org/wiki/Stochastic_control) problems, where in general, to obtain a candidate for an optimal control, one needs to solve a coupled forward-backward stochastic differential equation, where the backward components influence the dynamics of the forward component.

* * * * *

**Acknowledgements**

This work received support by the DARPA EQUiPS grant N66001-15-2-4055 and the AFOSR grant FA9550-17-1-0013. All data and codes are publicly available on [GitHub](https://github.com/maziarraissi/FBSNNs).

* * * * *
## Citation

	@article{raissi2018forwardbackward,
	  title={Forward-Backward Stochastic Neural Networks: Deep Learning of High-dimensional Partial Differential Equations},
	  author={Raissi, Maziar},
	  journal={arXiv preprint arXiv:1804.07010},
	  year={2018}
	}

