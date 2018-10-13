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

To solve the aforementioned problem, we proceed by approximating the latent functions $$u(t,x,y)$$, $$v(t,x,y)$$, $$p(t,x,y)$$, and $$\eta(t)$$ by a single neural network outputting four variables while taking as input $$t, x$$, and $$y$$. This prior assumption along with the incompressible Navier-Stokes equations result in the following Navier-Stokes informed neural networks (see the following figure)

$$
\begin{array}{l}
e_1 := u_t + u u_x + v u_y + p_x - Re^{-1}(u_{xx} + u_{yy}),\\
e_2 := v_t + u v_x + v v_y + p_y - Re^{-1}(v_{xx} + v_{yy}) + \eta_{tt},\\
e_3 := u_x + v_y.
\end{array}
$$

![](http://www.dam.brown.edu/people/mraissi/assets/img/DeepVIV_2.png)
> _Navier-Stokes informed neural networks:_ A plain vanilla densely connected (physics uninformed) neural network, with 10 hidden layers and 32 neurons per hidden layer per output variable (i.e., 4 x 32 = 128 neurons per hidden layer), takes the input variables t, x, y and outputs the dispacement, u, v, and p. As for the activation functions, we use sin(x). For illustration purposes only, the network depicted in this figure comprises of 2 hidden layers and 5 neurons per hidden layers. We employ automatic differentiation to obtain the required derivatives to compute the residual (physics informed) networks. If a term does not appear in the blue boxes, its coefficient is assumed to be zero. It is worth emphasizing that unless the coefficient in front of a term is non-zero, that term is not going to appear in the actual "compiled" computational graph and is not going to contribute to the computational cost of a feed forward evaluation of the resulting network. The total loss function is composed of the regression loss of the velocity fields u, v and the displacement on the training data, and the loss imposed by the differential equations. Here, the differential operators are computed using automatic differentiation and can be thought of as "activation operators". Moreover, the gradients of the loss function are back-propagated through the entire network to train the neural network parameters using the Adam optimizer.

We use automatic differentiation to obtain the required derivatives to compute the residual networks $$e_1$$, $$e_2$$, and $$e_3$$. The shared parameters of the neural networks $$u$$, $$v$$, $$p$$, $$\eta$$, $$e_1$$, $$e_2$$, and $$e_3$$ can be learned by minimizing the sum of squared errors loss function

$$
\begin{array}{l}
\sum_{n=1}^N \left( \vert u(t^n,x^n,y^n)-u^n \vert^2 + \vert v(t^n,x^n,y^n)-v^n \vert^2 \right)\\
+ \sum_{n=1}^N \vert \eta(t^n)-\eta^n \vert^2 + \sum_{i=1}^3\sum_{n=1}^N \left( \vert e_i(t^n,x^n,y^n) \vert^2 \right).
\end{array}
$$

Here, the first two summations correspond to the training data on the fluid velocity and the structure displacement while the last summation enforces the dynamics imposed by the incompresible Navier-Stokes equations.

The fluid forces on the cylinder are functions of the pressure and the velocity gradients. Consequently, having trained the neural networks, we can use

$$
F_D = \oint \left[-p n_x + 2 Re^{-1} u_x n_x + Re^{-1} \left(u_y + v_x\right)n_y\right]ds,
$$

$$
F_L = \oint \left[-p n_y + 2 Re^{-1} v_y n_y + Re^{-1} \left(u_y + v_x\right)n_x\right]ds,
$$

to obtain the lift and drag forces exerted by the fluid on the cylinder. Here, $$(n_x,n_y)$$ is the outward normal on the cylinder and $$ds$$ is the arc length on the surface of the cylinder. We use the trapezoidal rule to approximately compute these integrals, and we use the above equations to obtain the required data on the lift force. These data are then used to estimate the structural parameters $$b$$ and $$k$$ by minimizing the first loss function introduced in this document.

**Inferring Lift and Drag Forces from Flow Visualizations**

We now consider the second VIV learning problem by taking one step further and circumvent the need for having access to measurements of the velocity field by leveraging the following equation

$$
c_t + u c_x + v c_y = Pe^{-1} (c_{xx} + c_{yy}),
$$

governing the evolution of the concentration $$c(t,x,y)$$ of a passive scalar injected into the fluid flow dynamics described by the incompressible Navier-Stokes equations. Here, $$Pe$$ denotes the [Péclet number](https://en.wikipedia.org/wiki/Péclet_number), defined based on the cylinder diameter, the free-stream velocity and the diffusivity of the concentration species.


**Problem 2 (VIV-II):** Given scattered and potentially noisy measurements $$\{t^n,x^n,y^n,c^n\}_{n=1}^N$$ of the concentration $$c(t,x,y)$$ of the passive scalar in space-time, we are interested in inferring the latent (hidden) quantities $$u(t,x,y)$$, $$v(t,x,y)$$, and $$p(t,x,y)$$ while leveraging the governing equations of the flow as well as the transport equation describing the evolution of the passive scalar. Typically, the data points are of the order of a few thousands or less in space. Moreover, the equations for lift and drag enable us to consequently compute the drag and lift forces, respectively, as functions of the inferred pressure and velocity gradients. Unlike the first VIV problem, here we assume that we do not have access to direct observations of the velocity field.

To solve the second VIV problem, in addition to approximating $$u(t,x,y)$$, $$v(t,x,y)$$, $$p(t,x,y)$$, and $$\eta(t)$$ by deep neural networks as before, we represent $$c(t,x,y)$$ by yet another output of the network taking $$t, x,$$ and $$y$$ as inputs. This prior assumption along with the scalar transport equation result in the following additional component of the Navier-Stokes informed neural network (see the following figure)

$$
\begin{array}{l}
e_4 := c_t + u c_x + v c_y - Pe^{-1}(c_{xx} + c_{yy}).
\end{array}
$$

![](http://www.dam.brown.edu/people/mraissi/assets/img/DeepVIV_3.png)
> _Navier-Stokes informed neural networks:_ A plain vanilla densely connected (physics uninformed) neural network, with 10 hidden layers and 32 neurons per hidden layer per output variable (i.e., 5 x 32 = 160 neurons per hidden layer), takes the input variables t, x, y and outputs the displacement, c, u, v, w, and p. As for the activation functions, we use sin(x). For illustration purposes only, the network depicted in this figure comprises of 2 hidden layers and 6 neurons per hidden layers. We employ automatic differentiation to obtain the required derivatives to compute the residual (physics informed) networks. If a term does not appear in the blue boxes, its coefficient is assumed to be zero. It is worth emphasizing that unless the coefficient in front of a term is non-zero, that term is not going to appear in the actual "compiled" computational graph and is not going to contribute to the computational cost of a feed forward evaluation of the resulting network. The total loss function is composed of the regression loss of the passive scalar c and the displacement on the training data, and the loss imposed by the differential equations. Here, the differential operators are computed using automatic differentiation and can be thought of as "activation operators". Moreover, the gradients of the loss function are back-propagated through the entire network to train the neural networks parameters using the Adam optimizer.

The residual networks $$e_1$$, $$e_2$$, and $$e_3$$ are defined as before. We use automatic differentiation to obtain the required derivatives to compute the additional residual network $$e_4$$. The shared parameters of the neural networks $$c$$, $$u$$, $$v$$, $$p$$, $$\eta$$, $$e_1$$, $$e_2$$, $$e_3$$, and $$e_4$$ can be learned by minimizing the sum of squared errors loss function

$$
\begin{array}{l}
\sum_{n=1}^N \left(\vert c(t^n,x^n,y^n)-c^n \vert^2 + \vert \eta(t^n)-\eta^n \vert^2\right)\\
+ \sum_{m=1}^M \left( \vert u(t^m,x^m,y^m)-u^m \vert^2 + \vert v(t^m,x^m,y^m)-v^m \vert^2 \right)\\
+ \sum_{i=1}^4\sum_{n=1}^N \left( \vert e_i(t^n,x^n,y^n) \vert^2 \right).
\end{array}
$$

Here, the first summation corresponds to the training data on the concentration of the passive scalar and the structure's displacement, the second summation corresponds to the Dirichlet boundary data on the velocity field, and the last summation enforces the dynamics imposed by Navier-Stokes equations and the equation corresponding to the passive scalar. Upon training, we use the lift and drag equations to obtain the required data on the lift force. Such data are then used to estimate the structural parameters $$b$$ and $$k$$ by minimizing the first loss function introduced in this document.

* * * * * *
#### Results

To generate a high-resolution dataset for the VIV problem we have performed direct numerical simulations (DNS) employing the high-order [spectral-element method](https://global.oup.com/academic/product/spectralhp-element-methods-for-computational-fluid-dynamics-9780198528692?cc=us&lang=en&), together with the coordinate transformation method to take account of the boundary deformation. The computational domain is $$[-6.5\,D,23.5\,D] \times [-10\,D,10 \, D]$$, consisting of 1,872 quadrilateral elements. The cylinder center was placed at $$(0, 0)$$. On the inflow, located at $$x/D=-6.5$$, we prescribe $$(u=U_{\infty},v=0)$$. On the outflow, where $$x/D=23.5$$, zero-pressure boundary condition $$(p=0)$$ is imposed. On both top and bottom boundaries where $$y/D=\pm 10$$, a periodic boundary condition is used. The Reynolds number is $$Re=100$$, $$\rho=2$$, $$b=0.084$$ and $$k=2.2020$$. For the case with dye, we assumed the Péclet number $$Pe=90$$. First, the simulation is carried out until $$t=1000 \, \frac{D}{U_{\infty}}$$ when the system is in steady periodic state. Then, an additional simulation for $$\Delta t=14 \,\frac{D}{U_{\infty}}$$ is performed to collect the data that are saved in 280 field snapshots. The time interval between two consecutive snapshots is $$\Delta t= 0.05 \frac{D}{U_{\infty}}$$. Note here $$D=1$$ is the diameter of the cylinder and $$U_{\infty}=1$$ is the inflow velocity. We use the DNS results to compute the lift and drag forces exerted by the fluid on the cylinder. All data and codes used in this manuscript will be publicly available on [GitHub](https://github.com/maziarraissi/DeepVIV).

**A Pedagogical Example**

To illustrate the effectiveness of our approach, let us start with the two time series depicted in the following figure consisting of $$N=111$$ observations of the displacement and the lift force. These data correspond to damping and stiffness parameters with exact values $$b=0.084$$ and $$k=2.2020$$, respectively. Here, the cylinder is assumed to have a mass of $$\rho = 2.0$$. This data-set is then used to train a 10-layer deep neural network with 32 neurons per hidden layers by minimizing the sum of squared errors loss function (the first loss function introduced in the current work) using the [Adam optimizer](https://arxiv.org/abs/1412.6980). Upon training, the network is used to predict the entire solution functions $$\eta(t)$$ and $$f_L(t)$$, as well as the unknown structural parameters $$b$$ and $$k$$. In addition to almost perfect reconstructions of the two time series for displacement and lift force, the proposed framework is capable of identifying the correct values for the structural parameters $$b$$ and $$k$$ with remarkable accuracy. The learned values for the damping and stiffness parameters are $$b = 0.08438281$$ and $$k = 2.2015007$$. This corresponds to around $$0.45\%$$ and $$0.02\%$$ relative errors in the estimated values for $$b$$ and $$k$$, respectively.

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

