# Burgers_Equation1D
Physics informed neural network (PINN) for the 1D Burgers Equation




This module implements the Physics Informed Neural Network (PINN) model for the 1D Burgers equation. The Burgers equation is given by (du/dt -  mu d^2u/dx^2u + u du/dt)= 0, where mu is 0.01/pi. It has an initial condition u(t=0, x) = -sin(pi x). Dirichlet boundary condition is given at x = -1,+1. The PINN model predicts u(t, x) for the input (t, x).

The effectiveness of PINNs is validated in the following works.

+  M. Raissi, et al., Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations, arXiv: 1711.10561 (2017). (https://arxiv.org/abs/1711.10561)

+  M. Raissi, et al., Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations, arXiv: 1711.10566 (2017). (https://arxiv.org/abs/1711.10566)

It is based on hysics informed neural network (PINN) for the 1D Wave equation on https://github.com/okada39/pinn_wave

