# Burgers_Equation1D
Physics informed neural network (PINN) for the 1D Burgers Equation




This module implements the Physics Informed Neural Network (PINN) model for the 1D Burgers equation. The Burgers equation is given by (du/dt -  mu d^2u/dx^2u + u du/dt)= 0, where mu is 0.01/pi. It has an initial condition u(t=0, x) = sin(2 pi x). Dirichlet boundary condition is given at x = -1,+1. The PINN model predicts u(t, x) for the input (t, x).

It is based on hysics informed neural network (PINN) for the 1D Wave equation on https://github.com/okada39/pinn_wa
