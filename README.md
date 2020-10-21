# pde-fourier-transform
Generates PDE solutions using the Fast Fourier Transform and a numeric ODE integrator.

## 2D Wave Equation Solution
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial^2%20u}{\partial%20t^2}%20=%20v^2\nabla^2%20u=v^2(\frac{\partial^2%20u}{\partial%20x^2}%20+%20\frac{\partial^2%20u}{\partial%20y^2})"/>
<br/>Boundary conditions
<br/><img src="https://latex.codecogs.com/gif.latex?\text{Solution%20is%20defined%20on%20}%20D%20\implies%20\frac{\partial%20u}{\partial%20t}\rvert_{(\partial%20D,t)}=0"/>
<br/><img src="https://latex.codecogs.com/gif.latex?u(\vec{x},%200)=f(\vec{x})"/>
<br/><img src="https://latex.codecogs.com/gif.latex?\frac{\partial%20u}{\partial%20t}\rvert_{(\vec{x},%200)}=g(\vec{x})"/>
<br/>The ode integrator solves the equation
<br/><img src="https://latex.codecogs.com/gif.latex?\frac{d\hat{u}}{dt}=-v^2||\vec{\omega}||^2%20\hat{u}=-v^2(\omega_1^2%20+%20\omega_2^2)\hat{u}"/>