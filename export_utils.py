# Animate a function as a time series

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from math_utils import closest_index

# Initialize the animation
def init():
    line.set_data([], [])
    return line,

# Generate animation mp4
def gen_animation(u, inp_range, x_vals, t_vals, dt):
    # Initialize figure
    fig = plt.figure()
    ax = plt.axes(xlim=x_vals, ylim=(-2,2))
    line, = ax.plot([], [], lw=2)

    # Indicies for x range
    left = closest_index(x_vals[0], inp_range)
    right = closest_index(x_vals[1], inp_range)
    trunc_x = inp_range[left:right]

    # Get the time values to model the function over
    t_range = np.arange(t_vals[0], t_vals[1], dt)

    # Generate each animation frame
    def animate(i):
        y = u(i)[left:right]

        line.set_data(trunc_x, y)
        return line

    # Generate MatPlotLib FuncAnimation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=t_range, interval=1000*dt)

    # Try saving a function mp4
    try:
        anim.save('wave.mp4')
    except Exception:
        print('Please install a working animation writer')
