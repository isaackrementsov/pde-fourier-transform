# Animate a function as a time series

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from math_utils import closest_index

# Generate & save animation mp4
def gen_animation(data, x_domain, t_domain, y_range, labels):
    dt = np.abs(t_domain[0] - t_domain[1])

    # Initialize line
    fig = plt.figure()
    ax = plt.axes(xlim=(x_domain[0], x_domain[-1]), ylim=y_range)
    line, = ax.plot([], [], lw=2)

    # Label axes
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    # Initialize the animation
    def init():
        line.set_data([], [])
        return line,

    # Generate each animation frame
    def animate(i):
        u = data[i]

        line.set_data(x_domain, u)
        return line

    # Generate MatPlotLib FuncAnimation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(t_range), interval=1000*dt)

    # Try saving a function mp4
    try:
        anim.save('wave.mp4')
    except Exception as e:
        print(e)
        print('Please install a working animation writer')


# Generate & save graph image
def graph(data, x_domain, labels):
    # Initialize plot
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)

    ax.plot(x_domain, data)

    # Label axes
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    # Save figure
    fig.savefig('wave.png')
