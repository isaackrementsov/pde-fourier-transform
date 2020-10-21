# Animate a function as a time series

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from math_utils import closest_index

# Generate & save animation mp4
def gen_animation(data, x, t, z_range, labels):
    dt = np.abs(t[0] - t[1])

    plot_args = {'cmap': 'coolwarm', 'linewidth': 0}
    # Initialize line
    fig = plt.figure(figsize=(10,8), dpi=200)
    ax = fig.gca(projection='3d')
    plot = ax.plot_surface(x[0], x[1], data[0], **plot_args)
    ax.set_zlim(z_range[0], z_range[1])
    
    # Generate each animation frame
    def animate(i):
        nonlocal plot
        
        u = data[i]
        
        plot.remove()
        plot = ax.plot_surface(x[0], x[1], u, **plot_args)
        return plot,

    # Generate MatPlotLib FuncAnimation
    anim = animation.FuncAnimation(fig, animate, frames=len(t), interval=1000*dt)

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
