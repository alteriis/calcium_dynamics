import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(2)

x = np.arange(0, 2*np.pi, 0.01)
line, = ax[0].plot(x, np.sin(x))
line2, = ax[1].plot(x, np.cos(x))

def animate(i):
    line.set_ydata(np.sin(x + i / 50))  # update the data.
    line2.set_ydata(np.cos(x + i / 50))  # update the data.
    return line,line2


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
