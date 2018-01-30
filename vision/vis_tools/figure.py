__author__ = 'xyliao'
import numpy as np


def fig2data(fig):
    """Brief convert a Matplotlib figure to numpy.ndarray.

    Args:
        fig: a matplotlib figure

    Returns:
        return a 4D numpy.ndarray with RGBA channels.
    """
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig4board(fig):
    from matplotlib import pyplot as plot
    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.uint8)
    plot.close()
    return img_data[:, :, :3]
