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
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig2img(fig):
    """Convert a Matplotlib figure to tensorbaord image, which is a 3D np.ndarray, shape is :math:`(height, width, 3)`.
    This is in RGB format and the range of its value is :math:`[0, 255]`.

    Args:
        fig: Matplotlib figure

    Returns:
        a tensorboard image with shape :math:`(height, width, 3)`, value is :math:`[0, 255]`.
    """
    from matplotlib import pyplot as plot
    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.uint8)
    plot.close()
    return img_data[:, :, :3]
