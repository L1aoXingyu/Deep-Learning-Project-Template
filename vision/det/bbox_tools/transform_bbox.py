import numpy as np


def crop_bbox(
        bbox, y_slice=None, x_slice=None,
        allow_outside_center=True, return_param=False):
    """Translate bounding boxes to fit within the cropped area of an image.

    This method is mainly used together with image cropping.
    This method translates the coordinates of bounding boxes like
    :func:`~chainercv.transforms.translate_bbox`. In addition,
    this function truncates the bounding boxes to fit within the cropped area.
    If a bounding box does not overlap with the cropped area,
    this bounding box will be removed.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_slice (slice): The slice of y axis.
        x_slice (slice): The slice of x axis.
        allow_outside_center (bool): If this argument is :obj:`False`,
            bounding boxes whose centers are outside of the cropped area
            are removed. The default value is :obj:`True`.
        return_param (bool): If :obj:`True`, this function returns
            indices of kept bounding boxes.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`, returns an array :obj:`bbox`.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`bbox, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **index** (*numpy.ndarray*): An array holding indices of used \
            bounding boxes.

    """

    t, b = _slice_to_bounds(y_slice)
    l, r = _slice_to_bounds(x_slice)
    crop_bb = np.array((t, l, b, r))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]) \
                 .all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox


def _slice_to_bounds(slice_):
    if slice_ is None:
        return 0, np.inf

    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """Flip bounding boxes accordingly.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        y_flip (bool): Flip bounding box according to a vertical flip of
            an image.
        x_flip (bool): Flip bounding box according to a horizontal flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.

    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def translate_bbox(bbox, y_offset=0, x_offset=0):
    """Translate bounding boxes.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the left top point of the image from
    coordinate :math:`(0, 0)` to coordinate
    :math:`(y, x) = (y\_offset, x\_offset)`.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    Returns:
        ~numpy.ndarray:
        Bounding boxes translated according to the given offsets.

    """

    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox
