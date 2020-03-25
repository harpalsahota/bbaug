"""
Module for object detection augmentation using google policies
Paper: https://arxiv.org/abs/1906.11172
"""
from functools import wraps

from imgaug import augmenters as iaa
import numpy as np

_MAX_MAGNITUDE = 10.0
BBOX_TRANSLATION = 120
CUTOUT_BBOX = 50
CUTOUT_CONST = 100

__all__ = [
    'auto_contrast',
    'brightness',
    'colour',
    'contrast',
    'cutout',
    'cutout_bbox',
    'equalise',
    'posterize',
    'rotate_bbox',
    'sharpness',
    'shear_y_bbox',
    'solarize_add',
    'translate_x_bbox',
    'translate_y_bbox',
]

def negate(func):
    """
    Wrapper function to randomly reverse the direction of an augmentation

    :param func: Augmentation function
    :return: func
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if np.random.random() < 0.5:
            return -func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper


def _img_enhance_to_arg(magnitude: int) -> float:
    """
    Determine the magnitude of the image enhancement

    :type magnitude: int
    :param magnitude: Magnitude of enhancement
    :rtype: float
    :return: Magnitude of enhancement to apply
    """
    return (magnitude / _MAX_MAGNITUDE) * 1.8 + 0.1


@negate
def _rotate_mag_to_arg(magnitude: int) -> float:
    """
    Determine rotation magnitude

    :type magnitude: int
    :param magnitude: Magnitude of rotation
    :rtype: float
    :return: Rotation in degrees
    """
    return (magnitude / _MAX_MAGNITUDE) * 30


@negate
def _shear_mag_to_arg(magnitude: int) -> float:
    """
    Determine shear magnitude

    :type magnitude: int
    :param magnitude: magnitude of shear
    :rtype: float
    :return: shear magnitude
    """
    return (magnitude / _MAX_MAGNITUDE) * 0.3


@negate
def _translate_bbox_mag_to_arg(magnitude: int) -> int:
    """
    Determine translation magnitude in pixels

    :type magnitude: int
    :param magnitude: Magnitude of translation
    :rtype: int
    :return: Translation in pixels
    """
    return int((magnitude / _MAX_MAGNITUDE) * BBOX_TRANSLATION)


def auto_contrast(_: int) -> iaa.pillike.Autocontrast:
    """
    Apply auto contrast to image

    :type _: int
    :param _: unused magnitude
    :rtype: iaa.pillike.Autocontrast
    :return: Method to auto contrast image
    """
    return iaa.pillike.Autocontrast(0)


def brightness(magnitude: int) -> iaa.pillike.EnhanceBrightness:
    """
    Adjust the brightness of an image

    :type magnitude: int
    :param magnitude: Magnitude of brightness change
    :rtype: iaa.pillike.EnhanceBrightness
    :return: Method to adjust brightness in image
    """
    level = _img_enhance_to_arg(magnitude)
    return iaa.pillike.EnhanceBrightness(level)


def colour(magnitude: int) -> iaa.pillike.EnhanceColor:
    """
    Adjust the brightness of an image

    :type magnitude: int
    :param magnitude: Magnitude of colour change
    :rtype: iaa.pillike.EnhanceColor
    :return: Method to adjust colour in image
    """
    level = _img_enhance_to_arg(magnitude)
    return iaa.pillike.EnhanceColor(level)


def contrast(magnitude: int) -> iaa.GammaContrast:
    """
    Adjust the contrast of an image

    :type magnitude: int
    :param magnitude: magnitude of contrast change
    :rtype: iaa.GammaContrast
    :return: Method to adjust contrast of image
    """
    level = _img_enhance_to_arg(magnitude)
    return iaa.GammaContrast(level)


def cutout(magnitude: int, **kwargs) -> iaa.Cutout:
    """
    Apply cutout to an image

    The cutout value in the policies is at a pixel level. The imgaug cutout
    augmentation method requires the cutout to be a percentage of the image.
    Passing the image height and width as kwargs will scale the cutout to
    the appropriate percentage. Otherwise the imgaug default of 20% will be
    used.

    :type magnitude: int
    :param magnitude: magnitude of cutout
    :rtype: iaa.Cutout
    :return: Method to apply cutout to image
    """
    level = int((magnitude / _MAX_MAGNITUDE) * CUTOUT_CONST)
    cutout_args = {}
    if 'height' in kwargs and 'width' in kwargs:
        size = tuple([
            (level / kwargs['height']) * 2,
            (level / kwargs['width']) * 2
        ])
        cutout_args['size'] = size
    return iaa.Cutout(**cutout_args)


def cutout_bbox(magnitude: int, **kwargs) -> iaa.BlendAlphaBoundingBoxes:
    """
    Apply cutout only to the bounding box region

    The cutout value in the policies is at a pixel level. The imgaug cutout
    augmentation method requires the cutout to be a percentage of the image.
    Passing the image height and width as kwargs will scale the cutout to the
    appropriate percentage. Otherwise the imgaug default of 20% will be used.

    Note: the cutout may not always be present in the bounding box dut to
    randomness in the location of the cutout centre

    :type magnitude: int
    :param magnitude: magnitude of cutout
    :rtype: iaa.Cutout
    :return: Method to apply cutout to bounding boxes
    """
    level = int((magnitude / _MAX_MAGNITUDE) * CUTOUT_BBOX)
    cutout_args = {}
    if 'height' in kwargs and 'width' in kwargs:
        size = tuple([
            (level / kwargs['height']) * 2,
            (level / kwargs['width']) * 2
        ])
        cutout_args['size'] = size
    return iaa.BlendAlphaBoundingBoxes(
        None,
        foreground=iaa.Cutout(**cutout_args)
    )


def equalise(_: int) -> iaa.AllChannelsHistogramEqualization:
    """
    Apply auto histogram equalisation to the image

    :type _: int
    :param _: unused magnitude
    :rtype: iaa.AllChannelsHistogramEqualization
    :return: Method to equalise image
    """
    return iaa.AllChannelsHistogramEqualization()


def posterize(magnitude: int):
    """
    Posterize image

    :type magnitude: int
    :param magnitude: magnitude of posterize
    :rtype: iaa.AllChannelsHistogramEqualization
    :return: Method to posterize image
    """
    nbits = int((magnitude / _MAX_MAGNITUDE) * 4)
    if nbits == 0:
        nbits += 1
    return iaa.color.Posterize(nb_bits=nbits)


def rotate_bbox(magnitude: int) -> iaa.BlendAlphaBoundingBoxes:
    """
    Rotate the bounding box in an image

    :type magnitude: int
    :param magnitude: magnitude of rotation
    :rtype: iaa.BlendAlphaBoundingBoxes
    :return: Method to apply rotation
    """
    level = _rotate_mag_to_arg(magnitude)
    return iaa.BlendAlphaBoundingBoxes(
        None,
        foreground=iaa.Rotate(level)
    )


def sharpness(magnitude: int) -> iaa.pillike.EnhanceSharpness:
    """
    Add sharpness to the image

    :type magnitude: int
    :param magnitude: magnitude of sharpness
    :rtype: iaa.pillike.EnhanceSharpness
    :return: Method to adjust sharpness
    """
    level = _img_enhance_to_arg(magnitude)
    return iaa.pillike.EnhanceSharpness(level)


def shear_y_bbox(magnitude: int) -> iaa.ShearY:
    """
    Apply y shear only to the bounding box

    :type magnitude: int
    :param magnitude: magnitude of y shear
    :rtype: iaa.ShearY
    :return: Method to y shear bounding boxes
    """
    level = _shear_mag_to_arg(magnitude)
    return iaa.ShearY(level)


def solarize_add(magnitude: int):
    """
    Add solarize to an image

    :type magnitude:int
    :param magnitude: Magnitude of solarization
    :rtype: aug
    :return: Method to apply solarization
    """
    level = int((magnitude / _MAX_MAGNITUDE) * 110)

    def aug(image, bounding_boxes, threshold=128):
        image_added, image_copy = image.copy(), image.copy()
        image_added = image_added + level
        image_added = np.clip(image_added, 0, 255)
        image_copy[np.where(image_copy < threshold)] = image_added[np.where(image_copy < threshold)]
        return image_copy, bounding_boxes
    return aug


def translate_x_bbox(magnitude: int) -> iaa.BlendAlphaBoundingBoxes:
    """
    Translate bounding boxes only on the x-axis

    :type magnitude: int
    :param magnitude: Magnitude of translation
    :rtype: iaa.BlendAlphaBoundingBoxes
    :return: Method to apply x translation to bounding boxes
    """
    level = _translate_bbox_mag_to_arg(magnitude)
    return iaa.BlendAlphaBoundingBoxes(
        None,
        foreground=iaa.geometric.TranslateX(px=level),
    )


def translate_y_bbox(magnitude: int):
    """
    Translate bounding boxes only on the y-axis

    :type magnitude: int
    :param magnitude: magnitude of translation
    :rtype: iaa.BlendAlphaBoundingBoxes
    :return: Method to apply y translation to bounding boxes
    """
    level = _translate_bbox_mag_to_arg(magnitude)
    return iaa.BlendAlphaBoundingBoxes(
        None,
        foreground=iaa.geometric.TranslateY(px=level)
    )


NAME_TO_AUGMENTATION = {
    'AutoContrast': auto_contrast,
    'BBox_Cutout': cutout_bbox,
    'Brightness': brightness,
    'Cutout': cutout,
    'Color': colour,
    'Contrast': contrast,
    'Equalize': equalise,
    'Posterize': posterize,
    'Rotate_BBox': rotate_bbox,
    'Sharpness': sharpness,
    'ShearY_BBox': shear_y_bbox,
    'SolarizeAdd': solarize_add,
    'TranslateX_BBox': translate_x_bbox,
    'TranslateY_BBox': translate_y_bbox,
}
