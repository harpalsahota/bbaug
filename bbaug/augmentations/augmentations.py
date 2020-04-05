"""
Module for object detection augmentation using google policies
Paper: https://arxiv.org/abs/1906.11172
"""
from functools import wraps

from imgaug import augmenters as iaa
import numpy as np

from bbaug.exceptions import InvalidMagnitude

_MAX_MAGNITUDE = 10.0
BBOX_TRANSLATION = 120
CUTOUT_BBOX = 50
CUTOUT_MAX_PAD_FRACTION = 0.75
CUTOUT_CONST = 100
TRANSLATION_CONST = 250

__all__ = [
    'negate',
    'NAME_TO_AUGMENTATION',
    'auto_contrast',
    'brightness',
    'colour',
    'contrast',
    'cutout',
    'cutout_bbox',
    'cutout_fraction',
    'equalise',
    'fliplr_boxes',
    'posterize',
    'rotate',
    'sharpness',
    'shear_x',
    'shear_y',
    'solarize',
    'solarize_add',
    'translate_x',
    'translate_x_bbox',
    'translate_y',
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


def validate_magnitude(func):
    """
    Wrapper func to ensure magnitude is within the expected range

    :param func: func to test magnitude of
    :return: func
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        magnitude = args[0]
        if (magnitude < 0) or (magnitude > 10):
            raise InvalidMagnitude(
                f'Magnitude should be > 0 and < 10. Actual value: {magnitude}'
            )
        return func(*args, **kwargs)
    return wrapper


@validate_magnitude
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
@validate_magnitude
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
@validate_magnitude
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
@validate_magnitude
def _translate_mag_to_arg(magnitude: int, bbox=False) -> int:
    """
    Determine translation magnitude in pixels

    :type magnitude: int
    :param magnitude: Magnitude of translation
    :rtype: int
    :return: Translation in pixels
    """
    if bbox:
        return int((magnitude / _MAX_MAGNITUDE) * BBOX_TRANSLATION)
    return int((magnitude / _MAX_MAGNITUDE) * TRANSLATION_CONST)


def auto_contrast(_: int) -> iaa.pillike.Autocontrast:
    """
    Apply auto contrast to image

    Tensorflow Policy Equivalent: autocontrast

    :type _: int
    :param _: unused magnitude
    :rtype: iaa.pillike.Autocontrast
    :return: Method to auto contrast image
    """
    return iaa.pillike.Autocontrast(0)


def brightness(magnitude: int) -> iaa.pillike.EnhanceBrightness:
    """
    Adjust the brightness of an image

    Tensorflow Policy Equivalent: brightness

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

    Tensorflow Policy Equivalent: color

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

    Tensorflow Policy Equivalent: contrast

    :type magnitude: int
    :param magnitude: magnitude of contrast change
    :rtype: iaa.GammaContrast
    :return: Method to adjust contrast of image
    """
    level = _img_enhance_to_arg(magnitude)
    return iaa.GammaContrast(level)


@validate_magnitude
def cutout(magnitude: int, **kwargs) -> iaa.Cutout:
    """
    Apply cutout anywhere in the image. Passing the height and width
    of the image as integers and as keywords will scale the bounding
    box according to the policy

    Tensorflow Policy Equivalent: cutout

    The cutout value in the policies is at a pixel level. The imgaug cutout
    augmentation method requires the cutout to be a percentage of the image.
    Passing the image height and width as kwargs will scale the cutout to
    the appropriate percentage. Otherwise the imgaug default of 20% will be
    used.

    :type magnitude: int
    :param magnitude: magnitude of cutout
    :param kwargs:
        height: height of the image as int
        width: width of the image as int
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


@validate_magnitude
def cutout_fraction(magnitude: int, **kwargs) -> iaa.Cutout:
    """
    Applies cutout to the image according to bbox information. This will
    apply only to a single bounding box in the image. For the augmentation
    to apply the policy correctly the image height and width along with the
    bounding box height and width are required as keyword arguments.

    Tensorflow Policy Equivalent: bbox_cutout

    The cutout size is determined as a fraction of the bounding box size.
    The cutout value in the policies is at a pixel level. The imgaug cutout
    augmentation method requires the cutout to be a percentage of the image.
    Passing the image height and width as kwargs will scale the cutout to the
    appropriate percentage. Otherwise the imgaug default of 20% will be used.

    Note: the cutout may not always be present in the bounding box dut to
    randomness in the location of the cutout centre

    :type magnitude: int
    :param magnitude: magnitude of cutout
    :param kwargs:
        height: height of the image as int
        width: width of the image as int
        height_bbox: height of the bounding box as int
        width_bbox: width of the bounding box as int
    :rtype: iaa.Cutout
    :return: Method to apply cutout to bounding boxes
    """
    level = (magnitude / _MAX_MAGNITUDE) * CUTOUT_MAX_PAD_FRACTION
    cutout_args = {}
    if all(
            i in kwargs
            for i in ['height', 'width', 'height_bbox', 'width_bbox']
    ):
        size = tuple([
            (level * kwargs['height_bbox']) / kwargs['height'],
            (level * kwargs['width_bbox']) / kwargs['width']
        ])
        cutout_args['size'] = size
    return iaa.Cutout(**cutout_args)


def cutout_bbox(magnitude: int, **kwargs) -> iaa.BlendAlphaBoundingBoxes:
    """
    Only apply cutout to the bounding box area. Passing the
    height and width of the image as integers and as keywords
    will scale the bounding box according to the policy. Note, the
    cutout location is chosen randomly and will only appear if it
    falls within the bounding box.

    :type magnitude: int
    :param magnitude: magnitude of cutout
    :param kwargs:
        height: height of the image as int
        width: width of the image as int
    :rtype: iaa.BlendAlphaBoundingBoxes
    :return: Method to apply cutout only to bounding boxes
    """
    level = int((magnitude/_MAX_MAGNITUDE) * CUTOUT_BBOX)
    cutout_args = {}
    if 'height' in kwargs and 'width' in kwargs:
        size = tuple([level / kwargs['height'], level / kwargs['width']])
        cutout_args['size'] = size
    return iaa.BlendAlphaBoundingBoxes(
        None,
        foreground=iaa.Cutout(**cutout_args)
    )


def equalise(_: int) -> iaa.AllChannelsHistogramEqualization:
    """
    Apply auto histogram equalisation to the image

    Tensorflow Policy Equivalent: equalize

    :type _: int
    :param _: unused magnitude
    :rtype: iaa.AllChannelsHistogramEqualization
    :return: Method to equalise image
    """
    return iaa.AllChannelsHistogramEqualization()


def fliplr_boxes(_: int) -> iaa.BlendAlphaBoundingBoxes:
    """
    Flip only the bounding boxes horizontally

    Tensorflow Policy Equivalent: flip_only_bboxes

    :type _: int
    :param _: Unused, kept to fit within the ecosystem
    :rtype: iaa.AllChannelsHistogramEqualization
    :return: Method to flip bounding boxes horizontally
    """
    return iaa.BlendAlphaBoundingBoxes(
        None,
        foreground=iaa.Fliplr(1.0)
    )


@validate_magnitude
def posterize(magnitude: int):
    """
    Posterize image

    Tensorflow Policy Equivalent: posterize

    :type magnitude: int
    :param magnitude: magnitude of posterize
    :rtype: iaa.AllChannelsHistogramEqualization
    :return: Method to posterize image
    """
    nbits = int((magnitude / _MAX_MAGNITUDE) * 4)
    if nbits == 0:
        nbits += 1
    return iaa.color.Posterize(nb_bits=nbits)


def rotate(magnitude: int) -> iaa.BlendAlphaBoundingBoxes:
    """
    Rotate the bounding box in an image

    Tensorflow Policy Equivalent: rotate_with_bboxes

    :type magnitude: int
    :param magnitude: magnitude of rotation
    :rtype: iaa.BlendAlphaBoundingBoxes
    :return: Method to apply rotation
    """
    level = _rotate_mag_to_arg(magnitude)
    return iaa.Rotate(level)


def sharpness(magnitude: int) -> iaa.pillike.EnhanceSharpness:
    """
    Add sharpness to the image

    Tensorflow Policy Equivalent: sharpness

    :type magnitude: int
    :param magnitude: magnitude of sharpness
    :rtype: iaa.pillike.EnhanceSharpness
    :return: Method to adjust sharpness
    """
    level = _img_enhance_to_arg(magnitude)
    return iaa.pillike.EnhanceSharpness(level)


def shear_x(magnitude: int) -> iaa.ShearY:
    """
    Apply x shear to the image and boxes

    Tensorflow Policy Equivalent: shear_x

    :type magnitude: int
    :param magnitude: magnitude of y shear
    :rtype: iaa.ShearY
    :return: Method to y shear bounding boxes
    """
    level = _shear_mag_to_arg(magnitude)
    return iaa.ShearX(level)


def shear_x_bbox(magnitude: int) -> iaa.BlendAlphaBoundingBoxes:
    """
    Apply x shear only to bboxes

    Tensorflow Policy Equivalent: shear_x_only_bboxes

    :type magnitude: int
    :param magnitude: magnitude of x shear
    :rtype: iaa.BlendAlphaBoundingBoxes
    :return: Method to x shear bounding boxes
    """
    level = _shear_mag_to_arg(magnitude)
    return iaa.BlendAlphaBoundingBoxes(
        None,
        foreground=iaa.ShearX(level),
    )


def shear_y(magnitude: int) -> iaa.ShearY:
    """
    Apply y shear image and boxes

    Tensorflow Policy Equivalent: shear_y

    :type magnitude: int
    :param magnitude: magnitude of y shear
    :rtype: iaa.ShearY
    :return: Method to y shear bounding boxes
    """
    level = _shear_mag_to_arg(magnitude)
    return iaa.ShearY(level)


def shear_y_bbox(magnitude: int) -> iaa.BlendAlphaBoundingBoxes:
    """
    Apply y shear only to bboxes

    Tensorflow Policy Equivalent: shear_y_only_bboxes

    :type magnitude: int
    :param magnitude: magnitude of y shear
    :rtype: iaa.BlendAlphaBoundingBoxes
    :return: Method to y shear bounding boxes
    """
    level = _shear_mag_to_arg(magnitude)
    return iaa.BlendAlphaBoundingBoxes(
        None,
        foreground=iaa.ShearY(level),
    )


def solarize(_: int) -> iaa.pillike.Solarize:
    """
    Solarize the image

    :type _: int
    :param _: Unused, kept to fit within the ecosystem
    :rtype: iaa.pillike.Solarize
    :return: Method to solarize image
    """
    return iaa.pillike.Solarize(threshold=128)


@validate_magnitude
def solarize_add(magnitude: int):
    """
    Add solarize to an image

    Tensorflow Policy Equivalent: solarize_add

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
        image_copy[np.where(image_copy < threshold)] = image_added[np.where(image_copy < threshold)]  # noqa: 501
        return image_copy, bounding_boxes
    return aug


def translate_x(magnitude: int) -> iaa.geometric.TranslateX:
    """
    Translate bounding boxes only on the x-axis

    Tensorflow Policy Equivalent: translate_x_only_bboxes

    :type magnitude: int
    :param magnitude: Magnitude of translation
    :rtype: iaa.BlendAlphaBoundingBoxes
    :return: Method to apply x translation to bounding boxes
    """
    level = _translate_mag_to_arg(magnitude)
    return iaa.geometric.TranslateX(px=level)


def translate_x_bbox(magnitude: int) -> iaa.BlendAlphaBoundingBoxes:
    """
    Translate bounding boxes only on the x-axis

    Tensorflow Policy Equivalent: translate_x

    :type magnitude: int
    :param magnitude: Magnitude of translation
    :rtype: iaa.BlendAlphaBoundingBoxes
    :return: Method to apply x translation to bounding boxes
    """
    level = _translate_mag_to_arg(magnitude, bbox=True)
    return iaa.BlendAlphaBoundingBoxes(
        None,
        foreground=iaa.geometric.TranslateX(px=level),
    )


def translate_y(magnitude: int) -> iaa.geometric.TranslateY:
    """
    Translate bounding boxes only on the y-axis

    Tensorflow Policy Equivalent: translate_y_only_bboxes

    :type magnitude: int
    :param magnitude: magnitude of translation
    :rtype: iaa.BlendAlphaBoundingBoxes
    :return: Method to apply y translation to bounding boxes
    """
    level = _translate_mag_to_arg(magnitude)
    return iaa.geometric.TranslateY(px=level)


def translate_y_bbox(magnitude: int) -> iaa.BlendAlphaBoundingBoxes:
    """
    Translate bounding boxes only on the y-axis

    Tensorflow Policy Equivalent: translate_y

    :type magnitude: int
    :param magnitude: magnitude of translation
    :rtype: iaa.BlendAlphaBoundingBoxes
    :return: Method to apply y translation to bounding boxes
    """
    level = _translate_mag_to_arg(magnitude, bbox=True)
    return iaa.BlendAlphaBoundingBoxes(
        None,
        foreground=iaa.geometric.TranslateY(px=level)
    )


NAME_TO_AUGMENTATION = {
    'Auto_Contrast': auto_contrast,
    'Brightness': brightness,
    'Cutout': cutout,
    'Cutout_BBox': cutout_bbox,
    'Cutout_Fraction': cutout_fraction,
    'Color': colour,
    'Contrast': contrast,
    'Equalize': equalise,
    'Fliplr_BBox': fliplr_boxes,
    'Posterize': posterize,
    'Rotate': rotate,
    'Sharpness': sharpness,
    'Shear_X': shear_x,
    'Shear_X_BBox': shear_x_bbox,
    'Shear_Y': shear_y,
    'Shear_Y_BBox': shear_y_bbox,
    'Solarize': solarize,
    'Solarize_Add': solarize_add,
    'Translate_X': translate_x,
    'Translate_X_BBox': translate_x_bbox,
    'Translate_Y': translate_y,
    'Translate_Y_BBox': translate_y_bbox,
}
