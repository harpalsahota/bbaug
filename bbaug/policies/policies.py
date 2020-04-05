"""
Module containing augmentation policies
Ref: https://github.com/tensorflow/tpu/blob/2264f53d95852efbfb82ea27f03ca749e1205968/models/official/detection/utils/autoaugment_utils.py  # noqa: 501
"""

from collections import namedtuple
import random
from typing import (
    Callable,
    Dict,
    List,
    NamedTuple,
    Tuple,
)

from imgaug.augmentables.bbs import (
    BoundingBox,
    BoundingBoxesOnImage,
)
import numpy as np

from bbaug.augmentations.augmentations import NAME_TO_AUGMENTATION

POLICY_TUPLE_TYPE = NamedTuple(
    'policy',
    [('name', str), ('probability', float), ('magnitude', str)]
)
POLICY_TUPLE = namedtuple('policy', ['name', 'probability', 'magnitude'])

__all__ = [
    'POLICY_TUPLE_TYPE',
    'POLICY_TUPLE',
    'list_policies',
    'policies_v0',
    'policies_v1',
    'policies_v2',
    'policies_v3',
    'PolicyContainer',
]


def policies_v0():
    """
    Version of the policies used in the paper
​
    :rtype: List[List[POLICY_TUPLE_TYPE]]
    :return: List of policies
    """
    policy = [
      [
          POLICY_TUPLE('Translate_X', 0.6, 4),
          POLICY_TUPLE('Equalize', 0.8, 10)
      ],
      [
          POLICY_TUPLE('Translate_Y_BBox', 0.2, 2),
          POLICY_TUPLE('Cutout', 0.8, 8)
      ],
      [
          POLICY_TUPLE('Sharpness', 0.0, 8),
          POLICY_TUPLE('Shear_X', 0.4, 0)
      ],
      [
          POLICY_TUPLE('Shear_Y', 1.0, 2),
          POLICY_TUPLE('Translate_Y_BBox', 0.6, 6)
      ],
      [
          POLICY_TUPLE('Rotate', 0.6, 10),
          POLICY_TUPLE('Color', 1.0, 6)],
    ]
    return policy


def policies_v1() -> List[List[POLICY_TUPLE_TYPE]]:
    """
    Version 1 of augmentation policies
​
    :rtype: List[List[POLICY_TUPLE_TYPE]]
    :return: List of policies
    """
    policy = [
        [
            POLICY_TUPLE('Translate_X', 0.6, 4),
            POLICY_TUPLE('Equalize', 0.8, 10)
        ],
        [
            POLICY_TUPLE('Translate_Y_BBox', 0.2, 2),
            POLICY_TUPLE('Cutout', 0.8, 8)],
        [
            POLICY_TUPLE('Sharpness', 0.0, 8),
            POLICY_TUPLE('Shear_X', 0.4, 0)
        ],
        [
            POLICY_TUPLE('Shear_Y', 1.0, 2),
            POLICY_TUPLE('Translate_Y_BBox', 0.6, 6)
        ],
        [
            POLICY_TUPLE('Rotate', 0.6, 10),
            POLICY_TUPLE('Color', 1.0, 6)
        ],
        [
            POLICY_TUPLE('Color', 0.0, 0),
            POLICY_TUPLE('Shear_X_BBox', 0.8, 4)
        ],
        [
            POLICY_TUPLE('Shear_Y_BBox', 0.8, 2),
            POLICY_TUPLE('Fliplr_BBox', 0.0, 10)
        ],
        [
            POLICY_TUPLE('Equalize', 0.6, 10),
            POLICY_TUPLE('Translate_X', 0.2, 2)
        ],
        [
            POLICY_TUPLE('Color', 1.0, 10),
            POLICY_TUPLE('Translate_Y_BBox', 0.4, 6)
        ],
        [
            POLICY_TUPLE('Rotate', 0.8, 10),
            POLICY_TUPLE('Contrast', 0.0, 10)
        ],
        [
            POLICY_TUPLE('Cutout', 0.2, 2),
            POLICY_TUPLE('Brightness', 0.8, 10)
        ],
        [
            POLICY_TUPLE('Color', 1.0, 6),
            POLICY_TUPLE('Equalize', 1.0, 2)
        ],
        [
            POLICY_TUPLE('Cutout_BBox', 0.4, 6),
            POLICY_TUPLE('Translate_Y_BBox', 0.8, 2)
        ],
        [
            POLICY_TUPLE('Color', 0.2, 8),
            POLICY_TUPLE('Rotate', 0.8, 10)
        ],
        [
            POLICY_TUPLE('Sharpness', 0.4, 4),
            POLICY_TUPLE('Translate_Y_BBox', 0.0, 4)
        ],
        [
            POLICY_TUPLE('Sharpness', 1.0, 4),
            POLICY_TUPLE('Solarize_Add', 0.4, 4)
        ],
        [
            POLICY_TUPLE('Rotate', 1.0, 8),
            POLICY_TUPLE('Sharpness', 0.2, 8)
        ],
        [
            POLICY_TUPLE('Shear_Y', 0.6, 10),
            POLICY_TUPLE('Translate_Y_BBox', 0.6, 8)
        ],
        [
            POLICY_TUPLE('Shear_X', 0.2, 6),
            POLICY_TUPLE('Translate_Y_BBox', 0.2, 10)
        ],
        [
            POLICY_TUPLE('Solarize_Add', 0.6, 8),
            POLICY_TUPLE('Brightness', 0.8, 10)
        ],
    ]
    return policy


def policies_v2() -> List[List[POLICY_TUPLE_TYPE]]:
    """
    Version 1 of augmentation policies
    ​
    :rtype: List[List[POLICY_TUPLE_TYPE]]
    :return: List of policies
    """
    policy = [
      [
          POLICY_TUPLE('Color', 0.0, 6),
          POLICY_TUPLE('Cutout', 0.6, 8),
          POLICY_TUPLE('Sharpness', 0.4, 8)
      ],
      [
          POLICY_TUPLE('Rotate', 0.4, 8),
          POLICY_TUPLE('Sharpness', 0.4, 2),
          POLICY_TUPLE('Rotate', 0.8, 10)
      ],
      [
          POLICY_TUPLE('Translate_Y', 1.0, 8),
          POLICY_TUPLE('Auto_Contrast', 0.8, 2)
      ],
      [
          POLICY_TUPLE('Auto_Contrast', 0.4, 6),
          POLICY_TUPLE('Shear_X', 0.8, 8),
          POLICY_TUPLE('Brightness', 0.0, 10)
      ],
      [
          POLICY_TUPLE('Solarize_Add', 0.2, 6),
          POLICY_TUPLE('Contrast', 0.0, 10),
          POLICY_TUPLE('Auto_Contrast', 0.6, 0)
      ],
      [
          POLICY_TUPLE('Cutout', 0.2, 0),
          POLICY_TUPLE('Solarize', 0.8, 8),
          POLICY_TUPLE('Color', 1.0, 4)
      ],
      [
          POLICY_TUPLE('Translate_Y', 0.0, 4),
          POLICY_TUPLE('Equalize', 0.6, 8),
          POLICY_TUPLE('Solarize', 0.0, 10)
      ],
      [
          POLICY_TUPLE('Translate_Y', 0.2, 2),
          POLICY_TUPLE('Shear_Y', 0.8, 8),
          POLICY_TUPLE('Rotate', 0.8, 8)
      ],
      [
          POLICY_TUPLE('Cutout', 0.8, 8),
          POLICY_TUPLE('Brightness', 0.8, 8),
          POLICY_TUPLE('Cutout', 0.2, 2)
      ],
      [
          POLICY_TUPLE('Color', 0.8, 4),
          POLICY_TUPLE('Translate_Y', 1.0, 6),
          POLICY_TUPLE('Rotate', 0.6, 6)
      ],
      [
          POLICY_TUPLE('Rotate', 0.6, 10),
          POLICY_TUPLE('Cutout_Fraction', 1.0, 4),
          POLICY_TUPLE('Cutout', 0.2, 8)
      ],
      [
          POLICY_TUPLE('Rotate', 0.0, 0),
          POLICY_TUPLE('Equalize', 0.6, 6),
          POLICY_TUPLE('Shear_Y', 0.6, 8)
      ],
      [
          POLICY_TUPLE('Brightness', 0.8, 8),
          POLICY_TUPLE('Auto_Contrast', 0.4, 2),
          POLICY_TUPLE('Brightness', 0.2, 2)
      ],
      [
          POLICY_TUPLE('Translate_Y', 0.4, 8),
          POLICY_TUPLE('Solarize', 0.4, 6),
          POLICY_TUPLE('Solarize_Add', 0.2, 10)
      ],
      [
          POLICY_TUPLE('Contrast', 1.0, 10),
          POLICY_TUPLE('Solarize_Add', 0.2, 8),
          POLICY_TUPLE('Equalize', 0.2, 4)
      ],
    ]
    return policy


def policies_v3() -> List[List[POLICY_TUPLE_TYPE]]:
    """
    Version 3 of augmentation policies
​
    :rtype: List[List[POLICY_TUPLE_TYPE]]
    :return: List of policies
    """
    policy = [
        [
            POLICY_TUPLE('Posterize', 0.8, 2),
            POLICY_TUPLE('Translate_X', 1.0, 8)
        ],
        [
            POLICY_TUPLE('Cutout_Fraction', 0.2, 10),
            POLICY_TUPLE('Sharpness', 1.0, 8)
        ],
        [
            POLICY_TUPLE('Rotate', 0.6, 8),
            POLICY_TUPLE('Rotate', 0.8, 10)
        ],
        [
            POLICY_TUPLE('Equalize', 0.8, 10),
            POLICY_TUPLE('Auto_Contrast', 0.2, 10)
        ],
        [
            POLICY_TUPLE('Solarize_Add', 0.2, 2),
            POLICY_TUPLE('Translate_Y', 0.2, 8)
        ],
        [
            POLICY_TUPLE('Sharpness', 0.0, 2),
            POLICY_TUPLE('Color', 0.4, 8)
        ],
        [
            POLICY_TUPLE('Equalize', 1.0, 8),
            POLICY_TUPLE('Translate_Y', 1.0, 8)
        ],
        [
            POLICY_TUPLE('Posterize', 0.6, 2),
            POLICY_TUPLE('Rotate', 0.0, 10)
        ],
        [
            POLICY_TUPLE('Auto_Contrast', 0.6, 0),
            POLICY_TUPLE('Rotate', 1.0, 6)
        ],
        [
            POLICY_TUPLE('Equalize', 0.0, 4),
            POLICY_TUPLE('Cutout', 0.8, 10)
        ],
        [
            POLICY_TUPLE('Brightness', 1.0, 2),
            POLICY_TUPLE('Translate_Y', 1.0, 6)
        ],
        [
            POLICY_TUPLE('Contrast', 0.0, 2),
            POLICY_TUPLE('Shear_Y', 0.8, 0)
        ],
        [
            POLICY_TUPLE('Auto_Contrast', 0.8, 10),
            POLICY_TUPLE('Contrast', 0.2, 10)
        ],
        [
            POLICY_TUPLE('Rotate', 1.0, 10),
            POLICY_TUPLE('Cutout', 1.0, 10)
        ],
        [
            POLICY_TUPLE('Solarize_Add', 0.8, 6),
            POLICY_TUPLE('Equalize', 0.8, 8)
        ],
    ]
    return policy


def list_policies() -> List:
    """
    Returns a list of policies available

    :rtype: List
    :return: List of available policies
    """
    return [
        policies_v0.__name__,
        policies_v1.__name__,
        policies_v2.__name__,
        policies_v3.__name__,
    ]


class PolicyContainer:
    """
    Policy container for all the policies available during augmentation
    """

    def __init__(
            self,
            policy_set: List[List[POLICY_TUPLE_TYPE]],
            name_to_augmentation: Dict[str, Callable] = NAME_TO_AUGMENTATION,
            return_yolo: bool = False
    ):
        """
        Policy container initialisation

        :type policy_list: List[List[POLICY_TUPLE_TYPE]]
        :param policy_list: List of policies available for augmentation
        :type name_to_augmentation: Dict[str, Callable]
        :param name_to_augmentation: Mapping of augmentation name to function
                                     reference
        :type return_yolo: bool
        :param return_yolo: Flag for returning the bounding boxes in YOLO
                            format
        """
        self.policies = policy_set
        self.augmentations = name_to_augmentation
        self.return_yolo = return_yolo

    def __getitem__(self, item: str) -> Callable:
        """
        Returns the augmentation method reference

        :type item: str
        :param item: Name of augmentation method
        :rtype Callable
        :return: Augmentation method
        """
        return self.augmentations[item]

    def _bbs_to_percent(
            self,
            bounding_boxes: List[BoundingBox],
            image_height: int,
            image_width: int,
    ) -> np.array:
        """
        Convert the augmented bounding boxes to YOLO format:
        [x_centre, y_centre, box_width, box_height]

        :type bounding_boxes: List[BoundingBox]
        :param bounding_boxes: list of augmented bounding boxes
        :type image_height: int
        :param image_height: Height of the image
        :type image_width: int
        :param image_width: Width of the image
        :rtype: np.array
        :return: Numpy array of augmented bounding boxes
        """
        return np.array([
            [
                bb.center_x / image_width,
                bb.center_y / image_height,
                bb.width / image_width,
                bb.height / image_height
            ]
            for bb in bounding_boxes
        ])

    def _bbs_to_pixel(self, bounding_boxes: List[BoundingBox]) -> np.array:
        """
        Return the augmented bounding boxes in pixel format:
        [x_min, y_min, x_max, y_max]

        :type bounding_boxes: List[BoundingBox]
        :param bounding_boxes:
        :rtype: np.array
        :return: Numpy array of augmented bounding boxes
        """
        return np.array([
            [
                bb.x1,
                bb.y1,
                bb.x2,
                bb.y2
            ]
            for bb in bounding_boxes
        ]).astype('int32')

    def select_random_policy(self) -> List[POLICY_TUPLE]:
        """
        Selects a random policy from the list of available policies

        :rtype: List[POLICY_TUPLE]
        :return: Randomly selected policy
        """
        return random.choice(self.policies)

    def apply_augmentation(
            self,
            policy: List[POLICY_TUPLE],
            image: np.array,
            bounding_boxes: List[List[int]]
    ) -> Tuple[np.array, np.array]:
        """
        Applies the augmentations to the image.

        :type policy: List[POLICY_TUPLE]
        :param policy: Augmentation policy to apply to the image
        :type image: np.array
        :param image: Image to augment
        :type bounding_boxes: List[List[int]]
        :param bounding_boxes: Bounding boxes for the image in the format:
                               [x_min, y_min, x_max, y_max]
        :rtype: Tuple[np.array, np.array]
        :return: Tuple containing the augmented image and bounding boxes
        """
        bbs = BoundingBoxesOnImage(
            [BoundingBox(*bb) for bb in bounding_boxes],
            image.shape
        )
        for i in policy:
            if np.random.random() < i.probability:
                if (i.name == 'Cutout') or (i.name == 'Cutout_BBox'):
                    kwargs = {
                        'height': image.shape[0],
                        'width': image.shape[1]
                    }
                    aug = self[i.name](i.magnitude, **kwargs)
                elif i.name == 'Cutout_Fraction':
                    if len(bbs) == 0:
                        aug = self[i.name](i.magnitude)
                    else:
                        random_bb = np.random.choice(bbs)
                        kwargs = {
                            'height_bbox': random_bb.height,
                            'width_bbox': random_bb.width,
                            'height': image.shape[0],
                            'width': image.shape[1]
                        }
                        aug = self[i.name](i.magnitude, **kwargs)
                else:
                    aug = self[i.name](i.magnitude)
                image, bbs = aug(image=image, bounding_boxes=bbs)
                bbs = bbs.remove_out_of_image().clip_out_of_image()
        if self.return_yolo:
            bbs = self._bbs_to_percent(bbs, image.shape[0], image.shape[1])
        else:
            bbs = self._bbs_to_pixel(bbs)
        return image, bbs
