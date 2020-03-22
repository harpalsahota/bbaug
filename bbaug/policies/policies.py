"""
Module containing augmentation policies
Ref: https://github.com/tensorflow/tpu/blob/2264f53d95852efbfb82ea27f03ca749e1205968/models/official/detection/utils/autoaugment_utils.py  # noqa: 501
"""

from collections import namedtuple
import random
from typing import (
    List,
    Tuple,
)

from imgaug.augmentables.bbs import (
    BoundingBox,
    BoundingBoxesOnImage,
)
import numpy as np

from bbaug.augmentations.augmentations import NAME_TO_AUGMENTATION

POLICY_TUPLE = namedtuple('policy', ['name', 'probability', 'magnitude'])


def policies_v3() -> List[POLICY_TUPLE[str, float, int]]:
    """
    Version 3 of augmentation policies
​
    :rtype: List[Tuple[str, float, int]]
    :return: List of policies
    """
    policy = [
        [
            POLICY_TUPLE('Posterize', 0.8, 2),
            POLICY_TUPLE('TranslateX_BBox', 1.0, 8)
        ],
        [
            POLICY_TUPLE('BBox_Cutout', 0.2, 10),
            POLICY_TUPLE('Sharpness', 1.0, 8)
        ],
        [
            POLICY_TUPLE('Rotate_BBox', 0.6, 8),
            POLICY_TUPLE('Rotate_BBox', 0.8, 10)
        ],
        [
            POLICY_TUPLE('Equalize', 0.8, 10),
            POLICY_TUPLE('AutoContrast', 0.2, 10)
        ],
        [
            POLICY_TUPLE('SolarizeAdd', 0.2, 2),
            POLICY_TUPLE('TranslateY_BBox', 0.2, 8)
        ],
        [
            POLICY_TUPLE('Sharpness', 0.0, 2),
            POLICY_TUPLE('Color', 0.4, 8)
        ],
        [
            POLICY_TUPLE('Equalize', 1.0, 8),
            POLICY_TUPLE('TranslateY_BBox', 1.0, 8)
        ],
        [
            POLICY_TUPLE('Posterize', 0.6, 2),
            POLICY_TUPLE('Rotate_BBox', 0.0, 10)
        ],
        [
            POLICY_TUPLE('AutoContrast', 0.6, 0),
            POLICY_TUPLE('Rotate_BBox', 1.0, 6)
        ],
        [
            POLICY_TUPLE('Equalize', 0.0, 4),
            POLICY_TUPLE('Cutout', 0.8, 10)
        ],
        [
            POLICY_TUPLE('Brightness', 1.0, 2),
            POLICY_TUPLE('TranslateY_BBox', 1.0, 6)
        ],
        [
            POLICY_TUPLE('Contrast', 0.0, 2),
            POLICY_TUPLE('ShearY_BBox', 0.8, 0)
        ],
        [
            POLICY_TUPLE('AutoContrast', 0.8, 10),
            POLICY_TUPLE('Contrast', 0.2, 10)
        ],
        [
            POLICY_TUPLE('Rotate_BBox', 1.0, 10),
            POLICY_TUPLE('Cutout', 1.0, 10)
        ],
        [
            POLICY_TUPLE('SolarizeAdd', 0.8, 6),
            POLICY_TUPLE('Equalize', 0.8, 8)
        ],
    ]
    return policy


class PolicyContainer:

    def __init__(self, policy_list, name_to_augmentation=NAME_TO_AUGMENTATION):
        self.policies = policy_list
        self.augmentations = name_to_augmentation

    def __getitem__(self, item):
        return self.augmentations[item]

    def _bbs_to_percent(self, bounding_boxes, image_height, image_width):
        return np.array([
            [
                bb.center_x / image_width,
                bb.center_y / image_height,
                bb.width / image_width,
                bb.height / image_height
            ]
            for bb in bounding_boxes
        ])

    def select_random_policy(self):
        return random.choice(self.policies)

    def apply_augmentation(self, policy, image, bounding_boxes):
        bbs = BoundingBoxesOnImage(
            [BoundingBox(*bb) for bb in bounding_boxes],
            image.shape
        )
        for i in policy:
            if np.random.random() < i.probability:
                if (i.name == 'Cutout') or (i.name == 'BBox_Cutout'):
                    kwargs = {
                        'height': image.shape[0],
                        'width': image.shape[1]
                    }
                    aug = self[i.name](i.magnitude, **kwargs)
                else:
                    aug = self[i.name](i.magnitude)
                image, bbs = aug(image=image, bounding_boxes=bbs)
                bbs = bbs.remove_out_of_image().clip_out_of_image()
        return image, self._bbs_to_percent(bbs, image.shape[0], image.shape[1])
