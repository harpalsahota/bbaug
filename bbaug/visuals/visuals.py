""" Module to visualise policies """

from typing import (
    List
)

import imageio
from imgaug.augmentables.bbs import (
    BoundingBox,
    BoundingBoxesOnImage,
)
import matplotlib.pyplot as plt

from bbaug.augmentations import NAME_TO_AUGMENTATION
from bbaug.policies import POLICY_TUPLE_TYPE


def _aug_to_function(augmentation: POLICY_TUPLE_TYPE):
    """
    Obtain the method to apply the augmentation

    :type augmentation: POLICY_TUPLE_TYPE
    :param augmentation: Single augmentation
    :rtype: Function
    :return: Method to apply the augmentation
    """
    return NAME_TO_AUGMENTATION[augmentation.name](augmentation.magnitude)


def visualise_policy(
        image_path: str,
        bounding_boxes: List[List[int]],
        policy: List[POLICY_TUPLE_TYPE],
) -> None:
    """
    Visualise a single policy on an image

    :type image_path: str
    :param image_path: Path of the image
    :type bounding_boxes: List[List[int]]
    :param bounding_boxes: Bounding boxes for the image
    :type policy: List[POLICY_TUPLE_TYPE]
    :param policy: Single policy to apply to image
    :rtype: None
    """
    font_dict = {'fontsize': 9, 'fontweight': 'medium'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    image = imageio.imread(image_path)
    bbs = BoundingBoxesOnImage([
        BoundingBox(*box)
        for box in bounding_boxes
    ], shape=image.shape)
    [ax.axis('off') for ax in axes]

    img_aug, bbs_aug = _aug_to_function(policy[0])(image=image, bounding_boxes=bbs)
    axes[0].imshow(bbs_aug.draw_on_image(img_aug, size=2))
    axes[0].set_title(policy[0], fontdict=font_dict)

    img_aug, bbs_aug = _aug_to_function(policy[1])(image=image, bounding_boxes=bbs)
    axes[1].imshow(bbs_aug.draw_on_image(img_aug, size=2))
    axes[1].set_title(policy[1], fontdict=font_dict)

    img_aug, bbs_aug = _aug_to_function(policy[0])(image=image, bounding_boxes=bbs)
    img_aug, bbs_aug = _aug_to_function(policy[1])(image=img_aug, bounding_boxes=bbs_aug)
    axes[2].imshow(bbs_aug.draw_on_image(img_aug, size=2))
    axes[2].set_title(f'{policy[0]}\n{policy[1]}', fontdict=font_dict)

    fig.tight_layout()
    plt.show()
