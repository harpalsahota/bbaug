""" Module to visualise policies """

from typing import (
    Callable,
    Dict,
    List
)

import imageio
from imgaug.augmentables.bbs import (
    BoundingBox,
    BoundingBoxesOnImage,
)
import matplotlib.pyplot as plt

from bbaug.augmentations import NAME_TO_AUGMENTATION
from bbaug.policies import POLICY_TUPLE_TYPE, PolicyContainer

__all__ = [
    'visualise_policy'
]


def visualise_policy(
        image_path: str,
        save_path: str,
        bounding_boxes: List[List[int]],
        policy: List[List[POLICY_TUPLE_TYPE]],
        name_to_augmentation: Dict[str, Callable] = NAME_TO_AUGMENTATION
) -> None:
    """
    Visualise a single policy on an image

    :type image_path: str
    :param image_path: Path of the image
    :type save_path: str
    :param save_path: Directory where to save the images
    :type bounding_boxes: List[List[int]]
    :param bounding_boxes: Bounding boxes for the image
    :type policy: List[List[POLICY_TUPLE_TYPE]]
    :param policy: The policy set to apply to the image
    :type name_to_augmentation: Dict[str, Callable]
    :param name_to_augmentation: Dictionary mapping of the augmentation name
                                 to the augmentation method
    :rtype: None
    """
    policy_container = PolicyContainer(
        policy,
        name_to_augmentation=name_to_augmentation
    )

    for i, pol in enumerate(policy):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        image = imageio.imread(image_path)
        [ax.axis('off') for ax in axes]
        for ax in range(3):
            img_aug, bbs_aug = policy_container.apply_augmentation(
                pol, image,
                bounding_boxes
            )
            bbs_aug = BoundingBoxesOnImage([
                BoundingBox(*box)
                for box in bbs_aug
            ], shape=image.shape)
            axes[ax].imshow(bbs_aug.draw_on_image(img_aug, size=2))
        fig.suptitle(pol)
        fig.tight_layout()
        fig.savefig(f'{save_path}/sub_policy_{i}.png')
