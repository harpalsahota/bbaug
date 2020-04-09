from unittest.mock import call

import numpy as np
import pytest

from bbaug.policies import (
    POLICY_TUPLE,
    PolicyContainer,
    policies_v3
)
from bbaug.visuals import visualise_policy

def test_visualise_policy(mocker):
    mock_plt = mocker.patch('bbaug.visuals.visuals.plt')
    colour_mock = mocker.patch('bbaug.augmentations.augmentations.colour')
    imgio_mock = mocker.patch('bbaug.visuals.visuals.imageio')
    policy_container_mock = mocker.patch('bbaug.visuals.visuals.PolicyContainer')

    def aug_mock(image, bounding_boxes):
        return image, bounding_boxes

    class MockShape:

        @property
        def shape(self):
            return 100, 100, 3

    axes_mock = mocker.MagicMock()
    fig_mock = mocker.MagicMock()
    mock_plt.subplots.return_value = (fig_mock, axes_mock)
    colour_mock.return_value = aug_mock
    imgio_mock.imread.return_value = MockShape()
    policy_container_mock().apply_augmentation.return_value = (
        np.zeros((100, 100, 3)).astype('uint8'),
        np.array([[0, 50, 25, 75]])
    )
    visualise_policy(
        './test/image/dir.png',
        './test/save/dir',
        [[0, 50, 25, 75]],
        [9],
        [[POLICY_TUPLE('Color', 0.2, 10)]],
        name_to_augmentation={'Color': colour_mock}
    )

    mock_plt.subplots.assert_called_with(1, 3, figsize=(15, 4))
    imgio_mock.imread.assert_called_with('./test/image/dir.png')
    fig_mock.suptitle.assert_called_with([POLICY_TUPLE('Color', pytest.approx(0.2), 10)])
    assert fig_mock.savefig.call_args_list == [call('./test/save/dir/sub_policy_0.png')]
