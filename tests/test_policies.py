from imgaug.augmentables.bbs import (
    BoundingBox,
    BoundingBoxesOnImage,
)
import numpy as np
import pytest

from bbaug.policies import policies


def test_list_policies():
    res = policies.list_policies()
    assert len(res) == 1
    assert 'policies_v3' in res


def test_policies_v3():

    v3_policies = policies.policies_v3()
    assert len(v3_policies) == 15

    for policy in v3_policies:
        assert len(policy) == 2
        for sub_policy in policy:
            assert isinstance(sub_policy, policies.POLICY_TUPLE)
            assert type(sub_policy.name) is str
            assert type(sub_policy.probability) is float
            assert type(sub_policy.magnitude) is int
            assert sub_policy.name in policies.NAME_TO_AUGMENTATION
            assert 0.0 <= sub_policy.probability <= 1.0
            assert 0 <= sub_policy.magnitude <= 10


class TestPolicyContainer:

    def test___get__item(self):
        p = policies.PolicyContainer(policies.policies_v3())
        assert p['Color'].__name__ == 'colour'

    def test__bbs_to_percent(self):
        p = policies.PolicyContainer(policies.policies_v3())
        bbs = BoundingBoxesOnImage(
            [BoundingBox(*bb) for bb in [[0, 0, 25, 25]]],
            (100, 100)
        )
        res = p._bbs_to_percent(bbs, 100, 100)
        assert np.allclose(
            np.array([[0.125, 0.125, 0.25, 0.25]]),
            res
        )

    def test__bbs_to_pixek(self):
        p = policies.PolicyContainer(policies.policies_v3())
        bbs = BoundingBoxesOnImage(
            [BoundingBox(*bb) for bb in [[0, 0, 25, 25]]],
            (100, 100)
        )
        res = p._bbs_to_pixel(bbs)
        assert np.array_equal(
            np.array([[0, 0, 25, 25]]),
            res
        )

    def test_select_random_policy(self):
        p = policies.PolicyContainer(policies.policies_v3())
        random_policy = p.select_random_policy()
        assert random_policy in p.policies

    def test_apply_augmentation(self, mocker):
        numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
        numpy_random_mock.return_value = 0.0
        bbs_to_percent_mock = mocker.patch('bbaug.policies.policies.PolicyContainer._bbs_to_percent')
        bbs_to_pixel_mock = mocker.patch('bbaug.policies.policies.PolicyContainer._bbs_to_pixel')

        def aug_mock(image, bounding_boxes):
            return image, bounding_boxes

        bbcutout_mock = mocker.patch('bbaug.augmentations.augmentations.cutout_bbox')
        bbcutout_mock.return_value = aug_mock
        p = policies.PolicyContainer(
            policies.policies_v3(),
            name_to_augmentation={'BBox_Cutout': bbcutout_mock},
        )
        policy = [policies.POLICY_TUPLE('BBox_Cutout', 0.2, 10)]
        bbs = [[0, 0, 25, 25]]
        p.apply_augmentation(policy, np.zeros((100, 100, 3)).astype('uint8'), bbs)
        assert bbcutout_mock.called
        bbcutout_mock.assert_called_with(10, height=100, width=100)
        assert not bbs_to_percent_mock.called
        assert bbs_to_pixel_mock.called

        bbs_to_percent_mock.reset_mock()
        bbs_to_pixel_mock.reset_mock()
        colour_mock = mocker.patch('bbaug.augmentations.augmentations.colour')
        colour_mock.return_value = aug_mock
        p = policies.PolicyContainer(
            policies.policies_v3(),
            name_to_augmentation={'Color': colour_mock},
            return_yolo=True
        )
        policy = [policies.POLICY_TUPLE('Color', 0.2, 10)]
        p.apply_augmentation(policy, np.zeros((100, 100, 3)).astype('uint8'), bbs)
        assert colour_mock.called
        assert bbs_to_percent_mock.called
        assert not bbs_to_pixel_mock.called
