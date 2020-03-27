import pytest

from bbaug.augmentations import augmentations
from bbaug.exceptions import InvalidMagnitude


class TestNegate:

    def test_negate_positive(self, mocker):
        numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
        numpy_random_mock.return_value = 0.5

        @augmentations.negate
        def dumb_return():
            return 1
        assert dumb_return() > 0

        numpy_random_mock.return_value = 0.5
        assert dumb_return() > 0

    def test_negate_negative(self, mocker):
        numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
        numpy_random_mock.return_value = 0.49

        @augmentations.negate
        def dumb_return():
            return 1

        assert dumb_return() < 0


def test_validate_magnitude():

    @augmentations.validate_magnitude
    def dumb_func(magnitude):
        return magnitude

    with pytest.raises(InvalidMagnitude):
        dumb_func(11)

    with pytest.raises(InvalidMagnitude):
        dumb_func(-0.1)

    dumb_func(0)
    dumb_func(5)
    dumb_func(10)

# def test__img_enhance_to_art():
#
#     augmentations._img_enhance_to_arg()