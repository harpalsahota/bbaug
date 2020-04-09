import numpy as np
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


def test__img_enhance_to_arg():

    res = augmentations._img_enhance_to_arg(2)
    assert res == pytest.approx(0.46)

    res = augmentations._img_enhance_to_arg(8)
    assert res == pytest.approx(1.54)


def test__rotate_mag_to_arg(mocker):

    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.5

    res = augmentations._rotate_mag_to_arg(8)
    assert res == pytest.approx(24.0)

    res = augmentations._rotate_mag_to_arg(10)
    assert res == pytest.approx(30.0)


def test__shear_mag_to_arg(mocker):

    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.5

    res = augmentations._shear_mag_to_arg(0)
    assert res == pytest.approx(0.0)

    res = augmentations._shear_mag_to_arg(7)
    assert res == pytest.approx(0.21)


def test__translate_mag_to_arg(mocker):

    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.5

    res = augmentations._translate_mag_to_arg(8)
    assert res == 200

    res = augmentations._translate_mag_to_arg(2)
    assert res == 50


def test_auto_contrast(mocker):

    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.pillike.Autocontrast')

    augmentations.auto_contrast(10)
    assert aug_mock.called
    aug_mock.assert_called_with(0)


def test_brightness(mocker):

    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.pillike.EnhanceBrightness')

    augmentations.brightness(10)
    assert aug_mock.called
    aug_mock.assert_called_with(pytest.approx(1.9))

    aug_mock.reset_mock()
    augmentations.brightness(3)
    aug_mock.assert_called_with(pytest.approx(0.64))


def test_colour(mocker):

    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.pillike.EnhanceColor')

    augmentations.colour(6)
    assert aug_mock.called
    aug_mock.assert_called_with(pytest.approx(1.18))

    aug_mock.reset_mock()
    augmentations.colour(7)
    aug_mock.assert_called_with(pytest.approx(1.36))


def test_contrast(mocker):

    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.GammaContrast')

    augmentations.contrast(1)
    assert aug_mock.called
    aug_mock.assert_called_with(pytest.approx(0.28))

    aug_mock.reset_mock()
    augmentations.contrast(0)
    aug_mock.assert_called_with(pytest.approx(0.1))


def test_cutout(mocker):

    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.Cutout')

    augmentations.cutout(2)
    assert aug_mock.called
    aug_mock.assert_called_with()

    aug_mock.reset_mock()
    augmentations.cutout(5, height=1000, width=1000)
    args, kwargs = aug_mock.call_args_list[0]
    assert {'size': pytest.approx((0.1, 0.1))} == kwargs

    aug_mock.reset_mock()
    augmentations.cutout(10, height=10, width=10)
    args, kwargs = aug_mock.call_args_list[0]
    assert {'size': pytest.approx((1.0, 1.0))} == kwargs


def test_cutout_bbox(mocker):

    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.BlendAlphaBoundingBoxes')
    aug_cutout_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.Cutout')

    augmentations.cutout_bbox(10)
    args, kwargs = aug_mock.call_args_list[0]
    assert tuple([None]) == args
    assert 'foreground' in kwargs
    aug_cutout_mock.assert_called_with()

    aug_mock.reset_mock()
    aug_cutout_mock.reset_mock()
    augmentations.cutout_bbox(10, height=500, width=500)
    args, kwargs = aug_mock.call_args_list[0]
    assert tuple([None]) == args
    assert 'foreground' in kwargs
    args, kwargs = aug_cutout_mock.call_args_list[0]
    assert tuple() == args
    assert 'size' in kwargs
    assert {'size': pytest.approx((0.1, 0.1))} == kwargs
    aug_cutout_mock.assert_called_with(size=pytest.approx((0.1, 0.1)))

    aug_mock.reset_mock()
    aug_cutout_mock.reset_mock()
    augmentations.cutout_bbox(10, height=250, width=5)
    args, kwargs = aug_mock.call_args_list[0]
    assert tuple([None]) == args
    assert 'foreground' in kwargs
    args, kwargs = aug_cutout_mock.call_args_list[0]
    assert tuple() == args
    assert 'size' in kwargs
    assert {'size': pytest.approx((0.2, 1.0))} == kwargs
    aug_cutout_mock.assert_called_with(size=pytest.approx((0.2, 1.0)))


def test_cutout_fraction(mocker):

    aug_mock_cutout = mocker.patch('bbaug.augmentations.augmentations.iaa.Cutout')

    augmentations.cutout_fraction(7)
    assert aug_mock_cutout.called
    aug_mock_cutout.assert_called_with()

    aug_mock_cutout.reset_mock()
    augmentations.cutout_fraction(9, height=1000, width=1000)
    aug_mock_cutout.assert_called_with()

    aug_mock_cutout.reset_mock()
    augmentations.cutout_fraction(10, height=1000, width=1000, height_bbox=100, width_bbox=100)
    args, kwargs = aug_mock_cutout.call_args_list[0]
    assert tuple() == args
    assert 'size' in kwargs
    assert {'size': pytest.approx((0.075, 0.075))} == kwargs


def test_equalise(mocker):

    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.AllChannelsHistogramEqualization')

    augmentations.equalise(1)
    assert aug_mock.called
    aug_mock.assert_called_with()


def test_fliplr(mocker):

    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.BlendAlphaBoundingBoxes')
    aug_fliplr_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.Fliplr')

    augmentations.fliplr_boxes(1)
    args, kwargs = aug_mock.call_args_list[0]
    assert tuple([None]) == args
    assert 'foreground' in kwargs
    aug_fliplr_mock.assert_called_with(pytest.approx(1.0))


def test_posterize(mocker):

    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.color.Posterize')

    augmentations.posterize(1)
    assert aug_mock.called
    args, kwargs = aug_mock.call_args_list[0]
    assert {'nb_bits': 1} == kwargs
    assert tuple() == args

    aug_mock.reset_mock()
    augmentations.posterize(9)
    args, kwargs = aug_mock.call_args_list[0]
    assert {'nb_bits': 3} == kwargs
    assert tuple() == args


def test_rotate(mocker):

    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.5
    aug_mock_rotate = mocker.patch('bbaug.augmentations.augmentations.iaa.Rotate')

    augmentations.rotate(8)
    assert aug_mock_rotate.called
    aug_mock_rotate.assert_called_with(pytest.approx(24.0))

    aug_mock_rotate.reset_mock()
    augmentations.rotate(10)
    aug_mock_rotate.assert_called_with(pytest.approx(30.0))


def test_sharpness(mocker):

    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.pillike.EnhanceSharpness')

    augmentations.sharpness(4)
    assert aug_mock.called
    aug_mock.assert_called_with(pytest.approx(0.82))


def test_shear_x(mocker):

    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.49
    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.ShearX')

    augmentations.shear_x(1)
    assert aug_mock.called
    aug_mock.assert_called_with(pytest.approx(-0.03))


def test_shear_x_bbox(mocker):
    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.0
    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.BlendAlphaBoundingBoxes')
    aug_shear_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.ShearX')

    augmentations.shear_x_bbox(7)
    args, kwargs = aug_mock.call_args_list[0]
    assert tuple([None]) == args
    assert 'foreground' in kwargs
    aug_shear_mock.assert_called_with(pytest.approx(-0.21))


def test_shear_y(mocker):

    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.32
    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.ShearY')

    augmentations.shear_y(3)
    assert aug_mock.called
    aug_mock.assert_called_with(pytest.approx(-0.09))


def test_shear_y_bbox(mocker):

    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.5
    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.ShearY')

    augmentations.shear_y_bbox(2)
    assert aug_mock.called
    aug_mock.assert_called_with(pytest.approx(0.06))


def test_solarize(mocker):

    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.pillike.Solarize')

    augmentations.solarize(50)
    assert aug_mock.called
    aug_mock.assert_called_with(threshold=128)


def test_solarize_add():

    aug = augmentations.solarize_add(2)
    img_aug, bbs_aug = aug(np.zeros((2, 2)).astype('uint8'), [])
    exp = np.array([22]*4).reshape(2, 2)
    assert np.array_equal(exp, img_aug)

    aug = augmentations.solarize_add(5)
    img_aug, bbs_aug = aug(np.array([[22, 255], [0, 128]]).astype('uint8'), [])
    exp = np.array([[77, 255], [55, 128]])
    assert np.array_equal(exp, img_aug)


def test_translate_x(mocker):

    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.72
    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.geometric.TranslateX')

    augmentations.translate_x(8)
    assert aug_mock.called
    aug_mock.assert_called_with(px=200)


def test_translate_x_bbox(mocker):

    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.5
    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.BlendAlphaBoundingBoxes')
    aug_mock_translate = mocker.patch('bbaug.augmentations.augmentations.iaa.geometric.TranslateX')

    augmentations.translate_x_bbox(10)
    assert aug_mock.called
    args, kwargs = aug_mock.call_args_list[0]
    assert tuple([None]) == args
    assert 'foreground' in kwargs
    args, kwargs = aug_mock_translate.call_args_list[0]
    assert tuple() == args
    assert {'px': 120} == kwargs

    aug_mock.reset_mock()
    aug_mock_translate.reset_mock()
    augmentations.translate_x_bbox(3)
    assert aug_mock.called
    args, kwargs = aug_mock.call_args_list[0]
    assert tuple([None]) == args
    assert 'foreground' in kwargs
    args, kwargs = aug_mock_translate.call_args_list[0]
    assert tuple() == args
    assert {'px': 36} == kwargs


def test_translate_y(mocker):

    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.22
    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.geometric.TranslateY')

    augmentations.translate_y(7)
    assert aug_mock.called
    aug_mock.assert_called_with(px=-175)


def test_translate_y_bbox(mocker):

    numpy_random_mock = mocker.patch('bbaug.augmentations.augmentations.np.random.random')
    numpy_random_mock.return_value = 0.5
    aug_mock = mocker.patch('bbaug.augmentations.augmentations.iaa.BlendAlphaBoundingBoxes')
    aug_mock_translate = mocker.patch('bbaug.augmentations.augmentations.iaa.geometric.TranslateY')

    augmentations.translate_y_bbox(0)
    assert aug_mock.called
    args, kwargs = aug_mock.call_args_list[0]
    assert tuple([None]) == args
    assert 'foreground' in kwargs
    args, kwargs = aug_mock_translate.call_args_list[0]
    assert tuple() == args
    assert {'px': 0} == kwargs

    aug_mock.reset_mock()
    aug_mock_translate.reset_mock()
    augmentations.translate_y_bbox(1)
    assert aug_mock.called
    args, kwargs = aug_mock.call_args_list[0]
    assert tuple([None]) == args
    assert 'foreground' in kwargs
    args, kwargs = aug_mock_translate.call_args_list[0]
    assert tuple() == args
    assert {'px': 12} == kwargs
