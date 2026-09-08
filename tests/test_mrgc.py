"""Tests for the mRGC model."""

import numpy as np

from retinamodel.mrgc import MRGCModel


def test_model_constructs():
    """Check that the mRGC model can be created successfully."""
    model = MRGCModel()

    assert isinstance(model, MRGCModel)
    assert len(model.lut["wavelength_nm"]) == 63


def test_core_spectral_arrays():
    """Check that the main spectral arrays have the expected size and values."""
    model = MRGCModel()

    arrays = [
        model.log_ext_s,
        model.log_ext_m,
        model.log_ext_l,
        model.ext_s,
        model.ext_m,
        model.ext_l,
        model.ext_surround_lm,
        model.ext_surround_lms,
        model.ext_center_inhib_lm,
    ]

    for array in arrays:
        assert len(array) == 63
        assert np.isfinite(array).all()


def test_white_reference_outputs():
    """Check white-light outputs against the original notebook baseline."""
    model = MRGCModel()

    actual = np.array(
        [
            model.l_center_white,
            model.m_center_white,
            model.s_center_white,
            model.surround_white,
            model.lm_center_white,
            model.dog_l_white,
            model.dog_m_white,
            model.dog_h2_white,
            model.dog_s_white,
        ]
    )

    expected = np.array(
        [
            4537.338264696156,
            4537.338264696156,
            4537.338264696156,
            4537.338263078949,
            4537.338264696156,
            1.617207090021111e-06,
            1.617207090021111e-06,
            1.617207090021111e-06,
            0.0,
        ]
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-12,
        atol=1.0e-9,
    )


def test_yellow_reference_outputs():
    """Check yellow-light outputs against the original notebook baseline."""
    model = MRGCModel()

    actual = np.array(
        [
            model.l_center_yellow,
            model.m_center_yellow,
            model.surround_yellow,
            model.dog_l_yellow,
            model.dog_m_yellow,
            model.percent_dog_l_yellow,
            model.percent_dog_m_yellow,
        ]
    )

    expected = np.array(
        [
            4234.125064542216,
            3980.384448817143,
            4124.127673848555,
            109.99739069366115,
            -143.74322503141184,
            0.025978776965000634,
            -0.0361128998667775,
        ]
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-12,
        atol=1.0e-9,
    )