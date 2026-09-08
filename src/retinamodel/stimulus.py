"""Stimulus generation and mRGC export pipeline from Cells 11–14."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np

@dataclass
class StimulusConfig:
    """User-adjustable settings for stimulus generation."""

    image_size_px: int
    obj_diam_ratio: float
    gray_light_threshold: float

    obj_gray_n: int
    obj_gray_min: float
    obj_gray_max: float

    obj_chrom_n_int: int
    obj_chrom_int_min: float
    obj_chrom_int_max: float

    obj_chrom_n_sat: int
    obj_chrom_sat_min: float
    obj_chrom_sat_max: float

    bg_gray_n_int: int
    bg_gray_delta_int_min: float
    bg_gray_delta_int_max: float

    bg_chrom_n_int: int
    bg_chrom_delta_int_min: float
    bg_chrom_delta_int_max: float

    bg_chrom_n_sat: int
    bg_chrom_delta_sat_min: float
    bg_chrom_delta_sat_max: float

    include_bg_chrom_sat_1: bool

    @classmethod
    def from_toml(cls, path):
        """Create a stimulus configuration from a TOML file."""
        path = Path(path)

        with path.open("rb") as file:
            values = tomllib.load(file)

        values = {
            key.lower(): value
            for key, value in values.items()
        }

        config = cls(**values)
        config.validate()
        return config
    
    def validate(self):
        """Check that stimulus configuration values are valid."""

        if self.image_size_px <= 0:
            raise ValueError("image_size_px must be greater than 0.")

        if not 0.0 < self.obj_diam_ratio <= 1.0:
            raise ValueError("obj_diam_ratio must be in (0, 1].")

        if not 0.0 <= self.gray_light_threshold <= 1.0:
            raise ValueError("gray_light_threshold must be in [0, 1].")

        if self.obj_gray_n <= 0:
            raise ValueError("obj_gray_n must be greater than 0.")

        if not 0.0 <= self.obj_gray_min <= self.obj_gray_max <= 1.0:
            raise ValueError(
                "obj_gray_min and obj_gray_max must satisfy "
                "0 <= min <= max <= 1."
            )

        if self.obj_chrom_n_int <= 0:
            raise ValueError("obj_chrom_n_int must be greater than 0.")

        if not 0.0 <= self.obj_chrom_int_min <= self.obj_chrom_int_max <= 1.0:
            raise ValueError(
                "obj_chrom_int_min and obj_chrom_int_max must satisfy "
                "0 <= min <= max <= 1."
            )

        if self.obj_chrom_n_sat <= 0:
            raise ValueError("obj_chrom_n_sat must be greater than 0.")

        if not 0.0 <= self.obj_chrom_sat_min <= self.obj_chrom_sat_max <= 1.0:
            raise ValueError(
                "obj_chrom_sat_min and obj_chrom_sat_max must satisfy "
                "0 <= min <= max <= 1."
            )

        if self.bg_gray_n_int <= 0:
            raise ValueError("bg_gray_n_int must be greater than 0.")

        if self.bg_chrom_n_int <= 0:
            raise ValueError("bg_chrom_n_int must be greater than 0.")

        if self.bg_chrom_n_sat <= 0:
            raise ValueError("bg_chrom_n_sat must be greater than 0.")

def fft_2d(image, kernel, pad_mode="reflect"):
    """Convolve a 2D image with a 2D kernel using FFT convolution."""
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    image_padded = np.pad(
        image,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode=pad_mode,
    )

    padded_height, padded_width = image_padded.shape
    output_shape = (
        padded_height + kernel_height - 1,
        padded_width + kernel_width - 1,
    )

    convolved = np.fft.ifft2(
        np.fft.fft2(image_padded, output_shape)
        * np.fft.fft2(kernel, output_shape)
    ).real

    row_start = kernel_height // 2
    col_start = kernel_width // 2

    convolved_padded = convolved[
        row_start:row_start + padded_height,
        col_start:col_start + padded_width,
    ]

    return convolved_padded[
        pad_y:pad_y + image_height,
        pad_x:pad_x + image_width,
    ]