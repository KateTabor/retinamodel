"""Stimulus generation and mRGC export pipeline from Cells 11–14."""

import argparse
import colorsys
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

from retinamodel.mrgc import MRGCModel
from retinamodel.reporting import format_stimulus_report

GRAY_TOL = 1e-6

REC709_W = np.array([0.2126, 0.7152, 0.0722], dtype=float)

HUE_RGB = {
    "gray": np.array([1.0, 1.0, 1.0], dtype=float),
    "red": np.array([1.0, 0.0, 0.0], dtype=float),
    "green": np.array([0.0, 1.0, 0.0], dtype=float),
    "blue": np.array([0.0, 0.0, 1.0], dtype=float),
    "yellow": np.array([1.0, 1.0, 0.0], dtype=float),
}

BG_HUES = ("gray", "red", "green", "blue", "yellow")

CHROMATIC_HUES = ("red", "green", "blue", "yellow")

TRUE_LABELS = ("gray_l", "gray_d", "red", "green", "blue", "yellow")

RESPONSE_KEYS = (
    "l_on",
    "l_off",
    "m_on",
    "m_off",
    "l_h2_on",
    "m_h2_on",
    "l_h2_off",
    "m_h2_off",
    "s_on",
)

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

    def __post_init__(self):
        """Validate the configuration values after initialization."""
        self.validate()

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

        return cls(**values)


    def validate(self):
        """Check that stimulus configuration values are valid."""

        # Positive integer settings
        count_fields = {
            "image_size_px": self.image_size_px,
            "obj_gray_n": self.obj_gray_n,
            "obj_chrom_n_int": self.obj_chrom_n_int,
            "obj_chrom_n_sat": self.obj_chrom_n_sat,
            "bg_gray_n_int": self.bg_gray_n_int,
            "bg_chrom_n_int": self.bg_chrom_n_int,
            "bg_chrom_n_sat": self.bg_chrom_n_sat,
        }

        for name, value in count_fields.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        # Image/object settings
        if not 0.0 < self.obj_diam_ratio <= 1.0:
            raise ValueError("obj_diam_ratio must be in (0, 1].")

        if not 0.0 <= self.gray_light_threshold <= 1.0:
            raise ValueError("gray_light_threshold must be in [0, 1].")

        # Gray object range
        if not 0.0 <= self.obj_gray_min <= self.obj_gray_max <= 1.0:
            raise ValueError(
                "obj_gray_min and obj_gray_max must satisfy "
                "0 <= min <= max <= 1."
            )

        # Chromatic object intensity range
        if not 0.0 <= self.obj_chrom_int_min <= self.obj_chrom_int_max <= 1.0:
            raise ValueError(
                "obj_chrom_int_min and obj_chrom_int_max must satisfy "
                "0 <= min <= max <= 1."
            )

        # Chromatic object saturation range
        if not 0.0 <= self.obj_chrom_sat_min <= self.obj_chrom_sat_max <= 1.0:
            raise ValueError(
                "obj_chrom_sat_min and obj_chrom_sat_max must satisfy "
                "0 <= min <= max <= 1."
            )

        # Background delta ranges
        if self.bg_gray_delta_int_min > self.bg_gray_delta_int_max:
            raise ValueError(
                "bg_gray_delta_int_min must be <= bg_gray_delta_int_max."
            )

        if self.bg_chrom_delta_int_min > self.bg_chrom_delta_int_max:
            raise ValueError(
                "bg_chrom_delta_int_min must be <= bg_chrom_delta_int_max."
            )

        if self.bg_chrom_delta_sat_min > self.bg_chrom_delta_sat_max:
            raise ValueError(
                "bg_chrom_delta_sat_min must be <= bg_chrom_delta_sat_max."
            )

        # Boolean option
        if type(self.include_bg_chrom_sat_1) is not bool:
            raise ValueError("include_bg_chrom_sat_1 must be true or false.")


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


def rec709_luma(rgb):
    """Return Rec.709 luminance for an RGB color."""
    rgb = np.asarray(rgb, dtype=float)
    return float(np.dot(rgb, REC709_W))


def rgb_from_hue_int_sat(hue_rgb, intensity, saturation=1.0):
    """Build an RGB color from hue, intensity, and saturation."""
    hue_rgb = np.asarray(hue_rgb, dtype=float)
    intensity = float(intensity)
    saturation = float(saturation)

    if not 0.0 <= intensity <= 1.0:
        raise ValueError("intensity must be in [0, 1].")

    if not 0.0 <= saturation <= 1.0:
        raise ValueError("saturation must be in [0, 1].")

    max_channel = float(np.max(hue_rgb))

    if max_channel <= 0.0:
        raise ValueError("hue_rgb must have a positive maximum channel.")

    hue_unit = hue_rgb / max_channel

    rgb_unit = (
        saturation * hue_unit
        + (1.0 - saturation) * np.ones(3)
    )

    return np.clip(intensity * rgb_unit, 0.0, 1.0)


def rgb_from_label(label, intensity, saturation=1.0):
    """Build an RGB color from a named hue."""
    if label not in HUE_RGB:
        raise ValueError(
            f"Unknown label '{label}'. Use {list(HUE_RGB)}."
        )

    return rgb_from_hue_int_sat(
        HUE_RGB[label],
        intensity,
        saturation,
    )


def true_label(obj_hue_label, obj_rgb, gray_light_threshold):
    """Return the experimental label for an object's hue and RGB value."""
    if obj_hue_label != "gray":
        return obj_hue_label

    obj_rgb = np.asarray(obj_rgb, dtype=float)

    gray_difference = np.max(
        np.abs(obj_rgb - float(np.mean(obj_rgb)))
    )

    if float(gray_difference) > GRAY_TOL:
        raise ValueError(
            "obj_hue_label='gray' but obj_rgb is not grayscale."
        )

    obj_intensity = rec709_luma(obj_rgb)

    if obj_intensity >= gray_light_threshold:
        return "gray_l"

    return "gray_d"


def rgb_to_hsv01(rgb):
    """Convert an RGB color in [0, 1] to HSV in [0, 1]."""
    r, g, b = [
        float(value)
        for value in np.clip(rgb, 0.0, 1.0)
    ]

    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    return np.array([h, s, v], dtype=float)

def hsv_delta(hsv_obj, hsv_bg):
    """Return object-minus-background HSV differences with wrapped hue."""
    hsv_obj = np.asarray(hsv_obj, dtype=float)
    hsv_bg = np.asarray(hsv_bg, dtype=float)

    delta_h = ((hsv_obj[0] - hsv_bg[0] + 0.5) % 1.0) - 0.5

    return np.array([
        delta_h,
        hsv_obj[1] - hsv_bg[1],
        hsv_obj[2] - hsv_bg[2],
    ], dtype=float)

def make_circle_stimulus(bg_rgb, obj_rgb, size_px, obj_diam_ratio):
    """Create a square RGB image with a centered circular object."""
    size_px = int(size_px)
    if size_px <= 0:
        raise ValueError("size_px must be positive.")

    obj_diam_ratio = float(obj_diam_ratio)
    if not 0.0 < obj_diam_ratio <= 1.0:
        raise ValueError("obj_diam_ratio must be in (0, 1].")

    bg_rgb = np.asarray(bg_rgb, dtype=float)
    obj_rgb = np.asarray(obj_rgb, dtype=float)

    image = np.broadcast_to(
        bg_rgb,
        (size_px, size_px, 3),
    ).copy()

    center_y = center_x = size_px // 2
    radius = 0.5 * obj_diam_ratio * size_px

    yy, xx = np.ogrid[:size_px, :size_px]

    circle_mask = (
        (yy - center_y) ** 2
        + (xx - center_x) ** 2
        <= radius ** 2
    )

    image[circle_mask] = obj_rgb

    return image

def compute_rgb_to_quanta_weights(model):
    """Compute RGB-to-quanta weights from a constructed mRGC model."""
    primaries_rgb = np.stack(
        [
            model.lut["primary_r"],
            model.lut["primary_g"],
            model.lut["primary_b"],
        ],
        axis=1,
    )

    return {
        "L": model.ext_l @ primaries_rgb,
        "M": model.ext_m @ primaries_rgb,
        "S": model.ext_s @ primaries_rgb,
        "LM_sur": model.ext_surround_lm @ primaries_rgb,
        "LMS_sur": model.ext_surround_lms @ primaries_rgb,
        "LM_inhib_center": model.ext_center_inhib_lm @ primaries_rgb,
    }

def rgb_image_to_drive(img_rgb, rgb2q):
    """Convert an RGB image into a cone or surround drive image."""
    img_rgb = np.asarray(img_rgb, dtype=float)
    rgb2q = np.asarray(rgb2q, dtype=float)

    if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
        raise ValueError("img_rgb must have shape (H, W, 3).")

    if rgb2q.shape != (3,):
        raise ValueError("rgb2q must contain exactly 3 RGB weights.")

    return (
        img_rgb[..., 0] * rgb2q[0]
        + img_rgb[..., 1] * rgb2q[1]
        + img_rgb[..., 2] * rgb2q[2]
    )

def run_mrgc_model(img_rgb, model, rgb2q_weights):
    """Run the mRGC model on one RGB stimulus image."""
    drive_l = rgb_image_to_drive(img_rgb, rgb2q_weights["L"])
    drive_m = rgb_image_to_drive(img_rgb, rgb2q_weights["M"])
    drive_s = rgb_image_to_drive(img_rgb, rgb2q_weights["S"])

    drive_lm_sur = rgb_image_to_drive(
        img_rgb,
        rgb2q_weights["LM_sur"],
    )
    drive_lms_sur = rgb_image_to_drive(
        img_rgb,
        rgb2q_weights["LMS_sur"],
    )
    drive_lm_inhib = rgb_image_to_drive(
        img_rgb,
        rgb2q_weights["LM_inhib_center"],
    )

    center_l = fft_2d(drive_l, model.gaussian_center)
    center_m = fft_2d(drive_m, model.gaussian_center)
    center_s = fft_2d(drive_s, model.gaussian_center)

    surround_lm = fft_2d(
        drive_lm_sur,
        model.gaussian_surround,
    )
    surround_lms = fft_2d(
        drive_lms_sur,
        model.gaussian_surround,
    )
    inhib_center_lm = fft_2d(
        drive_lm_inhib,
        model.gaussian_center,
    )

    l_on = model.scalar_l * center_l - surround_lm
    m_on = model.scalar_m * center_m - surround_lm

    l_off = -l_on
    m_off = -m_on

    l_h2_on = (
        model.scalar_l * center_l
        + model.scalar_lms * surround_lms
    ) - (
        model.scalar_s * center_s
        + surround_lm
    )

    m_h2_on = (
        model.scalar_m * center_m
        + model.scalar_lms * surround_lms
    ) - (
        model.scalar_s * center_s
        + surround_lm
    )

    l_h2_off = -l_h2_on
    m_h2_off = -m_h2_on

    s_on = (
        model.scalar_s_inhib_lm * center_s
        - inhib_center_lm
    )

    responses = {
        "l_on": l_on,
        "l_off": l_off,
        "m_on": m_on,
        "m_off": m_off,
        "l_h2_on": l_h2_on,
        "m_h2_on": m_h2_on,
        "l_h2_off": l_h2_off,
        "m_h2_off": m_h2_off,
        "s_on": s_on,
    }

    if tuple(responses.keys()) != RESPONSE_KEYS:
        raise RuntimeError(
            "mRGC response key order does not match RESPONSE_KEYS."
        )

    return responses


def truncated_delta_grid(center_value, delta_min, delta_max, n, low=0.0, high=1.0):
    """Create evenly spaced deltas while keeping values within bounds."""
    center_value = float(center_value)

    dmin = max(float(delta_min), low - center_value)
    dmax = min(float(delta_max), high - center_value)

    deltas = np.linspace(
        dmin,
        dmax,
        int(n),
        endpoint=True,
    )

    values = center_value + deltas

    return values.astype(float), deltas.astype(float)


def rgb_key(rgb):
    """Convert an RGB color into a stable tuple for duplicate checking."""
    rgb = np.asarray(rgb, dtype=float)
    return tuple(np.round(rgb, 8).tolist())


def build_center_specs(config):
    """Build all gray and chromatic center-object settings."""
    gray_int_levels = np.linspace(config.obj_gray_min, config.obj_gray_max, config.obj_gray_n)
    chrom_int_levels = np.linspace(config.obj_chrom_int_min, config.obj_chrom_int_max, config.obj_chrom_n_int)
    chrom_sat_levels = np.linspace(config.obj_chrom_sat_min, config.obj_chrom_sat_max, config.obj_chrom_n_sat)

    center_specs = []

    for intensity in gray_int_levels:
        intensity = float(intensity)
        obj_rgb = np.array([intensity, intensity, intensity], dtype=float)

        center_specs.append({
            "obj_hue": "gray",
            "obj_int": intensity,
            "obj_sat": 0.0,
            "obj_rgb": obj_rgb,
        })

    for hue in CHROMATIC_HUES:
        for intensity in chrom_int_levels:
            for saturation in chrom_sat_levels:
                intensity = float(intensity)
                saturation = float(saturation)
                obj_rgb = rgb_from_label(hue, intensity, saturation)

                center_specs.append({
                    "obj_hue": hue,
                    "obj_int": intensity,
                    "obj_sat": saturation,
                    "obj_rgb": obj_rgb,
                })

    return center_specs


def build_background_specs(center_spec, config):
    """Build all background settings for one center object."""
    obj_int = float(center_spec["obj_int"])
    obj_sat = float(center_spec["obj_sat"])
    background_specs = []

    for bg_hue in BG_HUES:
        if bg_hue == "gray":
            bg_ints, delta_ints = truncated_delta_grid(
                obj_int, config.bg_gray_delta_int_min,
                config.bg_gray_delta_int_max, config.bg_gray_n_int
            )

            for bg_int, delta_int in zip(bg_ints, delta_ints):
                bg_int = float(bg_int)
                bg_sat = 0.0

                background_specs.append({
                    "bg_hue": "gray",
                    "bg_int": bg_int,
                    "bg_sat": bg_sat,
                    "delta_int": float(delta_int),
                    "delta_sat": float(bg_sat - obj_sat),
                    "bg_rgb": np.array([bg_int, bg_int, bg_int], dtype=float),
                })

        else:
            bg_ints, delta_ints = truncated_delta_grid(
                obj_int, config.bg_chrom_delta_int_min,
                config.bg_chrom_delta_int_max, config.bg_chrom_n_int
            )
            bg_sats, _ = truncated_delta_grid(
                obj_sat, config.bg_chrom_delta_sat_min,
                config.bg_chrom_delta_sat_max, config.bg_chrom_n_sat
            )

            if config.include_bg_chrom_sat_1:
                bg_sats = np.unique(
                    np.concatenate([bg_sats, np.array([1.0], dtype=float)])
                )

            delta_sats = bg_sats - obj_sat

            for bg_int, delta_int in zip(bg_ints, delta_ints):
                for bg_sat, delta_sat in zip(bg_sats, delta_sats):
                    bg_int = float(bg_int)
                    bg_sat = float(bg_sat)

                    background_specs.append({
                        "bg_hue": bg_hue,
                        "bg_int": bg_int,
                        "bg_sat": bg_sat,
                        "delta_int": float(delta_int),
                        "delta_sat": float(delta_sat),
                        "bg_rgb": rgb_from_label(bg_hue, bg_int, bg_sat),
                    })

    return background_specs


def build_stimulus_candidates(config):
    """Build all stimulus candidates and remove duplicate RGB pairs."""
    center_specs = build_center_specs(config)

    unique_candidates = []
    seen_rgb_pairs = set()
    raw_bg_counts_per_center = []

    raw_label_counts = {label: 0 for label in TRUE_LABELS}
    unique_label_counts = {label: 0 for label in TRUE_LABELS}
    raw_candidate_count = 0

    for center_spec in center_specs:
        obj_hue = center_spec["obj_hue"]
        obj_rgb = np.asarray(center_spec["obj_rgb"], dtype=float)

        obj_label = true_label(
            obj_hue, obj_rgb, config.gray_light_threshold
        )

        background_specs = build_background_specs(center_spec, config)
        raw_bg_counts_per_center.append(len(background_specs))

        for background_spec in background_specs:
            raw_candidate_count += 1
            raw_label_counts[obj_label] += 1

            bg_rgb = np.asarray(background_spec["bg_rgb"], dtype=float)
            rgb_pair = (rgb_key(bg_rgb), rgb_key(obj_rgb))

            if rgb_pair in seen_rgb_pairs:
                continue

            seen_rgb_pairs.add(rgb_pair)
            unique_label_counts[obj_label] += 1

            candidate = {
                **center_spec,
                **background_spec,
                "true_label": obj_label,
            }
            unique_candidates.append(candidate)

    raw_bg_counts_per_center = np.asarray(
        raw_bg_counts_per_center, dtype=int
    )

    summary = {
        "n_centers": len(center_specs),
        "n_gray_centers": sum(
            spec["obj_hue"] == "gray" for spec in center_specs
        ),
        "n_chrom_centers": sum(
            spec["obj_hue"] != "gray" for spec in center_specs
        ),
        "bg_per_center_min": int(raw_bg_counts_per_center.min()),
        "bg_per_center_max": int(raw_bg_counts_per_center.max()),
        "n_total_raw": raw_candidate_count,
        "n_total_unique": len(unique_candidates),
        "n_redundant": raw_candidate_count - len(unique_candidates),
        "raw_label_counts": raw_label_counts,
        "unique_label_counts": unique_label_counts,
    }

    return unique_candidates, summary


def build_metadata_row(candidate, stim_id):
    """Build one metadata row for a unique stimulus."""
    bg_rgb = np.asarray(candidate["bg_rgb"], dtype=float)
    obj_rgb = np.asarray(candidate["obj_rgb"], dtype=float)

    hsv_bg = rgb_to_hsv01(bg_rgb)
    hsv_obj = rgb_to_hsv01(obj_rgb)
    delta_hsv = hsv_delta(hsv_obj, hsv_bg)

    return {
        "stim_ID": int(stim_id),
        "true_label": candidate["true_label"],
        "bg_hue": candidate["bg_hue"],
        "bg_r": float(bg_rgb[0]),
        "bg_g": float(bg_rgb[1]),
        "bg_b": float(bg_rgb[2]),
        "obj_r": float(obj_rgb[0]),
        "obj_g": float(obj_rgb[1]),
        "obj_b": float(obj_rgb[2]),
        "bg_int": float(candidate["bg_int"]),
        "obj_int": float(candidate["obj_int"]),
        "delta_int": float(candidate["delta_int"]),
        "bg_sat": float(candidate["bg_sat"]),
        "obj_sat": float(candidate["obj_sat"]),
        "delta_sat": float(candidate["delta_sat"]),
        "hsv_bg": f"[{hsv_bg[0]:.6f},{hsv_bg[1]:.6f},{hsv_bg[2]:.6f}]",
        "hsv_obj": f"[{hsv_obj[0]:.6f},{hsv_obj[1]:.6f},{hsv_obj[2]:.6f}]",
        "delta_hsv": f"[{delta_hsv[0]:.6f},{delta_hsv[1]:.6f},{delta_hsv[2]:.6f}]",
    }


def build_metadata_table(candidates):
    """Build the metadata table for all unique stimuli."""
    rows = [
        build_metadata_row(candidate, stim_id)
        for stim_id, candidate in enumerate(candidates)
    ]

    return pd.DataFrame(rows)


def create_export_dir(base_dir="data/interm", timestamp=None):
    """Create a new timestamped directory for one export run."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    export_dir = base_dir / f"retina_dataset_{timestamp}"
    export_dir.mkdir(exist_ok=False)

    return export_dir


def create_zarr_store(export_dir, config, summary):
    """Create the Zarr store for stimulus images and mRGC responses."""
    export_dir = Path(export_dir)
    zarr_path = export_dir / "dataset.zarr"

    n_stimuli = int(summary["n_total_unique"])
    image_size = int(config.image_size_px)
    n_responses = len(RESPONSE_KEYS)
    chunk_hw = min(256, image_size)

    compressor = zarr.codecs.BloscCodec(
        cname="zstd", clevel=3, shuffle="bitshuffle"
    )

    root = zarr.open_group(zarr_path, mode="w")

    root.attrs.update({
        "dataset_name": export_dir.name,
        "IMAGE_SIZE_PX": image_size,
        "OBJ_DIAM_RATIO": float(config.obj_diam_ratio),
        "GRAY_LIGHT_THRESHOLD": float(config.gray_light_threshold),
        "RESPONSE_KEYS": list(RESPONSE_KEYS),
        "BG_HUES": list(BG_HUES),
        "INCLUDE_BG_CHROM_SAT_1": bool(config.include_bg_chrom_sat_1),
        "N_TOTAL_RAW": int(summary["n_total_raw"]),
        "N_TOTAL_UNIQUE": n_stimuli,
        "N_REDUNDANT_REMOVED": int(summary["n_redundant"]),
        "STIM_SETTINGS": asdict(config),
    })

    imgs_z = root.create_array(
        name="imgs",
        shape=(n_stimuli, image_size, image_size, 3),
        chunks=(1, chunk_hw, chunk_hw, 3),
        dtype="f4",
        compressors=compressor,
    )

    outs_z = root.create_array(
        name="outs",
        shape=(n_stimuli, n_responses, image_size, image_size),
        chunks=(1, 1, chunk_hw, chunk_hw),
        dtype="f4",
        compressors=compressor,
    )

    return root, imgs_z, outs_z, zarr_path


def process_stimulus_candidate(candidate, config, model, rgb2q_weights):
    """Create one stimulus image and its ordered mRGC response maps."""
    image = make_circle_stimulus(
        candidate["bg_rgb"], candidate["obj_rgb"],
        config.image_size_px, config.obj_diam_ratio
    )

    responses = run_mrgc_model(image, model, rgb2q_weights)

    response_stack = np.stack(
        [responses[key] for key in RESPONSE_KEYS],
        axis=0,
    )

    return image, response_stack


def write_stimulus_to_zarr(stim_id, image, response_stack, imgs_z, outs_z):
    """Write one stimulus image and its response maps to Zarr."""
    imgs_z[stim_id] = image
    outs_z[stim_id] = response_stack


def validate_export_structure(zarr_path, metadata_path, config, summary):
    """Check that the saved Zarr arrays and metadata are structurally aligned."""
    root = zarr.open_group(zarr_path, mode="r")
    metadata = pd.read_csv(metadata_path)

    expected_n = int(summary["n_total_unique"])
    image_size = int(config.image_size_px)

    if len(metadata) != expected_n:
        raise RuntimeError("Metadata row count does not match the expected number of stimuli.")

    if root["imgs"].shape != (expected_n, image_size, image_size, 3):
        raise RuntimeError("Saved stimulus image array has an unexpected shape.")

    if root["outs"].shape != (expected_n, len(RESPONSE_KEYS), image_size, image_size):
        raise RuntimeError("Saved mRGC response array has an unexpected shape.")

    if metadata["stim_ID"].tolist() != list(range(expected_n)):
        raise RuntimeError("Metadata stimulus IDs are not aligned with Zarr array indices.")

    if tuple(root.attrs["RESPONSE_KEYS"]) != RESPONSE_KEYS:
        raise RuntimeError("Saved response key order does not match RESPONSE_KEYS.")


def export_retina_dataset(config, base_dir="data/interm", timestamp=None):
    """Generate and save one complete retina dataset."""
    candidates, summary = build_stimulus_candidates(config)
    metadata = build_metadata_table(candidates)

    export_dir = create_export_dir(base_dir, timestamp)
    _, imgs_z, outs_z, zarr_path = create_zarr_store(
        export_dir, config, summary
    )

    model = MRGCModel()
    rgb2q_weights = compute_rgb_to_quanta_weights(model)

    for stim_id, candidate in enumerate(tqdm(candidates, desc="Processing stimuli")):
        image, response_stack = process_stimulus_candidate(
            candidate, config, model, rgb2q_weights
        )
        write_stimulus_to_zarr(
            stim_id, image, response_stack, imgs_z, outs_z
        )

    metadata_path = export_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    validate_export_structure(zarr_path, metadata_path, config, summary)

    return {
        "export_dir": export_dir,
        "zarr_path": zarr_path,
        "metadata_path": metadata_path,
        "summary": summary,
        "metadata": metadata,
    }


def main():
    """Generate a retina dataset from a stimulus configuration file."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic stimuli and mRGC response datasets."
    )
    parser.add_argument(
        "--config",
        default="config/stimulus.toml",
        help="Path to the stimulus TOML configuration file.",
    )
    args = parser.parse_args()

    config = StimulusConfig.from_toml(args.config)
    result = export_retina_dataset(config)

    print()
    print(format_stimulus_report(result))