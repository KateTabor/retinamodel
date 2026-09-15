"""Human-audit configuration and dataset validation for Module 3."""

from datetime import datetime, timezone
import json
from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import zarr

from retinamodel.stimulus import BG_HUES, TRUE_LABELS, rgb_key, rgb_to_hsv01


REQUIRED_METADATA_COLUMNS = (
    "stim_ID",
    "true_label",
    "bg_hue",
    "bg_r",
    "bg_g",
    "bg_b",
    "obj_r",
    "obj_g",
    "obj_b",
    "bg_int",
    "obj_int",
    "delta_int",
    "bg_sat",
    "obj_sat",
    "delta_sat",
)

REQUIRED_DATASET_ATTRS = (
    "IMAGE_SIZE_PX",
    "OBJ_DIAM_RATIO",
    "GRAY_LIGHT_THRESHOLD",
    "STIM_SETTINGS",
)

GALLERY_STIM_SETTING_KEYS = (
    "obj_gray_min",
    "obj_gray_max",
    "obj_chrom_int_min",
    "obj_chrom_int_max",
    "obj_chrom_sat_min",
    "obj_chrom_sat_max",
)

HUMAN_LABEL_REQUIRED_COLUMNS = (
    "user_label",
    "labeled_at",
    "obj_r",
    "obj_g",
    "obj_b",
    "bg_r",
    "bg_g",
    "bg_b",
    "obj_diam_ratio",
)

HUMAN_LABEL_LOOKUP_COLUMNS = (
    "obj_r",
    "obj_g",
    "obj_b",
    "bg_r",
    "bg_g",
    "bg_b",
    "obj_diam_ratio",
    "human_label",
    "n_human_labels",
    "label_resolution",
    "last_labeled_at",
)

SESSION_REQUIRED_COLUMNS = (
    *REQUIRED_METADATA_COLUMNS,
    "obj_diam_ratio",
    "user_label",
    "label_source",
    "matched_obj_diam_ratio",
    "existing_n_human_labels",
    "existing_label_resolution",
    "existing_last_labeled_at",
    "audit_order",
    "labeled_at",
)

RAW_HUMAN_LABEL_COLUMNS = (
    "label_id",
    "labeled_at",
    "user_label",
    "obj_r",
    "obj_g",
    "obj_b",
    "bg_r",
    "bg_g",
    "bg_b",
    "obj_diam_ratio",
    "image_size_px",
    "source_dataset",
    "source_stim_ID",
    "pipeline_label",
    "bg_hue",
    "obj_int",
    "obj_sat",
    "bg_int",
    "bg_sat",
    "delta_int",
    "delta_sat",
)

RATIO_ATOL = 1e-9


@dataclass
class AuditConfig:
    """User-adjustable settings for selecting stimuli for human audit."""

    labels: list[str]
    bg_hues: list[str]
    random_seed: int

    filter_delta_int: bool
    delta_int_min: float
    delta_int_max: float

    filter_delta_sat: bool
    delta_sat_min: float
    delta_sat_max: float

    def __post_init__(self):
        """Validate the audit configuration after initialization."""
        self.validate()

    def validate(self):
        """Check that audit configuration values are valid."""

        if not isinstance(self.labels, list) or not self.labels:
            raise ValueError("audit.labels must be a non-empty list.")

        if len(self.labels) != len(set(self.labels)):
            raise ValueError("audit.labels must not contain duplicates.")

        unknown_labels = set(self.labels) - set(TRUE_LABELS)
        if unknown_labels:
            raise ValueError(
                f"Unknown audit label(s): {sorted(unknown_labels)}. "
                f"Use only {list(TRUE_LABELS)}."
            )

        if not isinstance(self.bg_hues, list) or not self.bg_hues:
            raise ValueError("audit.bg_hues must be a non-empty list.")

        if len(self.bg_hues) != len(set(self.bg_hues)):
            raise ValueError("audit.bg_hues must not contain duplicates.")

        unknown_bg_hues = set(self.bg_hues) - set(BG_HUES)
        if unknown_bg_hues:
            raise ValueError(
                f"Unknown background hue(s): {sorted(unknown_bg_hues)}. "
                f"Use only {list(BG_HUES)}."
            )

        if type(self.random_seed) is not int:
            raise ValueError("audit.random_seed must be an integer.")

        if type(self.filter_delta_int) is not bool:
            raise ValueError("audit.filter_delta_int must be true or false.")

        if type(self.filter_delta_sat) is not bool:
            raise ValueError("audit.filter_delta_sat must be true or false.")

        _validate_delta_range(
            "delta_int",
            self.delta_int_min,
            self.delta_int_max,
        )

        _validate_delta_range(
            "delta_sat",
            self.delta_sat_min,
            self.delta_sat_max,
        )


@dataclass
class GalleryConfig:
    """User-adjustable settings for the stimulus gallery."""

    target_label: str
    n_cols: int
    save_png: bool
    dpi: int

    def __post_init__(self):
        """Validate the gallery configuration after initialization."""
        self.validate()

    def validate(self):
        """Check that gallery configuration values are valid."""

        if self.target_label not in TRUE_LABELS:
            raise ValueError(
                f"gallery.target_label must be one of {list(TRUE_LABELS)}."
            )

        if type(self.n_cols) is not int or self.n_cols <= 0:
            raise ValueError("gallery.n_cols must be a positive integer.")

        if type(self.save_png) is not bool:
            raise ValueError("gallery.save_png must be true or false.")

        if type(self.dpi) is not int or self.dpi <= 0:
            raise ValueError("gallery.dpi must be a positive integer.")


@dataclass
class AuditDataset:
    """Validated stimulus dataset used by the audit and gallery."""

    dataset_dir: Path
    zarr_path: Path
    metadata_path: Path

    root: object
    imgs: object
    metadata: pd.DataFrame

    image_size_px: int
    obj_diam_ratio: float
    gray_light_threshold: float
    stim_settings: dict


@dataclass
class LabelCoverage:
    """Coverage of selected stimuli by existing human labels."""

    n_selected: int
    n_same_ratio: int
    n_different_ratio: int

    @property
    def n_any_match(self):
        """Return stimuli with either same- or different-ratio matches."""
        return self.n_same_ratio + self.n_different_ratio

    @property
    def n_no_match(self):
        """Return selected stimuli with no existing RGB match."""
        return self.n_selected - self.n_any_match


@dataclass
class AuditSession:
    """One active or completed human-audit working session."""

    session_dir: Path
    table_path: Path
    state_path: Path
    table: pd.DataFrame
    state: dict


def _validate_delta_range(name, minimum, maximum):
    """Validate one signed delta range."""

    try:
        minimum = float(minimum)
        maximum = float(maximum)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} limits must be numeric.") from error

    if not np.isfinite([minimum, maximum]).all():
        raise ValueError(f"{name} limits must be finite.")

    if not -1.0 <= minimum <= maximum <= 1.0:
        raise ValueError(
            f"{name} limits must satisfy -1 <= min <= max <= 1."
        )


def load_audit_config(path="config/audit.toml"):
    """Load and validate the audit and gallery TOML configuration."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Audit configuration not found: {path}")

    with path.open("rb") as file:
        values = tomllib.load(file)

    if "audit" not in values:
        raise ValueError("audit.toml must contain an [audit] section.")

    if "gallery" not in values:
        raise ValueError("audit.toml must contain a [gallery] section.")

    audit_values = {
        key.lower(): value
        for key, value in values["audit"].items()
    }

    gallery_values = {
        key.lower(): value
        for key, value in values["gallery"].items()
    }

    audit_config = AuditConfig(**audit_values)
    gallery_config = GalleryConfig(**gallery_values)

    return audit_config, gallery_config


def load_audit_dataset(dataset_dir):
    """Load and validate a Module 2 stimulus dataset."""

    dataset_dir = Path(dataset_dir)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    zarr_path = dataset_dir / "dataset.zarr"
    metadata_path = dataset_dir / "metadata.csv"

    if not zarr_path.exists():
        raise FileNotFoundError(
            f"Dataset Zarr store not found: {zarr_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Dataset metadata CSV not found: {metadata_path}"
        )

    root = zarr.open_group(str(zarr_path), mode="r")

    if "imgs" not in root:
        raise ValueError("dataset.zarr must contain an 'imgs' array.")

    imgs = root["imgs"]

    if len(imgs.shape) != 4 or imgs.shape[-1] != 3:
        raise ValueError(
            "The Zarr 'imgs' array must have shape (N, H, W, 3)."
        )

    if imgs.shape[0] == 0:
        raise ValueError("The dataset contains no stimulus images.")

    metadata = pd.read_csv(metadata_path)

    missing_columns = [
        column
        for column in REQUIRED_METADATA_COLUMNS
        if column not in metadata.columns
    ]

    if missing_columns:
        raise ValueError(
            "metadata.csv is missing required column(s): "
            + ", ".join(missing_columns)
        )

    if len(metadata) != imgs.shape[0]:
        raise ValueError(
            "metadata.csv row count does not match the number of "
            "images in dataset.zarr."
        )

    _validate_stimulus_ids(metadata, imgs.shape[0])
    _validate_metadata_values(metadata)

    missing_attrs = [
        name
        for name in REQUIRED_DATASET_ATTRS
        if name not in root.attrs
    ]

    if missing_attrs:
        raise ValueError(
            "dataset.zarr is missing required attribute(s): "
            + ", ".join(missing_attrs)
        )

    image_size_px = root.attrs["IMAGE_SIZE_PX"]
    obj_diam_ratio = root.attrs["OBJ_DIAM_RATIO"]
    gray_light_threshold = root.attrs["GRAY_LIGHT_THRESHOLD"]
    stim_settings_raw = root.attrs["STIM_SETTINGS"]

    if not isinstance(stim_settings_raw, dict):
        raise ValueError(
            "Zarr attribute STIM_SETTINGS must contain a dictionary."
        )

    stim_settings = {
        str(key).lower(): value
        for key, value in stim_settings_raw.items()
    }

    if type(image_size_px) is not int or image_size_px <= 0:
        raise ValueError(
            "Zarr attribute IMAGE_SIZE_PX must be a positive integer."
        )

    if imgs.shape[1] != image_size_px or imgs.shape[2] != image_size_px:
        raise ValueError(
            "Zarr image dimensions do not match IMAGE_SIZE_PX."
        )

    obj_diam_ratio = float(obj_diam_ratio)
    if not 0.0 < obj_diam_ratio <= 1.0:
        raise ValueError(
            "Zarr attribute OBJ_DIAM_RATIO must be in (0, 1]."
        )

    gray_light_threshold = float(gray_light_threshold)
    if not 0.0 <= gray_light_threshold <= 1.0:
        raise ValueError(
            "Zarr attribute GRAY_LIGHT_THRESHOLD must be in [0, 1]."
        )

    missing_gallery_settings = [
        name
        for name in GALLERY_STIM_SETTING_KEYS
        if name not in stim_settings
    ]

    if missing_gallery_settings:
        raise ValueError(
            "STIM_SETTINGS is missing gallery setting(s): "
            + ", ".join(missing_gallery_settings)
        )

    return AuditDataset(
        dataset_dir=dataset_dir,
        zarr_path=zarr_path,
        metadata_path=metadata_path,
        root=root,
        imgs=imgs,
        metadata=metadata,
        image_size_px=image_size_px,
        obj_diam_ratio=obj_diam_ratio,
        gray_light_threshold=gray_light_threshold,
        stim_settings=dict(stim_settings),
    )


def _validate_stimulus_ids(metadata, n_images):
    """Validate that stim_ID uniquely indexes every Zarr image."""

    stim_ids = pd.to_numeric(
        metadata["stim_ID"],
        errors="coerce",
    )

    if stim_ids.isna().any():
        raise ValueError("metadata.csv contains invalid stim_ID values.")

    stim_id_values = stim_ids.to_numpy(dtype=float)

    if not np.equal(stim_id_values, np.floor(stim_id_values)).all():
        raise ValueError("Every stim_ID must be an integer.")

    stim_id_values = stim_id_values.astype(int)

    if len(np.unique(stim_id_values)) != len(stim_id_values):
        raise ValueError("metadata.csv contains duplicate stim_ID values.")

    expected_ids = np.arange(n_images, dtype=int)

    if not np.array_equal(
        np.sort(stim_id_values),
        expected_ids,
    ):
        raise ValueError(
            "stim_ID must uniquely cover every image index from "
            f"0 through {n_images - 1}."
        )


def _validate_metadata_values(metadata):
    """Validate labels and numeric stimulus metadata."""

    observed_labels = set(metadata["true_label"].dropna().astype(str))
    unknown_labels = observed_labels - set(TRUE_LABELS)

    if unknown_labels:
        raise ValueError(
            f"metadata.csv contains unknown true_label value(s): "
            f"{sorted(unknown_labels)}."
        )

    if metadata["true_label"].isna().any():
        raise ValueError("metadata.csv contains missing true_label values.")

    observed_bg_hues = set(metadata["bg_hue"].dropna().astype(str))
    unknown_bg_hues = observed_bg_hues - set(BG_HUES)

    if unknown_bg_hues:
        raise ValueError(
            f"metadata.csv contains unknown bg_hue value(s): "
            f"{sorted(unknown_bg_hues)}."
        )

    if metadata["bg_hue"].isna().any():
        raise ValueError("metadata.csv contains missing bg_hue values.")

    unit_interval_columns = (
        "bg_r",
        "bg_g",
        "bg_b",
        "obj_r",
        "obj_g",
        "obj_b",
        "bg_int",
        "obj_int",
        "bg_sat",
        "obj_sat",
    )

    delta_columns = (
        "delta_int",
        "delta_sat",
    )

    numeric_columns = unit_interval_columns + delta_columns

    numeric = metadata.loc[:, numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric.isna().any().any():
        raise ValueError(
            "metadata.csv contains missing or non-numeric stimulus values."
        )

    numeric_values = numeric.to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "metadata.csv contains non-finite stimulus values."
        )

    unit_values = numeric.loc[:, unit_interval_columns].to_numpy(dtype=float)

    if ((unit_values < 0.0) | (unit_values > 1.0)).any():
        raise ValueError(
            "RGB, intensity, and saturation values must be in [0, 1]."
        )

    delta_values = numeric.loc[:, delta_columns].to_numpy(dtype=float)

    if ((delta_values < -1.0) | (delta_values > 1.0)).any():
        raise ValueError(
            "delta_int and delta_sat values must be in [-1, 1]."
        )


def select_audit_stimuli(dataset, config):
    """Select stimuli using audit labels, backgrounds, and optional deltas."""

    metadata = dataset.metadata.copy()

    keep = (
        metadata["true_label"].isin(config.labels)
        & metadata["bg_hue"].isin(config.bg_hues)
    )

    if config.filter_delta_int:
        keep &= metadata["delta_int"].between(
            config.delta_int_min,
            config.delta_int_max,
            inclusive="both",
        )

    if config.filter_delta_sat:
        keep &= metadata["delta_sat"].between(
            config.delta_sat_min,
            config.delta_sat_max,
            inclusive="both",
        )

    selected = metadata.loc[keep].copy().reset_index(drop=True)

    if selected.empty:
        raise ValueError(
            "The audit configuration selected no stimuli."
        )

    selected["obj_diam_ratio"] = float(dataset.obj_diam_ratio)

    selected["user_label"] = ""
    selected["label_source"] = ""
    selected["matched_obj_diam_ratio"] = np.nan
    selected["existing_n_human_labels"] = pd.NA
    selected["existing_label_resolution"] = ""
    selected["existing_last_labeled_at"] = ""

    return selected


def _empty_human_label_lookup():
    """Return an empty human-label lookup with the expected columns."""

    return pd.DataFrame(columns=HUMAN_LABEL_LOOKUP_COLUMNS)


def _validate_human_label_observations(observations):
    """Validate and normalize raw human-label observations."""

    missing_columns = [
        column
        for column in HUMAN_LABEL_REQUIRED_COLUMNS
        if column not in observations.columns
    ]

    if missing_columns:
        raise ValueError(
            "Human-label observations are missing required column(s): "
            + ", ".join(missing_columns)
        )

    work = observations.copy()

    work["user_label"] = (
        work["user_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if work["user_label"].eq("").any():
        raise ValueError(
            "Human-label observations contain empty user_label values."
        )

    unknown_labels = set(work["user_label"]) - set(TRUE_LABELS)

    if unknown_labels:
        raise ValueError(
            "Human-label observations contain unknown label(s): "
            f"{sorted(unknown_labels)}."
        )

    try:
        work["_labeled_at"] = pd.to_datetime(
            work["labeled_at"],
            utc=True,
            errors="raise",
        )
    except (ValueError, TypeError) as error:
        raise ValueError(
            "Human-label observations contain invalid labeled_at values."
        ) from error

    rgb_columns = (
        "obj_r",
        "obj_g",
        "obj_b",
        "bg_r",
        "bg_g",
        "bg_b",
    )

    numeric_columns = rgb_columns + ("obj_diam_ratio",)

    numeric = work.loc[:, numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric.isna().any().any():
        raise ValueError(
            "Human-label observations contain missing or non-numeric "
            "RGB/ratio values."
        )

    values = numeric.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Human-label observations contain non-finite RGB/ratio values."
        )

    rgb_values = numeric.loc[:, rgb_columns].to_numpy(dtype=float)

    if ((rgb_values < 0.0) | (rgb_values > 1.0)).any():
        raise ValueError(
            "Human-label RGB values must be in [0, 1]."
        )

    ratios = numeric["obj_diam_ratio"].to_numpy(dtype=float)

    if ((ratios <= 0.0) | (ratios > 1.0)).any():
        raise ValueError(
            "Human-label obj_diam_ratio values must be in (0, 1]."
        )

    for column in numeric_columns:
        work[column] = numeric[column]

    return work


def load_raw_human_labels(raw_dir="data/raw/human_labels"):
    """Load immutable human-label observation CSV files."""

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        return pd.DataFrame(columns=HUMAN_LABEL_REQUIRED_COLUMNS)

    csv_paths = sorted(raw_dir.glob("*.csv"))

    if not csv_paths:
        return pd.DataFrame(columns=HUMAN_LABEL_REQUIRED_COLUMNS)

    tables = []

    for path in csv_paths:
        table = pd.read_csv(path)
        table["source_file"] = path.name
        tables.append(table)

    observations = pd.concat(
        tables,
        ignore_index=True,
        sort=False,
    )

    return _validate_human_label_observations(observations)


def _add_rgb_keys(table):
    """Add canonical object/background RGB keys to a table."""

    work = table.copy()

    work["_obj_rgb_key"] = [
        rgb_key(rgb)
        for rgb in work[
            ["obj_r", "obj_g", "obj_b"]
        ].to_numpy(dtype=float)
    ]

    work["_bg_rgb_key"] = [
        rgb_key(rgb)
        for rgb in work[
            ["bg_r", "bg_g", "bg_b"]
        ].to_numpy(dtype=float)
    ]

    return work


def consolidate_human_labels(observations):
    """Consolidate raw observations into one label per RGB pair and ratio."""

    if observations.empty:
        return _empty_human_label_lookup()

    work = _validate_human_label_observations(observations)
    work = _add_rgb_keys(work)

    work["_observation_order"] = np.arange(
        len(work),
        dtype=int,
    )

    rows = []

    group_columns = (
        "_obj_rgb_key",
        "_bg_rgb_key",
        "obj_diam_ratio",
    )

    for (
        obj_key,
        bg_key,
        obj_diam_ratio,
    ), group in work.groupby(
        list(group_columns),
        sort=False,
        dropna=False,
    ):
        label_counts = group["user_label"].value_counts()

        maximum_count = int(label_counts.max())

        tied_labels = label_counts[
            label_counts == maximum_count
        ].index.tolist()

        if len(tied_labels) == 1:
            human_label = tied_labels[0]
            resolution = "majority"

        else:
            tied_observations = group[
                group["user_label"].isin(tied_labels)
            ].sort_values(
                ["_labeled_at", "_observation_order"]
            )

            human_label = tied_observations.iloc[-1]["user_label"]
            resolution = "latest_tiebreak"

        latest_observation = group.sort_values(
            ["_labeled_at", "_observation_order"]
        ).iloc[-1]

        rows.append(
            {
                "obj_r": float(obj_key[0]),
                "obj_g": float(obj_key[1]),
                "obj_b": float(obj_key[2]),
                "bg_r": float(bg_key[0]),
                "bg_g": float(bg_key[1]),
                "bg_b": float(bg_key[2]),
                "obj_diam_ratio": float(obj_diam_ratio),
                "human_label": str(human_label),
                "n_human_labels": int(len(group)),
                "label_resolution": resolution,
                "last_labeled_at": (
                    latest_observation["_labeled_at"].isoformat()
                ),
            }
        )

    lookup = pd.DataFrame(
        rows,
        columns=HUMAN_LABEL_LOOKUP_COLUMNS,
    )

    return lookup


def rebuild_human_label_lookup(
    raw_dir="data/raw/human_labels",
    lookup_path=(
        "data/interm/human_labels/"
        "human_label_lookup.csv"
    ),
):
    """Rebuild the reusable lookup from immutable raw observations."""

    observations = load_raw_human_labels(raw_dir)
    lookup = consolidate_human_labels(observations)

    lookup_path = Path(lookup_path)
    lookup_path.parent.mkdir(parents=True, exist_ok=True)

    lookup.to_csv(lookup_path, index=False)

    return lookup


def load_human_label_lookup(
    path="data/interm/human_labels/human_label_lookup.csv",
):
    """Load the generated reusable human-label lookup."""

    path = Path(path)

    if not path.is_file():
        return _empty_human_label_lookup()

    lookup = pd.read_csv(path)

    missing_columns = [
        column
        for column in HUMAN_LABEL_LOOKUP_COLUMNS
        if column not in lookup.columns
    ]

    if missing_columns:
        raise ValueError(
            "Human-label lookup is missing required column(s): "
            + ", ".join(missing_columns)
        )

    return lookup


def _prepare_lookup_for_matching(lookup):
    """Prepare a human-label lookup for RGB and ratio matching."""

    if lookup.empty:
        return _empty_human_label_lookup()

    work = lookup.copy()

    numeric_columns = (
        "obj_r",
        "obj_g",
        "obj_b",
        "bg_r",
        "bg_g",
        "bg_b",
        "obj_diam_ratio",
        "n_human_labels",
    )

    numeric = work.loc[:, numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric.isna().any().any():
        raise ValueError(
            "Human-label lookup contains invalid numeric values."
        )

    for column in numeric_columns:
        work[column] = numeric[column]

    unknown_labels = (
        set(work["human_label"].astype(str))
        - set(TRUE_LABELS)
    )

    if unknown_labels:
        raise ValueError(
            "Human-label lookup contains unknown label(s): "
            f"{sorted(unknown_labels)}."
        )

    work["_last_labeled_at"] = pd.to_datetime(
        work["last_labeled_at"],
        utc=True,
        errors="raise",
    )

    return _add_rgb_keys(work)


def _lookup_candidates_for_stimulus(row, prepared_lookup):
    """Return lookup rows with the same object and background RGB."""

    if prepared_lookup.empty:
        return prepared_lookup

    obj_key = rgb_key(
        [row.obj_r, row.obj_g, row.obj_b]
    )

    bg_key = rgb_key(
        [row.bg_r, row.bg_g, row.bg_b]
    )

    return prepared_lookup.loc[
        (prepared_lookup["_obj_rgb_key"] == obj_key)
        & (prepared_lookup["_bg_rgb_key"] == bg_key)
    ]


def existing_label_coverage(selected, lookup):
    """Count mutually exclusive same- and different-ratio matches."""

    prepared_lookup = _prepare_lookup_for_matching(lookup)

    same_ratio = 0
    different_ratio = 0

    for row in selected.itertuples(index=False):
        candidates = _lookup_candidates_for_stimulus(
            row,
            prepared_lookup,
        )

        if candidates.empty:
            continue

        ratio_matches = np.isclose(
            candidates["obj_diam_ratio"].to_numpy(dtype=float),
            float(row.obj_diam_ratio),
            rtol=0.0,
            atol=RATIO_ATOL,
        )

        if ratio_matches.any():
            same_ratio += 1
        else:
            different_ratio += 1

    return LabelCoverage(
        n_selected=int(len(selected)),
        n_same_ratio=int(same_ratio),
        n_different_ratio=int(different_ratio),
    )


def _most_recent_lookup_row(rows):
    """Return the most recently labeled lookup row."""

    return rows.sort_values(
        "_last_labeled_at"
    ).iloc[-1]


def _choose_existing_label(
    row,
    prepared_lookup,
    allow_different_ratio,
):
    """Choose the best existing label for one selected stimulus."""

    candidates = _lookup_candidates_for_stimulus(
        row,
        prepared_lookup,
    )

    if candidates.empty:
        return None, None

    current_ratio = float(row.obj_diam_ratio)

    ratio_values = candidates[
        "obj_diam_ratio"
    ].to_numpy(dtype=float)

    exact_mask = np.isclose(
        ratio_values,
        current_ratio,
        rtol=0.0,
        atol=RATIO_ATOL,
    )

    if exact_mask.any():
        exact_rows = candidates.loc[exact_mask]
        chosen = _most_recent_lookup_row(exact_rows)

        return chosen, "existing_same_ratio"

    if not allow_different_ratio:
        return None, None

    distances = np.abs(
        ratio_values - current_ratio
    )

    minimum_distance = float(distances.min())

    nearest_mask = np.isclose(
        distances,
        minimum_distance,
        rtol=0.0,
        atol=RATIO_ATOL,
    )

    nearest_rows = candidates.loc[nearest_mask]
    chosen = _most_recent_lookup_row(nearest_rows)

    return chosen, "existing_different_ratio"


def apply_existing_labels(
    selected,
    lookup,
    use_existing,
    allow_different_ratio=False,
):
    """Fill selected stimuli with reusable human labels when requested."""

    result = selected.copy()

    if not use_existing or lookup.empty:
        return result

    prepared_lookup = _prepare_lookup_for_matching(lookup)

    for index, row in enumerate(
        result.itertuples(index=False)
    ):
        chosen, source = _choose_existing_label(
            row,
            prepared_lookup,
            allow_different_ratio,
        )

        if chosen is None:
            continue

        result.at[index, "user_label"] = str(
            chosen["human_label"]
        )

        result.at[index, "label_source"] = source

        result.at[index, "matched_obj_diam_ratio"] = float(
            chosen["obj_diam_ratio"]
        )

        result.at[index, "existing_n_human_labels"] = int(
            chosen["n_human_labels"]
        )

        result.at[index, "existing_label_resolution"] = str(
            chosen["label_resolution"]
        )

        result.at[index, "existing_last_labeled_at"] = str(
            chosen["last_labeled_at"]
        )

    return result


def _ask_yes_no(prompt, input_func=input):
    """Ask one yes/no question until the answer is valid."""

    while True:
        answer = input_func(prompt).strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Please enter y or n.")


def prompt_label_reuse(
    coverage,
    input_func=input,
    print_func=print,
):
    """Ask the two dataset-level existing-label reuse questions."""

    print_func(
        f"Previous human labels match "
        f"{coverage.n_any_match} of "
        f"{coverage.n_selected} selected stimuli:"
    )

    print_func(
        f"  {coverage.n_same_ratio} with the same "
        "object/background ratio"
    )

    print_func(
        f"  {coverage.n_different_ratio} with a different "
        "object/background ratio"
    )

    if coverage.n_any_match == 0:
        return False, False

    use_existing = _ask_yes_no(
        "Use existing human labels? [y/n] ",
        input_func=input_func,
    )

    if not use_existing:
        return False, False

    allow_different_ratio = _ask_yes_no(
        "Also allow labels from different "
        "object/background ratios? [y/n] ",
        input_func=input_func,
    )

    return True, allow_different_ratio


def randomize_pending_stimuli(audit_table, random_seed):
    """Randomize only stimuli that still require a human label."""

    user_labels = (
        audit_table["user_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    pending = audit_table.loc[
        user_labels.eq("")
    ].copy()

    pending = pending.sample(
        frac=1.0,
        random_state=random_seed,
    ).reset_index(drop=True)

    pending["audit_order"] = np.arange(
        len(pending),
        dtype=int,
    )

    return pending


def _utc_timestamp(value=None):
    """Return a timezone-aware UTC pandas timestamp."""

    if value is None:
        return pd.Timestamp.now(tz="UTC")

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def _atomic_write_csv(table, path):
    """Write a CSV through a temporary file before replacing the target."""

    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    table.to_csv(temp_path, index=False)
    temp_path.replace(path)


def _atomic_write_json(values, path):
    """Write JSON through a temporary file before replacing the target."""

    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            values,
            file,
            indent=2,
            sort_keys=True,
        )

    temp_path.replace(path)


def _validate_session_table(table):
    """Validate the structure of a saved audit-session table."""

    missing_columns = [
        column
        for column in SESSION_REQUIRED_COLUMNS
        if column not in table.columns
    ]

    if missing_columns:
        raise ValueError(
            "Audit session is missing required column(s): "
            + ", ".join(missing_columns)
        )

    if table.empty:
        raise ValueError(
            "Audit session contains no selected stimuli."
        )

    if table["stim_ID"].duplicated().any():
        raise ValueError(
            "Audit session contains duplicate stim_ID values."
        )

    user_labels = (
        table["user_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    labeled = user_labels.ne("")

    unknown_labels = (
        set(user_labels.loc[labeled])
        - set(TRUE_LABELS)
    )

    if unknown_labels:
        raise ValueError(
            "Audit session contains unknown user label(s): "
            f"{sorted(unknown_labels)}."
        )

    valid_sources = {
        "",
        "existing_same_ratio",
        "existing_different_ratio",
        "new_human",
    }

    label_sources = (
        table["label_source"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    unknown_sources = set(label_sources) - valid_sources

    if unknown_sources:
        raise ValueError(
            "Audit session contains unknown label_source value(s): "
            f"{sorted(unknown_sources)}."
        )

    raw_order = table["audit_order"]

    has_order = (
        raw_order.notna()
        & raw_order.astype(str).str.strip().ne("")
    )

    numeric_order = pd.to_numeric(
        raw_order,
        errors="coerce",
    )

    if (has_order & numeric_order.isna()).any():
        raise ValueError(
            "Audit session contains invalid audit_order values."
        )

    order_values = numeric_order.loc[has_order].to_numpy(dtype=float)

    if not np.equal(
        order_values,
        np.floor(order_values),
    ).all():
        raise ValueError(
            "Audit session audit_order values must be integers."
        )

    order_values = order_values.astype(int)

    if len(np.unique(order_values)) != len(order_values):
        raise ValueError(
            "Audit session contains duplicate audit_order values."
        )

    if len(order_values):
        expected = np.arange(
            len(order_values),
            dtype=int,
        )

        if not np.array_equal(
            np.sort(order_values),
            expected,
        ):
            raise ValueError(
                "Audit session audit_order must cover "
                "0 through N-1 without gaps."
            )


def audit_session_progress(table):
    """Summarize reused, newly labeled, and pending stimuli."""

    user_labels = (
        table["user_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    label_sources = (
        table["label_source"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    reused = (
        user_labels.ne("")
        & label_sources.isin(
            [
                "existing_same_ratio",
                "existing_different_ratio",
            ]
        )
    )

    new_human = (
        user_labels.ne("")
        & label_sources.eq("new_human")
    )

    pending = (
        user_labels.eq("")
        & table["audit_order"].notna()
    )

    return {
        "n_selected": int(len(table)),
        "n_reused": int(reused.sum()),
        "n_new_human": int(new_human.sum()),
        "n_pending": int(pending.sum()),
    }


def create_audit_session(
    audit_table,
    dataset,
    config,
    use_existing,
    allow_different_ratio,
    base_dir="data/interm/audit_runs",
    created_at=None,
):
    """Create a new resumable working audit session."""

    table = audit_table.copy()

    if table.empty:
        raise ValueError(
            "Cannot create an audit session with no stimuli."
        )

    pending = randomize_pending_stimuli(
        table,
        config.random_seed,
    )

    order_by_stim_id = dict(
        zip(
            pending["stim_ID"],
            pending["audit_order"],
        )
    )

    table["audit_order"] = (
        table["stim_ID"]
        .map(order_by_stim_id)
        .astype("Int64")
    )

    table["labeled_at"] = ""

    timestamp = _utc_timestamp(created_at)

    session_id = (
        "audit_"
        + timestamp.strftime("%Y%m%d_%H%M%S_%f")
    )

    base_dir = Path(base_dir)
    session_dir = base_dir / session_id

    session_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    table_path = session_dir / "audit_table.csv"
    state_path = session_dir / "session.json"

    progress = audit_session_progress(table)

    state = {
        "version": 1,
        "session_id": session_id,
        "created_at": timestamp.isoformat(),
        "updated_at": timestamp.isoformat(),
        "status": (
            "completed"
            if progress["n_pending"] == 0
            else "active"
        ),
        "dataset_dir": str(dataset.dataset_dir),
        "random_seed": int(config.random_seed),
        "use_existing": bool(use_existing),
        "allow_different_ratio": bool(
            allow_different_ratio
        ),
        "selection": {
            "labels": list(config.labels),
            "bg_hues": list(config.bg_hues),
            "filter_delta_int": bool(
                config.filter_delta_int
            ),
            "delta_int_min": float(
                config.delta_int_min
            ),
            "delta_int_max": float(
                config.delta_int_max
            ),
            "filter_delta_sat": bool(
                config.filter_delta_sat
            ),
            "delta_sat_min": float(
                config.delta_sat_min
            ),
            "delta_sat_max": float(
                config.delta_sat_max
            ),
        },
        **progress,
    }

    _validate_session_table(table)

    _atomic_write_csv(
        table,
        table_path,
    )

    _atomic_write_json(
        state,
        state_path,
    )

    return AuditSession(
        session_dir=session_dir,
        table_path=table_path,
        state_path=state_path,
        table=table,
        state=state,
    )


def load_audit_session(session_dir):
    """Load an existing audit session so labeling can resume."""

    session_dir = Path(session_dir)

    if not session_dir.is_dir():
        raise FileNotFoundError(
            f"Audit session directory not found: {session_dir}"
        )

    table_path = session_dir / "audit_table.csv"
    state_path = session_dir / "session.json"

    if not table_path.is_file():
        raise FileNotFoundError(
            f"Audit session table not found: {table_path}"
        )

    if not state_path.is_file():
        raise FileNotFoundError(
            f"Audit session state not found: {state_path}"
        )

    table = pd.read_csv(table_path)

    table["user_label"] = (
        table["user_label"]
        .fillna("")
        .astype(str)
    )

    table["label_source"] = (
        table["label_source"]
        .fillna("")
        .astype(str)
    )

    table["labeled_at"] = (
        table["labeled_at"]
        .fillna("")
        .astype(str)
    )

    numeric_order = pd.to_numeric(
        table["audit_order"],
        errors="coerce",
    )

    table["audit_order"] = numeric_order.astype("Int64")

    _validate_session_table(table)

    with state_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        state = json.load(file)

    if state.get("session_id") != session_dir.name:
        raise ValueError(
            "session.json session_id does not match "
            "the audit-session directory name."
        )

    if state.get("n_selected") != len(table):
        raise ValueError(
            "session.json n_selected does not match "
            "the audit table."
        )

    return AuditSession(
        session_dir=session_dir,
        table_path=table_path,
        state_path=state_path,
        table=table,
        state=state,
    )


def next_pending_stimulus(session):
    """Return the next unlabeled stimulus in randomized audit order."""

    table = session.table

    user_labels = (
        table["user_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    pending = table.loc[
        table["audit_order"].notna()
        & user_labels.eq("")
    ].copy()

    if pending.empty:
        return None

    pending = pending.sort_values(
        "audit_order"
    )

    return pending.iloc[0].copy()


def record_audit_label(
    session,
    audit_order,
    user_label,
    labeled_at=None,
):
    """Record one new human judgment and save it immediately."""

    if user_label not in TRUE_LABELS:
        raise ValueError(
            f"user_label must be one of {list(TRUE_LABELS)}."
        )

    if type(audit_order) is not int or audit_order < 0:
        raise ValueError(
            "audit_order must be a non-negative integer."
        )

    table = session.table.copy()

    order_matches = (
        table["audit_order"]
        .eq(audit_order)
        .fillna(False)
    )

    matching_indices = table.index[
        order_matches
    ].tolist()

    if len(matching_indices) != 1:
        raise ValueError(
            f"No unique stimulus has audit_order={audit_order}."
        )

    index = matching_indices[0]

    existing_label = str(
        table.at[index, "user_label"]
        if pd.notna(table.at[index, "user_label"])
        else ""
    ).strip()

    if existing_label:
        raise ValueError(
            f"audit_order={audit_order} is already labeled."
        )

    timestamp = _utc_timestamp(labeled_at)

    table.at[index, "user_label"] = user_label
    table.at[index, "label_source"] = "new_human"
    table.at[index, "labeled_at"] = timestamp.isoformat()

    _validate_session_table(table)

    progress = audit_session_progress(table)

    state = dict(session.state)

    state.update(progress)
    state["updated_at"] = timestamp.isoformat()
    state["status"] = (
        "completed"
        if progress["n_pending"] == 0
        else "active"
    )

    _atomic_write_csv(
        table,
        session.table_path,
    )

    _atomic_write_json(
        state,
        session.state_path,
    )

    session.table = table
    session.state = state

    return table.loc[index].copy()


def finalize_audit_session(
    session,
    raw_dir="data/raw/human_labels",
    lookup_path="data/interm/human_labels/human_label_lookup.csv",
    finalized_at=None,
):
    """Finalize new human judgments into raw storage and rebuild the lookup."""

    _validate_session_table(session.table)

    progress = audit_session_progress(session.table)

    if progress["n_pending"] != 0:
        raise ValueError(
            "Cannot finalize an audit session while stimuli remain unlabeled."
        )

    dataset_dir = session.state.get("dataset_dir")

    if not dataset_dir:
        raise ValueError(
            "Audit session does not identify its source dataset."
        )

    dataset = load_audit_dataset(dataset_dir)

    label_sources = (
        session.table["label_source"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    new_rows = session.table.loc[
        label_sources.eq("new_human")
    ].copy()

    new_rows = new_rows.sort_values(
        "audit_order"
    ).reset_index(drop=True)

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = None

    if not new_rows.empty:
        labeled_at = pd.to_datetime(
            new_rows["labeled_at"],
            utc=True,
            errors="coerce",
        )

        if labeled_at.isna().any():
            raise ValueError(
                "New human judgments contain invalid labeled_at values."
            )

        session_id = str(
            session.state["session_id"]
        )

        raw_table = pd.DataFrame(
            {
                "label_id": [
                    f"{session_id}:{int(order):06d}"
                    for order in new_rows["audit_order"]
                ],
                "labeled_at": [
                    value.isoformat()
                    for value in labeled_at
                ],
                "user_label": new_rows[
                    "user_label"
                ].astype(str),
                "obj_r": new_rows["obj_r"].astype(float),
                "obj_g": new_rows["obj_g"].astype(float),
                "obj_b": new_rows["obj_b"].astype(float),
                "bg_r": new_rows["bg_r"].astype(float),
                "bg_g": new_rows["bg_g"].astype(float),
                "bg_b": new_rows["bg_b"].astype(float),
                "obj_diam_ratio": new_rows[
                    "obj_diam_ratio"
                ].astype(float),
                "image_size_px": int(
                    dataset.image_size_px
                ),
                "source_dataset": (
                    dataset.dataset_dir.name
                ),
                "source_stim_ID": new_rows[
                    "stim_ID"
                ].astype(int),
                "pipeline_label": new_rows[
                    "true_label"
                ].astype(str),
                "bg_hue": new_rows[
                    "bg_hue"
                ].astype(str),
                "obj_int": new_rows[
                    "obj_int"
                ].astype(float),
                "obj_sat": new_rows[
                    "obj_sat"
                ].astype(float),
                "bg_int": new_rows[
                    "bg_int"
                ].astype(float),
                "bg_sat": new_rows[
                    "bg_sat"
                ].astype(float),
                "delta_int": new_rows[
                    "delta_int"
                ].astype(float),
                "delta_sat": new_rows[
                    "delta_sat"
                ].astype(float),
            },
            columns=RAW_HUMAN_LABEL_COLUMNS,
        )

        raw_path = (
            raw_dir
            / f"human_labels_{session_id}.csv"
        )

        new_csv_text = raw_table.to_csv(
            index=False
        )

        if raw_path.exists():
            existing_csv_text = raw_path.read_text(
                encoding="utf-8"
            )

            if existing_csv_text != new_csv_text:
                raise ValueError(
                    "A raw human-label file already exists for "
                    "this session but its contents differ."
                )

        else:
            _atomic_write_csv(
                raw_table,
                raw_path,
            )

    lookup = rebuild_human_label_lookup(
        raw_dir=raw_dir,
        lookup_path=lookup_path,
    )

    state = dict(session.state)

    if state.get("finalized_at"):
        timestamp = _utc_timestamp(
            state["finalized_at"]
        )
    else:
        timestamp = _utc_timestamp(
            finalized_at
        )

    state["status"] = "finalized"
    state["finalized_at"] = timestamp.isoformat()
    state["raw_label_path"] = (
        str(raw_path)
        if raw_path is not None
        else ""
    )
    state["human_label_lookup_path"] = str(
        lookup_path
    )
    state["n_finalized_new_human"] = int(
        len(new_rows)
    )

    _atomic_write_json(
        state,
        session.state_path,
    )

    session.state = state

    return raw_path, lookup


def _format_rgb(row, prefix):
    """Format one RGB triplet for audit-analysis tables."""

    values = [
        float(row[f"{prefix}_r"]),
        float(row[f"{prefix}_g"]),
        float(row[f"{prefix}_b"]),
    ]

    return (
        f"({values[0]:.3f}, "
        f"{values[1]:.3f}, "
        f"{values[2]:.3f})"
    )


def analyze_audit_session(
    session,
    save=True,
):
    """Build summary, crosstab, and mismatch tables for an audit."""

    table = session.table.copy()

    user_labels = (
        table["user_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    labeled = user_labels.ne("")

    matches = pd.Series(
        pd.NA,
        index=table.index,
        dtype="boolean",
    )

    matches.loc[labeled] = (
        user_labels.loc[labeled]
        == table.loc[labeled, "true_label"].astype(str)
    )

    table["match_true_label"] = matches

    n_total = int(len(table))
    n_labeled = int(labeled.sum())
    n_unlabeled = n_total - n_labeled
    n_matches = int(matches.fillna(False).sum())
    n_mismatches = n_labeled - n_matches

    agreement_rate = (
        n_matches / n_labeled
        if n_labeled
        else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "n_total": n_total,
                "n_labeled": n_labeled,
                "n_unlabeled": n_unlabeled,
                "n_matches": n_matches,
                "n_mismatches": n_mismatches,
                "agreement_rate": agreement_rate,
            }
        ]
    )

    labeled_table = table.loc[
        labeled
    ].copy()

    if labeled_table.empty:
        crosstab = pd.DataFrame()

    else:
        crosstab = pd.crosstab(
            labeled_table["user_label"],
            labeled_table["true_label"],
        )

    mismatch_mask = (
        table["match_true_label"]
        .fillna(True)
        .eq(False)
    )

    mismatches = table.loc[
        mismatch_mask
    ].copy()

    if not mismatches.empty:
        mismatches["obj_rgb"] = mismatches.apply(
            _format_rgb,
            axis=1,
            prefix="obj",
        )

        mismatches["bg_rgb"] = mismatches.apply(
            _format_rgb,
            axis=1,
            prefix="bg",
        )

    mismatch_columns = (
        "stim_ID",
        "true_label",
        "user_label",
        "label_source",
        "bg_hue",
        "obj_rgb",
        "bg_rgb",
        "obj_int",
        "obj_sat",
        "bg_int",
        "bg_sat",
        "delta_int",
        "delta_sat",
    )

    if mismatches.empty:
        mismatches = pd.DataFrame(
            columns=mismatch_columns
        )
    else:
        mismatches = mismatches.loc[
            :,
            mismatch_columns,
        ].reset_index(drop=True)

    paths = {}

    if save:
        results_path = (
            session.session_dir
            / "audit_results.csv"
        )

        summary_path = (
            session.session_dir
            / "audit_summary.csv"
        )

        crosstab_path = (
            session.session_dir
            / "audit_crosstab.csv"
        )

        mismatch_path = (
            session.session_dir
            / "audit_mismatches.csv"
        )

        _atomic_write_csv(
            table,
            results_path,
        )

        _atomic_write_csv(
            summary,
            summary_path,
        )

        _atomic_write_csv(
            crosstab,
            crosstab_path,
        )

        _atomic_write_csv(
            mismatches,
            mismatch_path,
        )

        paths = {
            "results": results_path,
            "summary": summary_path,
            "crosstab": crosstab_path,
            "mismatches": mismatch_path,
        }

    return summary, crosstab, mismatches, paths


def select_gallery_stimuli(
    dataset,
    gallery_config,
):
    """Select the object setting nearest the configured mid setting."""

    metadata = dataset.metadata.copy()

    target_label = gallery_config.target_label

    matching_label = metadata.loc[
        metadata["true_label"].eq(
            target_label
        )
    ].copy()

    if matching_label.empty:
        raise ValueError(
            f"No stimuli have true_label='{target_label}'."
        )

    settings = dataset.stim_settings

    if target_label == "gray_d":
        target_obj_int = 0.5 * (
            float(settings["obj_gray_min"])
            + float(dataset.gray_light_threshold)
        )

        target_obj_sat = 0.0

    elif target_label == "gray_l":
        target_obj_int = 0.5 * (
            float(dataset.gray_light_threshold)
            + float(settings["obj_gray_max"])
        )

        target_obj_sat = 0.0

    else:
        target_obj_int = 0.5 * (
            float(settings["obj_chrom_int_min"])
            + float(settings["obj_chrom_int_max"])
        )

        target_obj_sat = 0.5 * (
            float(settings["obj_chrom_sat_min"])
            + float(settings["obj_chrom_sat_max"])
        )

    object_settings = (
        matching_label[
            ["obj_int", "obj_sat"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    object_settings["_distance"] = (
        np.abs(
            object_settings["obj_int"]
            - target_obj_int
        )
        + np.abs(
            object_settings["obj_sat"]
            - target_obj_sat
        )
    )

    object_settings = object_settings.sort_values(
        "_distance",
        kind="stable",
    )

    chosen = object_settings.iloc[0]

    chosen_obj_int = float(
        chosen["obj_int"]
    )

    chosen_obj_sat = float(
        chosen["obj_sat"]
    )

    selected = matching_label.loc[
        np.isclose(
            matching_label["obj_int"],
            chosen_obj_int,
            rtol=0.0,
            atol=1e-9,
        )
        & np.isclose(
            matching_label["obj_sat"],
            chosen_obj_sat,
            rtol=0.0,
            atol=1e-9,
        )
    ].copy()

    hsv_values = [
        rgb_to_hsv01(
            [
                row.bg_r,
                row.bg_g,
                row.bg_b,
            ]
        )
        for row in selected.itertuples(
            index=False
        )
    ]

    selected["_bg_h"] = [
        hsv[0]
        for hsv in hsv_values
    ]

    selected["_bg_s"] = [
        hsv[1]
        for hsv in hsv_values
    ]

    selected["_bg_v"] = [
        hsv[2]
        for hsv in hsv_values
    ]

    selected["_gray_first"] = (
        selected["bg_hue"]
        .ne("gray")
        .astype(int)
    )

    selected = selected.sort_values(
        [
            "_gray_first",
            "_bg_h",
            "_bg_s",
            "_bg_v",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return (
        selected,
        chosen_obj_int,
        chosen_obj_sat,
    )


def plot_stimulus_gallery(
    dataset,
    gallery_config,
):
    """Display the configured stimulus gallery and optionally save it."""

    import matplotlib.pyplot as plt

    (
        selected,
        chosen_obj_int,
        chosen_obj_sat,
    ) = select_gallery_stimuli(
        dataset,
        gallery_config,
    )

    n_images = len(selected)
    n_cols = int(gallery_config.n_cols)

    n_rows = int(
        np.ceil(
            n_images / n_cols
        )
    )

    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            3.2 * n_cols,
            3.2 * n_rows,
        ),
        squeeze=False,
    )

    axes = axes.ravel()

    for axis, row in zip(
        axes,
        selected.itertuples(index=False),
    ):
        image = np.asarray(
            dataset.imgs[
                int(row.stim_ID)
            ]
        )

        axis.imshow(
            np.clip(
                image,
                0.0,
                1.0,
            )
        )

        axis.axis("off")

        obj_rgb = (
            row.obj_r,
            row.obj_g,
            row.obj_b,
        )

        bg_rgb = (
            row.bg_r,
            row.bg_g,
            row.bg_b,
        )

        obj_hsv = rgb_to_hsv01(
            obj_rgb
        )

        bg_hsv = rgb_to_hsv01(
            bg_rgb
        )

        axis.set_title(
            (
                f"stim {int(row.stim_ID)} | "
                f"{row.true_label}\n"
                f"obj RGB={tuple(round(v, 3) for v in obj_rgb)}\n"
                f"obj HSV={tuple(round(float(v), 3) for v in obj_hsv)}\n"
                f"bg RGB={tuple(round(v, 3) for v in bg_rgb)}\n"
                f"bg HSV={tuple(round(float(v), 3) for v in bg_hsv)}\n"
                f"Δint={row.delta_int:.3f}, "
                f"Δsat={row.delta_sat:.3f}"
            ),
            fontsize=8,
        )

    for axis in axes[n_images:]:
        axis.axis("off")

    figure.suptitle(
        (
            f"Gallery for true_label='{gallery_config.target_label}' "
            f"at obj_int={chosen_obj_int:.3f}, "
            f"obj_sat={chosen_obj_sat:.3f} "
            f"({n_images} images)"
        ),
        fontsize=14,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    png_path = None

    if gallery_config.save_png:
        timestamp = pd.Timestamp.now(
            tz="UTC"
        ).strftime(
            "%Y%m%d_%H%M%S"
        )

        figures_dir = Path("figures")
        figures_dir.mkdir(parents=True, exist_ok=True)

        png_path = (
            figures_dir
            / (
                f"gallery_"
                f"{gallery_config.target_label}_"
                f"{timestamp}.png"
            )
        )

        figure.savefig(
            png_path,
            dpi=gallery_config.dpi,
            bbox_inches="tight",
        )

    return (
        figure,
        selected,
        png_path,
    )