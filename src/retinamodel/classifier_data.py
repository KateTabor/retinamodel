"""Shared data loading and preparation for the classifier pipeline."""

from __future__ import annotations

import json
import tomllib
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import zoom


_REQUIRED_METADATA_COLUMNS = ["stim_ID", "true_label", "obj_int", "obj_sat", "bg_int", "bg_sat", "bg_hue"]
_CLASS_ORDER = ("gray_d", "gray_l", "red", "green", "blue", "yellow")
_SPLIT_METADATA_COLUMNS = ("true_label", "obj_int", "obj_sat", "bg_hue")
_DEFAULT_TRAIN_FRACTION = 0.70
_DEFAULT_VALIDATION_FRACTION = 0.15
_DEFAULT_TEST_FRACTION = 0.15
_DEFAULT_SPLIT_SEEDS = (12,)
_DEFAULT_INTENSITY_BINS = 3
_DEFAULT_SATURATION_BINS = 3
_DEFAULT_BALANCE_BACKGROUND_HUE = True
_DEFAULT_INPUT_SIZES = ("full", 256)
_DOWNSAMPLE_ORDER = 1
_DEFAULT_OUTPUT_ROOT = Path("data/interm")


@dataclass(frozen=True)
class _SplitConfig:
    """Experimenter-controlled settings for shared classifier splits."""

    train_fraction: float
    validation_fraction: float
    test_fraction: float
    seeds: tuple[int, ...]
    intensity_bins: int
    saturation_bins: int
    balance_background_hue: bool


@dataclass(frozen=True)
class _InputConfig:
    """Experimenter-controlled spatial input sizes."""

    sizes: tuple[str | int, ...]


@dataclass(frozen=True)
class _OutputConfig:
    """Location for lightweight classifier-data run records."""

    root_dir: Path


@dataclass(frozen=True)
class _ClassifierDataConfig:
    """Experimenter-controlled settings loaded from classifier_data.toml."""

    source_dataset_dir: Path
    retina_model_dir: Path
    config_path: Path
    class_weights: dict[str, float]
    channel_groups: dict[str, tuple[str, ...]]
    split: _SplitConfig
    inputs: _InputConfig
    output: _OutputConfig


@dataclass
class _ClassifierDataSources:
    """Internal container for validated source and retinal-response data."""

    source_dataset_dir: Path
    retina_model_dir: Path
    imgs: Any
    outs_fill: Any
    metadata: pd.DataFrame
    response_keys: tuple[str, ...]
    processing_settings: dict[str, Any]


@dataclass(frozen=True)
class _ClassifierDataSchema:
    """Validated shared definitions used by all classifier models."""

    class_order: tuple[str, ...]
    active_class_order: tuple[str, ...]
    class_weights: dict[str, float]
    class_fractions: dict[str, float]
    channel_groups: dict[str, tuple[str, ...]]
    channel_indices: dict[str, tuple[int, ...]]
    split_metadata_columns: tuple[str, ...]


@dataclass(frozen=True)
class _WeightedSamplePlan:
    """Deterministic weighted-sampling totals before stimulus IDs are drawn."""

    dataset_total: int
    available_counts: dict[str, int]
    selected_total: int
    selected_counts: dict[str, int]
    unused_total: int
    unused_counts: dict[str, int]


@dataclass(frozen=True)
class _SharedSplit:
    """One deterministic shared train/validation/test split for all classifiers."""

    seed: int
    train_stim_ids: np.ndarray
    validation_stim_ids: np.ndarray
    test_stim_ids: np.ndarray
    unused_stim_ids: np.ndarray


@dataclass(frozen=True)
class _LabeledStimulusSet:
    """Stimulus IDs, labels, and metadata for supervised model use."""

    stim_ids: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class _UnlabeledStimulusSet:
    """Stimulus IDs and metadata exposed without held-out labels."""

    stim_ids: np.ndarray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class _ModelSplitView:
    """Model-facing split that keeps held-out test labels unavailable."""

    seed: int
    train: _LabeledStimulusSet
    validation: _LabeledStimulusSet
    test: _UnlabeledStimulusSet


@dataclass(frozen=True)
class LabeledClassifierInput:
    """Prepared model input paired with labels for train or validation use."""

    stim_ids: np.ndarray
    X: Any
    labels: np.ndarray
    channel_group: str
    input_size: str | int


@dataclass(frozen=True)
class UnlabeledClassifierInput:
    """Prepared held-out model input without test labels."""

    stim_ids: np.ndarray
    X: Any
    channel_group: str
    input_size: str | int


class ResponseMapInput:
    """Lazy float32 retinal-response maps for CNN-style inputs."""

    def __init__(self, outs_fill: Any, stim_ids: np.ndarray, channel_indices: tuple[int, ...], input_size: str | int):
        self._outs_fill = outs_fill
        self.stim_ids = np.asarray(stim_ids, dtype=np.int64).copy()
        self.channel_indices = tuple(int(i) for i in channel_indices)
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.stim_ids)

    def __getitem__(self, item: int) -> np.ndarray:
        stim_id = int(self.stim_ids[item])
        maps = np.asarray(self._outs_fill[stim_id, :, :, :], dtype=np.float32)[list(self.channel_indices), :, :]
        return _resize_response_stack(maps, self.input_size)


@dataclass
class ClassifierDataBundle:
    """Shared classifier data, split views, provenance, and on-demand input settings."""

    config: _ClassifierDataConfig
    schema: _ClassifierDataSchema
    sample_plan: _WeightedSamplePlan
    splits: dict[int, _SharedSplit]
    model_views: dict[int, _ModelSplitView]
    provenance: dict[str, Any]
    _data_sources: _ClassifierDataSources
    output_dir: Path | None = None
    manifest_path: Path | None = None


def _validate_class_weights(class_weights: dict[str, Any]) -> dict[str, float]:
    """Validate experimenter-specified relative class weights."""

    missing_labels = sorted(set(_CLASS_ORDER) - set(class_weights))
    unknown_labels = sorted(set(class_weights) - set(_CLASS_ORDER))

    if missing_labels:
        raise ValueError(f"Class weights are missing canonical classes: {missing_labels}")

    if unknown_labels:
        raise ValueError(f"Class weights contain unknown classes: {unknown_labels}")

    validated = {}

    for label in _CLASS_ORDER:
        try:
            weight = float(class_weights[label])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Class weight for '{label}' must be numeric.") from exc

        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"Class weight for '{label}' must be finite and >= 0, found {class_weights[label]!r}.")

        validated[label] = weight

    if sum(weight > 0 for weight in validated.values()) < 2:
        raise ValueError("At least two classes must have positive sampling weights for classifier training.")

    return validated


def _validate_channel_group_definitions(channel_groups: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Validate the structure of experimenter-defined retinal channel groups."""

    if not channel_groups:
        raise ValueError("At least one retinal channel group must be defined.")

    validated = {}

    for group_name, channel_names in channel_groups.items():
        group_name = str(group_name)

        if not group_name.strip():
            raise ValueError("Channel-group names must not be empty.")

        if not isinstance(channel_names, list | tuple) or len(channel_names) == 0:
            raise ValueError(f"Channel group '{group_name}' must contain at least one retinal channel.")

        channels = tuple(str(channel) for channel in channel_names)

        if any(not channel.strip() for channel in channels):
            raise ValueError(f"Channel group '{group_name}' contains an empty channel name.")

        if len(set(channels)) != len(channels):
            raise ValueError(f"Channel group '{group_name}' contains duplicate retinal channels.")

        validated[group_name] = channels

    return validated


def _validate_split_config(split_settings: dict[str, Any] | None) -> _SplitConfig:
    """Validate shared split settings and apply documented defaults."""

    split_settings = {} if split_settings is None else split_settings

    if not isinstance(split_settings, dict):
        raise ValueError("classifier_data.toml [split] must be a table when provided.")

    try:
        train_fraction = float(split_settings.get("train_fraction", _DEFAULT_TRAIN_FRACTION))
        validation_fraction = float(split_settings.get("validation_fraction", _DEFAULT_VALIDATION_FRACTION))
        test_fraction = float(split_settings.get("test_fraction", _DEFAULT_TEST_FRACTION))
    except (TypeError, ValueError) as exc:
        raise ValueError("Split fractions must be numeric.") from exc

    fractions = (train_fraction, validation_fraction, test_fraction)

    if any(not np.isfinite(value) or value <= 0 for value in fractions):
        raise ValueError("Train, validation, and test fractions must be finite and > 0.")

    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("train_fraction + validation_fraction + test_fraction must equal 1.0.")

    seeds_raw = split_settings.get("seeds", list(_DEFAULT_SPLIT_SEEDS))

    if not isinstance(seeds_raw, list | tuple) or len(seeds_raw) == 0:
        raise ValueError("split.seeds must contain at least one integer seed.")

    if any(isinstance(seed, bool) or not isinstance(seed, int | np.integer) for seed in seeds_raw):
        raise ValueError("split.seeds must contain only integers.")

    seeds = tuple(int(seed) for seed in seeds_raw)

    if len(set(seeds)) != len(seeds):
        raise ValueError("split.seeds contains duplicate seeds.")

    intensity_bins = split_settings.get("intensity_bins", _DEFAULT_INTENSITY_BINS)
    saturation_bins = split_settings.get("saturation_bins", _DEFAULT_SATURATION_BINS)

    if isinstance(intensity_bins, bool) or not isinstance(intensity_bins, int) or intensity_bins < 1:
        raise ValueError("split.intensity_bins must be an integer >= 1.")

    if isinstance(saturation_bins, bool) or not isinstance(saturation_bins, int) or saturation_bins < 1:
        raise ValueError("split.saturation_bins must be an integer >= 1.")

    balance_background_hue = split_settings.get("balance_background_hue", _DEFAULT_BALANCE_BACKGROUND_HUE)

    if not isinstance(balance_background_hue, bool):
        raise ValueError("split.balance_background_hue must be true or false.")

    return _SplitConfig(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seeds=seeds,
        intensity_bins=intensity_bins,
        saturation_bins=saturation_bins,
        balance_background_hue=balance_background_hue,
    )


def _validate_input_config(input_settings: dict[str, Any] | None) -> _InputConfig:
    """Validate requested full-resolution and square resized input representations."""

    input_settings = {} if input_settings is None else input_settings

    if not isinstance(input_settings, dict):
        raise ValueError("classifier_data.toml [inputs] must be a table when provided.")

    sizes_raw = input_settings.get("sizes", list(_DEFAULT_INPUT_SIZES))

    if not isinstance(sizes_raw, list | tuple) or len(sizes_raw) == 0:
        raise ValueError("inputs.sizes must contain at least one entry.")

    sizes = []

    for value in sizes_raw:
        if isinstance(value, str):
            if value.lower() != "full":
                raise ValueError("String input sizes must be 'full'.")
            normalized = "full"
        elif isinstance(value, bool) or not isinstance(value, int | np.integer) or int(value) <= 0:
            raise ValueError("Numeric input sizes must be positive integers.")
        else:
            normalized = int(value)

        if normalized in sizes:
            raise ValueError(f"inputs.sizes contains duplicate entry: {normalized!r}.")

        sizes.append(normalized)

    return _InputConfig(sizes=tuple(sizes))


def _validate_output_config(output_settings: dict[str, Any] | None, project_root: Path) -> _OutputConfig:
    """Validate the root directory used for lightweight bundle records."""

    output_settings = {} if output_settings is None else output_settings

    if not isinstance(output_settings, dict):
        raise ValueError("classifier_data.toml [output] must be a table when provided.")

    root_dir = Path(str(output_settings.get("root_dir", _DEFAULT_OUTPUT_ROOT))).expanduser()

    if not root_dir.is_absolute():
        root_dir = project_root / root_dir

    return _OutputConfig(root_dir=root_dir.resolve())


def _load_classifier_data_config(config_path: str | Path) -> _ClassifierDataConfig:
    """Load experimenter-controlled classifier-data settings from TOML."""

    config_path = Path(config_path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Missing classifier-data config: {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    paths = config.get("paths")
    class_weights = config.get("class_weights")
    channel_groups = config.get("channel_groups")

    if not isinstance(paths, dict):
        raise ValueError("classifier_data.toml must contain a [paths] table.")

    if not isinstance(class_weights, dict):
        raise ValueError("classifier_data.toml must contain a [class_weights] table.")

    if not isinstance(channel_groups, dict):
        raise ValueError("classifier_data.toml must contain a [channel_groups] table.")

    missing_path_keys = [key for key in ("source_dataset_dir", "retina_model_dir") if key not in paths]

    if missing_path_keys:
        raise ValueError(f"classifier_data.toml [paths] is missing required settings: {missing_path_keys}")

    # Relative paths in config/classifier_data.toml are interpreted from the repository root.
    project_root = config_path.parent.parent
    source_dataset_dir = Path(str(paths["source_dataset_dir"])).expanduser()
    retina_model_dir = Path(str(paths["retina_model_dir"])).expanduser()

    if not source_dataset_dir.is_absolute():
        source_dataset_dir = project_root / source_dataset_dir

    if not retina_model_dir.is_absolute():
        retina_model_dir = project_root / retina_model_dir

    return _ClassifierDataConfig(
        source_dataset_dir=source_dataset_dir.resolve(),
        retina_model_dir=retina_model_dir.resolve(),
        config_path=config_path,
        class_weights=_validate_class_weights(class_weights),
        channel_groups=_validate_channel_group_definitions(channel_groups),
        split=_validate_split_config(config.get("split")),
        inputs=_validate_input_config(config.get("inputs")),
        output=_validate_output_config(config.get("output"), project_root),
    )


def _load_and_validate_data_sources(source_dataset_dir: str | Path, retina_model_dir: str | Path) -> _ClassifierDataSources:
    """Load the source stimuli, retinal responses, metadata, and processing provenance."""

    source_dataset_dir = Path(source_dataset_dir).expanduser().resolve()
    retina_model_dir = Path(retina_model_dir).expanduser().resolve()
    source_zarr_path = source_dataset_dir / "dataset.zarr"
    source_metadata_path = source_dataset_dir / "metadata.csv"
    retina_zarr_path = retina_model_dir / "retina_outputs.zarr"
    retina_metadata_path = retina_model_dir / "metadata.csv"
    processing_settings_path = retina_model_dir / "processing_settings.json"
    required_paths = {
        "source Zarr": source_zarr_path,
        "source metadata": source_metadata_path,
        "retina-response Zarr": retina_zarr_path,
        "retina metadata": retina_metadata_path,
        "processing settings": processing_settings_path,
    }

    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    dataset_root = zarr.open_group(str(source_zarr_path), mode="r")
    retina_root = zarr.open_group(str(retina_zarr_path), mode="r")

    if "imgs" not in dataset_root:
        raise ValueError("dataset.zarr must contain top-level array 'imgs'.")

    if "outs_fill" not in retina_root:
        raise ValueError("retina_outputs.zarr must contain top-level array 'outs_fill'.")

    if "RESPONSE_KEYS" not in retina_root.attrs:
        raise ValueError("retina_outputs.zarr must contain the RESPONSE_KEYS attribute.")

    imgs = dataset_root["imgs"]
    outs_fill = retina_root["outs_fill"]
    response_keys = tuple(str(key) for key in retina_root.attrs["RESPONSE_KEYS"])

    if imgs.ndim != 4:
        raise ValueError(f"Expected imgs to have 4 dimensions, found shape {imgs.shape}.")

    if outs_fill.ndim != 4:
        raise ValueError(f"Expected outs_fill to have 4 dimensions, found shape {outs_fill.shape}.")

    n_samples, n_channels, height, width = map(int, outs_fill.shape)

    if int(imgs.shape[0]) != n_samples:
        raise ValueError(f"Sample-count mismatch: imgs={imgs.shape[0]}, outs_fill={n_samples}.")

    if tuple(int(x) for x in imgs.shape[1:3]) != (height, width):
        raise ValueError(f"Spatial-dimension mismatch: imgs={imgs.shape[1:3]}, outs_fill={(height, width)}.")

    if len(response_keys) != n_channels:
        raise ValueError(f"Channel-count mismatch: outs_fill has {n_channels} channels but RESPONSE_KEYS has {len(response_keys)} entries.")

    with open(processing_settings_path, "r") as f:
        processing_settings = json.load(f)

    if "source_dataset" not in processing_settings:
        raise ValueError("processing_settings.json is missing 'source_dataset'.")

    saved_source_dataset = Path(processing_settings["source_dataset"]).expanduser().resolve()

    if saved_source_dataset != source_dataset_dir:
        raise ValueError(
            "Processed retina run does not reference the selected source dataset.\n"
            f"processing_settings.json source_dataset: {saved_source_dataset}\n"
            f"selected source dataset:                  {source_dataset_dir}"
        )

    metadata = pd.read_csv(retina_metadata_path).reset_index(drop=True)
    missing_columns = [column for column in _REQUIRED_METADATA_COLUMNS if column not in metadata.columns]

    if missing_columns:
        raise ValueError(f"Retina metadata is missing required columns: {missing_columns}")

    if len(metadata) != n_samples:
        raise ValueError(f"Metadata-row mismatch: metadata={len(metadata)}, outs_fill={n_samples}.")

    stim_ids_numeric = pd.to_numeric(metadata["stim_ID"], errors="coerce")

    if not stim_ids_numeric.notna().all():
        raise ValueError("Found missing or non-numeric values in metadata['stim_ID'].")

    metadata["stim_ID"] = stim_ids_numeric.astype(np.int64)
    expected_stim_ids = np.arange(n_samples, dtype=np.int64)

    if not np.array_equal(metadata["stim_ID"].to_numpy(dtype=np.int64), expected_stim_ids):
        raise ValueError("Expected metadata stim_ID order to be exactly [0, 1, ..., N-1] to match the source images and retinal-response rows.")

    return _ClassifierDataSources(
        source_dataset_dir=source_dataset_dir,
        retina_model_dir=retina_model_dir,
        imgs=imgs,
        outs_fill=outs_fill,
        metadata=metadata,
        response_keys=response_keys,
        processing_settings=processing_settings,
    )


def _define_shared_data_schema(data_sources: _ClassifierDataSources, class_weights: dict[str, float], channel_groups: dict[str, tuple[str, ...]]) -> _ClassifierDataSchema:
    """Validate and define the shared class, sampling, channel, and split-metadata schema."""

    weights = _validate_class_weights(class_weights)
    groups = _validate_channel_group_definitions(channel_groups)
    active_class_order = tuple(label for label in _CLASS_ORDER if weights[label] > 0)
    total_positive_weight = float(sum(weights[label] for label in active_class_order))
    class_fractions = {label: (weights[label] / total_positive_weight if weights[label] > 0 else 0.0) for label in _CLASS_ORDER}
    dataset_labels = set(data_sources.metadata["true_label"].astype(str).unique())
    unknown_dataset_labels = sorted(dataset_labels - set(_CLASS_ORDER))

    if unknown_dataset_labels:
        raise ValueError(f"Metadata contains unknown classifier labels: {unknown_dataset_labels}")

    missing_active_classes = [label for label in active_class_order if label not in dataset_labels]

    if missing_active_classes:
        raise ValueError(f"Classes with positive sampling weights are absent from the dataset: {missing_active_classes}")

    missing_split_columns = [column for column in _SPLIT_METADATA_COLUMNS if column not in data_sources.metadata.columns]

    if missing_split_columns:
        raise ValueError(f"Metadata is missing columns required for classifier splitting: {missing_split_columns}")

    response_key_to_index = {key: i for i, key in enumerate(data_sources.response_keys)}
    channel_indices = {}

    for group_name, channel_names in groups.items():
        missing_channels = [channel for channel in channel_names if channel not in response_key_to_index]

        if missing_channels:
            raise ValueError(f"Channel group '{group_name}' requires missing retinal channels: {missing_channels}")

        channel_indices[group_name] = tuple(response_key_to_index[channel] for channel in channel_names)

    return _ClassifierDataSchema(
        class_order=tuple(_CLASS_ORDER),
        active_class_order=active_class_order,
        class_weights=weights,
        class_fractions=class_fractions,
        channel_groups=groups,
        channel_indices=channel_indices,
        split_metadata_columns=tuple(_SPLIT_METADATA_COLUMNS),
    )


def _allocate_counts(total: int, labels: list[str] | tuple[str, ...], frac_dict: dict[str, float]) -> dict[str, int]:
    """Allocate an integer total by largest remainder while preserving label order for ties."""

    quotas = {label: float(total) * float(frac_dict[label]) for label in labels}
    base = {label: int(np.floor(quotas[label])) for label in labels}
    need = int(total - sum(base.values()))

    if need > 0:
        remainders = sorted(labels, key=lambda label: quotas[label] - base[label], reverse=True)
        for label in remainders[:need]:
            base[label] += 1

    return base


def _build_weighted_sample_plan(data_sources: _ClassifierDataSources, schema: _ClassifierDataSchema) -> _WeightedSamplePlan:
    """Calculate the largest sample that preserves the requested relative class weights."""

    label_series = data_sources.metadata["true_label"].astype(str)
    available_counts = {label: int((label_series == label).sum()) for label in schema.class_order}
    total_eff_float = min(float(available_counts[label]) / float(schema.class_fractions[label]) for label in schema.active_class_order)
    selected_total = int(np.floor(total_eff_float))

    while selected_total > 0:
        active_selected_counts = _allocate_counts(selected_total, schema.active_class_order, schema.class_fractions)
        if all(active_selected_counts[label] <= available_counts[label] for label in schema.active_class_order):
            break
        selected_total -= 1

    if selected_total <= 0:
        raise ValueError("Could not create a weighted sample from this dataset.")

    selected_counts = {label: int(active_selected_counts.get(label, 0)) for label in schema.class_order}
    unused_counts = {label: int(available_counts[label] - selected_counts[label]) for label in schema.class_order}

    return _WeightedSamplePlan(
        dataset_total=int(len(data_sources.metadata)),
        available_counts=available_counts,
        selected_total=selected_total,
        selected_counts=selected_counts,
        unused_total=int(len(data_sources.metadata) - selected_total),
        unused_counts=unused_counts,
    )


def _bin_continuous(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin continuous values exactly as in the classifier reference notebook."""

    values = np.asarray(values, dtype=float)
    edges = np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), int(n_bins) + 1)
    mids = edges[1:-1]
    out = np.digitize(values, mids, right=False).astype(np.int64)
    return np.clip(out, 0, int(n_bins) - 1)


def _sample_balanced(idxs: np.ndarray, n_select: int, rng: np.random.Generator, int_bin_series: pd.Series, sat_bin_series: pd.Series, bg_id_series: pd.Series | None = None) -> np.ndarray:
    """Select IDs across intensity, saturation, and optional background-hue cells."""

    idxs = np.asarray(idxs, dtype=np.int64)

    if n_select <= 0:
        return np.array([], dtype=np.int64)

    if n_select >= len(idxs):
        out = idxs.copy()
        rng.shuffle(out)
        return out

    cells = {}

    for i in idxs:
        if bg_id_series is None:
            key = (int(int_bin_series.loc[int(i)]), int(sat_bin_series.loc[int(i)]))
        else:
            key = (int(int_bin_series.loc[int(i)]), int(sat_bin_series.loc[int(i)]), int(bg_id_series.loc[int(i)]))
        cells.setdefault(key, []).append(int(i))

    cell_keys = list(cells.keys())
    rng.shuffle(cell_keys)
    base = {cell_key: (n_select // len(cell_keys)) for cell_key in cell_keys}
    remainder = int(n_select - sum(base.values()))

    while remainder > 0:
        progressed = False

        for cell_key in cell_keys:
            if remainder <= 0:
                break

            if base[cell_key] < len(cells[cell_key]):
                base[cell_key] += 1
                remainder -= 1
                progressed = True

        if not progressed:
            break

    selected = []

    for cell_key in cell_keys:
        arr = np.asarray(cells[cell_key], dtype=np.int64)
        rng.shuffle(arr)
        take = int(base[cell_key])

        if take > 0:
            selected.append(arr[:take])

    selected = np.concatenate(selected).astype(np.int64) if selected else np.array([], dtype=np.int64)

    if len(selected) < n_select:
        remaining = np.setdiff1d(idxs, selected, assume_unique=False).astype(np.int64)
        rng.shuffle(remaining)
        need = int(n_select - len(selected))
        selected = np.concatenate([selected, remaining[:need]]).astype(np.int64)

    rng.shuffle(selected)
    return selected


def _validate_shared_split(train_stim_ids: np.ndarray, validation_stim_ids: np.ndarray, test_stim_ids: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check split uniqueness, disjointness, and membership in the validated dataset."""

    train_stim_ids = np.asarray(train_stim_ids, dtype=np.int64)
    validation_stim_ids = np.asarray(validation_stim_ids, dtype=np.int64)
    test_stim_ids = np.asarray(test_stim_ids, dtype=np.int64)

    for name, stim_ids in (("train", train_stim_ids), ("validation", validation_stim_ids), ("test", test_stim_ids)):
        if len(np.unique(stim_ids)) != len(stim_ids):
            raise ValueError(f"{name} split contains duplicate stim_ID values.")

    if np.intersect1d(train_stim_ids, validation_stim_ids).size > 0:
        raise ValueError("Train and validation splits overlap.")

    if np.intersect1d(train_stim_ids, test_stim_ids).size > 0:
        raise ValueError("Train and test splits overlap.")

    if np.intersect1d(validation_stim_ids, test_stim_ids).size > 0:
        raise ValueError("Validation and test splits overlap.")

    valid_stim_ids = metadata["stim_ID"].to_numpy(dtype=np.int64)
    selected_stim_ids = np.concatenate([train_stim_ids, validation_stim_ids, test_stim_ids]).astype(np.int64)
    unknown_stim_ids = np.setdiff1d(selected_stim_ids, valid_stim_ids, assume_unique=False)

    if unknown_stim_ids.size > 0:
        raise ValueError(f"Shared split contains stim_ID values not present in metadata: {unknown_stim_ids.tolist()}")

    unused_stim_ids = np.setdiff1d(valid_stim_ids, selected_stim_ids, assume_unique=False).astype(np.int64)
    return train_stim_ids, validation_stim_ids, test_stim_ids, unused_stim_ids


def _generate_shared_split(data_sources: _ClassifierDataSources, schema: _ClassifierDataSchema, sample_plan: _WeightedSamplePlan, split_config: _SplitConfig, seed: int) -> _SharedSplit:
    """Generate one deterministic weighted split using the reference notebook procedure."""

    class_order = schema.active_class_order
    validation_total = int(round(split_config.validation_fraction * sample_plan.selected_total))
    test_total = int(round(split_config.test_fraction * sample_plan.selected_total))
    train_total = int(sample_plan.selected_total - validation_total - test_total)

    if min(train_total, validation_total, test_total) < len(class_order):
        raise ValueError("Weighted sample is too small to place every active class in train, validation, and test.")

    train_counts = _allocate_counts(train_total, class_order, schema.class_fractions)
    validation_counts = _allocate_counts(validation_total, class_order, schema.class_fractions)
    test_counts = {label: int(sample_plan.selected_counts[label] - train_counts[label] - validation_counts[label]) for label in class_order}

    if any(test_counts[label] < 0 for label in class_order):
        raise ValueError("Integer allocation produced a negative test count for at least one class.")

    if sum(train_counts.values()) != train_total:
        raise ValueError("Training class counts do not sum to the requested training total.")

    if sum(validation_counts.values()) != validation_total:
        raise ValueError("Validation class counts do not sum to the requested validation total.")

    if sum(test_counts.values()) != test_total:
        raise ValueError("Test class counts do not sum to the requested test total.")

    rng = np.random.default_rng(int(seed))
    metadata = data_sources.metadata.copy()
    metadata["_obj_int_r"] = np.round(metadata["obj_int"].astype(float), 6)
    metadata["_obj_sat_r"] = np.round(metadata["obj_sat"].astype(float), 6)
    int_bin_series = pd.Series(_bin_continuous(metadata["_obj_int_r"].to_numpy(), split_config.intensity_bins), index=metadata.index)
    sat_bin_series = pd.Series(_bin_continuous(metadata["_obj_sat_r"].to_numpy(), split_config.saturation_bins), index=metadata.index)

    if split_config.balance_background_hue:
        bg_str = metadata["bg_hue"].astype(str).to_numpy()
        bg_levels = sorted(np.unique(bg_str).tolist())
        bg_map = {background: i for i, background in enumerate(bg_levels)}
        bg_id_series = pd.Series(np.asarray([bg_map[background] for background in bg_str], dtype=np.int64), index=metadata.index)
    else:
        bg_id_series = None

    label_series = metadata["true_label"].astype(str)
    idx_by_label = {label: metadata.index[label_series == label].to_numpy(dtype=np.int64) for label in class_order}
    train_parts = []
    validation_parts = []
    test_parts = []

    for label in class_order:
        idxs = idx_by_label[label]

        # Protect the final test set first, then validation, then draw training from what remains.
        test_ids = _sample_balanced(idxs, int(test_counts[label]), rng, int_bin_series, sat_bin_series, bg_id_series)
        remaining = np.setdiff1d(idxs, test_ids, assume_unique=False).astype(np.int64)
        validation_ids = _sample_balanced(remaining, int(validation_counts[label]), rng, int_bin_series, sat_bin_series, bg_id_series)
        remaining = np.setdiff1d(remaining, validation_ids, assume_unique=False).astype(np.int64)
        train_ids = _sample_balanced(remaining, int(train_counts[label]), rng, int_bin_series, sat_bin_series, bg_id_series)
        train_parts.append(train_ids)
        validation_parts.append(validation_ids)
        test_parts.append(test_ids)

    train_stim_ids = np.concatenate(train_parts).astype(np.int64)
    validation_stim_ids = np.concatenate(validation_parts).astype(np.int64)
    test_stim_ids = np.concatenate(test_parts).astype(np.int64)
    rng.shuffle(train_stim_ids)
    rng.shuffle(validation_stim_ids)
    rng.shuffle(test_stim_ids)
    train_stim_ids, validation_stim_ids, test_stim_ids, unused_stim_ids = _validate_shared_split(train_stim_ids, validation_stim_ids, test_stim_ids, metadata)

    actual_train_counts = metadata.loc[train_stim_ids, "true_label"].astype(str).value_counts().reindex(class_order).fillna(0).astype(int).to_dict()
    actual_validation_counts = metadata.loc[validation_stim_ids, "true_label"].astype(str).value_counts().reindex(class_order).fillna(0).astype(int).to_dict()
    actual_test_counts = metadata.loc[test_stim_ids, "true_label"].astype(str).value_counts().reindex(class_order).fillna(0).astype(int).to_dict()

    if actual_train_counts != train_counts:
        raise ValueError("Actual training counts do not match requested counts.")

    if actual_validation_counts != validation_counts:
        raise ValueError("Actual validation counts do not match requested counts.")

    if actual_test_counts != test_counts:
        raise ValueError("Actual test counts do not match requested counts.")

    return _SharedSplit(
        seed=int(seed),
        train_stim_ids=train_stim_ids,
        validation_stim_ids=validation_stim_ids,
        test_stim_ids=test_stim_ids,
        unused_stim_ids=unused_stim_ids,
    )


def _generate_shared_splits(data_sources: _ClassifierDataSources, schema: _ClassifierDataSchema, sample_plan: _WeightedSamplePlan, split_config: _SplitConfig) -> dict[int, _SharedSplit]:
    """Generate one shared deterministic split for each configured seed."""

    return {seed: _generate_shared_split(data_sources, schema, sample_plan, split_config, seed) for seed in split_config.seeds}


def _labels_for_stim_ids(metadata: pd.DataFrame, stim_ids: np.ndarray) -> np.ndarray:
    """Return labels for validated stimulus IDs in the supplied order."""

    stim_ids = np.asarray(stim_ids, dtype=np.int64)
    return metadata.loc[stim_ids, "true_label"].astype(str).to_numpy(copy=True)


def _build_model_split_view(data_sources: _ClassifierDataSources, shared_split: _SharedSplit) -> _ModelSplitView:
    """Expose labeled train/validation views while withholding held-out test labels."""

    train_ids = np.asarray(shared_split.train_stim_ids, dtype=np.int64).copy()
    validation_ids = np.asarray(shared_split.validation_stim_ids, dtype=np.int64).copy()
    test_ids = np.asarray(shared_split.test_stim_ids, dtype=np.int64).copy()
    train_metadata = data_sources.metadata.loc[train_ids].copy().reset_index(drop=True)
    validation_metadata = data_sources.metadata.loc[validation_ids].copy().reset_index(drop=True)
    test_metadata = data_sources.metadata.loc[test_ids, ["stim_ID"]].copy().reset_index(drop=True)
    train = _LabeledStimulusSet(stim_ids=train_ids, labels=_labels_for_stim_ids(data_sources.metadata, train_ids), metadata=train_metadata)
    validation = _LabeledStimulusSet(stim_ids=validation_ids, labels=_labels_for_stim_ids(data_sources.metadata, validation_ids), metadata=validation_metadata)
    test = _UnlabeledStimulusSet(stim_ids=test_ids, metadata=test_metadata)
    return _ModelSplitView(seed=shared_split.seed, train=train, validation=validation, test=test)


def _build_model_split_views(data_sources: _ClassifierDataSources, shared_splits: dict[int, _SharedSplit]) -> dict[int, _ModelSplitView]:
    """Build one leakage-protected model-facing view for each shared split."""

    return {seed: _build_model_split_view(data_sources, split) for seed, split in shared_splits.items()}


def _load_classifier_data_and_schema(config_path: str | Path) -> tuple[_ClassifierDataSources, _ClassifierDataSchema]:
    """Load and validate classifier data sources and their shared schema."""

    config = _load_classifier_data_config(config_path)
    data_sources = _load_and_validate_data_sources(config.source_dataset_dir, config.retina_model_dir)
    schema = _define_shared_data_schema(data_sources, config.class_weights, config.channel_groups)
    return data_sources, schema

def _validate_configured_input_sizes(data_sources: _ClassifierDataSources, input_config: _InputConfig) -> None:
    """Reject configured numeric sizes that would upsample retinal-response maps."""

    height, width = map(int, data_sources.outs_fill.shape[2:4])

    for size in input_config.sizes:
        if size != "full" and (int(size) > height or int(size) > width):
            raise ValueError(f"Configured input size {size} would upsample {height}x{width} response maps. Use 'full' or a size <= {min(height, width)}.")


def _resize_response_stack(maps: np.ndarray, input_size: str | int) -> np.ndarray:
    """Return contiguous float32 maps at full resolution or a requested square size."""

    maps = np.asarray(maps, dtype=np.float32)

    if input_size == "full":
        return np.ascontiguousarray(maps, dtype=np.float32)

    size = int(input_size)
    height, width = map(int, maps.shape[-2:])
    z0 = float(size) / float(height)
    z1 = float(size) / float(width)
    resized = [zoom(maps[channel], (z0, z1), order=_DOWNSAMPLE_ORDER).astype(np.float32, copy=False) for channel in range(maps.shape[0])]
    return np.ascontiguousarray(np.stack(resized, axis=0), dtype=np.float32)


def _validate_input_request(bundle: ClassifierDataBundle, seed: int, split_name: str, channel_group: str, input_size: str | int) -> tuple[_ModelSplitView, tuple[int, ...]]:
    """Validate one on-demand model-input request against the recorded bundle settings."""

    if seed not in bundle.model_views:
        raise ValueError(f"Unknown split seed {seed}. Available seeds: {list(bundle.model_views)}")

    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name must be 'train', 'validation', or 'test'.")

    if channel_group not in bundle.schema.channel_indices:
        raise ValueError(f"Unknown channel group '{channel_group}'. Available groups: {list(bundle.schema.channel_indices)}")

    normalized_size = "full" if isinstance(input_size, str) and input_size.lower() == "full" else input_size

    if normalized_size not in bundle.config.inputs.sizes:
        raise ValueError(f"Input size {input_size!r} is not configured. Available sizes: {list(bundle.config.inputs.sizes)}")

    return bundle.model_views[seed], bundle.schema.channel_indices[channel_group]


def _stimulus_set_for_split(view: _ModelSplitView, split_name: str) -> _LabeledStimulusSet | _UnlabeledStimulusSet:
    """Return one protected split view by name."""

    return getattr(view, split_name)


def build_cnn_input(bundle: ClassifierDataBundle, seed: int, split_name: str, channel_group: str, input_size: str | int) -> LabeledClassifierInput | UnlabeledClassifierInput:
    """Build lazy channel-first float32 response-map inputs for CNN-style models."""

    view, channel_indices = _validate_input_request(bundle, seed, split_name, channel_group, input_size)
    input_size = "full" if isinstance(input_size, str) and input_size.lower() == "full" else input_size
    stimulus_set = _stimulus_set_for_split(view, split_name)
    maps = ResponseMapInput(bundle._data_sources.outs_fill, stimulus_set.stim_ids, channel_indices, input_size)

    if isinstance(stimulus_set, _LabeledStimulusSet):
        return LabeledClassifierInput(stim_ids=stimulus_set.stim_ids.copy(), X=maps, labels=stimulus_set.labels.copy(), channel_group=channel_group, input_size=input_size)

    return UnlabeledClassifierInput(stim_ids=stimulus_set.stim_ids.copy(), X=maps, channel_group=channel_group, input_size=input_size)


def build_sklearn_input(bundle: ClassifierDataBundle, seed: int, split_name: str, channel_group: str, input_size: str | int, mmap_path: str | Path | None = None) -> LabeledClassifierInput | UnlabeledClassifierInput:
    """Build flattened float32 response features using the notebook's stack-and-flatten behavior."""

    view, channel_indices = _validate_input_request(bundle, seed, split_name, channel_group, input_size)
    input_size = "full" if isinstance(input_size, str) and input_size.lower() == "full" else input_size
    stimulus_set = _stimulus_set_for_split(view, split_name)
    maps = ResponseMapInput(bundle._data_sources.outs_fill, stimulus_set.stim_ids, channel_indices, input_size)
    sample_shape = maps[0].shape
    n_features = int(np.prod(sample_shape, dtype=np.int64))

    if mmap_path is None:
        X = np.empty((len(maps), n_features), dtype=np.float32)
    else:
        mmap_path = Path(mmap_path).expanduser().resolve()
        mmap_path.parent.mkdir(parents=True, exist_ok=True)
        X = np.memmap(str(mmap_path), dtype=np.float32, mode="w+", shape=(len(maps), n_features))

    for row in range(len(maps)):
        X[row] = maps[row].reshape(-1)

    if isinstance(X, np.memmap):
        X.flush()

    if isinstance(stimulus_set, _LabeledStimulusSet):
        return LabeledClassifierInput(stim_ids=stimulus_set.stim_ids.copy(), X=X, labels=stimulus_set.labels.copy(), channel_group=channel_group, input_size=input_size)

    return UnlabeledClassifierInput(stim_ids=stimulus_set.stim_ids.copy(), X=X, channel_group=channel_group, input_size=input_size)


def _class_counts(metadata: pd.DataFrame, stim_ids: np.ndarray, class_order: tuple[str, ...]) -> dict[str, int]:
    """Count labels for recorded split summaries."""

    counts = metadata.loc[np.asarray(stim_ids, dtype=np.int64), "true_label"].astype(str).value_counts()
    return {label: int(counts.get(label, 0)) for label in class_order}


def _bundle_manifest(bundle: ClassifierDataBundle) -> dict[str, Any]:
    """Build a JSON-safe reproducibility record for one shared classifier-data run."""

    sources = bundle._data_sources
    split_records = {}

    for seed, split in bundle.splits.items():
        split_records[str(seed)] = {
            "train_n": int(len(split.train_stim_ids)),
            "validation_n": int(len(split.validation_stim_ids)),
            "test_n": int(len(split.test_stim_ids)),
            "unused_n": int(len(split.unused_stim_ids)),
            "train_class_counts": _class_counts(sources.metadata, split.train_stim_ids, bundle.schema.active_class_order),
            "validation_class_counts": _class_counts(sources.metadata, split.validation_stim_ids, bundle.schema.active_class_order),
            "split_file": f"split_seed{seed}.npz",
            "train_metadata_file": f"train_metadata_seed{seed}.csv",
            "validation_metadata_file": f"validation_metadata_seed{seed}.csv",
            "test_metadata_file": f"test_metadata_seed{seed}.csv",
        }

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "config_path": str(bundle.config.config_path),
        "source_dataset_dir": str(sources.source_dataset_dir),
        "retina_model_dir": str(sources.retina_model_dir),
        "imgs_shape": [int(x) for x in sources.imgs.shape],
        "outs_fill_shape": [int(x) for x in sources.outs_fill.shape],
        "response_keys": list(sources.response_keys),
        "processing_settings": sources.processing_settings,
        "class_order": list(bundle.schema.class_order),
        "active_class_order": list(bundle.schema.active_class_order),
        "class_weights": bundle.schema.class_weights,
        "class_fractions": bundle.schema.class_fractions,
        "channel_groups": {name: list(channels) for name, channels in bundle.schema.channel_groups.items()},
        "channel_indices": {name: list(indices) for name, indices in bundle.schema.channel_indices.items()},
        "selected_total": int(bundle.sample_plan.selected_total),
        "selected_counts": bundle.sample_plan.selected_counts,
        "unused_total": int(bundle.sample_plan.unused_total),
        "unused_counts": bundle.sample_plan.unused_counts,
        "split": {
            "train_fraction": bundle.config.split.train_fraction,
            "validation_fraction": bundle.config.split.validation_fraction,
            "test_fraction": bundle.config.split.test_fraction,
            "seeds": list(bundle.config.split.seeds),
            "intensity_bins": bundle.config.split.intensity_bins,
            "saturation_bins": bundle.config.split.saturation_bins,
            "balance_background_hue": bundle.config.split.balance_background_hue,
        },
        "inputs": {"sizes": list(bundle.config.inputs.sizes), "interpolation_order": _DOWNSAMPLE_ORDER},
        "test_labels_saved": False,
        "splits": split_records,
    }


def _new_bundle_output_dir(root_dir: Path) -> Path:
    """Create a timestamped output directory without overwriting an existing run."""

    root_dir.mkdir(parents=True, exist_ok=True)
    stem = "classifier_data_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root_dir / stem
    suffix = 1

    while candidate.exists():
        candidate = root_dir / f"{stem}_{suffix:02d}"
        suffix += 1

    candidate.mkdir()
    return candidate


def record_classifier_data_bundle(bundle: ClassifierDataBundle, output_dir: str | Path | None = None) -> Path:
    """Save lightweight split IDs, protected metadata views, settings, and provenance."""

    if output_dir is None:
        run_dir = _new_bundle_output_dir(bundle.config.output.root_dir)
    else:
        run_dir = Path(output_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

    for seed, split in bundle.splits.items():
        np.savez(run_dir / f"split_seed{seed}.npz", train_idx=split.train_stim_ids, val_idx=split.validation_stim_ids, test_idx=split.test_stim_ids, unused_idx=split.unused_stim_ids)
        view = bundle.model_views[seed]
        view.train.metadata.to_csv(run_dir / f"train_metadata_seed{seed}.csv", index=False)
        view.validation.metadata.to_csv(run_dir / f"validation_metadata_seed{seed}.csv", index=False)
        view.test.metadata.to_csv(run_dir / f"test_metadata_seed{seed}.csv", index=False)

    config_snapshot_path = run_dir / "classifier_data_config.toml"

    if bundle.config.config_path.exists():
        config_snapshot_path.write_bytes(bundle.config.config_path.read_bytes())

    manifest_path = run_dir / "classifier_data_manifest.json"
    manifest_path.write_text(json.dumps(_bundle_manifest(bundle), indent=2, sort_keys=True), encoding="utf-8")
    bundle.output_dir = run_dir
    bundle.manifest_path = manifest_path
    return run_dir


def build_classifier_data_bundle(config_path: str | Path, output_dir: str | Path | None = None, record: bool = True) -> ClassifierDataBundle:
    """Build the validated shared classifier-data bundle and optionally record it to disk."""

    config = _load_classifier_data_config(config_path)
    data_sources = _load_and_validate_data_sources(config.source_dataset_dir, config.retina_model_dir)
    _validate_configured_input_sizes(data_sources, config.inputs)
    schema = _define_shared_data_schema(data_sources, config.class_weights, config.channel_groups)
    sample_plan = _build_weighted_sample_plan(data_sources, schema)
    splits = _generate_shared_splits(data_sources, schema, sample_plan, config.split)
    model_views = _build_model_split_views(data_sources, splits)
    provenance = {"source_dataset_dir": str(data_sources.source_dataset_dir), "retina_model_dir": str(data_sources.retina_model_dir), "processing_settings": data_sources.processing_settings}
    bundle = ClassifierDataBundle(config=config, schema=schema, sample_plan=sample_plan, splits=splits, model_views=model_views, provenance=provenance, _data_sources=data_sources)

    if record:
        record_classifier_data_bundle(bundle, output_dir=output_dir)

    return bundle


def load_classifier_data_bundle(bundle_dir: str | Path) -> ClassifierDataBundle:
    """Load a recorded lightweight bundle and validate it against its referenced data."""

    bundle_dir = Path(bundle_dir).expanduser().resolve()
    manifest_path = bundle_dir / "classifier_data_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing classifier-data manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_dataset_dir = Path(manifest["source_dataset_dir"]).expanduser().resolve()
    retina_model_dir = Path(manifest["retina_model_dir"]).expanduser().resolve()
    data_sources = _load_and_validate_data_sources(source_dataset_dir, retina_model_dir)

    if [int(x) for x in data_sources.imgs.shape] != manifest["imgs_shape"]:
        raise ValueError("Recorded imgs shape no longer matches the referenced source dataset.")

    if [int(x) for x in data_sources.outs_fill.shape] != manifest["outs_fill_shape"]:
        raise ValueError("Recorded outs_fill shape no longer matches the referenced retina-model run.")

    if list(data_sources.response_keys) != manifest["response_keys"]:
        raise ValueError("Recorded response keys no longer match the referenced retina-model run.")

    split_settings = manifest["split"]
    split_config = _validate_split_config(split_settings)
    if int(manifest["inputs"].get("interpolation_order", _DOWNSAMPLE_ORDER)) != _DOWNSAMPLE_ORDER:
        raise ValueError(f"Recorded interpolation order must be {_DOWNSAMPLE_ORDER}.")

    input_config = _validate_input_config(manifest["inputs"])
    _validate_configured_input_sizes(data_sources, input_config)
    output_config = _OutputConfig(root_dir=bundle_dir.parent)
    config_snapshot_path = bundle_dir / "classifier_data_config.toml"
    original_config_path = Path(manifest.get("config_path", config_snapshot_path)).expanduser()
    config_path = config_snapshot_path if config_snapshot_path.exists() else original_config_path
    channel_groups = {name: tuple(channels) for name, channels in manifest["channel_groups"].items()}
    config = _ClassifierDataConfig(source_dataset_dir=source_dataset_dir, retina_model_dir=retina_model_dir, config_path=config_path, class_weights=_validate_class_weights(manifest["class_weights"]), channel_groups=_validate_channel_group_definitions(channel_groups), split=split_config, inputs=input_config, output=output_config)
    schema = _define_shared_data_schema(data_sources, config.class_weights, config.channel_groups)
    sample_plan = _build_weighted_sample_plan(data_sources, schema)

    if sample_plan.selected_total != int(manifest["selected_total"]) or sample_plan.selected_counts != {key: int(value) for key, value in manifest["selected_counts"].items()}:
        raise ValueError("Recorded weighted-sample plan no longer matches the referenced dataset.")

    splits = {}

    for seed in split_config.seeds:
        split_path = bundle_dir / f"split_seed{seed}.npz"

        if not split_path.exists():
            raise FileNotFoundError(f"Missing recorded split: {split_path}")

        with np.load(split_path, allow_pickle=False) as saved:
            train_ids, validation_ids, test_ids, unused_ids = _validate_shared_split(saved["train_idx"], saved["val_idx"], saved["test_idx"], data_sources.metadata)

            if "unused_idx" in saved.files and not np.array_equal(unused_ids, saved["unused_idx"].astype(np.int64)):
                raise ValueError(f"Recorded unused IDs are inconsistent for seed {seed}.")

        splits[seed] = _SharedSplit(seed=seed, train_stim_ids=train_ids, validation_stim_ids=validation_ids, test_stim_ids=test_ids, unused_stim_ids=unused_ids)

    model_views = _build_model_split_views(data_sources, splits)
    provenance = {"source_dataset_dir": str(source_dataset_dir), "retina_model_dir": str(retina_model_dir), "processing_settings": data_sources.processing_settings}
    return ClassifierDataBundle(config=config, schema=schema, sample_plan=sample_plan, splits=splits, model_views=model_views, provenance=provenance, _data_sources=data_sources, output_dir=bundle_dir, manifest_path=manifest_path)


def summarize_classifier_data_bundle(bundle: ClassifierDataBundle) -> str:
    """Return a concise human-readable summary of the shared classifier data."""

    groups = ", ".join(bundle.schema.channel_groups)
    sizes = ", ".join(str(size) for size in bundle.config.inputs.sizes)
    seeds = ", ".join(str(seed) for seed in bundle.config.split.seeds)
    output = str(bundle.output_dir) if bundle.output_dir is not None else "not recorded"
    return f"Dataset: {bundle.sample_plan.dataset_total} stimuli | selected: {bundle.sample_plan.selected_total} | unused: {bundle.sample_plan.unused_total} | seeds: {seeds} | input sizes: {sizes} | channel groups: {groups} | output: {output}"

