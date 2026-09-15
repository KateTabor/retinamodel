"""Interactive viewer helpers for Module 4 retinal-response datasets."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import zarr


MAP_TYPES = (
    "filled",
    "extrema",
)


@dataclass
class RetinaModelViewerData:
    """Validated data needed to view one completed Module 4 run."""

    run_dir: Path
    outputs_zarr_path: Path
    metadata_path: Path
    processing_settings_path: Path
    source_dataset_dir: Path
    source_zarr_path: Path
    outputs_root: object
    source_root: object
    imgs: object
    outs_fill: object
    extrema_mask: object
    metadata: pd.DataFrame
    response_keys: list[str]


def load_retina_model_viewer_data(run_dir):
    """Load and validate one completed Module 4 run for viewing."""

    run_dir = Path(run_dir)

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Retina-model run directory not found: {run_dir}"
        )

    outputs_zarr_path = run_dir / "retina_outputs.zarr"
    metadata_path = run_dir / "metadata.csv"
    processing_settings_path = run_dir / "processing_settings.json"

    if not outputs_zarr_path.exists():
        raise FileNotFoundError(
            f"Retinal outputs not found: {outputs_zarr_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {metadata_path}"
        )

    if not processing_settings_path.is_file():
        raise FileNotFoundError(
            f"Processing settings not found: {processing_settings_path}"
        )

    outputs_root = zarr.open_group(
        str(outputs_zarr_path),
        mode="r",
    )

    if outputs_root.attrs.get("complete") is not True:
        raise ValueError(
            "The Module 4 run is not marked complete."
        )

    for name in ("outs_fill", "extrema_mask"):
        if name not in outputs_root:
            raise ValueError(
                f"retina_outputs.zarr must contain a '{name}' array."
            )

    outs_fill = outputs_root["outs_fill"]
    extrema_mask = outputs_root["extrema_mask"]

    if len(outs_fill.shape) != 4:
        raise ValueError(
            "The 'outs_fill' array must have shape (N, K, H, W)."
        )

    if extrema_mask.shape != outs_fill.shape:
        raise ValueError(
            "'extrema_mask' and 'outs_fill' must have matching shapes."
        )

    if "RESPONSE_KEYS" not in outputs_root.attrs:
        raise ValueError(
            "retina_outputs.zarr is missing the RESPONSE_KEYS attribute."
        )

    response_keys = list(
        outputs_root.attrs["RESPONSE_KEYS"]
    )

    if len(response_keys) != outs_fill.shape[1]:
        raise ValueError(
            "RESPONSE_KEYS count does not match outs_fill.shape[1]."
        )

    metadata = pd.read_csv(metadata_path)

    if "stim_ID" not in metadata.columns:
        raise ValueError(
            "metadata.csv must contain a 'stim_ID' column."
        )

    if len(metadata) != outs_fill.shape[0]:
        raise ValueError(
            "metadata.csv row count does not match the Module 4 outputs."
        )

    settings = json.loads(
        processing_settings_path.read_text(
            encoding="utf-8"
        )
    )

    if "source_dataset" not in settings:
        raise ValueError(
            "processing_settings.json is missing 'source_dataset'."
        )

    source_dataset_dir = Path(
        settings["source_dataset"]
    )

    source_zarr_path = source_dataset_dir / "dataset.zarr"

    if not source_zarr_path.exists():
        raise FileNotFoundError(
            f"Source dataset Zarr store not found: {source_zarr_path}"
        )

    source_root = zarr.open_group(
        str(source_zarr_path),
        mode="r",
    )

    if "imgs" not in source_root:
        raise ValueError(
            "Source dataset.zarr must contain an 'imgs' array."
        )

    imgs = source_root["imgs"]

    if len(imgs.shape) != 4 or imgs.shape[-1] != 3:
        raise ValueError(
            "Source 'imgs' must have shape (N, H, W, 3)."
        )

    if imgs.shape[0] != outs_fill.shape[0]:
        raise ValueError(
            "Source images and Module 4 outputs must contain "
            "the same number of stimuli."
        )

    if imgs.shape[1:3] != outs_fill.shape[2:]:
        raise ValueError(
            "Source image dimensions must match Module 4 response maps."
        )

    return RetinaModelViewerData(
        run_dir=run_dir,
        outputs_zarr_path=outputs_zarr_path,
        metadata_path=metadata_path,
        processing_settings_path=processing_settings_path,
        source_dataset_dir=source_dataset_dir,
        source_zarr_path=source_zarr_path,
        outputs_root=outputs_root,
        source_root=source_root,
        imgs=imgs,
        outs_fill=outs_fill,
        extrema_mask=extrema_mask,
        metadata=metadata,
        response_keys=response_keys,
    )


def response_maps_for_type(data, map_type):
    """Return the saved response array for the requested viewer map type."""

    map_type = str(map_type).strip().lower()

    if map_type == "filled":
        return data.outs_fill

    if map_type == "extrema":
        return data.extrema_mask

    raise ValueError(
        f"Unknown map type: {map_type!r}. "
        f"Use one of {list(MAP_TYPES)}."
    )


def display_limits_for_type(map_type, maps, viewer_config):
    """Return appropriate color limits for one viewer map type."""

    map_type = str(map_type).strip().lower()

    if map_type == "extrema":
        return 0.0, 1.0

    if map_type != "filled":
        raise ValueError(
            f"Unknown map type: {map_type!r}. "
            f"Use one of {list(MAP_TYPES)}."
        )

    if viewer_config.use_fixed_scale:
        return (
            float(viewer_config.vmin),
            float(viewer_config.vmax),
        )

    maps = np.asarray(maps)
    max_abs = float(np.max(np.abs(maps)))

    if not np.isfinite(max_abs):
        raise ValueError(
            "Filled response maps contain non-finite values."
        )

    if max_abs == 0.0:
        return -1.0, 1.0

    return -max_abs, max_abs


def format_metadata_text(row):
    """Format all available metadata for one stimulus."""

    lines = []

    for column, value in row.items():
        if pd.isna(value):
            continue

        lines.append(
            f"{column}: {value}"
        )

    return "\n".join(lines)


def render_stimulus_figure(
    data,
    stim_id,
    map_type,
    viewer_config,
    figure=None,
):
    """Render one stimulus and all saved retinal-response channels."""

    stim_id = int(stim_id)

    if stim_id < 0 or stim_id >= data.imgs.shape[0]:
        raise IndexError(
            f"stim_ID {stim_id} is outside the available stimulus range."
        )

    rows = data.metadata[
        data.metadata["stim_ID"] == stim_id
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one metadata row for stim_ID={stim_id}."
        )

    row = rows.iloc[0]

    image = np.asarray(
        data.imgs[stim_id]
    )

    maps_z = response_maps_for_type(
        data,
        map_type,
    )

    maps = np.asarray(
        maps_z[stim_id]
    )

    vmin, vmax = display_limits_for_type(
        map_type,
        maps,
        viewer_config,
    )

    n_channels = maps.shape[0]
    n_cols = int(viewer_config.n_cols)
    n_rows = int(
        np.ceil(
            n_channels / float(n_cols)
        )
    )

    if figure is None:
        figure = Figure(
            figsize=(10.5, 10.0),
        )
    else:
        figure.clear()

    grid = figure.add_gridspec(
        n_rows + 1,
        n_cols,
        height_ratios=[1.2] + [1.0] * n_rows,
    )

    image_ax = figure.add_subplot(
        grid[0, :]
    )

    image_ax.imshow(
        image,
        origin="lower",
    )
    image_ax.axis("off")

    title_parts = [
        f"stim_ID={stim_id}",
    ]

    if (
        "true_label" in row.index
        and pd.notna(row["true_label"])
    ):
        title_parts.append(
            f"true_label={row['true_label']}"
        )

    if (
        "bg_hue" in row.index
        and pd.notna(row["bg_hue"])
    ):
        title_parts.append(
            f"bg_hue={row['bg_hue']}"
        )

    image_ax.set_title(
        "   ".join(title_parts),
        fontsize=10,
    )

    response_axes = []
    last_image = None

    for channel in range(n_channels):
        row_index = channel // n_cols
        col_index = channel % n_cols

        ax = figure.add_subplot(
            grid[1 + row_index, col_index]
        )

        last_image = ax.imshow(
            maps[channel],
            origin="lower",
            cmap=viewer_config.resp_cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(
            str(data.response_keys[channel]),
            fontsize=9,
        )
        ax.axis("off")

        response_axes.append(ax)

    for empty_index in range(
        n_channels,
        n_rows * n_cols,
    ):
        row_index = empty_index // n_cols
        col_index = empty_index % n_cols

        ax = figure.add_subplot(
            grid[1 + row_index, col_index]
        )
        ax.axis("off")

    if last_image is not None:
        if str(map_type).strip().lower() == "filled":
            colorbar_label = "outs_fill (a.u.)"
        else:
            colorbar_label = "extrema_mask (0/1)"

        figure.colorbar(
            last_image,
            ax=response_axes,
            fraction=0.02,
            pad=0.02,
            label=colorbar_label,
        )

    figure.subplots_adjust(
        right=0.88,
        hspace=0.30,
    )

    return figure


def metadata_filter_values(metadata, column):
    """Return sorted non-missing values available for an optional filter."""

    if column not in metadata.columns:
        return []

    values = (
        metadata[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values)


def initial_filter_value(values, preferred):
    """Choose the configured default when available, otherwise the first value."""

    if not values:
        return None

    if preferred in values:
        return preferred

    return values[0]


def filter_stimuli(metadata, true_label=None, bg_hue=None):
    """Filter metadata using whichever optional viewer filters are active."""

    selected = metadata.copy()

    if true_label is not None and "true_label" in selected.columns:
        selected = selected[
            selected["true_label"].astype(str) == str(true_label)
        ]

    if bg_hue is not None and "bg_hue" in selected.columns:
        selected = selected[
            selected["bg_hue"].astype(str) == str(bg_hue)
        ]

    return selected.sort_values("stim_ID").reset_index(drop=True)