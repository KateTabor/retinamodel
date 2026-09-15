"""Tests for Module 4 retinal-response viewer helpers."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import zarr

from retinamodel.retina_model_viewer import (
    MAP_TYPES,
    display_limits_for_type,
    filter_stimuli,
    initial_filter_value,
    load_retina_model_viewer_data,
    metadata_filter_values,
    response_maps_for_type,
    format_metadata_text,
    render_stimulus_figure,
)


def _make_viewer_run(
    tmp_path,
    complete=True,
    include_source_images=True,
):
    """Create a tiny completed Module 4-style run for viewer tests."""

    source_dataset = tmp_path / "retina_dataset_test"
    source_dataset.mkdir()

    source_root = zarr.open_group(
        str(source_dataset / "dataset.zarr"),
        mode="w",
    )

    if include_source_images:
        imgs = source_root.create_array(
            "imgs",
            shape=(2, 4, 4, 3),
            dtype="f4",
        )
        imgs[:] = 0.5

    run_dir = tmp_path / "retina_model_test"
    run_dir.mkdir()

    outputs_root = zarr.open_group(
        str(run_dir / "retina_outputs.zarr"),
        mode="w",
    )

    outs_fill = outputs_root.create_array(
        "outs_fill",
        shape=(2, 2, 4, 4),
        dtype="f4",
    )
    extrema_mask = outputs_root.create_array(
        "extrema_mask",
        shape=(2, 2, 4, 4),
        dtype="u1",
    )

    outs_fill[:] = 1.0
    extrema_mask[:] = 0
    extrema_mask[:, :, 1, 1] = 1

    outputs_root.attrs["complete"] = complete
    outputs_root.attrs["RESPONSE_KEYS"] = [
        "response_a",
        "response_b",
    ]

    pd.DataFrame(
        {
            "stim_ID": [0, 1],
            "true_label": ["green", "red"],
            "bg_hue": ["gray", "blue"],
        }
    ).to_csv(
        run_dir / "metadata.csv",
        index=False,
    )

    settings = {
        "source_dataset": str(source_dataset.resolve()),
    }

    (run_dir / "processing_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )

    return run_dir, source_dataset


def test_metadata_filter_values():
    """Return sorted unique non-missing filter values."""

    metadata = pd.DataFrame(
        {
            "true_label": [
                "green",
                "red",
                np.nan,
                "green",
            ]
        }
    )

    values = metadata_filter_values(
        metadata,
        "true_label",
    )

    assert values == ["green", "red"]


def test_metadata_filter_values_missing_column():
    """Missing optional metadata produces no filter values."""

    metadata = pd.DataFrame(
        {
            "stim_ID": [0, 1],
        }
    )

    assert metadata_filter_values(
        metadata,
        "true_label",
    ) == []


def test_initial_filter_value():
    """Prefer the configured value when present."""

    values = ["blue", "green", "red"]

    assert initial_filter_value(
        values,
        "green",
    ) == "green"

    assert initial_filter_value(
        values,
        "yellow",
    ) == "blue"

    assert initial_filter_value(
        [],
        "green",
    ) is None


def test_filter_stimuli_with_both_filters():
    """Apply object and background filters together."""

    metadata = pd.DataFrame(
        {
            "stim_ID": [2, 0, 1],
            "true_label": [
                "green",
                "green",
                "red",
            ],
            "bg_hue": [
                "gray",
                "blue",
                "gray",
            ],
        }
    )

    selected = filter_stimuli(
        metadata,
        true_label="green",
        bg_hue="gray",
    )

    assert selected["stim_ID"].tolist() == [2]


def test_filter_stimuli_without_optional_columns():
    """Datasets without object/background metadata remain browsable."""

    metadata = pd.DataFrame(
        {
            "stim_ID": [2, 0, 1],
            "image_name": [
                "c.png",
                "a.png",
                "b.png",
            ],
        }
    )

    selected = filter_stimuli(
        metadata,
        true_label="green",
        bg_hue="gray",
    )

    assert selected["stim_ID"].tolist() == [0, 1, 2]


def test_load_retina_model_viewer_data(tmp_path):
    """Load outputs, metadata, and original stimulus images."""

    run_dir, source_dataset = _make_viewer_run(
        tmp_path
    )

    data = load_retina_model_viewer_data(
        run_dir
    )

    assert data.run_dir == run_dir
    assert data.source_dataset_dir == source_dataset.resolve()

    assert data.imgs.shape == (2, 4, 4, 3)
    assert data.outs_fill.shape == (2, 2, 4, 4)
    assert data.extrema_mask.shape == (2, 2, 4, 4)

    assert data.response_keys == [
        "response_a",
        "response_b",
    ]

    assert data.metadata["stim_ID"].tolist() == [0, 1]

    np.testing.assert_array_equal(
        data.imgs[0],
        np.full(
            (4, 4, 3),
            0.5,
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        data.extrema_mask[0, 0],
        np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_viewer_loader_rejects_incomplete_run(tmp_path):
    """Do not view a Module 4 run that did not finish building."""

    run_dir, _ = _make_viewer_run(
        tmp_path,
        complete=False,
    )

    with pytest.raises(
        ValueError,
        match="not marked complete",
    ):
        load_retina_model_viewer_data(
            run_dir
        )


def test_viewer_loader_requires_source_images(tmp_path):
    """The viewer always requires the original stimulus images."""

    run_dir, _ = _make_viewer_run(
        tmp_path,
        include_source_images=False,
    )

    with pytest.raises(
        ValueError,
        match="imgs",
    ):
        load_retina_model_viewer_data(
            run_dir
        )


def test_viewer_map_types():
    """Check the supported viewer map types."""

    assert MAP_TYPES == (
        "filled",
        "extrema",
    )


def test_response_maps_for_type(tmp_path):
    """Select either saved Module 4 map array."""

    run_dir, _ = _make_viewer_run(
        tmp_path
    )

    data = load_retina_model_viewer_data(
        run_dir
    )

    filled = response_maps_for_type(
        data,
        "filled",
    )
    extrema = response_maps_for_type(
        data,
        "extrema",
    )

    assert filled is data.outs_fill
    assert extrema is data.extrema_mask

    assert filled.shape == (2, 2, 4, 4)
    assert extrema.shape == (2, 2, 4, 4)


def test_response_maps_for_type_rejects_unknown_type(tmp_path):
    """Reject unsupported viewer map types."""

    run_dir, _ = _make_viewer_run(
        tmp_path
    )

    data = load_retina_model_viewer_data(
        run_dir
    )

    with pytest.raises(
        ValueError,
        match="Unknown map type",
    ):
        response_maps_for_type(
            data,
            "raw",
        )


def test_display_limits_filled_fixed_scale():
    """Filled responses use configured limits when fixed scaling is enabled."""

    config = SimpleNamespace(
        use_fixed_scale=True,
        vmin=-5e7,
        vmax=5e7,
    )

    maps = np.array(
        [
            [-10.0, 20.0],
            [30.0, -40.0],
        ]
    )

    limits = display_limits_for_type(
        "filled",
        maps,
        config,
    )

    assert limits == (-5e7, 5e7)


def test_display_limits_filled_auto_scale():
    """Filled responses auto-scale symmetrically when requested."""

    config = SimpleNamespace(
        use_fixed_scale=False,
        vmin=-5e7,
        vmax=5e7,
    )

    maps = np.array(
        [
            [-3.0, 2.0],
            [1.0, 5.0],
        ]
    )

    limits = display_limits_for_type(
        "filled",
        maps,
        config,
    )

    assert limits == (-5.0, 5.0)


def test_display_limits_extrema_are_binary():
    """Extrema masks always use their binary 0-to-1 range."""

    config = SimpleNamespace(
        use_fixed_scale=True,
        vmin=-5e7,
        vmax=5e7,
    )

    maps = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.uint8,
    )

    limits = display_limits_for_type(
        "extrema",
        maps,
        config,
    )

    assert limits == (0.0, 1.0)


def test_display_limits_reject_nonfinite_filled_maps():
    """Auto-scaling should not silently display invalid response values."""

    config = SimpleNamespace(
        use_fixed_scale=False,
        vmin=-5e7,
        vmax=5e7,
    )

    maps = np.array(
        [
            [0.0, np.nan],
        ]
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        display_limits_for_type(
            "filled",
            maps,
            config,
        )


def test_format_metadata_text():
    """Display available metadata while omitting missing values."""

    row = pd.Series(
        {
            "stim_ID": 4,
            "true_label": "green",
            "notes": np.nan,
        }
    )

    text = format_metadata_text(
        row
    )

    assert "stim_ID: 4" in text
    assert "true_label: green" in text
    assert "notes" not in text


def test_render_stimulus_figure(tmp_path):
    """Render the original stimulus and every response channel."""

    run_dir, _ = _make_viewer_run(
        tmp_path
    )

    data = load_retina_model_viewer_data(
        run_dir
    )

    config = SimpleNamespace(
        resp_cmap="viridis",
        use_fixed_scale=True,
        vmin=-5.0,
        vmax=5.0,
        n_cols=2,
        dpi=100,
    )

    figure = render_stimulus_figure(
        data=data,
        stim_id=0,
        map_type="filled",
        viewer_config=config,
    )

    titles = [
        ax.get_title()
        for ax in figure.axes
    ]

    assert any(
        "stim_ID=0" in title
        for title in titles
    )

    assert "response_a" in titles
    assert "response_b" in titles