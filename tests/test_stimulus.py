from dataclasses import asdict

import numpy as np
import zarr

from retinamodel.stimulus import (
    RESPONSE_KEYS,
    StimulusConfig,
    build_metadata_table,
    build_stimulus_candidates,
    hsv_delta,
    truncated_delta_grid,
    export_retina_dataset,
)


def make_small_config():
    """Return a small configuration for fast automated tests."""
    config = StimulusConfig.from_toml("config/stimulus.toml")
    values = asdict(config)

    values.update({
        "image_size_px": 30,
        "obj_gray_n": 1,
        "obj_chrom_n_int": 1,
        "obj_chrom_n_sat": 1,
        "bg_gray_n_int": 1,
        "bg_chrom_n_int": 1,
        "bg_chrom_n_sat": 1,
    })

    return StimulusConfig(**values)


def test_response_keys_order():
    assert RESPONSE_KEYS == (
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


def test_truncated_delta_grid_stays_within_bounds():
    values, deltas = truncated_delta_grid(0.3, -0.4, 0.4, 5)

    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    np.testing.assert_allclose(values, 0.3 + deltas)


def test_hsv_delta_wraps_hue():
    result = hsv_delta(
        np.array([0.05, 0.8, 0.7]),
        np.array([0.95, 0.5, 0.4]),
    )

    np.testing.assert_allclose(result, [0.1, 0.3, 0.3])


def test_candidate_summary_is_consistent():
    config = make_small_config()
    candidates, summary = build_stimulus_candidates(config)

    assert len(candidates) == summary["n_total_unique"]
    assert summary["n_total_unique"] <= summary["n_total_raw"]
    assert summary["n_redundant"] == (
        summary["n_total_raw"] - summary["n_total_unique"]
    )
    assert sum(summary["raw_label_counts"].values()) == summary["n_total_raw"]
    assert sum(summary["unique_label_counts"].values()) == summary["n_total_unique"]


def test_metadata_table_matches_candidates():
    config = make_small_config()
    candidates, _ = build_stimulus_candidates(config)
    metadata = build_metadata_table(candidates)

    assert len(metadata) == len(candidates)
    assert metadata["stim_ID"].tolist() == list(range(len(candidates)))
    assert metadata.loc[0, "true_label"] == candidates[0]["true_label"]

def test_export_retina_dataset(tmp_path):
    config = make_small_config()
    result = export_retina_dataset(config, tmp_path, "test")

    root = zarr.open_group(result["zarr_path"], mode="r")
    n_stimuli = result["summary"]["n_total_unique"]

    assert result["zarr_path"].exists()
    assert result["metadata_path"].exists()
    assert len(result["metadata"]) == n_stimuli
    assert root["imgs"].shape == (n_stimuli, 30, 30, 3)
    assert root["outs"].shape == (n_stimuli, len(RESPONSE_KEYS), 30, 30)
    assert tuple(root.attrs["RESPONSE_KEYS"]) == RESPONSE_KEYS