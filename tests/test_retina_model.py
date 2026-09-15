"""Tests for Module 4 retinal-model configuration and architecture."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import distance_transform_edt, gaussian_filter
import zarr

from retinamodel.retina_model import (
    ExtremaConfig,
    FillConfig,
    ViewerConfig,
    create_retina_model_run_paths,
    load_retina_input_dataset,
    load_retina_model_config,
    write_processing_settings,
    compute_extrema_mask,
    ridge_valley_mask,
    compute_filled_response,
    build_retina_model,
    estimate_retina_model_storage,
)


RESPONSE_KEYS = [
    "l_on",
    "l_off",
    "m_on",
    "m_off",
    "l_h2_on",
    "m_h2_on",
    "l_h2_off",
    "m_h2_off",
    "s_on",
]


def _make_module2_dataset(tmp_path):
    """Create a tiny valid Module 2-style dataset."""

    dataset_dir = tmp_path / "retina_dataset_test"
    dataset_dir.mkdir()

    zarr_path = dataset_dir / "dataset.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")

    imgs = root.create_array(
        "imgs",
        shape=(2, 4, 4, 3),
        dtype="f4",
    )
    outs = root.create_array(
        "outs",
        shape=(2, 9, 4, 4),
        dtype="f4",
    )

    imgs[:] = 0.5
    outs[:] = 1.0

    root.attrs["RESPONSE_KEYS"] = RESPONSE_KEYS

    metadata = pd.DataFrame(
        {
            "stim_ID": [0, 1],
            "true_label": ["red", "green"],
            "bg_hue": ["gray", "blue"],
            "obj_r": [0.7, 0.2],
            "obj_g": [0.2, 0.7],
            "obj_b": [0.2, 0.2],
            "bg_r": [0.3, 0.2],
            "bg_g": [0.3, 0.2],
            "bg_b": [0.3, 0.6],
            "obj_int": [0.7, 0.7],
            "obj_sat": [0.7, 0.7],
            "bg_int": [0.3, 0.6],
            "bg_sat": [0.0, 0.7],
            "delta_int": [-0.4, -0.1],
            "delta_sat": [-0.7, 0.0],
        }
    )

    metadata.to_csv(
        dataset_dir / "metadata.csv",
        index=False,
    )

    return dataset_dir


def test_load_retina_model_config_defaults():
    """Check the repository Module 4 defaults."""

    config = load_retina_model_config()

    assert config.extrema.sigma_px == pytest.approx(0.5)
    assert config.extrema.amp_thresh == pytest.approx(0.0)
    assert config.extrema.grad_tol_frac == pytest.approx(0.7)
    assert config.extrema.curv_tol_frac == pytest.approx(0.20)

    assert config.fill.fillradius is None
    assert config.fill.beyondradius is False

    assert config.viewer.obj_label == "green"
    assert config.viewer.bg_label == "gray"
    assert config.viewer.resp_cmap == "berlin"
    assert config.viewer.use_fixed_scale is True
    assert config.viewer.vmin == pytest.approx(-5e7)
    assert config.viewer.vmax == pytest.approx(5e7)
    assert config.viewer.n_cols == 3
    assert config.viewer.dpi == 200


def test_fill_config_rejects_invalid_radius():
    """Check fillradius must be None or a positive integer."""

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        FillConfig(
            fillradius=0,
            beyondradius=False,
        )


def test_extrema_config_rejects_negative_sigma():
    """Check Gaussian sigma cannot be negative."""

    with pytest.raises(
        ValueError,
        match="sigma_px",
    ):
        ExtremaConfig(
            sigma_px=-0.5,
            amp_thresh=0.0,
            grad_tol_frac=0.7,
            curv_tol_frac=0.20,
        )


def test_viewer_config_rejects_reversed_scale():
    """Check fixed-scale limits are ordered."""

    with pytest.raises(
        ValueError,
        match="vmin",
    ):
        ViewerConfig(
            obj_label="red",
            bg_label="gray",
            resp_cmap="berlin",
            use_fixed_scale=True,
            vmin=5.0,
            vmax=-5.0,
            n_cols=3,
            dpi=200,
        )


def test_load_retina_input_dataset(tmp_path):
    """Check a valid Module 2 dataset loads correctly."""

    dataset_dir = _make_module2_dataset(
        tmp_path
    )

    dataset = load_retina_input_dataset(
        dataset_dir
    )

    assert dataset.imgs.shape == (2, 4, 4, 3)
    assert dataset.outs.shape == (2, 9, 4, 4)
    assert dataset.response_keys == RESPONSE_KEYS
    assert len(dataset.metadata) == 2


def test_input_dataset_rejects_response_key_mismatch(tmp_path):
    """Check channel names must match the response array."""

    dataset_dir = _make_module2_dataset(
        tmp_path
    )

    root = zarr.open_group(
        str(dataset_dir / "dataset.zarr"),
        mode="a",
    )
    root.attrs["RESPONSE_KEYS"] = RESPONSE_KEYS[:-1]

    with pytest.raises(
        ValueError,
        match="RESPONSE_KEYS",
    ):
        load_retina_input_dataset(
            dataset_dir
        )


def test_create_retina_model_run_paths(tmp_path):
    """Check the agreed Module 4 output structure."""

    paths = create_retina_model_run_paths(
        base_dir=tmp_path,
        run_tag="20260914_120000",
    )

    assert paths.run_dir.name == "retina_model_20260914_120000"
    assert paths.run_dir.is_dir()
    assert paths.outputs_zarr.name == "retina_outputs.zarr"
    assert paths.metadata_csv.name == "metadata.csv"
    assert paths.processing_settings_json.name == "processing_settings.json"


def test_write_processing_settings(tmp_path):
    """Check scientific settings and source provenance are recorded."""

    dataset_dir = _make_module2_dataset(
        tmp_path
    )
    dataset = load_retina_input_dataset(
        dataset_dir
    )
    config = load_retina_model_config()

    paths = create_retina_model_run_paths(
        base_dir=tmp_path / "runs",
        run_tag="20260914_120000",
    )

    settings = write_processing_settings(
        paths,
        dataset,
        config,
    )

    saved = json.loads(
        paths.processing_settings_json.read_text(
            encoding="utf-8"
        )
    )

    assert saved == settings
    assert saved["extrema"]["sigma_px"] == pytest.approx(0.5)
    assert saved["fill"]["fillradius"] is None
    assert saved["response_keys"] == RESPONSE_KEYS

    assert "viewer" not in saved


def _notebook_ridge_valley_mask(
    A,
    amp_thresh=0.0,
    grad_tol_frac=0.02,
    curv_tol_frac=0.10,
):
    """Direct reference implementation copied from notebook Cell 19."""

    A = np.asarray(A, dtype=float)

    gy, gx = np.gradient(A)
    gxx = np.gradient(gx, axis=1)
    gyy = np.gradient(gy, axis=0)
    gxy = np.gradient(gx, axis=0)

    tr = gxx + gyy
    det = gxx * gyy - gxy * gxy
    tmp = np.sqrt(
        np.maximum(
            0.0,
            (0.5 * tr) ** 2 - det,
        )
    )

    lam1 = 0.5 * tr - tmp
    lam2 = 0.5 * tr + tmp

    vx1 = gxy
    vy1 = lam1 - gxx

    nrm = np.hypot(vx1, vy1) + 1e-12
    vx1 /= nrm
    vy1 /= nrm

    d1 = gx * vx1 + gy * vy1

    grad_ref = np.percentile(np.abs(d1), 99)
    curv_ref = np.percentile(np.abs(lam1), 99)

    grad_tol = grad_tol_frac * grad_ref
    curv_tol = curv_tol_frac * curv_ref

    ridge_max = (
        (np.abs(d1) < grad_tol)
        & (lam1 < -curv_tol)
        & (np.abs(A) > amp_thresh)
    )

    vx2 = -vy1
    vy2 = vx1
    d2 = gx * vx2 + gy * vy2

    valley_min = (
        (np.abs(d2) < grad_tol)
        & (lam2 > curv_tol)
        & (np.abs(A) > amp_thresh)
    )

    return ridge_max | valley_min


def test_ridge_valley_mask_matches_notebook():
    """Check the extracted calculation exactly matches Cell 19."""

    rng = np.random.default_rng(14)
    response = rng.normal(
        size=(31, 31)
    ).astype(np.float32)

    expected = _notebook_ridge_valley_mask(
        response,
        amp_thresh=0.0,
        grad_tol_frac=0.7,
        curv_tol_frac=0.20,
    )

    actual = ridge_valley_mask(
        response,
        amp_thresh=0.0,
        grad_tol_frac=0.7,
        curv_tol_frac=0.20,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_compute_extrema_mask_matches_notebook_pipeline():
    """Check smoothing plus extrema detection matches Cell 19."""

    rng = np.random.default_rng(14)
    response = rng.normal(
        size=(31, 31)
    ).astype(np.float32)

    config = ExtremaConfig(
        sigma_px=0.5,
        amp_thresh=0.0,
        grad_tol_frac=0.7,
        curv_tol_frac=0.20,
    )

    smoothed = gaussian_filter(
        response,
        sigma=0.5,
        mode="reflect",
    )

    expected = _notebook_ridge_valley_mask(
        smoothed,
        amp_thresh=0.0,
        grad_tol_frac=0.7,
        curv_tol_frac=0.20,
    )

    actual = compute_extrema_mask(
        response,
        config,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_compute_extrema_mask_shape_and_dtype():
    """Check extrema masks remain Boolean maps matching the input shape."""

    response = np.zeros(
        (15, 17),
        dtype=np.float32,
    )

    config = ExtremaConfig(
        sigma_px=0.5,
        amp_thresh=0.0,
        grad_tol_frac=0.7,
        curv_tol_frac=0.20,
    )

    mask = compute_extrema_mask(
        response,
        config,
    )

    assert mask.shape == response.shape
    assert mask.dtype == np.bool_


def _notebook_filled_response(
    response,
    seed_mask,
    fillradius=None,
    beyondradius=False,
):
    """Direct reference implementation copied from notebook Cell 20."""

    A = np.asarray(response, dtype=float)
    seed_mask = np.asarray(seed_mask, dtype=bool)

    if not np.any(seed_mask):
        return np.zeros_like(A, dtype=np.float32)

    dist_map, inds = distance_transform_edt(
        ~seed_mask,
        return_indices=True,
    )

    nearest_vals = A[
        inds[0],
        inds[1],
    ]

    fill_mask = ~seed_mask

    if fillradius is not None:
        fill_mask = (
            fill_mask
            & (dist_map <= float(fillradius))
        )

    filled = A.copy()
    filled[fill_mask] = nearest_vals[fill_mask]

    if (
        fillradius is not None
        and not beyondradius
    ):
        beyond = (
            (~seed_mask)
            & (dist_map > float(fillradius))
        )
        filled[beyond] = 0.0

    return filled.astype(np.float32)


def test_filled_response_matches_notebook_fill_everywhere():
    """Check default nearest-extrema filling matches Cell 20."""

    response = np.arange(
        25,
        dtype=np.float32,
    ).reshape(5, 5)

    seed_mask = np.zeros(
        (5, 5),
        dtype=bool,
    )
    seed_mask[1, 1] = True
    seed_mask[3, 3] = True

    config = FillConfig(
        fillradius=None,
        beyondradius=False,
    )

    expected = _notebook_filled_response(
        response,
        seed_mask,
        fillradius=None,
        beyondradius=False,
    )

    actual = compute_filled_response(
        response,
        seed_mask,
        config,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "beyondradius",
    [False, True],
)
def test_filled_response_matches_notebook_finite_radius(
    beyondradius,
):
    """Check both beyond-radius behaviors match Cell 20."""

    response = np.arange(
        49,
        dtype=np.float32,
    ).reshape(7, 7)

    seed_mask = np.zeros(
        (7, 7),
        dtype=bool,
    )
    seed_mask[3, 3] = True

    config = FillConfig(
        fillradius=2,
        beyondradius=beyondradius,
    )

    expected = _notebook_filled_response(
        response,
        seed_mask,
        fillradius=2,
        beyondradius=beyondradius,
    )

    actual = compute_filled_response(
        response,
        seed_mask,
        config,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_filled_response_no_seeds_returns_zeros():
    """Preserve the notebook convention for maps with no extrema."""

    response = np.arange(
        25,
        dtype=np.float32,
    ).reshape(5, 5)

    seed_mask = np.zeros(
        (5, 5),
        dtype=bool,
    )

    config = FillConfig(
        fillradius=None,
        beyondradius=False,
    )

    filled = compute_filled_response(
        response,
        seed_mask,
        config,
    )

    np.testing.assert_array_equal(
        filled,
        np.zeros_like(response),
    )

    assert filled.dtype == np.float32


def test_estimate_retina_model_storage(tmp_path):
    """Check storage estimates include only retained Module 4 arrays."""

    dataset_dir = _make_module2_dataset(
        tmp_path
    )
    dataset = load_retina_input_dataset(
        dataset_dir
    )

    sizes = estimate_retina_model_storage(
        dataset
    )

    n_values = 2 * 9 * 4 * 4

    assert sizes["extrema_mask_bytes"] == n_values
    assert sizes["outs_fill_bytes"] == n_values * 4
    assert sizes["total_bytes"] == n_values * 5


def test_build_retina_model_outputs(tmp_path):
    """Check a complete Module 4 run has the agreed structure."""

    dataset_dir = _make_module2_dataset(
        tmp_path
    )

    paths = build_retina_model(
        dataset_dir,
        base_dir=tmp_path / "runs",
        run_tag="20260914_120000",
    )

    assert paths.run_dir.is_dir()
    assert paths.outputs_zarr.exists()
    assert paths.metadata_csv.is_file()
    assert paths.processing_settings_json.is_file()

    root = zarr.open_group(
        str(paths.outputs_zarr),
        mode="r",
    )

    assert set(root.array_keys()) == {
        "extrema_mask",
        "outs_fill",
    }

    assert root["extrema_mask"].shape == (
        2,
        9,
        4,
        4,
    )
    assert root["outs_fill"].shape == (
        2,
        9,
        4,
        4,
    )

    assert root["extrema_mask"].dtype == np.dtype(
        "u1"
    )
    assert root["outs_fill"].dtype == np.dtype(
        "f4"
    )

    assert root.attrs["complete"] is True
    assert list(
        root.attrs["RESPONSE_KEYS"]
    ) == RESPONSE_KEYS

    assert (
        paths.metadata_csv.read_bytes()
        == (dataset_dir / "metadata.csv").read_bytes()
    )


def test_build_retina_model_matches_direct_calculation(tmp_path):
    """Check saved Module 4 arrays match the validated calculations."""

    dataset_dir = _make_module2_dataset(
        tmp_path
    )

    paths = build_retina_model(
        dataset_dir,
        base_dir=tmp_path / "runs",
        run_tag="20260914_120001",
    )

    config = load_retina_model_config()

    source = zarr.open_group(
        str(dataset_dir / "dataset.zarr"),
        mode="r",
    )
    result = zarr.open_group(
        str(paths.outputs_zarr),
        mode="r",
    )

    response = source["outs"][0, 0]

    expected_mask = compute_extrema_mask(
        response,
        config.extrema,
    )
    expected_fill = compute_filled_response(
        response,
        expected_mask,
        config.fill,
    )

    np.testing.assert_array_equal(
        result["extrema_mask"][0, 0],
        expected_mask.astype(np.uint8),
    )

    np.testing.assert_array_equal(
        result["outs_fill"][0, 0],
        expected_fill,
    )

    assert "derived" not in source
    assert "extrema_mask" not in source
    assert "outs_fill" not in source


def test_viewer_config_allows_dataset_specific_labels():
    """Viewer filter defaults are not restricted to synthetic color labels."""

    config = ViewerConfig(
        obj_label="banana",
        bg_label="natural_scene",
        resp_cmap="berlin",
        use_fixed_scale=True,
        vmin=-5e7,
        vmax=5e7,
        n_cols=3,
        dpi=200,
    )

    assert config.obj_label == "banana"
    assert config.bg_label == "natural_scene"


def test_input_dataset_allows_minimal_metadata(tmp_path):
    """Module 4 does not require synthetic object/background metadata."""

    dataset_dir = tmp_path / "retina_dataset_minimal"
    dataset_dir.mkdir()

    root = zarr.open_group(
        str(dataset_dir / "dataset.zarr"),
        mode="w",
    )

    imgs = root.create_array(
        "imgs",
        shape=(2, 4, 4, 3),
        dtype="f4",
    )
    outs = root.create_array(
        "outs",
        shape=(2, 2, 4, 4),
        dtype="f4",
    )

    imgs[:] = 0.0
    outs[:] = 0.0

    root.attrs["RESPONSE_KEYS"] = [
        "response_a",
        "response_b",
    ]

    pd.DataFrame(
        {
            "stim_ID": [0, 1],
        }
    ).to_csv(
        dataset_dir / "metadata.csv",
        index=False,
    )

    dataset = load_retina_input_dataset(dataset_dir)

    assert dataset.imgs.shape == (2, 4, 4, 3)
    assert dataset.outs.shape == (2, 2, 4, 4)
    assert dataset.metadata.columns.tolist() == ["stim_ID"]
    assert dataset.response_keys == [
        "response_a",
        "response_b",
    ]


def test_input_dataset_requires_images(tmp_path):
    """A Module 4 input dataset must contain original stimulus images."""

    dataset_dir = tmp_path / "retina_dataset_no_images"
    dataset_dir.mkdir()

    root = zarr.open_group(
        str(dataset_dir / "dataset.zarr"),
        mode="w",
    )

    outs = root.create_array(
        "outs",
        shape=(2, 2, 4, 4),
        dtype="f4",
    )
    outs[:] = 0.0

    root.attrs["RESPONSE_KEYS"] = [
        "response_a",
        "response_b",
    ]

    pd.DataFrame(
        {
            "stim_ID": [0, 1],
        }
    ).to_csv(
        dataset_dir / "metadata.csv",
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="imgs",
    ):
        load_retina_input_dataset(dataset_dir)


def test_viewer_config_allows_omitted_filter_defaults(tmp_path):
    """Viewer object/background defaults may be omitted from TOML."""

    config_path = tmp_path / "retina_model.toml"

    config_path.write_text(
        """
[extrema]
SIGMA_PX = 0.5
AMP_THRESH = 0.0
GRAD_TOL_FRAC = 0.7
CURV_TOL_FRAC = 0.20

[fill]
FILLRADIUS = "none"
BEYONDRADIUS = false

[viewer]
RESP_CMAP = "berlin"
USE_FIXED_SCALE = true
VMIN = -5e7
VMAX = 5e7
N_COLS = 3
DPI = 200
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_retina_model_config(config_path)

    assert config.viewer.obj_label is None
    assert config.viewer.bg_label is None