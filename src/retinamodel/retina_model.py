"""Configuration and dataset architecture for the completed retinal model."""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import tomllib

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter
from tqdm.auto import tqdm
import zarr


REQUIRED_METADATA_COLUMNS = (
    "stim_ID",
)


@dataclass
class ExtremaConfig:
    """User-adjustable settings for detecting retinal response extrema."""

    sigma_px: float
    amp_thresh: float
    grad_tol_frac: float
    curv_tol_frac: float

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Check extrema settings."""

        if self.sigma_px < 0:
            raise ValueError("sigma_px must be >= 0.")

        if self.amp_thresh < 0:
            raise ValueError("amp_thresh must be >= 0.")

        if self.grad_tol_frac < 0:
            raise ValueError("grad_tol_frac must be >= 0.")

        if self.curv_tol_frac < 0:
            raise ValueError("curv_tol_frac must be >= 0.")


@dataclass
class FillConfig:
    """User-adjustable settings for nearest-extrema response filling."""

    fillradius: int | None
    beyondradius: bool

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Check fill settings."""

        if self.fillradius is not None:
            if type(self.fillradius) is not int or self.fillradius <= 0:
                raise ValueError(
                    "fillradius must be None or a positive integer number of pixels."
                )

        if type(self.beyondradius) is not bool:
            raise ValueError("beyondradius must be true or false.")


@dataclass
class ViewerConfig:
    """User-adjustable settings for viewing completed retinal responses."""

    obj_label: str | None
    bg_label: str | None
    resp_cmap: str
    use_fixed_scale: bool
    vmin: float
    vmax: float
    n_cols: int
    dpi: int

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Check viewer settings."""

        if self.obj_label is not None:
            if not isinstance(self.obj_label, str) or not self.obj_label.strip():
                raise ValueError("viewer.obj_label must be a non-empty string or None.")

        if self.bg_label is not None:
            if not isinstance(self.bg_label, str) or not self.bg_label.strip():
                raise ValueError("viewer.bg_label must be a non-empty string or None.")

        if not isinstance(self.resp_cmap, str) or not self.resp_cmap.strip():
            raise ValueError("resp_cmap must be a non-empty string.")

        if type(self.use_fixed_scale) is not bool:
            raise ValueError("use_fixed_scale must be true or false.")

        if self.vmin >= self.vmax:
            raise ValueError("vmin must be smaller than vmax.")

        if type(self.n_cols) is not int or self.n_cols <= 0:
            raise ValueError("n_cols must be a positive integer.")

        if type(self.dpi) is not int or self.dpi <= 0:
            raise ValueError("dpi must be a positive integer.")


@dataclass
class RetinaModelConfig:
    """Complete Module 4 configuration."""

    extrema: ExtremaConfig
    fill: FillConfig
    viewer: ViewerConfig


@dataclass
class RetinaInputDataset:
    """Validated Module 2 dataset used as input to Module 4."""

    dataset_dir: Path
    zarr_path: Path
    metadata_path: Path
    root: object
    imgs: object
    outs: object
    metadata: pd.DataFrame
    response_keys: list[str]


@dataclass
class RetinaModelRunPaths:
    """Paths belonging to one Module 4 retinal-model run."""

    run_dir: Path
    outputs_zarr: Path
    metadata_csv: Path
    processing_settings_json: Path


def _lowercase_keys(values):
    """Return a dictionary with lowercase string keys."""

    return {
        str(key).lower(): value
        for key, value in values.items()
    }


def _parse_fillradius(value):
    """Convert the TOML fillradius setting to Python None or int."""

    if isinstance(value, str) and value.strip().lower() == "none":
        return None

    if type(value) is int and value > 0:
        return value

    raise ValueError(
        'FILLRADIUS must be "none" or a positive integer number of pixels.'
    )


def load_retina_model_config(path="config/retina_model.toml"):
    """Load and validate Module 4 TOML configuration."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Retina-model configuration not found: {path}"
        )

    with path.open("rb") as file:
        values = tomllib.load(file)

    required_sections = ("extrema", "fill", "viewer")
    missing_sections = [
        section
        for section in required_sections
        if section not in values
    ]

    if missing_sections:
        raise ValueError(
            "retina_model.toml is missing section(s): "
            + ", ".join(missing_sections)
        )

    extrema_values = _lowercase_keys(values["extrema"])
    fill_values = _lowercase_keys(values["fill"])
    viewer_values = _lowercase_keys(values["viewer"])
    viewer_values.setdefault("obj_label", None)
    viewer_values.setdefault("bg_label", None)

    fill_values["fillradius"] = _parse_fillradius(
        fill_values["fillradius"]
    )

    config = RetinaModelConfig(
        extrema=ExtremaConfig(**extrema_values),
        fill=FillConfig(**fill_values),
        viewer=ViewerConfig(**viewer_values),
    )

    return config


def ridge_valley_mask(
    response,
    amp_thresh=0.0,
    grad_tol_frac=0.02,
    curv_tol_frac=0.10,
):
    """Return the notebook ridge/valley extrema mask for one response map."""

    A = np.asarray(response, dtype=float)

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

    grad_ref = np.percentile(
        np.abs(d1),
        99,
    )
    curv_ref = np.percentile(
        np.abs(lam1),
        99,
    )

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


def compute_extrema_mask(response, config):
    """Smooth one raw retinal response and compute its extrema mask."""

    smoothed = gaussian_filter(
        np.asarray(response),
        sigma=config.sigma_px,
        mode="reflect",
    )

    return ridge_valley_mask(
        smoothed,
        amp_thresh=config.amp_thresh,
        grad_tol_frac=config.grad_tol_frac,
        curv_tol_frac=config.curv_tol_frac,
    )


def compute_filled_response(response, seed_mask, config):
    """Fill one retinal response from its nearest extrema pixels."""

    A = np.asarray(response, dtype=float)
    seed_mask = np.asarray(seed_mask, dtype=bool)

    if A.shape != seed_mask.shape:
        raise ValueError(
            "response and seed_mask must have the same shape."
        )

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

    if config.fillradius is not None:
        fill_mask = (
            fill_mask
            & (dist_map <= float(config.fillradius))
        )

    filled = A.copy()
    filled[fill_mask] = nearest_vals[fill_mask]

    if (
        config.fillradius is not None
        and not config.beyondradius
    ):
        beyond = (
            (~seed_mask)
            & (dist_map > float(config.fillradius))
        )
        filled[beyond] = 0.0

    return filled.astype(np.float32)


def _validate_stimulus_ids(metadata, n_images):
    """Check that stimulus IDs map exactly onto Zarr array indices."""

    stim_ids = pd.to_numeric(
        metadata["stim_ID"],
        errors="coerce",
    )

    if stim_ids.isna().any():
        raise ValueError("metadata.csv contains invalid stim_ID values.")

    values = stim_ids.to_numpy(dtype=float)

    if not np.equal(values, np.floor(values)).all():
        raise ValueError("Every stim_ID must be an integer.")

    values = values.astype(int)

    if len(np.unique(values)) != len(values):
        raise ValueError("metadata.csv contains duplicate stim_ID values.")

    expected = np.arange(n_images, dtype=int)

    if not np.array_equal(np.sort(values), expected):
        raise ValueError(
            "stim_ID must uniquely cover every image index from "
            f"0 through {n_images - 1}."
        )


def load_retina_input_dataset(dataset_dir):
    """Load and validate a Module 2 dataset for Module 4 processing."""

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

    for name in ("imgs", "outs"):
        if name not in root:
            raise ValueError(
                f"dataset.zarr must contain a '{name}' array."
            )

    imgs = root["imgs"]
    outs = root["outs"]

    if len(imgs.shape) != 4 or imgs.shape[-1] != 3:
        raise ValueError(
            "The 'imgs' array must have shape (N, H, W, 3)."
        )

    if len(outs.shape) != 4:
        raise ValueError(
            "The 'outs' array must have shape (N, K, H, W)."
        )

    if imgs.shape[0] == 0:
        raise ValueError("The dataset contains no stimuli.")

    if outs.shape[0] != imgs.shape[0]:
        raise ValueError(
            "'imgs' and 'outs' must contain the same number of stimuli."
        )

    if outs.shape[2:] != imgs.shape[1:3]:
        raise ValueError(
            "Spatial dimensions of 'outs' must match 'imgs'."
        )

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
            "metadata.csv row count does not match the number "
            "of stimuli in dataset.zarr."
        )

    _validate_stimulus_ids(
        metadata,
        imgs.shape[0],
    )

    if "RESPONSE_KEYS" not in root.attrs:
        raise ValueError(
            "dataset.zarr is missing the RESPONSE_KEYS attribute."
        )

    response_keys = list(root.attrs["RESPONSE_KEYS"])

    if len(response_keys) != outs.shape[1]:
        raise ValueError(
            "RESPONSE_KEYS count does not match outs.shape[1]."
        )

    return RetinaInputDataset(
        dataset_dir=dataset_dir,
        zarr_path=zarr_path,
        metadata_path=metadata_path,
        root=root,
        imgs=imgs,
        outs=outs,
        metadata=metadata,
        response_keys=response_keys,
    )


def create_retina_model_run_paths(
    base_dir="data/interm",
    run_tag=None,
):
    """Create the directory structure for one Module 4 run."""

    base_dir = Path(base_dir)

    if run_tag is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = base_dir / f"retina_model_{run_tag}"

    if run_dir.exists():
        raise FileExistsError(
            f"Retina-model run directory already exists: {run_dir}"
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return RetinaModelRunPaths(
        run_dir=run_dir,
        outputs_zarr=run_dir / "retina_outputs.zarr",
        metadata_csv=run_dir / "metadata.csv",
        processing_settings_json=run_dir / "processing_settings.json",
    )


def write_processing_settings(paths, dataset, config):
    """Save the scientific settings and source dataset for one run."""

    settings = {
        "source_dataset": str(dataset.dataset_dir.resolve()),
        "source_dataset_zarr": str(dataset.zarr_path.resolve()),
        "response_keys": list(dataset.response_keys),
        "extrema": asdict(config.extrema),
        "fill": asdict(config.fill),
    }

    paths.processing_settings_json.write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )

    return settings


def estimate_retina_model_storage(dataset):
    """Estimate uncompressed storage required for Module 4 arrays."""

    n_values = int(np.prod(dataset.outs.shape))

    mask_bytes = (
        n_values
        * np.dtype("u1").itemsize
    )
    fill_bytes = (
        n_values
        * np.dtype("f4").itemsize
    )
    total_bytes = mask_bytes + fill_bytes

    gib = float(1024 ** 3)

    return {
        "extrema_mask_bytes": mask_bytes,
        "outs_fill_bytes": fill_bytes,
        "total_bytes": total_bytes,
        "extrema_mask_gib": mask_bytes / gib,
        "outs_fill_gib": fill_bytes / gib,
        "total_gib": total_bytes / gib,
    }


def build_retina_model(
    dataset_dir,
    config_path="config/retina_model.toml",
    base_dir="data/interm",
    run_tag=None,
    show_progress=False,
):
    """Build and save the completed retinal-model dataset."""

    config = load_retina_model_config(
        config_path
    )
    dataset = load_retina_input_dataset(
        dataset_dir
    )

    paths = create_retina_model_run_paths(
        base_dir=base_dir,
        run_tag=run_tag,
    )

    shutil.copy2(
        dataset.metadata_path,
        paths.metadata_csv,
    )

    write_processing_settings(
        paths,
        dataset,
        config,
    )

    N, K, H, W = dataset.outs.shape
    chunk_hw = int(min(256, H, W))

    compressor = zarr.codecs.BloscCodec(
        cname="zstd",
        clevel=3,
        shuffle="bitshuffle",
    )

    root = zarr.open_group(
        str(paths.outputs_zarr),
        mode="w",
    )

    root.attrs.update(
        {
            "complete": False,
            "source_dataset": str(
                dataset.dataset_dir.resolve()
            ),
            "RESPONSE_KEYS": list(
                dataset.response_keys
            ),
        }
    )

    mask_z = root.create_array(
        name="extrema_mask",
        shape=(N, K, H, W),
        chunks=(1, 1, chunk_hw, chunk_hw),
        dtype="u1",
        compressors=compressor,
    )

    fill_z = root.create_array(
        name="outs_fill",
        shape=(N, K, H, W),
        chunks=(1, 1, chunk_hw, chunk_hw),
        dtype="f4",
        compressors=compressor,
    )

    progress = tqdm(
        total=N * K,
        desc="Building retinal model",
        disable=not show_progress,
    )

    try:
        for stim_id in range(N):
            for ch in range(K):
                response = dataset.outs[
                    stim_id,
                    ch,
                ]

                mask = compute_extrema_mask(
                    response,
                    config.extrema,
                )

                filled = compute_filled_response(
                    response,
                    mask,
                    config.fill,
                )

                mask_z[
                    stim_id,
                    ch,
                ] = mask.astype(np.uint8)

                fill_z[
                    stim_id,
                    ch,
                ] = filled

                progress.update(1)

    finally:
        progress.close()

    root.attrs["complete"] = True

    return paths