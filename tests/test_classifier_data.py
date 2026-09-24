"""Tests for classifier-data loading, schema, weighted sampling, and shared splits."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.ndimage import zoom

import retinamodel.classifier_data as classifier_data


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

BASE_WEIGHTS = {
    "gray_d": 25.0,
    "gray_l": 25.0,
    "red": 12.5,
    "green": 12.5,
    "blue": 12.5,
    "yellow": 12.5,
}

BASE_GROUPS = {
    "Classic4": ("l_on", "l_off", "m_on", "m_off"),
    "Classic4_plus_s": ("l_on", "l_off", "m_on", "m_off", "s_on"),
    "Neitz4": ("l_h2_on", "m_h2_on", "l_h2_off", "m_h2_off"),
    "Neitz8": ("l_on", "l_off", "m_on", "m_off", "l_h2_on", "m_h2_on", "l_h2_off", "m_h2_off"),
    "All9": RESPONSE_KEYS,
}


def _make_metadata(labels):
    """Create valid classifier metadata with stable stim_ID order."""

    n = len(labels)
    backgrounds = ["gray", "red", "green", "blue"]

    return pd.DataFrame(
        {
            "stim_ID": np.arange(n, dtype=np.int64),
            "true_label": labels,
            "obj_int": np.linspace(0.2, 0.8, n),
            "obj_sat": np.linspace(0.1, 0.9, n),
            "bg_int": np.linspace(0.3, 0.7, n),
            "bg_sat": np.linspace(0.0, 0.8, n),
            "bg_hue": [backgrounds[i % len(backgrounds)] for i in range(n)],
        }
    )


def _make_classifier_data_dirs(tmp_path, labels=None):
    """Create tiny valid stimulus and retina-response outputs for tests."""

    if labels is None:
        labels = ["gray_d", "gray_l", "red", "green", "blue", "yellow"] * 2

    metadata = _make_metadata(labels)
    n = len(metadata)
    source_dir = tmp_path / "retina_dataset_test"
    retina_dir = tmp_path / "retina_model_test"
    source_dir.mkdir()
    retina_dir.mkdir()
    source_root = zarr.open_group(str(source_dir / "dataset.zarr"), mode="w")
    source_root.create_array("imgs", shape=(n, 4, 4, 3), dtype="f4")[:] = 0.5
    metadata.to_csv(source_dir / "metadata.csv", index=False)
    retina_root = zarr.open_group(str(retina_dir / "retina_outputs.zarr"), mode="w")
    retina_root.create_array("outs_fill", shape=(n, len(RESPONSE_KEYS), 4, 4), dtype="f4")[:] = 1.0
    retina_root.attrs["RESPONSE_KEYS"] = list(RESPONSE_KEYS)
    metadata.to_csv(retina_dir / "metadata.csv", index=False)
    (retina_dir / "processing_settings.json").write_text(json.dumps({"source_dataset": str(source_dir.resolve())}), encoding="utf-8")
    return source_dir, retina_dir


def _write_config(tmp_path, source_dir, retina_dir, seeds="[12]", extra_group="", include_split=True, input_sizes='["full", 2]'):
    """Write a complete temporary classifier_data.toml."""

    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    path = config_dir / "classifier_data.toml"
    split_block = ""

    if include_split:
        split_block = f'''
[split]
train_fraction = 0.70
validation_fraction = 0.15
test_fraction = 0.15
seeds = {seeds}
intensity_bins = 3
saturation_bins = 3
balance_background_hue = true
'''

    path.write_text(
        f'''
[paths]
source_dataset_dir = "{source_dir}"
retina_model_dir = "{retina_dir}"

[class_weights]
gray_d = 25.0
gray_l = 25.0
red = 12.5
green = 12.5
blue = 12.5
yellow = 12.5
{split_block}
[inputs]
sizes = {input_sizes}

[output]
root_dir = "{tmp_path / 'classifier_data_outputs'}"

[channel_groups]
Classic4 = ["l_on", "l_off", "m_on", "m_off"]
Classic4_plus_s = ["l_on", "l_off", "m_on", "m_off", "s_on"]
Neitz4 = ["l_h2_on", "m_h2_on", "l_h2_off", "m_h2_off"]
Neitz8 = ["l_on", "l_off", "m_on", "m_off", "l_h2_on", "m_h2_on", "l_h2_off", "m_h2_off"]
All9 = ["l_on", "l_off", "m_on", "m_off", "l_h2_on", "m_h2_on", "l_h2_off", "m_h2_off", "s_on"]
{extra_group}
'''.strip(),
        encoding="utf-8",
    )

    return path


def _schema_source(metadata, response_keys=RESPONSE_KEYS):
    """Return the minimal object needed by schema and split helpers."""

    return SimpleNamespace(metadata=metadata, response_keys=tuple(response_keys))


def _production_like_metadata():
    """Create the production class counts without the large image arrays."""

    return _make_metadata(["gray_d"] * 20 + ["gray_l"] * 20 + ["red"] * 88 + ["green"] * 88 + ["blue"] * 88 + ["yellow"] * 88)


def _notebook_allocate_counts(total, labels, frac_dict):
    """Direct reference implementation from the classifier notebook."""

    quotas = {lab: float(total) * float(frac_dict[lab]) for lab in labels}
    base = {lab: int(np.floor(quotas[lab])) for lab in labels}
    need = int(total - sum(base.values()))

    if need > 0:
        rema = sorted(labels, key=lambda lab: quotas[lab] - base[lab], reverse=True)
        for lab in rema[:need]:
            base[lab] += 1

    return base


def _notebook_bin_continuous(values, n_bins):
    """Direct reference implementation from the classifier notebook."""

    values = np.asarray(values, dtype=float)
    edges = np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), int(n_bins) + 1)
    mids = edges[1:-1]
    out = np.digitize(values, mids, right=False).astype(np.int64)
    return np.clip(out, 0, int(n_bins) - 1)


def _notebook_sample_balanced(idxs, n_select, rng, int_bin_series, sat_bin_series, bg_id_series=None):
    """Direct reference implementation from the classifier notebook."""

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


def _notebook_reference_split(metadata, selected_counts, class_order, class_fractions, seed=12, train_fraction=0.70, validation_fraction=0.15, test_fraction=0.15, intensity_bins=3, saturation_bins=3, balance_background_hue=True):
    """Build a small split with the notebook's exact random-operation order."""

    selected_total = int(sum(selected_counts.values()))
    validation_total = int(round(validation_fraction * selected_total))
    test_total = int(round(test_fraction * selected_total))
    train_total = int(selected_total - validation_total - test_total)
    train_counts = _notebook_allocate_counts(train_total, class_order, class_fractions)
    validation_counts = _notebook_allocate_counts(validation_total, class_order, class_fractions)
    test_counts = {lab: int(selected_counts[lab] - train_counts[lab] - validation_counts[lab]) for lab in class_order}
    rng = np.random.default_rng(int(seed))
    meta_split = metadata.copy()
    meta_split["_obj_int_r"] = np.round(meta_split["obj_int"].astype(float), 6)
    meta_split["_obj_sat_r"] = np.round(meta_split["obj_sat"].astype(float), 6)
    int_bin_series = pd.Series(_notebook_bin_continuous(meta_split["_obj_int_r"].to_numpy(), intensity_bins), index=meta_split.index)
    sat_bin_series = pd.Series(_notebook_bin_continuous(meta_split["_obj_sat_r"].to_numpy(), saturation_bins), index=meta_split.index)

    if balance_background_hue:
        bg_str = meta_split["bg_hue"].astype(str).to_numpy()
        bg_levels = sorted(np.unique(bg_str).tolist())
        bg_map = {background: i for i, background in enumerate(bg_levels)}
        bg_id_series = pd.Series(np.asarray([bg_map[background] for background in bg_str], dtype=np.int64), index=meta_split.index)
    else:
        bg_id_series = None

    label_series = meta_split["true_label"].astype(str)
    idx_by_label = {lab: meta_split.index[label_series == lab].to_numpy(dtype=np.int64) for lab in class_order}
    train_parts = []
    validation_parts = []
    test_parts = []

    for lab in class_order:
        idxs = idx_by_label[lab]
        test_ids = _notebook_sample_balanced(idxs, int(test_counts[lab]), rng, int_bin_series, sat_bin_series, bg_id_series)
        remaining = np.setdiff1d(idxs, test_ids, assume_unique=False).astype(np.int64)
        validation_ids = _notebook_sample_balanced(remaining, int(validation_counts[lab]), rng, int_bin_series, sat_bin_series, bg_id_series)
        remaining = np.setdiff1d(remaining, validation_ids, assume_unique=False).astype(np.int64)
        train_ids = _notebook_sample_balanced(remaining, int(train_counts[lab]), rng, int_bin_series, sat_bin_series, bg_id_series)
        train_parts.append(train_ids)
        validation_parts.append(validation_ids)
        test_parts.append(test_ids)

    train_ids = np.concatenate(train_parts).astype(np.int64)
    validation_ids = np.concatenate(validation_parts).astype(np.int64)
    test_ids = np.concatenate(test_parts).astype(np.int64)
    rng.shuffle(train_ids)
    rng.shuffle(validation_ids)
    rng.shuffle(test_ids)
    unused_ids = np.setdiff1d(meta_split.index.to_numpy(dtype=np.int64), np.concatenate([train_ids, validation_ids, test_ids]), assume_unique=False).astype(np.int64)
    return train_ids, validation_ids, test_ids, unused_ids


def test_load_classifier_data_config_baseline_settings(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    config_path = _write_config(tmp_path, source_dir, retina_dir, seeds="[12, 13, 14, 15]")
    config = classifier_data._load_classifier_data_config(config_path)

    assert config.source_dataset_dir == source_dir.resolve()
    assert config.retina_model_dir == retina_dir.resolve()
    assert config.class_weights == BASE_WEIGHTS
    assert config.split.train_fraction == pytest.approx(0.70)
    assert config.split.validation_fraction == pytest.approx(0.15)
    assert config.split.test_fraction == pytest.approx(0.15)
    assert config.split.seeds == (12, 13, 14, 15)
    assert config.split.intensity_bins == 3
    assert config.split.saturation_bins == 3
    assert config.split.balance_background_hue is True


def test_split_config_uses_documented_defaults(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    config_path = _write_config(tmp_path, source_dir, retina_dir, include_split=False)
    config = classifier_data._load_classifier_data_config(config_path)

    assert config.split.train_fraction == pytest.approx(0.70)
    assert config.split.validation_fraction == pytest.approx(0.15)
    assert config.split.test_fraction == pytest.approx(0.15)
    assert config.split.seeds == (12,)
    assert config.split.intensity_bins == 3
    assert config.split.saturation_bins == 3
    assert config.split.balance_background_hue is True


def test_split_config_rejects_duplicate_seeds(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    config_path = _write_config(tmp_path, source_dir, retina_dir, seeds="[12, 12]")

    with pytest.raises(ValueError, match="duplicate seeds"):
        classifier_data._load_classifier_data_config(config_path)


def test_load_and_validate_data_sources(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    sources = classifier_data._load_and_validate_data_sources(source_dir, retina_dir)

    assert sources.imgs.shape == (12, 4, 4, 3)
    assert sources.outs_fill.shape == (12, 9, 4, 4)
    assert sources.response_keys == RESPONSE_KEYS
    assert sources.metadata["stim_ID"].tolist() == list(range(12))


def test_data_sources_reject_stim_id_misalignment(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    metadata_path = retina_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    metadata.loc[[0, 1], "stim_ID"] = [1, 0]
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="stim_ID order"):
        classifier_data._load_and_validate_data_sources(source_dir, retina_dir)


def test_data_sources_reject_wrong_processing_provenance(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    (retina_dir / "processing_settings.json").write_text(json.dumps({"source_dataset": str(tmp_path / "different_dataset")}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not reference"):
        classifier_data._load_and_validate_data_sources(source_dir, retina_dir)


def test_data_sources_reject_response_key_count_mismatch(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    retina_root = zarr.open_group(str(retina_dir / "retina_outputs.zarr"), mode="a")
    retina_root.attrs["RESPONSE_KEYS"] = list(RESPONSE_KEYS[:-1])

    with pytest.raises(ValueError, match="Channel-count mismatch"):
        classifier_data._load_and_validate_data_sources(source_dir, retina_dir)


def test_data_sources_reject_missing_required_metadata_column(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    metadata_path = retina_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_path).drop(columns=["obj_sat"])
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        classifier_data._load_and_validate_data_sources(source_dir, retina_dir)


def test_shared_schema_baseline_and_custom_channel_group(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    config_path = _write_config(tmp_path, source_dir, retina_dir, extra_group='Off2 = ["l_off", "m_off"]')
    config = classifier_data._load_classifier_data_config(config_path)
    sources = classifier_data._load_and_validate_data_sources(source_dir, retina_dir)
    schema = classifier_data._define_shared_data_schema(sources, config.class_weights, config.channel_groups)

    assert schema.class_order == ("gray_d", "gray_l", "red", "green", "blue", "yellow")
    assert schema.active_class_order == schema.class_order
    assert tuple(schema.class_fractions.values()) == pytest.approx((0.25, 0.25, 0.125, 0.125, 0.125, 0.125))
    assert schema.channel_indices["Classic4"] == (0, 1, 2, 3)
    assert schema.channel_indices["Classic4_plus_s"] == (0, 1, 2, 3, 8)
    assert schema.channel_indices["Neitz4"] == (4, 5, 6, 7)
    assert schema.channel_indices["Neitz8"] == (0, 1, 2, 3, 4, 5, 6, 7)
    assert schema.channel_indices["All9"] == (0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert schema.channel_indices["Off2"] == (1, 3)


def test_zero_weight_class_is_excluded():
    metadata = _production_like_metadata()
    source = _schema_source(metadata)
    weights = dict(BASE_WEIGHTS)
    weights["yellow"] = 0.0
    schema = classifier_data._define_shared_data_schema(source, weights, BASE_GROUPS)
    plan = classifier_data._build_weighted_sample_plan(source, schema)

    assert "yellow" not in schema.active_class_order
    assert schema.class_fractions["yellow"] == 0.0
    assert plan.selected_counts["yellow"] == 0
    assert plan.unused_counts["yellow"] == 88


def test_class_weights_reject_negative_values():
    weights = dict(BASE_WEIGHTS)
    weights["red"] = -1.0

    with pytest.raises(ValueError, match=">= 0"):
        classifier_data._validate_class_weights(weights)


def test_class_weights_require_two_active_classes():
    weights = {label: 0.0 for label in BASE_WEIGHTS}
    weights["red"] = 1.0

    with pytest.raises(ValueError, match="At least two classes"):
        classifier_data._validate_class_weights(weights)


def test_unknown_channel_in_group_is_rejected():
    metadata = _make_metadata(["gray_d", "gray_l", "red", "green", "blue", "yellow"])
    source = _schema_source(metadata)
    groups = dict(BASE_GROUPS)
    groups["BadGroup"] = ("l_on", "not_a_channel")

    with pytest.raises(ValueError, match="missing retinal channels"):
        classifier_data._define_shared_data_schema(source, BASE_WEIGHTS, groups)


def test_allocate_counts_matches_notebook_reference():
    labels = ["gray_d", "gray_l", "red", "green", "blue", "yellow"]
    fractions = {"gray_d": 0.25, "gray_l": 0.25, "red": 0.125, "green": 0.125, "blue": 0.125, "yellow": 0.125}

    for total in (12, 56, 80):
        assert classifier_data._allocate_counts(total, labels, fractions) == _notebook_allocate_counts(total, labels, fractions)

    assert classifier_data._allocate_counts(12, labels, fractions) == {"gray_d": 3, "gray_l": 3, "red": 2, "green": 2, "blue": 1, "yellow": 1}


def test_weighted_sample_plan_matches_production_baseline_counts():
    source = _schema_source(_production_like_metadata())
    schema = classifier_data._define_shared_data_schema(source, BASE_WEIGHTS, BASE_GROUPS)
    plan = classifier_data._build_weighted_sample_plan(source, schema)

    assert plan.dataset_total == 392
    assert plan.selected_total == 80
    assert plan.unused_total == 312
    assert plan.available_counts == {"gray_d": 20, "gray_l": 20, "red": 88, "green": 88, "blue": 88, "yellow": 88}
    assert plan.selected_counts == {"gray_d": 20, "gray_l": 20, "red": 10, "green": 10, "blue": 10, "yellow": 10}
    assert plan.unused_counts == {"gray_d": 0, "gray_l": 0, "red": 78, "green": 78, "blue": 78, "yellow": 78}


def test_equivalent_relative_weights_give_same_weighted_plan():
    source = _schema_source(_production_like_metadata())
    baseline_schema = classifier_data._define_shared_data_schema(source, BASE_WEIGHTS, BASE_GROUPS)
    ratio_schema = classifier_data._define_shared_data_schema(source, {"gray_d": 2.0, "gray_l": 2.0, "red": 1.0, "green": 1.0, "blue": 1.0, "yellow": 1.0}, BASE_GROUPS)
    baseline_plan = classifier_data._build_weighted_sample_plan(source, baseline_schema)
    ratio_plan = classifier_data._build_weighted_sample_plan(source, ratio_schema)

    assert ratio_plan.selected_total == baseline_plan.selected_total
    assert ratio_plan.selected_counts == baseline_plan.selected_counts
    assert ratio_plan.unused_counts == baseline_plan.unused_counts


def test_shared_split_baseline_counts_and_integrity():
    metadata = _production_like_metadata()
    source = _schema_source(metadata)
    schema = classifier_data._define_shared_data_schema(source, BASE_WEIGHTS, BASE_GROUPS)
    plan = classifier_data._build_weighted_sample_plan(source, schema)
    split_config = classifier_data._validate_split_config({"seeds": [12]})
    split = classifier_data._generate_shared_split(source, schema, plan, split_config, seed=12)

    assert len(split.train_stim_ids) == 56
    assert len(split.validation_stim_ids) == 12
    assert len(split.test_stim_ids) == 12
    assert len(split.unused_stim_ids) == 312
    assert np.intersect1d(split.train_stim_ids, split.validation_stim_ids).size == 0
    assert np.intersect1d(split.train_stim_ids, split.test_stim_ids).size == 0
    assert np.intersect1d(split.validation_stim_ids, split.test_stim_ids).size == 0
    assert len(np.unique(np.concatenate([split.train_stim_ids, split.validation_stim_ids, split.test_stim_ids, split.unused_stim_ids]))) == 392

    train_counts = metadata.loc[split.train_stim_ids, "true_label"].value_counts().reindex(schema.active_class_order).fillna(0).astype(int).to_dict()
    validation_counts = metadata.loc[split.validation_stim_ids, "true_label"].value_counts().reindex(schema.active_class_order).fillna(0).astype(int).to_dict()
    test_counts = metadata.loc[split.test_stim_ids, "true_label"].value_counts().reindex(schema.active_class_order).fillna(0).astype(int).to_dict()

    assert train_counts == {"gray_d": 14, "gray_l": 14, "red": 7, "green": 7, "blue": 7, "yellow": 7}
    assert validation_counts == {"gray_d": 3, "gray_l": 3, "red": 2, "green": 2, "blue": 1, "yellow": 1}
    assert test_counts == {"gray_d": 3, "gray_l": 3, "red": 1, "green": 1, "blue": 2, "yellow": 2}


def test_shared_split_is_deterministic_for_same_seed():
    source = _schema_source(_production_like_metadata())
    schema = classifier_data._define_shared_data_schema(source, BASE_WEIGHTS, BASE_GROUPS)
    plan = classifier_data._build_weighted_sample_plan(source, schema)
    split_config = classifier_data._validate_split_config({"seeds": [12]})
    split_a = classifier_data._generate_shared_split(source, schema, plan, split_config, seed=12)
    split_b = classifier_data._generate_shared_split(source, schema, plan, split_config, seed=12)

    assert np.array_equal(split_a.train_stim_ids, split_b.train_stim_ids)
    assert np.array_equal(split_a.validation_stim_ids, split_b.validation_stim_ids)
    assert np.array_equal(split_a.test_stim_ids, split_b.test_stim_ids)
    assert np.array_equal(split_a.unused_stim_ids, split_b.unused_stim_ids)


def test_shared_split_matches_notebook_reference_algorithm():
    metadata = _production_like_metadata()
    source = _schema_source(metadata)
    schema = classifier_data._define_shared_data_schema(source, BASE_WEIGHTS, BASE_GROUPS)
    plan = classifier_data._build_weighted_sample_plan(source, schema)
    split_config = classifier_data._validate_split_config({"seeds": [12]})
    actual = classifier_data._generate_shared_split(source, schema, plan, split_config, seed=12)
    expected = _notebook_reference_split(metadata, plan.selected_counts, schema.active_class_order, schema.class_fractions, seed=12)

    assert np.array_equal(actual.train_stim_ids, expected[0])
    assert np.array_equal(actual.validation_stim_ids, expected[1])
    assert np.array_equal(actual.test_stim_ids, expected[2])
    assert np.array_equal(actual.unused_stim_ids, expected[3])


def test_multiple_seeds_create_shared_split_for_each_seed():
    source = _schema_source(_production_like_metadata())
    schema = classifier_data._define_shared_data_schema(source, BASE_WEIGHTS, BASE_GROUPS)
    plan = classifier_data._build_weighted_sample_plan(source, schema)
    split_config = classifier_data._validate_split_config({"seeds": [12, 13, 14, 15]})
    splits = classifier_data._generate_shared_splits(source, schema, plan, split_config)

    assert tuple(splits) == (12, 13, 14, 15)
    assert all(split.seed == seed for seed, split in splits.items())
    assert all(len(split.train_stim_ids) == 56 for split in splits.values())
    assert all(len(split.validation_stim_ids) == 12 for split in splits.values())
    assert all(len(split.test_stim_ids) == 12 for split in splits.values())
    assert not np.array_equal(splits[12].test_stim_ids, splits[13].test_stim_ids)


def test_zero_weight_class_never_enters_shared_split():
    metadata = _production_like_metadata()
    source = _schema_source(metadata)
    weights = dict(BASE_WEIGHTS)
    weights["yellow"] = 0.0
    schema = classifier_data._define_shared_data_schema(source, weights, BASE_GROUPS)
    plan = classifier_data._build_weighted_sample_plan(source, schema)
    split_config = classifier_data._validate_split_config({"seeds": [12]})
    split = classifier_data._generate_shared_split(source, schema, plan, split_config, seed=12)
    selected_ids = np.concatenate([split.train_stim_ids, split.validation_stim_ids, split.test_stim_ids])

    assert "yellow" not in set(metadata.loc[selected_ids, "true_label"])
    assert set(metadata.index[metadata["true_label"] == "yellow"]).issubset(set(split.unused_stim_ids))


def test_local_production_split_matches_notebook_reference_when_available():
    """Compare exact seed-12 stim_ID arrays when the local research artifacts are available."""

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "classifier_data.toml"
    config = classifier_data._load_classifier_data_config(config_path)
    reference_path = config.source_dataset_dir / "split_cache_shared" / "split_seed12_train70_val15_test15_weighted_v1.npz"

    if not config.source_dataset_dir.exists() or not config.retina_model_dir.exists() or not reference_path.exists():
        pytest.skip("Local production data or notebook split reference is not available.")

    sources = classifier_data._load_and_validate_data_sources(config.source_dataset_dir, config.retina_model_dir)
    schema = classifier_data._define_shared_data_schema(sources, config.class_weights, config.channel_groups)
    plan = classifier_data._build_weighted_sample_plan(sources, schema)
    actual = classifier_data._generate_shared_split(sources, schema, plan, config.split, seed=12)

    with np.load(reference_path, allow_pickle=False) as reference:
        assert np.array_equal(actual.train_stim_ids, reference["train_idx"].astype(np.int64))
        assert np.array_equal(actual.validation_stim_ids, reference["val_idx"].astype(np.int64))
        assert np.array_equal(actual.test_stim_ids, reference["test_idx"].astype(np.int64))
        if "unused_idx" in reference.files:
            assert np.array_equal(actual.unused_stim_ids, reference["unused_idx"].astype(np.int64))



def test_model_split_view_exposes_train_and_validation_labels_but_not_test_labels():
    metadata = _production_like_metadata()
    source = _schema_source(metadata)
    schema = classifier_data._define_shared_data_schema(source, BASE_WEIGHTS, BASE_GROUPS)
    plan = classifier_data._build_weighted_sample_plan(source, schema)
    split_config = classifier_data._validate_split_config({"seeds": [12]})
    split = classifier_data._generate_shared_split(source, schema, plan, split_config, seed=12)
    view = classifier_data._build_model_split_view(source, split)

    assert view.seed == 12
    assert np.array_equal(view.train.stim_ids, split.train_stim_ids)
    assert np.array_equal(view.validation.stim_ids, split.validation_stim_ids)
    assert np.array_equal(view.test.stim_ids, split.test_stim_ids)
    assert hasattr(view.train, "labels")
    assert hasattr(view.validation, "labels")
    assert "true_label" in view.train.metadata.columns
    assert "true_label" in view.validation.metadata.columns
    assert list(view.test.metadata.columns) == ["stim_ID"]
    assert not hasattr(view.test, "labels")
    assert not hasattr(view, "test_labels")
    assert not hasattr(view, "y_test")


def test_model_split_view_labels_match_authoritative_metadata():
    metadata = _production_like_metadata()
    source = _schema_source(metadata)
    schema = classifier_data._define_shared_data_schema(source, BASE_WEIGHTS, BASE_GROUPS)
    plan = classifier_data._build_weighted_sample_plan(source, schema)
    split_config = classifier_data._validate_split_config({"seeds": [12]})
    split = classifier_data._generate_shared_split(source, schema, plan, split_config, seed=12)
    view = classifier_data._build_model_split_view(source, split)

    expected_train = metadata.loc[split.train_stim_ids, "true_label"].astype(str).to_numpy()
    expected_validation = metadata.loc[split.validation_stim_ids, "true_label"].astype(str).to_numpy()
    assert np.array_equal(view.train.labels, expected_train)
    assert np.array_equal(view.validation.labels, expected_validation)


def test_model_split_views_preserve_shared_ids_for_multiple_seeds():
    source = _schema_source(_production_like_metadata())
    schema = classifier_data._define_shared_data_schema(source, BASE_WEIGHTS, BASE_GROUPS)
    plan = classifier_data._build_weighted_sample_plan(source, schema)
    split_config = classifier_data._validate_split_config({"seeds": [12, 13]})
    splits = classifier_data._generate_shared_splits(source, schema, plan, split_config)
    views = classifier_data._build_model_split_views(source, splits)

    assert tuple(views) == (12, 13)
    assert all(np.array_equal(views[seed].train.stim_ids, splits[seed].train_stim_ids) for seed in views)
    assert all(np.array_equal(views[seed].validation.stim_ids, splits[seed].validation_stim_ids) for seed in views)
    assert all(np.array_equal(views[seed].test.stim_ids, splits[seed].test_stim_ids) for seed in views)
    assert all(not hasattr(views[seed].test, "labels") for seed in views)


def test_input_config_accepts_full_and_multiple_sizes(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    config_path = _write_config(tmp_path, source_dir, retina_dir, input_sizes='["full", 2, 3]')
    config = classifier_data._load_classifier_data_config(config_path)

    assert config.inputs.sizes == ("full", 2, 3)


def test_input_config_uses_documented_defaults(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    config_path = _write_config(tmp_path, source_dir, retina_dir)
    text = config_path.read_text(encoding="utf-8")
    start = text.index("[inputs]")
    end = text.index("[output]")
    config_path.write_text(text[:start] + text[end:], encoding="utf-8")
    config = classifier_data._load_classifier_data_config(config_path)

    assert config.inputs.sizes == ("full", 256)


def test_input_config_rejects_none_and_duplicate_sizes(tmp_path):
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path)
    bad_name = _write_config(tmp_path, source_dir, retina_dir, input_sizes='["none", 2]')

    with pytest.raises(ValueError, match="must be 'full'"):
        classifier_data._load_classifier_data_config(bad_name)

    duplicate = _write_config(tmp_path, source_dir, retina_dir, input_sizes='["full", 2, 2]')

    with pytest.raises(ValueError, match="duplicate"):
        classifier_data._load_classifier_data_config(duplicate)


def test_configured_input_size_cannot_upsample(tmp_path):
    labels = ["gray_d"] * 20 + ["gray_l"] * 20 + ["red"] * 20 + ["green"] * 20 + ["blue"] * 20 + ["yellow"] * 20
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path, labels=labels)
    config_path = _write_config(tmp_path, source_dir, retina_dir, input_sizes='["full", 5]')

    with pytest.raises(ValueError, match="would upsample"):
        classifier_data.build_classifier_data_bundle(config_path, record=False)


def _make_input_bundle(tmp_path, input_sizes='["full", 2]'):
    labels = ["gray_d"] * 20 + ["gray_l"] * 20 + ["red"] * 20 + ["green"] * 20 + ["blue"] * 20 + ["yellow"] * 20
    source_dir, retina_dir = _make_classifier_data_dirs(tmp_path, labels=labels)
    retina_root = zarr.open_group(str(retina_dir / "retina_outputs.zarr"), mode="a")
    arr = retina_root["outs_fill"]

    for stim_id in range(len(labels)):
        for channel in range(len(RESPONSE_KEYS)):
            base = np.arange(16, dtype=np.float32).reshape(4, 4)
            arr[stim_id, channel, :, :] = base + np.float32(stim_id * 100 + channel * 10)

    config_path = _write_config(tmp_path, source_dir, retina_dir, input_sizes=input_sizes)
    bundle = classifier_data.build_classifier_data_bundle(config_path, record=False)
    return bundle


def test_cnn_full_input_matches_source_maps_exactly(tmp_path):
    bundle = _make_input_bundle(tmp_path)
    prepared = classifier_data.build_cnn_input(bundle, 12, "train", "Classic4", "full")
    stim_id = int(prepared.stim_ids[0])
    expected = np.asarray(bundle._data_sources.outs_fill[stim_id, :, :, :], dtype=np.float32)[[0, 1, 2, 3], :, :]

    assert prepared.X[0].dtype == np.float32
    assert prepared.X[0].shape == (4, 4, 4)
    assert np.array_equal(prepared.X[0], expected)
    assert np.array_equal(prepared.labels, bundle.model_views[12].train.labels)


def test_cnn_resized_input_matches_scipy_order_one(tmp_path):
    bundle = _make_input_bundle(tmp_path)
    prepared = classifier_data.build_cnn_input(bundle, 12, "validation", "Classic4", 2)
    stim_id = int(prepared.stim_ids[0])
    source = np.asarray(bundle._data_sources.outs_fill[stim_id, :, :, :], dtype=np.float32)[[0, 1, 2, 3], :, :]
    expected = np.stack([zoom(source[ch], (0.5, 0.5), order=1).astype(np.float32, copy=False) for ch in range(4)], axis=0)

    assert prepared.X[0].dtype == np.float32
    assert prepared.X[0].shape == (4, 2, 2)
    assert np.array_equal(prepared.X[0], expected)


def test_sklearn_input_stacks_and_flattens_notebook_order(tmp_path):
    bundle = _make_input_bundle(tmp_path)
    prepared = classifier_data.build_sklearn_input(bundle, 12, "train", "Classic4", 2)
    cnn_input = classifier_data.build_cnn_input(bundle, 12, "train", "Classic4", 2)

    assert prepared.X.dtype == np.float32
    assert prepared.X.shape == (len(prepared.stim_ids), 4 * 2 * 2)
    assert np.array_equal(prepared.X[0], cnn_input.X[0].reshape(-1))
    assert np.array_equal(prepared.stim_ids, cnn_input.stim_ids)
    assert np.array_equal(prepared.labels, cnn_input.labels)


def test_sklearn_memmap_input_matches_in_memory_input(tmp_path):
    bundle = _make_input_bundle(tmp_path)
    memory = classifier_data.build_sklearn_input(bundle, 12, "validation", "Classic4", 2)
    mmap_path = tmp_path / "features.dat"
    mapped = classifier_data.build_sklearn_input(bundle, 12, "validation", "Classic4", 2, mmap_path=mmap_path)

    assert isinstance(mapped.X, np.memmap)
    assert mmap_path.exists()
    assert np.array_equal(mapped.X, memory.X)


def test_test_inputs_never_expose_labels(tmp_path):
    bundle = _make_input_bundle(tmp_path)
    sklearn_test = classifier_data.build_sklearn_input(bundle, 12, "test", "Classic4", 2)
    cnn_test = classifier_data.build_cnn_input(bundle, 12, "test", "Classic4", "full")

    assert not hasattr(sklearn_test, "labels")
    assert not hasattr(cnn_test, "labels")
    assert np.array_equal(sklearn_test.stim_ids, bundle.splits[12].test_stim_ids)
    assert np.array_equal(cnn_test.stim_ids, bundle.splits[12].test_stim_ids)


def test_input_representations_preserve_shared_stimulus_ids(tmp_path):
    bundle = _make_input_bundle(tmp_path)
    full = classifier_data.build_cnn_input(bundle, 12, "validation", "Classic4", "full")
    resized = classifier_data.build_cnn_input(bundle, 12, "validation", "Classic4", 2)
    flat = classifier_data.build_sklearn_input(bundle, 12, "validation", "Classic4", 2)

    assert np.array_equal(full.stim_ids, resized.stim_ids)
    assert np.array_equal(full.stim_ids, flat.stim_ids)
    assert np.array_equal(full.stim_ids, bundle.splits[12].validation_stim_ids)


def test_unconfigured_input_size_is_rejected(tmp_path):
    bundle = _make_input_bundle(tmp_path)

    with pytest.raises(ValueError, match="not configured"):
        classifier_data.build_cnn_input(bundle, 12, "train", "Classic4", 3)


def test_bundle_records_lightweight_reproducibility_files(tmp_path):
    bundle = _make_input_bundle(tmp_path)
    output_dir = tmp_path / "recorded_bundle"
    classifier_data.record_classifier_data_bundle(bundle, output_dir)
    manifest_path = output_dir / "classifier_data_manifest.json"
    split_path = output_dir / "split_seed12.npz"
    test_metadata_path = output_dir / "test_metadata_seed12.csv"

    assert manifest_path.exists()
    assert (output_dir / "classifier_data_config.toml").exists()
    assert split_path.exists()
    assert (output_dir / "train_metadata_seed12.csv").exists()
    assert (output_dir / "validation_metadata_seed12.csv").exists()
    assert test_metadata_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_metadata = pd.read_csv(test_metadata_path)

    assert manifest["inputs"] == {"interpolation_order": 1, "sizes": ["full", 2]}
    assert manifest["test_labels_saved"] is False
    assert manifest["selected_total"] == bundle.sample_plan.selected_total
    assert manifest["splits"]["12"]["test_n"] == len(bundle.splits[12].test_stim_ids)
    assert list(test_metadata.columns) == ["stim_ID"]

    with np.load(split_path, allow_pickle=False) as saved:
        assert np.array_equal(saved["train_idx"], bundle.splits[12].train_stim_ids)
        assert np.array_equal(saved["val_idx"], bundle.splits[12].validation_stim_ids)
        assert np.array_equal(saved["test_idx"], bundle.splits[12].test_stim_ids)
        assert np.array_equal(saved["unused_idx"], bundle.splits[12].unused_stim_ids)


def test_bundle_summary_reports_core_user_readouts(tmp_path):
    bundle = _make_input_bundle(tmp_path)
    summary = classifier_data.summarize_classifier_data_bundle(bundle)

    assert "selected:" in summary
    assert "unused:" in summary
    assert "seeds: 12" in summary
    assert "input sizes: full, 2" in summary
    assert "Classic4" in summary




def test_recorded_bundle_round_trip_preserves_shared_ids_and_protection(tmp_path):
    bundle = _make_input_bundle(tmp_path)
    output_dir = tmp_path / "round_trip_bundle"
    classifier_data.record_classifier_data_bundle(bundle, output_dir)
    loaded = classifier_data.load_classifier_data_bundle(output_dir)

    assert np.array_equal(loaded.splits[12].train_stim_ids, bundle.splits[12].train_stim_ids)
    assert np.array_equal(loaded.splits[12].validation_stim_ids, bundle.splits[12].validation_stim_ids)
    assert np.array_equal(loaded.splits[12].test_stim_ids, bundle.splits[12].test_stim_ids)
    assert np.array_equal(loaded.splits[12].unused_stim_ids, bundle.splits[12].unused_stim_ids)
    assert list(loaded.model_views[12].test.metadata.columns) == ["stim_ID"]
    assert not hasattr(loaded.model_views[12].test, "labels")
