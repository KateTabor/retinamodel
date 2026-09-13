"""Tests for human-audit configuration and dataset validation."""

import numpy as np
import pandas as pd
import pytest
import zarr

from retinamodel.audit import (
    AuditConfig,
    GalleryConfig,
    LabelCoverage,
    apply_existing_labels,
    consolidate_human_labels,
    existing_label_coverage,
    load_audit_config,
    load_audit_dataset,
    prompt_label_reuse,
    randomize_pending_stimuli,
    rebuild_human_label_lookup,
    select_audit_stimuli,
    AuditSession,
    audit_session_progress,
    create_audit_session,
    load_audit_session,
    next_pending_stimulus,
    record_audit_label,
    finalize_audit_session,
    analyze_audit_session,
    select_gallery_stimuli,
    plot_stimulus_gallery,
)


def _valid_stim_settings():
    """Return the minimum stimulus settings required by the gallery."""

    return {
        "image_size_px": 4,
        "obj_diam_ratio": 0.5,
        "gray_light_threshold": 0.5,
        "obj_gray_min": 0.3,
        "obj_gray_max": 0.7,
        "obj_chrom_int_min": 0.3,
        "obj_chrom_int_max": 0.7,
        "obj_chrom_sat_min": 0.3,
        "obj_chrom_sat_max": 0.7,
    }


def _valid_metadata():
    """Return a small valid metadata table."""

    return pd.DataFrame(
        {
            "stim_ID": [0, 1],
            "true_label": ["gray_d", "red"],
            "bg_hue": ["gray", "blue"],
            "bg_r": [0.2, 0.1],
            "bg_g": [0.2, 0.2],
            "bg_b": [0.2, 0.4],
            "obj_r": [0.3, 0.7],
            "obj_g": [0.3, 0.3],
            "obj_b": [0.3, 0.3],
            "bg_int": [0.2, 0.4],
            "obj_int": [0.3, 0.7],
            "delta_int": [-0.1, -0.3],
            "bg_sat": [0.0, 0.5],
            "obj_sat": [0.0, 0.6],
            "delta_sat": [0.0, -0.1],
        }
    )


def _make_dataset(tmp_path, metadata=None, stim_settings=None):
    """Create a minimal Module 2-style dataset for testing."""

    dataset_dir = tmp_path / "retina_dataset_test"
    zarr_path = dataset_dir / "dataset.zarr"
    metadata_path = dataset_dir / "metadata.csv"

    dataset_dir.mkdir()

    root = zarr.open_group(zarr_path, mode="w")

    imgs = root.create_array(
        name="imgs",
        shape=(2, 4, 4, 3),
        dtype="f4",
    )
    imgs[:] = 0.5

    root.attrs.update(
        {
            "IMAGE_SIZE_PX": 4,
            "OBJ_DIAM_RATIO": 0.5,
            "GRAY_LIGHT_THRESHOLD": 0.5,
            "STIM_SETTINGS": (
                _valid_stim_settings()
                if stim_settings is None
                else stim_settings
            ),
        }
    )

    if metadata is None:
        metadata = _valid_metadata()

    metadata.to_csv(metadata_path, index=False)

    return dataset_dir


def test_project_audit_config_loads():
    """Check that the repository audit configuration is valid."""

    audit_config, gallery_config = load_audit_config(
        "config/audit.toml"
    )

    assert isinstance(audit_config, AuditConfig)
    assert isinstance(gallery_config, GalleryConfig)


def test_audit_config_rejects_unknown_label():
    """Check that unsupported audit labels are rejected."""

    with pytest.raises(ValueError, match="Unknown audit label"):
        AuditConfig(
            labels=["red", "purple"],
            bg_hues=["gray"],
            random_seed=14,
            filter_delta_int=False,
            delta_int_min=-1.0,
            delta_int_max=1.0,
            filter_delta_sat=False,
            delta_sat_min=-1.0,
            delta_sat_max=1.0,
        )


def test_gallery_config_rejects_invalid_columns():
    """Check that the gallery requires at least one column."""

    with pytest.raises(
        ValueError,
        match="n_cols must be a positive integer",
    ):
        GalleryConfig(
            target_label="red",
            n_cols=0,
            save_png=False,
            dpi=200,
        )


def test_valid_dataset_loads(tmp_path):
    """Check that a valid Module 2 dataset loads successfully."""

    dataset_dir = _make_dataset(tmp_path)

    dataset = load_audit_dataset(dataset_dir)

    assert dataset.image_size_px == 4
    assert dataset.obj_diam_ratio == pytest.approx(0.5)
    assert dataset.gray_light_threshold == pytest.approx(0.5)
    assert dataset.imgs.shape == (2, 4, 4, 3)
    assert len(dataset.metadata) == 2
    assert dataset.stim_settings["obj_gray_min"] == pytest.approx(0.3)


def test_dataset_rejects_missing_metadata_column(tmp_path):
    """Check that required metadata columns cannot be missing."""

    metadata = _valid_metadata().drop(columns=["delta_sat"])
    dataset_dir = _make_dataset(tmp_path, metadata=metadata)

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        load_audit_dataset(dataset_dir)


def test_dataset_rejects_invalid_stimulus_ids(tmp_path):
    """Check that stim_ID must map exactly onto the Zarr images."""

    metadata = _valid_metadata()
    metadata["stim_ID"] = [0, 2]

    dataset_dir = _make_dataset(tmp_path, metadata=metadata)

    with pytest.raises(
        ValueError,
        match="stim_ID must uniquely cover every image index",
    ):
        load_audit_dataset(dataset_dir)


def test_dataset_normalizes_stim_setting_names(tmp_path):
    """Check compatibility with uppercase notebook-style setting names."""

    uppercase_settings = {
        key.upper(): value
        for key, value in _valid_stim_settings().items()
    }

    dataset_dir = _make_dataset(
        tmp_path,
        stim_settings=uppercase_settings,
    )

    dataset = load_audit_dataset(dataset_dir)

    assert "obj_gray_min" in dataset.stim_settings
    assert "OBJ_GRAY_MIN" not in dataset.stim_settings
    assert dataset.stim_settings["obj_gray_min"] == pytest.approx(0.3)


def _human_observation(
    label,
    labeled_at,
    obj_rgb=(0.7, 0.3, 0.3),
    bg_rgb=(0.2, 0.2, 0.2),
    ratio=0.5,
):
    """Build one human-label observation for testing."""

    return {
        "user_label": label,
        "labeled_at": labeled_at,
        "obj_r": obj_rgb[0],
        "obj_g": obj_rgb[1],
        "obj_b": obj_rgb[2],
        "bg_r": bg_rgb[0],
        "bg_g": bg_rgb[1],
        "bg_b": bg_rgb[2],
        "obj_diam_ratio": ratio,
    }


def test_select_audit_stimuli_applies_filters(tmp_path):
    """Check label, background, and signed delta filtering."""

    dataset_dir = _make_dataset(tmp_path)
    dataset = load_audit_dataset(dataset_dir)

    config = AuditConfig(
        labels=["red"],
        bg_hues=["blue"],
        random_seed=14,
        filter_delta_int=True,
        delta_int_min=-0.3,
        delta_int_max=-0.3,
        filter_delta_sat=True,
        delta_sat_min=-0.1,
        delta_sat_max=-0.1,
    )

    selected = select_audit_stimuli(
        dataset,
        config,
    )

    assert len(selected) == 1
    assert selected.iloc[0]["stim_ID"] == 1
    assert selected.iloc[0]["true_label"] == "red"
    assert selected.iloc[0]["obj_diam_ratio"] == pytest.approx(0.5)


def test_consolidate_human_labels_uses_unique_most_frequent():
    """Check unique highest-count label consolidation."""

    observations = pd.DataFrame(
        [
            _human_observation(
                "red",
                "2026-09-01T12:00:00Z",
            ),
            _human_observation(
                "red",
                "2026-09-02T12:00:00Z",
            ),
            _human_observation(
                "green",
                "2026-09-03T12:00:00Z",
            ),
        ]
    )

    lookup = consolidate_human_labels(observations)

    assert len(lookup) == 1
    assert lookup.iloc[0]["human_label"] == "red"
    assert lookup.iloc[0]["n_human_labels"] == 3
    assert lookup.iloc[0]["label_resolution"] == "majority"


def test_consolidate_human_labels_uses_latest_tiebreak():
    """Check that tied labels use the latest tied observation."""

    observations = pd.DataFrame(
        [
            _human_observation(
                "red",
                "2026-09-01T12:00:00Z",
            ),
            _human_observation(
                "green",
                "2026-09-03T12:00:00Z",
            ),
        ]
    )

    lookup = consolidate_human_labels(observations)

    assert lookup.iloc[0]["human_label"] == "green"
    assert lookup.iloc[0]["label_resolution"] == "latest_tiebreak"


def test_consolidate_keeps_different_ratios_separate():
    """Check that object/background ratio remains part of the key."""

    observations = pd.DataFrame(
        [
            _human_observation(
                "red",
                "2026-09-01T12:00:00Z",
                ratio=0.4,
            ),
            _human_observation(
                "green",
                "2026-09-02T12:00:00Z",
                ratio=0.6,
            ),
        ]
    )

    lookup = consolidate_human_labels(observations)

    assert len(lookup) == 2
    assert set(lookup["obj_diam_ratio"]) == {0.4, 0.6}


def test_rebuild_human_label_lookup_from_raw_files(tmp_path):
    """Check raw observations can regenerate the interim lookup."""

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    lookup_path = tmp_path / "interm" / "human_label_lookup.csv"

    first = pd.DataFrame(
        [
            _human_observation(
                "red",
                "2026-09-01T12:00:00Z",
            )
        ]
    )

    second = pd.DataFrame(
        [
            _human_observation(
                "red",
                "2026-09-02T12:00:00Z",
            )
        ]
    )

    first.to_csv(raw_dir / "session_1.csv", index=False)
    second.to_csv(raw_dir / "session_2.csv", index=False)

    lookup = rebuild_human_label_lookup(
        raw_dir=raw_dir,
        lookup_path=lookup_path,
    )

    assert lookup_path.is_file()
    assert len(lookup) == 1
    assert lookup.iloc[0]["human_label"] == "red"
    assert lookup.iloc[0]["n_human_labels"] == 2


def test_existing_label_coverage_is_mutually_exclusive():
    """Check exact matches take precedence over different-ratio matches."""

    selected = pd.DataFrame(
        [
            {
                "obj_r": 0.7,
                "obj_g": 0.3,
                "obj_b": 0.3,
                "bg_r": 0.2,
                "bg_g": 0.2,
                "bg_b": 0.2,
                "obj_diam_ratio": 0.5,
            },
            {
                "obj_r": 0.2,
                "obj_g": 0.7,
                "obj_b": 0.2,
                "bg_r": 0.1,
                "bg_g": 0.1,
                "bg_b": 0.1,
                "obj_diam_ratio": 0.5,
            },
            {
                "obj_r": 0.2,
                "obj_g": 0.2,
                "obj_b": 0.7,
                "bg_r": 0.4,
                "bg_g": 0.4,
                "bg_b": 0.4,
                "obj_diam_ratio": 0.5,
            },
        ]
    )

    lookup = pd.DataFrame(
        [
            {
                "obj_r": 0.7,
                "obj_g": 0.3,
                "obj_b": 0.3,
                "bg_r": 0.2,
                "bg_g": 0.2,
                "bg_b": 0.2,
                "obj_diam_ratio": 0.5,
                "human_label": "red",
                "n_human_labels": 2,
                "label_resolution": "majority",
                "last_labeled_at": "2026-09-01T12:00:00Z",
            },
            {
                "obj_r": 0.7,
                "obj_g": 0.3,
                "obj_b": 0.3,
                "bg_r": 0.2,
                "bg_g": 0.2,
                "bg_b": 0.2,
                "obj_diam_ratio": 0.4,
                "human_label": "green",
                "n_human_labels": 1,
                "label_resolution": "majority",
                "last_labeled_at": "2026-09-02T12:00:00Z",
            },
            {
                "obj_r": 0.2,
                "obj_g": 0.7,
                "obj_b": 0.2,
                "bg_r": 0.1,
                "bg_g": 0.1,
                "bg_b": 0.1,
                "obj_diam_ratio": 0.4,
                "human_label": "green",
                "n_human_labels": 1,
                "label_resolution": "majority",
                "last_labeled_at": "2026-09-03T12:00:00Z",
            },
        ]
    )

    coverage = existing_label_coverage(
        selected,
        lookup,
    )

    assert coverage.n_selected == 3
    assert coverage.n_same_ratio == 1
    assert coverage.n_different_ratio == 1
    assert coverage.n_any_match == 2
    assert coverage.n_no_match == 1


def test_apply_existing_labels_prefers_exact_ratio():
    """Check that exact-ratio labels beat different-ratio labels."""

    selected = pd.DataFrame(
        [
            {
                "obj_r": 0.7,
                "obj_g": 0.3,
                "obj_b": 0.3,
                "bg_r": 0.2,
                "bg_g": 0.2,
                "bg_b": 0.2,
                "obj_diam_ratio": 0.5,
                "user_label": "",
                "label_source": "",
                "matched_obj_diam_ratio": np.nan,
                "existing_n_human_labels": pd.NA,
                "existing_label_resolution": "",
                "existing_last_labeled_at": "",
            }
        ]
    )

    lookup = pd.DataFrame(
        [
            {
                "obj_r": 0.7,
                "obj_g": 0.3,
                "obj_b": 0.3,
                "bg_r": 0.2,
                "bg_g": 0.2,
                "bg_b": 0.2,
                "obj_diam_ratio": 0.4,
                "human_label": "green",
                "n_human_labels": 1,
                "label_resolution": "majority",
                "last_labeled_at": "2026-09-03T12:00:00Z",
            },
            {
                "obj_r": 0.7,
                "obj_g": 0.3,
                "obj_b": 0.3,
                "bg_r": 0.2,
                "bg_g": 0.2,
                "bg_b": 0.2,
                "obj_diam_ratio": 0.5,
                "human_label": "red",
                "n_human_labels": 2,
                "label_resolution": "majority",
                "last_labeled_at": "2026-09-01T12:00:00Z",
            },
        ]
    )

    result = apply_existing_labels(
        selected,
        lookup,
        use_existing=True,
        allow_different_ratio=True,
    )

    assert result.iloc[0]["user_label"] == "red"
    assert result.iloc[0]["label_source"] == "existing_same_ratio"
    assert result.iloc[0]["matched_obj_diam_ratio"] == pytest.approx(0.5)


def test_different_ratio_tie_uses_most_recent():
    """Check equal-distance ratio matches use the most recent row."""

    selected = pd.DataFrame(
        [
            {
                "obj_r": 0.7,
                "obj_g": 0.3,
                "obj_b": 0.3,
                "bg_r": 0.2,
                "bg_g": 0.2,
                "bg_b": 0.2,
                "obj_diam_ratio": 0.5,
                "user_label": "",
                "label_source": "",
                "matched_obj_diam_ratio": np.nan,
                "existing_n_human_labels": pd.NA,
                "existing_label_resolution": "",
                "existing_last_labeled_at": "",
            }
        ]
    )

    lookup = pd.DataFrame(
        [
            {
                "obj_r": 0.7,
                "obj_g": 0.3,
                "obj_b": 0.3,
                "bg_r": 0.2,
                "bg_g": 0.2,
                "bg_b": 0.2,
                "obj_diam_ratio": 0.4,
                "human_label": "red",
                "n_human_labels": 1,
                "label_resolution": "majority",
                "last_labeled_at": "2026-09-01T12:00:00Z",
            },
            {
                "obj_r": 0.7,
                "obj_g": 0.3,
                "obj_b": 0.3,
                "bg_r": 0.2,
                "bg_g": 0.2,
                "bg_b": 0.2,
                "obj_diam_ratio": 0.6,
                "human_label": "green",
                "n_human_labels": 1,
                "label_resolution": "majority",
                "last_labeled_at": "2026-09-03T12:00:00Z",
            },
        ]
    )

    result = apply_existing_labels(
        selected,
        lookup,
        use_existing=True,
        allow_different_ratio=True,
    )

    assert result.iloc[0]["user_label"] == "green"
    assert result.iloc[0]["label_source"] == "existing_different_ratio"
    assert result.iloc[0]["matched_obj_diam_ratio"] == pytest.approx(0.6)


def test_prompt_label_reuse_asks_two_questions():
    """Check the two one-time reuse decisions."""

    answers = iter(["y", "n"])
    printed = []

    coverage = LabelCoverage(
        n_selected=92,
        n_same_ratio=52,
        n_different_ratio=16,
    )

    result = prompt_label_reuse(
        coverage,
        input_func=lambda prompt: next(answers),
        print_func=printed.append,
    )

    assert result == (True, False)
    assert "68 of 92" in printed[0]


def test_randomize_pending_stimuli_excludes_reused_labels():
    """Check only unresolved stimuli are randomized."""

    audit_table = pd.DataFrame(
        {
            "stim_ID": [0, 1, 2, 3],
            "user_label": ["red", "", "green", ""],
        }
    )

    first = randomize_pending_stimuli(
        audit_table,
        random_seed=14,
    )

    second = randomize_pending_stimuli(
        audit_table,
        random_seed=14,
    )

    assert len(first) == 2
    assert set(first["stim_ID"]) == {1, 3}
    assert first["stim_ID"].tolist() == second["stim_ID"].tolist()
    assert first["audit_order"].tolist() == [0, 1]


def _make_session_inputs(tmp_path):
    """Create a tiny dataset and audit table for session tests."""

    dataset_dir = _make_dataset(tmp_path)
    dataset = load_audit_dataset(dataset_dir)

    config = AuditConfig(
        labels=["gray_d", "red"],
        bg_hues=["gray", "blue"],
        random_seed=14,
        filter_delta_int=False,
        delta_int_min=-1.0,
        delta_int_max=1.0,
        filter_delta_sat=False,
        delta_sat_min=-1.0,
        delta_sat_max=1.0,
    )

    table = select_audit_stimuli(
        dataset,
        config,
    )

    return dataset, config, table


def test_create_audit_session_writes_working_files(tmp_path):
    """Check that a new audit session is saved under interm-style storage."""

    dataset, config, table = _make_session_inputs(
        tmp_path
    )

    table.at[0, "user_label"] = "gray_d"
    table.at[0, "label_source"] = "existing_same_ratio"

    session = create_audit_session(
        table,
        dataset,
        config,
        use_existing=True,
        allow_different_ratio=False,
        base_dir=tmp_path / "audit_runs",
        created_at="2026-09-12T12:00:00Z",
    )

    assert isinstance(session, AuditSession)
    assert session.table_path.is_file()
    assert session.state_path.is_file()

    progress = audit_session_progress(
        session.table
    )

    assert progress == {
        "n_selected": 2,
        "n_reused": 1,
        "n_new_human": 0,
        "n_pending": 1,
    }

    reused = session.table.loc[
        session.table["stim_ID"] == 0
    ].iloc[0]

    pending = session.table.loc[
        session.table["stim_ID"] == 1
    ].iloc[0]

    assert pd.isna(reused["audit_order"])
    assert pending["audit_order"] == 0


def test_load_audit_session_preserves_order(tmp_path):
    """Check that a saved audit can be resumed with its order intact."""

    dataset, config, table = _make_session_inputs(
        tmp_path
    )

    session = create_audit_session(
        table,
        dataset,
        config,
        use_existing=False,
        allow_different_ratio=False,
        base_dir=tmp_path / "audit_runs",
        created_at="2026-09-12T12:00:00Z",
    )

    original_order = (
        session.table
        .sort_values("audit_order")
        ["stim_ID"]
        .tolist()
    )

    resumed = load_audit_session(
        session.session_dir
    )

    resumed_order = (
        resumed.table
        .sort_values("audit_order")
        ["stim_ID"]
        .tolist()
    )

    assert resumed_order == original_order
    assert resumed.state["status"] == "active"


def test_record_audit_label_saves_immediately(tmp_path):
    """Check one judgment is written and survives a reload."""

    dataset, config, table = _make_session_inputs(
        tmp_path
    )

    session = create_audit_session(
        table,
        dataset,
        config,
        use_existing=False,
        allow_different_ratio=False,
        base_dir=tmp_path / "audit_runs",
        created_at="2026-09-12T12:00:00Z",
    )

    next_row = next_pending_stimulus(
        session
    )

    record_audit_label(
        session,
        int(next_row["audit_order"]),
        "red",
        labeled_at="2026-09-12T12:01:00Z",
    )

    resumed = load_audit_session(
        session.session_dir
    )

    saved = resumed.table.loc[
        resumed.table["stim_ID"]
        == next_row["stim_ID"]
    ].iloc[0]

    assert saved["user_label"] == "red"
    assert saved["label_source"] == "new_human"
    assert saved["labeled_at"] == "2026-09-12T12:01:00+00:00"

    assert resumed.state["n_new_human"] == 1
    assert resumed.state["n_pending"] == 1


def test_completed_session_has_no_next_stimulus(tmp_path):
    """Check completion after every pending stimulus is labeled."""

    dataset, config, table = _make_session_inputs(
        tmp_path
    )

    table.at[0, "user_label"] = "gray_d"
    table.at[0, "label_source"] = "existing_same_ratio"

    session = create_audit_session(
        table,
        dataset,
        config,
        use_existing=True,
        allow_different_ratio=False,
        base_dir=tmp_path / "audit_runs",
        created_at="2026-09-12T12:00:00Z",
    )

    next_row = next_pending_stimulus(
        session
    )

    record_audit_label(
        session,
        int(next_row["audit_order"]),
        "red",
        labeled_at="2026-09-12T12:01:00Z",
    )

    assert next_pending_stimulus(session) is None
    assert session.state["status"] == "completed"
    assert session.state["n_pending"] == 0
    assert session.state["n_reused"] == 1
    assert session.state["n_new_human"] == 1


def test_record_audit_label_rejects_overwrite(tmp_path):
    """Check that an existing judgment cannot be silently replaced."""

    dataset, config, table = _make_session_inputs(
        tmp_path
    )

    session = create_audit_session(
        table,
        dataset,
        config,
        use_existing=False,
        allow_different_ratio=False,
        base_dir=tmp_path / "audit_runs",
        created_at="2026-09-12T12:00:00Z",
    )

    next_row = next_pending_stimulus(
        session
    )

    audit_order = int(
        next_row["audit_order"]
    )

    record_audit_label(
        session,
        audit_order,
        "red",
        labeled_at="2026-09-12T12:01:00Z",
    )

    with pytest.raises(
        ValueError,
        match="already labeled",
    ):
        record_audit_label(
            session,
            audit_order,
            "green",
        )


def test_finalize_writes_only_new_human_labels(tmp_path):
    """Check that reused labels are not written as new raw observations."""

    dataset, config, table = _make_session_inputs(
        tmp_path
    )

    table.at[0, "user_label"] = "gray_d"
    table.at[0, "label_source"] = "existing_same_ratio"

    session = create_audit_session(
        table,
        dataset,
        config,
        use_existing=True,
        allow_different_ratio=False,
        base_dir=tmp_path / "audit_runs",
        created_at="2026-09-12T12:00:00Z",
    )

    next_row = next_pending_stimulus(
        session
    )

    record_audit_label(
        session,
        int(next_row["audit_order"]),
        "red",
        labeled_at="2026-09-12T12:01:00Z",
    )

    raw_dir = tmp_path / "raw"
    lookup_path = (
        tmp_path
        / "interm"
        / "human_label_lookup.csv"
    )

    raw_path, lookup = finalize_audit_session(
        session,
        raw_dir=raw_dir,
        lookup_path=lookup_path,
        finalized_at="2026-09-12T12:02:00Z",
    )

    raw = pd.read_csv(
        raw_path
    )

    assert len(raw) == 1
    assert raw.iloc[0]["user_label"] == "red"
    assert raw.iloc[0]["source_stim_ID"] == int(
        next_row["stim_ID"]
    )

    assert lookup_path.is_file()
    assert len(lookup) == 1

    assert session.state["status"] == "finalized"
    assert session.state["n_finalized_new_human"] == 1


def test_finalize_rejects_incomplete_session(tmp_path):
    """Check that unfinished audits cannot enter permanent raw storage."""

    dataset, config, table = _make_session_inputs(
        tmp_path
    )

    session = create_audit_session(
        table,
        dataset,
        config,
        use_existing=False,
        allow_different_ratio=False,
        base_dir=tmp_path / "audit_runs",
        created_at="2026-09-12T12:00:00Z",
    )

    with pytest.raises(
        ValueError,
        match="remain unlabeled",
    ):
        finalize_audit_session(
            session,
            raw_dir=tmp_path / "raw",
            lookup_path=tmp_path / "lookup.csv",
        )


def test_finalize_is_idempotent(tmp_path):
    """Check that finalizing twice does not duplicate raw observations."""

    dataset, config, table = _make_session_inputs(
        tmp_path
    )

    session = create_audit_session(
        table,
        dataset,
        config,
        use_existing=False,
        allow_different_ratio=False,
        base_dir=tmp_path / "audit_runs",
        created_at="2026-09-12T12:00:00Z",
    )

    while True:
        row = next_pending_stimulus(
            session
        )

        if row is None:
            break

        record_audit_label(
            session,
            int(row["audit_order"]),
            "red",
            labeled_at=(
                "2026-09-12T12:01:00Z"
            ),
        )

    raw_dir = tmp_path / "raw"
    lookup_path = tmp_path / "lookup.csv"

    first_path, _ = finalize_audit_session(
        session,
        raw_dir=raw_dir,
        lookup_path=lookup_path,
        finalized_at="2026-09-12T12:02:00Z",
    )

    second_path, _ = finalize_audit_session(
        session,
        raw_dir=raw_dir,
        lookup_path=lookup_path,
        finalized_at="2026-09-12T12:03:00Z",
    )

    assert first_path == second_path

    raw_files = list(
        raw_dir.glob("*.csv")
    )

    assert len(raw_files) == 1

    raw = pd.read_csv(
        raw_files[0]
    )

    assert len(raw) == 2


def test_analyze_audit_session_counts_matches(tmp_path):
    """Check audit summary, crosstab, and mismatch exports."""

    dataset, config, table = _make_session_inputs(
        tmp_path
    )

    session = create_audit_session(
        table,
        dataset,
        config,
        use_existing=False,
        allow_different_ratio=False,
        base_dir=tmp_path / "audit_runs",
        created_at="2026-09-12T12:00:00Z",
    )

    pending = (
        session.table
        .dropna(subset=["audit_order"])
        .sort_values("audit_order")
    )

    first = pending.iloc[0]
    second = pending.iloc[1]

    record_audit_label(
        session,
        int(first["audit_order"]),
        str(first["true_label"]),
        labeled_at="2026-09-12T12:01:00Z",
    )

    wrong_label = (
        "green"
        if second["true_label"] != "green"
        else "red"
    )

    record_audit_label(
        session,
        int(second["audit_order"]),
        wrong_label,
        labeled_at="2026-09-12T12:02:00Z",
    )

    (
        summary,
        crosstab,
        mismatches,
        paths,
    ) = analyze_audit_session(
        session,
        save=True,
    )

    assert summary.iloc[0]["n_total"] == 2
    assert summary.iloc[0]["n_labeled"] == 2
    assert summary.iloc[0]["n_matches"] == 1
    assert summary.iloc[0]["n_mismatches"] == 1
    assert summary.iloc[0]["agreement_rate"] == pytest.approx(0.5)

    assert crosstab.to_numpy().sum() == 2
    assert len(mismatches) == 1

    assert paths["results"].is_file()
    assert paths["summary"].is_file()
    assert paths["crosstab"].is_file()
    assert paths["mismatches"].is_file()


def test_analyze_incomplete_audit_counts_unlabeled(tmp_path):
    """Check analysis also works before an audit is complete."""

    dataset, config, table = _make_session_inputs(
        tmp_path
    )

    session = create_audit_session(
        table,
        dataset,
        config,
        use_existing=False,
        allow_different_ratio=False,
        base_dir=tmp_path / "audit_runs",
        created_at="2026-09-12T12:00:00Z",
    )

    row = next_pending_stimulus(
        session
    )

    record_audit_label(
        session,
        int(row["audit_order"]),
        str(row["true_label"]),
        labeled_at="2026-09-12T12:01:00Z",
    )

    summary, _, mismatches, _ = analyze_audit_session(
        session,
        save=False,
    )

    assert summary.iloc[0]["n_total"] == 2
    assert summary.iloc[0]["n_labeled"] == 1
    assert summary.iloc[0]["n_unlabeled"] == 1
    assert summary.iloc[0]["n_matches"] == 1
    assert summary.iloc[0]["n_mismatches"] == 0
    assert mismatches.empty


def test_gallery_selects_configured_mid_setting(tmp_path):
    """Check gallery selection uses the saved generation ranges."""

    metadata = pd.DataFrame(
        {
            "stim_ID": [0, 1],
            "true_label": ["red", "red"],
            "bg_hue": ["gray", "blue"],
            "bg_r": [0.2, 0.1],
            "bg_g": [0.2, 0.2],
            "bg_b": [0.2, 0.4],
            "obj_r": [0.3, 0.5],
            "obj_g": [0.2, 0.25],
            "obj_b": [0.2, 0.25],
            "bg_int": [0.2, 0.4],
            "obj_int": [0.3, 0.5],
            "delta_int": [-0.1, -0.1],
            "bg_sat": [0.0, 0.5],
            "obj_sat": [0.3, 0.5],
            "delta_sat": [-0.3, 0.0],
        }
    )

    dataset_dir = _make_dataset(
        tmp_path,
        metadata=metadata,
    )

    dataset = load_audit_dataset(
        dataset_dir
    )

    gallery = GalleryConfig(
        target_label="red",
        n_cols=6,
        save_png=False,
        dpi=200,
    )

    selected, obj_int, obj_sat = select_gallery_stimuli(
        dataset,
        gallery,
    )

    assert obj_int == pytest.approx(0.5)
    assert obj_sat == pytest.approx(0.5)
    assert len(selected) == 1
    assert selected.iloc[0]["stim_ID"] == 1


def test_plot_stimulus_gallery_builds_figure(tmp_path):
    """Check the gallery figure can be constructed headlessly."""

    import matplotlib.pyplot as plt

    dataset_dir = _make_dataset(
        tmp_path
    )

    dataset = load_audit_dataset(
        dataset_dir
    )

    gallery = GalleryConfig(
        target_label="red",
        n_cols=2,
        save_png=False,
        dpi=200,
    )

    figure, selected, png_path = plot_stimulus_gallery(
        dataset,
        gallery,
    )

    assert figure is not None
    assert len(selected) == 1
    assert png_path is None

    plt.close(figure)