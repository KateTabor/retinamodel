"""Tests for the human-audit command-line parser."""

from retinamodel.audit_cli import (
    DEFAULT_CONFIG,
    DEFAULT_LOOKUP_PATH,
    DEFAULT_RAW_DIR,
    DEFAULT_RUNS_DIR,
    build_parser,
)


def test_audit_start_parser():
    """Check parsing of the start command."""

    args = build_parser().parse_args(
        [
            "start",
            "--dataset",
            "data/interm/example",
        ]
    )

    assert args.command == "start"
    assert args.dataset == "data/interm/example"
    assert args.config == DEFAULT_CONFIG
    assert args.runs_dir == DEFAULT_RUNS_DIR
    assert args.raw_dir == DEFAULT_RAW_DIR
    assert args.lookup_path == DEFAULT_LOOKUP_PATH


def test_audit_resume_parser():
    """Check parsing of the resume command."""

    args = build_parser().parse_args(
        [
            "resume",
            "data/interm/audit_runs/example",
        ]
    )

    assert args.command == "resume"
    assert (
        args.session_dir
        == "data/interm/audit_runs/example"
    )


def test_audit_analyze_parser():
    """Check parsing of the analyze command."""

    args = build_parser().parse_args(
        [
            "analyze",
            "data/interm/audit_runs/example",
        ]
    )

    assert args.command == "analyze"
    assert (
        args.session_dir
        == "data/interm/audit_runs/example"
    )


def test_audit_gallery_parser():
    """Check parsing of the gallery command."""

    args = build_parser().parse_args(
        [
            "gallery",
            "--dataset",
            "data/interm/example",
        ]
    )

    assert args.command == "gallery"
    assert args.dataset == "data/interm/example"
    assert args.config == DEFAULT_CONFIG