"""Command-line workflow for human stimulus auditing."""

import argparse

from retinamodel.audit import (
    analyze_audit_session,
    apply_existing_labels,
    create_audit_session,
    existing_label_coverage,
    finalize_audit_session,
    load_audit_config,
    load_audit_dataset,
    load_audit_session,
    plot_stimulus_gallery,
    prompt_label_reuse,
    rebuild_human_label_lookup,
    select_audit_stimuli,
)
from retinamodel.audit_ui import run_audit_ui
from retinamodel.reporting import (
    format_audit_analysis_report,
    format_audit_session_report,
    format_gallery_report,
)


DEFAULT_CONFIG = "config/audit.toml"
DEFAULT_RUNS_DIR = "data/interm/audit_runs"
DEFAULT_RAW_DIR = "data/raw/human_labels"
DEFAULT_LOOKUP_PATH = (
    "data/interm/human_labels/"
    "human_label_lookup.csv"
)


def _start_audit(args):
    """Start a new audit from a stimulus dataset."""

    audit_config, _ = load_audit_config(
        args.config
    )

    dataset = load_audit_dataset(
        args.dataset
    )

    selected = select_audit_stimuli(
        dataset,
        audit_config,
    )

    lookup = rebuild_human_label_lookup(
        raw_dir=args.raw_dir,
        lookup_path=args.lookup_path,
    )

    coverage = existing_label_coverage(
        selected,
        lookup,
    )

    (
        use_existing,
        allow_different_ratio,
    ) = prompt_label_reuse(
        coverage
    )

    selected = apply_existing_labels(
        selected,
        lookup,
        use_existing=use_existing,
        allow_different_ratio=allow_different_ratio,
    )

    session = create_audit_session(
        selected,
        dataset,
        audit_config,
        use_existing=use_existing,
        allow_different_ratio=allow_different_ratio,
        base_dir=args.runs_dir,
    )

    print()
    print(
        format_audit_session_report(
            session,
            title="Audit session created",
        )
    )

    if session.state["n_pending"] == 0:
        finalize_audit_session(
            session,
            raw_dir=args.raw_dir,
            lookup_path=args.lookup_path,
        )

        print()
        print(
            format_audit_session_report(
                session,
                title="Audit session finalized",
            )
        )

        return

    session = run_audit_ui(
        session.session_dir,
        finalize_on_complete=True,
        raw_dir=args.raw_dir,
        lookup_path=args.lookup_path,
    )

    print()
    print(
        format_audit_session_report(
            session,
            title="Audit session finished",
        )
    )


def _resume_audit(args):
    """Resume an existing audit session."""

    session = load_audit_session(
        args.session_dir
    )

    if session.state.get("status") == "finalized":
        print(
            format_audit_session_report(
                session,
                title="Audit session already finalized",
            )
        )
        return

    if session.state.get("n_pending", 0) == 0:
        finalize_audit_session(
            session,
            raw_dir=args.raw_dir,
            lookup_path=args.lookup_path,
        )

        print(
            format_audit_session_report(
                session,
                title="Audit session finalized",
            )
        )
        return

    session = run_audit_ui(
        session.session_dir,
        finalize_on_complete=True,
        raw_dir=args.raw_dir,
        lookup_path=args.lookup_path,
    )

    print()
    print(
        format_audit_session_report(
            session,
            title="Audit session finished",
        )
    )


def _analyze_audit(args):
    """Analyze a saved audit session."""

    session = load_audit_session(
        args.session_dir
    )

    (
        summary,
        _,
        _,
        paths,
    ) = analyze_audit_session(
        session,
        save=True,
    )

    print(
        format_audit_analysis_report(
            summary,
            paths,
        )
    )


def _show_gallery(args):
    """Display the configured stimulus gallery."""

    _, gallery_config = load_audit_config(
        args.config
    )

    if args.save_png:
        gallery_config.save_png = True

    dataset = load_audit_dataset(
        args.dataset
    )

    (
        figure,
        selected,
        png_path,
    ) = plot_stimulus_gallery(
        dataset,
        gallery_config,
    )

    print(
        format_gallery_report(
            dataset,
            gallery_config,
            selected,
            png_path,
        )
    )

    import matplotlib.pyplot as plt

    plt.show(
        block=True
    )

    return figure


def build_parser():
    """Build the human-audit command-line parser."""

    parser = argparse.ArgumentParser(
        prog="audit",
        description=(
            "Start, resume, analyze, or inspect "
            "human stimulus audits."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    start_parser = subparsers.add_parser(
        "start",
        help="Start a new human audit.",
    )

    start_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a Module 2 retina dataset.",
    )

    start_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to audit.toml.",
    )

    start_parser.add_argument(
        "--runs-dir",
        default=DEFAULT_RUNS_DIR,
        help="Directory for resumable audit sessions.",
    )

    start_parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help="Directory containing permanent raw human labels.",
    )

    start_parser.add_argument(
        "--lookup-path",
        default=DEFAULT_LOOKUP_PATH,
        help="Path to the generated human-label lookup CSV.",
    )

    start_parser.set_defaults(
        handler=_start_audit
    )

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an existing human audit.",
    )

    resume_parser.add_argument(
        "session_dir",
        help="Path to the audit-session directory.",
    )

    resume_parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help="Directory containing permanent raw human labels.",
    )

    resume_parser.add_argument(
        "--lookup-path",
        default=DEFAULT_LOOKUP_PATH,
        help="Path to the generated human-label lookup CSV.",
    )

    resume_parser.set_defaults(
        handler=_resume_audit
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a saved audit session.",
    )

    analyze_parser.add_argument(
        "session_dir",
        help="Path to the audit-session directory.",
    )

    analyze_parser.set_defaults(
        handler=_analyze_audit
    )

    gallery_parser = subparsers.add_parser(
        "gallery",
        help="Display a stimulus gallery.",
    )

    gallery_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a Module 2 retina dataset.",
    )

    gallery_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to audit.toml.",
    )

    gallery_parser.add_argument(
    "--save-png",
    action="store_true",
    help="Save the gallery PNG to figures/.",
    )

    gallery_parser.set_defaults(
        handler=_show_gallery
    )

    return parser


def main(argv=None):
    """Run the human-audit command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)