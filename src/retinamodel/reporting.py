"""Formatting helpers for command-line reports."""


import math


def format_number(value, significant_digits=2):
    """Format a number with a compact number of significant digits."""
    return f"{value:.{significant_digits}g}"


def format_mrgc_report(model):
    """Return a concise report of the mRGC reference calculations."""
    return "\n".join(
        [
            "mRGC reference outputs",
            (
                "White DoG: "
                f"L={format_number(model.dog_l_white)}, "
                f"M={format_number(model.dog_m_white)}, "
                f"H2={format_number(model.dog_h2_white)}, "
                f"S={format_number(model.dog_s_white)}"
            ),
            (
                "Yellow DoG: "
                f"L={format_number(model.dog_l_yellow)}, "
                f"M={format_number(model.dog_m_yellow)}"
            ),
            (
                "Yellow DoG fraction of center: "
                f"L={format_number(model.percent_dog_l_yellow)}, "
                f"M={format_number(model.percent_dog_m_yellow)}"
            ),
        ]
    )


def format_stimulus_report(result):
    """Return a concise report of a completed stimulus export."""
    summary = result["summary"]

    return "\n".join(
        [
            "Stimulus export complete",
            f"Dataset: {result['export_dir']}",
            f"Raw stimuli: {summary['n_total_raw']}",
            f"Unique stimuli: {summary['n_total_unique']}",
            f"Redundant stimuli removed: {summary['n_redundant']}",
        ]
    )


def format_audit_session_report(session, title="Audit session"):
    """Return a concise report of an audit-session state."""

    state = session.state

    return "\n".join(
        [
            title,
            f"Session: {session.session_dir}",
            f"Status: {state.get('status', 'unknown')}",
            f"Selected stimuli: {state.get('n_selected', len(session.table))}",
            f"Reused labels: {state.get('n_reused', 0)}",
            f"New human labels: {state.get('n_new_human', 0)}",
            f"Remaining: {state.get('n_pending', 0)}",
        ]
    )


def format_audit_analysis_report(summary, paths):
    """Return a concise report of audit-analysis results."""

    row = summary.iloc[0]

    agreement_rate = float(
        row["agreement_rate"]
    )

    if math.isfinite(agreement_rate):
        agreement_text = (
            f"{100.0 * agreement_rate:.1f}%"
        )
    else:
        agreement_text = "n/a"

    lines = [
        "Audit analysis complete",
        f"Selected stimuli: {int(row['n_total'])}",
        f"Labeled: {int(row['n_labeled'])}",
        f"Unlabeled: {int(row['n_unlabeled'])}",
        f"Matches: {int(row['n_matches'])}",
        f"Mismatches: {int(row['n_mismatches'])}",
        f"Agreement: {agreement_text}",
    ]

    if paths:
        lines.extend(
            [
                f"Results: {paths['results']}",
                f"Summary: {paths['summary']}",
                f"Crosstab: {paths['crosstab']}",
                f"Mismatches: {paths['mismatches']}",
            ]
        )

    return "\n".join(lines)


def format_gallery_report(
    dataset,
    gallery_config,
    selected,
    png_path,
):
    """Return a concise report of a stimulus gallery."""

    lines = [
        "Stimulus gallery",
        f"Dataset: {dataset.dataset_dir}",
        f"Target label: {gallery_config.target_label}",
        f"Images: {len(selected)}",
    ]

    if png_path is None:
        lines.append("PNG: not saved")
    else:
        lines.append(f"PNG: {png_path}")

    return "\n".join(lines)