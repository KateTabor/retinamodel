"""Formatting helpers for command-line reports."""


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