"""Interactive Matplotlib interface for human color labeling."""

from pathlib import Path

import numpy as np

from retinamodel.audit import (
    audit_session_progress,
    finalize_audit_session,
    load_audit_dataset,
    load_audit_session,
    next_pending_stimulus,
    record_audit_label,
)


KEY_TO_LABEL = {
    "1": "gray_d",
    "2": "gray_l",
    "3": "red",
    "4": "green",
    "5": "blue",
    "6": "yellow",
}

KEY_MAP_TEXT = (
    "1=gray_d   "
    "2=gray_l   "
    "3=red   "
    "4=green   "
    "5=blue   "
    "6=yellow"
)


def backend_is_interactive(backend):
    """Return whether a Matplotlib backend can support this audit UI."""

    backend = str(backend).strip().lower()

    noninteractive = {
        "agg",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "cairo",
        "template",
    }

    if backend in noninteractive:
        return False

    if "matplotlib_inline" in backend:
        return False

    if backend.endswith("backend_inline"):
        return False

    return True


def matplotlib_backend():
    """Return the currently selected Matplotlib backend."""

    import matplotlib

    return str(matplotlib.get_backend())


def run_audit_ui(
    session_dir,
    finalize_on_complete=True,
    raw_dir="data/raw/human_labels",
    lookup_path="data/interm/human_labels/human_label_lookup.csv",
):
    """Run a blind keyboard-driven human-labeling session."""

    import matplotlib
    import matplotlib.pyplot as plt

    backend = str(
        matplotlib.get_backend()
    )

    if not backend_is_interactive(backend):
        raise RuntimeError(
            "The current Matplotlib backend "
            f"('{backend}') is not interactive. "
            "Run the audit from a Python environment "
            "with an interactive GUI backend."
        )

    session = load_audit_session(
        session_dir
    )

    dataset_path = session.state.get(
        "dataset_dir"
    )

    if not dataset_path:
        raise ValueError(
            "Audit session does not identify its source dataset."
        )

    dataset = load_audit_dataset(
        dataset_path
    )

    total_to_label = int(
        session.table["audit_order"]
        .notna()
        .sum()
    )

    figure, axis = plt.subplots(
        figsize=(7, 7)
    )

    finalization_done = (
        session.state.get("status")
        == "finalized"
    )

    def show_complete():
        """Show completion and finalize once."""

        nonlocal finalization_done

        axis.clear()
        axis.axis("off")

        message = (
            "Audit complete.\n"
            f"Session saved to:\n{session.session_dir}"
        )

        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

        figure.suptitle(
            "Audit complete"
        )

        figure.canvas.draw_idle()

        if (
            finalize_on_complete
            and not finalization_done
        ):
            raw_path, lookup = finalize_audit_session(
                session,
                raw_dir=raw_dir,
                lookup_path=lookup_path,
            )

            finalization_done = True

            if raw_path is None:
                print(
                    "Audit finalized. "
                    "No new human judgments required."
                )
            else:
                print(
                    "New human judgments saved to:",
                    raw_path,
                )

            print(
                "Human-label lookup rows:",
                len(lookup),
            )

    def show_next():
        """Display the next pending stimulus without revealing metadata."""

        row = next_pending_stimulus(
            session
        )

        if row is None:
            show_complete()
            return

        stim_id = int(
            row["stim_ID"]
        )

        audit_order = int(
            row["audit_order"]
        )

        image = np.asarray(
            dataset.imgs[stim_id]
        )

        axis.clear()

        axis.imshow(
            np.clip(
                image,
                0.0,
                1.0,
            )
        )

        axis.axis("off")

        figure.suptitle(
            f"Audit {audit_order + 1}/{total_to_label}\n"
            f"{KEY_MAP_TEXT}"
        )

        figure.canvas.draw_idle()

    def on_key(event):
        """Record a valid keyboard label and advance."""

        user_label = KEY_TO_LABEL.get(
            event.key
        )

        if user_label is None:
            return

        row = next_pending_stimulus(
            session
        )

        if row is None:
            return

        record_audit_label(
            session,
            int(row["audit_order"]),
            user_label,
        )

        show_next()

    figure.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    show_next()

    plt.show(
        block=True
    )

    progress = audit_session_progress(
        session.table
    )

    if progress["n_pending"] > 0:
        print(
            "Audit paused."
        )
        print(
            "Remaining stimuli:",
            progress["n_pending"],
        )
        print(
            "Resume from:",
            Path(session.session_dir),
        )

    return session