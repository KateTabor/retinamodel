"""Tk interface for viewing completed Module 4 retinal-response runs."""

from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from retinamodel.retina_model import load_retina_model_config
from retinamodel.retina_model_viewer import (
    filter_stimuli,
    format_metadata_text,
    initial_filter_value,
    load_retina_model_viewer_data,
    metadata_filter_values,
    render_stimulus_figure,
)


ALL_FILTER = "All"

MAP_DISPLAY_TO_KEY = {
    "Filled response": "filled",
    "Extrema mask": "extrema",
}


class RetinaModelViewerWindow:
    """Interactive viewer for one completed Module 4 run."""

    def __init__(
        self,
        root,
        data,
        viewer_config,
    ):
        self.root = root
        self.data = data
        self.viewer_config = viewer_config

        self.selected = None
        self.index = 0

        self.obj_values = metadata_filter_values(
            data.metadata,
            "true_label",
        )

        self.bg_values = metadata_filter_values(
            data.metadata,
            "bg_hue",
        )

        obj_default = initial_filter_value(
            self.obj_values,
            viewer_config.obj_label,
        )

        bg_default = initial_filter_value(
            self.bg_values,
            viewer_config.bg_label,
        )

        self.obj_var = tk.StringVar(
            value=obj_default or ALL_FILTER
        )

        self.bg_var = tk.StringVar(
            value=bg_default or ALL_FILTER
        )

        self.map_var = tk.StringVar(
            value="Filled response"
        )

        self.status_var = tk.StringVar()
        self.message_var = tk.StringVar()

        self._build_window()
        self._bind_keys()
        self._apply_filters()

    def _build_window(self):
        """Create a viewer whose controls and navigation remain visible."""

        self.root.title(
            f"retina-response viewer — {self.data.run_dir.name}"
        )

        self.root.geometry(
            "1200x900"
        )

        self.root.minsize(
            900,
            700,
        )

        self.root.grid_rowconfigure(
            1,
            weight=1,
        )

        self.root.grid_columnconfigure(
            0,
            weight=1,
        )

        self._build_controls()
        self._build_main_area()
        self._build_navigation()

    def _build_controls(self):
        """Create the fixed top control bar."""

        controls = ttk.Frame(
            self.root,
            padding=8,
        )

        controls.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        column = 0

        if self.obj_values:
            ttk.Label(
                controls,
                text="Object/label:",
            ).grid(
                row=0,
                column=column,
                padx=(0, 4),
            )

            column += 1

            self.obj_menu = ttk.Combobox(
                controls,
                textvariable=self.obj_var,
                values=[
                    ALL_FILTER,
                    *self.obj_values,
                ],
                state="readonly",
                width=14,
            )

            self.obj_menu.grid(
                row=0,
                column=column,
                padx=(0, 14),
            )

            self.obj_menu.bind(
                "<<ComboboxSelected>>",
                self._on_filter_change,
            )

            column += 1

        if self.bg_values:
            ttk.Label(
                controls,
                text="Background hue:",
            ).grid(
                row=0,
                column=column,
                padx=(0, 4),
            )

            column += 1

            self.bg_menu = ttk.Combobox(
                controls,
                textvariable=self.bg_var,
                values=[
                    ALL_FILTER,
                    *self.bg_values,
                ],
                state="readonly",
                width=14,
            )

            self.bg_menu.grid(
                row=0,
                column=column,
                padx=(0, 14),
            )

            self.bg_menu.bind(
                "<<ComboboxSelected>>",
                self._on_filter_change,
            )

            column += 1

        ttk.Label(
            controls,
            text="Map type:",
        ).grid(
            row=0,
            column=column,
            padx=(0, 4),
        )

        column += 1

        self.map_menu = ttk.Combobox(
            controls,
            textvariable=self.map_var,
            values=list(MAP_DISPLAY_TO_KEY),
            state="readonly",
            width=18,
        )

        self.map_menu.grid(
            row=0,
            column=column,
            padx=(0, 14),
        )

        self.map_menu.bind(
            "<<ComboboxSelected>>",
            self._on_map_change,
        )

        column += 1

        ttk.Button(
            controls,
            text="Show All",
            command=self._show_all,
        ).grid(
            row=0,
            column=column,
            padx=(0, 8),
        )

        column += 1

        ttk.Button(
            controls,
            text="Reset Defaults",
            command=self._reset_filters,
        ).grid(
            row=0,
            column=column,
        )

    def _build_main_area(self):
        """Create the resizable figure plus stimulus metadata panel."""

        main = ttk.Frame(
            self.root,
            padding=(8, 0, 8, 0),
        )

        main.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        main.grid_rowconfigure(
            0,
            weight=1,
        )

        main.grid_columnconfigure(
            0,
            weight=1,
        )

        figure_frame = ttk.Frame(
            main,
        )

        figure_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.figure = Figure(
            figsize=(9.0, 7.0),
            dpi=100,
        )

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=figure_frame,
        )

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
        )

        info = ttk.LabelFrame(
            main,
            text="Current stimulus metadata",
            padding=8,
        )

        info.grid(
            row=0,
            column=1,
            sticky="ns",
            padx=(8, 0),
        )

        self.metadata_text = tk.Text(
            info,
            width=31,
            height=18,
            wrap="word",
        )

        self.metadata_text.pack(
            fill="both",
            expand=True,
        )

        self.metadata_text.configure(
            state="disabled",
        )

    def _build_navigation(self):
        """Create the fixed bottom paging controls."""

        navigation = ttk.Frame(
            self.root,
            padding=8,
        )

        navigation.grid(
            row=2,
            column=0,
            sticky="ew",
        )

        navigation.grid_columnconfigure(
            1,
            weight=1,
        )

        self.prev_button = ttk.Button(
            navigation,
            text="← Previous",
            command=self._previous,
        )

        self.prev_button.grid(
            row=0,
            column=0,
            padx=(0, 10),
        )

        ttk.Label(
            navigation,
            textvariable=self.status_var,
            anchor="center",
        ).grid(
            row=0,
            column=1,
            sticky="ew",
        )

        self.next_button = ttk.Button(
            navigation,
            text="Next →",
            command=self._next,
        )

        self.next_button.grid(
            row=0,
            column=2,
            padx=(10, 10),
        )

        self.save_button = ttk.Button(
            navigation,
            text="Save PNG",
            command=self._save_png,
        )

        self.save_button.grid(
            row=0,
            column=3,
        )

        ttk.Label(
            navigation,
            textvariable=self.message_var,
            anchor="w",
        ).grid(
            row=1,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(5, 0),
        )

    def _bind_keys(self):
        """Allow keyboard paging through the selected stimuli."""

        self.root.bind(
            "<Left>",
            self._previous,
        )

        self.root.bind(
            "<Right>",
            self._next,
        )

        self.root.bind(
            "<Prior>",
            self._previous,
        )

        self.root.bind(
            "<Next>",
            self._next,
        )

    def _current_filter_values(self):
        """Return active optional object/background filters."""

        true_label = None
        bg_hue = None

        if (
            self.obj_values
            and self.obj_var.get() != ALL_FILTER
        ):
            true_label = self.obj_var.get()

        if (
            self.bg_values
            and self.bg_var.get() != ALL_FILTER
        ):
            bg_hue = self.bg_var.get()

        return true_label, bg_hue

    def _apply_filters(self):
        """Apply active filters and start at the first matching stimulus."""

        true_label, bg_hue = self._current_filter_values()

        self.selected = filter_stimuli(
            self.data.metadata,
            true_label=true_label,
            bg_hue=bg_hue,
        )

        self.index = 0
        self.message_var.set("")

        if len(self.selected) == 0:
            self._show_no_matches()
            return

        self._set_navigation_enabled(
            True
        )

        self._render_current()

    def _on_filter_change(self, event=None):
        """Refresh after changing an object or background filter."""

        self._apply_filters()

    def _on_map_change(self, event=None):
        """Refresh after changing the displayed map type."""

        if (
            self.selected is not None
            and len(self.selected) > 0
        ):
            self._render_current()

    def _show_all(self):
        """Clear object/background filtering."""

        if self.obj_values:
            self.obj_var.set(
                ALL_FILTER
            )

        if self.bg_values:
            self.bg_var.set(
                ALL_FILTER
            )

        self._apply_filters()

    def _reset_filters(self):
        """Restore object/background defaults from the viewer config."""

        if self.obj_values:
            value = initial_filter_value(
                self.obj_values,
                self.viewer_config.obj_label,
            )

            self.obj_var.set(
                value or ALL_FILTER
            )

        if self.bg_values:
            value = initial_filter_value(
                self.bg_values,
                self.viewer_config.bg_label,
            )

            self.bg_var.set(
                value or ALL_FILTER
            )

        self._apply_filters()

    def _current_stim_id(self):
        """Return the current stimulus ID."""

        return int(
            self.selected.iloc[
                self.index
            ]["stim_ID"]
        )

    def _current_map_type(self):
        """Return the internal map-type key."""

        return MAP_DISPLAY_TO_KEY[
            self.map_var.get()
        ]

    def _current_metadata_row(self):
        """Return the metadata row for the current stimulus."""

        stim_id = self._current_stim_id()

        return self.data.metadata[
            self.data.metadata["stim_ID"] == stim_id
        ].iloc[0]

    def _render_current(self):
        """Render the current stimulus and update viewer information."""

        stim_id = self._current_stim_id()
        map_type = self._current_map_type()

        render_stimulus_figure(
            data=self.data,
            stim_id=stim_id,
            map_type=map_type,
            viewer_config=self.viewer_config,
            figure=self.figure,
        )

        self.canvas.draw_idle()

        row = self._current_metadata_row()

        self._set_metadata_text(
            format_metadata_text(row)
        )

        self.status_var.set(
            f"Stimulus {self.index + 1} of {len(self.selected)} "
            f"(stim_ID={stim_id})"
        )

        self._update_navigation_buttons()

    def _previous(self, event=None):
        """Move to the previous matching stimulus."""

        if (
            self.selected is None
            or len(self.selected) == 0
            or self.index <= 0
        ):
            return

        self.index -= 1
        self.message_var.set("")
        self._render_current()

    def _next(self, event=None):
        """Move to the next matching stimulus."""

        if (
            self.selected is None
            or len(self.selected) == 0
            or self.index >= len(self.selected) - 1
        ):
            return

        self.index += 1
        self.message_var.set("")
        self._render_current()

    def _update_navigation_buttons(self):
        """Enable Previous/Next only when another page exists."""

        if (
            self.selected is None
            or len(self.selected) == 0
        ):
            self.prev_button.configure(
                state="disabled"
            )
            self.next_button.configure(
                state="disabled"
            )
            return

        self.prev_button.configure(
            state=(
                "normal"
                if self.index > 0
                else "disabled"
            )
        )

        self.next_button.configure(
            state=(
                "normal"
                if self.index < len(self.selected) - 1
                else "disabled"
            )
        )

    def _save_png(self):
        """Save the current stimulus/maps figure inside this run."""

        if (
            self.selected is None
            or len(self.selected) == 0
        ):
            return

        stim_id = self._current_stim_id()
        map_type = self._current_map_type()

        output_dir = Path("figures")
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"{self.data.run_dir.name}_"
            f"stim{stim_id:06d}_"
            f"{map_type}_"
            f"{timestamp}.png"
        )

        output_path = (
            output_dir
            / filename
        )

        save_figure = render_stimulus_figure(
            data=self.data,
            stim_id=stim_id,
            map_type=map_type,
            viewer_config=self.viewer_config,
        )

        save_figure.savefig(
            output_path,
            dpi=self.viewer_config.dpi,
            bbox_inches="tight",
        )

        self.message_var.set(
            f"Saved: {output_path}"
        )

    def _set_metadata_text(self, text):
        """Replace the read-only metadata display."""

        self.metadata_text.configure(
            state="normal"
        )

        self.metadata_text.delete(
            "1.0",
            "end",
        )

        self.metadata_text.insert(
            "1.0",
            text,
        )

        self.metadata_text.configure(
            state="disabled"
        )

    def _set_navigation_enabled(self, enabled):
        """Enable or disable viewer navigation/save controls."""

        state = (
            "normal"
            if enabled
            else "disabled"
        )

        self.save_button.configure(
            state=state
        )

        if enabled:
            self._update_navigation_buttons()
        else:
            self.prev_button.configure(
                state="disabled"
            )

            self.next_button.configure(
                state="disabled"
            )

    def _show_no_matches(self):
        """Display a clear message when filters select no stimuli."""

        self.figure.clear()

        ax = self.figure.add_subplot(
            111
        )

        ax.text(
            0.5,
            0.5,
            "No stimuli match the selected filters.",
            ha="center",
            va="center",
        )

        ax.axis(
            "off"
        )

        self.canvas.draw_idle()

        self._set_metadata_text(
            ""
        )

        self.status_var.set(
            "No matching stimuli"
        )

        self._set_navigation_enabled(
            False
        )


def launch_retina_model_viewer(
    run_dir,
    config_path="config/retina_model.toml",
):
    """Open the interactive viewer for one completed Module 4 run."""

    data = load_retina_model_viewer_data(
        run_dir
    )

    config = load_retina_model_config(
        config_path
    )

    root = tk.Tk()

    RetinaModelViewerWindow(
        root=root,
        data=data,
        viewer_config=config.viewer,
    )

    root.mainloop()
