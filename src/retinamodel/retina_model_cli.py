"""Command-line interface for retinal-response processing."""

import argparse

from retinamodel.retina_model import (
    build_retina_model,
    estimate_retina_model_storage,
    load_retina_input_dataset,
)


def _build_command(args):
    """Build a completed Module 4 retinal-response dataset."""

    dataset = load_retina_input_dataset(
        args.dataset
    )

    sizes = estimate_retina_model_storage(
        dataset
    )

    print("Retinal-response build")
    print("Source dataset:", dataset.dataset_dir)
    print("Input responses:", dataset.outs.shape)
    print("Estimated uncompressed storage:")
    print(
        "  extrema_mask:",
        f"{sizes['extrema_mask_gib']:.3f} GiB",
    )
    print(
        "  outs_fill:",
        f"{sizes['outs_fill_gib']:.3f} GiB",
    )
    print(
        "  total:",
        f"{sizes['total_gib']:.3f} GiB",
    )
    print()

    paths = build_retina_model(
        dataset_dir=args.dataset,
        config_path=args.config,
        base_dir=args.output_dir,
        show_progress=True,
    )

    print()
    print("Retinal-response build complete")
    print("Run directory:", paths.run_dir)
    print("Retinal outputs:", paths.outputs_zarr)
    print("Metadata:", paths.metadata_csv)
    print(
        "Processing settings:",
        paths.processing_settings_json,
    )


def _launch_viewer(run_dir, config_path):
    """Import and launch the graphical viewer only when requested."""

    from retinamodel.retina_model_viewer_ui import (
        launch_retina_model_viewer,
    )

    launch_retina_model_viewer(
        run_dir=run_dir,
        config_path=config_path,
    )


def _view_command(args):
    """Open an existing completed Module 4 run."""

    _launch_viewer(
        run_dir=args.run,
        config_path=args.config,
    )


def build_parser():
    """Create the retina-response command-line parser."""

    parser = argparse.ArgumentParser(
        prog="retina-response",
        description=(
            "Build and inspect completed retinal-response datasets."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    build = subparsers.add_parser(
        "build",
        help=(
            "Create extrema masks and final filled retinal responses."
        ),
    )

    build.add_argument(
        "--dataset",
        required=True,
        help="Path to the Module 2 retina_dataset_<id> directory.",
    )

    build.add_argument(
        "--config",
        default="config/retina_model.toml",
        help="Path to the Module 4 TOML configuration.",
    )

    build.add_argument(
        "--output-dir",
        default="data/interm",
        help="Directory in which retina_model_<id> is created.",
    )

    build.set_defaults(
        func=_build_command
    )

    view = subparsers.add_parser(
        "view",
        help="Open an interactive completed retinal-response run.",
    )

    view.add_argument(
        "--run",
        required=True,
        help="Path to the retina_model_<id> run directory.",
    )

    view.add_argument(
        "--config",
        default="config/retina_model.toml",
        help="Path to the Module 4 TOML configuration.",
    )

    view.set_defaults(
        func=_view_command
    )

    return parser


def main():
    """Run the retina-response command-line interface."""

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()