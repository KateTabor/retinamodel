"""Command-line interface for shared classifier-data preparation."""

import argparse

from retinamodel.classifier_data import build_classifier_data_bundle, summarize_classifier_data_bundle


DEFAULT_CONFIG = "config/classifier_data.toml"


def _build_command(args):
    """Build and record the shared classifier-data bundle."""

    print("Classifier-data build")
    print("Configuration:", args.config)

    bundle = build_classifier_data_bundle(args.config, output_dir=args.output_dir, record=True)

    print()
    print("Classifier-data build complete")
    print(summarize_classifier_data_bundle(bundle))


def build_parser():
    """Create the classifier-data command-line parser."""

    parser = argparse.ArgumentParser(
        prog="classifier-data",
        description="Build the shared, reproducible data bundle used by classifier models.",
    )

    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the classifier-data TOML configuration.")
    parser.add_argument("--output-dir", default=None, help="Optional exact output directory. Otherwise the configured output root is used.")

    parser.set_defaults(func=_build_command)
    return parser


def main():
    """Run the classifier-data command-line interface."""

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
