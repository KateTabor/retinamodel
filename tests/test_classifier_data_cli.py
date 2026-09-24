"""Tests for the classifier-data command-line interface."""

from types import SimpleNamespace

from retinamodel import classifier_data_cli


def test_build_parser_defaults():
    """Check the default classifier-data command settings."""

    parser = classifier_data_cli.build_parser()
    args = parser.parse_args([])

    assert args.config == "config/classifier_data.toml"
    assert args.output_dir is None
    assert args.func is classifier_data_cli._build_command


def test_build_parser_custom_paths():
    """Check custom configuration and output paths."""

    parser = classifier_data_cli.build_parser()
    args = parser.parse_args(["--config", "config/custom.toml", "--output-dir", "data/interm/custom_run"])

    assert args.config == "config/custom.toml"
    assert args.output_dir == "data/interm/custom_run"


def test_build_command_calls_classifier_data(monkeypatch, capsys):
    """Check the CLI delegates work to the classifier-data pipeline."""

    bundle = SimpleNamespace()
    calls = {}

    def fake_build(config_path, output_dir=None, record=True):
        calls["config_path"] = config_path
        calls["output_dir"] = output_dir
        calls["record"] = record
        return bundle

    def fake_summary(input_bundle):
        assert input_bundle is bundle
        return "Dataset: 392 stimuli | selected: 80 | unused: 312"

    monkeypatch.setattr(classifier_data_cli, "build_classifier_data_bundle", fake_build)
    monkeypatch.setattr(classifier_data_cli, "summarize_classifier_data_bundle", fake_summary)

    args = SimpleNamespace(config="config/classifier_data.toml", output_dir="data/interm/classifier_data_test")
    classifier_data_cli._build_command(args)

    assert calls == {
        "config_path": "config/classifier_data.toml",
        "output_dir": "data/interm/classifier_data_test",
        "record": True,
    }

    output = capsys.readouterr().out

    assert "Classifier-data build" in output
    assert "Classifier-data build complete" in output
    assert "Dataset: 392 stimuli" in output